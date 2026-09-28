"""Lab 7 — the one pipeline that ships.

The service (`service.py`) and the regression gate (`gate.py`) both call
`Pipeline.answer()`, so CI measures the exact code path users hit.

    question ─► guard.input (PII redaction) ─► retrieve (Lab 3 dense, Lab 5
    document expansion) ─► guard.retrieved (Lab 6 layer 2: drop chunks shaped
    like instructions) ─► generate (Lab 4, prompt v3, SMALL) ─► validate /
    repair (Lab 4) ─► guard.output (Lab 6 layer 5) ─► response

Why each choice is here, in one line each (evidence in labs/lab7/PLAN.md):
* Lab 5 retriever over Lab 4's: correctness 0.949 vs 0.936, faster, 1.14x cost.
* Prompt v3: the only prompt that clears refusal precision >= 0.75 (0.833).
* SMALL tier: MAIN is ~36 s/call on this key and alone breaks the 6 s SLO.
* Layer 2 runs on retrieved text, not on the user's question: Lab 6 measured
  the question-side detector blocking innocent customers ("ignore what the
  agent told me"). It flags 0 of 231 corpus chunks, so it costs no recall.
"""
from __future__ import annotations

import contextvars
import dataclasses
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cost, tracing  # noqa: E402
from aip.guards import detect_injection, redact_pii  # noqa: E402
from aip.retrieval import Hit, Retriever  # noqa: E402

CONFIG = {
    "retriever": "lab5-doc-expansion(lab3 markdown-400 dense, no ARCHIVED)",
    "k": 12,
    "final_k": 6,
    "prompt": "v1",
    "tier": "SMALL",
}

# --------------------------------------------------------------------------
# Per-request cost. aip.cost keeps one process-wide ledger plus a *global*
# stack of Budgets, so two concurrent requests would each see the other's
# spend. A context variable is per-request (FastAPI runs each sync handler in
# its own copied context), so wrapping cost.record() gives an exact per-request
# ledger without touching the shared toolkit.
# --------------------------------------------------------------------------
_REQUEST_USAGE: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "aip_request_usage", default=None)
_toolkit_record = cost.record


def _record(usage: cost.Usage) -> None:
    _toolkit_record(usage)
    acc = _REQUEST_USAGE.get()
    if acc is not None:
        acc.append(usage)


cost.record = _record


class GuardedRetriever:
    """Lab 6 layer 2 on the RAG path: drop retrieved chunks that contain text
    shaped like instructions. Dropping (not rewriting) means the generator sees
    fewer sources and refuses if nothing clean is left -- the safe failure."""

    def __init__(self, base: Retriever):
        self.base = base
        self.last_flags: list[str] = []

    def search(self, query: str, k: int = 8) -> list[Hit]:
        with tracing.trace("retrieve", k=k) as span:
            hits = self.base.search(query, k=k)
            span["n_hits"] = len(hits)
        clean, flags = [], []
        with tracing.trace("guard.retrieved") as span:
            for h in hits:
                v = detect_injection(h.chunk.text)
                if v.flagged:
                    flags.extend(v.signals)
                else:
                    clean.append(h)
            span["dropped"] = len(hits) - len(clean)
        self.last_flags = flags
        return clean


_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Result:
    answer: str
    refused: bool
    partial: bool
    citations: list[dict]
    sources: list[dict]
    retrieved_doc_ids: list[str]
    cost_usd: float
    llm_calls: int
    llm_cached: bool           # every model call was an aip.cache replay
    repaired: bool
    guards: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cache_layer: str = ""      # "" | "exact" | "semantic" (answer cache, B1)
    cache_similarity: float = 0.0
    context: str = ""          # exactly what the generator saw; the gate's judge reads it
    # What these calls cost at list price even when replayed from aip.cache
    # (cost_usd is 0 for a replay). The gate's cost metric reads this.
    list_price_usd: float = 0.0


class Pipeline:
    def __init__(self):
        from labs.lab4.rag import PROMPTS
        from labs.lab5.diagnose import build_fixed_retriever
        from labs.lab7.answer_cache import AnswerCache
        with tracing.trace("pipeline.build"):
            self.retriever = GuardedRetriever(build_fixed_retriever())
        self.system = PROMPTS[CONFIG["prompt"]]
        self.cache = AnswerCache(CONFIG, embed_model=self.retriever.base.base.model)

    @staticmethod
    def _guard_input(question: str) -> tuple[str, list[str]]:
        with tracing.trace("guard.input") as span:
            # PII never leaves the process: redact before the question is
            # embedded, cached, or sent to the model.
            q, pii = redact_pii(question)
            span["pii"] = pii
        return q, (["input_pii:" + ",".join(sorted(pii))] if pii else [])

    def _from_cache(self, q: str, final_k: int, guards: list[str], t0: float) -> Result | None:
        hit = self.cache.get(q, final_k)
        if hit is None:
            return None
        return dataclasses.replace(
            hit.result, cost_usd=0.0, llm_calls=0, llm_cached=False,
            guards=guards + [g for g in hit.result.guards if not g.startswith("input_pii")],
            cache_layer=hit.layer, cache_similarity=round(hit.similarity, 4),
            latency_ms=(time.perf_counter() - t0) * 1000)

    def answer(self, question: str, *, final_k: int | None = None,
               use_cache: bool = True) -> Result:
        """The non-streaming path. The gate calls this with use_cache=False:
        it measures the pipeline, not the answer cache."""
        from aip.retrieval import format_context
        from labs.lab4.rag import answer_question
        from labs.lab6.agent import filter_output

        t0 = time.perf_counter()
        final_k = final_k or CONFIG["final_k"]
        q, guards = self._guard_input(question)
        if use_cache and (cached := self._from_cache(q, final_k, guards, t0)):
            return cached

        usage: list[cost.Usage] = []
        token = _REQUEST_USAGE.set(usage)
        try:
            with tracing.trace("rag.answer", prompt=CONFIG["prompt"]) as span:
                a = answer_question(q, self.retriever, k=CONFIG["k"],
                                    final_k=final_k,
                                    tier=CONFIG["tier"], system=self.system)
                span.update(refused=a.refused, partial=a.partial, repaired=a.repaired)
            if self.retriever.last_flags:
                guards.append("retrieved_injection:" + ",".join(self.retriever.last_flags))

            with tracing.trace("guard.output") as span:
                text, fired = filter_output(a.text)
                guards += [f"output:{f}" for f in fired]
                span["fired"] = fired
        finally:
            _REQUEST_USAGE.reset(token)

        sources = [{"index": i, "doc_id": h.doc_id, "excerpt": h.chunk.text}
                   for i, h in enumerate(a.hits, start=1)]
        cited = sorted({int(n) for n in _CITE.findall(text)})
        result = Result(
            answer=text, refused=a.refused, partial=a.partial,
            citations=[s for s in sources if s["index"] in cited],
            sources=sources, retrieved_doc_ids=[h.doc_id for h in a.hits],
            cost_usd=sum(u.cost_usd for u in usage), llm_calls=len(usage),
            llm_cached=bool(usage) and all(u.cached for u in usage),
            repaired=a.repaired, guards=guards,
            latency_ms=(time.perf_counter() - t0) * 1000, context=format_context(a.hits),
            list_price_usd=sum(cost.price_of(u.model, u.prompt_tokens, u.completion_tokens)
                               for u in usage),
        )
        if use_cache:
            self.cache.put(q, final_k, result)
        return result

    # ----------------------------------------------------------------------
    # B2/B3 — streaming.
    #
    # The problem: citations cannot be validated until the answer is complete,
    # but by then it has been sent. Strategy: STREAM SENTENCE BY SENTENCE, THEN
    # A VALIDATION EVENT THE UI ACTS ON.
    #   * Tokens are buffered to sentence boundaries and each sentence passes
    #     the Lab 6 output filter before it leaves -- a URL or PII split across
    #     two tokens cannot slip through. Answers are 2-3 sentences, so TTFT is
    #     the first sentence, not the whole answer.
    #   * Citation markers stream as text; the resolved citations (doc, excerpt)
    #     are sent once, at the end, only after validate_answer() passes.
    #   * If validation fails, a `replace` event carries the answer from the
    #     non-streaming path (which repairs once, then refuses), and the UI
    #     swaps it in. Lab 4's repair rate on this pipeline is 0/45, so the
    #     retract path is rare -- but an unvalidated citation is never final.
    # Rejected: buffer-then-stream (TTFT = full generation, no benefit); raw
    # token streaming (unfiltered text reaches the client).
    #
    # No span is held open across a `yield`: aip.tracing keeps its span stack
    # per thread and the SSE server may resume the generator on another one.
    # ----------------------------------------------------------------------
    def stream(self, question: str, *, final_k: int | None = None) -> Iterator[dict]:
        from litellm import completion

        from aip.config import resolve_model
        from aip.retrieval import format_context
        from labs.lab4.rag import build_user_prompt, validate_answer
        from labs.lab6.agent import filter_output

        t0 = time.perf_counter()
        final_k = final_k or CONFIG["final_k"]
        q, guards = self._guard_input(question)

        cached = self._from_cache(q, final_k, guards, t0)
        if cached is not None:
            yield {"event": "meta", "cache_layer": cached.cache_layer,
                   "cache_similarity": cached.cache_similarity}
            yield {"event": "token", "text": cached.answer}
            yield from self._final_events(cached, t0, ttft_ms=cached.latency_ms)
            return

        hits = self.retriever.search(q, k=CONFIG["k"])[:final_k]
        context = format_context(hits)
        yield {"event": "meta", "cache_layer": "", "n_sources": len(hits),
               "retrieve_ms": round((time.perf_counter() - t0) * 1000, 1)}

        model = resolve_model(CONFIG["tier"])
        t_gen = time.perf_counter()
        resp = completion(
            model=model, temperature=0.0, max_tokens=700, stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": build_user_prompt(q, context)}])

        full, buf, fired = "", "", []
        ttft_ms = first_token_ms = None
        finish, pt, ct = None, 0, 0
        for chunk in resp:
            if getattr(chunk, "usage", None):
                pt = int(chunk.usage.prompt_tokens or 0)
                ct = int(chunk.usage.completion_tokens or 0)
            if not chunk.choices:
                continue
            finish = chunk.choices[0].finish_reason or finish
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000
            full += delta
            buf += delta
            ready, buf = _split_sentences(buf)
            if ready:
                safe, f = filter_output(ready)
                fired += f
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                yield {"event": "token", "text": safe}
        if buf:
            safe, f = filter_output(buf)
            fired += f
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            yield {"event": "token", "text": safe}

        gen_ms = (time.perf_counter() - t_gen) * 1000
        usd = cost.price_of(model, pt, ct)
        _record(cost.Usage(model, pt, ct, usd, gen_ms, cached=False, calls=1,
                           priced=cost.is_priced(model)))
        tracing.event("llm.stream", model=model, prompt_tokens=pt, completion_tokens=ct,
                      cost_usd=round(usd, 6), latency_ms=round(gen_ms, 1),
                      first_token_ms=round(first_token_ms or 0, 1), finish_reason=finish)

        full = full.strip()
        v = validate_answer(full, len(hits), finish)
        if not v["valid"]:
            # Retract: the non-streaming path repairs once, then refuses.
            r = self.answer(question, final_k=final_k, use_cache=False)
            r = dataclasses.replace(r, cost_usd=r.cost_usd + usd, llm_calls=r.llm_calls + 1,
                                    repaired=True)
            yield {"event": "replace", "text": r.answer, "reason": v["reason"]}
            yield from self._final_events(r, t0, ttft_ms=ttft_ms)
            return

        text, f2 = filter_output(full)
        sources = [{"index": i, "doc_id": h.doc_id, "excerpt": h.chunk.text}
                   for i, h in enumerate(hits, start=1)]
        cited = sorted({int(n) for n in _CITE.findall(text)})
        result = Result(
            answer=text, refused=v["refused"], partial=v["partial"],
            citations=[s for s in sources if s["index"] in cited], sources=sources,
            retrieved_doc_ids=[h.doc_id for h in hits], cost_usd=usd, llm_calls=1,
            llm_cached=False, repaired=False,
            guards=guards + [f"output:{x}" for x in sorted(set(fired + f2))],
            latency_ms=(time.perf_counter() - t0) * 1000, context=context,
            list_price_usd=usd)
        self.cache.put(q, final_k, result)
        yield from self._final_events(result, t0, ttft_ms=ttft_ms)

    @staticmethod
    def _final_events(r: Result, t0: float, *, ttft_ms: float | None) -> Iterator[dict]:
        total_ms = (time.perf_counter() - t0) * 1000
        yield {"event": "citations", "citations": r.citations, "sources": r.sources}
        yield {"event": "validation", "valid": True, "refused": r.refused,
               "partial": r.partial, "repaired": r.repaired}
        tracing.event("http.ask_stream", ttft_ms=round(ttft_ms or 0, 1),
                      total_ms=round(total_ms, 1), cache_layer=r.cache_layer,
                      refused=r.refused, cost_usd=round(r.cost_usd, 6))
        yield {"event": "done", "ttft_ms": round(ttft_ms or 0, 1),
               "total_ms": round(total_ms, 1), "cost_usd": r.cost_usd,
               "cache_layer": r.cache_layer, "guards": r.guards, "answer": r.answer}


# A sentence ends at . ! ? or a newline, followed by whitespace. "Rs. 5,000" and
# "[2]." are fine: the split only moves text between flushes, never drops it.
_SENT_END = re.compile(r"(?<=[.!?\n])\s+")


def _split_sentences(buf: str) -> tuple[str, str]:
    """Return (complete sentences ready to send, remainder to keep buffering)."""
    parts = _SENT_END.split(buf)
    if len(parts) < 2:
        return "", buf
    ready_len = len(buf) - len(parts[-1])
    return buf[:ready_len], buf[ready_len:]


_PIPELINE: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Built once per process. Building per request re-embeds the corpus."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE
