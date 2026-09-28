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
import re
import sys
import time
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
    "prompt": "v3",
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
    cached: bool
    repaired: bool
    guards: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class Pipeline:
    def __init__(self):
        from labs.lab4.rag import PROMPTS
        from labs.lab5.diagnose import build_fixed_retriever
        with tracing.trace("pipeline.build"):
            self.retriever = GuardedRetriever(build_fixed_retriever())
        self.system = PROMPTS[CONFIG["prompt"]]

    def answer(self, question: str, *, final_k: int | None = None) -> Result:
        from labs.lab4.rag import answer_question
        from labs.lab6.agent import filter_output

        t0 = time.perf_counter()
        usage: list[cost.Usage] = []
        token = _REQUEST_USAGE.set(usage)
        try:
            guards: list[str] = []
            with tracing.trace("guard.input") as span:
                # PII never leaves the process: redact before the question is
                # embedded or sent to the model.
                q, pii = redact_pii(question)
                if pii:
                    guards.append("input_pii:" + ",".join(sorted(pii)))
                span["pii"] = pii

            with tracing.trace("rag.answer", prompt=CONFIG["prompt"]) as span:
                a = answer_question(q, self.retriever, k=CONFIG["k"],
                                    final_k=final_k or CONFIG["final_k"],
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
        return Result(
            answer=text, refused=a.refused, partial=a.partial,
            citations=[s for s in sources if s["index"] in cited],
            sources=sources, retrieved_doc_ids=[h.doc_id for h in a.hits],
            cost_usd=sum(u.cost_usd for u in usage), llm_calls=len(usage),
            cached=bool(usage) and all(u.cached for u in usage),
            repaired=a.repaired, guards=guards,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


_PIPELINE: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Built once per process. Building per request re-embeds the corpus."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE
