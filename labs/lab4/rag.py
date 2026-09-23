#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted, enforce_citations  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# TODO A: written before reading aip/rag.py::ANSWER_SYSTEM (see report.md for the diff).
# Six required elements, in order: (1) sources-only, (2) cite by index,
# (3) never invent an index, (4) exact refusal string, (5) surface conflicts,
# (6) length discipline.
ANSWER_SYSTEM = f"""\
You are Aurora Health Insurance's support assistant. You answer questions for \
support agents using ONLY the numbered sources supplied in the user message.

RULES

1. SOURCES ONLY. Every factual statement must come from the numbered sources \
below. You have no other knowledge. If you know something about insurance from \
general knowledge and it is not in the sources, you must not state it. Do not \
estimate or fill gaps with what is typical in the industry.
You MAY combine facts that are stated in different sources to answer a \
question that spans them: if [1] says out-patient treatment is not covered and \
[3] says the OPD rider is unavailable on Bronze, answering "no, and here is \
why [1][3]" is correct and expected. What you may not do is add a fact that no \
source states. Combining stated facts is answering; inventing a fact is not.

2. CITE EVERY FACTUAL SENTENCE by source index, in square brackets, e.g. \
"Claims must be filed within 30 days of discharge [2]." Use [1][4] when a \
sentence draws on more than one source. Place the citation at the end of the \
sentence it supports.

3. NEVER cite an index you were not given. If five sources are supplied, the \
only legal citations are [1] through [5]. A citation to a source that does not \
exist is the worst failure in this system.

4. WHEN THE SOURCES DO NOT ANSWER THE QUESTION AT ALL, reply with exactly this \
sentence and nothing else:
{REFUSAL}
Copy it character for character. Do not paraphrase it, do not append a topic \
to it, do not attach a citation to it, do not apologise, do not add a \
suggestion. The sentence must appear exactly as written above or downstream \
checks will not recognise it as a refusal.

4a. PARTIAL ANSWERS ARE REQUIRED when the sources support part of the \
question. A blanket refusal is WRONG whenever any part is supported. Structure \
it exactly like this:
  - first, state every part the sources do support, with citations;
  - then, on a new line, name the part you cannot support and follow it with \
the refusal sentence above, verbatim.
Example shape: "Treatment outside India is excluded except under the Platinum \
international emergency benefit [2]. The limit for that benefit is not stated \
in these sources. {REFUSAL}"
Check before refusing: does ANY source mention the subject of the question, \
even partially? If yes, you must answer that part.

5. WHEN SOURCES DISAGREE, say so explicitly and cite both, e.g. "[2] states 30 \
days while [4] states 15 days." Never silently pick one. If one source is \
marked superseded, archived or dated, say which and prefer the current one, \
citing both.

6. BE BRIEF. Two or three sentences unless the question genuinely needs more. \
Answer the question asked; do not restate it, do not add background, do not \
summarise the sources. Give figures exactly as the source gives them.

{UNTRUSTED_SYSTEM_CLAUSE}
"""

# C4: a stricter variant, used to move the refusal dial. Only rule 4 changes --
# everything else is identical, so the comparison isolates refusal strictness.
STRICT_REFUSAL_CLAUSE = f"""\

ADDITIONAL RULE (STRICT MODE). Prefer refusing over answering when you are not \
certain. Before answering, check that the sources state the answer explicitly. \
If the answer requires you to combine, infer, assume the question's premise, or \
generalise from a related but different product, plan, or scenario, do not \
answer: reply with exactly:
{REFUSAL}
An unsupported answer is far more damaging than an unnecessary refusal.
"""

ANSWER_SYSTEM_STRICT = ANSWER_SYSTEM + STRICT_REFUSAL_CLAUSE

# A citation is [n]; the index is what we validate against the source count.
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False
    # Added for Lab 4 reporting: repair rate (B3) and p95 latency (E1).
    repaired: bool = False
    latency_ms: float = 0.0
    n_sources: int = 0


def is_refusal(text: str) -> bool:
    """Exact-match refusal detection.

    Matching the exact sentence (not "sorry" or "cannot") is what makes refusal
    machine-detectable; a paraphrase the model invents is a bug we want to see
    as a non-refusal rather than silently count as one.
    """
    return REFUSAL.lower() in text.lower()


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """B2. Structural validation of a generated answer.

    Returns {"valid", "refused", "invalid_citations", "n_citations",
             "truncated", "reason"}.
    """
    text = (text or "").strip()
    refused = is_refusal(text)
    truncated = finish_reason == "length"
    cited = sorted({int(m) for m in _CITE.findall(text)})
    # enforce_citations owns the index check (aip/guards.py); it also requires
    # at least one citation, which is wrong for a pure refusal, so the refusal
    # case is handled separately below.
    _, invalid = enforce_citations(text, n_sources)

    if not text:
        reason = "empty answer"
    elif truncated:
        # T1 failure mode 4: a cut-off prose answer reads as complete.
        reason = "truncated (finish_reason == 'length')"
    elif invalid:
        reason = f"citation(s) out of range 1..{n_sources}: {invalid}"
    elif not cited and not refused:
        reason = "no citation on a non-refusal answer"
    else:
        reason = "ok"

    return {
        "valid": reason == "ok",
        "refused": refused,
        "invalid_citations": invalid,
        "n_citations": len(cited),
        "truncated": truncated,
        "reason": reason,
    }


def _build_answer(question: str, text: str, hits: list[Hit], v: dict, *,
                  repaired: bool, latency_ms: float, n_sources: int) -> Answer:
    return Answer(
        question=question, text=text, hits=hits, refused=v["refused"],
        citations_valid=not v["invalid_citations"], invalid_citations=v["invalid_citations"],
        n_citations=v["n_citations"], truncated=v["truncated"],
        repaired=repaired, latency_ms=latency_ms, n_sources=n_sources,
    )


def _generate(question: str, context: str, *, tier: str, system: str,
              max_tokens: int) -> tuple[str, str | None]:
    prompt = (
        f"{delimit_untrusted(context, 'RETRIEVED_DOCUMENT')}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the numbered sources above, citing each factual "
        f"sentence."
    )
    res = chat(prompt, system=system, tier=tier, max_tokens=max_tokens,
               temperature=0.0, return_full=True)
    return (res["text"] or "").strip(), res.get("finish_reason")


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "MAIN",
                    system: str = ANSWER_SYSTEM, max_tokens: int = 700) -> Answer:
    """Retrieve -> (rerank) -> generate -> validate -> repair -> else refuse.

    B3, the failure policy, and why:

    Repair once, then fall back to refusal. A single corrective retry is cheap
    (one extra call on ~9% of questions) and fixes the common case, which is a
    model that cited [7] when five sources were supplied -- it had the right
    content and the wrong label. Stripping the bad citation instead would leave
    an uncited factual sentence, which is unauditable and therefore worse than
    no answer on an insurance helpdesk. If the retry still fails validation we
    return the exact refusal string: a refusal is a known, measurable,
    recoverable state, while a confident answer carrying a fabricated citation
    is precisely the failure this system exists to prevent.

    Invariant: this function never returns citations_valid=False with
    refused=False.
    """
    t0 = time.perf_counter()
    hits = retriever.search(question, k=k)
    if reranker is not None:
        hits = reranker.rerank(question, hits, k=final_k)
    else:
        hits = hits[:final_k]
    context = format_context(hits)
    n_sources = len(hits)

    text, finish = _generate(question, context, tier=tier, system=system,
                             max_tokens=max_tokens)
    v = validate_answer(text, n_sources, finish)
    repaired = False

    if not v["valid"]:
        repaired = True
        corrective = (
            f"Your previous answer was rejected: {v['reason']}.\n"
            f"There are exactly {n_sources} sources, numbered 1 to {n_sources}. "
            f"Rewrite the answer using only those indices, citing every factual "
            f"sentence. If the sources do not answer the question, reply with "
            f"exactly: {REFUSAL}\n\n"
            f"Previous answer:\n{text}"
        )
        prompt = (
            f"{delimit_untrusted(context, 'RETRIEVED_DOCUMENT')}\n\n"
            f"Question: {question}\n\n{corrective}"
        )
        # A truncated answer is not a schema problem -- it needs more room, so
        # the retry gets double the budget (the aip.llm.structured rule).
        res = chat(prompt, system=system, tier=tier,
                   max_tokens=max_tokens * 2 if v["truncated"] else max_tokens,
                   temperature=0.0, return_full=True)
        text2 = (res["text"] or "").strip()
        v2 = validate_answer(text2, n_sources, res.get("finish_reason"))
        if v2["valid"]:
            text, v = text2, v2
        else:
            text = REFUSAL
            v = validate_answer(text, n_sources, None)

    return _build_answer(question, text, hits, v, repaired=repaired,
                         latency_ms=(time.perf_counter() - t0) * 1000,
                         n_sources=n_sources)


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN", system: str = ANSWER_SYSTEM,
                             final_k: int = 5, max_tokens: int = 700) -> Answer:
    """E2: same generator, gold context, no retrieval.

    The gold documents are chunked with the same chunker and size as the Lab 3
    winning configuration and the same number of sources is passed, so the ONLY
    difference from answer_question() is which passages arrive. Passing whole
    documents instead would change context length as well and confound the
    comparison.
    """
    t0 = time.perf_counter()
    chunks = [c for i, text in enumerate(gold_docs)
              for c in markdown_chunks(text, f"gold-{i}", size=400)]
    # Rank gold chunks by word overlap with the question so the generator sees
    # the same *number* of sources as the retrieved path, not the whole document.
    q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    chunks.sort(key=lambda c: len(q_terms & set(re.findall(r"[a-z0-9]+", c.text.lower()))),
                reverse=True)
    hits = [Hit(c, 1.0, "gold", i) for i, c in enumerate(chunks[:final_k])]
    context = format_context(hits)
    n_sources = len(hits)

    text, finish = _generate(question, context, tier=tier, system=system,
                             max_tokens=max_tokens)
    v = validate_answer(text, n_sources, finish)
    if not v["valid"] and not v["refused"]:
        text = REFUSAL
        v = validate_answer(text, n_sources, None)
    return _build_answer(question, text, hits, v, repaired=False,
                         latency_ms=(time.perf_counter() - t0) * 1000,
                         n_sources=n_sources)
