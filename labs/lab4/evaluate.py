#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import judge_agreement, llm_judge  # noqa: E402
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import (  # noqa: E402
    ANSWER_SYSTEM,
    ANSWER_SYSTEM_STRICT,
    REFUSAL,
    answer_question,
    answer_with_gold_context,
    is_refusal,
)

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"

# Generation tier. The handout's default is MAIN, but on this free-tier key
# MAIN (gemini-3.7-flash) is throttled to tens of seconds per call, which makes
# a 45-question run impossible and blows the 6 s p95 target on its own. SMALL
# (gemini-3.5-flash-lite) generates; the judge stays on LARGE, so judge and
# generator are still different models (D3). Stated as a deviation in report.md.
GEN_TIER = os.environ.get("LAB4_GEN_TIER", "SMALL")
# Judge tier. The handout wants a *different, stronger* tier than the generator
# (D3), and llm_judge defaults to LARGE for exactly that reason. On this
# free-tier key LARGE (gemini-3.5-flash) is capped at 20 generate requests PER
# DAY, and a full run needs 90 judge calls, so LARGE cannot judge this set at
# all. SMALL judges instead -- which makes judge and generator the same model.
# That is self-preference bias (T3 S4.1), so D3 measures it rather than waving
# at it: --cross-check re-judges a sample on LARGE and reports the agreement.
JUDGE_TIER = os.environ.get("LAB4_JUDGE_TIER", "SMALL")


def build_retriever():
    """Lab 3 winning configuration: markdown-aware chunking at 400 characters,
    exact dense retrieval, final_k=5, archived documents excluded.

    Lab 3 measured nDCG@10 0.8600 / recall@5 0.9028 / hit_rate@1 0.8095 for
    this config (reports/lab3_sweeps.json). The archived filter is the D3 result:
    dropping `*-ARCHIVED` chunks at ingest fixed Q30 and changed nothing else.
    No golden question has an ARCHIVED document as its gold source.
    """
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)
              if "ARCHIVED" not in doc_id]
    return DenseRetriever(chunks, show_progress=False)


# ---------------------------------------------------------------------------
# judges (yours)
# ---------------------------------------------------------------------------
# D1. Both rubrics are rewritten from aip.evals' starting templates. The three
# cases the shipped faithfulness rubric does not handle are called out
# explicitly: partial refusal, paraphrase-into-a-stronger-claim, and
# right-about-the-world-wrong-about-the-context.
RUBRIC_FAITHFULNESS = """\
You are grading whether an ANSWER is supported by the provided CONTEXT.

Judge support ONLY. Do not judge helpfulness, style, or whether the answer
matches your own knowledge of insurance.

Score 1 (supported) if EVERY factual statement in the answer is stated in, or
is a direct restatement of, the context. Specifically:
- A REFUSAL IS ALWAYS SCORE 1 unless the context plainly contains the answer.
  A refusal asserts nothing, so it cannot be unsupported. Score a bare refusal
  1 even though it cites nothing and even though it is unhelpful; helpfulness
  is not what you are grading here.
- A PARTIAL answer that states what the context supports, cites it, and then
  refuses the remainder is SUPPORTED, provided the part it does answer is in
  the context. Do not penalise it for refusing the rest.
- Numbers, time limits, percentages and names must match the context exactly.

Score 0 (unsupported) if the answer does any of these:
- states anything absent from the context, EVEN IF IT IS TRUE IN THE REAL
  WORLD. Being right about insurance is not being supported by this context.
- strengthens or generalises the context: the context says "usually" and the
  answer says "always"; the context describes one plan and the answer states
  it for all plans; the context gives a condition and the answer drops it.
- refuses when the context plainly does contain the answer.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON: {{"score": 0 or 1, "unsupported_claims": [...], "reason": "one sentence"}}
"""

# The correctness rubric MUST handle refusal, or it punishes the exactly
# correct behaviour on the 5 unanswerable questions. The gold answers for those
# start with "REFUSE" or "PARTIAL REFUSE", which is the signal used here.
RUBRIC_CORRECTNESS = """\
Compare a CANDIDATE answer to a REFERENCE answer for the same question.

FIRST, check whether the REFERENCE begins with "REFUSE" or "PARTIAL REFUSE".

If the REFERENCE says REFUSE, the correct behaviour is to decline:
  2 = the candidate declines to answer (any clear statement that the sources
      do not contain the information).
  1 = the candidate declines but adds an unsupported claim, or hedges so
      weakly that a reader would take it as an answer.
  0 = the candidate confidently answers. Being fluent and plausible here is
      the worst outcome, not a partial success.

If the REFERENCE says PARTIAL REFUSE, the correct behaviour is to state the
supported part AND decline the unsupported part:
  2 = candidate does both.
  1 = candidate does only one of the two (answers the supported part but
      invents the rest, or refuses everything including the supported part).
  0 = candidate answers the unsupported part as though it were supported.

Otherwise (the reference is a normal answer):
  2 = the candidate answers WHAT THE QUESTION ASKED, and every number and
      condition it gives matches the reference.
  1 = partially correct (see the three rules below).
  0 = wrong, contradicts the reference, or refuses outright when the
      reference answers.

Judge against the QUESTION, not against the reference's word count. Three
rules that decide most cases:

  (a) EXTRA SUPPORTED DETAIL DOES NOT REDUCE THE SCORE. If the candidate
      answers the question and then adds further correct material the
      reference happens not to mention, that is still 2. Answering "30 days,
      and 60 days for group claims" when the reference says "30 days" is 2.

  (b) OMITTING DETAIL THE QUESTION DID NOT ASK FOR DOES NOT REDUCE THE SCORE.
      References often carry background the question never requested. Score 2
      if the candidate answers the actual question. Score 1 only when the
      missing material is part of what was ASKED -- for example the question
      asks to compare all four plans and the candidate does not identify which
      figure belongs to which plan, or asks to "list everything" and the
      candidate lists some of it.

  (c) A PARTIAL ANSWER SCORES 1. If the candidate answers part of the question
      and explicitly declines the rest, that is 1, not 0. It is 0 only if the
      part it does answer is wrong, or if it declines the whole question.

Ignore citation markers like [2] when judging content.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply as JSON: {{"score": 0|1|2, "reason": "one sentence"}}
"""


def _with_retry(fn, tries: int = 6, base: float = 15.0):
    """Retry transient provider failures: 429 (free-tier rate limit) and 503
    ("this model is currently experiencing high demand"). Neither is a result,
    so failing the run on one would silently turn provider weather into a
    missing measurement."""
    import random
    import re as _re
    import time as _time
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:                            # noqa: BLE001
            s = str(exc)
            transient = any(t in s for t in ("429", "RESOURCE_EXHAUSTED", "503",
                                             "UNAVAILABLE", "overloaded"))
            if not transient or attempt == tries - 1:
                raise
            m = _re.search(r"retry in ([\d.]+)s", s)
            delay = float(m.group(1)) + 3 if m else base * (attempt + 1) * (0.5 + random.random())
            print(f"    (transient {'429' if '429' in s else '503'}, retry "
                  f"{attempt + 1}/{tries} in {delay:.0f}s)", flush=True)
            _time.sleep(delay)


def _score(verdict: dict) -> int | None:
    """None, not 0, when the judge failed to produce a verdict.

    The war story in the handout: llm_judge returns {"score": 0,
    "parse_error": True} on an unparseable verdict. Taking that 0 at face value
    puts a silent downward bias on the headline metric. A parse failure is
    missing data, so it is excluded from the mean and counted separately.
    """
    if verdict.get("parse_error"):
        return None
    score = verdict.get("score")
    return int(score) if isinstance(score, (int, float)) else None


def judge_faithfulness(answer_text: str, context: str) -> int | None:
    return _score(_with_retry(lambda: llm_judge(RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text), tier=JUDGE_TIER)))


def judge_correctness(question: str, candidate: str, reference: str) -> int | None:
    return _score(_with_retry(lambda: llm_judge(RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate), tier=JUDGE_TIER)))


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _p95(values: list[float]) -> float:
    return sorted(values)[int(0.95 * (len(values) - 1))] if values else 0.0


# ---------------------------------------------------------------------------
def run_full(save: str = "", *, strict: bool = False, limit: int = 0,
             budget_usd: float = 1.00, judge: bool = True) -> dict:
    questions = load_questions(include_unanswerable=True)
    if limit:
        questions = questions[:limit]
    retriever = build_retriever()
    system = ANSWER_SYSTEM_STRICT if strict else ANSWER_SYSTEM
    rows = []

    with Budget(limit_usd=budget_usd, label="lab4-full") as b:
        for q in questions:
            a = _with_retry(lambda q=q: answer_question(
                q["question"], retriever, tier=GEN_TIER, system=system))
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "question": q["question"], "gold_answer": q["gold_answer"],
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "n_citations": a.n_citations, "n_sources": a.n_sources,
                "truncated": a.truncated, "repaired": a.repaired,
                "latency_ms": round(a.latency_ms, 1),
                "faithfulness": judge_faithfulness(a.text, ctx) if judge else None,
                "correctness": (judge_correctness(q["question"], a.text, q["gold_answer"])
                                if judge else None),
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
                "context": ctx,          # Lab 5 needs the context to diagnose
                "strict_mode": strict,
            })
            print(f"  {q['id']:<4} {'REFUSED' if a.refused else 'answered':<9} "
                  f"cit={a.n_citations} valid={a.citations_valid} "
                  f"{'REPAIRED ' if a.repaired else ''}{a.latency_ms:.0f}ms", flush=True)

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]
    correct_refusals = [r for r in refusals if r["unanswerable"]]
    lat = [r["latency_ms"] for r in rows]
    n_parse_fail = (0 if not judge else
                    sum(1 for r in rows if r["faithfulness"] is None)
                    + sum(1 for r in rows if r["correctness"] is None))

    corr = _mean(r["correctness"] for r in ans)
    summary = {
        "n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
        "citation_validity": _mean(r["citations_valid"] for r in rows),
        "faithfulness": _mean(r["faithfulness"] for r in rows),
        "correctness_0_2": corr,
        "correctness_normalised": None if corr is None else corr / 2,
        "refusal_recall": (len(correct_refusals) / len(una)) if una else None,
        "refusal_precision": (len(correct_refusals) / len(refusals)) if refusals else None,
        "refused_total": len(refusals), "refused_correctly": len(correct_refusals),
        "repair_rate": _mean(r["repaired"] for r in rows),
        "truncated": sum(1 for r in rows if r["truncated"]),
        "judge_parse_failures": n_parse_fail,
        "latency_p50_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": _p95(lat),
        "cost_usd": b.spent_usd, "cost_per_query_usd": b.spent_usd / max(len(rows), 1),
        "calls": b.calls, "cached_calls": b.cached_calls,
        "gen_tier": GEN_TIER, "judge_tier": JUDGE_TIER, "strict_mode": strict,
    }

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)"
          f"   mode={'STRICT' if strict else 'default'}")
    print(f"citation validity   {_fmt(summary['citation_validity'])}   (target 1.000)")
    print(f"faithfulness        {_fmt(summary['faithfulness'])}")
    print(f"correctness (0-2)   {_fmt(summary['correctness_0_2'])}"
          f"  normalised {_fmt(summary['correctness_normalised'])}")
    print(f"refusal recall      {_fmt(summary['refusal_recall'])}"
          f"   ({len(correct_refusals)}/{len(una)} unanswerable declined)")
    print(f"refusal precision   {_fmt(summary['refusal_precision'])}"
          f"   ({len(correct_refusals)}/{len(refusals)} of all refusals were right)")
    print(f"repair rate         {_fmt(summary['repair_rate'])}"
          f"   truncated={summary['truncated']}  judge parse failures={n_parse_fail}")
    print(f"latency p50/p95     {summary['latency_p50_ms']:.0f} / "
          f"{summary['latency_p95_ms']:.0f} ms   (target p95 <= 6000)")
    print(f"cost                ${b.spent_usd:.4f} total, "
          f"${summary['cost_per_query_usd']:.5f}/query   (target <= $0.01)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    for kind in sorted({r["kind"] for r in ans}):
        sub = [r for r in ans if r["kind"] == kind]
        m = _mean(r["correctness"] for r in sub)
        print(f"  {kind:<16} {_fmt(None if m is None else m / 2)}  n={len(sub)}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"summary": summary, "rows": rows},
                                indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")
    return summary


def run_gold_context(limit: int = 0) -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    if limit:
        questions = questions[:limit]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores, rows = [], [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = _with_retry(lambda q=q: answer_question(
                q["question"], retriever, tier=GEN_TIER))
            rb = judge_correctness(q["question"], a.text, q["gold_answer"])
            g = _with_retry(lambda q=q: answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus],
                tier=GEN_TIER))
            ga = judge_correctness(q["question"], g.text, q["gold_answer"])
            if rb is not None:
                retrieved_scores.append(rb / 2)
            if ga is not None:
                gold_scores.append(ga / 2)
            rows.append({"id": q["id"], "kind": q["kind"], "retrieved": rb, "gold": ga,
                         "retrieved_answer": a.text, "gold_answer_text": g.text,
                         "retrieved_docs": [h.doc_id for h in a.hits],
                         "relevant": q["relevant_docs"]})
            print(f"  {q['id']:<4} retrieved={rb} gold={ga}", flush=True)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")
    out = ROOT / "reports/lab4_decomposition.json"
    out.write_text(json.dumps({"A_gold": A, "B_retrieved": B,
                               "retrieval_loss": A - B, "generation_loss": 1 - A,
                               "n": len(rows), "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")


def run_rejudge(src: str, dst: str = "") -> None:
    """Re-score saved answers with the CURRENT rubrics, without regenerating.

    Needed to compare two prompt versions fairly: if the rubric changed between
    runs, a difference in faithfulness is a measurement change, not a system
    change. This re-judges stored (answer, context) pairs so both versions are
    scored by one rubric.
    """
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    with Budget(limit_usd=1.00, label="lab4-rejudge") as b:
        for r in rows:
            r["faithfulness"] = judge_faithfulness(r["answer"], r["context"])
            r["correctness"] = judge_correctness(r["question"], r["answer"],
                                                 r["gold_answer"])
            print(f"  {r['id']:<4} faith={r['faithfulness']} corr={r['correctness']}",
                  flush=True)
    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]
    ok = [r for r in refusals if r["unanswerable"]]
    corr = _mean(r["correctness"] for r in ans)
    print(f"\nre-judged {len(rows)} rows from {src}")
    print(f"citation validity {_fmt(_mean(r['citations_valid'] for r in rows))}")
    print(f"faithfulness      {_fmt(_mean(r['faithfulness'] for r in rows))}")
    print(f"correctness       {_fmt(corr)}  normalised {_fmt(None if corr is None else corr/2)}")
    print(f"refusal recall    {_fmt(len(ok)/len(una) if una else None)} ({len(ok)}/{len(una)})")
    print(f"refusal precision {_fmt(len(ok)/len(refusals) if refusals else None)} "
          f"({len(ok)}/{len(refusals)})")
    print(b.report())
    if dst:
        out = ROOT / dst
        out.write_text(json.dumps({"summary": {"rejudged_from": src}, "rows": rows},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> {out}")


def run_cross_check(n: int = 10) -> None:
    """D3: quantify self-preference. Re-judge a sample with LARGE (a different,
    stronger model) and compare with the SMALL judge's scores. LARGE is capped
    at 20 requests/day on this key, so n=10 answers x 1 faithfulness call is the
    most that fits; correctness is included only if the budget allows."""
    data = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    sample = [r for r in rows if r["faithfulness"] is not None][:n]
    pairs = []
    for r in sample:
        big = _score(_with_retry(lambda r=r: llm_judge(RUBRIC_FAITHFULNESS.format(
            context=r["context"][:8000], answer=r["answer"]), tier="LARGE")))
        pairs.append({"id": r["id"], "small_judge": r["faithfulness"], "large_judge": big})
        print(f"  {r['id']:<4} SMALL={r['faithfulness']}  LARGE={big}", flush=True)
    usable = [(p["small_judge"], p["large_judge"]) for p in pairs
              if p["large_judge"] is not None]
    out = {"n": len(usable), "pairs": pairs}
    if usable:
        agree = sum(1 for a, b in usable if a == b) / len(usable)
        small_mean = statistics.fmean(a for a, _ in usable)
        large_mean = statistics.fmean(b for _, b in usable)
        out.update({"raw_agreement": agree, "small_mean": small_mean,
                    "large_mean": large_mean, "self_preference_gap": small_mean - large_mean})
        print(f"\nagreement {agree:.3f} over n={len(usable)}   "
              f"SMALL mean {small_mean:.3f} vs LARGE mean {large_mean:.3f}   "
              f"gap {small_mean - large_mean:+.3f}")
        print("A positive gap is the self-preference direction: the generator's own "
              "model rates its answers higher than a different model does.")
    p = ROOT / "reports/lab4_cross_check.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {p}")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    data = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    # Spread the sample across kinds so the labels are not 20 single_hop cases;
    # kappa on a sample with one class is meaningless.
    by_kind: dict[str, list] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    sample, i = [], 0
    while len(sample) < 20 and any(v[i:] for v in by_kind.values()):
        for kind in sorted(by_kind):
            if len(sample) < 20 and i < len(by_kind[kind]):
                sample.append(by_kind[kind][i])
        i += 1
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "kind": r["kind"], "question": r["question"],
        "gold_answer": r["gold_answer"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}  ({len(sample)} cases)")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    data = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    machine = {r["id"]: r for r in rows}
    out = {}
    for field in ("faithfulness", "correctness"):
        pairs = [(machine[r["id"]][field], r[f"human_{field}"]) for r in human
                 if r.get(f"human_{field}") is not None
                 and machine.get(r["id"], {}).get(field) is not None]
        if not pairs:
            print(f"{field}: no human labels yet")
            continue
        m, h = [p[0] for p in pairs], [p[1] for p in pairs]
        res = judge_agreement(m, h)
        out[field] = res
        print(f"{field}: {res}")
        disagree = [(machine[r['id']]['id'], machine[r['id']][field], r[f'human_{field}'])
                    for r in human
                    if r.get(f"human_{field}") is not None
                    and machine.get(r["id"], {}).get(field) is not None
                    and machine[r["id"]][field] != r[f"human_{field}"]]
        if disagree:
            print(f"  disagreements (id, judge, human): {disagree}")
    if out:
        p = ROOT / "reports/lab4_kappa.json"
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved -> {p}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--rejudge", default="", help="re-score a saved run with current rubrics")
    ap.add_argument("--rejudge-save", default="", help="where to write the re-scored run")
    ap.add_argument("--cross-check", type=int, default=0,
                    help="D3: re-judge N answers on LARGE to measure self-preference")
    ap.add_argument("--strict", action="store_true", help="C4: stricter refusal prompt")
    ap.add_argument("--limit", type=int, default=0, help="pilot on the first N questions")
    ap.add_argument("--no-judge", action="store_true",
                    help="C4: refusal counts only, skip the judges (saves quota)")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save, strict=a.strict, limit=a.limit, judge=not a.no_judge)
    if a.gold_context:
        run_gold_context(a.limit)
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if a.rejudge:
        run_rejudge(a.rejudge, a.rejudge_save)
    if a.cross_check:
        run_cross_check(a.cross_check)
    if not any([a.full, a.gold_context, a.calibrate, a.kappa, a.cross_check, a.rejudge]):
        ap.print_help()
