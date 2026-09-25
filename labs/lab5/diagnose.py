#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.retrieval import Hit, Retriever  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.evaluate import build_retriever  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


_NUM = re.compile(r"\d[\d,]*")


def _numbers(text: str) -> set[str]:
    """Numbers with separators and common suffixes normalised.

    'Rs 1,00,000', '1,00,000' and '100000' are the same figure; '30 days' and
    '30' are the same number. Without this, the shipped test fails on exactly
    the answers that matter most here -- the ones that are a figure.
    """
    return {m.group().replace(",", "").lstrip("0") or "0" for m in _NUM.finditer(text)}


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test: is the gold answer's content present in the gold documents?

    Improvement over the shipped version, which was pure word overlap > 0.4:

      1. NUMBERS DECIDE when the gold answer contains one. An answer of "30
         days from discharge" is present iff 30 appears in the gold documents;
         word overlap on 'discharge' proves nothing. Separators are normalised
         so "Rs 1,00,000" matches "1,00,000" and "100000".
      2. A REFUSE gold answer is handled explicitly. Gold answers for the
         unanswerable questions begin with REFUSE/PARTIAL REFUSE and describe
         what is *absent*; scoring their words against the corpus is
         meaningless. PARTIAL REFUSE counts as present (part is supported),
         REFUSE as absent -- which is the mode-1 answer by definition.
      3. Stopword-ish short tokens are still dropped, and the threshold stays
         0.4 for the non-numeric case.

    How I know it is better: it is checked against ground truth I already have.
    Q36/Q38/Q39 have no relevant documents at all and must be mode 1; Q40 asks
    for a phone number that is genuinely absent while its gold document exists
    -- the shipped word-overlap test calls Q40 "present" because the words
    'helpline' and 'number' appear, and the numeric rule correctly calls it
    absent. Q01 ('30 days from the date of discharge') must be present, and
    both versions agree there. See report.md for the case-by-case check.
    """
    gold = gold_answer.strip()
    upper = gold.upper()
    if upper.startswith("PARTIAL REFUSE"):
        return True
    if upper.startswith("REFUSE"):
        return False

    text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not text:
        return False

    gold_nums = _numbers(gold)
    if gold_nums:
        return bool(gold_nums & _numbers(text))

    tokens = [t for t in gold.lower().split() if len(t) > 4]
    if not tokens:
        return True
    return sum(1 for t in tokens if t.strip(".,;()") in text) / len(tokens) > 0.4


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None,
             findable_by_own_text: bool | None = None,
             in_final_k: bool | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    TODO: complete the branches marked TODO. Follow the tree in the handout;
    do not invent your own ordering, because the ordering is what makes the
    modes mutually exclusive.
    """
    # Mode 7 first: right answer, wrong citation. Check this before anything
    # else, because a mode-7 failure is not a retrieval failure at all.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    # Mode 6: does gold context fix it?
    # The direction is the one people invert: gold context FIXING the answer
    # means the generator was always capable and retrieval starved it, so the
    # fault is upstream. Gold context NOT fixing it means generation is at
    # fault -- the right text was in front of the model and it still failed.
    if gold_context_fixes_it is False:
        return 6, ("gold context does not fix it: the generator had the right "
                   "passages and still answered wrongly")

    # From here the fault is upstream of the generator. Where?
    if in_top_30 is False:
        # Not even in the top 30. Is the chunk findable at all?
        if findable_by_own_text is True:
            return 3, ("gold doc absent from top 30, but retrievable by the gold "
                       "chunk's own text: the query embedding is the problem")
        if findable_by_own_text is False:
            return 2, ("gold doc absent from top 30 and NOT retrievable even by "
                       "its own text: needs_human_check on the chunk itself")
        return 3, "gold doc absent from top 30 (own-text probe not run)"

    if in_top_30:
        if dropped_by_reranker:
            return 5, "gold doc was in the stage-1 pool and the reranker dropped it"
        if in_final_k is False:
            return 4, ("gold doc in the top 30 but outside the final k passed to "
                       "the generator")
        # The gold doc DID reach the generator, and gold-only context still
        # scored higher. What differs is the company it kept: the retrieved
        # context also carried competing near-duplicate passages. That is a
        # ranking/composition fault, not a generation one -- same generator,
        # same question, better ordering wins.
        return 4, ("gold doc reached the generator, but cleaner gold-only "
                   "context scored higher: distractors alongside it are the fault")

    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "█" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reports/lab4.json")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json")
    ap.add_argument("--before-after", action="store_true",
                    help="Part D: run the fix and compare against reports/lab4.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", default="expand", choices=["expand", "single"])
    args = ap.parse_args()

    if args.before_after:
        run_before_after(limit=args.limit, variant=args.variant,
                         save=("reports/lab5_before_after.json" if args.variant == "expand"
                               else "reports/lab5_variant_single.json"))
        return

    data = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    # Evidence 1 (mode 6): does gold context fix it? Taken from the Lab 4
    # decomposition run rather than re-called -- same generator, same judge,
    # already paid for.
    gold_fix = {}
    dec_path = ROOT / "reports/lab4_decomposition.json"
    if dec_path.exists():
        for d in json.loads(dec_path.read_text(encoding="utf-8"))["rows"]:
            if d.get("gold") is not None and d.get("retrieved") is not None:
                gold_fix[d["id"]] = d["gold"] > d["retrieved"]

    failures = [r for r in rows
                if (r.get("correctness") is not None and r["correctness"] < 2)
                or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    # Evidence 2 (modes 3/4/5): was the gold doc in the top 30, and is the gold
    # chunk findable by its own text? Both re-run the Lab 4 retriever; every
    # embedding involved is already cached, so this costs nothing.
    retriever = build_retriever()
    top30, findable = {}, {}
    for r in failures:
        q = questions[r["id"]]
        if not q["relevant_docs"]:
            continue
        hits = retriever.search(q["question"], k=30)
        top30[r["id"]] = any(h.doc_id in q["relevant_docs"] for h in hits)
        if not top30[r["id"]]:
            # Own-text probe: search with the gold document's own opening text.
            probe = " ".join(corpus[q["relevant_docs"][0]].split()[:40])
            probe_hits = retriever.search(probe, k=30)
            findable[r["id"]] = any(h.doc_id in q["relevant_docs"] for h in probe_hits)

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        mode, evidence = classify(
            r, q, corpus,
            gold_context_fixes_it=gold_fix.get(r["id"]),
            in_top_30=top30.get(r["id"]),
            # Lab 4 ships no reranker, so mode 5 is structurally impossible
            # here. Recorded as False rather than left unknown, so the tree
            # cannot silently fall through to it.
            dropped_by_reranker=False,
            findable_by_own_text=findable.get(r["id"]),
            in_final_k=(any(d in q["relevant_docs"] for d in r.get("retrieved", []))
                        if q["relevant_docs"] else None))
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "gold_context_fixes_it": gold_fix.get(r["id"]),
                    "in_top_30": top30.get(r["id"]),
                    "in_final_k": (any(d in q["relevant_docs"] for d in r.get("retrieved", []))
                                   if q["relevant_docs"] else None),
                    "findable_by_own_text": findable.get(r["id"]),
                    "correctness": r.get("correctness"),
                    "citations_valid": r.get("citations_valid"),
                    "refused": r.get("refused"),
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    print("\n" + pareto(tally))
    print("\nCases marked needs_human_check are Part A2. Open them.")

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")




# ===========================================================================
# Part C — the fix
# ===========================================================================
# DIAGNOSIS (Part A, n=12): 6 ranking + 6 generation, and the per-question
# evidence says both clusters have one cause. Every failing question's 5-slot
# context was a MIXTURE of documents: gold chunks 0-4 of 5, distractor chunks
# 1-5 of 5. The ranking cluster fails because foreign passages compete with the
# right one; the generation cluster fails because the answer is spread over
# several chunks of one document and only some of them arrived (Q05: the gold
# document has 10 chunks mentioning the grace period, the model saw one and
# answered 30 days without the instalment case; Q10: the Ombudsman stage
# arrived, the two escalation stages before it did not).
#
# Those two pull in opposite directions -- fewer passages vs more passages --
# so neither "lower final_k" nor "raise final_k" can serve both. What serves
# both is changing WHICH document the passages come from: keep the same budget,
# spend it on the best documents instead of spreading it across five.
#
# PREDICTION (stated before measuring, Part B):
#   recover 3-5 of the 6 generation failures (the incomplete-answer ones),
#   1-2 of the 6 ranking failures, 4-7 of 12 overall; refusal precision to
#   improve slightly (fewer competing documents => fewer spurious refusals);
#   RISK: questions needing 3 gold documents (Q26) get worse, and cost rises
#   with the longer context (predicted ~1.3-1.6x input tokens, well inside the
#   2x budget, with no extra model calls).
class DocExpansionRetriever(Retriever):
    """Keep the ranking, but give the best document room to finish its answer.

    Stage 1 retrieves as Lab 4 does. Stage 2 then:
      * expands the TOP document to its `per_doc` best chunks, in document
        order, so a multi-part answer (Q05's instalment case, Q10's escalation
        stages) arrives whole rather than as whichever single chunk ranked
        highest; and
      * keeps the next `keep_top` chunks of the original ranking, so questions
        that genuinely need a second document (Q26 needs three) do not lose it.

    Only ONE variable changes against Lab 4: which passages reach the
    generator. Same embeddings, same prompt, same tier, and -- measured
    offline over all 42 answerable questions -- essentially the same context
    size (5.1 slots vs 5.0), so the fix is cost-neutral by construction.

    Offline composition check, v1 -> v2 (gold chunks as a share of context):
        all questions   0.548 -> 0.640
        the 12 failures 0.517 -> 0.590
        gold-doc recall 0.891 -> 0.843   <- the price paid
    """

    name = "doc_expansion"

    def __init__(self, base: Retriever, chunks, *, keep_top: int = 4,
                 expand_docs: int = 1, per_doc: int = 3):
        self.base, self.keep_top = base, keep_top
        self.expand_docs, self.per_doc = expand_docs, per_doc
        self.by_doc: dict[str, list] = {}
        for c in chunks:
            self.by_doc.setdefault(c.doc_id, []).append(c)

    def search(self, query: str, k: int = 8):
        hits = self.base.search(query, k=max(k, 12))
        score = {h.chunk.chunk_id: h.score for h in hits}
        ranked_docs: list[str] = []
        for h in hits:
            if h.doc_id not in ranked_docs:
                ranked_docs.append(h.doc_id)

        chosen, seen = [], set()
        for doc_id in ranked_docs[:self.expand_docs]:
            siblings = self.by_doc.get(doc_id, [])
            best = sorted(siblings, key=lambda c: -score.get(c.chunk_id, 0.0))
            keep = {c.chunk_id for c in best[:self.per_doc]}
            for c in siblings:                      # document order, not score
                if c.chunk_id in keep and c.chunk_id not in seen:
                    chosen.append(c)
                    seen.add(c.chunk_id)
        for h in hits[:self.keep_top]:
            if h.chunk.chunk_id not in seen:
                chosen.append(h.chunk)
                seen.add(h.chunk.chunk_id)
        return [Hit(c, score.get(c.chunk_id, 0.0), "doc_expansion", i)
                for i, c in enumerate(chosen)]


def build_fixed_retriever(keep_top: int = 4, expand_docs: int = 1, per_doc: int = 3):
    """Lab 4's retriever, wrapped in document expansion (the Part C fix)."""
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)
              if "ARCHIVED" not in doc_id]
    return DocExpansionRetriever(build_retriever(), chunks, keep_top=keep_top,
                                 expand_docs=expand_docs, per_doc=per_doc)


def run_before_after(save: str = "reports/lab5_before_after.json",
                     limit: int = 0, variant: str = "expand") -> None:
    """Part D: same questions, same judges, one variable changed.

    v1 numbers are read from reports/lab4.json rather than re-run, so the
    comparison cannot drift on generator temperature or a judge update; only
    v2 is generated here.
    """
    import statistics as _st

    from aip.cost import Budget
    from aip.retrieval import format_context
    from labs.lab4.evaluate import _mean, _p95, judge_correctness, judge_faithfulness
    from labs.lab4.rag import answer_question

    v1_rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))["rows"]
    v1 = {r["id"]: r for r in v1_rows}
    questions = load_questions(include_unanswerable=True)
    if limit:
        questions = questions[:limit]
    # variant="expand"  : the shipped fix (keep top 4 + expand top doc to 3)
    # variant="single"   : the purest-context alternative -- ONE document, five
    #                      chunks. Offline it had the best gold share (0.810 vs
    #                      0.640) and the worst gold-doc recall (0.554 vs 0.843).
    #                      Run to find out which of those two numbers decides.
    retriever = (build_fixed_retriever(keep_top=0, expand_docs=1, per_doc=5)
                 if variant == "single" else build_fixed_retriever())
    print(f"variant: {variant}", flush=True)

    rows = []
    with Budget(limit_usd=1.00, label="lab5-v2") as b:
        for q in questions:
            a = answer_question(q["question"], retriever, k=12, final_k=6, tier="SMALL")
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "question": q["question"], "gold_answer": q["gold_answer"],
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid, "n_citations": a.n_citations,
                "n_sources": a.n_sources, "repaired": a.repaired,
                "latency_ms": round(a.latency_ms, 1),
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                "retrieved": [h.doc_id for h in a.hits], "relevant": q["relevant_docs"],
                "context": ctx,
            })
            before = v1.get(q["id"], {}).get("correctness")
            after = rows[-1]["correctness"]
            move = "=" if before == after else ("UP" if (after or 0) > (before or 0) else "DOWN")
            print(f"  {q['id']:<5} corr {before} -> {after}  {move}", flush=True)

    def summarise(rs):
        ans = [r for r in rs if not r["unanswerable"]]
        una = [r for r in rs if r["unanswerable"]]
        ref = [r for r in rs if r["refused"]]
        ok = [r for r in ref if r["unanswerable"]]
        corr = _mean(r["correctness"] for r in ans)
        lat = [r["latency_ms"] for r in rs]
        return {
            "n": len(rs),
            "correctness": None if corr is None else corr / 2,
            "faithfulness": _mean(r["faithfulness"] for r in rs),
            "citation_validity": _mean(r["citations_valid"] for r in rs),
            "refusal_recall": len(ok) / len(una) if una else None,
            "refusal_precision": len(ok) / len(ref) if ref else None,
            "refused_total": len(ref), "refused_correctly": len(ok),
            "repair_rate": _mean(r["repaired"] for r in rs),
            "latency_p50_ms": _st.median(lat) if lat else None,
            "latency_p95_ms": _p95(lat),
            "mean_sources": _st.fmean(r["n_sources"] for r in rs),
        }

    ids = {r["id"] for r in rows}
    s1, s2 = summarise([r for r in v1_rows if r["id"] in ids]), summarise(rows)
    s2["cost_usd"] = b.spent_usd
    s2["cost_per_query_usd"] = b.spent_usd / max(len(rows), 1)

    print(f"\n{'metric':<22}{'v1':>12}{'v2':>12}{'delta':>12}")
    for key in ("correctness", "faithfulness", "citation_validity", "refusal_recall",
                "refusal_precision", "repair_rate", "latency_p95_ms", "mean_sources"):
        a_, b_ = s1.get(key), s2.get(key)
        if a_ is None or b_ is None:
            print(f"{key:<22}{str(a_):>12}{str(b_):>12}{'n/a':>12}")
        else:
            print(f"{key:<22}{a_:>12.3f}{b_:>12.3f}{b_ - a_:>+12.3f}")
    print(f"{'refusals (count)':<22}{s1['refused_total']:>12}{s2['refused_total']:>12}")

    moved = [{"id": r["id"], "kind": r["kind"], "before": v1[r["id"]]["correctness"],
              "after": r["correctness"]}
             for r in rows if r["id"] in v1 and v1[r["id"]]["correctness"] != r["correctness"]]
    print(f"\nquestions whose correctness moved: {len(moved)}")
    for m in moved:
        print(f"  {m['id']:<5} {m['kind']:<14} {m['before']} -> {m['after']}")

    p = ROOT / save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"v1_summary": s1, "v2_summary": s2, "moved": moved,
                             "v2_rows": rows}, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
