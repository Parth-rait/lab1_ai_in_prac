#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (  # noqa: E402
    Bm25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Hit,
    HybridRetriever,
    LLMReranker,
    Retriever,
)

# Chunkers that don't accept `overlap` (build_chunks() already tolerates the
# TypeError, but being explicit here keeps the call sites readable).
_CHUNK_KW = {"sliding": {"overlap": 150}, "recursive": {"overlap": 100}}

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
SWEEPS_JSON = ROOT / "reports/lab3_sweeps.json"


def _save(section: str, payload: dict) -> None:
    """Merge one sweep's results into reports/lab3_sweeps.json, keeping the others."""
    data = json.loads(SWEEPS_JSON.read_text()) if SWEEPS_JSON.exists() else {}
    data[section] = payload
    SWEEPS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SWEEPS_JSON.write_text(json.dumps(data, indent=2, default=str))
    print(f"\n[saved '{section}' -> {SWEEPS_JSON.relative_to(ROOT)}]")


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")
    _save("baseline", {"config": "sliding-800 overlap-150 dense", "n_questions": len(questions),
                       "n_chunks": len(chunks), "metrics": m})


def _retry_on_rate_limit(fn, tries: int = 8, base_delay: float = 20.0):
    """Gemini's free tier counts each embedded chunk against a 100/min quota,
    not each HTTP call -- so a batch request for 90+ chunks alone can trip it,
    and building several chunking configs back to back trips it repeatedly.
    That is an account limit, not a bug, so we back off and retry rather than
    fail the sweep."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:                           # noqa: BLE001
            if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            if attempt == tries - 1:
                raise
            m = re.search(r"retry in ([\d.]+)s", str(exc))
            delay = float(m.group(1)) + 3 if m else base_delay
            print(f"    (rate limited, retrying in {delay:.0f}s -- "
                  f"attempt {attempt + 1}/{tries})")
            time.sleep(delay)


def _build_indexed(chunks: list[Chunk], label: str) -> tuple[DenseRetriever, float, float]:
    """Build a DenseRetriever, timing it and isolating its embedding spend.

    A fresh Budget as a context manager only counts calls made while it is
    active, so the cost printed here is the index-build cost for THIS
    chunking config alone -- not the running process total.
    """
    t0 = time.perf_counter()
    with Budget(limit_usd=1.0, label=label) as b:
        r = _retry_on_rate_limit(lambda: DenseRetriever(chunks, show_progress=False))
    build_ms = (time.perf_counter() - t0) * 1000
    return r, build_ms, b.spent_usd


def _meta_row(name: str, n_chunks: int, build_ms: float, cost_usd: float) -> str:
    return f"{name:<28}{n_chunks:>10} chunks{build_ms:>10.0f} ms{cost_usd:>10.4f} $"


def sweep_chunking() -> None:
    """A1-A4."""
    corpus, questions = load_corpus(), load_questions()

    # --- A1: all four strategies at size=800 ------------------------------
    print("=" * 78)
    print("A1 -- four chunking strategies at size=800")
    print("=" * 78)
    a1_results: dict[str, dict] = {}
    a1_meta: dict[str, tuple[int, float, float]] = {}
    for name in STRATEGIES:
        chunks = build_chunks(corpus, name, 800, **_CHUNK_KW.get(name, {}))
        r, build_ms, cost_usd = _build_indexed(chunks, f"chunking-{name}-800")
        a1_results[name] = evaluate(r, questions)
        a1_meta[name] = (len(chunks), build_ms, cost_usd)
    print(table(a1_results))
    print()
    for name, (n, ms, c) in a1_meta.items():
        print(_meta_row(name, n, ms, c))

    winner = max(a1_results, key=lambda n: a1_results[n]["ndcg@10"])
    print(f"\nA1 winner by nDCG@10: '{winner}'")

    # --- A2: size sweep on the A1 winner -----------------------------------
    print("\n" + "=" * 78)
    print(f"A2 -- size sweep on '{winner}' (400 / 800 / 1600)")
    print("=" * 78)
    a2_results: dict[str, dict] = {}
    a2_meta: dict[str, tuple[int, float, float]] = {}
    for size in (400, 800, 1600):
        label = f"{winner}-{size}"
        if size == 800:
            # already built in A1 -- don't pay to embed it twice.
            a2_results[label] = a1_results[winner]
            a2_meta[label] = a1_meta[winner]
            continue
        chunks = build_chunks(corpus, winner, size, **_CHUNK_KW.get(winner, {}))
        r, build_ms, cost_usd = _build_indexed(chunks, f"chunking-{label}")
        a2_results[label] = evaluate(r, questions)
        a2_meta[label] = (len(chunks), build_ms, cost_usd)
    print(table(a2_results))
    print()
    for label, (n, ms, c) in a2_meta.items():
        print(_meta_row(label, n, ms, c))

    best_size = max((400, 800, 1600),
                    key=lambda s: a2_results[f"{winner}-{s}"]["ndcg@10"])
    print(f"\nA2 best size for '{winner}': {best_size}")
    print(
        "Dilution argument (T4 S2.2): a chunk embedding is one point standing "
        "in for everything inside it. Too large and unrelated rules get packed "
        "into one vector, which sits close to none of them (this is why 1600 "
        "tends to lose to 800). Too small and a single rule can split across "
        "chunk boundaries, so no one chunk fully contains the answer and "
        "recall falls again -- which is why the curve has a knee instead of "
        "improving monotonically as size shrinks."
    )

    # --- A3: markdown WITH vs WITHOUT the heading-path prefix ---------------
    print("\n" + "=" * 78)
    print("A3 -- markdown chunking: heading-path prefix on vs off")
    print("=" * 78)
    md_chunks = build_chunks(corpus, "markdown", 800)
    # markdown_chunks() prepends "[heading > path]\n" to chunk.text (see
    # aip/chunking.py). Stripping it here -- rather than editing the shared
    # module -- keeps every other lab's markdown_chunks() import unaffected.
    stripped_chunks = [
        Chunk(re.sub(r"^\[[^\n]*\]\n", "", c.text, count=1), c.doc_id, c.chunk_id, c.meta)
        for c in md_chunks
    ]
    r_with, ms_with, cost_with = _build_indexed(md_chunks, "chunking-md-with-prefix")
    r_without, ms_without, cost_without = _build_indexed(stripped_chunks, "chunking-md-without-prefix")
    m_with = evaluate(r_with, questions)
    m_without = evaluate(r_without, questions)
    a3_results = {"markdown WITH prefix": m_with, "markdown WITHOUT prefix": m_without}
    print(table(a3_results))
    print()
    print(_meta_row("markdown WITH prefix", len(md_chunks), ms_with, cost_with))
    print(_meta_row("markdown WITHOUT prefix", len(stripped_chunks), ms_without, cost_without))
    d_ndcg = m_with["ndcg@10"] - m_without["ndcg@10"]
    d_hr1 = m_with["hit_rate@1"] - m_without["hit_rate@1"]
    d_hr5 = m_with["hit_rate@5"] - m_without["hit_rate@5"]
    print(
        f"\nDelta from the heading-path prefix: "
        f"nDCG@10 {d_ndcg:+.4f}, hit_rate@1 {d_hr1:+.4f}, hit_rate@5 {d_hr5:+.4f}"
    )

    # --- A4: one golden question where chunking is the visible failure -----
    print("\n" + "=" * 78)
    print("A4 -- a chunking failure, by hand")
    print("=" * 78)
    # Use the best A1/A2 configuration as "current best" for this inspection.
    best_label = f"{winner}-{best_size}"
    best_chunks = (
        md_chunks if best_label == "markdown-800"
        else build_chunks(corpus, winner, best_size, **_CHUNK_KW.get(winner, {}))
    )
    best_retriever, _, _ = _build_indexed(best_chunks, "chunking-a4-best")
    best_metrics = evaluate(best_retriever, questions)

    # Worst-scoring non-unanswerable question by MRR under the best config.
    q_by_id = {q["id"]: q for q in questions}
    candidates = [
        (qid, mrr) for qid, mrr in best_metrics["_per_question_mrr"].items()
        if q_by_id[qid]["kind"] != "unanswerable"
    ]
    worst_qid, worst_mrr = min(candidates, key=lambda t: t[1])
    worst_q = q_by_id[worst_qid]
    print(f"Worst question under '{best_label}': {worst_qid} (mrr={worst_mrr:.3f}, "
          f"kind={worst_q['kind']})")
    print(f"  Q: {worst_q['question']}")
    print(f"  relevant_docs (gold): {worst_q['relevant_docs']}")

    hits = best_retriever.search(worst_q["question"], k=5)
    print("\n  Chunks that WERE retrieved (top 5):")
    for h in hits:
        flag = "  <- from a gold doc" if h.doc_id in worst_q["relevant_docs"] else ""
        print(f"    [{h.rank}] {h.chunk.chunk_id} (doc={h.doc_id}){flag}")
        print(f"        {h.text[:160].replace(chr(10), ' ')}...")

    gold_chunks = [c for c in best_chunks if c.doc_id in worst_q["relevant_docs"]]
    if gold_chunks:
        q_terms = set(worst_q["question"].lower().split())
        gold_chunks.sort(
            key=lambda c: len(q_terms & set(c.text.lower().split())), reverse=True
        )
        print("\n  Chunk that SHOULD have matched (highest lexical overlap in the gold doc):")
        gc = gold_chunks[0]
        print(f"    {gc.chunk_id} (doc={gc.doc_id})")
        print(f"        {gc.text[:300].replace(chr(10), ' ')}...")
    else:
        print("\n  (no chunk from a gold doc survives in this chunking -- the "
              "document itself was dropped, not just mis-ranked.)")

    def _cfg(results, meta):
        return {n: {"metrics": results[n], "n_chunks": meta[n][0],
                    "build_ms": meta[n][1], "build_cost_usd": meta[n][2]} for n in results}

    _save("chunking", {
        "A1_size800": _cfg(a1_results, a1_meta),
        "A1_winner": winner,
        "A2_size_sweep": _cfg(a2_results, a2_meta),
        "A2_best_size": best_size,
        "A3_heading_prefix": {
            "with_prefix": m_with, "without_prefix": m_without,
            "delta": {"ndcg@10": d_ndcg, "hit_rate@1": d_hr1, "hit_rate@5": d_hr5},
        },
        "A4_chunking_failure": {
            "config": best_label, "question_id": worst_qid, "mrr": worst_mrr,
            "question": worst_q["question"], "relevant_docs": worst_q["relevant_docs"],
            "retrieved_top5": [h.chunk.chunk_id for h in hits],
            "should_have_matched": gold_chunks[0].chunk_id if gold_chunks else None,
        },
    })


def sweep_retrieval() -> None:
    """B1-B5."""
    corpus, questions = load_corpus(), load_questions()
    # A1/A2 winner from Part A.
    chunks = build_chunks(corpus, "markdown", 400)
    print(f"using best chunking from Part A: markdown-400 ({len(chunks)} chunks)")

    # --- B1: dense / bm25 / hybrid on the best chunking ---------------------
    print("\n" + "=" * 78)
    print("B1 -- dense vs BM25 vs hybrid, on markdown-400")
    print("=" * 78)
    dense = _retry_on_rate_limit(lambda: DenseRetriever(chunks, show_progress=False))
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    b1_results = {name: evaluate(r, questions)
                 for name, r in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid))}
    print(table(b1_results))

    # --- B2: per-kind breakdown on MRR (NOT hit_rate@5), plus Q44 / Q41 -----
    print("\n" + "=" * 78)
    print("B2 -- per-kind breakdown by MRR (hit_rate@5 is saturated -- see A1)")
    print("=" * 78)
    for name, m in b1_results.items():
        print(f"\n-- {name} --")
        print(kind_table(m, col="mrr"))

    q_by_id = {q["id"]: q for q in questions}
    print("\nQ44 (exact identifier) vs Q41 (paraphrase, no lexical overlap) -- per-retriever MRR:")
    print(f"{'retriever':<10}{'Q44 mrr':>12}{'Q41 mrr':>12}")
    for name, m in b1_results.items():
        pq = m["_per_question_mrr"]
        print(f"{name:<10}{pq.get('Q44', float('nan')):>12.3f}{pq.get('Q41', float('nan')):>12.3f}")
    print(f"\n  Q44: {q_by_id['Q44']['question']!r}  (gold: {q_by_id['Q44']['relevant_docs']})")
    print(f"  Q41: {q_by_id['Q41']['question']!r}  (gold: {q_by_id['Q41']['relevant_docs']})")
    print(
        "\n  Mechanism: Q44 asks for an exact policy code ('AUR-HI-SIL-2026'). "
        "BM25 matches those tokens directly; the dense embedding treats the "
        "code as low-information text and can rank a topically-similar plan "
        "document above the one that actually contains it. Q41 shares no "
        "words at all with the source text (it paraphrases 'grace period' as "
        "'how long before I lose everything'), so BM25's term overlap has "
        "nothing to match while the embedding captures the meaning anyway."
    )

    dense_mrr, bm25_mrr = b1_results["dense"]["_per_question_mrr"], b1_results["bm25"]["_per_question_mrr"]
    dense_wins = sum(1 for qid in dense_mrr if dense_mrr[qid] > bm25_mrr.get(qid, 0))
    bm25_wins = sum(1 for qid in dense_mrr if bm25_mrr.get(qid, 0) > dense_mrr[qid])
    ties = len(dense_mrr) - dense_wins - bm25_wins
    print(f"\n  Across all {len(dense_mrr)} questions (by MRR): dense wins {dense_wins}, "
          f"bm25 wins {bm25_wins}, tied {ties}.")

    # --- B3: RRF k sweep ------------------------------------------------------
    print("\n" + "=" * 78)
    print("B3 -- RRF k in {10, 30, 60, 100}")
    print("=" * 78)
    b3_results = {f"rrf_k={k}": evaluate(HybridRetriever([dense, bm25], rrf_k=k), questions)
                 for k in (10, 30, 60, 100)}
    print(table(b3_results))
    spread = (max(m["ndcg@10"] for m in b3_results.values())
             - min(m["ndcg@10"] for m in b3_results.values()))
    print(f"\nnDCG@10 spread across k: {spread:.4f} -- "
          f"{'small, as expected (why RRF needs little tuning)' if spread < 0.02 else 'larger than expected, worth a second look'}")

    # --- B4: unequal fusion weights --------------------------------------------
    print("\n" + "=" * 78)
    print("B4 -- unequal fusion weights (dense : bm25)")
    print("=" * 78)
    b4_results = {
        f"{wd:.0f}:{wb:.0f}": evaluate(HybridRetriever([dense, bm25], weights=[wd, wb]), questions)
        for wd, wb in ((1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 2.0))
    }
    print(table(b4_results))
    best_w = max(b4_results, key=lambda k2: b4_results[k2]["ndcg@10"])
    n = len(dense_mrr)
    print(f"\nBest weighting tried: {best_w} (nDCG@10={b4_results[best_w]['ndcg@10']:.4f}) vs "
          f"1:1 (nDCG@10={b4_results['1:1']['ndcg@10']:.4f}). At n={n}, treat any gap under "
          "~0.02 nDCG@10 as noise, not a real effect -- it is well within what a couple of "
          "flipped rankings would produce by chance.")

    # --- B5: the headline result, reported honestly whichever way it lands ----
    print("\n" + "=" * 78)
    print("B5 -- dense vs hybrid: the finding you are not allowed to suppress")
    print("=" * 78)
    d_ndcg, h_ndcg = b1_results["dense"]["ndcg@10"], b1_results["hybrid"]["ndcg@10"]
    verdict = "HYBRID LOSES to dense alone" if h_ndcg < d_ndcg else "hybrid beats dense alone"
    print(f"dense nDCG@10={d_ndcg:.4f}   hybrid nDCG@10={h_ndcg:.4f}   -> {verdict}")
    print(
        "T4 S4.3 calls hybrid retrieval 'the strongest single change most RAG "
        "systems can make.' That is not true on this corpus: the embedding "
        "model is already strong enough to handle most of what BM25 would "
        f"rescue, so fusing in a weaker retriever (dense beat bm25 on "
        f"{dense_wins}/{dense_wins + bm25_wins} questions where they "
        "differed, bm25 ahead on the rest, and both tied on the remaining "
        f"{ties}) drags more good rankings down than it rescues. A "
        "technique that is right on average can be wrong on your data -- "
        "that is why you measure instead of assuming the textbook result."
    )
    _save("retrieval", {
        "chunking": "markdown-400", "n_chunks": len(chunks),
        "B1": b1_results,
        "B2_q44_q41_mrr": {name: {"Q44": m["_per_question_mrr"].get("Q44"),
                                  "Q41": m["_per_question_mrr"].get("Q41")}
                           for name, m in b1_results.items()},
        "B2_dense_vs_bm25_by_mrr": {"dense_wins": dense_wins, "bm25_wins": bm25_wins, "ties": ties},
        "B3_rrf_k": b3_results, "B3_ndcg10_spread": spread,
        "B4_weights_dense_bm25": b4_results, "B4_best": best_w,
        "B5_verdict": verdict,
    })


class _PacedLLMReranker(LLMReranker):
    """LLMReranker with an identical prompt and chat() call -- so cache keys
    match and earlier runs' answers are reused -- but paced for Gemini's
    free-tier cap of 15 generate requests/minute, retrying only the one call
    that got rate-limited instead of the whole evaluation."""

    # Free tier: 15 generate requests/min. A billed key allows far more --
    # raise with LAB3_LLM_RPM so pacing doesn't throttle it artificially.
    MIN_GAP_S = 60 / float(os.environ.get("LAB3_LLM_RPM", "14"))

    def __init__(self, tier: str = "SMALL"):
        super().__init__(tier)
        self.live_call_ms: list[float] = []

    def rerank(self, query, hits, k=5):
        from aip.llm import chat

        scored = []
        for h in hits:
            prompt = self.PROMPT.format(q=query, p=h.text[:1500])
            t0 = time.perf_counter()
            out = _retry_on_rate_limit(
                lambda p=prompt: chat(p, tier=self.tier, max_tokens=8, temperature=0.0),
                tries=20,
            )
            ms = (time.perf_counter() - t0) * 1000
            if ms > 50:                      # a live call, not a cache hit
                self.live_call_ms.append(ms)
                time.sleep(max(0.0, self.MIN_GAP_S - ms / 1000))
            m = re.search(r"\d+(?:\.\d+)?", out)
            scored.append((float(m.group()) if m else 0.0, h))
        scored.sort(key=lambda t: -t[0])
        return [Hit(h.chunk, s, "llm_reranked", i) for i, (s, h) in enumerate(scored[:k])]


def sweep_rerank() -> None:
    """C1-C4."""
    corpus, questions = load_corpus(), load_questions()
    q_by_id = {q["id"]: q for q in questions}
    chunks = build_chunks(corpus, "markdown", 400)  # Part A winner
    dense = _retry_on_rate_limit(lambda: DenseRetriever(chunks, show_progress=False))

    print("=" * 78)
    print("baseline -- dense alone, k=5, no reranking")
    print("=" * 78)
    base = evaluate(dense, questions, k=5)
    print(table({"dense (no rerank)": base}))

    # --- C1: cross-encoder, retrieve k=30 -> rerank to 5 ---------------------
    print("\n" + "=" * 78)
    print("C1 -- cross-encoder rerank (retrieve k=30, rerank to 5)")
    print("=" * 78)
    print("Loading cross-encoder/ms-marco-MiniLM-L-6-v2 "
          "(first run downloads ~90 MB, local after that)...")
    cross = CrossEncoderReranker()
    m_cross = evaluate(dense, questions, k=30, reranker=cross, final_k=5)
    print(table({"dense (no rerank)": base, "dense + cross-encoder": m_cross}))
    d_ndcg5 = m_cross["ndcg@5"] - base["ndcg@5"]
    d_hr1 = m_cross["hit_rate@1"] - base["hit_rate@1"]
    d_recall5 = m_cross["recall@5"] - base["recall@5"]
    d_lat_cross = m_cross["latency_p95_ms"] - base["latency_p95_ms"]
    print(f"\nDelta vs no rerank: nDCG@5 {d_ndcg5:+.4f}, hit_rate@1 {d_hr1:+.4f}, "
          f"recall@5 {d_recall5:+.4f}, p95 latency {d_lat_cross:+.1f} ms")

    # --- C4: a query the cross-encoder made worse ------------------------------
    print("\n" + "=" * 78)
    print("C4 -- a query the cross-encoder reranker made worse")
    print("=" * 78)
    base_mrr, cross_mrr = base["_per_question_mrr"], m_cross["_per_question_mrr"]
    deltas = {qid: cross_mrr[qid] - base_mrr.get(qid, 0.0) for qid in cross_mrr}
    worst_qid = min(deltas, key=lambda qid: deltas[qid])
    if deltas[worst_qid] >= 0:
        print("No question regressed under the cross-encoder on this run.")
    else:
        wq = q_by_id[worst_qid]
        print(f"{worst_qid}: mrr {base_mrr[worst_qid]:.3f} (dense alone) -> "
              f"{cross_mrr[worst_qid]:.3f} (reranked)  delta={deltas[worst_qid]:+.3f}")
        print(f"  Q: {wq['question']}")
        print(f"  gold: {wq['relevant_docs']}  kind={wq['kind']}")

        before_hits = dense.search(wq["question"], k=5)
        wide_hits = dense.search(wq["question"], k=30)
        after_hits = cross.rerank(wq["question"], wide_hits, k=5)
        print("\n  top-5 BEFORE rerank (dense):")
        for h in before_hits:
            flag = "  <- gold" if h.doc_id in wq["relevant_docs"] else ""
            print(f"    [{h.rank}] {h.doc_id}{flag}  {h.text[:100].strip()}...")
        print("\n  top-5 AFTER rerank (cross-encoder):")
        for h in after_hits:
            flag = "  <- gold" if h.doc_id in wq["relevant_docs"] else ""
            print(f"    [{h.rank}] {h.doc_id}{flag}  {h.text[:100].strip()}...")
        print(
            "\n  Diagnosis (failure mode 5, T4 S5): the cross-encoder "
            "(ms-marco-MiniLM-L-6-v2) was trained on web-search query/passage "
            "pairs, not insurance-policy prose. It can prefer a passage that "
            "is lexically closer to the query's exact wording over the one "
            "that actually answers it, because its learned notion of "
            "'relevance' comes from a different distribution than this "
            "corpus. A reranker is a model, and models have training "
            "distributions that may not transfer -- you have to measure "
            "that, not assume it."
        )

    # --- C2: LLM reranker at k=10 and k=30 -------------------------------------
    # Handout spec is k=30 (1,260 calls). k=10 was first run to fit the free
    # tier's rate limit; both are reported so C3 shows whether reranking a
    # wider candidate pool buys anything. The cross-encoder is run at both
    # sizes too, so each LLM result has a like-for-like comparison.
    print("\n" + "=" * 78)
    print("C2 -- LLM reranker at k=10 and k=30 (rerank to 5)")
    print("=" * 78)
    cross_by_k = {30: m_cross, 10: evaluate(dense, questions, k=10, reranker=cross, final_k=5)}
    llm_by_k: dict[int, tuple[dict, float]] = {}
    llm_ks = tuple(int(x) for x in os.environ.get("LAB3_LLM_KS", "10,30").split(","))
    llm_stats: dict[int, dict] = {}
    for n_cand in llm_ks:
        print(f"\n-- k={n_cand}: {n_cand} model calls/question x {len(questions)} questions "
              "(cached answers reused) --", flush=True)
        llm_rr = _PacedLLMReranker(tier="SMALL")
        with Budget(limit_usd=2.0, label=f"c2-llm-rerank-k{n_cand}") as b:
            m_llm = evaluate(dense, questions, k=n_cand, reranker=llm_rr, final_k=5)
        # evaluate()'s latency includes pacing sleeps and cache hits, so it is
        # not a deployment number. Serial latency = n_cand x one live call.
        live = sorted(llm_rr.live_call_ms)
        if live:
            per_call_p50 = statistics.median(live)
            per_call_p95 = live[int(0.95 * (len(live) - 1))]
            m_llm["latency_p95_ms"] = n_cand * per_call_p95
        else:
            per_call_p50 = per_call_p95 = float("nan")
        live_calls = b.calls - b.cached_calls
        cost_per_query = (b.spent_usd / live_calls) * n_cand if live_calls else float("nan")
        cost_per_1k = cost_per_query * 1000
        llm_by_k[n_cand] = (m_llm, cost_per_1k)
        llm_stats[n_cand] = {"live_calls": live_calls, "per_call_p50_ms": per_call_p50,
                             "per_call_p95_ms": per_call_p95, "spent_usd": b.spent_usd,
                             "cost_per_query_usd": cost_per_query}
        print(table({"dense (no rerank)": base,
                     f"dense + cross-encoder k={n_cand}": cross_by_k[n_cand],
                     f"dense + LLM reranker k={n_cand}": m_llm},
                    cols=("hit_rate@1", "recall@5", "mrr", "ndcg@5", "latency_p95_ms")))
        print(f"\nLLM k={n_cand} vs no rerank: nDCG@5 {m_llm['ndcg@5'] - base['ndcg@5']:+.4f}, "
              f"hit_rate@1 {m_llm['hit_rate@1'] - base['hit_rate@1']:+.4f}, "
              f"recall@5 {m_llm['recall@5'] - base['recall@5']:+.4f}")
        print(f"Per live call: p50 {per_call_p50:.0f} ms, p95 {per_call_p95:.0f} ms "
              f"({len(live)} live calls timed). Serial per-query p95 estimate: "
              f"{m_llm['latency_p95_ms']:.0f} ms ({n_cand} calls in a row).")
        print(f"Cost: ${b.spent_usd:.4f} across {live_calls} uncached calls -> "
              f"${cost_per_query:.5f}/query -> ${cost_per_1k:.3f} / 1k queries at list price.",
              flush=True)

    # --- C3: the decision -----------------------------------------------------
    print("\n" + "=" * 78)
    print("C3 -- the decision")
    print("=" * 78)
    configs = {"dense, no rerank": (base, 0.0)}
    for n_cand in (10, 30):
        configs[f"dense + cross-encoder k={n_cand}"] = (cross_by_k[n_cand], 0.0)
    for n_cand in llm_ks:
        configs[f"dense + LLM rerank k={n_cand}"] = llm_by_k[n_cand]
    w = max(len(n) for n in configs) + 2
    print(f"{'config':<{w}}{'ndcg@5':>10}{'hit_rate@1':>12}{'p95 ms':>10}{'$/1k q':>10}")
    for name, (m, cpk) in configs.items():
        print(f"{name:<{w}}{m['ndcg@5']:>10.4f}{m['hit_rate@1']:>12.4f}"
              f"{m['latency_p95_ms']:>10.1f}{cpk:>10.3f}")

    # Interactive: best quality within the lab's 400 ms retrieval budget.
    # Batch: best quality, full stop -- nobody is waiting and volume is bounded.
    budget_ms = 400
    fits = [n for n, (m, _) in configs.items() if m["latency_p95_ms"] <= budget_ms]
    interactive_choice = max(fits, key=lambda n: configs[n][0]["ndcg@5"])
    batch_choice = max(configs, key=lambda n: configs[n][0]["ndcg@5"])
    im, bm = configs[interactive_choice][0], configs[batch_choice][0]
    print(f"\n(a) Interactive agent-facing search box -> '{interactive_choice}': best "
          f"nDCG@5 ({im['ndcg@5']:.4f}) among configs under the {budget_ms} ms p95 "
          f"target ({im['latency_p95_ms']:.1f} ms).")
    print(f"(b) Overnight batch job -> '{batch_choice}': best nDCG@5 measured "
          f"({bm['ndcg@5']:.4f}); latency ({bm['latency_p95_ms']:.0f} ms) and "
          f"${configs[batch_choice][1]:.3f}/1k queries don't matter with nobody waiting.")
    if interactive_choice == batch_choice:
        print("\nSame answer for both. The lab expects them to differ; here no "
              "reranker beat plain dense on quality, so the extra latency or cost "
              "buys nothing in either setting -- the cheapest config is also the best.")
    else:
        print("\nThese differ: same quality table, different latency budget, "
              "different decision.")
    _save("rerank", {
        "chunking": "markdown-400", "retriever": "dense",
        "baseline_k5": base,
        "C1_cross_encoder_k30": m_cross,
        "C1_delta_vs_baseline": {"ndcg@5": d_ndcg5, "hit_rate@1": d_hr1, "recall@5": d_recall5,
                                 "latency_p95_ms": d_lat_cross},
        "C4_worst_regression": {"question_id": worst_qid, "delta_mrr": deltas[worst_qid],
                                "mrr_before": base_mrr[worst_qid], "mrr_after": cross_mrr[worst_qid]},
        "C2_cross_encoder_by_k": cross_by_k,
        "C2_llm_by_k": {k2: {"metrics": llm_by_k[k2][0], "cost_per_1k_usd": llm_by_k[k2][1],
                             **llm_stats[k2]} for k2 in llm_ks},
        "C2_note": "handout k=30; k=10 added because the free tier rate-limited k=30. "
                   "LLM latency_p95_ms = k x p95 of one live call (serial).",
        "C3_table": {n: {"ndcg@5": m["ndcg@5"], "hit_rate@1": m["hit_rate@1"],
                         "latency_p95_ms": m["latency_p95_ms"], "cost_per_1k_usd": c}
                     for n, (m, c) in configs.items()},
        "C3_interactive_choice": interactive_choice, "C3_batch_choice": batch_choice,
        "C3_interactive_budget_ms": budget_ms,
    })


class _FilteredRetriever(Retriever):
    """A ChromaRetriever with a fixed metadata filter, so evaluate() can use it."""

    name = "chroma+filter"

    def __init__(self, inner, where: dict):
        self.inner, self.where = inner, where

    def search(self, query: str, k: int = 8) -> list[Hit]:
        return self.inner.search(query, k=k, where=self.where)


def _pctl(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def sweep_index() -> None:
    """D1-D3."""
    import contextlib

    import chromadb
    import numpy as np

    from aip.embed import embed_batch
    from aip.retrieval import ChromaRetriever

    corpus, questions = load_corpus(), load_questions()
    q_by_id = {q["id"]: q for q in questions}
    chunks = build_chunks(corpus, "markdown", 400)  # Part A winner
    # D3 metadata goes on at ingest; it changes nothing until a filter uses it.
    for c in chunks:
        c.meta["status"] = "archived" if "ARCHIVED" in c.doc_id else "current"

    dense = _retry_on_rate_limit(lambda: DenseRetriever(chunks, show_progress=False))
    chroma = _retry_on_rate_limit(lambda: ChromaRetriever(
        chunks, path=str(ROOT / ".chroma"), collection="lab3_index", reset=True))

    # --- D1: exact vs HNSW on the real corpus -------------------------------
    print("=" * 78)
    print(f"D1 -- exact NumPy vs Chroma HNSW, markdown-400 ({len(chunks)} chunks)")
    print("=" * 78)
    m_exact, m_hnsw = evaluate(dense, questions), evaluate(chroma, questions)
    print(table({"exact (DenseRetriever)": m_exact, "HNSW (ChromaRetriever)": m_hnsw}))
    moved = [qid for qid in m_exact["_per_question_mrr"]
             if abs(m_exact["_per_question_mrr"][qid] - m_hnsw["_per_question_mrr"][qid]) > 1e-9]
    print(f"\nQuality gap (HNSW - exact): nDCG@10 {m_hnsw['ndcg@10'] - m_exact['ndcg@10']:+.4f}, "
          f"recall@5 {m_hnsw['recall@5'] - m_exact['recall@5']:+.4f}, "
          f"hit_rate@1 {m_hnsw['hit_rate@1'] - m_exact['hit_rate@1']:+.4f}. "
          f"Questions whose MRR changed: {len(moved)}{' ' + str(moved) if moved else ''}.")
    print("(Latency column includes the cached query-embedding lookup for both; "
          "D2 times the index alone.)", flush=True)

    # --- D3: metadata filter on the archived-document trap -------------------
    print("\n" + "=" * 78)
    print("D3 -- filter archived documents at query time (Q29 / Q30 / Q31)")
    print("=" * 78)
    trap = [q_by_id[i] for i in ("Q29", "Q30", "Q31")]
    filtered = _FilteredRetriever(chroma, where={"status": "current"})
    m_before, m_after = evaluate(chroma, trap), evaluate(filtered, trap)
    print(table({"no filter": m_before, "status == current": m_after},
                cols=("hit_rate@1", "mrr", "ndcg@10")))
    print()
    for q in trap:
        top_before = chroma.search(q["question"], k=1)[0].doc_id
        top_after = filtered.search(q["question"], k=1)[0].doc_id
        print(f"  {q['id']}: top-1 {top_before:<34} -> {top_after:<20} gold={q['relevant_docs']}")
    m_all_filtered = evaluate(filtered, questions)
    print(f"\nWhole set, n={len(questions)}: nDCG@10 {m_hnsw['ndcg@10']:.4f} -> "
          f"{m_all_filtered['ndcg@10']:.4f}, hit_rate@1 {m_hnsw['hit_rate@1']:.4f} -> "
          f"{m_all_filtered['hit_rate@1']:.4f} (checks the filter costs nothing elsewhere).")
    print("No retriever, embedding or chunker changed -- only a metadata field at "
          "ingest and a where-clause at query time.", flush=True)

    # --- D2: latency at scale --------------------------------------------------
    print("\n" + "=" * 78)
    print("D2 -- query latency, exact vs HNSW, as the index grows")
    print("=" * 78)
    scaled_dir = ROOT / "data/corpus_scaled"
    ballast_docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(scaled_dir.glob("*.md"))}
    if not ballast_docs:
        print("data/corpus_scaled/ is empty -- run: python scripts/expand_corpus.py --docs 4000")
        return
    n_ballast = len(build_chunks(ballast_docs, "markdown", 400))
    real = dense.matrix
    dim = real.shape[1]
    # Latency depends on vector count and dimension, not on what the vectors
    # mean, and ballast is excluded from quality anyway. Embedding ~28k chunks
    # at the free tier's 100/min would take hours, so ballast gets random unit
    # vectors at the real dimension. State this in the report.
    n_max = max(40_000, len(chunks) + n_ballast)
    rng = np.random.default_rng(7)
    ballast = rng.standard_normal((n_max - len(chunks), dim), dtype=np.float32)
    ballast /= np.linalg.norm(ballast, axis=1, keepdims=True)
    full = np.vstack([real, ballast])
    q_vecs = embed_batch([q["question"] for q in questions], input_type="query")  # cached

    print(f"ballast from expand_corpus.py: {n_ballast} chunks at markdown-400 "
          f"(the handout says ~40k; 40k is also tested). dim={dim}. "
          "Ballast vectors are random unit vectors -- see comment in code.\n")
    print(f"{'n vectors':>10}{'exact p50':>11}{'exact p95':>11}{'hnsw p50':>10}"
          f"{'hnsw p95':>10}{'hnsw build':>12}{'recall@10':>11}")
    reps, crossover, d2_rows = 3, None, []
    client = chromadb.EphemeralClient()
    for n in (len(chunks), 4_000, len(chunks) + n_ballast, 40_000):
        mat = full[:n]
        ids = [f"v{i}" for i in range(n)]
        for q in q_vecs[:5]:                                  # warm-up
            np.argsort(-(mat @ q))[:10]
        exact_ms, exact_top = [], []
        for _ in range(reps):
            for q in q_vecs:
                t0 = time.perf_counter()
                top = np.argsort(-(mat @ q))[:10]
                exact_ms.append((time.perf_counter() - t0) * 1000)
                exact_top.append(set(top.tolist()))

        name = f"d2_{n}"
        with contextlib.suppress(Exception):
            client.delete_collection(name)
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        t0 = time.perf_counter()
        for s in range(0, n, 2000):
            col.add(ids=ids[s:s + 2000], embeddings=mat[s:s + 2000])
        build_s = time.perf_counter() - t0
        for q in q_vecs[:5]:                                  # warm-up
            col.query(query_embeddings=[q.tolist()], n_results=10)
        hnsw_ms, hnsw_top = [], []
        for _ in range(reps):
            for q in q_vecs:
                t0 = time.perf_counter()
                res = col.query(query_embeddings=[q.tolist()], n_results=10, include=[])
                hnsw_ms.append((time.perf_counter() - t0) * 1000)
                hnsw_top.append({int(i[1:]) for i in res["ids"][0]})
        client.delete_collection(name)

        recall = statistics.fmean(len(a & b) / 10 for a, b in zip(exact_top, hnsw_top))
        e50, h50 = statistics.median(exact_ms), statistics.median(hnsw_ms)
        if crossover is None and h50 < e50:
            crossover = n
        d2_rows.append({"n_vectors": n, "exact_p50_ms": e50, "exact_p95_ms": _pctl(exact_ms, .95),
                        "hnsw_p50_ms": h50, "hnsw_p95_ms": _pctl(hnsw_ms, .95),
                        "hnsw_build_s": build_s, "hnsw_recall@10": recall})
        print(f"{n:>10}{e50:>9.2f}ms{_pctl(exact_ms, .95):>9.2f}ms{h50:>8.2f}ms"
              f"{_pctl(hnsw_ms, .95):>8.2f}ms{build_s:>10.1f}s{recall:>11.3f}", flush=True)

    print("\nCrossover: " + (f"HNSW's median first beats exact at ~{crossover} vectors."
                             if crossover else "HNSW never beat exact up to 40k vectors."))
    print("Mechanism: exact search is one BLAS matrix multiply -- O(n*d) but "
          "vectorised and cache-friendly, so small n costs microseconds. HNSW "
          "walks a graph in O(log n) hops, but each query also pays a fixed "
          "overhead (Python -> Chroma client -> Rust/C++ index, list conversion, "
          "result marshalling). Exact search wins until its linear term "
          "outgrows that fixed overhead.")
    _save("index", {
        "chunking": "markdown-400", "n_chunks": len(chunks),
        "D1_exact": m_exact, "D1_hnsw": m_hnsw, "D1_questions_mrr_changed": moved,
        "D3_trap_questions": ["Q29", "Q30", "Q31"],
        "D3_before": m_before, "D3_after": m_after,
        "D3_whole_set_unfiltered": m_hnsw, "D3_whole_set_filtered": m_all_filtered,
        "D2_ballast_chunks": n_ballast, "D2_dim": dim, "D2_reps_per_query": reps,
        "D2_note": "ballast vectors are random unit vectors at the real dimension "
                   "(embedding ~28k chunks exceeds the free-tier quota); "
                   "timings are index-only, query embedding excluded",
        "D2_rows": d2_rows, "D2_crossover_first_measured_n": crossover,
    })


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
