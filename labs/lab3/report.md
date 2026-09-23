# Lab 3 — Semantic Search That Actually Works: Report

Numbers are from `reports/lab3_sweeps.json`, written by `labs/lab3/search.py`; `AIP_PROFILE=gemini`,
embeddings `gemini-embedding-001` (3072-d), LLM reranker `gemini-3.5-flash-lite`.

**n = 42.** Q36, Q38 and Q39 have no relevant document, so recall and nDCG are undefined for them
and they are excluded. Scoring is per document: a document counts at rank r if any of its chunks
does. **Metric:** `hit_rate@5` is 0.93–1.00 for every configuration tested, so it never decides
anything; decisions use nDCG@10 (nDCG@5 for reranking), MRR, recall@5 and hit_rate@1.

## Recommended configuration

Markdown-aware chunking at 400 characters (heading path kept), dense retrieval, exact NumPy
search, no reranker, archived documents excluded via `status` metadata.

| | nDCG@10 | recall@5 | hit_rate@1 | MRR | paraphrase MRR | search p95 | index cost |
|---|---|---|---|---|---|---|---|
| Baseline (sliding-800 dense) | 0.8053 | 0.8452 | 0.7857 | 0.8451 | — | — | — |
| **Recommended** | **0.8600** | **0.9028** | **0.8095** | **0.8919** | **0.800** | 0.03 ms + query embed | $0.0013 |
| Target | ≥ 0.80 | ≥ 0.85 | ≥ 0.65 | — | ≥ 0.75 | ≤ 400 ms | report |

All targets met. For an overnight batch job, add the LLM reranker (C3). Three textbook techniques
lost here and are reported as losses: hybrid retrieval, the cross-encoder and HNSW.

## Part A — chunking

| A1: strategy @ 800 | chunks | nDCG@10 | recall@5 | hit_rate@1 | MRR | build cost¹ |
|---|---|---|---|---|---|---|
| fixed | 83 | 0.7952 | 0.8373 | 0.7381 | 0.8387 | cached |
| sliding | 91 | 0.8053 | 0.8452 | 0.7857 | 0.8451 | cached |
| recursive | 98 | 0.8251 | 0.8750 | 0.7619 | 0.8611 | $0.0022 |
| **markdown-aware** | 164 | **0.8458** | **0.8988** | 0.7619 | **0.8720** | $0.0021 |

| A2: markdown @ size | chunks | nDCG@10 | recall@5 | hit_rate@1 |
|---|---|---|---|---|
| **400** | 235 | **0.8527** | **0.9028** | **0.7857** |
| 800 | 164 | 0.8458 | 0.8988 | 0.7619 |
| 1600 | 150 | 0.8075 | 0.8750 | 0.7143 |

¹ First cold run; later reruns were fully cached, so the JSON shows $0.

**The size curve and dilution (T4 §2.2).** A chunk's embedding is one point standing in for
everything inside it; pack several rules together and the vector sits between them, close to none.
The markdown chunker never merges sections, only splits long ones, so going from 800 to 1600 keeps
just 12 sections whole, and they are the multi-topic ones: the glossary, the FAQ, the ombudsman
office list, the exclusions list and the plan-comparison table. Those 12 vectors cost 3.8 points,
against 0.7 from 400 to 800. Over these three sizes the curve is monotonic; the knee where chunks
become too small to hold a whole rule must lie below 400 and was not measured.

**A3. Heading-path prefix** (markdown @ 800): with vs without, nDCG@10 **+0.054**, hit_rate@1
**+0.143**, hit_rate@5 **−0.024**. It improves ranking, not recall: plans repeat the same headings
("Co-payment", "Sum insured"), and the path is often the only text saying *which* plan a chunk is.

**A4. A chunking failure: Q15, "What is a day-care procedure?"** (gold `glossary`, MRR 0.333).
`plan-bronze::m7` "[Aurora Bronze > Day-care procedures] 150 listed day-care procedures are
covered…" ranks first; `glossary::m1`, which holds the definition, ranks third. The glossary chunk
packs several unrelated definitions under one `[Glossary]` path, so its vector is an average of all
of them (failure mode 2, T4 §5). One definition per chunk would fix it without touching retrieval.

## Part B — dense vs BM25 vs hybrid (markdown-400)

| B1/B2: MRR by kind | n | dense | BM25 | hybrid (RRF k=60) |
|---|---|---|---|---|
| single_hop | 18 | 0.907 | 0.852 | **0.944** |
| multi_hop | 10 | **1.000** | 0.650 | 0.817 |
| paraphrase | 5 | **0.800** | 0.487 | 0.650 |
| aggregation | 4 | **0.875** | 0.375 | 0.583 |
| trap_archived | 3 | **0.833** | 0.511 | 0.667 |
| unanswerable (with docs) | 2 | 0.313 | 0.417 | 0.375 |
| **all: nDCG@10** | 42 | **0.8527** | 0.6978 | 0.7949 |
| **all: MRR** | 42 | **0.8800** | 0.6698 | 0.7976 |

| Per-question MRR | dense | BM25 | hybrid |
|---|---|---|---|
| **Q44** "AUR-HI-SIL-2026 — what are the sum insured options?" | 0.50 | **1.00** | 1.00 |
| **Q41** "If I skip paying on time, how long before I lose everything I've built up?" | **1.00** | 0.00 | 0.25 |

**Q44:** BM25 matches the rare token `sil` in Silver's product code and ranks `plan-silver` first.
The embedding treats the code as noise and matches the meaning "sum insured options", putting
`topup-and-super-topup` first. **Q41:** the question shares no words with "grace period", so BM25
returns FAQ chunks and scores 0, while the embedding maps "skip paying on time" to the grace-period
meaning and ranks `policy-renewal-and-portability` first.

**B5: hybrid loses, by 5.8 nDCG@10 points.** Where dense and BM25 disagree on MRR, dense wins
**20 of 26** (16 tie). RRF averages ranks, so fusing a retriever that is worse on 77% of
disagreements drags down more good rankings (Q41) than it rescues (Q44). Hybrid wins only
`single_hop`, where BM25 is also strong. **B3:** RRF k=10/30/60/100 gives 0.816/0.795/0.795/0.790,
favouring small k (more trust in each list's top rank); even the best is 3.7 points below dense.
**B4:** dense:BM25 weights 1:1/2:1/3:1/1:2 give 0.795/0.807/0.814/0.793. Up-weighting dense helps,
but 0.019 at n=42 is two or three flipped rankings: this trends towards dense-only, it is not a
real win. Routing identifier-shaped queries to BM25 would keep Q44 without losing Q41.

## Part C — reranking (dense, markdown-400, rerank to 5)

**Deviation: the LLM reranker ran at k=10, not the handout's k=30.** k=30 needs 1,260 calls; the
free tier throttled this key to ~5 calls/minute and then stalled, so k=30 could not finish. The
cross-encoder was also run at k=10 for a like-for-like comparison.

| C3 decision table | nDCG@5 | hit_rate@1 | recall@5 | p95 per query | $/1k queries |
|---|---|---|---|---|---|
| dense, no rerank | 0.8313 | 0.7857 | 0.8909 | **1 ms** | $0 |
| + cross-encoder k=30 (C1) | 0.8174 | 0.7619 | 0.8889 | 192 ms | $0 |
| + cross-encoder k=10 | 0.8139 | 0.7381 | 0.8849 | 150 ms | $0 |
| **+ LLM reranker k=10 (C2)** | **0.8715** | **0.8810** | **0.8988** | 9,919 ms² | $0.41³ |

² One live call p95 992 ms × 10 calls made in sequence; `evaluate()`'s own timing includes pacing sleeps.
³ $0.0073 over 180 uncached calls at list price; the free tier billed $0.

**C1.** The cross-encoder (`ms-marco-MiniLM-L-6-v2`) is worse on every metric at both k and adds
103–192 ms across runs. It was trained on web search, not policy prose.

**C2.** The LLM reranker is the best configuration in the lab: +4.0 nDCG@5, +9.5 hit_rate@1. It
fixed four questions, including Q15 from A4 (0.33 → 1.0), because an LLM recognises a definition
that a diluted embedding missed. It broke one, **Q44 (0.5 → 0.0)**. The only chunk containing the
code is Silver's header, which has no sum-insured facts, so it scores 0. The chunk with the answer
never mentions the code, so it ties at 10 with six other plans' "Sum insured" chunks. Ties keep dense
order, and Silver's answer chunk (dense rank 8) falls out of the top 5.

**C3. The two deployments.**
- **(a) Interactive agent search box: dense, no reranker.** It is the best quality inside the
  400 ms target. The only better option takes ~10 s per search; the cross-encoder is slower *and* worse.
- **(b) Overnight batch: dense + LLM reranker k=10.** Nobody waits: 10,000 queries take ~28 h
  serially or well under an hour in parallel, and +9.5 points of hit_rate@1 costs ~$4.

Parallelising the 10 calls would bring (b) to ~1 s per query, still 2.5× over target, and every
search would become a paid, rate-limited external call. Fixing the chunking the LLM compensates for
(Q15, Q44) is the cheaper route to the same gain.

**C4. A query reranking made worse: Q32, "Which plans have no co-payment?"**, cross-encoder, MRR
1.0 → 0.2. Dense ranked `plan-gold` ("Co-payment: **Nil** at all ages") first. The cross-encoder
put Silver (10%), Senior Care (20%), a corporate-plan table and Bronze (20%) above it, and pushed
the second gold document out of the top 5. Wrong answers repeat the query's word "co-payment";
the right one answers with "Nil", which the model does not connect to "no" (failure mode 5, T4 §5).

## Part D — index and metadata

**D1.** At 235 chunks, HNSW (Chroma) and exact search give **identical** results on every metric,
and no question's MRR changes. End to end HNSW is ~5× slower (3.58 ms vs 0.74 ms p95).

| D2: index-only p50 | 235 | 4,000 | 28,235⁴ | 40,000 |
|---|---|---|---|---|
| exact | **0.03 ms** | **1.11 ms** | 9.46 ms | 12.47 ms |
| HNSW | 1.48 ms | 3.07 ms | **6.52 ms** | **7.65 ms** |
| HNSW build | 0.1 s | 8.3 s | 116 s | 161 s |

⁴ `expand_corpus.py --docs 4000` produced 28,000 ballast chunks, not ~40k, so both sizes were
tested. Ballast vectors are random unit vectors: latency depends only on count and dimension, and
embedding 28k chunks on the free tier would take ~5 h. HNSW recall@10 was 1.000 at every size, but
that is optimistic, because random vectors never crowd a real query's neighbourhood.

**Crossover ~14k vectors** (interpolated between 4k and 28k). Exact search is one vectorised matrix
multiply, so its cost grows linearly from microseconds. HNSW checks only a small fraction of
vectors, but every query pays ~1.5 ms fixed overhead crossing from Python into Chroma's native index
and back. **Decision:** exact search at this size; HNSW only past ~15k chunks.

**D3. Archived-document trap.** Chunks get `status = archived` if the document id contains
`ARCHIVED`, else `current`, and queries filter on `current`.

| Q29–Q31 | hit_rate@1 | MRR | nDCG@10 |
|---|---|---|---|
| no filter | 0.667 | 0.833 | 0.898 |
| `status == current` | **1.000** | **1.000** | **1.000** |

Only **Q30** was wrong: its top result was `claims-timelines-2024-ARCHIVED` and is now
`claims-timelines`. Across all 42 questions, nDCG@10 goes 0.8527 → 0.8600 and hit_rate@1 0.7857 →
0.8095, and **no other question's MRR changed**. The retriever was not wrong: the archived file
really is the most similar text, with out-of-date numbers. No embedding model can separate current
from stale when both say the same thing; only metadata can. **When retrieval is plausibly wrong,
check the data before tuning the model.**

## What the greedy sweep could have missed

Each axis was fixed before the next was tested. **Chunking × retriever:** markdown-400 was chosen
under dense retrieval, and BM25 may prefer larger chunks, so hybrid was only tested on chunking that
suits its dense half. **Chunking × reranker:** Q44 fails under the LLM reranker because 400-character
chunks split the product code from the answer; larger chunks or the code in the heading path might
remove that regression. **Prefix × retriever:** the heading prefix was measured only with dense.

## Not measured

The LLM reranker at k=30 (`LAB3_LLM_KS=10,30` runs it). D2 with real ballast embeddings. The exact
D2 crossover and the A2 knee are bracketed, not pinned.

## One surprising result

The biggest reliable fix in the lab needed no retrieval technique: one metadata field and a `where`
clause took the trap questions from 0.667 to 1.000 and changed nothing else. Meanwhile the
cross-encoder, a standard quality upgrade, made answers worse by preferring passages that repeat
"co-payment" over the one that answers with "Nil".
