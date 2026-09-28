# Reproducing the Lab 3–6 results

Every number in `labs/lab*/report.md` comes from a file in `reports/`, and every file in
`reports/` carries a `_provenance` block naming the models, git SHA, cache state and
temperature it was produced under. Start by reading that block — it is what tells you
whether a cost or latency figure is a deployment number or a cache artefact.

```bash
python -c "import json;print(json.load(open('reports/lab4.json'))['_provenance'])"
python scripts/provenance.py --check      # which files lack provenance
```

## 0. Environment

```bash
make setup-full          # ~1.8 GB: chromadb, rank_bm25, sentence-transformers (PyTorch)
make env                 # creates .env; put one provider key in it
make check               # must print "Environment is ready." with no warn lines
```

**Free-tier quotas decide what is runnable in a day**, and they shaped several of our
results:

| Model tier | Model | Free-tier limit |
|---|---|---|
| SMALL | `gemini-3.5-flash-lite` | 500 generate requests/day |
| MAIN | `gemini-3.7-flash` | ~20/day, tens of seconds per call |
| LARGE | `gemini-3.5-flash` | **20 generate requests/day** |
| EMBED | `gemini-embedding-001` | 100 embeddings/minute |

Quotas reset at midnight US Pacific. A full Lab 4 run is ~135 calls, Lab 6's cumulative
sweep ~300, so two labs in one day is roughly the ceiling on one free key.

## 1. Order matters

```
Lab 3  ──►  Lab 4  ──►  Lab 5
                 └──►  Lab 6
```

Lab 4 hard-depends on Lab 3's winning configuration (`build_retriever()` in
`labs/lab4/evaluate.py`), Lab 5 reads Lab 4's output, and Lab 6's `search_policy` tool is
backed by the same Lab 3 retriever.

## 2. Lab 3 — retrieval sweeps

```bash
python scripts/expand_corpus.py --docs 4000     # REQUIRED before --sweep index (D2 ballast)
python labs/lab3/search.py --baseline
python labs/lab3/search.py --sweep chunking     # slowest: 9 configs, embeds ~900 new chunks
python labs/lab3/search.py --sweep retrieval
python labs/lab3/search.py --sweep rerank       # ~420 LLM calls at k=10
python labs/lab3/search.py --sweep index        # ~5 min, builds HNSW at 40k vectors
```

Each command merges one section into `reports/lab3_sweeps.json` (plural). Saving happens at
the **end** of each sweep: interrupt it and nothing is written.

Expect rate-limit backoffs during `--sweep chunking`; they are logged and retried
automatically. `--sweep rerank` at the handout's k=30 needs ~1,260 calls and will exhaust a
day's quota — ours ran at k=10, which is stated in the report.

## 3. Lab 4 — grounded answers

```bash
python labs/lab4/evaluate.py --full --save reports/lab4.json     # ~135 calls
python labs/lab4/evaluate.py --gold-context                      # E2 decomposition
python labs/lab4/evaluate.py --full --strict --no-judge --save reports/lab4_strict.json
python labs/lab4/evaluate.py --calibrate                          # writes the label sheet
#   -> hand-label 20 answers in labs/lab4/calibration_labels.jsonl BEFORE the next step
python labs/lab4/evaluate.py --kappa
python labs/lab4/evaluate.py --cross-check 10                     # self-preference, LARGE
```

**Two traps worth knowing before you run this.**

1. **Snapshot `lab4.json` before re-judging it.** Lab 5 derives its failure list from Lab 4's
   scores, so re-judging Lab 4 silently changes Lab 5's input. We keep
   `reports/lab4_pre_rubric_fix.json` for exactly this reason: six of Lab 5's twelve
   failures are not failures under the corrected rubric.
   ```bash
   cp reports/lab4.json reports/lab4_pre_rubric_fix.json
   python labs/lab4/evaluate.py --rejudge reports/lab4.json --rejudge-save reports/lab4.json
   ```
2. **κ gates the correctness number.** If `--kappa` reports below 0.4, fix the rubric and
   re-judge; do not quote correctness until it passes. Ours went 0.189 → 0.519 that way.

## 4. Lab 5 — diagnose, fix, prove

```bash
python labs/lab5/diagnose.py --input reports/lab4_pre_rubric_fix.json --pareto
python labs/lab5/diagnose.py --before-after                    # the shipped fix
python labs/lab5/diagnose.py --before-after --variant single   # the fix that did not work
```

`--input` matters: pass the snapshot, not today's `lab4.json`, or the tally will not match
the report. The classifier reads gold-context outcomes from
`reports/lab4_decomposition.json`, so run Lab 4's `--gold-context` first.

## 5. Lab 6 — tools and red-teaming

```bash
python labs/lab6/redteam.py --no-guards --save reports/lab6_unguarded.json
python labs/lab6/redteam.py --cumulative --save reports/lab6_cumulative.json   # ~300 calls
python labs/lab6/redteam.py --layers 1 2 3 4 5 --extra --save reports/lab6_extra.json
```

Measure **both** rates at every layer: block rate over the 17 attacks and false positives
over the 4 controls. Running only "unguarded vs all five" reads 1.00/0.00 at both ends and
hides that layer 2 is net-negative and layer 3 repairs it.

The harness writes poisoned documents to a temp copy of the corpus; `data/corpus/` is never
modified.

## 6. Quoting cost or latency

Runs replay a warm cache, so repeated runs report ~$0 and ~10 ms. Neither is a deployment
figure. To quote either:

```bash
AIP_CACHE=0 python ...      # forces live calls
```

Every number we quote for cost or p95 was measured this way, and the `_provenance` block
records `cache_enabled` so a reader can tell.

## What is not reproducible from a clean clone

- **`data/corpus_scaled/`** (16 MB, 4,000 generated documents) is gitignored. Regenerate it
  with `scripts/expand_corpus.py` before Lab 3's D2.
- **`.chroma/`** is gitignored and rebuilt by `--sweep index`.
- **The response cache** (`.aip_cache/`) is not committed, so a first run pays full price and
  hits rate limits ours did not.
- **Exact latency figures** depend on provider load. We measured the same configuration at
  p95 8,574 ms and 2,300 ms an hour apart. Treat latency as a magnitude.
