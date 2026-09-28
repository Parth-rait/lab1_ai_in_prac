# Lab 7 (Option 2) — change plan

Scope: **Lab 7 only (Option 2).** The Option 1 track (Policy / Legal / Healthcare /
Fintech) is deliberately out of scope and will be planned separately.

Deliverable for submission (Friday EOD): a ≤ 4-slide presentation and a concept note
for Lab 7. Demo/viva before 6 Oct: 5 min presentation + 3 min questions.

---

## 0 · What has already been done (29 Sep)

| Step | Result |
|---|---|
| Backup before unzipping | `../backup_before_lab7_20260929_0202/aip-lab1_full_snapshot.tgz` (everything except `.venv`, `.chroma`, `data/corpus_scaled`) |
| `unzip -o AI-in-Practice-Lab7.zip -d aip-lab1` | 70 files: 18 new (Lab 7 + CI workflow + missing handouts), 52 refreshed |
| What actually changed in existing files | Only 5 differed: `Makefile`, `labs/lab1/README.md`, `labs/lab2/README.md`, `labs/lab2/OVERVIEW.md`, `tests/test_aip.py`. `aip/` was already identical. No `report.md`, lab code, `reports/`, `.env` or `.aip_cache/` touched |
| **Repair: `tests/test_aip.py`** | The zip's version dropped our own Lab 1 tests (`apply_business_rules`, `extract_deterministic`). Merged them back on top of the new file, which also adds two new toolkit tests. **39/39 pass** |
| `Makefile` | Zip removed the `make quiz` target. Harmless — `scripts/build_quiz.py` never existed here |
| Service stack | `fastapi, uvicorn, streamlit, sse_starlette, yaml` import OK; `make check` → "Environment is ready." |

---

## 1 · Do Labs 1–6 fit Lab 7? — compatibility check (measured, not assumed)

| Lab 7 needs | We have | Status |
|---|---|---|
| Lab 3 retriever, importable | `labs.lab4.evaluate.build_retriever()` — markdown-400, dense, k=5, no ARCHIVED | ✅ builds in **0.08 s** |
| Lab 4 generator | `labs.lab4.rag.answer_question(q, retriever, tier="SMALL")` with citation validation + repair | ✅ |
| Lab 5 fix | `labs.lab5.diagnose.build_fixed_retriever()` (document expansion) | ✅ builds in **0.06 s** |
| Offline replay for CI (`AIP_OFFLINE=1`) | Cache replays **45/45** golden questions for both v1 (Lab 4) and v2 (Lab 5) | ✅ |
| Lab 6 guards | `LAYERS` 1–5 in `labs/lab6/agent.py`, `ToolGuard`, `filter_output()` | ⚠️ built for the **tool loop**, not the RAG path — needs wiring (change A2b) |
| Lab 6 agent, offline | `run_agent()` → `CacheMiss` on golden questions (cache holds only red-team payloads); defaults to **MAIN** tier (~36 s/call, ~20/day) | ⚠️ `mode="tools"` cannot meet the 6 s SLO on MAIN; not gate-able offline |
| `hit_rate_at_5 ≥ 0.85` | Lab 3: **0.976** | ✅ |
| `correctness ≥ 0.75` | 0.936 (v1) / 0.949 (v2), corrected rubric | ✅ |
| `faithfulness ≥ 0.90` | 0.978 / 0.956 | ✅ |
| `citation_validity ≥ 0.98` | 1.000 | ✅ |
| `refusal_recall ≥ 0.80` | 1.000 | ✅ |
| **`refusal_precision ≥ 0.75`** | **0.625 (v1) / 0.714 (v2)** | ❌ **fails the gate as shipped** |
| `cost_per_query_usd ≤ 0.010` | $0.00099 / $0.00113 | ✅ |
| `p95_latency_ms ≤ 6000` | 2,300 / 1,075 ms (live) | ✅ live — but see D1b: offline replay reads ~0 ms |
| CI can see the cache | **`.aip_cache/` is in `.gitignore`** | ❌ CI will fail with "missing key/cache miss" |
| GitHub repo | `origin = github.com/Parth-rait/lab1_ai_in_prac` | ✅ workflow sits at repo root `.github/workflows/eval.yml` |

**Bottom line:** the earlier labs *do* carry into Lab 7 — the pipeline imports cleanly,
starts in under 0.1 s and replays offline. Three things block a green gate and must change
first: **refusal precision**, **the ignored cache**, and **Lab 6 guards on the RAG path**.

---

## 2 · The pipeline we ship, and why

```
question ─► [L2 injection screen] ─► Lab 3 retriever ─► Lab 5 doc-expansion ─►
         Lab 4 generator (SMALL, prompt v1 wording) ─► citation validation/repair ─►
         [L5 output filter] ─► response {answer, citations, cost_usd, trace_id}
```

- **Lab 5 v2 retriever over v1**: correctness 0.949 vs 0.936, p95 1,075 vs 2,300 ms, cost
  1.14× (inside 2×). The gain is small (+0.013 ≈ 1 question, see Lab 5 addendum), but it is
  not worse and it is faster.
- **SMALL tier for generation**: MAIN is ~36 s/call — it alone breaks the 6 s SLO (Lab 4).
- **RAG mode is the default; `mode="tools"` is optional** and runs the Lab 6 agent on SMALL
  with layers 1–5 and the layer-4 `ToolGuard`. It is excluded from the offline gate.

---

## 3 · Every change, and why

### Pre-work (fixes carried over from Labs 4–5 — needed for a green gate)

| # | Change | Why | Cost |
|---|---|---|---|
| P1 | **Un-ignore the cache**: remove `.aip_cache/` from `.gitignore`, commit `.aip_cache/calls.sqlite3` (23 MB) | CI runs `AIP_OFFLINE=1`; with no committed cache every call is a miss and the gate fails for the wrong reason | free |
| P2 | **Restore Lab 4's v1 refusal wording** as a prompt option in `labs/lab4/rag.py` (`ANSWER_SYSTEM_V1`), keeping v2 intact | Refusal precision 0.714 < 0.75 gate. Lab 4 measured v1 at **0.833** and said "v1 is the better system; next step: revert and re-measure" — this is that step | ~45 gen + ~90 judge calls, live |
| P3 | **Re-measure v1-prompt + Lab 5 retriever** once, live, `AIP_CACHE=0` for cost/latency, then a cached run so the gate can replay it | Gives the numbers Part E quotes *and* populates the cache CI replays | ~135 calls (≈ ¼ of a day's SMALL quota) |
| P4 | **Addenda** (append-only, no existing text changed) to `labs/lab4/report.md` and `labs/lab5/report.md` with the P3 result | User rule: never rewrite past reports, only add on | free |
| P5 | *(optional, if quota allows)* Lab 5 **adaptive expansion** (`--variant adaptive`) | Targets the Q04/Q11 regressions; Lab 5 D4 estimates +0.02 | ~135 calls |

**If P2 does not reach 0.75**, we do **not** lower the threshold to make CI green —
that is exactly the "unreachable/loosened gate" failure the brief warns about. We report
the breach honestly and set the gate at measured − 1 SE with the reasoning written down.

### Part A — the service (`service.py`, `ui.py`)

| # | Change | Why |
|---|---|---|
| A1 | `POST /ask` returns `answer, refused, citations[{index, doc_id, excerpt}], sources, latency_ms, cost_usd, cached, trace_id` | Brief: cost and trace id in the body so a debugger can find the trace and a payer can see the cost |
| A2 | `pipeline()` builds retriever + generator **once at startup** (FastAPI lifespan) and caches it | Trap #1: per-request build re-embeds the corpus (the "40-second p95") |
| A2b | Lab 6 guards on the RAG path: **L1** untrusted-delimiting (already in `ANSWER_SYSTEM` via `UNTRUSTED_SYSTEM_CLAUSE`), **L2** `detect_injection` on input, **L5** `filter_output` on output, **citation enforcement** from Lab 4; **L4** `ToolGuard` allowlist for `mode="tools"` | Trap #2 and explicitly graded. Lab 6 showed L2 alone has a false positive (C02) that L3 repairs — so L2 *flags* into the trace and only blocks on high-confidence signatures, rather than refusing everything it dislikes |
| A3 | Errors: 422 (free from Pydantic — verify), **503 + `Retry-After`** on provider outage/rate limit, **429** on `BudgetExceeded`, 500 only for genuine bugs | A 500 on a rate limit makes clients retry immediately — the wrong behaviour. Test all three with `curl` |
| A4 | UI: citations as expanders with source excerpt (skeleton mostly there); add the `sources` list | Grounding the user cannot check is decoration |

### Part B — caching and streaming

| # | Change | Why |
|---|---|---|
| B1a | Exact response cache: key = hash of normalised question (lower-case, whitespace/punctuation collapsed) + pipeline version | Zero-risk; 15–30% hit rate. Including the pipeline version stops a prompt change serving stale answers |
| B1b | Semantic cache: question embedding, cosine ≥ threshold | +10–20% hit rate, but can answer a *different* question |
| B1c | **Threshold sweep** script: pairs of near-duplicate questions (Gold vs Silver, 30 vs 60 days, Bronze vs Platinum…) + true paraphrases; sweep 0.80 → 0.99, record the cosine where the first wrong hit appears | **The threshold is the deliverable.** The 0.95 in the starter is reasoned, not measured |
| B2 | `POST /ask/stream` via `sse-starlette`; record TTFT and total | Target TTFT ≤ 1,500 ms |
| B3 | Strategy: **stream the prose, hold citations to the end, then send a `validation` event**; if validation fails the UI replaces the answer with the refusal | Buffering throws away TTFT; streaming unvalidated citations is un-sendable. Lab 4's repair rate was 0/45, so the retract path is rare but must exist |
| B4 | Latency budget per stage (embed / retrieve / expand / generate / validate) from traces, p50/p95 | Section 5 of the report; identifies what to optimise (expected: generate) |

### Part C — observability

| # | Change | Why |
|---|---|---|
| C1 | Own spans: `http.ask`, `cache.exact`, `cache.semantic`, `guard.input`, `retrieve`, `expand`, `generate`, `validate`, `guard.output` | So "why did request X take 9 s?" is answerable from traces alone |
| C2 | `/metrics`: cost today, cost/query, cache hit rate (exact vs semantic), p50/p95/p99, error rate **by type** (422/429/503/500), refusal rate, tool-call counts | Brief C2 |
| C3 | Dashboard: p50/p95 per span, latency over time, cumulative cost, error rate, cache hit rate | TODO C3 |
| C4 | Alert: **refusal rate doubles vs trailing baseline** → check index size in `/health`, re-run gate | Best signal that the index broke; quality metrics lag it |
| — | `/health`: index size (chunks/docs), model profile, cache stats, uptime | TODO C |

### Part D — regression gate

| # | Change | Why |
|---|---|---|
| D1 | `gate.measure()`: run the 45 golden questions through the **same `pipeline()` the service uses**, judge with the Lab 4 corrected rubrics, return all 8 metrics (hit_rate_at_5 from `aip.evals.retrieval_metrics`) | Gate must test what ships, not a copy |
| D1b | **Latency/cost gates read the live-measured numbers** (from the P3 run file), not the offline replay | Offline replay reports ~0 ms and $0 — a p95 gate on replay can never fail, which is a fake gate |
| D1c | Tune `thresholds.yml` to measured − ~1 SE (at n=45 the SE on a proportion is ~0.03–0.05) | Brief: at-the-number fails on noise; far-below never fires |
| D2 | Commit cache (P1), push, confirm the **green** Actions run | Workflow provided |
| D3 | Deliberate break on a branch: `final_k=1` (or delete a corpus doc) → push → **screenshot the red build** → revert | "A gate you have not seen fail is a gate you do not have." Graded + demo item 4 |
| D3b | Check the workflow can fail: exit code propagates, no `|| true`, thresholds reachable | Brief lists swallowed exit codes / unreachable thresholds as fake-green causes |

### Part E — deliverables

| # | Change | Why |
|---|---|---|
| E1 | `EVALUATION_REPORT.md` (repo root), 2 pages, 7 sections | 16% of the module |
| E2 | §3 failures with counts: currently **Q04, Q11** (multi-doc regressions from expansion), **Q29, Q32** (generation), **Q37** (retrieval / paraphrase, gold docs never retrieved), Q44 (judge parse — missing data) | Demo item 5 — **worst remaining failure: Q37** (Singapore treatment; no prompt can fix it; next fix = route paraphrases to hybrid) |
| E3 | §4 cost per query / per 1,000 / per year at 10,000/day, from live-measured $/query | |
| E4 | §6 not safe for: coverage/claim decisions without human review (a wrong deadline costs a valid claim; judge κ = 0.52 means correctness itself is only moderately reliable), multi-plan comparisons (Q04/Q11 class), anything outside the corpus, semantic cache above its measured threshold | Concrete boundary + reason, not a disclaimer |
| E5 | §7 next steps ranked with EV: (1) adaptive expansion, +0.02; (2) hybrid routing for paraphrases (Q37); (3) judge on LARGE/cross-check to remove self-preference (+0.10 faithfulness overstatement) | |
| E6 | Root `README.md` "run in < 5 min on a clean machine" — **tested on partner's laptop** | Our machine has the venv, keys and warm cache |
| E7 | 4-slide deck + concept note (Option 2 deliverable from the email) | Friday EOD |

---

## 4 · Risks to watch

- **Quota.** P3 ≈ 135 calls; semantic sweep + streaming tests + demo add more. One free key ≈ 500 SMALL calls/day — run P3 early in the day, don't combine with P5.
- **CI install is heavy.** `requirements.txt` pulls `sentence-transformers` (PyTorch); the first CI run will be slow (~5–8 min). Acceptable; don't strip it without checking Lab 3 imports.
- **CI runs Python 3.11**, we run 3.12 locally — verify the first green run rather than assuming.
- **Latency is environment-dependent** (same config measured 2,300 and 8,574 ms an hour apart). Quote p95 as a magnitude with the date.
- **Uncommitted work.** Everything since 25 Sep (Lab 5 addendum, runlog, provenance, Lab 6 edits) is still uncommitted — commit it before starting Lab 7 so the D3 red/green history is clean.

## 5 · Order of work

1. Commit current state (Labs 5/6 add-ons + Lab 7 unzip) → P1
2. P2 → P3 → P4 (live, one session)
3. A1–A4 → B1a → C1/C2 (service usable end-to-end)
4. B1b/B1c threshold sweep → B2/B3 → B4
5. D1–D3 (green, then red screenshot)
6. C3/C4 dashboard + alert
7. E1–E7, rehearse the 5-minute demo

---

## 6 · Recalibration (29 Sep, after P3)

### What P3 showed

v1 prompt on the Lab 5 retriever: refusal precision **0.667** (4/6), recall **0.800**
(`reports/lab7_p3_v1prompt.json`). Worse than v2 (0.714, 1.000). Lab 4's 0.833 for v1 was
measured on a different retriever and does not transfer. **P2 as planned is refuted.**

### Comparison: reference repo (neeti-kurulkar/ai-in-practice-lab1, commit e874997, Labs 1–6)

Same golden set, corpus and `aip/`. Different pipeline: MAIN generator, 300-char chunks,
LARGE judge.

| | ours: v2 + expansion (SMALL) | theirs: Lab 4 (MAIN) | theirs: Lab 5 small-to-big |
|---|---|---|---|
| correctness | 0.949 | 0.863 | 0.925 |
| refusal recall | 1.000 | 0.800 exact / 1.000 incl. partials | 1.000 |
| **refusal precision** | **0.714 ❌** | **1.000 ✅** | **0.833 ✅** |
| cost/query | $0.0011 | $0.0089 | **$0.0123 ❌** |
| p95 | 1,075 ms | 4,237 ms | **8,912 ms ❌** |

**They pass the gate we fail.** (Their Lab 5 fix fails cost and latency instead, so their
Lab 4 pipeline is the one that clears all eight.)

### Why: partial answers do not use the refusal sentence

Our v1 and v2 both tell the model to answer the supported part and then append the
**exact refusal sentence**. Exact-match detection then counts every partial answer as a
refusal, and because the question was answerable, as a *false* refusal. Their rule 4:

- partial answer = supported fact [n] + "**The sources do not state** <missing part>."
  No refusal sentence.
- refuse **only when the sources say nothing** about the subject.
- **no substitution**: a different quantity (sum insured for premium) is not a partial
  answer. Refuse instead. (This is the clause that stopped their Q36 regression.)
- if the sources confirm something *exists* but not the detail, say it exists, cite it,
  and say the detail is not stated. That is their Q40 fix ("24×7 helpline, number on the
  policy schedule [1][4]"), scored correctness 2.

Per question, on our two false refusals: **Q23** becomes a partial (their corr 1, ours 2
as a refusal); **Q44** becomes a partial that is still wrong (their corr 0). Q44 is a
retrieval/paraphrase failure either way.

**Honesty caveat (goes in the report):** part of the precision gain is definitional.
Q44 still fails. It just stops being counted as a refusal and shows up in correctness
instead. So we report **both** definitions, as they do: exact refusals (what downstream
code sees, and what the gate uses) and refusals + partial declines.

### Revised pre-work (replaces P2–P4)

| # | Change | Why |
|---|---|---|
| P2′ | `ANSWER_SYSTEM_V3` in `labs/lab4/rag.py` = v2 with rules 4/4a rewritten to the shape above (partial → "The sources do not state …", refuse only on nothing, no substitution, exists-but-no-detail). v1/v2 kept. Add `partial_decline` detection (regex on "The sources do not state") to `validate_answer` and to the rows | Moves correct partials out of the false-refusal bucket, and fixes Q40's appended-topic refusal |
| P3′ | One live run: v3 + Lab 5 retriever, SMALL, `--baseline reports/lab5_v2_rejudged.json` (~135 calls; ~110 used today) | One variable against v2, same as P3 |
| P3′ decision | **Ship v3** if precision ≥ 0.75 **and** recall ≥ 0.80 **and** correctness ≥ 0.919 (v2 − 1 SE) **and** no new Q36-type substitution. Otherwise **ship v2** and fall back to "report the breach, gate at measured − 1 SE" | Pre-registered, so we cannot pick the rule after seeing the numbers |
| P4′ | Addenda cover **both** runs: v1 refuted (P3), v3 result (P3′), both refusal definitions | Lab 4's "revert to v1" next step is now answered: tested, and it did not hold |

**Not adopted from their repo:** MAIN tier (4.2 s p95 on its own, ~20 calls/day) and
small-to-big (p95 8.9 s, $0.0123/query, both over the gate).

### Knock-on changes to later steps

- **Service / UI (A, B3):** the response carries `partial: bool`, and the UI shows the
  "not stated" part distinctly. The streaming `validation` event treats a cited partial
  as valid, not as a retraction.
- **Gate (D1):** `refusal_precision` and `refusal_recall` stay exact-match (gated).
  `partial_decline_rate` is reported but not gated. Watch it in C4: a jump means the
  index lost content.
- **Pipeline diagram (§2):** "prompt v1 wording" → "prompt v3 (or v2, per P3′ decision)".
- **E2 failures:** add Q29 (motor-claim distractor, 0 under v1) and Q44 (unchanged,
  retrieval). Q37 is still the worst remaining failure.

### Revised order of work

1. ✅ Commit + P1 · ✅ P2/P3 (v1: refuted, committed `8934a7f`)
2. **P2′ → P3′ → decision → P4′** (live, today)
3. A1–A4 → B1a → C1/C2 · 4. B1b/B1c → B2/B3 → B4 · 5. D1–D3 · 6. C3/C4 · 7. E1–E7

**P3′ result (29 Sep, `reports/lab7_p3_v3prompt.json`): v3 ships.** Precision 0.833 (5/6),
recall 1.000, correctness 0.923 (≥ 0.919, thin), Q36 still refused → all four conditions
met. Cost: correctness −0.026 vs v2 (Q04, Q20, Q45 down to hedged partials; Q29 up). The
one remaining false refusal is Q44 (retrieval). **No further prompt work; on to A.**
