# Aurora Policy Assistant — Evaluation Report

*Lab 7 capstone · measured 29 Sep 2026 · pipeline: Lab 3 dense retriever + Lab 5 document
expansion, answer prompt v3, `gemini-3.5-flash-lite` (SMALL). Every number below is reproducible
from committed files. `AIP_OFFLINE=1 python labs/lab7/gate.py` replays the full test set with no
API key (CI: [green run](https://github.com/Parth-rait/lab1_ai_in_prac/actions/runs/36487871404)).*

## 1 · What it does

Support agents at Aurora Health Insurance type a customer's question, for example "how long do I
have to file a reimbursement claim?". The assistant finds the relevant passages in Aurora's 29
policy documents and writes a two-to-three-sentence answer. Every factual sentence carries a
numbered citation that opens to the exact source text. When the documents do not answer the
question, it says so in a fixed sentence instead of guessing. When they answer only part of it,
it answers that part and names what is missing. It is a lookup aid for the agent, not a
decision-maker.

## 2 · How well it works

45 test questions (40 answerable, 5 not answerable from the documents), judged against written
reference answers. Source: `reports/lab7_gate.json`; the live run is in `reports/lab7_p3_v3prompt.json`.

| Metric | Result | Gate | What it means |
|---|---|---|---|
| Correctness (0–1) | **0.923** | ≥ 0.88 | mean judged score, 39 answerable questions (1 judge parse failure) |
| Faithfulness | **0.978** (44/45) | ≥ 0.95 | the answer contains nothing the cited sources do not say |
| Citation validity | **1.000** | ≥ 0.98 | enforced in code: an invalid citation is repaired once, else refused |
| Refusal recall | **1.000** (5/5) | ≥ 0.80 | every unanswerable question was declined |
| Refusal precision | **0.833** (5/6) | ≥ 0.75 | 1 answerable question was wrongly declined (Q44) |
| Retrieval hit rate @5 | **0.976** (41/42) | ≥ 0.95 | a correct document was in the top 5 |
| Partial answers | 6 of 40 answerable | not gated | answered in part, remainder declared "not stated" |

**How much to trust these numbers.** Each correct refusal moves precision and recall by
0.12–0.20, so differences in those two rows below about 0.15 are noise. The judge agreed with
human labels at κ = 0.52 (moderate), and a cross-check with a stronger model put our
faithfulness about 0.10 too high (self-preference). **Faithfulness 0.978 is an upper bound.**

**How we got here.** The previous prompt (v2) failed the refusal-precision gate (0.714). Reverting
to Lab 4's v1 made it worse (0.667, recall 0.80). v3 fixed the actual cause: partial answers had
ended with the exact refusal sentence, so every correct partial answer counted as a false
refusal. That cost 0.026 correctness, about one question. The v3 choice was made by a rule
written before the run (Lab 4 report addendum, 29 Sep).

## 3 · Where it fails

10 of 45 questions are imperfect. Grouped by cause:

| Failure mode | n | Questions | Evidence |
|---|---|---|---|
| **Over-hedged partial answer** (generation) | 4 | Q04 (score 0), Q20, Q32, Q45 (1 each) | Correct first sentence, then "the sources do not state…" for a part they do support. Q45 will not conclude that 6 dioptres < 7.5 means "not covered". The side effect of the v3 prompt. |
| **Answer spans two documents** (context) | 2 | Q04, Q11 | Document expansion fills the context from the top document and pushes out the second: the rider that shortens Q04's waiting period, and Gold/Platinum air ambulance for Q11. Both are Lab 5 regressions. |
| **Retrieval miss** | 2 | Q37, Q44 | Q37 ("Singapore"): the gold documents are never retrieved (hit@5 = 0), so refusing is the correct response to the context it got. Q44 (a plan-code query, `AUR-HI-SIL-2026`) retrieves the document but not the sum-insured passage, and it is wrongly refused. |
| **Derived claim judged unsupported** | 2 | Q25, Q34 | Correct answers (score 2) that compute (₹1,60,000 − ₹1,00,000) or recommend a plan. The derived statement is not verbatim in a source, so faithfulness scores it 0. |

Q04 appears in two rows, so there are 10 distinct questions. Q23 and Q26 are partial answers
that scored 2, so they are not failures.

**Worst remaining failure: Q37.** It is the only question whose evidence never reaches the
model, so no prompt can fix it. It also represents a whole class: a place name ("Singapore")
standing in for the corpus's wording ("outside India"). We would route queries like this to
hybrid retrieval (§7).

## 4 · What it costs

Generation dominates. Query embedding costs about $0.000003 and search, guards and caching run
locally. Per query, at Gemini list prices, from recorded token counts (`gate.py`, 45 questions):

| | Cost |
|---|---|
| Per query (mean; max $0.00063) | **$0.00046** |
| Per 1,000 queries | **$0.46** |
| Per year at 10,000 / day (3.65 M queries) | **$1,690** |
| … with a 20% answer-cache hit rate | ≈ $1,350 |

The gate caps cost at $0.0010/query, so switching to the MAIN tier or doubling the context
fails CI. Not included: hosting, evaluation runs (judging roughly doubles a test run's cost:
$0.00084/query live), and the fact that 10,000/day needs a **paid** key. The free tier allows
about 500 generations a day. Observed answer-cache hit rate on our own mixed test traffic was
8%. The brief's 15–30% assumes real repeat traffic, which we have not measured.

## 5 · How fast it is

Per stage, from traces alone (`labs/lab7/latency_budget.py`). 5 live requests on questions not
seen before, 29 Sep, 02:52:

| Stage | p50 | p95 | Share |
|---|---|---|---|
| Embed query (Gemini API) | 1,082 ms | 1,371 ms | 48% |
| Search + document expansion | 0 ms | 0 ms | 0% |
| Guards (input PII, retrieved-text injection, output filter) | 0 ms | 1 ms | 0% |
| Generate (SMALL) | 1,136 ms | 1,639 ms | 50% |
| Validation, formatting | 8 ms | 10 ms | 0% |
| **Total** | **2,274 ms** | **2,516 ms** | SLO p95 ≤ 6,000 ms ✅ |

The 45-question live run measured **p50 1,225 / p95 1,783 ms**. Its query embeddings were
already cached, so those figures are almost pure generation. An exact answer-cache hit returns
in **0.5 ms**.

**Streaming does not help this system.** Time to first sentence was 2,186 ms against 2,190 ms
total (n = 6). The model writes a 2–3 sentence answer in one or two chunks, so almost all the
wait happens before the first word. Streaming is implemented properly anyway: each sentence
passes the output filter before it is sent, citations are validated at the end, and a
`replace` event retracts a failed answer. **Optimise first: the query embedding (48%).** Answer
caching already removes it for repeat questions.

## 6 · What it is not safe for

1. **Coverage or claim decisions without a human.** About 1 in 13 answers is incomplete or
   wrong (correctness 0.923), and that figure comes from a judge that agrees with humans only
   moderately (κ = 0.52). The failures are hard to spot. Q04 answers the most common question
   (pre-existing waiting period) with "36 months" and then wrongly says the rest is not stated,
   omitting the rider that reduces it to 24 or 12 months. A customer told the wrong deadline or
   exclusion can lose a valid claim.
2. **Questions comparing plans or spanning documents.** 4 of the 6 lowest-scoring answers are
   multi-part (Q04, Q11, Q20, Q32). The context shows the top document in depth at the expense
   of the second.
3. **Reading a refusal as "not covered".** A refusal means *not found*, not *excluded*. Q37 and
   Q44 are refused although the documents contain the answer. Agents must check the source
   before telling a customer something is not covered.
4. **Computed amounts.** Out-of-pocket figures (Q25) are the model's arithmetic, not a quoted
   policy figure, and nothing verifies them.
5. **The semantic cache beyond what we measured.** The 0.93 threshold was validated on 20
   near-miss pairs; wrong pairs reached 0.906 ("settlement ratio" vs "settlement time", Gold vs
   Silver at ~0.89). Unmeasured phrasings could cross it and return another plan's answer.
   The cache also lives in memory with no invalidation, so a document update needs a restart.
6. **Latency promises under load.** Measured on a free key on one day. The same configuration
   has measured 2.3 s and 8.6 s an hour apart (Lab 4). The 6 s SLO is a target, not a
   guarantee.
7. **Hostile input as a security boundary.** The injection and PII guards are pattern-based,
   and their protective value is unproven. On Lab 6's 17-attack suite the model refused every
   attack even with no guards on, so that suite cannot show what the guards add. What we have
   measured is their cost: 0 of 231 real passages flagged, 0 of 45 answers altered. A
   determined attacker can phrase around a pattern.

## 7 · What we would do next

| Rank | Change | Expected value | Cost |
|---|---|---|---|
| 1 | **Adaptive document expansion**: expand only when the second-ranked document is clearly weaker | Recovers Q04 (0→2) and Q11 (1→2): **≈ +0.04 correctness**; targets failure rows 1–2 | No extra model calls per query; one live re-run (~135 calls) |
| 2 | **Route paraphrase and code-style queries to hybrid retrieval** (dense + keyword) | Q37 and Q44 reach their evidence: **precision 0.833 → 1.0**, ≈ +0.03 correctness | Routing only. Lab 3 measured hybrid as worse when applied to every query |
| 3 | **Remove the query-embedding round trip** (local embedder or a warm regional endpoint) | **≈ −1.1 s p50 (−48%)** on new questions | Must re-validate retrieval (hit@5 0.976) on the new embedder first, via the Lab 3 sweep |

Rank 1 comes first because it is the cheapest change aimed at the most frequent failure. Rank 2
removes the worst remaining failure. Rank 3 is pure speed, and the system already meets its SLO.
