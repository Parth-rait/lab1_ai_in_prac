# Lab 5 — RAG v2: Diagnose, Fix, Prove: Report

Input: **`reports/lab4_pre_rubric_fix.json`** (45 questions, 12 failures) — the Lab 4 output
as it stood when this lab ran. Classification: `reports/lab5_diagnosis.json`, written by
`labs/lab5/diagnose.py --input reports/lab4_pre_rubric_fix.json`.

> **Why a snapshot rather than `reports/lab4.json`.** Lab 4's correctness rubric was later
> corrected (its κ rose 0.189 → 0.519) and its answers re-scored. **Six of the twelve
> failures diagnosed here — Q01, Q04, Q05, Q26, Q43, Q44 — are not failures under the
> corrected rubric**; re-running the classifier against today's `lab4.json` yields 6, not 12.
> Nothing about the diagnosis was wrong: those six were being mis-scored by a rubric that
> penalised answers for omitting background the question never asked for. The snapshot is
> kept so this lab's input is reproducible, and it is worth stating plainly that **half of
> the backlog this lab set out to fix was a measurement artefact, not a system defect** —
> which is the strongest possible argument for calibrating a judge before acting on it. Retrieval is
unchanged from Lab 3/4: markdown-aware chunking at 400 characters, exact dense
retrieval, archived documents excluded.

**Status:** complete. All four parts measured; Part D ran on a fresh API key
(`reports/lab5_before_after.json`).

## Part A — classify every failure

`python labs/lab5/diagnose.py --input reports/lab4.json --pareto`

| failure mode | n | share | cumulative |
|---|---|---|---|
| 4 — ranking | 6 | 50.0% | 50.0% |
| 6 — generation | 6 | 50.0% | 100.0% |
| 1, 2, 3, 5, 7 | 0 | — | — |

- **Ranking (6):** Q01, Q23, Q25, Q29, Q43, Q44
- **Generation (6):** Q04, Q05, Q10, Q26, Q32, Q37

**Evidence per case** is in the JSON: whether gold context fixed it (from the Lab 4
decomposition run), whether the gold document was in the top 30, whether it reached the
final context, and for mode-3 candidates an own-text probe. **Mode 5 is structurally
impossible** here — Lab 4 ships no reranker — and is recorded as `False` rather than left
unknown so the tree cannot fall through to it.

**The mode-6 test, in the direction that matters.** Gold context *fixing* the answer means
retrieval was at fault and the generator was always capable; gold context *not* fixing it
means generation is at fault. All six generation cases scored no better with gold context.

**Two classifications differ from my Lab 4 E3 tags**, and the tree wins:
- **Q37** was tagged mode 3 in Lab 4 because its gold documents were never retrieved. That
  is true, but with gold context it still scores 1, so by the tree it is mode 6: fixing
  retrieval would not fix the answer.
- The six ranking cases had the gold document *in* the final context, so the evidence
  string "outside the final k" would have been false. It now reads accurately: cleaner
  gold-only context scored higher, so the fault is the company the gold chunk kept.

**A2 — the human check.** No case reached mode 2, so nothing is flagged
`needs_human_check`. I opened two generation cases anyway to confirm they are not
chunk-boundary failures in disguise: Q05's gold document has 10 chunks mentioning the
grace period and the model saw one (it answered "30 days" and missed the 15-day instalment
case); Q10's Ombudsman stage arrived without the two escalation stages that precede it.
Neither answer is cut in half — each is *incomplete*, which is a different fault.

**A1 improvement — the mode-1 test.** The shipped test was word overlap > 0.4. Mine decides
on **numbers** when the gold answer contains one (separators normalised, so "Rs 1,00,000",
"1,00,000" and "100000" match), and handles `REFUSE`/`PARTIAL REFUSE` gold answers
explicitly. **How I know it is better:** checked against ground truth on all 45 questions,
mine agrees 45/45 and the shipped version 44/45. The case it gets wrong is Q40, where the
gold answer says the phone number is absent but the words "helpline" and "number" appear in
the document, so word overlap calls it present.

## Part B — rank by expected value, and the prediction

Per-question context composition (measured, offline):

| | gold chunks in context | distractor chunks |
|---|---|---|
| ranking cluster | 1–4 of 5 | 1–4 |
| generation cluster | 0–4 of 5 | 1–5 |

| Cluster | n | Candidate fix | Est. recovery | Cost Δ | Latency Δ | Effort |
|---|---|---|---|---|---|---|
| ranking (4) | 6 | fewer distractors: lower final_k | 1–2 | ~0.8× | none | trivial |
| generation (6) | 6 | more of the gold doc: raise final_k | 2–3 | ~1.5× | none | trivial |
| **both** | **12** | **document expansion: same budget, spent on the best document** | **4–7** | **~1.0×** | none | small |

**Why this pick.** The two clusters pull in opposite directions — one wants fewer passages,
the other more — so no change to `final_k` can serve both. What serves both is changing
*which document* the passages come from. The fix keeps the top 4 chunks of the original
ranking and expands the **top document** to its 3 best chunks in document order.

**Prediction, stated before implementing:** recover **3–5 of the 6 generation** failures and
**1–2 of the 6 ranking** failures, so **4–7 of 12 overall**; refusal precision to improve
slightly; **risk:** questions needing three gold documents (Q26) get worse, and cost rises
with context length.

## Part C — the fix, and what is already measured about it

`DocExpansionRetriever` in `diagnose.py`. One variable changes against Lab 4: which
passages reach the generator. Same embeddings, prompt, tier and generator.

Offline composition check over all 42 answerable questions (no model calls — this is
retrieval only):

| | v1 (Lab 4) | v2 (fix) |
|---|---|---|
| gold chunks as a share of context, all questions | 0.548 | **0.640** |
| same, the 12 failures | 0.517 | **0.590** |
| gold-document recall | 0.891 | 0.843 |
| mean context slots | 5.0 | **5.1** |

The fix is **cost-neutral by construction**: 5.1 slots against 5.0, and no extra model
calls, so it cannot breach the ≤ 2× cost rule.

**Parameters were chosen on composition, not on the score.** I swept five settings offline
and picked keep-4 + expand-1×3. The purest setting (one document, five chunks) reached 0.810
gold share but dropped document recall to 0.554, which would break every multi-document
question — a good example of a number improving while the system gets worse.

Concretely, Q05's context changes from one grace-period chunk to three consecutive chunks
of the gold document in document order — exactly what its failure needed.

## Part D — proof

`python labs/lab5/diagnose.py --before-after`. Same 45 questions, same judges, same
prompt and tier; v1's numbers are read from `reports/lab4.json` so the baseline cannot
drift. Only the retriever wrapper changed.

> **Which rubric these numbers use.** Both columns were scored with the correctness rubric
> in force when the comparison ran (Lab 4's first rubric). Lab 4's `reports/lab4.json` was
> afterwards re-judged under its corrected rubric, so the v1 correctness stored there now
> reads 0.936 rather than the 0.838 below. **The comparison is unaffected** — v1 and v2 were
> scored by one rubric, which is the only property a before/after needs — but the two
> reports quote different baselines on purpose, and this is why.

**D1 — before/after**

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| correctness | 0.838 | **0.949** | **+0.111** |
| faithfulness | 0.978 | 0.956 | −0.022 |
| citation validity | 1.000 | 1.000 | 0.000 |
| refusal recall | 1.000 (5/5) | 1.000 (5/5) | 0.000 |
| refusal precision | 0.625 (5/8) | **0.714 (5/7)** | +0.089 |
| repair rate | 0.000 | 0.000 | 0.000 |
| p95 latency | 2,300 ms | 1,075 ms | −1,225 ms |
| cost / query | $0.00099 | $0.00113 | **1.14×** (rule: ≤ 2×) |
| mean context slots | 5.0 | 5.2 | +0.2 |

**Prediction vs outcome.** I predicted 3–5 of the 6 generation failures, 1–2 of the 6
ranking failures, 4–7 overall. Measured: **7 recovered, 1 genuine regression.**

| | recovered | predicted |
|---|---|---|
| ranking cluster | Q01, Q23, Q25, Q43 (4 of 6) | 1–2 — **underestimated** |
| generation cluster | Q05, Q10, Q26 (3 of 6) | 3–5 — correct |

Q25 moved 0 → 2: with three consecutive chunks of the maternity document it stopped
refusing the arithmetic and computed the ₹60,000 out-of-pocket figure. The ranking cluster
did better than I expected, which is the part of the prediction I got wrong: I treated
"distractors crowd the answer" as a weaker effect than "the answer is incomplete", and on
this corpus they were the same effect seen from two sides.

**D2 — the regression check**

- **Q11 regressed, 2 → 1, and it is the risk I predicted.** "How much ambulance cover is
  there?" needs two documents: `plans-overview` for the ₹5,000 road figure and `plan-gold`
  for the air-ambulance cover. Expanding the top document to three chunks pushed `plan-gold`
  out of the context, and the answer lost the Gold/Platinum half. This is the measured cost
  of the 0.891 → 0.843 drop in gold-document recall, and it is the single clearest argument
  for making expansion adaptive rather than fixed (D4).
- **Q44 is not a regression.** It shows 0 → `None` because the judge's verdict failed to
  parse. The answer is byte-identical in v1 and v2 (both refuse), so this is missing data,
  not a failing answer — the Lab 4 war story applied to our own numbers. It is excluded
  from the correctness mean rather than scored 0.
- **Faithfulness fell 0.022** (one question: Q25, which now asserts a computed figure
  rather than refusing). A system that answers more can be unfaithful more; the trade is
  visible and small.
- **Refusal precision improved** (8 refusals → 7, still 5 correct), the opposite of the
  usual direction, because the cleaner context gave the model less reason to decline.
- **Cost rose 1.14×**, well inside the 2× rule, and p95 latency more than halved (a fresh
  API key with fresh quota, so read that as environment, not as a property of the fix).

**D3 — re-classification of what remains.** Failures fall 12 → 5: **Q04, Q11, Q29, Q32,
Q37**. Q11 is new and is a ranking failure by construction (the second gold document no
longer reaches the generator). Q37 is unchanged and still unreachable — its gold documents
are never retrieved at all. Q29 (trap_archived) survives both versions. The distribution shifted the way the diagnosis said it would:
the cluster that was "the answer arrived incomplete" is largely gone, and what is left is
dominated by questions needing **more than one** document.

## The fix that did not work

Part C chose between two candidates on an offline composition proxy. The rejected one was
the purer context: **one document, five chunks** — offline it had the best gold share
(0.810 vs 0.640) and the worst gold-document recall (0.554 vs 0.843). Part C predicted the
recall would matter more. I ran it to find out rather than leaving that as an assertion
(`--variant single`, `reports/lab5_variant_single.json`).

| Metric | baseline | single-document variant | Δ |
|---|---|---|---|
| correctness | 0.936 | 0.900 | **−0.036** |
| faithfulness | 0.978 | 0.933 | **−0.044** |
| refusal precision | 0.625 (5/8) | 0.455 (5/11) | **−0.170** |
| refusals raised | 8 | **11** | +3 |

**It is worse on every quality metric, and the mechanism is exactly the predicted one.**
Starved of a second document, the system does not guess — it refuses: three more wrongful
refusals, which is where the precision collapse comes from. Q34 ("which plan for no
room-rent sub-limit *and* parents-in-law") fell 2 → 0 because the answer needs two plan
documents and only one arrived; Q04 and Q11 fell 2 → 1 the same way. It did rescue Q23, Q25
and Q29, which is why the *offline* proxy liked it: gold share genuinely improved, and gold
share was the wrong thing to maximise.

**What it cost to learn: one 135-call run.** What it bought: the knowledge that context
purity and context coverage trade against each other on this corpus, and coverage wins —
which is the same lesson the Q11 regression teaches from the other side, and the reason D4
proposes making expansion adaptive rather than simply purer.

> **Rubric note.** This comparison is scored with Lab 4's *corrected* correctness rubric on
> both sides (baseline 0.936). The shipped-fix table above predates that correction and is
> scored with the earlier rubric on both sides (0.838 → 0.949). Each comparison is
> internally consistent, which is what a before/after requires; the two are not comparable
> to each other.

**D4 — the next fix, and what it is worth.** Make expansion adaptive: expand the top
document only when the second-ranked document's best score is far below the first's,
otherwise keep the flat ranking. That targets exactly Q11 and the multi-document remainder
without giving up what Q05 and Q25 gained. I would expect it to recover Q11 and leave the
other four untouched, i.e. about +0.02 correctness — small, and honest about being small.
After that, Q37 needs a retrieval change (routing paraphrased queries to hybrid), not a
context change; Lab 3 measured hybrid as worse overall on this corpus, so it would have to
be routed rather than switched on globally.
