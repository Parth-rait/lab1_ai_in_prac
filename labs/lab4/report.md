# Lab 4 — RAG v1: Grounded Answers with Citations: Report

Numbers come from `reports/lab4.json` (45 questions), `reports/lab4_strict.json`,
`reports/lab4_decomposition.json` and `reports/lab4_kappa.json`, produced by
`labs/lab4/evaluate.py`. Retrieval is the Lab 3 winner: **markdown-aware chunking at
400 characters, exact dense retrieval, final_k = 5, archived documents excluded**
(Lab 3: nDCG@10 0.860, recall@5 0.903).

**Two deviations from the handout, both forced by free-tier quotas.**
1. **Generation runs on SMALL (`gemini-3.5-flash-lite`), not MAIN.** MAIN
   (`gemini-3.7-flash`) answers in ~36 s per call here, which alone breaks the 6 s p95 target.
2. **The judge runs on SMALL too, not LARGE.** LARGE (`gemini-3.5-flash`) is capped at
   **20 generate requests per day** on this key; a full run needs 90 judge calls. This makes
   judge and generator the same model — self-preference bias, discussed in D3.

## E1 — the target table

| Metric | Target | Measured | |
|---|---|---|---|
| Citation validity | 1.00 | **1.000** (45/45) | ✅ |
| Faithfulness | ≥ 0.90 | **0.978** (44/45) | ✅ |
| Answer correctness (normalised) | ≥ 0.75 | **0.936** | ✅ (κ = 0.52, see D2) |
| Refusal recall | ≥ 4/5 | **5/5 = 1.000** | ✅ |
| Refusal precision | ≥ 0.70 | 0.625 (5/8) | ❌ |
| Repair rate | reported | **0.000** (0/45) | — |
| Cost per query | ≤ $0.01 | **$0.00099** | ✅ |
| p95 end-to-end latency | ≤ 6,000 ms | **2,300 ms** | ✅ |

Latency and cost are measured under free-tier throttling; an earlier identical run
measured p95 8,574 ms purely because the API was slower that hour, so treat p95 as
"~2–9 s, environment-dependent", not as a stable system property.

## Part A — the answer prompt

`ANSWER_SYSTEM` (full text in `labs/lab4/rag.py`) was written before reading
`aip/rag.py`. Abridged — rule text shortened, structure and wording otherwise as shipped:

> You are Aurora Health Insurance's support assistant… using ONLY the numbered sources
> supplied in the user message.
> **1. SOURCES ONLY.** Every factual statement must come from the numbered sources. You
> have no other knowledge… Do not estimate or fill gaps with what is typical in the
> industry. You MAY combine facts stated in different sources… Combining stated facts is
> answering; inventing a fact is not.
> **2. CITE EVERY FACTUAL SENTENCE** by source index, e.g. "Claims must be filed within 30
> days of discharge [2]." Use [1][4] when a sentence draws on more than one source.
> **3. NEVER cite an index you were not given.** If five sources are supplied, the only
> legal citations are [1]–[5].
> **4. WHEN THE SOURCES DO NOT ANSWER THE QUESTION AT ALL**, reply with exactly:
> *I don't have enough information in the provided sources to answer that.* Do not
> paraphrase it, append a topic to it, or attach a citation — downstream checks match it
> exactly.
> **4a. PARTIAL ANSWERS ARE REQUIRED** when the sources support part of the question:
> state the supported part with citations, then name the unsupported part followed by the
> refusal sentence verbatim.
> **5. WHEN SOURCES DISAGREE**, say so and cite both. If one is marked superseded or
> archived, say which and prefer the current one.
> **6. BE BRIEF.** Two or three sentences unless the question needs more… Give figures
> exactly as the source gives them.
> *(plus `UNTRUSTED_SYSTEM_CLAUSE`)*

That covers the six required elements — sources-only, cite by index, never invent an index,
an exact refusal string, surface conflicts, length discipline.

**Differences from the reference:**

| Mine | Reference |
|---|---|
| Sources-only is rule 1; refusal is rule 4 | Refusal is rule 1, "in priority order" |
| Explicit partial-answer protocol (answer the supported part, then the refusal sentence) | No partial-answer instruction |
| "Prefer the current document and say which is superseded" when sources conflict | "If sources disagree, say so and cite both" |
| Forbids paraphrasing, appending to, or citing the refusal sentence | States the sentence only |
| Explicitly permits combining facts stated across sources | Silent |
| Gives the assistant a role and forbids restating the question | Neither |

The reference is shorter and puts refusal first; mine is more specific about the two
behaviours this lab measures. Whether that specificity helps is measured below — partly it
does not.

## Part B — citation enforcement

**Citation validity 1.000 across all 45**, because it is a code guarantee, not a model
behaviour: `validate_answer()` extracts every `[n]` with a regex, rejects any index
outside 1..n_sources (via `aip.guards.enforce_citations`), rejects empty and truncated
answers (`finish_reason == "length"`), and requires at least one citation on any
non-refusal.

**B3 — failure policy: repair once, then refuse.** A corrective retry is cheap and fixes
the common case (the model had the right content and the wrong label). Stripping the bad
citation would leave an uncited factual sentence, which is unauditable and worse than no
answer for an insurance helpdesk. If the retry still fails, the pipeline returns the exact
refusal string. The function cannot return `citations_valid=False` with `refused=False`.

Repair rate was **0/45** in the final run and 3/45 (6.7%) in the previous one — the three
repairs were all out-of-range citation indices, all fixed on the retry.

## Part C — refusal, both directions

| Setting | Refusals | Recall | Precision |
|---|---|---|---|
| Default prompt | 8 | **5/5 = 1.000** | 5/8 = **0.625** |
| Strict prompt (C4) | 9 | **5/5 = 1.000** | 5/9 = **0.556** |
| Earlier prompt version (v1, see below) | 6 | **5/5 = 1.000** | 5/6 = **0.833** |

**These numbers are extremely noisy and no comparison below ~0.15 means anything.**
There are 5 unanswerable questions; one case moves recall by 0.20 and precision by ~0.12.
All three settings found all five unanswerable questions; they differ only in how many
answerable ones they wrongly declined (3, 4 and 1 respectively).

**C4:** strictness bought nothing. Recall was already saturated at 5/5, so the stricter
instruction could only add false refusals, and it did.

**C2 — Q37, the partial-answer case, is not a prompt problem.** Q37 asks about treatment
in Singapore. Its context contains **no mention of "Singapore", "international", "outside
India" or "Platinum"** — neither gold document (`exclusions`, `plans-overview`) was
retrieved. Given that context, refusing is the *correct* behaviour, and no prompt change
can fix it. It is a retrieval failure (mode 3). The partial-answer protocol does fire when
the context supports part of a question: Q23 answers "the OPD rider is not available on
Bronze [1]" and then refuses the rest, and Q25 states the ₹1,00,000 caesarean limit before
declining the arithmetic.

**Product recommendation: keep the default setting, and prefer the v1 prompt's
refusal wording.** For an insurance helpdesk the two error types are not symmetric, but
not in the direction usually assumed. A wrong claim deadline that an agent repeats to a
customer can cost that customer a valid claim and Aurora a regulatory finding; it is
discovered late, by the person harmed. An unnecessary refusal costs an agent about two
minutes and is discovered immediately, by someone who can escalate. That argues for
precision over recall on *answers*, i.e. refuse when genuinely unsupported — which is what
recall 5/5 already delivers. What it does not justify is strictness that converts
answerable questions into refusals: those are invisible in the refusal metrics unless you
measure both directions, and at 3–4 of 40 they are the larger failure here.

## Part D — the judge

**D1.** Two single-criterion rubrics, rewritten from the `aip.evals` templates. The
faithfulness rubric adds the three cases the shipped one mishandles: a partial refusal is
supported; strengthening or generalising the context ("usually" → "always") is not; and
being right about the world while unsupported by the context is not. The correctness
rubric adds explicit refusal handling — the gold answers for unanswerable questions begin
with `REFUSE`/`PARTIAL REFUSE`, so a correct refusal scores 2 and a confident answer 0.

**Parse failures are excluded, not scored 0** (`_score()` returns `None`). There were 0
parse failures in the final run.

**D2 — calibration. 20 hand-labelled answers, labelled before reading the judge's scores.**

| Rubric | Raw agreement | Cohen's κ | Verdict |
|---|---|---|---|
| Faithfulness | 1.00 | 1.00 (degenerate) | not usable evidence |
| Correctness, first rubric | 0.55 | 0.189 | below the 0.4 bar |
| **Correctness, fixed rubric** | **0.79** | **0.519** (n=19) | **above the bar — reportable** |

**Faithfulness κ is meaningless here, and saying "κ = 1.0" would be dishonest.** I labelled
all 20 answers faithful and the judge agreed on all 20, so there is no variance in either
label set; a judge that always says "faithful" scores identically. The criterion is
saturated on this corpus, which is the Lab 3 saturated-metric lesson in a new place.

**The first rubric scored κ = 0.189, below the 0.4 bar**, so no correctness number could be
reported against it. Reading its 9 disagreements showed the fault was my rubric, not the
model:

- **5 cases** (Q01, Q05, Q26, Q29, Q43) — judge 1, me 2. The answer answers the question
  but omits background the reference happens to carry. My rubric literally said "omits
  something the reference states → 1", so the judge was following it correctly.
- **3 cases** (Q20, Q33, Q35) — judge 2, me 1. The answer misses part of what the question
  *asked* ("compare all four plans", "list everything").
- **1 case** (Q25) — judge 0, me 1. A partial answer, which the rubric never mentioned.

**The rubric was fixed and re-run** (rules a/b/c in `RUBRIC_CORRECTNESS`): judge against
what the question asked; extra supported detail never reduces the score; omission counts
only when the question asked for it; a partial answer scores 1. `--rejudge` re-scored the
saved answers without regenerating, so **no answer changed — only its score.**

κ rose **0.189 → 0.519** and raw agreement **0.55 → 0.79**. Seven scores moved, every one in
a direction the new rules predict: Q01, Q04, Q05, Q26 and Q43 from 1 → 2 (they answered the
question and were being penalised for omitting background the reference happened to carry),
Q25 from 0 → 1 (a partial answer), and Q44 from 0 → `None` (the judge's verdict failed to
parse, which is missing data, not a zero).

**Headline correctness therefore moves 0.838 → 0.936, and that is a measurement change, not
a better system.** The answers are byte-identical; the first rubric was miscounting. Four
disagreements remain (Q20, Q33, Q35 where the judge says 2 and I say 1 — each omits part of
what the question explicitly asked; Q29 the reverse), which is what a κ of 0.52 looks like:
moderate agreement, not perfect.

**D3 — self-preference, measured.** The judge is the same model as the generator, because
LARGE's 20-requests/day cap cannot cover 90 judge calls. Self-preference runs **upward**, so
rather than assert the caveat I measured it: `--cross-check 10` re-judged ten answers on
LARGE, a different and stronger model.

| | our judge (SMALL) | independent judge (LARGE) |
|---|---|---|
| mean faithfulness over the same 10 answers | 1.000 | 0.900 |
| agreement | 0.900 | — |

**Gap +0.100, in the self-preference direction** — our faithfulness of 0.978 is an upper
bound, and the overstatement is on the order of a tenth of a point. The single disagreement
is Q04, which our own model called supported and the independent model called unsupported.
At n = 10 (the whole daily budget for that model) this is a direction with a magnitude, not
a precise correction, and it should be read that way.

## Part E — decomposition and failure modes

**E2.** Run on both prompt versions:

| | A (gold context) | B (retrieved) | Retrieval loss A−B | Generation loss 1−A |
|---|---|---|---|---|
| v1 prompt | 0.893 | 0.881 | **0.012** | **0.107** |
| v2 prompt (shipped) | 0.845 | 0.833 | **0.012** | 0.155 |

**Generation is ~9× the larger loss, so Lab 5 belongs in generation, not retrieval.** The
retrieval loss is identical (0.012) across two independent runs, which is a useful
stability check on the measurement. The small A−B also hides offsetting effects: gold
context *helped* 4 questions and *hurt* 4 others — at n = 42 the net is noise, which only
strengthens the conclusion that retrieval is not the bottleneck.

**E3 — the 12 imperfect answers, tagged against T4 §5.** ("Does gold context fix it?" is
taken from the decomposition run, which is the tree's branch test.)

| Mode | n | Questions | Evidence |
|---|---|---|---|
| **4 — ranking / distractors** | 6 | Q01, Q23, Q25, Q29, Q43, Q44 | gold document retrieved; gold context scores higher |
| **6 — generation** | 5 | Q04, Q05, Q10, Q26, Q32 | gold document at rank 1; gold context does not help |
| **3 — embedding mismatch** | 1 | Q37 | gold documents never retrieved at all |

**This is the Lab 5 backlog.** Note 11 of 12 had the right document in the top 5 and 9 had
it at rank 1: almost nothing here is fixable by better retrieval.

## An unplanned result: my prompt iteration made the system worse

I changed the prompt mid-lab to force partial answers (intending to fix Q37). Measured
against the same correctness rubric:

| | Correctness | Refusal precision | Refusals |
|---|---|---|---|
| v1 (original) | **0.887** | **0.833** (5/6) | 6 |
| v2 (partial-answer rule, shipped) | 0.838 | 0.625 (5/8) | 8 |

It did not fix Q37 (that was retrieval), and it made the model readier to refuse: Q25 and
Q44 became refusals. On the 18 questions where neither version refuses, the two are
**identical** (correctness 0.889, faithfulness 1.000 under one rubric) — the whole
difference is refusal behaviour. The faithfulness improvement I first attributed to this
change (0.889 → 0.978) was the **rubric fix**, not the prompt.

**Next step: revert to the v1 refusal wording and re-measure.** It is shipped as v2 only
because v2 is the version with a complete, self-consistent measurement set; the honest
reading of the table above is that v1 is the better system.

## What the quotas cost, and what is left

Free-tier limits shaped this lab more than any design choice: **500 generate requests/day**
on the generation model and **20/day** on the judge tier. That is why generation and judging
share a model (D3), why the cross-check is n = 10 rather than n = 45, and why the first
attempt at a full run died mid-way. All three originally-blocked items have since run on a
fresh key: the re-judge (κ 0.519), the cross-check (gap +0.100) and this report's numbers.

**One item remains deliberately undone.** A v3 prompt permitting arithmetic on stated
figures would likely recover Q25, which refused to subtract ₹1,00,000 from ₹1,60,000. It is
not implemented here because **Lab 5 solved that case a different way** — giving the
generator more of the gold document moved Q25 from 0 to 2 without touching the prompt. Two
fixes for one failure is how you stop being able to attribute either.
