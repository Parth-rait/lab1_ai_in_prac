# Lab 2 — The Prompt Lab: Report

All numbers below come from `reports/lab2_grid.json`, the latest full run of `labs/lab2/grid.py`
on the 60-ticket dev split (`data/eval/extraction_dev.jsonl`), `AIP_PROFILE=gemini`. Every number
in this report was recomputed directly from that file so it can be checked against the raw data.

## Headline finding

The cheap configuration wins. Zero-shot on the SMALL model (`gemini-3.5-flash-lite`) scores
51.7% record accuracy and 90.8% field accuracy at roughly $0.67 per 1,000 tickets. Adding
few-shot examples, adding a reasoning field, or routing through a cascade to the MAIN model
all cost more, and none of them beats zero-shot by a margin a paired test can tell apart from
noise. That is the intended result of this lab, not a failure to find an improvement.

## What could not be measured, and why

Google's free tier caps the MAIN model (`gemini-3.7-flash`) at 20 requests per day, per project,
total. That limit is shared across every session this project has ever run, so by the time this
grid was run there was almost no quota left. This affected three of the seven configurations
directly:

- **`few_shot_main` and `few_shot_reasoned_main` have no saved results at all.** Neither
  configuration completed a run in this project; there is no file anywhere in `reports/` that
  contains them. Any table that lists specific numbers for these two rows is not backed by data
  and should not be trusted.
- **`zero_shot_main` did run, but only 6 of its 60 calls ever reached the model.** The other 54
  failed with `RESOURCE_EXHAUSTED` and were scored using the fallback record (see
  `variants.py::_FALLBACK_FIELDS`), which is mostly wrong by construction. That drags the
  reported numbers (5.0% record accuracy, 9.2% field accuracy) down to a level that looks like
  "the big model is bad at this task." It isn't a measurement of the big model at all — it's
  mostly a measurement of a quota outage. The one honest use of this row is the 6 real predictions
  buried inside it, which is too small a sample to base a decision on.
- **The cascade also lost calls to the same wall.** 4 of its 60 tickets triggered a disagreement
  between the two SMALL-tier samples, tried to escalate to MAIN, and errored out instead of
  completing. Those 4 tickets are counted as failures in the table below, not as successful
  escalations, which understates the cascade's own escalation rate (more in Part C).

Because of this, **the model-tier axis cannot be answered with a real number in this report.**
Every claim below is about prompt strategy on the SMALL tier, where all four configurations ran
to completion with a real n of 60 (56 for the cascade, after the 4 quota errors).

## The grid table

| variant | record acc. | field acc. | schema valid | cost / 1k tickets | cost / yr @ 10k/day |
|---|---|---|---|---|---|
| **zero_shot** (SMALL) | 51.7% [39.3–63.8%] | 90.8% | 100% | **$0.67** | **$2,457** |
| few_shot (SMALL) | 50.0% [37.7–62.3%] | 91.3% | 100% | $0.83 | $3,024 |
| few_shot_reasoned (SMALL) | 51.7% [39.3–63.8%] | 91.5% | 100% | $1.37 | $4,992 |
| cascade (SMALL → MAIN) | 46.7% [34.6–59.1%] | 84.8%¹ | 93.3%¹ | $1.31 | $4,790 |
| zero_shot_main (MAIN)² | 5.0% | 9.2% | 10.0% | not measurable | not measurable |

¹ Lower than the other rows mainly because 4 of 60 tickets errored out entirely (quota, see
above) and score as zero across the board — not because the cascade's completed predictions are
worse.
² Included for completeness only. 54 of 60 calls failed to quota; treat this row as an outage
record, not a result. It is excluded from every comparison and every claim in this report.

Cost is derived from measured tokens per ticket, priced at Gemini's published per-token rates,
because the harness's own cost column reads close to $0 here — nearly every call in this run was
served from the local response cache, which is free to replay but not what a ticket costs in
production. Reporting the cached $0 as the real cost would be exactly the "cost meter says zero
for a paid API" mistake this module warns against.

Latency could not be freshly measured for `zero_shot` or `few_shot` this run either — both were
served entirely from cache, so there are zero live calls to time. The only variants with enough
live calls to say anything about latency are `few_shot_reasoned` (4 live calls, roughly
1.3–1.5 seconds each) and the cascade (28 live calls, p50 around 1.2s, p95 pulled up past 11s by
a few slow retries). Since every SMALL-tier variant makes the same one call per ticket, 1.3–1.5s
is the best available estimate for `zero_shot` and `few_shot` too, but it is an estimate carried
over from a different row, not something this session measured directly for them.

**Dominated configurations:** `few_shot`, `few_shot_reasoned`, and `cascade` are all dominated by
`zero_shot` — none of them scores higher on record accuracy, and all three cost more.

## Part A — few-shot selection

**A1 — the six examples**, from `variants.py::FEW_SHOT_IDS`, each chosen to teach a boundary
the model would otherwise miss:

| id | teaches |
|---|---|
| T0225 | An angry message about a double debit is `billing`, not `complaint`, because the transaction is the subject, not Aurora's conduct. |
| T0048 | No policy number anywhere in the ticket means `policy_number = null`. Never invent one. |
| T0112 | Hindi words written in Latin script ("Kripya", "batayiye") still count as `language = hi-en`, even inside a mostly-English sentence. |
| T0201 | Sentiment is not urgency. The tone here is calm, but a same-day filing deadline still means `urgency = 4`. |
| T0238 | Text after a forwarded reply marker (`> On ... wrote:`) is old context, not the live request — classify only the message above it. |
| T0095 | A case the Lab 1 extractor actually got wrong (urgency), included so the model sees a real failure mode, not just an invented one. |

**A4 — the leakage problem.** These six examples come from the dev set, and both `zero_shot` and
`few_shot` are then scored on the dev set. That means 6 of the 60 graded tickets are ones the
model was handed the answer to, as a worked example, before being asked to solve them. That can
inflate `few_shot`'s score on exactly those 6 tickets and nowhere else.

**The fix:** re-score both configurations on the 54 dev tickets that are *not* in the example
set, and compare on that held-out slice instead of the full 60.

| | zero_shot | few_shot |
|---|---|---|
| record accuracy, all 60 | 51.7% | 50.0% |
| record accuracy, held-out 54 | 51.9% | 44.4% |
| paired test, held-out 54 | — | b=8, c=4, p=0.39, no significant difference |

The direction confirms the leakage: `few_shot` gets all 6 of its own worked examples correct
(6/6), and dropping them takes its score from 50.0% down to 44.4%, while `zero_shot` barely moves
(51.7% → 51.9%) because it never saw those 6 as examples in the first place. Leakage was
propping up `few_shot`'s raw number, just not by enough to change the conclusion — both the raw
and the corrected number land on "no detectable difference."

## Part B — three questions

**Which axis mattered more, prompt or model?** This can't be answered honestly this run. The
model-tier axis is the one blocked by the quota wall (see above), so there is no clean
measurement to compare against the prompt-strategy axis. What can be said: within the SMALL
tier, moving from zero-shot to few-shot to few-shot-with-reasoning moved record accuracy by at
most 1.7 points (50.0% to 51.7%), and none of that movement is distinguishable from noise. If
model tier moves the numbers anywhere near as much as the (broken) `zero_shot_main` row suggests,
it would dwarf that, but that row isn't trustworthy enough to make the comparison.

**What did the reasoning field cost, and what did it buy?** `few_shot_reasoned` adds about 67
extra output tokens per ticket over plain `few_shot` (126 vs 58), which comes to roughly $0.54
more per 1,000 tickets. What it bought: one extra correct ticket out of 60, moving record
accuracy from 50.0% to 51.7%. The paired test on those two configurations gives b=8, c=9,
p=1.00, no significant difference. So the honest answer is that the extra spend bought nothing
measurable, not a small win.

**Is anything dominated?** Yes. `few_shot`, `few_shot_reasoned`, and `cascade` are all worse than
`zero_shot` on record accuracy and more expensive. There is no configuration among the
measurable ones that beats `zero_shot` on both quality and cost.

## Part C — the cascade

**Design.** Two SMALL-tier samples per ticket, the second drawn at temperature 0.7 instead of 0
so it is a genuine second draw rather than a cache replay of the first (this is the trap the lab
warns about: two temperature-0 calls are the same request, and the cache serves the second one
for free, so disagreement is never observed). Escalate to MAIN when the two samples disagree on
`category` or `urgency`.

- **Escalation rate:** 56 of 60 tickets had the two SMALL samples agree and stayed on the small
  path. The other 4 disagreed and triggered an escalation attempt — all 4 hit the MAIN quota wall
  and errored out instead of returning a result. So the true escalation-triggering rate is at
  least 4/60 (6.7%), even though the harness's own escalation counter reads 0%, because it only
  counts escalations that completed successfully.
- **Blended cost:** $1.31 per 1,000 tickets, almost double `zero_shot`'s $0.67, because the
  cascade always pays for two SMALL calls per ticket before it even considers escalating.
- **Blended accuracy:** 46.7% record accuracy, lower than plain `zero_shot`'s 51.7%, though not
  significantly so (b=3, c=0, p=0.25) — most of that gap is the 4 tickets that errored out
  entirely rather than a genuine accuracy loss.

**Does the disagreement trigger actually carry signal?** Comparing the two SMALL samples
directly against gold, independent of the cascade's own logic: when the record was correct
(n=31), the two samples agreed 93.5% of the time. When the record was wrong (n=29), they agreed
100% of the time. Of the 29 wrong records, the disagreement trigger caught **zero**.

That is the real finding of this part. The model isn't uncertain when it's wrong, it's
confidently and consistently wrong — the same mistake shows up in both samples. Self-consistency
sampling is built to catch variance (the model wobbling between two answers), and this dataset's
errors are bias (the model landing on the same wrong answer every time), so the trigger has
nothing to catch. A cascade built on this trigger is not a useful safety net for this task.

## Part D — is the difference real?

| comparison | b | c | p | verdict |
|---|---|---|---|---|
| zero_shot vs few_shot | 8 | 7 | 1.00 | no significant difference, choose on cost |
| zero_shot vs few_shot_reasoned | 10 | 10 | 1.00 | no significant difference, choose on cost |
| few_shot vs few_shot_reasoned | 8 | 9 | 1.00 | no significant difference, choose on cost |
| zero_shot vs cascade | 3 | 0 | 0.25 | no significant difference, choose on cost |

`grid.py` also reports `zero_shot vs zero_shot_main: b=29, c=1, p<0.0001`, which looks like the
most decisive result in the whole grid. It is not usable. 54 of `zero_shot_main`'s 60 tickets
failed to `RESOURCE_EXHAUSTED` and score as automatic losses, so this "highly significant"
result is a measurement of the quota outage, not evidence that MAIN performs worse than SMALL on
this task. Reporting a p-value without checking what produced it is exactly the mistake this lab
is built to catch, so it is named here and set aside rather than used anywhere in the
recommendation.

At n=60, the 95% confidence intervals on record accuracy are roughly ±12 points wide for every
SMALL-tier configuration, and they all overlap heavily (see the grid table above). That overlap
is exactly why the paired test, not the raw percentages, is what this report's conclusions rest
on.

## Part E — error analysis

**Worst field: `urgency`**, at 56.7% (34/60), far below every other field (sentiment 85.0%,
escalate 88.3%, category 96.7%, everything else 100%).

**Confusion matrix, urgency, zero_shot, n=60 (rows = gold, columns = predicted):**

```
        pred 1  pred 2  pred 3  pred 4  pred 5
gold 1     9       3       .       .       .
gold 2     2      12       2       .       .
gold 3     .       5       4       2       .
gold 4     .       .       5       4       5
gold 5     .       .       .       2       5
```

**What the matrix shows.** Every single urgency error in this run (26 of 26) is off by exactly
one step, never more. The model has the right neighborhood but not the right number. Hidden
inside that: 5 of the 60 tickets have a gold urgency of 4 or higher but a predicted urgency below
4, and because `escalate` is computed directly from urgency, every one of those 5 silently flips
`escalate` from true to false. All 7 of the `escalate` errors in this run trace back to an
urgency call landing on the wrong side of that 3/4 line. The 88.3% escalate accuracy on its own
would never point to that; it takes reading the confusion matrix to see that one upstream field
is responsible for the whole downstream error.

**Three error clusters, out of 29 record-level failures:**

1. **Urgency off-by-one (26 of 29 failures involve it).** The clearest concentration is right at
   the 3/4 boundary. These are tickets where a deadline or repeat-failure signal is present in
   the text but stated mildly, and the model doesn't reliably apply the "add a point for a
   same-day deadline" rule on borderline phrasing.
2. **Sentiment: frustrated confused with neutral or angry (9 of 29 failures).** 8 of those 9 are
   cases where gold is `frustrated` and the model predicted `neutral` or `angry` instead.
   `frustrated` is defined narrowly in the schema (a civil complaint about a *prior* failure, not
   a first-time request), and the model tends to default to the more obvious label on either
   side of it.
3. **Category: complaint mistaken for claims (2 of 29 failures, T0025 and T0153).** Both are
   gold `complaint` but predicted `claims`. This is the exact boundary the system prompt already
   states explicitly (a message about a claim is `claims` if the customer still wants it
   processed, and `complaint` only when Aurora's own conduct is the subject) — a small count, but
   it's the one cluster where an explicit rule in the prompt still isn't fully sticking.

## The recommendation

**Ship `zero_shot` on the SMALL model, single call, no examples.** Measured: 51.7% record
accuracy (95% CI 39.3–63.8%), 90.8% field accuracy, 100% schema validity, at roughly $0.67 per
1,000 tickets, or about $2,457 per year at 10,000 tickets/day. Every other configuration this
report can actually measure, few-shot, few-shot with reasoning, and the cascade, is
statistically indistinguishable from this baseline (every paired p-value is 0.25 or higher) while
costing 1.2 to 2.0 times more. None of them earns its extra cost on this evidence.

**I would change this recommendation if:** (a) a full, quota-unconstrained run on the MAIN tier
showed a paired-test-significant improvement specifically on `urgency`, since that is the one
field whose errors are both concentrated and business-costly (it silently flips `escalate` in
5 of 60 tickets), and a real gain there could justify MAIN's cost; or (b) the business puts a high
enough cost on those 5 silently-missed escalations per 60 tickets that even a small, unproven
accuracy edge becomes worth paying for despite the lack of statistical evidence today.

## Negative results

1. **None of the more expensive SMALL-tier configurations beat the cheap baseline by a
   detectable margin.** Few-shot, few-shot with reasoning, and the cascade all failed to
   separate from `zero_shot` under the paired test. This is the intended outcome of this lab,
   not a failure to find an improvement.
2. **The cascade's disagreement trigger carries no usable signal on this dataset.** The two
   SMALL samples agreed with each other on 100% of the records they both got wrong (0 of 29
   caught). Self-consistency sampling detects variance; this model's errors are bias, and bias
   looks the same in both samples every time.
3. **The model-tier axis is not measurable with this API key.** `few_shot_main` and
   `few_shot_reasoned_main` never completed a run and have no saved data anywhere in this
   project. `zero_shot_main`'s reported numbers rest on 54 failed calls out of 60 and should not
   be read as a real measurement of the MAIN model. This is reported as a limitation, not
   papered over with an invented number.
