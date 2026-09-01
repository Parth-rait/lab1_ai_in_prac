# Lab 1 — The Reliable Extractor — Report

## Part A — how v0 fails (40 dev tickets, `python labs/lab1/v0_naive.py --n 40`)

| Failure mode | Count in 40 | Example ticket id |
|---|---|---|
| Not valid JSON at all | 0 | — |
| JSON wrapped in a markdown fence | 40 | T0054 |
| Extra prose before or after the JSON | 0 | — |
| Valid JSON, missing a required field | 0 | — |
| Valid JSON, category outside the allowed set | 40 | T0054 |
| Urgency as a string instead of an int | 40 | T0054 |
| Policy number invented (not present in the text) | 0 | — |
| Unhandled exception | 0 | — |

**The arc:** `0/40` parsed by bare `json.loads()` → `40/40` after stripping
the fence → still `0/40` clean. The model fences its JSON deterministically
(100%), and behind that fence `urgency_is_string` and `category_out_of_set`
are *also* at 100% — one trivial parsing bug was hiding two independent,
100%-rate content defects. The zero rows are findings too: this model
doesn't invent policy numbers or drop required keys unprompted — its
failures are entirely about *shape*, not omission or fabrication.

**T1 §3 mapping.** Fence/prose/not-valid-JSON → **#5 Malformed output**.
Category-out-of-set / urgency-as-string → **#6 Schema violation** ("wrong
shape or an out-of-range enum"). Two rows have no clean entry: **unhandled
exception** isn't a failure type itself, it's the visible symptom of *not*
catching #1/#2/#3/#4/#7 — v0 has zero exception handling, so any of those
five would look identical here. **Missing a required field** sits between
entries: #6's own wording is about a *wrong* value at a present key, not an
absent one; a silently missing field reads closer to #4 Truncation (the
model stopped before reaching that key), but nothing forces that reading.

**Why one line isn't the fix.** Stripping the fence defeats #5 but does
nothing about #6 — that gap is the entire justification for Part B: tolerant
parsing gets *a* JSON object, not a *correct* one.

**What a reviewer would notice.** The fence failure is loud (`JSONDecodeError`
in a log). `urgency_is_string`/`category_out_of_set` are quiet — a dashboard
showing `urgency: "3"` or `category: "refund"` looks plausible at a glance.
The loud failure gets fixed fast; the quiet ones ship.

---

## Variant comparison — dev split (60 tickets, `--workers 1`, clean run)

| metric | v0 | B | C |
|---|---|---|---|
| parses/validates | 0/40 clean | 1.00 schema-valid | 1.00 schema-valid |
| field_accuracy | n/a (not measured against gold) | 0.891 | **0.908** |
| record_accuracy | n/a | 0.390 | **0.517** |
| error_rate | n/a | 0.017 | **0.000** |
| cost_usd (whole split) | $0.0016† | $0.0088 | **$0.0067** |
| p95 latency | 910ms† | 1307ms | **1199ms** |

† v0's run was 40 tickets (not 60) and 78% cache-hit, so its cost/latency
aren't comparable to B/C on a per-ticket basis — they're included only to
show that v0 "succeeding" cheaply is exactly the trap: it's cheap because it
does no validation, and 0% of that cheap output was actually usable.

**C beats B on every measured axis**: 24% lower cost (fewer fields requested,
shorter prompt) and *higher*, not just held, accuracy — `policy_number`,
`contains_pii`, `product`, `language` all reach 1.000 once computed in code.

---

## Test split — Part D (120 tickets, run once, `reports/lab1_test.json`)

| metric | value | target |
|---|---|---|
| field_accuracy | **0.907** | ≥0.90 ✓ |
| record_accuracy | **0.467** | ≥0.55 (short) |
| schema_valid | **1.000** | 100% ✓ |
| cost_usd (107 successful calls) | **$0.042** | ≤$0.15/120 ✓ |
| p95 latency | **1311ms** | ≤4000ms ✓ |
| unhandled exceptions | **0** | 0 ✓ |

**Per-field accuracy:** `urgency` 0.626 (worst), `sentiment` 0.766, `escalate`
0.925, `category` 0.935, `contains_pii`/`language`/`policy_number`/`product`
all **1.000**.

**Category confusion matrix:**

```
                 billing  claims  complaint  information  policy_change  technical
billing              14       .          .            .              .          .
claims                .      18          .            1              .          .
complaint              .       3          9            .              .          .
information            .       1          .           18              .          .
policy_change           .       .          .            .             20          .
technical               .       .          .            2              .         21
```

**Dev/test gap.** field_accuracy: dev 0.908 vs. test 0.907 (Δ0.001) —
essentially no gap, expected since labels here are exact by construction
(`data/README.md`, Known Limitations #1). record_accuracy: dev 0.517 vs. test
0.467 (Δ0.05) — a real gap, driven by more `urgency`-boundary tickets in this
test slice.

**Measurement caveat.** 13/120 test calls (10.8%) failed on Gemini's
free-tier rate limit (429, 15 req/min), not extraction quality — metrics are
computed over the 107 that succeeded. A first attempt at the default
`--workers 4` hit 60% rate-limit errors and was discarded as unusable.

---

## Top three error clusters (D4)

**1. `urgency` — adjacent-boundary confusion, not scatter (40 errors, 97.5%
off-by-one, spread evenly across 1↔2, 2↔3, 3↔4, 4↔5).** Clearest case is the
4/5 tense boundary the guide names explicitly: **T0129** — *"Refund it or I am
going to the ombudsman"* — gold=4 (conditional threat), predicted=5 (read as
a stated escalation). The rule is in the schema description verbatim; the
model still collapses "X or I will Y" into "I am Y."

**2. `sentiment` — under-detects `frustrated`/`satisfied`, defaults to
`neutral` (16 of 25 errors: frustrated→neutral 10, satisfied→neutral 6).**
**T0049**: *"...crashes every time I try to upload... Tried reinstalling
twice."* Gold=`frustrated` (a stated prior failure — the exact rule in the
description), predicted=`neutral`.

**3. `category` — complaint/claims boundary (3 of 7 errors).** Matches the
guide's own warning almost exactly ("roughly a third of the errors any
system makes on this dataset are on this one boundary").

**One fix, all three clusters:** the schema descriptions state every rule
correctly, but rules alone under-perform on the exact cases they're meant to
cover. Per T2 §2.2 that's what few-shot examples are for and prose is not —
add one contrastive pair per boundary (a 4-labelled conditional threat next
to a 5-labelled stated one; a frustrated repeat-attempt next to a satisfied
thanks; a claims-with-anger next to a complaint-about-conduct). Not
implemented here to keep the prompt short and cost down — worth measuring
against the cost budget if pursued.

---

## D5 — the economic question

Marginal cost per ticket (excluding cache hits, since production traffic
won't repeat tickets): $0.0424 total ÷ 63 non-cached calls ≈ **$0.00067 /
ticket** — consistent with the reference solution's $0.08 for a full clean
120-item run ($0.08/120 ≈ $0.00067).

At 10,000 tickets/day: **$6.70/day → ≈$2,446/year** (≈₹2.03 lakh/year at
₹83/USD).

Human baseline: 40s/ticket at ₹300/hour = ₹3.33/ticket. At 10,000/day:
**₹33,333/day → ≈₹1.22 crore/year.**

**The system is roughly 60x cheaper than the labor it replaces**, purely on
processing cost.

**Break-even accuracy, naively:** if every incorrect record falls back to the
same 40s manual handling it would have needed anyway (system cost is added
on top of, never in place of, a human catch):
`blended_cost(acc) = ₹0.056 + (1 − acc) × ₹3.33`. Set equal to the pure-manual
₹3.33/ticket: **acc_breakeven ≈ 1.7%** — almost any accuracy "wins"
financially. **This number is close to meaningless** — it assumes every wrong
record is caught and reworked at no extra cost, which is exactly the
assumption that fails in production.

**The number that actually matters:** of the 107 successful test predictions,
**5 tickets that should have escalated (`urgency` gold=4) were predicted
`urgency`=3 and did not escalate** — a 4.7% false-negative rate on the
business rule that exists specifically to catch at-risk tickets. All 5 are
the identical 4→3 miss. Escalation *is* robust to the 5→4 direction (2 cases)
because `escalate = urgency >= 4` still fires at 4 — but it is not robust to
4→3. That 4.7% is the figure worth deploying against, not the 46.7% record
accuracy: it says how often a ticket that should be flagged for urgent
handling silently isn't, which is the actual cost this system could impose
on Aurora if shipped as-is.

---

## One thing that did not work

Running the Part D test-split eval at the default `--workers 4` against the
Gemini free-tier key. The free tier caps at 15 requests/minute; four
concurrent workers burst well past that within seconds, and the first test
run came back with a **60% error rate (72/120)** — almost entirely
`RateLimitError`, not extraction failures. Re-running the identical code at
`--workers 1` dropped the error rate to 10.8% (13/120). The harness's
per-case `try/except` meant this never crashed the process (0 unhandled
exceptions either way), but it produced a report that was unusable as a
quality measurement — a reminder that concurrency has to be matched to the
provider's actual rate limit, not just to what the harness's `--workers` flag
will accept, and that "0 unhandled exceptions" and "a meaningful report" are
different guarantees.
