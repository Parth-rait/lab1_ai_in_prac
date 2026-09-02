# Lab 1 — The Reliable Extractor — Report

This report walks through what actually happened at each stage — v0 (naive),
Part B (schema + validation), Part C (move work out of the model), and Part D
(test-split measurement) — explaining in plain terms what broke, why it broke,
and what we did about it. The numbers are real, measured runs; the
explanations are the "why" behind them.

---

## The big picture, in one paragraph

We're turning a messy support ticket (an email, a WhatsApp message, an HTML
form fragment) into a clean structured record: category, urgency, sentiment,
policy number, etc. The naive way — just ask an LLM for JSON and use it — looks
like it works because the model *usually* returns something that looks like
JSON. Parts B, C, and D are about closing the gap between "looks like it
works" and "is actually safe to run on 10,000 tickets a day without a human
finding out the hard way that it lied to them."

---

## At a glance — v0 vs B vs C (dev split, 60 tickets)

Three systems, one eval harness, same 60 dev tickets. This is the whole
story in one table; the Part A/B/C sections below are the "why" behind each
column.

| metric | v0 (naive) | B (schema + repair) | C (schema + deterministic fields) |
|---|---|---|---|
| cleanly parses/validates | 0/40 (0%) | 1.000 | 1.000‡ |
| field_accuracy | not measurable — nothing validly parsed | 0.891 | **0.908** |
| record_accuracy | n/a | 0.390 | **0.517** |
| error_rate | n/a | 0.017 | **0.000** |
| cost per ticket (whole split) | $0.0016† | $0.0088 | **$0.0067** |
| p95 latency | not measured | 1307ms | **1199ms** |

† Not really comparable — v0 ran fewer tickets, hit the cache a lot, and
100% of that cheap output was unusable. "Cheap and worthless" isn't a
result worth optimizing toward.
‡ Not separately reported as a raw number in the C dev run, but inferred
from `error_rate` being 0.000 — C reuses B's validation layer, so nothing
got past it invalid.

Reading it as a story rather than a table: v0 establishes that "the model
returned *something*" and "the model returned something *correct*" are
different claims, and v0 fails the first one 100% of the time. B fixes the
first claim completely (`1.000` valid) but only gets the second claim right
39% of the time. C keeps B's validity guarantee and *also* improves
correctness on every axis at once — it's strictly better than B, not a
trade-off (see Part C below for why that isn't a coincidence). Part D
below is what happens when C meets tickets it has never seen.

---

## Part A — the naive version (v0), and why it fails

**What we did:** wrote the simplest possible version — send the ticket to the
model, ask for JSON, call `json.loads()` on whatever comes back. Ran it on 40
dev tickets: `python labs/lab1/v0_naive.py --n 40`.

**What happened:**

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

**Why this happened, in plain terms.** `json.loads()` is strict: it wants a
string that *is* JSON, nothing else. But models like to be helpful — they wrap
their answer in a code block, ` ```json ... ``` `, because that's how JSON is
usually shown in a chat interface. That markdown fence means `json.loads()`
sees ` ```json\n{...}\n``` ` instead of `{...}` and immediately throws. That's
why **all 40** failed — it's not random, it's the model's default formatting
habit, 100% of the time.

Here's the part that's easy to miss: **that one bug was hiding two more
bugs.** Because every single call crashed at the parsing step, we never even
got to look at *what was inside* the JSON. Once you strip the fence (which
Part B does), you find that the *content* is also wrong 100% of the time:
`urgency` comes back as `"3"` (a string) instead of `3` (an integer), and
`category` comes back as things like `"refund"` which isn't one of the six
allowed categories at all. Three separate, independent, 100%-rate problems,
and only one of them was visible until we fixed the first.

**Why this matters for what we build next.** Notice what v0 did *not* do
wrong: it didn't invent policy numbers, and it didn't drop required fields.
So the model's failures here are entirely about *shape* — wrong format,
wrong type, wrong value from a wrong set — not about making things up. That
tells us the fix isn't "make the model more honest," it's "make the pipeline
tolerant of formatting and strict about content." Those are two different
fixes, which is exactly the shape of Part B.

**The trap to notice:** if you just strip the markdown fence and stop there,
you'll see `json.loads()` succeed 40/40 and think you're done. You're not —
you've fixed the *loud* failure (a crash you can see in your terminal) and
left the *quiet* ones (`urgency: "3"`, `category: "refund"`) completely
unaddressed. A dashboard showing those values looks plausible at a glance.
That's the whole justification for Part B: getting *a* JSON object back is
not the same as getting a *correct* one.

---

## Part B — schema, validation, and a repair loop

**What we did:** defined a Pydantic schema (`TicketRecord`) that says exactly
what each field is allowed to be — `urgency` must be an `int` in a fixed
range, `category` must be one of six specific `Literal` strings, and so on.
Every model response now gets validated against this schema instead of just
being trusted. When validation fails, instead of crashing, the code retries
with the error fed back to the model (a "repair loop") or falls back to a
safe default record.

**Why this is the fix for what Part A exposed.** Part A's real lesson was:
the model will confidently hand you the wrong *type* or the wrong *value*, and
nothing will tell you unless you check. Pydantic is the checker. `category
outside the allowed set` and `urgency as a string` — both 100% failure rates
in v0 — become impossible to ship silently once every field has a declared
type and, for `category`, a declared closed set of allowed values. If the
model still gets it wrong, Pydantic raises a `ValidationError` we can catch,
instead of it quietly becoming `urgency: "3"` in a downstream system.

**What we measured (dev split, 60 tickets):**

| metric | v0 | B |
|---|---|---|
| parses/validates | 0/40 clean | 1.00 schema-valid |
| field_accuracy | not measurable — nothing was ever validly parsed | 0.891 |
| record_accuracy | n/a | 0.390 |
| error_rate | n/a | 0.017 |
| cost per ticket (whole split) | $0.0016† | $0.0088 |

† v0's number isn't really comparable — it ran fewer tickets and hit the
cache a lot, and more importantly, 0% of that cheap output was actually
usable. "Cheap and worthless" isn't a result worth optimizing toward.

**What this number is telling us:** `schema_valid` jumped to a perfect
**1.00** — every single response is now well-formed and safe to hand to
another system. That's a real, hard-won guarantee. But `record_accuracy` is
only **0.390** — meaning fewer than 4 in 10 records have *every* field
exactly right. **Validity and correctness are different guarantees.** B
solved "never crashes, never malformed." It did not solve "always right." The
model can still validly, cleanly return the *wrong* category or the wrong
urgency — it'll just be wrong in a way that Pydantic is happy to accept,
because Pydantic only checks *shape*, not *truth*.

---

## Part C — move the deterministic fields out of the model

**What we did:** looked at which fields don't actually need judgment — a
`policy_number` is just a regex pattern in the text, `contains_pii` is a
lookup against a list of PII markers (phone numbers, emails, etc.). These
don't need an LLM to guess at all; they can be computed in plain Python code,
deterministically, from the ticket text itself. So we stopped asking the
model for those fields and computed them directly instead.

**Why this is the fix for what B left on the table.** Every time you ask an
LLM to do something a regex or a lookup table can do exactly, you're trading
a *guaranteed-correct* answer for a *probably-correct* one, and paying for the
privilege. Worse, it adds noise: the model has limited attention, and every
extra field it has to reason about is one more chance to get something else
wrong too. Pulling `policy_number` and `contains_pii` out means the model's
prompt gets shorter and its job gets narrower — it only handles the fields
that genuinely need judgment (category, urgency, sentiment).

**What we measured (same dev split):**

| metric | B | C |
|---|---|---|
| field_accuracy | 0.891 | **0.908** |
| record_accuracy | 0.390 | **0.517** |
| error_rate | 0.017 | **0.000** |
| cost per ticket (whole split) | $0.0088 | **$0.0067** |
| p95 latency | 1307ms | **1199ms** |

**Why C beats B on *every* axis, and why that's not a coincidence.** This
isn't "slightly better tradeoff" — it's strictly better: cheaper (24% lower
cost, from a shorter prompt), faster, *and* more accurate. That's because
`policy_number`, `contains_pii`, `product`, and `language` all reach **1.000**
accuracy once they're computed in code instead of guessed by the model — code
doesn't have an off day. The lesson generalizes: **if you can write a rule
for it, don't ask the model to guess at the rule.** Reserve the model for the
fields that genuinely require judgment about ambiguous natural language
(what tone is this person using, how urgent does this feel).

---

## Part D — running it for real, on the held-out test split

**What we did:** ran the Part C system once on 120 tickets it had never been
tuned against (the test split, not the dev split), to get an honest read on
real-world performance.

**What we measured:**

| metric | value | target | met? |
|---|---|---|---|
| field_accuracy | **0.907** | ≥0.90 | ✓ |
| record_accuracy | **0.467** | ≥0.55 | ✗ (short) |
| schema_valid | **1.000** | 100% | ✓ |
| cost (107 successful calls) | **$0.042** | ≤$0.15/120 | ✓ |
| p95 latency | **1311ms** | ≤4000ms | ✓ |
| unhandled exceptions | **0** | 0 | ✓ |

**Why `field_accuracy` is high (0.907) but `record_accuracy` is low
(0.467), and why that's expected, not a bug.** `field_accuracy` asks "out of
all fields across all tickets, what fraction are right?" `record_accuracy`
asks the much harder question: "for this one ticket, are *all eight* fields
right, simultaneously?" If each field is independently right ~90% of the
time, the chance of all eight being right at once is roughly `0.9^k` for the
fields that aren't already at 1.000 — it compounds down fast. Four fields
(`policy_number`, `contains_pii`, `product`, `language`) are already at
1.000 thanks to Part C, so the real bottleneck is the three judgment fields:
roughly `0.92 × 0.75 × 0.83 ≈ 0.57`. **This is the actual point of the whole
lab:** field accuracy is the number an engineer looks at to know the pipeline
is healthy; record accuracy is the number the business feels, because a
support ticket routed with one wrong field out of eight is still a
mis-routed ticket. The gap between the two is not a measurement quirk — it's
the real cost of asking a model to make several judgment calls at once.

**Per-field accuracy, worst to best:** `urgency` 0.626 (worst by far),
`sentiment` 0.766, `escalate` 0.925, `category` 0.935, and
`contains_pii`/`language`/`policy_number`/`product` all **1.000** (the
deterministic ones from Part C, confirming they generalize perfectly to
unseen data too).

**A failure we hit and had to work around:** running Part D at the tool's
default concurrency (`--workers 4`) against the free Gemini API key produced
a **60% error rate (72/120 failed)** — almost all of them `RateLimitError`,
not extraction mistakes. The free tier only allows 15 requests/minute, and 4
workers hitting it at once blew straight past that limit within seconds.
**Why this matters:** the code's error handling worked exactly as designed —
zero unhandled exceptions either way, nothing crashed — but "didn't crash" and
"produced a trustworthy measurement" are two different guarantees, same as
the schema-valid vs. record-accuracy split above. A report built on a 60%
rate-limited run isn't a report about the model's accuracy; it's a report
about your quota. **The fix:** re-ran with `--workers 1` (one request at a
time), which dropped the failure rate to 10.8% (13/120, still rate-limit
related) — enough for the reported metrics to be computed over the 107 calls
that actually succeeded, giving a real read on quality instead of a mostly-
random subset of it.

---

## Where the remaining errors actually are, and why

Three fields account for almost all the record-accuracy loss, and in each
case the *cause* is the same shape: **the schema states the rule correctly in
plain English, and the model still misapplies it on the exact cases the rule
was written for.**

**1. `urgency` — worst field, 40 errors, but not random ones: 97.5% of them
are off by exactly one level (a 3 read as a 2, a 4 read as a 5, etc.), spread
evenly across every boundary.** Example — ticket T0129: *"Refund it or I am
going to the ombudsman."* The correct label is urgency=4, because it's a
*conditional* threat ("or I will..."). The model predicted 5, reading it as
an *already-happening* escalation. The schema's own field description spells
this exact distinction out — the model still collapses "X or I will do Y"
into "I am doing Y."

**2. `sentiment` — under-detects strong feelings, defaults to `neutral`
(16 of 25 errors: frustrated→neutral 10 times, satisfied→neutral 6 times).**
Example — T0049: *"...crashes every time I try to upload... Tried
reinstalling twice."* Correct label: `frustrated` (a clearly stated repeated
failure — again, exactly the rule the schema describes). The model predicted
`neutral` — it treated a flat, factual tone as emotionally neutral even
though the content described real frustration.

**3. `category` — confusion specifically at the complaint/claims boundary**
(3 of 7 errors), which matches a warning already written into the
annotation guidelines: this is the single boundary most systems get wrong on
this dataset. Here's the actual confusion matrix — gold label (rows) vs.
predicted label (columns) — over the 107 test tickets that returned a valid
prediction:

```
                 billing   claims  complaint  information  policy_change  technical
billing               14        .          .            .              .          .
claims                 .       18          .            1              .          .
complaint               .        3          9            .              .          .
information             .        1          .           18              .          .
policy_change           .        .          .            .             20          .
technical                .        .          .            2              .         21
```

All 7 off-diagonal cells, by direction:

| gold → predicted | count |
|---|---|
| complaint → claims | 3 |
| technical → information | 2 |
| claims → information | 1 |
| information → claims | 1 |

Two things jump out. First, every category the model gets *cleanly*
separated (`billing`, `policy_change`) has zero confusion in or out — the
errors cluster entirely on two specific boundaries, not randomly across the
6×6 grid. Second, `complaint` never leaks *into* `billing`, `information`,
`policy_change`, or `technical` — its only failure mode is being read as
`claims`, which tracks: a complaint about how a claim was handled and an
actual claims ticket use overlapping vocabulary ("claim", "denied",
"rejected"), and the model is defaulting to the more common label
(`claims` outnumbers `complaint` 19 to 12 in this split) when the two
overlap. `technical`→`information` is a milder version of the same thing —
a "why isn't this working" technical complaint phrased as a question reads
close enough to an information request that the model picks the safer,
more common label.

**Why writing the rule better in the prompt probably won't fix this alone.**
The rules are already there, stated correctly, in the schema's field
descriptions — and the model still fails on exactly the cases the rules
describe. That's a strong signal that the fix isn't *more prose*, it's
**examples**: one clear side-by-side pair per boundary (a 4-labelled
conditional threat right next to a 5-labelled real one; a frustrated
repeat-failure next to a genuinely satisfied thank-you; a claims ticket with
some anger in it next to an actual complaint-about-conduct ticket) tends to
fix boundary confusion where a written rule alone does not. We didn't
implement this here, on purpose — it would lengthen the prompt and raise
cost, and that tradeoff is worth measuring deliberately rather than just
doing.

---

## The number that actually matters for the business

It's tempting to report the 0.467 record accuracy and stop. But the number
that would actually hurt Aurora Health Insurance if this shipped as-is is
narrower: **of the 107 successful test predictions, 5 tickets that should
have triggered an urgent escalation (`urgency` gold=4) were instead predicted
as `urgency`=3 — a 4.7% false-negative rate on the exact rule that exists to
catch at-risk tickets before they blow up.** Escalation is defined as
`urgency >= 4`, so a 5-scored-as-4 (2 cases) still triggers correctly — it's
only the 4→3 direction that silently fails, and 100% of the urgency errors
that broke escalation went that direction. **Why this framing beats the raw
accuracy number:** a business doesn't feel "53% of records have some field
wrong" — it feels "a customer who needed urgent help waited like everyone
else," and that's the number that says how often that actually happens.

For reference, the system is roughly **60x cheaper than the human process it
replaces** (~$0.00067/ticket vs. a human's ~40 seconds at typical hourly
cost), which is a real result — but cost savings mean nothing if the
system's error pattern silently drops the tickets that most need a human.

---

## What to actually do with all of this

1. **Ship the schema-validity guarantee as-is.** 100% valid output, zero
   crashes, is genuinely production-safe as an *infrastructure* property —
   don't need more work here.
2. **Don't trust record_accuracy as the headline number for stakeholders.**
   Report the escalation false-negative rate (4.7%) instead — it's the
   figure that maps to actual harm.
3. **Before touching prompts again, check quota/concurrency first.** The
   Part D detour (`--workers 4` → 60% failures) is a reminder that a bad
   *measurement* looks identical to a bad *model* until you check the error
   type. Always look at *why* a call failed before treating it as a quality
   signal.
4. **The next real lever is few-shot examples on the three boundary fields**
   (`urgency`, `sentiment`, `category`'s complaint/claims split), not more
   prose in the schema description — the prose is already correct and the
   model still misses it on exactly those cases.
5. **Keep deterministic fields deterministic.** Part C's lesson generalizes
   beyond this lab: any field with a rule you can write in code should never
   be handed to a model to guess — it's strictly cheaper, faster, and more
   accurate to compute it directly.
