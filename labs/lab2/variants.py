#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import StructuredOutputError, structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    CATEGORIES, SYSTEM_PROMPT, TicketRecord, TicketRecordC,
    apply_business_rules, extract_deterministic,
)

# Fallback record on a StructuredOutputError -- same shape extract_c() uses
# in Lab 1, so a failed call degrades to a flagged, human-reviewable record
# instead of crashing the grid run.
_FALLBACK_FIELDS = {
    "evidence": "", "category": "information", "urgency": 1,
    "sentiment": "neutral", "product": "unknown", "language": "en",
    "needs_human_review": True,
}

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# Two of the six edge cases named in the runsheet do not actually occur in
# the dev split (verified by scanning all 60 rows):
#   - no ticket is satisfied-sentiment AND high-urgency (the two satisfied
#     tickets are both urgency=1)
#   - no ticket has a policy number that appears ONLY after the '>' quote
#     marker (extract.py's own docstring already says as much)
# Rather than force a case that isn't there, each is swapped for the nearest
# real trap in the data -- noted below.
FEW_SHOT_IDS: list[str] = [
    "T0225",  # teaches: billing/complaint boundary. Angry, repeated, even
              # names the ombudsman -- but the subject is a double debit
              # (a transaction), not Aurora's conduct, so gold = billing,
              # not complaint. Contrast with T0054 ("agent mis-sold me
              # this policy"), which IS complaint because the conduct
              # itself is the subject.
    "T0048",  # teaches: null policy_number. Clean, single-issue ticket
              # with no policy-shaped string anywhere -- isolates the
              # "absent means null, never invent" rule without other
              # confounds.
    "T0112",  # teaches: Hinglish. "Kripya ADD MY MOTHER... Koi solution
              # batayiye" -- Hindi words in Latin script mixed into an
              # English sentence, still language=hi-en even though most
              # of the ticket reads as English.
    "T0201",  # teaches: sentiment != urgency (substitutes for the
              # nonexistent satisfied+urgent case). Tone is neutral
              # throughout, but urgency=4 because a reimbursement filing
              # deadline is closing ("before tomorrow morning"). A model
              # that reads tone instead of stakes will under-call this.
    "T0238",  # teaches: quoted/forwarded history is context, not the
              # subject (substitutes for the nonexistent
              # quoted-only-policy-number case, which does not occur in
              # this split). Everything below '> On ... wrote:' is
              # Aurora's own auto-ack; the actual request (a stalled
              # portability, 21 days) is entirely in the live section.
    "T0095",  # teaches: a case Lab 1 got wrong. run_eval.py --variant c
              # --n 5 misclassified this record's urgency (predicted the
              # wrong value while everything else was right) -- it is a
              # wellness-points balance question, i.e. answerable from
              # account lookup alone, not urgency 3+.
]

# A2 note: FEW_SHOT_IDS point at rows in extraction_dev.jsonl, but that file's
# "expected" dict has no "evidence" field (it isn't graded -- GRADED_FIELDS in
# run_eval.py omits it) and carries policy_number/contains_pii/escalate, which
# the model is never asked for here: variants.py already imports
# extract_deterministic and apply_business_rules to compute those in code,
# exactly as Lab 1 Part C did. So the example OUTPUT below matches what the
# model actually produces: evidence (hand-written, a real verbatim quote from
# each ticket) + category, urgency, sentiment, product, language -- in that
# field order, because that is the order TicketRecordC declares them, and
# Pydantic/JSON-schema field order shapes generation order (T2 §3.3).
EXAMPLE_OUTPUTS: dict[str, dict] = {
    "T0225": {
        "evidence": "the double debit on AUR-5319507. Rs 4200 taken twice",
        "category": "billing", "urgency": 4, "sentiment": "angry",
        "product": "unknown", "language": "en",
    },
    "T0048": {
        "evidence": "Please add my mother as a dependent on my policy",
        "category": "policy_change", "urgency": 2, "sentiment": "neutral",
        "product": "unknown", "language": "en",
    },
    "T0112": {
        "evidence": "ADD MY MOTHER AS a dependent on my Aurora Bronze policy",
        "category": "policy_change", "urgency": 2, "sentiment": "neutral",
        "product": "bronze", "language": "hi-en",
    },
    "T0201": {
        "evidence": "How do I submit the post-hospitalisation bills for AUR-7453325",
        "category": "claims", "urgency": 4, "sentiment": "neutral",
        "product": "gold", "language": "hi-en",
    },
    "T0238": {
        "evidence": "I submitted a portability request 21 days ago and heard nothing",
        "category": "policy_change", "urgency": 3, "sentiment": "frustrated",
        "product": "bronze", "language": "en",
    },
    "T0095": {
        "evidence": "How many wellness points do I currently have on AUR-9746149",
        "category": "information", "urgency": 2, "sentiment": "neutral",
        "product": "gold", "language": "en",
    },
}

# B — the same six examples, rendered for TicketRecordReasoned: a `reasoning`
# key first (matching the schema's field order), everything else identical to
# EXAMPLE_OUTPUTS. This is the only thing that changes between few_shot and
# few_shot_reasoned -- one variable at a time (T3 §5.2).
EXAMPLE_OUTPUTS_REASONED: dict[str, dict] = {
    "T0225": {
        "reasoning": "The subject is a double debit -- a billing transaction "
                     "-- not Aurora's own conduct, so this is billing, not "
                     "complaint. It is the third attempt, money is at stake, "
                     "and the ombudsman is named: an escalation threat, "
                     "which is urgency 4.",
        **EXAMPLE_OUTPUTS["T0225"],
    },
    "T0048": {
        "reasoning": "Adding a dependent alters the policy contract, so "
                     "policy_change. Nothing has failed or is stuck, but it "
                     "requires acting on this specific account, so urgency 2, "
                     "not 1.",
        **EXAMPLE_OUTPUTS["T0048"],
    },
    "T0112": {
        "reasoning": "Same request as a dependent addition, written with "
                     "Hindi words in Latin script ('Kripya', 'batayiye'), so "
                     "language=hi-en. Category and urgency follow the same "
                     "logic as any other dependent-addition request: "
                     "policy_change, urgency 2.",
        **EXAMPLE_OUTPUTS["T0112"],
    },
    "T0201": {
        "reasoning": "The customer needs to file post-hospitalisation claim "
                     "documents, so claims. Tone is neutral throughout, but a "
                     "same-day deadline is stated ('before tomorrow "
                     "morning'), which by the urgency rule adds a point -- "
                     "urgency 4 despite the calm tone.",
        **EXAMPLE_OUTPUTS["T0201"],
    },
    "T0238": {
        "reasoning": "Ignoring the quoted auto-reply below '> On ... wrote:', "
                     "the live message is a portability request stalled for "
                     "21 days, so policy_change. Something has already gone "
                     "wrong and the customer is waiting: urgency 3.",
        **EXAMPLE_OUTPUTS["T0238"],
    },
    "T0095": {
        "reasoning": "A wellness-points balance question needs an account "
                     "lookup, not general product knowledge, so it is "
                     "urgency 2, not 1 -- but nothing is stuck or failing, so "
                     "information, not claims or technical.",
        **EXAMPLE_OUTPUTS["T0095"],
    },
}


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(l) for l in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8")]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def _render_block(ids: list[str], outputs: dict[str, dict]) -> str:
    examples = load_examples(ids)
    blocks = []
    for ex in examples:
        output = outputs[ex["id"]]
        blocks.append(
            f"Ticket:\n{ex['input'].strip()}\n\n"
            f"Output:\n{json.dumps(output, ensure_ascii=False)}"
        )
    return "\n\n---\n\n".join(blocks)


def few_shot_block(ids: list[str]) -> str:
    """A2: render the examples into the prompt.

    Each example is rendered as the same Ticket:/Output: shape the model will
    see at inference time, and the Output json uses the exact key set and
    order the schema declares (see EXAMPLE_OUTPUTS above) -- byte-identical
    to what real calls produce, so the model isn't shown one shape and asked
    for another.
    """
    return _render_block(ids, EXAMPLE_OUTPUTS)


def few_shot_block_reasoned(ids: list[str]) -> str:
    """B: the reasoned counterpart of few_shot_block, using
    EXAMPLE_OUTPUTS_REASONED so the shown examples match TicketRecordReasoned
    field-for-field (same A2 rule: shown format == requested format)."""
    return _render_block(ids, EXAMPLE_OUTPUTS_REASONED)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """Lab 1 Part C, no examples. This is the baseline.

    Same pattern as extract_c(): ask the model only for the fields it can't
    get for free (evidence, category, urgency, sentiment, product, language),
    then fill policy_number/contains_pii/escalate deterministically in code.
    """
    try:
        rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier=tier)
        fields = rec.model_dump()
    except StructuredOutputError as e:
        fields = {**_FALLBACK_FIELDS, "review_reason": str(e)[:200]}
    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """zero_shot + the six worked examples from A1/A2, prepended to the ticket."""
    prompt = f"{few_shot_block(FEW_SHOT_IDS)}\n\n---\n\nTicket:\n{ticket}"
    try:
        rec = structured(prompt, schema=TicketRecordC, system=SYSTEM_PROMPT, tier=tier)
        fields = rec.model_dump()
    except StructuredOutputError as e:
        fields = {**_FALLBACK_FIELDS, "review_reason": str(e)[:200]}
    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


class TicketRecordReasoned(BaseModel):
    """B: `reasoning` declared FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition
    the answer; putting it last makes it a post-hoc rationalisation of an
    answer already committed to. This does NOT subclass TicketRecord: that
    would put reasoning after the inherited fields (subclassing appends,
    it can't reorder), and it would also reintroduce policy_number /
    contains_pii, which every other variant here computes deterministically
    instead of asking the model for -- both would confound the comparison
    with few_shot. So every field except `reasoning` is copied field-for-
    field from TicketRecordC, same descriptions, same order, and reasoning
    is the one and only variable that changes.
    """
    reasoning: str = Field(
        max_length=400,
        description="Think step by step, in your own words: which category "
                    "applies and why, and what single fact sets the urgency "
                    "(a deadline, an escalation threat, money or access at "
                    "risk). Decide this BEFORE the fields below -- do not "
                    "restate the ticket, reason about it."
    )
    evidence: str = TicketRecordC.model_fields["evidence"]
    category: CATEGORIES = TicketRecordC.model_fields["category"]
    urgency: int = TicketRecordC.model_fields["urgency"]
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = \
        TicketRecordC.model_fields["sentiment"]
    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = \
        TicketRecordC.model_fields["product"]
    language: Literal["en", "hi-en"] = TicketRecordC.model_fields["language"]
    needs_human_review: bool = False
    review_reason: str = ""


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """few_shot with TicketRecordReasoned: same six examples, rendered with
    a reasoning field first (few_shot_block_reasoned), same prompt otherwise."""
    prompt = f"{few_shot_block_reasoned(FEW_SHOT_IDS)}\n\n---\n\nTicket:\n{ticket}"
    try:
        rec = structured(prompt, schema=TicketRecordReasoned, system=SYSTEM_PROMPT, tier=tier)
        fields = rec.model_dump()
    except StructuredOutputError as e:
        fields = {**_FALLBACK_FIELDS, "reasoning": "", "review_reason": str(e)[:200]}
    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


def cascade(ticket: str) -> dict:
    """C: SMALL first; escalate to MAIN when two SMALL samples disagree.

    Trigger chosen: draw two SMALL samples and escalate on disagreement --
    the runsheet ranks this the strongest of the four options (2x small
    cost, "much the best"), ahead of validation-failed, empty-evidence, and
    urgency>=4 (free, but not a confidence signal).

    THE TRAP: the two samples must NOT both be at the harness's default
    temperature. Two identical-temperature-0 calls are the same request, so
    the response cache serves the second from the first -- the answers come
    back byte-identical, disagreement is never observed, escalation reads
    0%, and nothing errors. The fix applied here: the second sample is drawn
    at temperature=0.7, which changes both the sampling AND the cache key,
    so a genuine second draw actually happens.

    Disagreement is checked on category and urgency -- the two fields that
    drive routing/SLA decisions downstream, and the two the system prompt's
    own rules (category boundaries, the urgency ladder) are most likely to
    catch the model wavering on.
    """
    try:
        first = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier="SMALL")
        second = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT,
                             tier="SMALL", temperature=0.7)
    except StructuredOutputError as e:
        fields = {**_FALLBACK_FIELDS, "review_reason": str(e)[:200]}
        fields.update(extract_deterministic(ticket))
        result = apply_business_rules(fields, ticket)
        result["_path"] = "small"
        return result

    disagree = (first.category != second.category) or (first.urgency != second.urgency)

    if not disagree:
        fields = first.model_dump()
        path = "small"
    else:
        path = "large"
        try:
            large = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT, tier="MAIN")
            fields = large.model_dump()
        except StructuredOutputError as e:
            # Escalation itself failed -- fall back to the first SMALL
            # sample rather than losing the record, and flag it for a human.
            fields = first.model_dump()
            fields["needs_human_review"] = True
            fields["review_reason"] = f"escalation failed: {str(e)[:150]}"

    fields.update(extract_deterministic(ticket))
    result = apply_business_rules(fields, ticket)
    result["_path"] = path
    return result


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
