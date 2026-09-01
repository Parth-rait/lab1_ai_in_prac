#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # B1a: `evidence` is declared BEFORE the fields it justifies (T2 §3.3).
    # Autoregressive models condition on what they have already written, so
    # this makes evidence act as reasoning that constrains category/urgency
    # (CoT-in-a-schema) rather than a cheaper, post-hoc citation that could
    # rationalise a wrong answer already committed to.

    evidence: str = Field(
        max_length=200,
        description="The span of the ticket that determined the category, "
                    "quoted verbatim from the ticket. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="One of: billing (premium, debits, refunds, invoices, "
                    "the 80D tax certificate, instalment options) · claims "
                    "(an actual or intended claim: cashless, reimbursement, "
                    "settlement amount, deduction, rejection) · "
                    "policy_change (altering the contract: add/remove a "
                    "member, upgrade, port, change contact details) · "
                    "technical (the app, portal, OTP, login, locator, or "
                    "document upload is broken) · complaint (the subject is "
                    "Aurora's own conduct — mis-selling, being kept on "
                    "hold, an ignored grievance) · information (a question "
                    "with no pending transaction behind it). Boundary rule: "
                    "an angry message about a claim is 'claims' if the "
                    "customer still wants the claim processed; it is "
                    "'complaint' only when Aurora's conduct itself is the "
                    "subject."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="1 = answerable from general product knowledge or a "
                    "self-service how-to, nothing to look up (e.g. 'what "
                    "is the waiting period for cataract surgery?'). "
                    "2 = requires looking up or acting on this customer's "
                    "account, or a transaction is in flight (e.g. 'please "
                    "add my newborn'). 3 = something has already gone "
                    "wrong or is stuck and the customer is waiting (e.g. "
                    "'debited twice'). 4 = repeated failure to resolve, "
                    "money or access at risk now, or an explicit "
                    "escalation threat (e.g. 'THIS IS THE THIRD TIME'). "
                    "5 = an emergency in progress, a formal denial "
                    "demanding immediate reversal, or the customer states "
                    "they ARE (not: are threatening to be) escalating to "
                    "the Ombudsman (e.g. 'father is in ICU and cashless is "
                    "DENIED'). Add 1, capped at 5, if the message states a "
                    "same-day or next-morning deadline. Judge the "
                    "situation, not the tone or length: a calm ICU message "
                    "can be 5; a furious message about a tax certificate "
                    "can be 2."
    )

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="The customer's tone, independent of urgency. angry = "
                    "hostile, accusatory, or threatening escalation. "
                    "frustrated = unhappy and tired of trying, but still "
                    "civil — the message must reference a prior failure (a "
                    "repeat attempt, an unanswered request, a delay, "
                    "something not working). A first-time request, however "
                    "terse, is neutral, not frustrated. neutral = factual, "
                    "no emotional content. satisfied = expresses thanks or "
                    "approval. Judge the customer's own words only, not "
                    "agent replies in quoted history."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The customer's plan tier, only if the ticket names it "
                    "explicitly. Use 'unknown' whenever no tier is stated — "
                    "never infer a tier from the customer's tone, the claim "
                    "amount, or anything else."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if the ticket mixes Hindi and English in any "
                    "amount, including Hindi words written in Latin script. "
                    "'en' only if the ticket is English throughout."
    )

    policy_number: str | None = Field(
        default=None,
        pattern=r"^AUR-\d{7}$",
        description="Format: AUR- followed by exactly 7 digits, copied "
                    "character for character from the ticket. Return null "
                    "when the ticket contains no such string. Never invent, "
                    "complete, or reformat a policy number."
    )
    contains_pii: bool = Field(
        default=False,
        description="True if the ticket contains a phone number, or an "
                    "email address that is NOT one of Aurora's own "
                    "published addresses (support@aurorahealth.example, "
                    "grievance@aurorahealth.example) — those two addresses "
                    "never count, including in signature blocks and quoted "
                    "history. A person's name alone does not count."
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number", mode="before")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        # B1j: return None rather than raising. A malformed policy number is a
        # missing policy number, not a corrupt record — and Part C computes
        # this field in code anyway, so failing the whole extraction over it
        # would cost a record to save a field.
        if v is None:
            return None
        v = str(v).strip()
        if v.lower() in {"", "null", "none", "n/a"}:
            return None
        return v if re.fullmatch(r"AUR-\d{7}", v) else None


SYSTEM_PROMPT = """\
You extract structured records from customer support tickets for Aurora \
Health Insurance.

Read the ticket and populate every field in the schema. The field descriptions \
define exactly what each value means — follow them literally.

Rules:
- Use only what the ticket says. Never infer, complete, or invent a value.
- When a ticket is a forwarded chain, classify the newest message. Quoted \
history below '>' lines is context, not the subject.
- Quote evidence verbatim from the ticket; do not paraphrase it.
- If a value is genuinely not present, use the schema's designated absent \
value rather than a guess.
"""


def extract_b(ticket: str) -> TicketRecord:
    try:
        return structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT)
    except StructuredOutputError as e:
        return TicketRecord(
            evidence="",
            category="information",
            urgency=1,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=str(e)[:200],
        )


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


# Aurora's own published support addresses. The annotation guidelines
# (data/README.md) say these never count as PII, and "Known limitations" #2
# documents the actual bug this guards against: the auto-reply tail quoted
# below a '>' line always contains support@aurorahealth.example, and a naive
# regex scan flagged 14% of dev tickets as containing PII purely because of
# it. Verified against gold: with this exclusion, 0/60 dev and 0/120 test
# contains_pii labels are wrong; without it, 8/60 dev are wrong.
_AURORA_EMAILS = {"support@aurorahealth.example", "grievance@aurorahealth.example"}


def extract_deterministic(ticket: str) -> dict:
    """Return {'policy_number', 'contains_pii'} without a model call.

    policy_number:
        C3 -- the trap. A ticket can contain a policy-number-shaped string in
        TWO places: the live body, and a quoted reply below a '>' line from an
        earlier thread, and they are not always the same number.

        Rule: only the live section (everything before the first '>' quote
        marker) is ever a source of truth for policy_number. A policy number
        that appears only in quoted history is not returned -- per
        data/README.md: "A ticket whose only policy-shaped string is inside a
        quoted reply is labelled null." This is not fitted to this dataset:
        it is the annotation guideline itself, so it should generalise to any
        ticket built the same way. (On the current dev/test splits no ticket
        actually has two *distinct* policy numbers, so a naive
        search-live-then-fall-back-to-whole-ticket also happens to score
        0 errors here -- but it would silently pick up a stale quoted number
        the moment one existed. Restricting to `live` is the defensible rule.)

    contains_pii:
        True if the ticket contains a phone number, or an email address that
        is not one of Aurora's own published addresses (`_AURORA_EMAILS`).
        aip.guards._PII_PATTERNS has the patterns. A *name* alone does not
        count for this dataset's labels -- see data/README.md, which flags
        that as a deliberately narrow definition, insufficient for DPDP Act
        purposes (a name + policy number is already personal data there).
    """
    m = QUOTE_MARKER.search(ticket)
    live = ticket[:m.start()] if m else ticket

    found = POLICY_RE.search(live)
    policy_number = found.group(0) if found else None

    patterns = (_PII_PATTERNS.values()
                if isinstance(_PII_PATTERNS, dict) else _PII_PATTERNS)
    contains_pii = any(
        match.group(0).lower() not in _AURORA_EMAILS
        for p in patterns
        for match in p.finditer(ticket)
    )

    return {"policy_number": policy_number, "contains_pii": contains_pii}


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """Compute `escalate` in code, not in the model.

    escalate = urgency >= 4 or 'ombudsman' appears in the ticket

    This is a business rule: it belongs in code where it can be read by a
    compliance officer, changed without touching a prompt, and unit-tested
    (see tests/test_aip.py::test_apply_business_rules_escalate).
    """
    escalate = (rec_fields.get("urgency", 1) >= 4
                or "ombudsman" in ticket.lower())
    return {**rec_fields, "escalate": escalate}


class TicketRecordC(BaseModel):
    """C2: the reduced schema the model sees in Part C.

    `policy_number`, `contains_pii`, and `escalate` are gone: they are now
    computed deterministically in `extract_deterministic` /
    `apply_business_rules`, so the model is never asked for them. Fewer
    fields means a shorter prompt, fewer output tokens, and those fields at
    100% accuracy -- and it is auditable, not just "usually right".
    Descriptions are copied verbatim from `TicketRecord` so behaviour on the
    remaining fields does not shift.
    """
    evidence: str = TicketRecord.model_fields["evidence"]
    category: CATEGORIES = TicketRecord.model_fields["category"]
    urgency: int = TicketRecord.model_fields["urgency"]
    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = \
        TicketRecord.model_fields["sentiment"]
    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = \
        TicketRecord.model_fields["product"]
    language: Literal["en", "hi-en"] = TicketRecord.model_fields["language"]
    needs_human_review: bool = False
    review_reason: str = ""


def extract_c(ticket: str) -> dict:
    try:
        rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT)
        fields = rec.model_dump()
    except StructuredOutputError as e:
        fields = {
            "evidence": "", "category": "information", "urgency": 1,
            "sentiment": "neutral", "product": "unknown", "language": "en",
            "needs_human_review": True, "review_reason": str(e)[:200],
        }

    fields.update(extract_deterministic(ticket))
    return apply_business_rules(fields, ticket)


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))
