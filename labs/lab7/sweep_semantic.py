#!/usr/bin/env python3
"""Lab 7 B1c — find the semantic-cache threshold where wrong hits start.

    python labs/lab7/sweep_semantic.py            # writes reports/lab7_semantic_sweep.json

For each base question (from the golden set) there are two variants:
  paraphrase  same answer, different words        -> a hit is CORRECT
  near-miss   one detail changed, different answer -> a hit is WRONG
Every pair of distinct golden questions is a further negative (990 pairs).

A wrong hit is worse than a miss: a miss costs one pipeline call, a wrong hit
tells a customer the Gold cataract limit when they asked about Silver. So the
threshold is chosen for ZERO wrong hits, and the paraphrase hit rate at that
threshold is the price.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.embed import embed  # noqa: E402
from labs.lab3.search import load_questions  # noqa: E402

# (golden id, paraphrase with the same answer, near-miss with a different answer)
PAIRS = [
    ("Q01", "After I'm discharged, how many days do I get to file for reimbursement?",
            "How many days do I have to submit a cashless claim after discharge?"),
    ("Q02", "What's the Silver plan's room rent cap?",
            "What is the room rent limit on the Bronze plan?"),
    ("Q03", "Does the Bronze plan include maternity cover?",
            "Is maternity covered on the Gold plan?"),
    ("Q04", "How long is the waiting period for a pre-existing condition?",
            "What is the waiting period for maternity benefits?"),
    ("Q05", "What grace period do I get on a yearly policy?",
            "How long is the grace period for an instalment policy?"),
    ("Q07", "How much no-claim bonus does the Gold plan give?",
            "What is the no-claim bonus on Silver?"),
    ("Q08", "Is IVF something I can make a claim for?",
            "Can I claim for cataract treatment?"),
    ("Q09", "On the Gold plan, what's the sub-limit for cataract?",
            "What is the cataract sub-limit on Silver?"),
    ("Q11", "What's the cover for an ambulance?",
            "How much air ambulance cover is there?"),
    ("Q12", "Does the policy cover mental illness?",
            "Is dental treatment covered?"),
    ("Q13", "How long is the free look period?",
            "How long is the grace period?"),
    ("Q14", "How many wellness points does an annual health check-up earn me?",
            "How many wellness points do I get for completing the quit-tobacco programme?"),
    ("Q16", "For an emergency cashless admission, within how many hours must the hospital tell Aurora?",
            "How many hours in advance must a planned cashless admission be notified?"),
    ("Q17", "Is a third party administrator used by Aurora?",
            "Does Aurora use a network of hospitals?"),
    ("Q26", "If I buy the policy today, is knee replacement covered right away?",
            "Is knee replacement covered after an accident?"),
    ("Q29", "How many days can I take to answer a query Aurora raises on my claim?",
            "How many days does Aurora take to settle a reimbursement claim?"),
    ("Q30", "How much notice does a planned cashless admission need?",
            "How much notice does an emergency cashless admission need?"),
    ("Q32", "Which plans have zero co-pay?",
            "Which plans have a 20% co-payment?"),
    ("Q38", "What was Aurora's claim settlement ratio last financial year?",
            "What was Aurora's claim settlement time last financial year?"),
    ("Q45", "Would LASIK for 6 dioptres be covered?",
            "Is a 9 dioptre lasik covered?"),
]

THRESHOLDS = [round(x, 3) for x in np.arange(0.80, 0.995, 0.005)]


def _vec(text: str, model: str) -> np.ndarray:
    v = np.asarray(embed(text, model=model, input_type="query"), dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def main() -> None:
    from labs.lab7.pipeline import get_pipeline
    model = get_pipeline().retriever.base.base.model      # the retriever's own embedder
    golden = {q["id"]: q["question"] for q in load_questions(include_unanswerable=True)}

    para, near = [], []
    for gid, p, n in PAIRS:
        g = _vec(golden[gid], model)
        para.append({"id": gid, "a": golden[gid], "b": p, "cos": float(g @ _vec(p, model))})
        near.append({"id": gid, "a": golden[gid], "b": n, "cos": float(g @ _vec(n, model))})

    gv = {k: _vec(q, model) for k, q in golden.items()}
    cross = sorted(({"a": a, "b": b, "cos": float(gv[a] @ gv[b])}
                    for a, b in itertools.combinations(sorted(gv), 2)),
                   key=lambda r: -r["cos"])

    rows = []
    for t in THRESHOLDS:
        rows.append({
            "threshold": t,
            "paraphrase_hit_rate": int(sum(r["cos"] >= t for r in para)) / len(para),
            "near_miss_wrong_hits": int(sum(r["cos"] >= t for r in near)),
            "golden_cross_wrong_hits": int(sum(r["cos"] >= t for r in cross)),
        })
    safe = [r for r in rows if r["near_miss_wrong_hits"] == 0 and r["golden_cross_wrong_hits"] == 0]
    first_safe = safe[0]["threshold"] if safe else None
    max_wrong = max([r["cos"] for r in near] + [cross[0]["cos"]])

    print(f"{'T':>6} {'para hit':>9} {'near-miss wrong':>16} {'golden-pair wrong':>18}")
    for r in rows[::2]:
        print(f"{r['threshold']:>6.3f} {r['paraphrase_hit_rate']:>9.2f} "
              f"{r['near_miss_wrong_hits']:>16} {r['golden_cross_wrong_hits']:>18}")
    print(f"\nhighest cosine of a WRONG pair : {max_wrong:.4f}")
    print(f"lowest threshold with 0 wrong  : {first_safe}")
    print("\nclosest near-misses (would be wrong hits):")
    for r in sorted(near, key=lambda r: -r["cos"])[:5]:
        print(f"  {r['cos']:.4f}  {r['a']!r}  vs  {r['b']!r}")
    print("paraphrases, lowest cosine (hardest to catch):")
    for r in sorted(para, key=lambda r: r["cos"])[:5]:
        print(f"  {r['cos']:.4f}  {r['a']!r}  vs  {r['b']!r}")

    out = ROOT / "reports/lab7_semantic_sweep.json"
    out.write_text(json.dumps({
        "embed_model": model, "n_paraphrase": len(para), "n_near_miss": len(near),
        "n_golden_pairs": len(cross), "max_wrong_cosine": max_wrong,
        "lowest_zero_wrong_threshold": first_safe, "sweep": rows,
        "paraphrases": para, "near_misses": near, "top_golden_pairs": cross[:10],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
