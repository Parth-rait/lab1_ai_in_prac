#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    AIP_OFFLINE=1 python labs/lab7/gate.py --config labs/lab7/thresholds.yml

It runs the 45 golden questions through labs/lab7/pipeline.py -- the same
Pipeline.answer() the service calls, with the answer cache off -- and judges
them with Lab 4's corrected rubrics. Under AIP_OFFLINE=1 every model call is a
replay of the committed .aip_cache, so CI needs no key and costs nothing.

Two metrics cannot be read off a replay, and faking them is how a gate goes
permanently green:
* cost_per_query_usd -- a replayed call costs $0. Instead, each call's recorded
  token counts are priced at today's rates (aip.cost.price_of), so a longer
  prompt or context still moves the number.
* p95_latency_ms -- a replayed call takes ~0 ms. Instead, the gate uses the
  p95 of the committed LIVE run (LIVE_RUN) -- but only if the replay
  reproduces that run's answers exactly. Change the pipeline and the live
  number no longer describes it, so the metric is reported as not measured
  and the gate fails until someone re-measures live.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.cache import CacheMiss  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402

LIVE_RUN = ROOT / "reports/lab7_p3_v3prompt.json"      # live, 29 Sep, prompt v3
REPORT = ROOT / "reports/lab7_gate.json"


def measure() -> dict[str, float]:
    from labs.lab3.search import load_questions
    from labs.lab4.evaluate import judge_correctness, judge_faithfulness
    from labs.lab7 import pipeline as pl

    p = pl.get_pipeline()
    questions = load_questions(include_unanswerable=True)
    live = {r["id"]: r for r in json.loads(LIVE_RUN.read_text(encoding="utf-8"))["v2_rows"]}

    rows, misses = [], []
    for q in questions:
        unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
        try:
            r = p.answer(q["question"], use_cache=False)
        except CacheMiss:
            misses.append(q["id"])
            continue
        ctx = r.context
        rows.append({
            "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
            "answer": r.answer, "refused": r.refused, "partial": r.partial,
            "citations_valid": all(1 <= c["index"] <= len(r.sources) for c in r.citations)
            and (r.refused or bool(r.citations)),
            "faithfulness": judge_faithfulness(r.answer, ctx),
            "correctness": judge_correctness(q["question"], r.answer, q["gold_answer"]),
            "hit_rate_at_5": (retrieval_metrics(r.retrieved_doc_ids, q["relevant_docs"],
                                                ks=(5,))["hit_rate@5"]
                              if q["relevant_docs"] else None),
            # Recorded token counts priced at list rates: a replay is $0.
            "priced_cost_usd": r.list_price_usd,
            "matches_live": live.get(q["id"], {}).get("answer") == r.answer,
        })
        print(f"  {q['id']:<4} {'REFUSED' if r.refused else 'partial' if r.partial else 'answer':<8}"
              f" corr={rows[-1]['correctness']} faith={rows[-1]['faithfulness']}", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else None

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    ref = [r for r in rows if r["refused"]]
    ok_ref = [r for r in ref if r["unanswerable"]]
    corr = mean(r["correctness"] for r in ans)
    reproduces_live = not misses and all(r["matches_live"] for r in rows)
    live_summary = json.loads(LIVE_RUN.read_text(encoding="utf-8"))["v2_summary"]

    metrics = {
        "correctness": None if corr is None else corr / 2,
        "faithfulness": mean(r["faithfulness"] for r in rows),
        "citation_validity": mean(float(r["citations_valid"]) for r in rows),
        "refusal_recall": len(ok_ref) / len(una) if una else None,
        "refusal_precision": len(ok_ref) / len(ref) if ref else None,
        "hit_rate_at_5": mean(r["hit_rate_at_5"] for r in rows),
        "cost_per_query_usd": mean(r["priced_cost_usd"] for r in rows),
        "p95_latency_ms": live_summary["latency_p95_ms"] if reproduces_live else None,
        # Coverage is gated too: a question that cannot be replayed was never
        # measured, and silently dropping it would inflate every mean.
        "replay_coverage": len(rows) / len(questions),
    }
    info = {
        "n": len(questions), "replayed": len(rows), "cache_misses": misses,
        "refused": len(ref), "refused_correctly": len(ok_ref),
        "partial_declines": sum(r["partial"] for r in rows),
        "judge_parse_failures": sum(r["faithfulness"] is None for r in rows)
        + sum(r["correctness"] is None for r in ans),
        "reproduces_live_run": reproduces_live,
        "live_run": str(LIVE_RUN.relative_to(ROOT)),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"metrics": metrics, "info": info, "config": pl.CONFIG,
                                  "rows": rows}, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nreplayed {len(rows)}/{len(questions)}; reproduces live run: {reproduces_live}; "
          f"refusals {len(ok_ref)}/{len(ref)} correct; partial declines {info['partial_declines']}")
    if misses:
        print(f"NOT IN REPLAY CACHE ({len(misses)}): {', '.join(misses)} -- the pipeline "
              "changed since the cache was recorded; run the gate once online, commit "
              ".aip_cache, then push.")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"\n{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.5g}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
