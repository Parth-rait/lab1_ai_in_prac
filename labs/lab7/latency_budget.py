#!/usr/bin/env python3
"""Lab 7 B4 — the latency budget, per stage, from traces alone.

    python labs/lab7/latency_budget.py                 # latest trace file
    python labs/lab7/latency_budget.py --run 20260929-024504

Only requests that reached the model (not answer-cache hits, not aip.cache
replays) are counted: a replayed call reports ~0 ms and would drag every
percentile toward zero.

Stages are derived from the span tree, per request:
    embed query   embed.batch            (0 when the query vector was cached)
    search        retrieve.dense - embed
    expand        retrieve - retrieve.dense   (Lab 5 document expansion)
    guards        guard.input + guard.retrieved + guard.output
    generate      llm.call / llm.stream  (sum: includes a repair call if any)
    other         request total - all of the above (validation, formatting)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402
from aip.tracing import read_traces  # noqa: E402

STAGES = ["embed query", "search", "expand", "guards", "generate", "other"]


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0.0


def request_breakdowns(spans: list[dict]) -> list[dict]:
    by_id = {s["span_id"]: s for s in spans}
    children: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        if s.get("parent_id"):
            children[s["parent_id"]].append(s)

    def descendants(sid: str):
        for c in children.get(sid, []):
            yield c
            yield from descendants(c["span_id"])

    out = []
    roots = [s for s in spans if s["name"] == "http.ask" and s.get("status") == "ok"]
    for root in roots:
        ds = list(descendants(root["span_id"]))
        names = defaultdict(float)
        live = False
        for d in ds:
            names[d["name"]] += d.get("duration_ms", 0.0)
            if d["name"] == "llm.call" and not d.get("cached"):
                live = True
        if not live:
            continue
        embed = names["embed.batch"]
        stage = {
            "embed query": embed,
            "search": max(names["retrieve.dense"] - embed, 0.0),
            "expand": max(names["retrieve"] - names["retrieve.dense"], 0.0),
            "guards": names["guard.input"] + names["guard.retrieved"] + names["guard.output"],
            "generate": names["llm.call"],
        }
        stage["other"] = max(root["duration_ms"] - sum(stage.values()), 0.0)
        stage["total"] = root["duration_ms"]
        out.append(stage)
    del by_id
    return out


def stream_stats(spans: list[dict]) -> list[dict]:
    return [s for s in spans if s["name"] == "http.ask_stream" and not s.get("cache_layer")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="trace run id (default: newest file)")
    args = ap.parse_args()
    run = args.run or max(settings.trace_dir.glob("*.jsonl"),
                          key=lambda p: p.stat().st_mtime).stem
    spans = read_traces(run)
    rows = request_breakdowns(spans)
    print(f"run {run}: {len(rows)} live /ask requests\n")
    if rows:
        print(f"{'stage':<14}{'p50 ms':>10}{'p95 ms':>10}{'share of p50 total':>22}")
        tot50 = _pct([r["total"] for r in rows], 50)
        for st in STAGES:
            xs = [r[st] for r in rows]
            print(f"{st:<14}{_pct(xs, 50):>10.0f}{_pct(xs, 95):>10.0f}"
                  f"{_pct(xs, 50) / tot50 * 100 if tot50 else 0:>21.0f}%")
        print("-" * 56)
        xs = [r["total"] for r in rows]
        print(f"{'total':<14}{_pct(xs, 50):>10.0f}{_pct(xs, 95):>10.0f}")
    ss = stream_stats(spans)
    if ss:
        print(f"\nstreaming ({len(ss)} live): TTFT p50 {_pct([s['ttft_ms'] for s in ss], 50):.0f} ms, "
              f"total p50 {_pct([s['total_ms'] for s in ss], 50):.0f} ms")
        fts = [s["first_token_ms"] for s in spans if s["name"] == "llm.stream"]
        lat = [s["latency_ms"] for s in spans if s["name"] == "llm.stream"]
        if fts:
            print(f"  provider first token after request start p50 {_pct(fts, 50):.0f} ms; "
                  f"generation call p50 {_pct(lat, 50):.0f} ms")


if __name__ == "__main__":
    main()
