#!/usr/bin/env python3
"""Run-level logging for the lab harnesses.

`aip.tracing` already records one span per model call (embed, retrieve, llm,
tool). What it does not record is the context a reader needs to interpret a
number afterwards: which input file, how many rows, whether the cache was warm,
what it cost, and whether the run finished at all.

Every failure this module guards against is one that actually happened here:

  * a sweep that wrote nothing because it returned early, with no line saying so
  * $0.0000 and 10 ms quoted as cost and latency from a fully cached replay
  * a defence layer spending money outside the request's budget, unnoticed
  * one lab's input changing because another lab re-scored its output
  * 45-minute background runs with no visible progress (buffered stdout)

Usage:

    from labs.runlog import run, step, cache_warning

    with run("lab4-full", input="data/eval/rag_golden.jsonl", n=45) as r:
        step("generating")
        ...
        r["rows_written"] = 45
        cache_warning(budget)          # loud if the cache did the work
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aip import tracing  # noqa: E402

RUN_LOG = ROOT / "reports/_runs.jsonl"


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def step(message: str, **attrs: Any) -> None:
    """A timestamped, flushed progress line, mirrored into the trace file.

    Flushed because these harnesses are run in the background with stdout
    redirected, where Python block-buffers and a 40-minute run looks identical
    to a hung one.
    """
    print(f"  [{_stamp()}] {message}", flush=True)
    tracing.event("runlog.step", message=message, **attrs)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Identify an input file well enough to detect that it changed.

    Lab 5 reads Lab 4's output. When Lab 4 was re-judged, Lab 5's input changed
    underneath it and the failure count went from 12 to 6 with nothing in any
    log to say why. A hash and a row count in the manifest make that visible.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    raw = p.read_bytes()
    out: dict[str, Any] = {"path": str(p.relative_to(ROOT) if p.is_absolute() else p),
                           "exists": True, "bytes": len(raw),
                           "sha256_12": hashlib.sha256(raw).hexdigest()[:12]}
    if p.suffix == ".json":
        try:
            data = json.loads(raw)
            rows = data.get("rows", data.get("detail")) if isinstance(data, dict) else data
            if isinstance(rows, list):
                out["rows"] = len(rows)
        except Exception:                                  # noqa: BLE001
            pass
    return out


def cache_warning(budget: Any, *, label: str = "") -> bool:
    """Say so, loudly, when the cache did the work.

    Returns True if the run's cost and latency are cache artefacts. A warm
    replay reports ~$0 and ~10 ms; quoting either as a deployment figure is the
    mistake this exists to prevent.
    """
    calls = getattr(budget, "calls", 0)
    cached = getattr(budget, "cached_calls", 0)
    if not calls:
        return False
    ratio = cached / calls
    if ratio < 0.5:
        return False
    print(f"  [{_stamp()}] NOTE{' ' + label if label else ''}: {cached}/{calls} calls "
          f"({ratio:.0%}) served from cache. Cost and latency from this run are "
          f"NOT deployment figures -- re-measure with AIP_CACHE=0 to quote either.",
          flush=True)
    tracing.event("runlog.cache_dominated", cached=cached, calls=calls, ratio=ratio)
    return True


@contextmanager
def run(name: str, **config: Any) -> Iterator[dict[str, Any]]:
    """Wrap a harness run: one trace span, one manifest line, always.

    The manifest is appended even when the body raises, because "this run
    started and did not finish" is the single most useful thing to know when a
    result file is missing or stale.
    """
    record: dict[str, Any] = {
        "run": name,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "run_id": tracing.current_run_id() if hasattr(tracing, "current_run_id") else None,
    }
    t0 = time.perf_counter()
    print(f"\n=== [{_stamp()}] {name} start "
          f"{json.dumps(config, default=str) if config else ''}", flush=True)
    try:
        with tracing.trace(f"runlog.{name}", **{k: str(v)[:120] for k, v in config.items()}):
            yield record
        record["status"] = "ok"
    except BaseException as exc:                           # noqa: BLE001
        record["status"] = f"failed: {type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        record["elapsed_s"] = round(time.perf_counter() - t0, 1)
        record["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        print(f"=== [{_stamp()}] {name} {record.get('status', '?')} "
              f"in {record['elapsed_s']}s -> {RUN_LOG.name}", flush=True)


def tail(n: int = 10) -> None:
    """Print the last n runs. `python -m labs.runlog` to read the history."""
    if not RUN_LOG.exists():
        print(f"no runs logged yet ({RUN_LOG.relative_to(ROOT)} absent)")
        return
    lines = RUN_LOG.read_text(encoding="utf-8").strip().splitlines()[-n:]
    for line in lines:
        r = json.loads(line)
        cfg = {k: v for k, v in (r.get("config") or {}).items()
               if k in ("input", "variant", "layers", "limit", "strict")}
        print(f"  {r['started_utc'][:19]}  {r['run']:<22} {r.get('status','?'):<28} "
              f"{r.get('elapsed_s','?'):>7}s  {cfg or ''}")


if __name__ == "__main__":
    tail(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
