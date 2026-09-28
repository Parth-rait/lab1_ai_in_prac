#!/usr/bin/env python3
"""Stamp result JSONs with the context needed to interpret them.

A number without its provenance is not reproducible and is easy to misread. Two
of this module's own consistency problems came from exactly that: Lab 4's
correctness read 0.838 in one file and 0.936 in another (different rubric
version), and Lab 6 rows read $0.00000 (warm cache, not a free system).

    python scripts/provenance.py            # stamp every reports/*.json
    python scripts/provenance.py --check    # report what is missing, write nothing
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                      # noqa: BLE001
        return "unknown"


def provenance() -> dict:
    from aip.config import MODELS, settings
    return {
        "stamped_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "profile": settings.profile,
        "models": dict(MODELS),
        "cache_enabled": settings.cache_enabled,
        "offline": settings.offline,
        "temperature": settings.temperature,
        "note": "Cost and latency measured with cache_enabled=true are NOT "
                "deployment figures: a warm cache reports ~$0 and ~0 ms. "
                "Re-measure with AIP_CACHE=0 to quote either.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    prov = provenance()
    # Labs 1 and 2 predate this script and are left exactly as submitted.
    skip = {"lab1_test.json", "lab2_grid.json", "lab2_partA_grid.json",
            "lab2_partB_grid.json"}
    missing, stamped = [], []
    for path in sorted((ROOT / "reports").glob("*.json")):
        if path.name in skip:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):          # a bare list of rows
            data = {"rows": data}
        if "_provenance" in data:
            continue
        missing.append(path.name)
        if not args.check:
            data["_provenance"] = prov
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            stamped.append(path.name)

    if args.check:
        print(f"{len(missing)} file(s) without provenance:")
        for m in missing:
            print(f"  {m}")
    else:
        print(f"stamped {len(stamped)} file(s) with git {prov['git_sha']}, "
              f"profile {prov['profile']}, cache={prov['cache_enabled']}")
        for f in stamped:
            print(f"  {f}")


if __name__ == "__main__":
    main()
