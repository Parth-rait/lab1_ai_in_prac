#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.

Everything here is derived from traces alone (C1): no service state is read,
so the page works on a trace file copied off a production box.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402
from labs.lab7.latency_budget import STAGES, request_breakdowns  # noqa: E402

SLO_P95_MS = 6000

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

# Default to the newest run that served real traffic: HTTP requests that reached
# a model. Test runs have requests but no model calls; gate runs the reverse.
def _served(p: Path) -> bool:
    t = p.read_text(encoding="utf-8")
    return '"name": "http.ask' in t and '"name": "llm.' in t


served = next((p.stem for p in runs if _served(p)), runs[0].stem)
chosen = st.sidebar.multiselect("runs", [p.stem for p in runs], default=[served])
rows = [json.loads(line) for p in runs if p.stem in chosen
        for line in p.open(encoding="utf-8") if line.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")
for col in ("cost_usd", "cached", "hit", "refused", "status", "error"):
    if col not in df:
        df[col] = None

# One row per user request: /ask spans and /ask/stream completion events.
req = df[df["name"].isin(["http.ask", "http.ask_stream"])].copy()
req["latency_ms"] = req["duration_ms"].where(req["name"] == "http.ask", req.get("total_ms"))
req["error"] = req["status"].eq("error")
req = req.sort_values("ts")

exact = df[df["name"] == "cache.exact"]
sem = df[df["name"] == "cache.semantic"]
n_lookups = len(exact)
exact_hits = int(exact["hit"].fillna(False).astype(bool).sum())
sem_hits = int(sem["hit"].fillna(False).astype(bool).sum())
model_cost = df[df["name"].isin(["llm.call", "llm.stream"])]["cost_usd"].fillna(0)

c = st.columns(6)
c[0].metric("requests", len(req))
c[1].metric("total model cost", f"${model_cost.sum():.4f}")
c[2].metric("cost / request", f"${model_cost.sum() / max(len(req), 1):.5f}")
c[3].metric("answer-cache hit rate",
            f"{(exact_hits + sem_hits) / n_lookups:.0%}" if n_lookups else "—",
            help=f"exact {exact_hits}, semantic {sem_hits}, of {n_lookups} lookups")
c[4].metric("error rate", f"{req['error'].mean():.0%}" if len(req) else "—")
c[5].metric("p95 latency", f"{req['latency_ms'].quantile(0.95):.0f} ms" if len(req) else "—")

# --------------------------------------------------------------------------
# C4 — alerts. Evaluated on the most recent requests against a trailing
# baseline, so they fire on a change, not on a level.
# --------------------------------------------------------------------------
st.subheader("Alerts")
WINDOW = st.sidebar.number_input("alert window (requests)", 5, 200, 20)
answered = req[~req["error"]]
recent, base = answered.tail(WINDOW), answered.iloc[:-WINDOW] if len(answered) > WINDOW else answered.iloc[0:0]
fired = False
if len(base) >= WINDOW and len(recent) == WINDOW:
    r_now = recent["refused"].fillna(False).astype(bool).mean()
    r_base = base["refused"].fillna(False).astype(bool).mean()
    if r_base > 0 and r_now >= 2 * r_base:
        fired = True
        st.error(
            f"**Refusal rate doubled**: {r_now:.0%} over the last {WINDOW} requests vs "
            f"{r_base:.0%} baseline.\n\n"
            "This usually means the index broke (empty or partial re-index, embedding "
            "model changed), not that users started asking harder questions. **Do:** "
            "check `index.chunks` in `/health` (expect 231), then run "
            "`AIP_OFFLINE=1 python labs/lab7/gate.py`. If the gate passes offline but "
            "live refusals are up, the live index differs from the committed one: "
            "rebuild it and restart the service.")
if len(recent) >= 5 and recent["latency_ms"].quantile(0.95) > SLO_P95_MS:
    fired = True
    st.warning(
        f"**p95 above SLO**: {recent['latency_ms'].quantile(0.95):.0f} ms > {SLO_P95_MS} ms "
        f"over the last {len(recent)} requests. **Do:** open the stage table below. If "
        "`generate` or `embed query` moved, it is the provider: check its status page and "
        "rely on the answer cache; if `search`/`expand` moved, it is us.")
if not fired:
    st.success(f"No alerts. Refusal-rate doubling needs {2 * WINDOW} answered requests "
               f"of history ({len(answered)} so far); p95 SLO is {SLO_P95_MS} ms.")

# --------------------------------------------------------------------------
st.subheader("Latency by stage (live requests only)")
bd = pd.DataFrame(request_breakdowns(rows))
if len(bd):
    table = pd.DataFrame({"p50 ms": bd[STAGES + ["total"]].median(),
                          "p95 ms": bd[STAGES + ["total"]].quantile(0.95)}).round(0)
    st.dataframe(table, width="stretch")
    st.caption("From span durations: embed query = embed.batch; search = retrieve.dense − "
               "embed; expand = retrieve − retrieve.dense; generate = llm.call. Answer-cache "
               "hits and aip.cache replays are excluded (they report ~0 ms).")
else:
    st.caption("No live (uncached) /ask requests in the selected runs.")

left, right = st.columns(2)
with left:
    st.subheader("Request latency over time")
    if len(req):
        st.line_chart(req.set_index("ts")["latency_ms"])
with right:
    st.subheader("Cumulative model cost")
    costs = df[df["name"].isin(["llm.call", "llm.stream"])].sort_values("ts")
    if len(costs):
        st.line_chart(costs.assign(cum=costs["cost_usd"].fillna(0).cumsum()).set_index("ts")["cum"])

st.subheader("Errors")
errs = df[df["status"] == "error"]
st.dataframe(errs[["ts", "name", "error"]] if len(errs) else pd.DataFrame({"errors": []}),
             width="stretch")
