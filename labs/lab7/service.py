#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE = None
_STARTED = time.time()


def pipeline():
    """TODO A2: build your Labs 3-5 pipeline once, at startup, and cache it.

    Building it per-request re-embeds the corpus every time. Students do this
    and then report a 40-second p95.
    """
    global _PIPELINE
    if _PIPELINE is None:
        raise NotImplementedError("wire in your pipeline")
    return _PIPELINE


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """TODO A1. Return cost and trace_id in the response -- they are how
    anyone debugs this later."""
    t0 = time.perf_counter()
    try:
        with tracing.trace("http.ask", question=req.question[:120]) as span:
            raise NotImplementedError
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except NotImplementedError:
        raise
    except Exception as exc:                                    # noqa: BLE001
        # TODO A3: distinguish a provider outage (503 + Retry-After) from a
        # genuine bug (500). Returning 500 for a rate limit makes every client
        # retry immediately, which is exactly wrong.
        raise HTTPException(status_code=503, detail="upstream model unavailable") from exc


@app.get("/health")
def health() -> dict:
    """TODO C: index size, model profile, cache stats, uptime."""
    return {"status": "ok", "uptime_s": round(time.time() - _STARTED, 1),
            "cache": cache.stats()}


@app.get("/metrics")
def metrics() -> dict:
    """TODO C2: cost today, cost/query, cache hit rate, p50/p95/p99, error rate."""
    b = global_budget()
    return {**b.as_dict(), "p99_latency_ms": round(b.percentile(99), 1)}


# TODO B2: POST /ask/stream with server-sent events.
#          Then solve B3: you cannot validate citations before you have sent
#          the answer. Pick one of the three strategies and defend it.
