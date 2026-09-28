#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq

The pipeline itself lives in labs/lab7/pipeline.py so the regression gate
calls exactly the same code.
"""
from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.config import settings  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.llm import _is_retryable  # noqa: E402
from labs.lab7.pipeline import CONFIG, get_pipeline  # noqa: E402

_STARTED = time.time()
RETRY_AFTER_S = 30


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # A2: build once, at startup. The first user should not pay for it.
    get_pipeline()
    yield


app = FastAPI(title="Aurora Policy Assistant", version="1.0", lifespan=lifespan)


def pipeline():
    return get_pipeline()


# --------------------------------------------------------------------------
# Request-level counters for /metrics (C2). The toolkit's global Budget counts
# model calls; these count HTTP requests, which is what a user experiences.
# --------------------------------------------------------------------------
_LOCK = threading.Lock()
_STATUS: Counter = Counter()
_LAT_MS: list[float] = []
_OUTCOME: Counter = Counter()      # answered / partial / refused / guarded
_CACHED = Counter()                # hit / miss (response cache)


def _count(status: int, latency_ms: float | None = None) -> None:
    with _LOCK:
        _STATUS[status] += 1
        if latency_ms is not None:
            _LAT_MS.append(latency_ms)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # Default = the measured configuration. The gate tests final_k=6; a
    # different default here would ship something CI never saw.
    top_k: int = Field(default=CONFIG["final_k"], ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    partial: bool
    citations: list[Citation]
    sources: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str
    guards: list[str]
    mode: str


@app.exception_handler(RequestValidationError)
async def _on_422(request: Request, exc: RequestValidationError):
    _count(422)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """A1. Cost and trace_id are in the response: they are how anyone debugs
    this later ("why did request X cost 10x?" -> grep the trace file)."""
    t0 = time.perf_counter()
    try:
        with tracing.trace("http.ask", question=req.question[:120], mode=req.mode) as span:
            trace_id = f"{tracing.RUN_ID}:{span['span_id']}"
            if req.mode == "tools":
                out = _ask_tools(req.question)
            else:
                r = pipeline().answer(req.question, final_k=req.top_k)
                out = {"answer": r.answer, "refused": r.refused, "partial": r.partial,
                       "citations": r.citations, "sources": r.sources,
                       "cost_usd": r.cost_usd, "cached": r.cached, "guards": r.guards}
            span.update(refused=out["refused"], partial=out["partial"],
                        cost_usd=out["cost_usd"], cached=out["cached"],
                        guards=out["guards"])
    except BudgetExceeded as exc:
        _count(429)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except cache.CacheMiss as exc:
        # Offline mode and the question was never answered online. Not a bug
        # in the service, and retrying will not help: 503 with no Retry-After.
        _count(503)
        raise HTTPException(status_code=503,
                            detail="offline mode: question not in the replay cache") from exc
    except Exception as exc:                                    # noqa: BLE001
        # A3: a provider outage / rate limit is 503 + Retry-After, so clients
        # back off. Anything else is our bug: 500, no stack trace to the client
        # (it is in the trace file under this trace_id).
        if _is_retryable(exc):
            _count(503)
            raise HTTPException(status_code=503, detail="upstream model unavailable",
                                headers={"Retry-After": str(RETRY_AFTER_S)}) from exc
        _count(500)
        raise HTTPException(status_code=500,
                            detail=f"internal error ({type(exc).__name__}); "
                                   f"see traces for run {tracing.RUN_ID}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    _count(200, latency_ms)
    with _LOCK:
        _OUTCOME["refused" if out["refused"] else "partial" if out["partial"] else "answered"] += 1
        if out["guards"]:
            _OUTCOME["guard_fired"] += 1
        _CACHED["hit" if out["cached"] else "miss"] += 1
    return AskResponse(**out, latency_ms=round(latency_ms, 1), trace_id=trace_id,
                       mode=req.mode)


def _ask_tools(question: str) -> dict:
    """mode="tools": the Lab 6 agent with all five layers and the layer-4
    ToolGuard (issue_refund off the allowlist and unconfirmable). SMALL tier:
    MAIN takes ~36 s/call. Not covered by the offline gate -- it is an opt-in
    mode, and the RAG path is what the gate measures."""
    from aip.guards import ToolGuard
    from labs.lab6 import agent
    agent.LAYERS.clear()
    agent.LAYERS.update({1, 2, 3, 4, 5})
    guard = ToolGuard(max_calls=6,
                      allow={"search_policy", "get_policy_details", "compute_premium"},
                      requires_confirmation={"issue_refund"},
                      confirm_fn=lambda name, a: False)
    res = agent.run_agent(question, guard=guard, tier="SMALL", max_seconds=20.0)
    text = res["answer"] or ""
    from labs.lab4.rag import is_refusal
    return {"answer": text, "refused": is_refusal(text), "partial": False,
            "citations": [], "sources": [], "cost_usd": res.get("cost_usd", 0.0),
            "cached": False, "guards": [f"flag:{f}" for f in res.get("flags", [])]
            + [f"stopped:{res['stopped_because']}"]}


@app.get("/health")
def health() -> dict:
    p = pipeline()
    base = p.retriever.base.base            # GuardedRetriever -> DocExpansion -> dense
    chunks = getattr(base, "chunks", None) or getattr(p.retriever.base, "chunks", [])
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index": {"chunks": len(chunks), "docs": len({c.doc_id for c in chunks})},
        "pipeline": CONFIG,
        "profile": os.getenv("AIP_PROFILE", ""),
        "offline": settings.offline,
        "response_cache_rows": cache.stats(),
    }


@app.get("/metrics")
def metrics() -> dict:
    b = global_budget()
    with _LOCK:
        lat = sorted(_LAT_MS)
        ok = _STATUS[200]
        total = sum(_STATUS.values())

        def pct(p: float) -> float:
            return round(lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))], 1) \
                if lat else 0.0

        return {
            "requests": total,
            "status_counts": {str(k): v for k, v in sorted(_STATUS.items())},
            "error_rate": round((total - ok) / total, 4) if total else 0.0,
            "outcomes": dict(_OUTCOME),
            "refusal_rate": round(_OUTCOME["refused"] / ok, 4) if ok else 0.0,
            "cache": dict(_CACHED),
            "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99),
                           "mean": round(statistics.fmean(lat), 1) if lat else 0.0},
            "cost_usd_total": round(b.spent_usd, 6),
            "cost_per_request_usd": round(b.spent_usd / ok, 6) if ok else 0.0,
            "model_calls": b.as_dict(),
        }


# TODO B2 (step 5): POST /ask/stream with server-sent events.
