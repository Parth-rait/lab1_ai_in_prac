"""Lab 7 A3: the service's error contract, without touching a model.

The pipeline is replaced by a stub that raises, so these run offline in
milliseconds and test the HTTP mapping only.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aip.cost import BudgetExceeded
from labs.lab7 import service


class _Raises:
    def __init__(self, exc: Exception):
        self.exc = exc

    def answer(self, question: str, **_):
        raise self.exc


class RateLimitError(Exception):
    """Named like litellm's, which is what _is_retryable() keys on."""


@pytest.fixture
def client(monkeypatch):
    def use(exc: Exception) -> TestClient:
        monkeypatch.setattr(service, "pipeline", lambda: _Raises(exc))
        return TestClient(service.app)      # no `with`: skips the startup build
    return use


def _ask(c: TestClient, **body):
    return c.post("/ask", json={"question": "Is cataract surgery covered?", **body})


def test_malformed_request_is_422(client):
    c = client(RuntimeError("unused"))
    assert c.post("/ask", json={"question": "hi"}).status_code == 422
    assert _ask(c, mode="agent").status_code == 422
    assert _ask(c, top_k=0).status_code == 422


def test_budget_exhaustion_is_429(client):
    r = _ask(client(BudgetExceeded("spent $1.01 > limit $1.00")))
    assert r.status_code == 429


def test_provider_outage_is_503_with_retry_after(client):
    r = _ask(client(RateLimitError("429 quota exceeded")))
    assert r.status_code == 503
    assert r.headers["retry-after"] == str(service.RETRY_AFTER_S)


def test_our_bug_is_500_without_stack_trace(client):
    r = _ask(client(KeyError("oops")))
    assert r.status_code == 500
    assert "Traceback" not in r.text and "KeyError" in r.json()["detail"]
