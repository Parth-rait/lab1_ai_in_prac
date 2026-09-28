#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json

import requests
import streamlit as st

API = st.sidebar.text_input("Service URL", "http://localhost:8000")

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")

stream = st.sidebar.toggle("Stream the answer", value=True)


def _ask_stream(question: str) -> dict:
    """B3 on the client: show sentences as they arrive, then act on the
    validation event. A `replace` event means the streamed text failed
    citation validation -- swap in the repaired (or refused) answer."""
    box, text, data, ev = st.empty(), "", {}, None
    with requests.post(f"{API}/ask/stream", json={"question": question},
                       stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                d = json.loads(line[5:])
                if ev == "token":
                    text += d["text"]
                    box.markdown(text + " ▌")
                elif ev == "replace":
                    st.caption(f"Streamed answer retracted ({d['reason']}); corrected below.")
                    text = d["text"]
                elif ev == "citations":
                    data.update(d)
                elif ev == "validation":
                    data.update(refused=d["refused"], partial=d["partial"])
                elif ev == "done":
                    data.update(latency_ms=d["total_ms"], ttft_ms=d["ttft_ms"],
                                cost_usd=d["cost_usd"], cached=bool(d["cache_layer"]),
                                cache_layer=d["cache_layer"], guards=d["guards"])
                elif ev == "error":
                    raise RuntimeError(f"{d['status']}: {d['detail']}")
    box.empty()
    data["answer"] = text
    return data


if st.button("Ask", type="primary") and q:
    with st.spinner("thinking"):
        try:
            if stream:
                data = _ask_stream(q)
            else:
                r = requests.post(f"{API}/ask", json={"question": q}, timeout=60)
                r.raise_for_status()
                data = r.json()
        except requests.HTTPError as exc:
            st.error(f"{exc.response.status_code}: {exc.response.text[:300]}")
            st.stop()
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"request failed: {exc}")
            st.stop()

    if data.get("refused"):
        st.warning(data["answer"])
    elif data.get("partial"):
        # v3 prompt: a cited answer to part of the question, then "The sources
        # do not state ...". Shown as an answer with a caveat, not a refusal.
        st.markdown(data["answer"])
        st.info("Partly answered: the policy documents do not cover part of this question.")
    else:
        st.markdown(data["answer"])

    # A4: every citation opens to the source text it points at. A citation
    # the user cannot open is not grounding.
    if data.get("citations"):
        st.subheader("Citations")
    for c in data.get("citations", []):
        with st.expander(f"[{c['index']}] {c['doc_id']}"):
            st.text(c["excerpt"])

    cited = {c["index"] for c in data.get("citations", [])}
    uncited = [s for s in data.get("sources", []) if s["index"] not in cited]
    if uncited:
        with st.expander(f"Other retrieved sources ({len(uncited)}, not cited)"):
            for s in uncited:
                st.markdown(f"**[{s['index']}] {s['doc_id']}**")
                st.text(s["excerpt"])

    if data.get("guards"):
        st.caption("Guards fired: " + ", ".join(data["guards"]))

    cols = st.columns(4)
    cols[0].metric("latency", f"{data.get('latency_ms', 0):.0f} ms")
    cols[1].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[2].metric("cache", data.get("cache_layer") or "miss")
    cols[3].metric("citations", len(data.get("citations", [])))
    if data.get("ttft_ms") is not None:
        st.caption(f"time to first sentence: {data['ttft_ms']:.0f} ms")
    if data.get("trace_id"):
        st.caption(f"trace: `{data['trace_id']}`")

# TODO stretch: a thumbs-down button that appends the case to a review queue.
# That queue is how real golden sets get built.
