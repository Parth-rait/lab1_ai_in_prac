"""Lab 7 B1 — two answer-cache layers in front of the pipeline.

This is a different cache from aip.cache. aip.cache memoises *model calls*
(same prompt -> same completion) and is what makes CI replay work. These
cache *answers to questions*, so a hit skips retrieval and generation entirely.

    exact     hash(normalised question, top_k, pipeline version)   no risk
    semantic  cosine(query embedding, stored embedding) >= T         can be wrong

The semantic layer reuses the retriever's own query embedding (same model,
input_type="query"), so a miss costs nothing extra: retrieval finds the vector
already in aip.cache. T is measured by labs/lab7/sweep_semantic.py, not
assumed; see SEMANTIC_THRESHOLD below.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass

import numpy as np

from aip import tracing
from aip.embed import embed

# Measured, not assumed (reports/lab7_semantic_sweep.json, gemini-embedding-001):
# the highest-scoring WRONG pair is 0.906 ("settlement ratio" vs "settlement
# time"; "6 vs 9 dioptre" is 0.902; Gold/Silver/Bronze swaps 0.88-0.89), so
# wrong hits start just below 0.91. 0.91 would be 0.004 above that on only 20
# near-misses -- too thin. 0.93 keeps a 0.024 margin and still serves 80% of
# true paraphrases (the starter's 0.95 serves 50%). Set by hand from that file
# so a code change cannot move it silently; re-run the sweep if the embedding
# model changes.
SEMANTIC_THRESHOLD = 0.93

_PUNCT = re.compile(r"[^\w\s₹]")
_WS = re.compile(r"\s+")


def normalise(question: str) -> str:
    """Case, whitespace and punctuation only. Nothing that can change meaning:
    stemming or stop-word removal would merge "covered" and "not covered"."""
    return _WS.sub(" ", _PUNCT.sub(" ", question.lower())).strip()


@dataclass
class Hit:
    layer: str             # "exact" | "semantic"
    result: object
    similarity: float = 1.0
    matched_question: str = ""


class AnswerCache:
    def __init__(self, version: dict, embed_model: str, *,
                 threshold: float = SEMANTIC_THRESHOLD, semantic: bool = True):
        # The pipeline config is part of every key: a prompt or retriever
        # change must never serve answers produced by the old pipeline.
        self.version = hashlib.sha256(
            json.dumps(version, sort_keys=True).encode()).hexdigest()[:12]
        self.embed_model = embed_model
        self.threshold = threshold
        self.semantic_enabled = semantic
        self._exact: dict[str, object] = {}
        self._vecs: list[np.ndarray] = []
        self._entries: list[tuple[str, int, object]] = []   # (question, top_k, result)
        self._lock = threading.Lock()
        self.stats = {"exact": 0, "semantic": 0, "miss": 0}

    def _key(self, question: str, top_k: int) -> str:
        blob = f"{self.version}|{top_k}|{normalise(question)}"
        return hashlib.sha256(blob.encode()).hexdigest()

    def _vec(self, question: str) -> np.ndarray:
        v = np.asarray(embed(question, model=self.embed_model, input_type="query"),
                       dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def get(self, question: str, top_k: int) -> Hit | None:
        with tracing.trace("cache.exact") as span:
            with self._lock:
                res = self._exact.get(self._key(question, top_k))
            span["hit"] = res is not None
        if res is not None:
            self._count("exact")
            return Hit("exact", res)
        if not self.semantic_enabled:
            self._count("miss")
            return None

        with tracing.trace("cache.semantic", threshold=self.threshold) as span:
            with self._lock:
                vecs, entries = list(self._vecs), list(self._entries)
            best, best_i = -1.0, -1
            if vecs:
                q = self._vec(question)
                sims = np.stack(vecs) @ q
                # Only entries answered with the same top_k are comparable.
                for i in np.argsort(-sims):
                    if entries[i][1] == top_k:
                        best, best_i = float(sims[i]), int(i)
                        break
            span.update(best_similarity=round(best, 4), hit=best >= self.threshold)
        if best_i >= 0 and best >= self.threshold:
            self._count("semantic")
            return Hit("semantic", entries[best_i][2], best, entries[best_i][0])
        self._count("miss")
        return None

    def put(self, question: str, top_k: int, result: object) -> None:
        key = self._key(question, top_k)
        vec = self._vec(question) if self.semantic_enabled else None
        with self._lock:
            if key in self._exact:
                return
            self._exact[key] = result
            if vec is not None:
                self._vecs.append(vec)
                self._entries.append((question, top_k, result))

    def _count(self, layer: str) -> None:
        with self._lock:
            self.stats[layer] += 1

    def hit_rates(self) -> dict:
        with self._lock:
            n = sum(self.stats.values())
            return {**self.stats, "lookups": n,
                    "exact_rate": round(self.stats["exact"] / n, 4) if n else 0.0,
                    "semantic_rate": round(self.stats["semantic"] / n, 4) if n else 0.0,
                    "threshold": self.threshold, "entries": len(self._exact)}
