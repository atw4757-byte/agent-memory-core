"""Naive baseline — append-only in-memory list, BM25-ish substring retrieval.

Zero external deps. Demonstrates the worst-case curve. Used as the lower bound
on every chart.
"""
from __future__ import annotations

from typing import Literal

from benchmark.amb_v2.adapters.base import Mode
from benchmark.amb_v2.chunks import Chunk


class NaiveAppendOnlyAdapter:
    """Stores every chunk; query returns the top-5 most-substring-matching by score."""

    def __init__(self, mode: Mode = "stock") -> None:
        self.mode: Mode = mode
        self._chunks: list[Chunk] = []

    def ingest(self, day: int, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def consolidate(self, day: int) -> None:
        # Naive baseline does nothing in either mode (no consolidation).
        return

    def query(self, question: str, scenario_id: str) -> str:
        if not self._chunks:
            return ""
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        scored: list[tuple[int, Chunk]] = []
        for c in self._chunks:
            if c.scenario_id != scenario_id:
                continue
            text_words = {w.lower() for w in c.text.split()}
            score = len(q_words & text_words)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda p: (-p[0], p[1].day))
        top = scored[:5]
        return " | ".join(c.text for _, c in top)

    @property
    def metadata(self) -> dict:
        return {
            "name": "naive-append-only",
            "version": "v2.0",
            "implements_consolidation": False,
            "mode": self.mode,
        }
