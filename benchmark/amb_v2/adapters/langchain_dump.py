"""langchain-dump — honest ConversationTokenBufferMemory simulation.

The prior `langchain-buffer` adapter stored chunks but used word-overlap top-5
at query time — byte-identical to the naive baseline (confirmed at regime M:
stock AUCs match naive exactly per-seed). That's not what LangChain's buffer
memory actually does, and it can't fail the way the user's intuition demands
(sessions > context window → accuracy must collapse).

This adapter fixes that. It stores every chunk, and at query time returns the
newest chunks concatenated up to a token budget. Oldest chunks fall out of
the window first — FIFO eviction, exactly like ConversationBufferMemory does
when paired with a finite LLM context.

Stock mode: 8k-token budget (typical small-model context).
Tuned mode: 32k-token budget (generous frontier-ish context).

Neither mode "manages memory" in any smart sense. The ONLY difference is
budget size. If bigger context alone solved working-memory, tuned should
dominate stock by the difference. If the gap collapses at scale, memory
management (not context size) is what matters — which is the v2.2 thesis.

Tokens are approximated at 4 chars/token (OpenAI tiktoken default for English).
"""
from __future__ import annotations

from benchmark.amb_v2.adapters.base import Mode
from benchmark.amb_v2.chunks import Chunk

CHARS_PER_TOKEN = 4
STOCK_BUDGET_TOKENS = 8_000
TUNED_BUDGET_TOKENS = 32_000


class LangChainDumpAdapter:
    """ConversationTokenBufferMemory simulation — FIFO eviction at token budget."""

    def __init__(self, mode: Mode = "stock") -> None:
        self.mode: Mode = mode
        self._chunks: list[Chunk] = []
        budget_tokens = TUNED_BUDGET_TOKENS if mode == "tuned" else STOCK_BUDGET_TOKENS
        self._budget_chars = budget_tokens * CHARS_PER_TOKEN

    def ingest(self, day: int, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def consolidate(self, day: int) -> None:
        return

    def query(self, question: str, scenario_id: str) -> str:
        scoped = [c for c in self._chunks if c.scenario_id == scenario_id]
        if not scoped:
            return ""

        scoped.sort(key=lambda c: c.day, reverse=True)
        kept: list[str] = []
        used = 0
        sep_len = 3
        for c in scoped:
            cost = len(c.text) + (sep_len if kept else 0)
            if used + cost > self._budget_chars:
                break
            kept.append(c.text)
            used += cost
        return " | ".join(kept)

    @property
    def metadata(self) -> dict:
        budget_tokens = TUNED_BUDGET_TOKENS if self.mode == "tuned" else STOCK_BUDGET_TOKENS
        return {
            "name": "langchain-dump",
            "version": "v1.0",
            "implements_consolidation": False,
            "mode": self.mode,
            "budget_tokens": budget_tokens,
        }
