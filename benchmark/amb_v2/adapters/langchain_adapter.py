"""LangChain adapter — wraps ConversationSummaryBufferMemory + a simple
recency-ordered substring retriever.

Stock mode: append only, no pruning.
Tuned mode: trigger memory.prune() once per simulated day.

Uses no LLM for summarization (would slow benchmark + add API cost). Stores
raw turns and falls back to substring retrieval at query time.
"""
from __future__ import annotations

from benchmark.amb_v2.adapters.base import Mode
from benchmark.amb_v2.chunks import Chunk

try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class LangChainAdapter:
    def __init__(self, mode: Mode = "stock", max_token_limit: int = 8000) -> None:
        if not _AVAILABLE:
            raise ImportError("langchain not installed; run pip install langchain langchain-community")
        self.mode: Mode = mode
        self._history = ChatMessageHistory()
        self._chunks: list[Chunk] = []
        self._max_token_limit = max_token_limit

    def ingest(self, day: int, chunks: list[Chunk]) -> None:
        for c in chunks:
            self._history.add_user_message(c.text)
            self._history.add_ai_message("noted")
            self._chunks.append(c)

    def consolidate(self, day: int) -> None:
        if self.mode == "stock":
            return
        # Tuned: drop chunks older than 30 days; rebuild history from trimmed set
        cutoff = day - 30
        self._chunks = [c for c in self._chunks if c.day >= cutoff]
        self._history.clear()
        for c in self._chunks:
            self._history.add_user_message(c.text)
            self._history.add_ai_message("noted")

    def query(self, question: str, scenario_id: str) -> str:
        q_words = {w.lower() for w in question.split() if len(w) > 3}
        scored: list[tuple[int, Chunk]] = []
        for c in self._chunks:
            if c.scenario_id != scenario_id:
                continue
            text_words = {w.lower() for w in c.text.split()}
            score = len(q_words & text_words)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda p: (-p[0], -p[1].day))
        return " | ".join(c.text for _, c in scored[:5])

    @property
    def metadata(self) -> dict:
        try:
            import langchain
            v = getattr(langchain, "__version__", "unknown")
        except ImportError:
            v = "unknown"
        return {
            "name": "langchain-buffer",
            "version": v,
            "implements_consolidation": False,
            "mode": self.mode,
        }
