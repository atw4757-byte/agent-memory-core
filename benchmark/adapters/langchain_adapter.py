"""
benchmark/adapters/langchain_adapter.py — LangChain ConversationBufferWindowMemory.

Simulates what a typical LangChain agent does: keep the last k=10 turns in a
sliding window buffer. No vector search, no salience — just a plain rolling
window of recent conversation.

This represents the "I just use LangChain memory" crowd.

Dependencies: langchain, langchain-community (optional — skips gracefully if absent)
"""

from __future__ import annotations

_IMPORT_ERROR: Exception | None = None

try:
    # langchain >= 1.0 moved memory to langchain_classic
    try:
        from langchain_classic.memory import ConversationBufferWindowMemory
    except ImportError:
        from langchain.memory import ConversationBufferWindowMemory
    _HAS_LANGCHAIN = True
except ImportError as _e:
    _HAS_LANGCHAIN = False
    _IMPORT_ERROR = _e


class LangChainWindowAdapter:
    """LangChain ConversationBufferWindowMemory with k=10 turns.

    Stores the last 10 human/AI message pairs. Query returns the full buffer
    as context — no semantic search, pure recency.

    Raises ImportError at instantiation if langchain is not installed.
    """

    WINDOW_K = 10  # turns (each turn = 1 human + 1 AI message)

    def __init__(self) -> None:
        if not _HAS_LANGCHAIN:
            raise ImportError(
                f"langchain is not installed (langchain_adapter will be skipped). "
                f"Original error: {_IMPORT_ERROR}. "
                f"Install with: pip install langchain langchain-community"
            )
        self._memory = ConversationBufferWindowMemory(k=self.WINDOW_K, return_messages=False)
        # We accumulate turns per session_id so we can replay them into the buffer
        self._pending_human: str | None = None

    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        if role == "user":
            # Hold until we see the assistant response (standard LangChain pattern)
            self._pending_human = content
        elif role == "assistant":
            if self._pending_human is not None:
                self._memory.save_context(
                    {"input": self._pending_human},
                    {"output": content},
                )
                self._pending_human = None
            # If there's no pending human turn, skip (orphaned assistant turn)
        # Flush any orphaned human turn as a turn with empty AI response
        # so we don't lose user-only sessions

    def _flush_pending(self) -> None:
        if self._pending_human is not None:
            self._memory.save_context(
                {"input": self._pending_human},
                {"output": ""},
            )
            self._pending_human = None

    def query(self, question: str) -> str:
        # Flush any unmatched final user turn
        self._flush_pending()
        history = self._memory.load_memory_variables({}).get("history", "")
        if not history:
            return ""
        # The question itself isn't answered by LangChain window memory —
        # it just provides the context buffer. We return the buffer as the
        # "answer" so the benchmark can score what information was available.
        return str(history)

    def reset(self) -> None:
        self._memory.clear()
        self._pending_human = None
