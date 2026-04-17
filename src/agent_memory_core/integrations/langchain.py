"""LangChain adapter for agent-memory-core.

Exposes a `BaseChatMessageHistory`-compatible class (langchain_core >=0.3)
that uses `MemoryStore` underneath.  Drop it into any LangChain chain via
`RunnableWithMessageHistory` or use it directly with legacy ConversationChain.

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from agent_memory_core.integrations.langchain import AgentMemoryStore

    def get_session_history(session_id: str) -> AgentMemoryStore:
        return AgentMemoryStore(agent=session_id)

    chain_with_history = RunnableWithMessageHistory(chain, get_session_history)

Underlying `MemoryStore` is available as `memory.store` for advanced use
(typed chunks, namespaces, consolidation, evals).

Note: langchain <0.3 exposed `BaseChatMemory` / `save_context` /
`load_memory_variables`.  That API was removed upstream in 0.3.  The modern
pattern is `BaseChatMessageHistory` (messages / add_messages / clear).
"""

from __future__ import annotations

from typing import Any, List, Sequence

from ..store import MemoryStore


try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover — optional dependency
    BaseChatMessageHistory = object  # type: ignore[misc,assignment]
    BaseMessage = Any  # type: ignore[misc,assignment]
    HumanMessage = Any  # type: ignore[misc,assignment]
    AIMessage = Any  # type: ignore[misc,assignment]
    _LANGCHAIN_AVAILABLE = False


class AgentMemoryStore(BaseChatMessageHistory):  # type: ignore[misc]
    """LangChain BaseChatMessageHistory backed by agent-memory-core.

    Stores each message as a `session` chunk. `messages` retrieves them in
    insertion order (oldest first) by replaying all chunks tagged with this
    store's source/agent label.

    Parameters
    ----------
    store : MemoryStore, optional
        An existing store to reuse. If None, a fresh local store is created.
    memory_key : str, default "history"
        Kept for drop-in compatibility with older call sites that inspect this
        attribute; not used internally by BaseChatMessageHistory.
    agent : str, optional
        Agent namespace. Chunks are tagged with this so multiple agents can
        share a single MemoryStore without cross-talk.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        memory_key: str = "history",
        agent: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is not installed. Install with: "
                'pip install "agent-memory-core[langchain]"'
            )
        super().__init__(**kwargs)
        self.store = store if store is not None else MemoryStore()
        self.memory_key = memory_key
        self.agent = agent
        # In-order buffer: we replay additions so ordering is preserved.
        self._buffer: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """Return all messages in insertion order."""
        return list(self._buffer)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist a batch of messages."""
        for message in messages:
            self._buffer.append(message)
            role = "User" if isinstance(message, HumanMessage) else "Agent"
            self.store.add(
                f"{role}: {message.content}",
                type="session",
                source="langchain",
                agent=self.agent if self.agent is not None else "shared",
            )

    def clear(self) -> None:
        """Clear this store's messages (scoped to agent namespace or source)."""
        self._buffer.clear()
        if self.agent:
            self.store.delete_by_agent(self.agent)
        else:
            self.store.delete_by_source("langchain")
