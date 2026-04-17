"""LlamaIndex adapter for agent-memory-core.

Exposes a `BaseMemory`-compatible class that uses `MemoryStore` underneath.
Drop it into any LlamaIndex agent that accepts a memory object.

    from llama_index.core.agent import ReActAgent
    from agent_memory_core.integrations.llamaindex import AgentMemoryStore

    memory = AgentMemoryStore()
    agent = ReActAgent.from_tools(tools, llm=llm, memory=memory, ...)

Underlying `MemoryStore` is available as `memory.store` for advanced use
(typed chunks, namespaces, consolidation, evals).
"""

from __future__ import annotations

from typing import Any, List

from ..store import MemoryStore


try:
    from llama_index.core.memory.types import BaseMemory
    from llama_index.core.llms import ChatMessage, MessageRole

    _LLAMAINDEX_AVAILABLE = True
except ImportError:  # pragma: no cover — optional dependency
    BaseMemory = object  # type: ignore[misc,assignment]
    ChatMessage = Any  # type: ignore[misc,assignment]
    MessageRole = Any  # type: ignore[misc,assignment]
    _LLAMAINDEX_AVAILABLE = False


class AgentMemoryStore(BaseMemory):  # type: ignore[misc]
    """LlamaIndex BaseMemory backed by agent-memory-core.

    Stores turns as `session` chunks. On `get`, retrieves the top-k most
    relevant chunks for the latest input and returns them as ChatMessages.

    Parameters
    ----------
    store : MemoryStore, optional
        An existing store to reuse. If None, a fresh local store is created.
    k : int, default 5
        Number of chunks to retrieve per query.
    agent : str, optional
        Agent namespace.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        k: int = 5,
        agent: str | None = None,
    ) -> None:
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "llama-index is not installed. Install with: "
                'pip install "agent-memory-core[llamaindex]"'
            )
        super().__init__()
        self.store = store if store is not None else MemoryStore()
        self.k = k
        self.agent = agent
        self._recent_user_input: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "AgentMemoryStore"

    def get(self, input: str | None = None, **kwargs: Any) -> List[Any]:
        query = input or self._recent_user_input
        if not query:
            return []

        results = self.store.search(query, n=self.k, agent=self.agent)
        messages = []
        for r in results:
            role = MessageRole.USER if r.text.startswith("User:") else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=r.text))
        return messages

    def get_all(self) -> List[Any]:
        results = self.store.search("", n=1000, agent=self.agent)
        return [
            ChatMessage(
                role=MessageRole.USER if r.text.startswith("User:") else MessageRole.ASSISTANT,
                content=r.text,
            )
            for r in results
        ]

    def put(self, message: Any) -> None:
        text = getattr(message, "content", str(message))
        role = getattr(message, "role", None)
        prefix = "User: " if role == MessageRole.USER else "Agent: "
        if text.startswith("User:") or text.startswith("Agent:"):
            prefix = ""
        self.store.add(
            f"{prefix}{text}",
            type="session",
            source="llamaindex",
            agent=self.agent,
        )
        if role == MessageRole.USER:
            self._recent_user_input = text

    def set(self, messages: List[Any]) -> None:
        self.reset()
        for m in messages:
            self.put(m)

    def reset(self) -> None:
        if self.agent:
            self.store.delete_by_agent(self.agent)
        else:
            self.store.delete_by_source("llamaindex")
