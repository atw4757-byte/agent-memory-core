"""LangChain adapter for agent-memory-core.

Exposes a `BaseMemory`-compatible class that uses `MemoryStore` underneath.
Drop it into any LangChain agent that accepts a memory object.

    from langchain.agents import AgentExecutor, initialize_agent
    from agent_memory_core.integrations.langchain import AgentMemoryStore

    memory = AgentMemoryStore()
    agent = initialize_agent(tools, llm, memory=memory, ...)

Underlying `MemoryStore` is available as `memory.store` for advanced use
(typed chunks, namespaces, consolidation, evals).
"""

from __future__ import annotations

from typing import Any

from ..store import MemoryStore


try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover — optional dependency
    BaseChatMemory = object  # type: ignore[misc,assignment]
    BaseMessage = Any  # type: ignore[misc,assignment]
    _LANGCHAIN_AVAILABLE = False


class AgentMemoryStore(BaseChatMemory):  # type: ignore[misc]
    """LangChain BaseChatMemory backed by agent-memory-core.

    Stores input/output turns as `session` chunks. On load, retrieves the
    top-k most relevant chunks for the current input and injects them into
    the chat history.

    Parameters
    ----------
    store : MemoryStore, optional
        An existing store to reuse. If None, a fresh local store is created.
    k : int, default 5
        Number of chunks to retrieve per turn.
    memory_key : str, default "history"
        Key under which retrieved history is injected into the chain input.
    input_key : str, default "input"
        Key of the user's input in the chain's input dict.
    output_key : str, default "output"
        Key of the chain's output to persist.
    agent : str, optional
        Agent namespace. Chunks written by this adapter are tagged with this
        namespace so multiple agents can share a store without cross-talk.
    """

    store: MemoryStore
    k: int = 5
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    agent: str | None = None

    def __init__(
        self,
        store: MemoryStore | None = None,
        k: int = 5,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        agent: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain is not installed. Install with: "
                'pip install "agent-memory-core[langchain]"'
            )
        super().__init__(**kwargs)
        self.store = store if store is not None else MemoryStore()
        self.k = k
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.agent = agent

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: ""}

        results = self.store.search(query, n=self.k, agent=self.agent)
        formatted = "\n".join(f"- {r.text}" for r in results)
        return {self.memory_key: formatted}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> None:
        user = inputs.get(self.input_key, "")
        agent_reply = outputs.get(self.output_key, "")
        if user:
            self.store.add(
                f"User: {user}",
                type="session",
                source="langchain",
                agent=self.agent,
            )
        if agent_reply:
            self.store.add(
                f"Agent: {agent_reply}",
                type="session",
                source="langchain",
                agent=self.agent,
            )

    def clear(self) -> None:
        """Clear only this agent's namespace, not the whole store."""
        if self.agent:
            self.store.delete_by_agent(self.agent)
        else:
            self.store.delete_by_source("langchain")
