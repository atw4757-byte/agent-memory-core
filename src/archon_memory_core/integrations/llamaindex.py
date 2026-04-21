"""LlamaIndex adapter for archon-memory-core.

Exposes a `BaseMemory`-compatible class that uses `MemoryStore` underneath.
Drop it into any LlamaIndex agent that accepts a memory object.

    from llama_index.core.agent import ReActAgent
    from archon_memory_core.integrations.llamaindex import AgentMemoryStore

    memory = AgentMemoryStore.from_defaults()
    agent = ReActAgent.from_tools(tools, llm=llm, memory=memory, ...)

Underlying `MemoryStore` is available as `memory.store` for advanced use
(typed chunks, namespaces, consolidation, evals).

Note: `BaseMemory` (llama-index-core >=0.12) is a Pydantic v2 BaseModel
subclass.  Fields must be declared at class level; arbitrary types require
`model_config = ConfigDict(arbitrary_types_allowed=True)`.  The `__init__`
override pattern used by earlier stubs does not work with this model.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..store import MemoryStore


try:
    from llama_index.core.memory.types import BaseMemory
    from llama_index.core.llms import ChatMessage, MessageRole
    import pydantic as _pydantic

    _LLAMAINDEX_AVAILABLE = True
except ImportError:  # pragma: no cover — optional dependency
    BaseMemory = object  # type: ignore[misc,assignment]
    ChatMessage = Any  # type: ignore[misc,assignment]
    MessageRole = Any  # type: ignore[misc,assignment]
    _pydantic = None  # type: ignore[assignment]
    _LLAMAINDEX_AVAILABLE = False


if _LLAMAINDEX_AVAILABLE:
    class AgentMemoryStore(BaseMemory):  # type: ignore[misc]
        """LlamaIndex BaseMemory backed by archon-memory-core.

        Stores turns as `session` chunks. On `get`, retrieves the top-k most
        relevant chunks for the latest input and returns them as ChatMessages.

        Parameters
        ----------
        store : MemoryStore
            Backing store instance.  Use `from_defaults()` to create one
            automatically.
        k : int, default 5
            Number of chunks to retrieve per query.
        agent : str, optional
            Agent namespace.
        """

        # Pydantic v2 field declarations — BaseMemory is a pydantic BaseModel.
        store: Any = None
        k: int = 5
        agent: Optional[str] = None

        # Private state — not part of the pydantic schema.
        _recent_user_input: str = _pydantic.PrivateAttr(default="")

        model_config = _pydantic.ConfigDict(arbitrary_types_allowed=True)

        @classmethod
        def from_defaults(
            cls,
            store: Optional[MemoryStore] = None,
            k: int = 5,
            agent: Optional[str] = None,
            **kwargs: Any,
        ) -> "AgentMemoryStore":
            """Create an AgentMemoryStore with default settings."""
            if not _LLAMAINDEX_AVAILABLE:
                raise ImportError(
                    "llama-index-core is not installed. Install with: "
                    'pip install "archon-memory-core[llamaindex]"'
                )
            return cls(
                store=store if store is not None else MemoryStore(),
                k=k,
                agent=agent,
            )

        def __init__(self, **data: Any) -> None:
            if not _LLAMAINDEX_AVAILABLE:
                raise ImportError(
                    "llama-index-core is not installed. Install with: "
                    'pip install "archon-memory-core[llamaindex]"'
                )
            if "store" not in data or data["store"] is None:
                data["store"] = MemoryStore()
            super().__init__(**data)

        @classmethod
        def class_name(cls) -> str:
            return "AgentMemoryStore"

        def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
            query = input or self._recent_user_input
            if not query:
                return []

            results = self.store.search(query, n=self.k, agent=self.agent)
            messages = []
            for r in results:
                role = MessageRole.USER if r.text.startswith("User:") else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=r.text))
            return messages

        def get_all(self) -> List[ChatMessage]:
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
                agent=self.agent if self.agent is not None else "shared",
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
            self._recent_user_input = ""

else:
    # Stub class when llama-index-core is not installed.
    # The ImportError is raised at instantiation time.
    class AgentMemoryStore:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "llama-index-core is not installed. Install with: "
                'pip install "archon-memory-core[llamaindex]"'
            )
