"""Framework adapters for agent-memory-core.

Import the adapter for the framework you're using. Adapters are thin — they
delegate to `MemoryStore` for all storage and retrieval. You never lose access
to the underlying store; you can always reach it via `adapter.store`.

    from agent_memory_core.integrations.langchain import AgentMemoryStore
    from agent_memory_core.integrations.llamaindex import AgentMemoryStore

Adapters are intentionally minimal. If you need advanced features (namespacing,
typed chunks, eval runs), instantiate `MemoryStore` directly and pass it into
the adapter constructor.
"""

from __future__ import annotations

__all__ = ["langchain", "llamaindex"]
