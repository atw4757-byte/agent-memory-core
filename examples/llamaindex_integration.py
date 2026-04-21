"""
LlamaIndex integration example — use archon-memory-core as a custom memory store
inside a LlamaIndex ReActAgent.

This wraps MemoryStore in a LlamaIndex-compatible ChatMemoryBuffer interface.
The agent writes observations and tool results to MemoryStore, and retrieves
relevant context at the start of each query.

Requirements:
    pip install archon-memory-core llama-index-core llama-index-llms-anthropic

Usage:
    python examples/llamaindex_integration.py
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# ArchonMemoryBuffer — wraps MemoryStore as a LlamaIndex memory component
# ---------------------------------------------------------------------------

class ArchonMemoryBuffer:
    """Drop-in replacement for LlamaIndex's ChatMemoryBuffer backed by MemoryStore.

    Stores each message as a typed memory chunk and retrieves the top-N most
    relevant past messages as context for the next query.

    Parameters
    ----------
    store:          Configured MemoryStore instance.
    retrieval_n:    How many memories to inject per query. Default: 5.
    default_type:   Chunk type for stored messages. Default: "session".
    """

    def __init__(self, store: Any, retrieval_n: int = 5, default_type: str = "session") -> None:
        self._store = store
        self._n = retrieval_n
        self._default_type = default_type
        self._recent: list[dict] = []  # in-session message buffer

    def put(self, message: Any) -> None:
        """Store a message (LlamaIndex ChatMessage or plain string)."""
        if hasattr(message, "content"):
            text = message.content
            role = getattr(message, "role", "user")
        else:
            text = str(message)
            role = "user"

        self._store.add(text, type=self._default_type, source=f"llamaindex/{role}")
        self._recent.append({"role": str(role), "text": text})

    def get(self, input: Optional[str] = None) -> list[dict]:
        """Retrieve relevant past messages for an input query.

        Returns a list of dicts with keys: role, text.
        Falls back to the last 5 in-session messages if no query is given.
        """
        if not input:
            return self._recent[-5:]

        results = self._store.search(input, n=self._n)
        retrieved = [{"role": "memory", "text": r.text} for r in results]
        # Append recent in-session messages for continuity
        recent = self._recent[-3:] if self._recent else []
        return retrieved + [m for m in recent if m not in retrieved]

    def reset(self) -> None:
        """Clear in-session message buffer (does not delete stored memories)."""
        self._recent.clear()


# ---------------------------------------------------------------------------
# Demo: wire it up with a LlamaIndex agent
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from llama_index.core.agent import ReActAgent
        from llama_index.core.tools import FunctionTool
        from llama_index.llms.anthropic import Anthropic
    except ImportError:
        print("Install LlamaIndex to run this example:")
        print("  pip install llama-index-core llama-index-llms-anthropic")
        return

    from archon_memory_core import MemoryStore

    store = MemoryStore()
    memory = ArchonMemoryBuffer(store, retrieval_n=5)

    # Seed a few memories
    store.add("The database password is stored in AWS Secrets Manager", type="credential")
    store.add("The project deadline is April 30, 2026", type="task")
    store.add("Use Python 3.12 for all new services", type="technical")

    # Example tool
    def get_memory(query: str) -> str:
        """Search agent memory for relevant context."""
        results = store.search(query, n=3)
        if not results:
            return "No relevant memory found."
        return "\n".join(f"- {r.text}" for r in results)

    llm = Anthropic(model="claude-sonnet-4-6")
    tools = [FunctionTool.from_defaults(fn=get_memory)]
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

    # Inject retrieved memory as context prefix
    query = "Where is the database password?"
    context = memory.get(query)
    context_str = "\n".join(f"[MEMORY] {m['text']}" for m in context)

    response = agent.chat(f"{context_str}\n\nUser: {query}")
    print(f"\nAgent: {response}")

    # Store the exchange
    memory.put(type("Msg", (), {"role": "user", "content": query})())
    memory.put(type("Msg", (), {"role": "assistant", "content": str(response)})())


if __name__ == "__main__":
    main()
