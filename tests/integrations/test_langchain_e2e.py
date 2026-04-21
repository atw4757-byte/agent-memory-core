"""End-to-end tests for the LangChain adapter (BaseChatMessageHistory).

Skipped automatically when langchain-core is not installed so CI stays green
regardless of which extras are present.

These tests exercise the full save → retrieve → clear cycle against an
in-memory/tmp-path MemoryStore — no real Ollama or ChromaDB on disk required.
"""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip(
    "langchain_core",
    reason="langchain-core not installed — skip LangChain e2e tests",
)


from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402
from archon_memory_core.store import MemoryStore  # noqa: E402
from archon_memory_core.integrations.langchain import AgentMemoryStore  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    """AgentMemoryStore backed by a fresh tmp-dir MemoryStore."""
    store = MemoryStore(db_path=str(tmp_path / "db"))
    return AgentMemoryStore(store=store, agent="test-session")


def test_instantiation(tmp_path):
    """Adapter constructs without error given an explicit store."""
    store = MemoryStore(db_path=str(tmp_path / "db"))
    m = AgentMemoryStore(store=store)
    assert m is not None


def test_add_messages_and_retrieve(mem):
    """save 3 conversations, confirm messages returns them in insertion order."""
    mem.add_messages([HumanMessage(content="What is 2+2?")])
    mem.add_messages([AIMessage(content="4")])
    mem.add_messages([HumanMessage(content="What is the capital of France?")])
    mem.add_messages([AIMessage(content="Paris")])
    mem.add_messages([HumanMessage(content="Who wrote Hamlet?")])
    mem.add_messages([AIMessage(content="Shakespeare")])

    msgs = mem.messages
    assert len(msgs) == 6

    # Ordering preserved
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "What is 2+2?"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "4"
    assert isinstance(msgs[-1], AIMessage)
    assert msgs[-1].content == "Shakespeare"


def test_messages_returns_correct_types(mem):
    """Each message comes back as the right LangChain message type."""
    mem.add_messages([HumanMessage(content="hi")])
    mem.add_messages([AIMessage(content="hello")])

    msgs = mem.messages
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)


def test_clear_empties_state(mem):
    """clear() removes all messages from in-memory buffer and backing store."""
    mem.add_messages([HumanMessage(content="remember this")])
    mem.add_messages([AIMessage(content="ok")])
    assert len(mem.messages) == 2

    mem.clear()
    assert len(mem.messages) == 0


def test_memory_key_attribute(tmp_path):
    """memory_key is preserved on the instance for legacy compatibility."""
    store = MemoryStore(db_path=str(tmp_path / "db"))
    m = AgentMemoryStore(store=store, memory_key="chat_history")
    assert m.memory_key == "chat_history"


def test_store_accessible(mem):
    """The underlying MemoryStore is exposed as mem.store."""
    assert isinstance(mem.store, MemoryStore)


def test_clear_does_not_affect_other_agents(tmp_path):
    """Clearing one agent's history does not delete another agent's chunks."""
    shared_store = MemoryStore(db_path=str(tmp_path / "db"))
    mem_a = AgentMemoryStore(store=shared_store, agent="agent-a")
    mem_b = AgentMemoryStore(store=shared_store, agent="agent-b")

    mem_a.add_messages([HumanMessage(content="agent a message")])
    mem_b.add_messages([HumanMessage(content="agent b message")])

    mem_a.clear()
    # mem_a is clear
    assert len(mem_a.messages) == 0
    # mem_b is untouched (its in-memory buffer)
    assert len(mem_b.messages) == 1
