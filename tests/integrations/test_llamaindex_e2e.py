"""End-to-end tests for the LlamaIndex adapter (BaseMemory).

Skipped automatically when llama-index-core is not installed so CI stays green
regardless of which extras are present.

These tests exercise the full put → get → get_all → reset cycle against an
in-memory/tmp-path MemoryStore — no real Ollama or ChromaDB on disk required.
"""

from __future__ import annotations

import pytest

llama_index_core = pytest.importorskip(
    "llama_index.core",
    reason="llama-index-core not installed — skip LlamaIndex e2e tests",
)


from llama_index.core.llms import ChatMessage, MessageRole  # noqa: E402
from archon_memory_core.store import MemoryStore  # noqa: E402
from archon_memory_core.integrations.llamaindex import AgentMemoryStore  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    """AgentMemoryStore backed by a fresh tmp-dir MemoryStore."""
    store = MemoryStore(db_path=str(tmp_path / "db"))
    return AgentMemoryStore.from_defaults(store=store, agent="test-session")


def test_from_defaults_instantiation(tmp_path):
    """from_defaults() creates an instance without error."""
    store = MemoryStore(db_path=str(tmp_path / "db"))
    m = AgentMemoryStore.from_defaults(store=store)
    assert m is not None


def test_direct_instantiation(tmp_path):
    """Direct constructor also works (needed for Pydantic model compat)."""
    store = MemoryStore(db_path=str(tmp_path / "db"))
    m = AgentMemoryStore(store=store, k=3, agent="direct")
    assert m.k == 3


def test_put_and_get_all(mem):
    """put() stores messages; get_all() retrieves them as ChatMessage objects."""
    mem.put(ChatMessage(role=MessageRole.USER, content="Hello"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))
    mem.put(ChatMessage(role=MessageRole.USER, content="What is Python?"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="A programming language"))

    all_msgs = mem.get_all()
    assert len(all_msgs) == 4
    # All results are ChatMessage instances
    assert all(isinstance(m, ChatMessage) for m in all_msgs)


def test_get_returns_chat_messages(mem):
    """get(input=...) returns a list of ChatMessage objects."""
    mem.put(ChatMessage(role=MessageRole.USER, content="What is Python?"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="A programming language"))

    results = mem.get(input="Python")
    assert isinstance(results, list)
    assert all(isinstance(m, ChatMessage) for m in results)
    # At least one result mentioning Python
    contents = [m.content for m in results]
    assert any("Python" in c for c in contents)


def test_role_roundtrip(mem):
    """USER and ASSISTANT roles survive the put → get_all cycle."""
    mem.put(ChatMessage(role=MessageRole.USER, content="user msg"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="assistant msg"))

    all_msgs = mem.get_all()
    roles = {m.role for m in all_msgs}
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT in roles


def test_reset_clears_state(mem):
    """reset() removes all messages from the backing store."""
    mem.put(ChatMessage(role=MessageRole.USER, content="remember this"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="noted"))

    before = mem.get_all()
    assert len(before) == 2

    mem.reset()
    after = mem.get_all()
    assert len(after) == 0


def test_set_replaces_history(mem):
    """set() clears existing history and inserts the given messages."""
    mem.put(ChatMessage(role=MessageRole.USER, content="old message"))
    mem.put(ChatMessage(role=MessageRole.ASSISTANT, content="old reply"))

    new_messages = [
        ChatMessage(role=MessageRole.USER, content="fresh start"),
        ChatMessage(role=MessageRole.ASSISTANT, content="fresh reply"),
    ]
    mem.set(new_messages)

    all_msgs = mem.get_all()
    contents = [m.content for m in all_msgs]
    assert len(all_msgs) == 2
    assert "fresh start" in contents[0] or "fresh start" in contents[1]


def test_class_name(mem):
    """class_name() returns the expected string for LlamaIndex serialization."""
    assert AgentMemoryStore.class_name() == "AgentMemoryStore"


def test_store_accessible(mem):
    """The underlying MemoryStore is exposed as mem.store."""
    assert isinstance(mem.store, MemoryStore)


def test_reset_scoped_to_agent(tmp_path):
    """reset() only clears the current agent's chunks, not another agent's."""
    shared_store = MemoryStore(db_path=str(tmp_path / "db"))
    mem_a = AgentMemoryStore.from_defaults(store=shared_store, agent="agent-a")
    mem_b = AgentMemoryStore.from_defaults(store=shared_store, agent="agent-b")

    mem_a.put(ChatMessage(role=MessageRole.USER, content="agent a says hi"))
    mem_b.put(ChatMessage(role=MessageRole.USER, content="agent b says hi"))

    mem_a.reset()

    # agent-a is gone
    assert len(mem_a.get_all()) == 0
    # agent-b is untouched
    assert len(mem_b.get_all()) == 1
