"""Smoke tests for framework adapters.

Each adapter module is importable and raises a friendly ImportError with
install instructions when the framework isn't installed. These tests do not
require LangChain or LlamaIndex — they verify the import shim behaves.
"""

from __future__ import annotations

import importlib

import pytest


def test_integrations_package_importable() -> None:
    mod = importlib.import_module("archon_memory_core.integrations")
    assert hasattr(mod, "__all__")


def test_langchain_adapter_import_shim() -> None:
    """If langchain is missing, constructor raises a helpful ImportError."""
    from archon_memory_core.integrations import langchain as lc

    if not lc._LANGCHAIN_AVAILABLE:
        with pytest.raises(ImportError, match="langchain"):
            lc.AgentMemoryStore()


def test_llamaindex_adapter_import_shim() -> None:
    """If llama_index is missing, constructor raises a helpful ImportError."""
    from archon_memory_core.integrations import llamaindex as li

    if not li._LLAMAINDEX_AVAILABLE:
        with pytest.raises(ImportError, match="llama"):
            li.AgentMemoryStore()
