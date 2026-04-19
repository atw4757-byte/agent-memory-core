"""Tests for langchain-dump — the honest ConversationTokenBufferMemory.

These tests enforce the adapter's contract:
  1. Stores every ingested chunk (no silent drops)
  2. At query time, returns newest-first chunks up to a token budget
  3. FIFO eviction: oldest chunks drop out once budget is exceeded
  4. Stock has smaller budget than tuned
  5. Scenario scoping: only returns chunks from the queried scenario
  6. No smart retrieval — this is the honest baseline that MUST fail at scale
"""
from __future__ import annotations

import pytest

from benchmark.amb_v2.adapters.base import validate_metadata
from benchmark.amb_v2.adapters.langchain_dump import (
    CHARS_PER_TOKEN,
    STOCK_BUDGET_TOKENS,
    TUNED_BUDGET_TOKENS,
    LangChainDumpAdapter,
)
from benchmark.amb_v2.chunks import Chunk


def _chunk(cid: str, scenario_id: str, day: int, text: str) -> Chunk:
    return Chunk(id=cid, scenario_id=scenario_id, day=day, text=text, type="fact")


def test_metadata_valid():
    a = LangChainDumpAdapter(mode="stock")
    validate_metadata(a.metadata)
    assert a.metadata["name"] == "langchain-dump"
    assert a.metadata["implements_consolidation"] is False
    assert a.metadata["budget_tokens"] == STOCK_BUDGET_TOKENS


def test_tuned_budget_larger_than_stock():
    stock = LangChainDumpAdapter(mode="stock")
    tuned = LangChainDumpAdapter(mode="tuned")
    assert tuned.metadata["budget_tokens"] > stock.metadata["budget_tokens"]
    assert tuned.metadata["budget_tokens"] == TUNED_BUDGET_TOKENS


def test_empty_store_returns_empty_string():
    a = LangChainDumpAdapter(mode="stock")
    assert a.query("anything?", "s1") == ""


def test_returns_only_scenario_scoped_chunks():
    a = LangChainDumpAdapter(mode="stock")
    a.ingest(0, [_chunk("c1", "s1", 0, "alpha fact")])
    a.ingest(0, [_chunk("c2", "s2", 0, "beta fact")])
    out = a.query("q", "s1")
    assert "alpha" in out
    assert "beta" not in out


def test_dumps_all_chunks_when_under_budget():
    a = LangChainDumpAdapter(mode="stock")
    a.ingest(0, [_chunk("c1", "s1", 0, "first"),
                 _chunk("c2", "s1", 1, "second"),
                 _chunk("c3", "s1", 2, "third")])
    out = a.query("q", "s1")
    assert "first" in out
    assert "second" in out
    assert "third" in out


def test_newest_first_ordering():
    a = LangChainDumpAdapter(mode="stock")
    a.ingest(0, [_chunk("c1", "s1", 0, "oldest")])
    a.ingest(10, [_chunk("c2", "s1", 10, "middle")])
    a.ingest(30, [_chunk("c3", "s1", 30, "newest")])
    out = a.query("q", "s1")
    assert out.index("newest") < out.index("middle") < out.index("oldest")


def test_fifo_eviction_at_budget():
    """With a budget of 8k tokens (32k chars), at ~1000 chars/chunk, we fit
    ~32 chunks. Ingesting 100 chunks should evict the oldest ~68."""
    a = LangChainDumpAdapter(mode="stock")
    big_text = "x" * 1000
    chunks = [_chunk(f"c{i:03d}", "s1", i, f"d{i:03d}-{big_text}") for i in range(100)]
    a.ingest(0, chunks)
    out = a.query("q", "s1")
    assert "d099" in out, "newest chunk must survive"
    assert "d000" not in out, "oldest chunk must be evicted at budget"


def test_tuned_fits_more_than_stock():
    """Tuned's 32k budget holds ~4x more than stock's 8k, so at the same input
    density tuned retains chunks that stock evicts."""
    text_1000 = "y" * 1000
    chunks = [_chunk(f"c{i:03d}", "s1", i, f"d{i:03d}-{text_1000}") for i in range(60)]

    stock = LangChainDumpAdapter(mode="stock")
    stock.ingest(0, chunks)
    stock_out = stock.query("q", "s1")

    tuned = LangChainDumpAdapter(mode="tuned")
    tuned.ingest(0, chunks)
    tuned_out = tuned.query("q", "s1")

    assert len(tuned_out) > len(stock_out)


def test_budget_never_exceeded():
    a = LangChainDumpAdapter(mode="stock")
    big_text = "z" * 500
    chunks = [_chunk(f"c{i:03d}", "s1", i, f"d{i:03d}-{big_text}") for i in range(500)]
    a.ingest(0, chunks)
    out = a.query("q", "s1")
    budget_chars = STOCK_BUDGET_TOKENS * CHARS_PER_TOKEN
    assert len(out) <= budget_chars


def test_consolidate_is_noop():
    """langchain-dump has no smart management — consolidate does nothing in
    either mode. Difference between modes is ONLY budget size."""
    a_stock = LangChainDumpAdapter(mode="stock")
    a_tuned = LangChainDumpAdapter(mode="tuned")
    for i in range(10):
        a_stock.ingest(i, [_chunk(f"c{i}", "s1", i, f"text {i}")])
        a_tuned.ingest(i, [_chunk(f"c{i}", "s1", i, f"text {i}")])
    before_stock = a_stock.query("q", "s1")
    before_tuned = a_tuned.query("q", "s1")
    a_stock.consolidate(10)
    a_tuned.consolidate(10)
    assert a_stock.query("q", "s1") == before_stock
    assert a_tuned.query("q", "s1") == before_tuned


def test_question_text_ignored():
    """Dump adapter returns the same content regardless of question — it has
    no retrieval logic. This is the honest 'dump into prompt' behavior."""
    a = LangChainDumpAdapter(mode="stock")
    a.ingest(0, [_chunk("c1", "s1", 0, "the sky is blue"),
                 _chunk("c2", "s1", 1, "Marcus prefers espresso")])
    ans1 = a.query("what color is the sky?", "s1")
    ans2 = a.query("what is Marcus's drink?", "s1")
    assert ans1 == ans2
