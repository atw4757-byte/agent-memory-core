"""
Tests for supersedes-aware consolidation and retrieval (v1.x → v2.1 feature).

v2.1 feature: MemoryStore.consolidate() detects supersedes metadata and marks
older chunks as superseded_by their replacements. search() then excludes
superseded chunks by default, so contradictions resolve to the latest value.

This is deterministic (metadata-driven, no LLM) — distinct from the existing
Consolidator class which uses Ollama for lossy compression.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_store(tmp_path):
    from agent_memory_core import MemoryStore
    return MemoryStore(
        db_path=tmp_path / "vectordb",
        collection_name="test-supersedes",
        facts_collection_name="test-supersedes-facts",
        hindsight_url=None,
    )


class TestConsolidateSupersedes:
    def test_consolidate_exists(self, tmp_store):
        """MemoryStore.consolidate() must exist and be callable."""
        assert callable(getattr(tmp_store, "consolidate", None))

    def test_consolidate_empty_store_is_noop(self, tmp_store):
        """Running on an empty store returns cleanly, no error."""
        report = tmp_store.consolidate()
        assert report is None or isinstance(report, dict)

    def test_consolidate_marks_superseded(self, tmp_store):
        """When chunk B supersedes chunk A, A should be marked superseded_by=B."""
        # Add A with a scenario-level id in extra_metadata
        a_id = tmp_store.add(
            "User lives in Seattle.",
            type="personal",
            source="sup-test",
            extra_metadata={"scenario_chunk_id": "ev-A"},
        )
        # Add B which supersedes A
        b_id = tmp_store.add(
            "User moved to Portland.",
            type="personal",
            source="sup-test",
            extra_metadata={"scenario_chunk_id": "ev-B", "supersedes": "ev-A"},
        )

        tmp_store.consolidate()

        all_chunks = tmp_store.get_all(include_archived=True)
        by_id = {c["id"]: c for c in all_chunks}
        assert a_id in by_id and b_id in by_id
        assert by_id[a_id]["metadata"].get("superseded_by") == b_id
        # B should NOT be marked superseded
        assert not by_id[b_id]["metadata"].get("superseded_by")

    def test_consolidate_is_idempotent(self, tmp_store):
        """Running consolidate twice produces the same state."""
        tmp_store.add("Old value.", type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "x1"})
        tmp_store.add("New value.", type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "x2", "supersedes": "x1"})
        tmp_store.consolidate()
        state1 = {c["id"]: c["metadata"].get("superseded_by")
                  for c in tmp_store.get_all(include_archived=True)}
        tmp_store.consolidate()
        state2 = {c["id"]: c["metadata"].get("superseded_by")
                  for c in tmp_store.get_all(include_archived=True)}
        assert state1 == state2

    def test_consolidate_handles_missing_target(self, tmp_store):
        """If supersedes points to a chunk that isn't in the store, no crash."""
        tmp_store.add("Dangling reference.", type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "y1",
                                      "supersedes": "nonexistent-id"})
        tmp_store.consolidate()  # must not raise

    def test_consolidate_dedupes_double_contradiction(self, tmp_store):
        """Two chunks superseding the same older chunk must NOT crash on
        ChromaDB's duplicate-id validation. Latest superseder wins."""
        a_id = tmp_store.add("Original value.", type="fact", source="t",
                             extra_metadata={"scenario_chunk_id": "a"})
        b_id = tmp_store.add("First replacement.", type="fact", source="t",
                             extra_metadata={"scenario_chunk_id": "b", "supersedes": "a"})
        c_id = tmp_store.add("Second replacement.", type="fact", source="t",
                             extra_metadata={"scenario_chunk_id": "c", "supersedes": "a"})
        tmp_store.consolidate()  # must not raise
        all_chunks = tmp_store.get_all(include_archived=True)
        by_id = {c["id"]: c for c in all_chunks}
        # A got marked as superseded; newest superseder (b or c) wins — both
        # are acceptable, but it must be one of them (not crash, not missing).
        assert by_id[a_id]["metadata"].get("superseded_by") in (b_id, c_id)


class TestSearchExcludesSuperseded:
    def test_search_skips_superseded_by_default(self, tmp_store):
        """A chunk with superseded_by=X should be filtered out of search results."""
        tmp_store.add("Old fact about Project Alpha.", type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "old-1"})
        tmp_store.add("New fact about Project Alpha replaces the old one.",
                      type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "new-1", "supersedes": "old-1"})
        tmp_store.consolidate()

        results = tmp_store.search("Project Alpha", n=5)
        texts = [r.text for r in results]
        # New fact should be present; old fact should be filtered
        assert any("New fact" in t for t in texts)
        assert not any("Old fact about Project Alpha" in t for t in texts)

    def test_search_include_superseded_returns_both(self, tmp_store):
        """Passing include_superseded=True restores the old chunk to results."""
        tmp_store.add("Alpha old value.", type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "old-2"})
        tmp_store.add("Alpha new value supersedes old.", type="fact", source="t",
                      extra_metadata={"scenario_chunk_id": "new-2", "supersedes": "old-2"})
        tmp_store.consolidate()

        results = tmp_store.search("Alpha value", n=5, include_superseded=True)
        texts = [r.text for r in results]
        assert any("old value" in t for t in texts)
        assert any("new value" in t for t in texts)
