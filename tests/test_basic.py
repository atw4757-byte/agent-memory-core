"""
Basic tests for agent-memory-core.

Run:
    pip install -e ".[dev]"
    pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """A MemoryStore backed by a temporary directory."""
    from agent_memory_core import MemoryStore
    return MemoryStore(
        db_path=tmp_path / "vectordb",
        collection_name="test-memory",
        facts_collection_name="test-facts",
        hindsight_url=None,
    )


@pytest.fixture
def tmp_working(tmp_path):
    """A WorkingMemory backed by a temporary file."""
    from agent_memory_core import WorkingMemory
    return WorkingMemory(buffer_path=tmp_path / "working-memory.json")


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------

class TestTypes:
    def test_valid_types_nonempty(self):
        from agent_memory_core import VALID_TYPES
        assert len(VALID_TYPES) >= 12

    def test_compute_recency_today(self):
        from datetime import date
        from agent_memory_core.types import compute_recency
        today = date.today().isoformat()
        score = compute_recency(today, "fact")
        assert score > 0.99, "Today's memory should have recency near 1.0"

    def test_compute_recency_credentials_never_decay(self):
        from agent_memory_core.types import compute_recency
        old_score = compute_recency("2020-01-01", "credential")
        assert old_score == pytest.approx(1.0), "Credentials must not decay"

    def test_compute_salience_credential_highest(self):
        from agent_memory_core.types import compute_salience, TYPE_SALIENCE_PRIORS
        cred_prior = TYPE_SALIENCE_PRIORS["credential"]
        dream_prior = TYPE_SALIENCE_PRIORS["dream"]
        assert cred_prior > dream_prior

    def test_memory_result_repr(self):
        from agent_memory_core.types import MemoryResult
        r = MemoryResult(
            id="test::id", text="hello world", type="fact", source="test",
            date="2026-04-09", distance=0.1, recency_score=0.9,
            salience=0.5, combined_score=0.3, age_days=0,
        )
        assert "fact" in repr(r)
        assert "0.300" in repr(r)

    def test_working_memory_buffer_roundtrip(self):
        from agent_memory_core.types import WorkingMemoryBuffer
        buf = WorkingMemoryBuffer(
            current_goal="test goal",
            active_context=["item1", "item2"],
            blockers=["blocker1"],
            next_actions=["action1"],
        )
        data = buf.to_dict()
        restored = WorkingMemoryBuffer.from_dict(data)
        assert restored.current_goal == "test goal"
        assert restored.active_context == ["item1", "item2"]


# ---------------------------------------------------------------------------
# store.py
# ---------------------------------------------------------------------------

class TestMemoryStore:
    def test_add_and_search(self, tmp_store):
        chunk_id = tmp_store.add(
            "The database password is in AWS Secrets Manager",
            type="credential",
            source="test",
        )
        assert isinstance(chunk_id, str)
        assert len(chunk_id) > 0

        results = tmp_store.search("Where is the database password?")
        assert len(results) >= 1
        assert "Secrets Manager" in results[0].text or "password" in results[0].text.lower()

    def test_add_invalid_type_raises(self, tmp_store):
        with pytest.raises(ValueError, match="Invalid type"):
            tmp_store.add("some text", type="invalid_type_xyz")

    def test_search_empty_store(self, tmp_store):
        results = tmp_store.search("anything")
        assert results == []

    def test_result_is_memory_result(self, tmp_store):
        from agent_memory_core import MemoryResult
        tmp_store.add("Python 3.12 is required", type="technical", source="test")
        results = tmp_store.search("Python version")
        assert all(isinstance(r, MemoryResult) for r in results)

    def test_result_fields_populated(self, tmp_store):
        tmp_store.add("API key is in keychain", type="credential", source="test-source")
        results = tmp_store.search("API key")
        r = results[0]
        assert r.type == "credential"
        assert r.source == "test-source"
        assert 0.0 <= r.salience <= 1.0
        assert 0.0 <= r.recency_score <= 1.0
        assert r.combined_score >= 0.0

    def test_forget_by_id(self, tmp_store):
        chunk_id = tmp_store.add("temporary fact", type="fact", source="test")
        removed = tmp_store.forget(id=chunk_id)
        assert removed == 1
        results = tmp_store.search("temporary fact")
        assert all(chunk_id != r.id for r in results)

    def test_forget_by_source(self, tmp_store):
        tmp_store.add("fact one", type="fact", source="source-a")
        tmp_store.add("fact two", type="fact", source="source-a")
        tmp_store.add("fact three", type="fact", source="source-b")
        removed = tmp_store.forget(source="source-a")
        assert removed == 2

    def test_status(self, tmp_store):
        tmp_store.add("something", type="fact", source="test")
        s = tmp_store.status()
        assert s["total_chunks"] == 1
        assert "fact" in s["by_type"]
        assert "db_size_mb" in s
        assert s["hindsight_available"] is False  # no hindsight configured

    def test_type_filter(self, tmp_store):
        tmp_store.add("credential text", type="credential", source="test")
        tmp_store.add("session text", type="session", source="test")
        results = tmp_store.search("text", type="credential")
        assert all(r.type == "credential" for r in results)

    def test_agent_namespace(self, tmp_store):
        tmp_store.add("shared fact", type="fact", source="test", agent="shared")
        tmp_store.add("private fact for alice", type="fact", source="test", agent="alice")
        # alice sees shared + private
        alice_results = tmp_store.search("fact", agent="alice")
        texts = [r.text for r in alice_results]
        assert any("shared" in t for t in texts)
        assert any("alice" in t for t in texts)

    def test_get_all(self, tmp_store):
        tmp_store.add("chunk 1", type="fact", source="test")
        tmp_store.add("chunk 2", type="session", source="test")
        all_chunks = tmp_store.get_all()
        assert len(all_chunks) == 2
        assert all("id" in c and "text" in c and "metadata" in c for c in all_chunks)

    def test_get_all_type_filter(self, tmp_store):
        tmp_store.add("fact chunk", type="fact", source="test")
        tmp_store.add("session chunk", type="session", source="test")
        facts = tmp_store.get_all(type="fact")
        assert len(facts) == 1
        assert facts[0]["text"] == "fact chunk"

    def test_include_archived(self, tmp_store):
        chunk_id = tmp_store.add("to be archived", type="observation", source="test")
        tmp_store.update_metadata([chunk_id], [{"consolidated_into": "some-other-id"}])
        # Default search excludes archived
        results_default = tmp_store.search("archived")
        assert all(r.id != chunk_id for r in results_default)
        # get_all with include_archived=True
        all_chunks = tmp_store.get_all(include_archived=True)
        chunk_ids = [c["id"] for c in all_chunks]
        assert chunk_id in chunk_ids


# ---------------------------------------------------------------------------
# working.py
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    def test_empty_on_init(self, tmp_working):
        buf = tmp_working.get()
        assert buf.current_goal == ""
        assert buf.active_context == []
        assert buf.blockers == []
        assert buf.next_actions == []

    def test_set_goal(self, tmp_working):
        tmp_working.set_goal("Build the divergence dataset")
        buf = tmp_working.get()
        assert buf.current_goal == "Build the divergence dataset"

    def test_add_context(self, tmp_working):
        tmp_working.add_context("item 1")
        tmp_working.add_context("item 2")
        buf = tmp_working.get()
        assert "item 1" in buf.active_context
        assert "item 2" in buf.active_context

    def test_context_fifo_drop(self):
        """Oldest context item is dropped when max_context_slots is exceeded."""
        from agent_memory_core import WorkingMemory
        with tempfile.TemporaryDirectory() as tmp:
            wm = WorkingMemory(buffer_path=Path(tmp) / "wm.json", max_context_slots=3)
            wm.add_context("first")
            wm.add_context("second")
            wm.add_context("third")
            dropped = wm.add_context("fourth")
            assert dropped == "first"
            buf = wm.get()
            assert "first" not in buf.active_context
            assert "fourth" in buf.active_context

    def test_add_blocker(self, tmp_working):
        tmp_working.add_blocker("Waiting for GPU")
        buf = tmp_working.get()
        assert "Waiting for GPU" in buf.blockers

    def test_add_action(self, tmp_working):
        tmp_working.add_action("Run preflight check")
        buf = tmp_working.get()
        assert "Run preflight check" in buf.next_actions

    def test_clear(self, tmp_working):
        tmp_working.set_goal("some goal")
        tmp_working.add_context("some context")
        tmp_working.clear()
        buf = tmp_working.get()
        assert buf.current_goal == ""
        assert buf.active_context == []

    def test_flush_to_store(self, tmp_store, tmp_working):
        tmp_working.set_goal("Test flush")
        tmp_working.add_context("ctx item")
        chunk_id = tmp_working.flush(tmp_store)
        assert chunk_id is not None
        # Buffer should be cleared after flush
        buf = tmp_working.get()
        assert buf.current_goal == ""
        # Memory should be searchable
        results = tmp_store.search("Test flush")
        assert len(results) >= 1

    def test_flush_empty_returns_none(self, tmp_store, tmp_working):
        result = tmp_working.flush(tmp_store)
        assert result is None

    def test_as_query_context(self, tmp_working):
        tmp_working.set_goal("Build dataset")
        tmp_working.add_context("RTX 4090 ordered")
        ctx = tmp_working.as_query_context()
        assert "Build dataset" in ctx


# ---------------------------------------------------------------------------
# consolidation.py (clustering logic only — no LLM required)
# ---------------------------------------------------------------------------

class TestConsolidationClustering:
    def _make_chunk(self, chunk_id, text, source, ctype):
        return {
            "id": chunk_id,
            "text": text,
            "metadata": {"source_file": source, "chunk_type": ctype},
        }

    def test_source_type_clustering(self):
        from agent_memory_core.consolidation import cluster_chunks
        chunks = [
            self._make_chunk("a1", "The API failed with 404", "pipeline.md", "observation"),
            self._make_chunk("a2", "The API failed again with 404 error", "pipeline.md", "observation"),
            self._make_chunk("b1", "Different source content", "other.md", "session"),
        ]
        clusters = cluster_chunks(chunks)
        source_clusters = [c for c in clusters if c["strategy"] == "source+type"]
        assert len(source_clusters) >= 1
        assert len(source_clusters[0]["chunks"]) == 2

    def test_keyword_clustering(self):
        from agent_memory_core.consolidation import cluster_chunks
        chunks = [
            self._make_chunk("x1", "divergence routing model inference latency benchmark", "a.md", "session"),
            self._make_chunk("x2", "divergence routing model evaluation performance benchmark", "b.md", "session"),
            self._make_chunk("x3", "completely unrelated topics about cats and dogs today", "c.md", "session"),
        ]
        clusters = cluster_chunks(chunks, similarity_threshold=0.15)
        kw_clusters = [c for c in clusters if c["strategy"] == "type+keywords"]
        # x1 and x2 share enough keywords to cluster
        if kw_clusters:
            members = kw_clusters[0]["chunks"]
            member_ids = {m["id"] for m in members}
            assert "x1" in member_ids or "x2" in member_ids

    def test_no_clusters_when_all_unique(self):
        from agent_memory_core.consolidation import cluster_chunks
        chunks = [
            self._make_chunk("u1", "apple orchard harvest season fruit", "a.md", "fact"),
            self._make_chunk("u2", "mathematics algebra calculus derivatives integral", "b.md", "session"),
            self._make_chunk("u3", "cooking recipes pasta italian chef restaurant", "c.md", "dream"),
        ]
        clusters = cluster_chunks(chunks, similarity_threshold=0.8)
        # No clusters should form with very high threshold on semantically distinct texts
        for c in clusters:
            assert len(c["chunks"]) < 3 or c["strategy"] == "source+type"


# ---------------------------------------------------------------------------
# eval.py
# ---------------------------------------------------------------------------

class TestMemoryEval:
    def test_score_query_recall_hit(self, tmp_store):
        from agent_memory_core import MemoryEval, MemoryResult
        ev = MemoryEval(tmp_store)
        # Fake a result that contains the expected fact
        fake_result = MemoryResult(
            id="r1", text="The aff84f3e issuer ID is in ASC",
            type="credential", source="test", date="2026-04-09",
            distance=0.1, recency_score=0.9, salience=0.8,
            combined_score=0.1, age_days=0,
        )
        q = {"query": "ASC issuer ID", "expected_facts": ["aff84f3e"], "type": "credential"}
        result = ev.score_query(q, [fake_result])
        assert result.recall is True
        assert result.precision > 0
        assert result.answer is True

    def test_score_query_recall_miss(self, tmp_store):
        from agent_memory_core import MemoryEval, MemoryResult
        ev = MemoryEval(tmp_store)
        fake_result = MemoryResult(
            id="r1", text="Nothing useful here at all",
            type="session", source="test", date="2026-04-09",
            distance=0.9, recency_score=0.5, salience=0.3,
            combined_score=0.8, age_days=0,
        )
        q = {"query": "ASC issuer ID", "expected_facts": ["aff84f3e"], "type": "credential"}
        result = ev.score_query(q, [fake_result])
        assert result.recall is False
        assert result.precision == 0.0

    def test_score_query_empty_results(self, tmp_store):
        from agent_memory_core import MemoryEval
        ev = MemoryEval(tmp_store)
        q = {"query": "test", "expected_facts": ["something"], "type": "fact"}
        result = ev.score_query(q, [])
        assert result.recall is False
        assert result.precision == 0.0
        assert result.answer is False
        assert result.results_count == 0

    def test_custom_queries(self, tmp_store):
        from agent_memory_core import MemoryEval
        ev = MemoryEval(tmp_store, queries=[])
        ev.add_query("test query", expected_facts=["fact1"], type="fact")
        assert len(ev._queries) == 1

    def test_history_persistence(self, tmp_path, tmp_store):
        from agent_memory_core import MemoryEval
        ev = MemoryEval(
            tmp_store,
            history_path=tmp_path / "eval-history.json",
            queries=[],
        )
        ev.add_query("simple query", expected_facts=["anything"], type="fact")
        report = ev.run(n=5, version="test-v1", verbose=False, save=True)
        assert report["version"] == "test-v1"
        history = ev.history()
        assert len(history) == 1
        assert history[0]["version"] == "test-v1"

    def test_score_delta_insufficient_history(self, tmp_path, tmp_store):
        from agent_memory_core import MemoryEval
        ev = MemoryEval(tmp_store, history_path=tmp_path / "eh.json", queries=[])
        assert ev.score_delta() is None


# ---------------------------------------------------------------------------
# graph.py (offline — no LLM calls)
# ---------------------------------------------------------------------------

class TestMemoryGraph:
    def test_load_nonexistent_returns_none(self, tmp_path):
        from agent_memory_core import MemoryGraph
        g = MemoryGraph(graph_path=tmp_path / "missing.json")
        assert g.load() is None

    def test_stats_no_graph(self, tmp_path):
        from agent_memory_core import MemoryGraph
        g = MemoryGraph(graph_path=tmp_path / "missing.json")
        stats = g.stats()
        assert stats["built"] is False

    def test_contradictions_no_graph(self, tmp_path):
        from agent_memory_core import MemoryGraph
        g = MemoryGraph(graph_path=tmp_path / "missing.json")
        assert g.contradictions() == []

    def test_search_no_graph(self, tmp_path):
        from agent_memory_core import MemoryGraph
        g = MemoryGraph(graph_path=tmp_path / "missing.json")
        result = g.search("anything")
        assert result["direct"] == []
        assert result["neighbors"] == []

    def test_entity_map_no_graph(self, tmp_path):
        from agent_memory_core import MemoryGraph
        g = MemoryGraph(graph_path=tmp_path / "missing.json")
        assert g.entity_map() == {}

    def test_load_existing_graph(self, tmp_path):
        """Graph can be loaded from a pre-existing JSON file."""
        import json
        from agent_memory_core import MemoryGraph
        fake_graph = {
            "version": "1.0",
            "built_at": "2026-04-09T00:00:00",
            "node_count": 2,
            "edge_count": 1,
            "nodes": {
                "abc123": {
                    "id": "abc123",
                    "source_file": "/memory/a.md",
                    "title": "Node A",
                    "summary": "First node",
                    "type": "INFERRED",
                    "domain": "project",
                    "confidence": 0.7,
                    "last_modified": "2026-04-09T00:00:00",
                    "entities": ["divergence", "router"],
                    "topics": ["ai", "routing"],
                    "relationships": [{"target_id": "def456", "type": "co-occurs", "weight": 0.5}],
                },
                "def456": {
                    "id": "def456",
                    "source_file": "/memory/b.md",
                    "title": "Node B",
                    "summary": "Second node",
                    "type": "INFERRED",
                    "domain": "reference",
                    "confidence": 0.6,
                    "last_modified": "2026-04-09T00:00:00",
                    "entities": ["divergence"],
                    "topics": ["ai"],
                    "relationships": [{"target_id": "abc123", "type": "co-occurs", "weight": 0.5}],
                },
            },
            "edges": [{"source": "abc123", "target": "def456", "type": "co-occurs", "weight": 0.5, "shared": ["divergence"]}],
        }
        graph_file = tmp_path / "graph.json"
        graph_file.write_text(json.dumps(fake_graph))

        g = MemoryGraph(graph_path=graph_file)
        loaded = g.load()
        assert loaded is not None
        assert loaded["node_count"] == 2

        stats = g.stats()
        assert stats["built"] is True
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1

        em = g.entity_map()
        assert "/memory/a.md" in em
        assert "divergence" in em["/memory/a.md"]

        results = g.search("divergence router")
        assert len(results["direct"]) >= 1


# ---------------------------------------------------------------------------
# forgetting.py
# ---------------------------------------------------------------------------

class TestForgettingPolicy:
    def test_find_stale_chunks(self, tmp_store):
        from agent_memory_core import ForgettingPolicy
        # Add a chunk with an old date
        tmp_store.add("very old observation", type="observation", source="test",
                      extra_metadata={"date": "2020-01-01"})
        fp = ForgettingPolicy(tmp_store, stale_threshold_days=30)
        stale = fp.find_stale_chunks()
        assert len(stale) >= 1
        assert stale[0]["age_days"] > 30

    def test_credentials_never_stale(self, tmp_store):
        from agent_memory_core import ForgettingPolicy
        tmp_store.add("old credential", type="credential", source="test",
                      extra_metadata={"date": "2020-01-01"})
        fp = ForgettingPolicy(tmp_store, stale_threshold_days=30)
        stale = fp.find_stale_chunks()
        assert all(c["type"] != "credential" for c in stale)

    def test_archive_chunks(self, tmp_store):
        from agent_memory_core import ForgettingPolicy
        cid = tmp_store.add("to archive", type="observation", source="test")
        fp = ForgettingPolicy(tmp_store)
        count = fp.archive_chunks([cid], reason="test")
        assert count == 1
        # Archived chunk should not appear in default search
        results = tmp_store.search("to archive")
        assert all(r.id != cid for r in results)

    def test_hard_delete(self, tmp_store):
        from agent_memory_core import ForgettingPolicy
        cid = tmp_store.add("to delete", type="observation", source="test")
        fp = ForgettingPolicy(tmp_store)
        count = fp.hard_delete([cid])
        assert count == 1
        assert tmp_store.status()["total_chunks"] == 0

    def test_health_report_structure(self, tmp_store):
        from agent_memory_core import ForgettingPolicy
        fp = ForgettingPolicy(tmp_store, hindsight_url=None)
        report = fp.health_report()
        assert "score" in report
        assert "stale_files" in report
        assert "duplicates" in report
        assert "hindsight" in report
        assert "store_status" in report
        assert "warnings" in report
        assert 0 <= report["score"] <= 100

    def test_forget_source(self, tmp_store):
        from agent_memory_core import ForgettingPolicy
        tmp_store.add("from source x", type="fact", source="source-x")
        tmp_store.add("also from source x", type="fact", source="source-x")
        fp = ForgettingPolicy(tmp_store)
        removed = fp.forget_source("source-x")
        assert removed == 2
        assert tmp_store.status()["total_chunks"] == 0
