"""
agent_memory_core.store — ChromaDB-backed memory store.

Core capabilities:
  - Persistent ChromaDB storage with cosine similarity
  - Salience scoring (type prior + access count + graph connectivity)
  - Adaptive retrieval: query-intent detection adjusts similarity/recency/salience weights
  - Cross-encoder re-ranking (optional — install with [reranker] extra)
  - MMR (Maximal Marginal Relevance) for result diversity
  - Active forgetting: archived/consolidated chunks excluded by default
  - Atomic fact layer: augments chunk results with fine-grained facts
  - Agent namespacing: private memories per agent, shared memories visible to all

Dependencies:
  - chromadb (required)
  - sentence-transformers (optional, for cross-encoder re-ranking)
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from .types import (
    DECAY_RATES,
    TYPE_SALIENCE_PRIORS,
    TYPE_TO_BANK,
    VALID_TYPES,
    MemoryResult,
    compute_recency,
    compute_salience,
)

# ---------------------------------------------------------------------------
# Optional cross-encoder re-ranker
# ---------------------------------------------------------------------------

try:
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False
    _RERANKER = None


# ---------------------------------------------------------------------------
# Query intent detection
# ---------------------------------------------------------------------------

_CREDENTIAL_WORDS = frozenset(["key", "password", "token", "api", "secret", "credential", "auth", "oauth"])
_RECENCY_WORDS    = frozenset(["current", "now", "today", "status", "latest", "recent", "right now"])
_LESSON_WORDS     = frozenset(["mistake", "wrong", "fix", "error", "rule", "should", "lesson", "correction", "bug"])
_PERSONAL_WORDS   = frozenset(["personal", "preference", "habit", "name", "family", "home", "location"])
_TECHNICAL_WORDS  = frozenset(["code", "script", "install", "config", "port", "ssh", "docker", "server", "python", "function"])
_PROJECT_WORDS    = frozenset(["project", "app", "build", "release", "version", "feature"])
_SESSION_WORDS    = frozenset(["session", "last", "yesterday", "week", "ago", "previous", "earlier", "history"])

# Cross-encoder relevance gate thresholds (raw CE scores, higher = more relevant).
# ms-marco-MiniLM-L-6-v2 typically ranges roughly -10 to +10.
CE_THRESHOLDS: dict[str, float] = {
    "credential":     -3.0,   # slightly loose to keep partial matches
    "lesson":         -4.0,
    "personal":       -4.0,
    "project_status": -5.0,   # broad — status queries vary widely
    "technical":      -4.0,
    "session":        -6.0,   # very broad — recency queries
    "default":        -5.0,
}


def detect_query_type(query: str) -> str:
    """Classify a query into a type label used to select the CE relevance-gate threshold.

    Returns one of: credential / lesson / personal / technical / project_status / session / default.
    This classification is independent of chunk_type — it is purely for threshold lookup.
    """
    words = set(query.lower().split())
    if words & _CREDENTIAL_WORDS:
        return "credential"
    if words & _LESSON_WORDS:
        return "lesson"
    if words & _PERSONAL_WORDS:
        return "personal"
    if words & _TECHNICAL_WORDS:
        return "technical"
    if words & _PROJECT_WORDS:
        return "project_status"
    if words & _SESSION_WORDS:
        return "session"
    return "default"


def _detect_query_weights(query: str) -> tuple[float, float, float]:
    """Return (w_similarity, w_recency, w_salience) based on query intent.

    Weights sum to 1.0. Lower combined_score = better match (distance-style).
    """
    words = set(query.lower().split())
    if words & _CREDENTIAL_WORDS:
        return 0.9, 0.0, 0.1
    if words & _RECENCY_WORDS:
        return 0.3, 0.5, 0.2
    if words & _LESSON_WORDS:
        return 0.5, 0.1, 0.4
    return 0.5, 0.2, 0.3


# ---------------------------------------------------------------------------
# MMR helpers
# ---------------------------------------------------------------------------

def _text_bigram_vec(text: str) -> dict[str, int]:
    vec: dict[str, int] = {}
    for i in range(len(text) - 1):
        bg = text[i:i + 2]
        vec[bg] = vec.get(bg, 0) + 1
    return vec


def _cosine(va: dict, vb: dict) -> float:
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in va)
    na = sum(v * v for v in va.values()) ** 0.5
    nb = sum(v * v for v in vb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mmr_rerank(candidates: list[dict], n: int, lambda_: float = 0.7) -> list[dict]:
    """Maximal Marginal Relevance — balances relevance with diversity."""
    if len(candidates) <= 1:
        return candidates[:n]

    vecs = [_text_bigram_vec(r["text"]) for r in candidates]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < n:
        if not selected:
            best = min(remaining, key=lambda idx: candidates[idx]["combined_score"])
        else:
            best = None
            best_mmr: Optional[float] = None
            for idx in remaining:
                relevance = 1.0 - candidates[idx]["combined_score"]
                max_sim = max(_cosine(vecs[idx], vecs[sel]) for sel in selected)
                mmr = lambda_ * relevance - (1.0 - lambda_) * max_sim
                if best_mmr is None or mmr > best_mmr:
                    best_mmr = mmr
                    best = idx
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


# ---------------------------------------------------------------------------
# Chunk ID helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(source: str, section_heading: str) -> str:
    return f"{source}::{section_heading}"


def _make_add_chunk_id(source: str, text: str, today: str) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return _make_chunk_id(source, f"added-{today}-{h}")


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """ChromaDB-backed semantic memory store with salience-weighted retrieval.

    Parameters
    ----------
    db_path:
        Directory where ChromaDB persists its data.
        Defaults to ``~/.agent-memory-core/vectordb``.
    collection_name:
        ChromaDB collection name. Override to isolate multiple stores.
    facts_collection_name:
        ChromaDB collection for atomic facts produced by consolidation.
    graph_path:
        Path to ``memory_graph.json`` used for graph-connectivity salience boosts.
        Set to ``None`` to disable graph-based boosting.
    hindsight_url:
        Base URL of a running Hindsight instance (e.g. ``http://localhost:8889``).
        Pass ``None`` to disable Hindsight integration.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        collection_name: str = "agent-memory",
        facts_collection_name: str = "agent-memory-facts",
        graph_path: Optional[str | Path] = None,
        hindsight_url: Optional[str] = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path else Path.home() / ".agent-memory-core" / "vectordb"
        self._collection_name = collection_name
        self._facts_collection_name = facts_collection_name
        self._graph_path = Path(graph_path) if graph_path else None
        if hindsight_url is not None and not hindsight_url.startswith(("http://", "https://")):
            raise ValueError(f"hindsight_url must start with http:// or https://, got: {hindsight_url!r}")
        self._hindsight_url = hindsight_url

        # Lazy-loaded ChromaDB handles
        self._client: Any = None
        self._collection: Any = None
        self._facts_collection: Any = None

        # Lazy-loaded graph connectivity cache: {source_file: connection_count}
        self._graph_connectivity: Optional[dict[str, int]] = None

    # ------------------------------------------------------------------
    # ChromaDB initialisation
    # ------------------------------------------------------------------

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        import chromadb
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def _get_facts_collection(self) -> Optional[Any]:
        if self._facts_collection is not None:
            return self._facts_collection
        try:
            # Ensure the primary client is initialised so we share a single connection
            if self._client is None:
                self._get_collection()
            col = self._client.get_or_create_collection(
                name=self._facts_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if col.count() > 0:
                self._facts_collection = col
                return self._facts_collection
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Graph connectivity (salience boost)
    # ------------------------------------------------------------------

    def _load_graph_connectivity(self) -> dict[str, int]:
        if self._graph_connectivity is not None:
            return self._graph_connectivity
        self._graph_connectivity = {}
        if not self._graph_path or not self._graph_path.exists():
            return self._graph_connectivity
        try:
            import json
            data = json.loads(self._graph_path.read_text())
            for node in data.get("nodes", {}).values():
                src = node.get("source_file", "")
                if src:
                    connections = len(node.get("relationships", []))
                    self._graph_connectivity[src] = (
                        self._graph_connectivity.get(src, 0) + connections
                    )
        except Exception:
            pass
        return self._graph_connectivity

    # ------------------------------------------------------------------
    # Hindsight integration
    # ------------------------------------------------------------------

    def _hindsight_available(self) -> bool:
        if not self._hindsight_url:
            return False
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._hindsight_url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                import json
                data = json.loads(resp.read())
                return data.get("status") == "healthy"
        except Exception:
            return False

    def _hindsight_retain(self, bank_id: str, items: list[dict]) -> bool:
        if not items or not self._hindsight_url:
            return False
        import json, urllib.request
        payload = json.dumps({"items": items, "async": True}).encode()
        req = urllib.request.Request(
            f"{self._hindsight_url}/v1/default/banks/{bank_id}/memories",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                return result.get("success", False)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core add / search / forget / status
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        type: str = "fact",
        source: str = "manual",
        agent: str = "shared",
        extra_metadata: Optional[dict] = None,
    ) -> str:
        """Add a memory chunk to the store.

        Parameters
        ----------
        text:    The memory text to store.
        type:    Chunk type — must be one of ``VALID_TYPES``.
        source:  Identifier for the origin of this memory (file path, script name, etc.).
        agent:   Namespace. ``"shared"`` (default) is visible to all agents.
                 Any other name creates a private memory for that agent.
        extra_metadata:
                 Additional key-value pairs merged into the stored metadata.

        Returns
        -------
        str: The deterministic chunk ID that was upserted.
        """
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid type {type!r}. Must be one of: {sorted(VALID_TYPES)}")

        collection = self._get_collection()
        today = date.today().isoformat()
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M")

        chunk_id = _make_add_chunk_id(source, text, today)
        metadata: dict[str, Any] = {
            "source_file": source,
            "section_heading": f"Added {today}",
            "chunk_type": type,
            "date": today,
            "timestamp": timestamp,
            "agent": agent,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        collection.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata])

        # Push to Hindsight if available
        if self._hindsight_available():
            bank_id = TYPE_TO_BANK.get(type, "core")
            self._hindsight_retain(bank_id, [{
                "content": text,
                "context": f"{source} / Added {today}",
                "document_id": chunk_id,
                "metadata": metadata,
            }])

        return chunk_id

    def upsert_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Low-level upsert of a pre-built chunk (used by Consolidator and indexing pipelines)."""
        collection = self._get_collection()
        collection.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata])

    def upsert_fact(self, fact_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Upsert an atomic fact into the facts sub-collection."""
        if self._client is None:
            self._get_collection()
        col = self._client.get_or_create_collection(
            name=self._facts_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        col.upsert(ids=[fact_id], documents=[text], metadatas=[metadata])
        self._facts_collection = col  # update cached ref

    def search(
        self,
        query: str,
        n: int = 5,
        type: Optional[str] = None,
        since: Optional[str] = None,
        agent: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[MemoryResult]:
        """Semantic search across stored memories.

        Parameters
        ----------
        query:    Natural language query.
        n:        Maximum number of results to return.
        type:     Restrict results to this chunk type.
        since:    Only return memories on or after this date (YYYY-MM-DD).
        agent:    If set, returns shared memories plus this agent's private memories.
        include_archived:
                  Include chunks that were archived during consolidation.
                  Useful for debugging; disabled by default.

        Returns
        -------
        list[MemoryResult]: Results sorted by combined_score (lower = better).
        """
        collection = self._get_collection()

        # Build ChromaDB where clause
        base_filters = []
        if type:
            base_filters.append({"chunk_type": {"$eq": type}})
        if since:
            base_filters.append({"date": {"$gte": since}})

        agent_filter = None
        if agent:
            agent_filter = {
                "$or": [
                    {"agent": {"$eq": "shared"}},
                    {"agent": {"$eq": agent}},
                ]
            }

        where = None
        if agent_filter and base_filters:
            where = {"$and": base_filters + [agent_filter]}
        elif agent_filter:
            where = agent_filter
        elif len(base_filters) == 2:
            where = {"$and": base_filters}
        elif len(base_filters) == 1:
            where = base_filters[0]

        # Retrieve wider candidate pool for re-ranking
        retrieval_k = max(n * 4, 20)
        total = collection.count()
        if total == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(retrieval_k, total),
        }
        if where:
            kwargs["where"] = where

        try:
            raw = collection.query(**kwargs)
        except Exception:
            if where:
                # Retry without filter (filter might match 0 docs)
                del kwargs["where"]
                try:
                    raw = collection.query(**kwargs)
                except Exception:
                    return []
            else:
                return []

        if not raw or not raw["ids"] or not raw["ids"][0]:
            return []

        # Adaptive weights from query intent
        w_sim, w_rec, w_sal = _detect_query_weights(query)
        graph = self._load_graph_connectivity()

        raw_results: list[dict] = []
        for i, doc_id in enumerate(raw["ids"][0]):
            metadata = raw["metadatas"][0][i] if raw.get("metadatas") else {}
            distance = raw["distances"][0][i] if raw.get("distances") else 1.0
            chunk_t = metadata.get("chunk_type", "unknown")

            # Skip archived/consolidated chunks unless explicitly requested
            if not include_archived and metadata.get("consolidated_into"):
                continue

            recency = compute_recency(metadata.get("date", ""), chunk_t)
            sal = compute_salience(chunk_t, metadata, graph)
            norm_distance = min(distance / 2.0, 1.0)
            combined = (
                w_sim * norm_distance
                + w_rec * (1.0 - recency)
                + w_sal * (1.0 - sal)
            )

            date_str = metadata.get("date", "")
            age_days = 0
            if date_str:
                try:
                    mem_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                    age_days = max((datetime.now() - mem_date).days, 0)
                except (ValueError, TypeError):
                    age_days = 30

            raw_results.append({
                "id": doc_id,
                "text": raw["documents"][0][i],
                "distance": distance,
                "recency_score": round(recency, 3),
                "salience": round(sal, 3),
                "combined_score": round(combined, 4),
                "age_days": age_days,
                "chunk_type": chunk_t,
                "source": metadata.get("source_file", ""),
                "date": date_str,
                "metadata": metadata,
                "has_fact": False,
            })

        raw_results.sort(key=lambda x: x["combined_score"])

        # BM25-style keyword boost: reward chunks that contain literal query words.
        # Applied before cross-encoder so CE sees pre-boosted ordering.
        # Lower combined_score = better, so we subtract the boost amount.
        _query_words = set(query.lower().split())
        for _r in raw_results:
            _text_lower = _r["text"].lower()
            _keyword_hits = sum(1 for w in _query_words if len(w) > 2 and w in _text_lower)
            _keyword_boost = min(_keyword_hits * 0.02, 0.1)  # cap at 0.1
            _r["combined_score"] = round(_r["combined_score"] - _keyword_boost, 4)
        raw_results.sort(key=lambda x: x["combined_score"])

        # Cross-encoder re-ranking (if sentence-transformers installed)
        if HAS_RERANKER and _RERANKER and raw_results:
            pairs = [(query, r["text"]) for r in raw_results]
            try:
                ce_scores = _RERANKER.predict(pairs)
                ce_min, ce_max = min(ce_scores), max(ce_scores)
                ce_range = ce_max - ce_min if ce_max != ce_min else 1.0
                for idx, r in enumerate(raw_results):
                    norm_ce = (ce_scores[idx] - ce_min) / ce_range
                    r["combined_score"] = round(
                        float(0.5 * r["combined_score"] + 0.5 * (1.0 - norm_ce)), 4
                    )
                    r["ce_score"] = round(float(ce_scores[idx]), 4)
                raw_results.sort(key=lambda x: x["combined_score"])

                # Relevance gate: drop results whose raw CE score falls below the
                # per-query-type threshold. Always keep at least 2 results.
                _qtype = detect_query_type(query)
                _ce_threshold = CE_THRESHOLDS.get(_qtype, CE_THRESHOLDS["default"])
                _gated = [r for r in raw_results if r.get("ce_score", 0.0) >= _ce_threshold]
                if len(_gated) >= 2:
                    raw_results = _gated
                elif len(raw_results) >= 2:
                    raw_results = raw_results[:2]  # floor: keep best 2 even below threshold
            except Exception:
                pass

        # MMR diversity re-ranking
        raw_results = _mmr_rerank(raw_results, n)

        # Atomic fact augmentation: boost parent chunks that have matching facts
        facts = self._search_facts(query, n=10)
        if facts:
            raw_results = self._merge_facts(raw_results, facts)
            raw_results = raw_results[:n]

        # Dynamic tail pruning: drop low-quality outliers if >= 3 results
        if len(raw_results) >= 3:
            scores = [r["combined_score"] for r in raw_results]
            gaps = [(scores[i + 1] - scores[i], i) for i in range(len(scores) - 1)]
            max_gap, gap_pos = max(gaps, key=lambda x: x[0])
            if max_gap > 0.05 and gap_pos + 1 >= 3:
                raw_results = raw_results[: gap_pos + 1]

        # Convert to MemoryResult dataclasses
        return [
            MemoryResult(
                id=r["id"],
                text=r["text"],
                type=r["chunk_type"],
                source=r["source"],
                date=r["date"],
                distance=r["distance"],
                recency_score=r["recency_score"],
                salience=r["salience"],
                combined_score=r["combined_score"],
                age_days=r["age_days"],
                metadata=r["metadata"],
                ce_score=r.get("ce_score"),
                has_fact=r.get("has_fact", False),
            )
            for r in raw_results
        ]

    def _search_facts(self, query: str, n: int = 10) -> list[dict]:
        """Query the atomic facts collection."""
        facts_col = self._get_facts_collection()
        if facts_col is None:
            return []
        total = facts_col.count()
        if total == 0:
            return []
        try:
            results = facts_col.query(
                query_texts=[query],
                n_results=min(n, total),
            )
        except Exception:
            return []

        facts = []
        if results and results["ids"] and results["ids"][0]:
            for i, fact_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                facts.append({
                    "fact_id": fact_id,
                    "fact_text": results["documents"][0][i],
                    "parent_chunk_id": meta.get("parent_chunk_id", ""),
                    "distance": distance,
                    "chunk_type": meta.get("chunk_type", "fact"),
                    "date": meta.get("date", ""),
                })
        return facts

    def _merge_facts(self, formatted: list[dict], facts: list[dict]) -> list[dict]:
        """Supplement chunk results with atomic fact matches (never replace)."""
        chunk_index = {r["id"]: i for i, r in enumerate(formatted)}

        # Best fact per parent chunk
        best_fact: dict[str, dict] = {}
        for f in facts:
            pid = f["parent_chunk_id"]
            if pid not in best_fact or f["distance"] < best_fact[pid]["distance"]:
                best_fact[pid] = f

        for pid, f in best_fact.items():
            if pid in chunk_index:
                idx = chunk_index[pid]
                formatted[idx]["combined_score"] = max(
                    0.0, round(formatted[idx]["combined_score"] - 0.03, 4)
                )
                formatted[idx]["text"] += "\n[FACT] " + f["fact_text"]
                formatted[idx]["has_fact"] = True

        formatted.sort(key=lambda x: x["combined_score"])
        return formatted

    def get_all(
        self,
        type: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[dict]:
        """Return all stored chunks (no embedding query). Useful for consolidation.

        Returns raw dicts with keys: id, text, metadata.
        """
        collection = self._get_collection()
        total = collection.count()
        if total == 0:
            return []

        result = collection.get(include=["documents", "metadatas"])
        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            if not include_archived and meta.get("consolidated_into"):
                continue
            if type and meta.get("chunk_type") != type:
                continue
            chunks.append({
                "id": chunk_id,
                "text": result["documents"][i],
                "metadata": meta,
            })
        return chunks

    def update_metadata(self, chunk_ids: list[str], metadata_updates: list[dict]) -> None:
        """Fetch existing metadata for chunk_ids and merge in metadata_updates, then upsert."""
        collection = self._get_collection()
        result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        updated_metas = []
        for i, meta in enumerate(result["metadatas"]):
            merged = dict(meta)
            merged.update(metadata_updates[i])
            updated_metas.append(merged)
        collection.upsert(
            ids=chunk_ids,
            documents=result["documents"],
            metadatas=updated_metas,
        )

    def forget(self, source: Optional[str] = None, id: Optional[str] = None) -> int:
        """Remove memories by source file or by chunk ID.

        Parameters
        ----------
        source: Remove all chunks whose ``source_file`` metadata matches this value.
        id:     Remove a single chunk by its exact ID.

        Returns
        -------
        int: Number of chunks removed.
        """
        collection = self._get_collection()
        if id:
            try:
                collection.delete(ids=[id])
                return 1
            except Exception:
                return 0
        if source:
            results = collection.get(where={"source_file": {"$eq": source}})
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
        return 0

    def status(self) -> dict:
        """Return DB stats: total chunk count, breakdown by type, disk size, Hindsight health."""
        collection = self._get_collection()
        total = collection.count()

        type_counts: dict[str, int] = {}
        if total > 0:
            all_data = collection.get(include=["metadatas"])
            for meta in all_data["metadatas"]:
                t = meta.get("chunk_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

        db_size = sum(f.stat().st_size for f in self._db_path.rglob("*") if f.is_file())

        facts_total = 0
        try:
            fc = self._get_facts_collection()
            if fc is not None:
                facts_total = fc.count()
        except Exception:
            pass

        return {
            "total_chunks": total,
            "by_type": type_counts,
            "facts_total": facts_total,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
            "db_path": str(self._db_path),
            "hindsight_available": self._hindsight_available(),
            "reranker_available": HAS_RERANKER,
        }
