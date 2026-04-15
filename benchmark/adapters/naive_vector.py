"""
benchmark/adapters/naive_vector.py — Plain ChromaDB baseline.

"What everyone does": embed turns, search top-5. No salience scoring,
no reranking, no working memory, no temporal weighting. This is the
floor — the state of the art circa 2022.

Dependencies: chromadb (required)
"""

from __future__ import annotations

import shutil
import tempfile

try:
    import chromadb
    from chromadb.utils import embedding_functions
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False


class NaiveVectorAdapter:
    """Plain ChromaDB: embed every user turn, return top-5 by cosine similarity.

    No salience. No reranking. No temporal awareness. Flat retrieval.
    """

    def __init__(self) -> None:
        if not _HAS_CHROMA:
            raise ImportError(
                "chromadb is required for NaiveVectorAdapter. "
                "Run: pip install chromadb"
            )
        self._tmp: str | None = None
        self._client = None
        self._collection = None
        self._counter = 0
        self._init_store()

    def _init_store(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="amb_naive_")
        self._client = chromadb.PersistentClient(path=self._tmp)
        self._collection = self._client.get_or_create_collection(
            name="naive_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = 0

    def ingest_turn(self, session_id: int, role: str, content: str) -> None:
        # Only store user turns — same as most naive implementations
        if role != "user":
            return
        doc_id = f"turn_{self._counter}"
        self._counter += 1
        self._collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[{"session_id": session_id, "role": role}],
        )

    def query(self, question: str) -> str:
        if self._counter == 0:
            return ""
        results = self._collection.query(
            query_texts=[question],
            n_results=min(5, self._counter),
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return " | ".join(docs)

    def reset(self) -> None:
        # Tear down and recreate a clean store
        if self._tmp:
            try:
                # Delete the collection first (ChromaDB cleanup)
                self._client.delete_collection("naive_memory")
            except Exception:
                pass
            try:
                shutil.rmtree(self._tmp, ignore_errors=True)
            except Exception:
                pass
        self._init_store()
