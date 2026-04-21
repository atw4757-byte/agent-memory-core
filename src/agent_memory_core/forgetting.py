"""Active forgetting: stale detection, health scoring, contradiction surfacing."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .store import MemoryStore


# Default health scoring weights

_DEFAULT_PENALTIES = {
    "stale_over_20":    15,   # deduct 15 pts if > 20 stale files
    "contradictions_over_10": 20,
    "has_duplicates":   10,
    "hindsight_not_all_healthy": 20,
    "graph_edges_under_50": 15,
}


class ForgettingPolicy:
    """Detect, surface, and optionally archive stale memories.

    Parameters
    ----------
    store:
        The MemoryStore instance to operate on.
    stale_threshold_days:
        Files not modified in this many days are considered stale. Default: 30.
    memory_dirs:
        List of directories to scan for markdown files. Used for stale/duplicate
        detection which is file-based (not ChromaDB-based).
    hindsight_url:
        Hindsight base URL for health checks. Pass None to skip.
    """

    def __init__(
        self,
        store: MemoryStore,
        stale_threshold_days: int = 30,
        memory_dirs: Optional[list[str | Path]] = None,
        hindsight_url: Optional[str] = None,
    ) -> None:
        self._store = store
        self._stale_days = stale_threshold_days
        self._memory_dirs = [Path(d) for d in memory_dirs] if memory_dirs else []
        self._hindsight_url = hindsight_url

    # Stale file detection

    def find_stale_files(self) -> list[dict]:
        """Return markdown files not modified within the stale threshold.

        Each entry has keys: file (str), name (str), age_days (int).
        Results are sorted by age descending (oldest first).
        """
        stale = []
        cutoff = datetime.now().timestamp() - (self._stale_days * 86400)
        for d in self._memory_dirs:
            if not d.exists():
                continue
            for f in d.rglob("*.md"):
                if f.stat().st_mtime < cutoff:
                    age = int((datetime.now().timestamp() - f.stat().st_mtime) / 86400)
                    stale.append({"file": str(f), "name": f.stem, "age_days": age})
        return sorted(stale, key=lambda x: -x["age_days"])

    # Stale chunk detection

    def find_stale_chunks(self, types: Optional[list[str]] = None) -> list[dict]:
        """Return ChromaDB chunks older than the stale threshold.

        Unlike ``find_stale_files``, this queries the vector store directly
        and respects the per-type decay profile: credential and lesson types
        are never flagged regardless of age.

        Each entry has keys: id, text, type, date, age_days, source.
        """
        all_chunks = self._store.get_all(include_archived=False)
        never_stale = {"credential", "lesson"}
        cutoff = datetime.now() - timedelta(days=self._stale_days)
        stale = []

        for chunk in all_chunks:
            meta = chunk["metadata"]
            chunk_type = meta.get("chunk_type", "unknown")
            if chunk_type in never_stale:
                continue
            if types and chunk_type not in types:
                continue

            date_str = meta.get("date", "")
            if not date_str:
                continue
            try:
                mem_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                age_days = (datetime.now() - mem_date).days
                if mem_date < cutoff:
                    stale.append({
                        "id": chunk["id"],
                        "text": chunk["text"][:100],
                        "type": chunk_type,
                        "date": date_str,
                        "age_days": age_days,
                        "source": meta.get("source_file", ""),
                    })
            except (ValueError, TypeError):
                continue

        return sorted(stale, key=lambda x: -x["age_days"])

    # Duplicate detection (file-based)

    def find_duplicate_files(self) -> dict[str, list[str]]:
        """Find markdown files with identical frontmatter 'name' fields.

        Returns a dict of {name: [file_path_1, file_path_2, ...]} for groups
        with more than one file. Groups with a single file are excluded.
        """
        files: dict[str, list[str]] = {}
        for d in self._memory_dirs:
            if not d.exists():
                continue
            for f in d.rglob("*.md"):
                try:
                    content = f.read_text()[:200].lower()
                    if content.startswith("---"):
                        end = content.find("---", 3)
                        if end > 0:
                            fm = content[3:end]
                            for line in fm.split("\n"):
                                if line.startswith("name:"):
                                    title = line.split(":", 1)[1].strip()
                                    files.setdefault(title, []).append(str(f))
                except Exception:
                    continue
        return {k: v for k, v in files.items() if len(v) > 1}

    # Hindsight health

    def check_hindsight(self) -> dict[str, str]:
        """Probe each Hindsight bank. Returns {bank_name: 'healthy' | 'unreachable'}."""
        banks = ["core", "projects", "sessions", "health", "dreams", "lessons"]
        results: dict[str, str] = {}
        if not self._hindsight_url:
            return {b: "not_configured" for b in banks}

        for bank in banks:
            try:
                req = urllib.request.Request(
                    f"{self._hindsight_url}/v1/default/banks/{bank}/memories/recall",
                    data=json.dumps({"query": "test", "n": 1}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as _:
                    results[bank] = "healthy"
            except Exception:
                results[bank] = "unreachable"
        return results

    # Archive management

    def archive_chunks(self, chunk_ids: list[str], reason: str = "manual") -> int:
        """Mark chunks as archived by setting ``consolidated_into`` metadata.

        This is a soft delete — chunks remain in ChromaDB but are excluded
        from default search results. Pass ``include_archived=True`` to surface them.

        Returns the number of chunks archived.
        """
        if not chunk_ids:
            return 0
        today = datetime.now().strftime("%Y-%m-%d")
        metadata_updates = [
            {"consolidated_into": f"archived:{reason}", "archived_at": today}
        ] * len(chunk_ids)
        self._store.update_metadata(chunk_ids, metadata_updates)
        return len(chunk_ids)

    def hard_delete(self, chunk_ids: list[str]) -> int:
        """Permanently remove chunks from ChromaDB by ID.

        Use with caution — this is irreversible. For soft removal use
        ``archive_chunks`` instead.

        Returns the number of chunks deleted.
        """
        count = 0
        for chunk_id in chunk_ids:
            count += self._store.forget(id=chunk_id)
        return count

    def forget_source(self, source_file: str) -> int:
        """Remove all chunks whose source_file matches. Delegates to MemoryStore.forget."""
        return self._store.forget(source=source_file)

    # Health report

    def health_report(
        self,
        graph_stats: Optional[dict] = None,
        penalties: Optional[dict] = None,
    ) -> dict:
        """Run all health checks and return a structured report with a health score.

        Parameters
        ----------
        graph_stats:
            Optional output from ``MemoryGraph.stats()``. If provided, edge count
            is included in the health score.
        penalties:
            Override default penalty weights. Keys correspond to ``_DEFAULT_PENALTIES``.

        Returns
        -------
        dict with keys:
          score (int 0-100), stale_files (list), duplicates (dict),
          hindsight (dict), store_status (dict), graph (dict | None), warnings (list)
        """
        p = {**_DEFAULT_PENALTIES, **(penalties or {})}
        score = 100
        warnings = []

        stale = self.find_stale_files()
        if len(stale) > 20:
            score -= p["stale_over_20"]
            warnings.append(f"{len(stale)} files stale (>{self._stale_days} days)")

        duplicates = self.find_duplicate_files()
        if duplicates:
            score -= p["has_duplicates"]
            warnings.append(f"{len(duplicates)} duplicate title groups found")

        hindsight = self.check_hindsight()
        healthy_count = sum(1 for v in hindsight.values() if v == "healthy")
        total_banks = len(hindsight)
        if healthy_count < total_banks and self._hindsight_url:
            score -= p["hindsight_not_all_healthy"]
            warnings.append(f"Hindsight: only {healthy_count}/{total_banks} banks healthy")

        store_status = self._store.status()

        graph_info = None
        if graph_stats:
            graph_info = graph_stats
            edge_count = graph_stats.get("edge_count", 0)
            if edge_count < 50:
                score -= p["graph_edges_under_50"]
                warnings.append(f"Memory graph has only {edge_count} edges (< 50)")

        # Contradictions (requires graph_stats from a MemoryGraph instance)
        # Caller can pass contradiction_count in graph_stats for scoring
        contradiction_count = (graph_stats or {}).get("contradiction_count", 0)
        if contradiction_count > 10:
            score -= p["contradictions_over_10"]
            warnings.append(f"{contradiction_count} contradictions detected in memory graph")

        return {
            "score": max(0, score),
            "timestamp": datetime.now().isoformat(),
            "stale_files": stale[:20],
            "stale_file_count": len(stale),
            "duplicates": duplicates,
            "hindsight": hindsight,
            "store_status": store_status,
            "graph": graph_info,
            "warnings": warnings,
        }
