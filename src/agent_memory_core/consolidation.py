"""
agent_memory_core.consolidation — Nightly lossy memory compression pipeline.

Extracted from archon-memory-consolidate. Clusters episodic memories by:
  1. Same source file + same type (strongest signal)
  2. Same type + keyword Jaccard overlap across sources
  3. Shared entities from a MemoryGraph

For clusters meeting the minimum size threshold, a local LLM (via Ollama)
compresses them into 1-3 permanent semantic facts. Original chunks are
archived (not deleted) by setting a ``consolidated_into`` metadata flag.

After compression, the consolidated text is decomposed into atomic facts
and stored in the ``archon-memory-facts`` (or equivalent) collection for
fine-grained retrieval.

Dependencies:
  - chromadb (via MemoryStore)
  - Ollama running locally with mistral:latest or qwen2.5:7b
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

from .types import TYPE_TO_BANK, VALID_TYPES
from .store import MemoryStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSOLIDATABLE_TYPES = frozenset([
    "observation",
    "session",
    "task",
    "dream",
    "project_status",
    "lesson",
])

# What consolidated output becomes
CONSOLIDATION_OUTPUT_TYPE: dict[str, str] = {
    "observation":    "fact",
    "session":        "fact",
    "task":           "lesson",
    "dream":          "fact",
    "project_status": "project_status",
    "lesson":         "lesson",
}

DEFAULT_MIN_CLUSTER = 3
MAX_CHUNKS_PER_PROMPT = 10
MAX_CHUNK_CHARS = 400

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
PREFERRED_MODELS = ["mistral:latest", "qwen2.5:7b"]


# ---------------------------------------------------------------------------
# Keyword fingerprinting
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "can", "this", "that", "these", "those", "it", "its", "from", "by",
    "as", "not", "no", "so", "if", "then", "than", "also", "more", "some",
    "all", "any", "each", "into", "about", "after", "before", "up", "out",
])


def _keyword_fingerprint(text: str, top_n: int = 8) -> frozenset:
    words = re.findall(r"[a-z]{3,}", text.lower())
    freq: dict[str, int] = defaultdict(int)
    for w in words:
        if w not in _STOPWORDS:
            freq[w] += 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return frozenset(w for w, _ in ranked[:top_n])


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_chunks(
    chunks: list[dict],
    entity_map: Optional[dict[str, set]] = None,
    similarity_threshold: float = 0.25,
) -> list[dict]:
    """Group chunks into clusters using three strategies.

    Strategy 1 — same source file + same type (strongest signal)
    Strategy 2 — same type + keyword Jaccard >= threshold
    Strategy 3 — shared entity from memory graph (entity_map lookup)

    Each cluster dict has keys: strategy, source, type, chunks.
    A chunk belongs to at most one cluster (first-match by strategy priority).

    Parameters
    ----------
    chunks:
        Raw chunk dicts from MemoryStore.get_all() with keys: id, text, metadata.
    entity_map:
        Optional {source_file: set_of_entity_strings} from MemoryGraph.
        Pass None to skip entity-based clustering.
    similarity_threshold:
        Minimum Jaccard score to form a keyword cluster (default: 0.25).
    """
    entity_map = entity_map or {}
    assigned: set[str] = set()
    clusters: list[dict] = []

    fingerprints = {c["id"]: _keyword_fingerprint(c["text"]) for c in chunks}

    # Strategy 1: same source + same type
    source_type_groups: dict[tuple, list] = defaultdict(list)
    for chunk in chunks:
        key = (
            chunk["metadata"].get("source_file", ""),
            chunk["metadata"].get("chunk_type", ""),
        )
        source_type_groups[key].append(chunk)

    for (source, ctype), group in source_type_groups.items():
        unassigned = [c for c in group if c["id"] not in assigned]
        if len(unassigned) >= 2:
            clusters.append({
                "strategy": "source+type",
                "source": source,
                "type": ctype,
                "chunks": unassigned,
            })
            for c in unassigned:
                assigned.add(c["id"])

    # Strategy 2: same type + keyword Jaccard
    by_type: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        if chunk["id"] not in assigned:
            by_type[chunk["metadata"].get("chunk_type", "")].append(chunk)

    for ctype, type_chunks in by_type.items():
        used: set[str] = set()
        for i, anchor in enumerate(type_chunks):
            if anchor["id"] in used:
                continue
            fp_a = fingerprints[anchor["id"]]
            members = [anchor]
            for j, candidate in enumerate(type_chunks):
                if i == j or candidate["id"] in used:
                    continue
                if _jaccard(fp_a, fingerprints[candidate["id"]]) >= similarity_threshold:
                    members.append(candidate)
                    used.add(candidate["id"])
            if len(members) >= 2:
                used.add(anchor["id"])
                clusters.append({
                    "strategy": "type+keywords",
                    "source": "(mixed)",
                    "type": ctype,
                    "chunks": members,
                })
                for c in members:
                    assigned.add(c["id"])

    # Strategy 3: shared entities from memory graph
    if entity_map:
        remaining = [c for c in chunks if c["id"] not in assigned]
        entity_to_chunks: dict[str, list] = defaultdict(list)
        for chunk in remaining:
            src = chunk["metadata"].get("source_file", "")
            for entity in entity_map.get(src, set()):
                entity_to_chunks[entity].append(chunk)

        for entity, entity_chunks in entity_to_chunks.items():
            unassigned = [c for c in entity_chunks if c["id"] not in assigned]
            seen: set[str] = set()
            deduped = []
            for c in unassigned:
                if c["id"] not in seen:
                    deduped.append(c)
                    seen.add(c["id"])
            if len(deduped) >= 2:
                clusters.append({
                    "strategy": "entity",
                    "source": entity,
                    "type": "mixed",
                    "chunks": deduped,
                })
                for c in deduped:
                    assigned.add(c["id"])

    return clusters


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ollama_available(model: str, ollama_url: str) -> bool:
    tags_url = ollama_url.replace("/api/generate", "/api/tags")
    try:
        req = urllib.request.Request(tags_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            names = [m["name"] for m in data.get("models", [])]
            return any(
                model == n or model.split(":")[0] == n.split(":")[0]
                for n in names
            )
    except Exception:
        return False


def _pick_model(ollama_url: str) -> Optional[str]:
    for model in PREFERRED_MODELS:
        if _ollama_available(model, ollama_url):
            return model
    return None


def _call_ollama(prompt: str, model: str, ollama_url: str, max_tokens: int = 600) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception:
        return ""


def _compress_cluster(cluster: dict, model: str, ollama_url: str, dry_run: bool = False) -> Optional[str]:
    """Ask the LLM to compress a cluster into 1-3 permanent facts."""
    chunks = cluster["chunks"][:MAX_CHUNKS_PER_PROMPT]
    ctype = cluster["type"]
    strategy = cluster["strategy"]
    topic_hint = cluster["source"] if strategy != "type+keywords" else ctype

    memory_texts = []
    for i, c in enumerate(chunks, 1):
        text = c["text"][:MAX_CHUNK_CHARS]
        if len(c["text"]) > MAX_CHUNK_CHARS:
            text += "..."
        memory_texts.append(f"[{i}] {text}")

    combined = "\n\n".join(memory_texts)
    n = len(chunks)

    prompt = (
        f'You are a memory consolidation system. Below are {n} memory chunks '
        f'that all relate to: "{topic_hint}" (type: {ctype}).\n\n'
        "Your task: Extract 1 to 3 permanent facts, lessons, or status updates "
        "that subsume all of these memories. Be concrete and specific. Omit "
        "redundancy. Write in present tense where possible. Each fact should "
        "stand alone without needing the originals.\n\n"
        "Format your response as a numbered list. No preamble, no explanation "
        "— just the facts.\n\n"
        f"MEMORIES:\n{combined}\n\nCONSOLIDATED FACTS:"
    )

    if dry_run:
        return f"[DRY RUN] Would compress {n} chunks about '{topic_hint}'"

    response = _call_ollama(prompt, model, ollama_url, max_tokens=400)
    return response if response else None


def _decompose_to_facts(
    consolidated_text: str,
    parent_chunk_id: str,
    chunk_type: str,
    chunk_date: str,
    model: str,
    ollama_url: str,
) -> list[dict]:
    """Split consolidated text into atomic facts via LLM."""
    prompt = (
        "Split this text into individual atomic facts, one per line. "
        "Each fact must be a single, self-contained statement that makes sense on its own. "
        "Do not number the lines. Do not add any preamble or explanation. "
        "Just output the facts, one per line.\n\n"
        f"Text: {consolidated_text}\n\nFacts (one per line):"
    )
    response = _call_ollama(prompt, model, ollama_url, max_tokens=600)
    if not response:
        return []

    facts = []
    for line in response.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) *")
        line = line.strip()
        if len(line) > 10:
            facts.append({
                "text": line,
                "parent_chunk_id": parent_chunk_id,
                "type": chunk_type,
                "date": chunk_date,
            })
    return facts


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

class Consolidator:
    """Nightly lossy memory compression pipeline.

    Parameters
    ----------
    store:
        The MemoryStore to read from and write consolidated chunks into.
    ollama_url:
        Base generate URL for local Ollama. Default: ``http://localhost:11434/api/generate``.
    min_cluster:
        Minimum cluster size before compression runs. Default: 3.
    allowed_types:
        Restrict consolidation to these chunk types. Default: all consolidatable types.
    """

    def __init__(
        self,
        store: MemoryStore,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        min_cluster: int = DEFAULT_MIN_CLUSTER,
        allowed_types: Optional[list[str]] = None,
    ) -> None:
        self._store = store
        self._ollama_url = ollama_url
        self._min_cluster = min_cluster
        self._allowed_types = (
            [t for t in allowed_types if t in CONSOLIDATABLE_TYPES]
            if allowed_types
            else list(CONSOLIDATABLE_TYPES)
        )

    def run(
        self,
        dry_run: bool = False,
        entity_map: Optional[dict[str, set]] = None,
        verbose: bool = True,
    ) -> dict:
        """Run the full consolidation pipeline.

        Returns a report dict with keys:
          dry_run, model, eligible, clusters_found, clusters_viable,
          consolidated, archived, errors.
        """
        def _log(msg: str) -> None:
            if verbose:
                print(msg)

        report: dict = {
            "dry_run": dry_run,
            "model": None,
            "eligible": 0,
            "clusters_found": 0,
            "clusters_viable": 0,
            "consolidated": 0,
            "archived": 0,
            "errors": [],
        }

        # 1. Check model availability
        if not dry_run:
            model = _pick_model(self._ollama_url)
            if model is None:
                raise RuntimeError(
                    "Ollama is not available or no suitable model found. "
                    "Start Ollama and pull mistral:latest or qwen2.5:7b, "
                    "or run with dry_run=True."
                )
        else:
            model = PREFERRED_MODELS[0]
        report["model"] = model
        _log(f"Model: {model}{' (dry-run)' if dry_run else ''}")

        # 2. Load eligible chunks
        chunks = []
        for ctype in self._allowed_types:
            chunks.extend(self._store.get_all(type=ctype, include_archived=False))
        report["eligible"] = len(chunks)
        _log(f"Eligible chunks: {len(chunks)}")

        if not chunks:
            _log("Nothing to consolidate.")
            return report

        # 3. Cluster
        all_clusters = cluster_chunks(chunks, entity_map)
        viable = [c for c in all_clusters if len(c["chunks"]) >= self._min_cluster]
        report["clusters_found"] = len(all_clusters)
        report["clusters_viable"] = len(viable)
        _log(f"Clusters found: {len(all_clusters)}, viable: {len(viable)}")

        if not viable:
            _log(f"No clusters meet min_cluster={self._min_cluster}.")
            return report

        # 4. Process each viable cluster
        today = date.today().isoformat()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        for i, cluster in enumerate(viable, 1):
            n = len(cluster["chunks"])
            ctype = cluster["type"]
            _log(f"  [{i}/{len(viable)}] strategy={cluster['strategy']} type={ctype} chunks={n}")

            working = cluster["chunks"][:MAX_CHUNKS_PER_PROMPT]
            original_ids = [c["id"] for c in working]

            compressed = _compress_cluster(cluster, model, self._ollama_url, dry_run)
            if not compressed:
                report["errors"].append(f"Cluster {i}: LLM returned empty response")
                continue

            output_type = CONSOLIDATION_OUTPUT_TYPE.get(ctype, "fact")
            content_hash = hashlib.md5(compressed.encode()).hexdigest()[:12]
            new_id = f"consolidated::{today}::{content_hash}"

            metadata = {
                "source_file": "consolidation",
                "section_heading": f"Consolidated ({cluster['strategy']}: {cluster['source'][:60]})",
                "chunk_type": output_type,
                "date": today,
                "timestamp": now,
                "agent": "shared",
                "consolidated_from": json.dumps(original_ids[:20]),
                "n_originals": str(len(original_ids)),
                "consolidation_strategy": cluster["strategy"],
                "salience": "0.75",
            }

            if not dry_run:
                self._store.upsert_chunk(new_id, compressed, metadata)

                # Decompose into atomic facts
                facts = _decompose_to_facts(
                    compressed, new_id, output_type, today, model, self._ollama_url
                )
                for j, fact in enumerate(facts):
                    fact_id = f"fact::{new_id}::{j}"
                    self._store.upsert_fact(fact_id, fact["text"], {
                        "parent_chunk_id": new_id,
                        "chunk_type": output_type,
                        "date": today,
                        "fact_index": str(j),
                        "source": "decomposition",
                    })

                # Archive originals
                archive_meta = [{"consolidated_into": new_id, "archived_at": today}] * len(original_ids)
                self._store.update_metadata(original_ids, archive_meta)

            report["consolidated"] += 1
            report["archived"] += len(original_ids)
            _log(f"    Consolidated -> {new_id} (archived {len(original_ids)} originals)")

        return report
