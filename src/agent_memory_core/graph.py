"""Entity relationship graph over memory files. Supports Ollama and Gemini extraction."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .types import GraphNode


# LLM helpers

def _call_ollama_json(
    prompt: str,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "mistral:latest",
    max_tokens: int = 512,
) -> Optional[dict]:
    """Call local Ollama requesting JSON output. Returns parsed dict or None."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "").strip()
            return json.loads(text)
    except Exception:
        return None


def _call_gemini_json(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 512,
) -> Optional[dict]:
    """Call Gemini Flash for entity/topic extraction. Returns parsed dict or None."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingBudget": 0},
            "maxOutputTokens": max_tokens,
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            text = "".join(
                p.get("text", "") for p in parts if not p.get("thought", False)
            )
            return json.loads(text.strip())
    except Exception:
        return None


# Node extraction

def _file_id(filepath: Path) -> str:
    return hashlib.md5(str(filepath).encode()).hexdigest()[:12]


def _parse_frontmatter(filepath: Path) -> tuple[dict, str]:
    """Extract YAML frontmatter from a markdown file."""
    try:
        content = filepath.read_text()
        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                fm_text = content[3:end].strip()
                fm: dict = {}
                for line in fm_text.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        fm[key.strip()] = val.strip().strip('"').strip("'")
                return fm, content[end + 3:].strip()
        return {}, content
    except Exception:
        return {}, ""


def _extract_node(filepath: Path) -> dict:
    """Build a raw node dict from a file (without entity enrichment)."""
    fm, content = _parse_frontmatter(filepath)
    fm_type = fm.get("type", "unknown")
    if fm_type in ("user", "feedback"):
        node_type, confidence = "EXTRACTED", 1.0
    elif fm_type in ("project", "reference"):
        node_type, confidence = "INFERRED", 0.7
    else:
        node_type, confidence = "INFERRED", 0.6

    return {
        "id": _file_id(filepath),
        "source_file": str(filepath),
        "title": fm.get("name", filepath.stem),
        "summary": fm.get("description", content[:200]),
        "_content_preview": content[:500],  # used during enrichment, stripped before save
        "type": node_type,
        "domain": fm_type,
        "confidence": confidence,
        "last_modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
        "entities": [],
        "topics": [],
        "relationships": [],
    }


# Relationship detection

def _to_str(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("name", x.get("entity", x.get("topic", str(x)))))
    return str(x)


def _find_relationships(nodes: list[dict]) -> list[dict]:
    """Find edges between nodes sharing entities or topics."""
    entity_index: dict[str, list[str]] = {}
    topic_index: dict[str, list[str]] = {}

    for node in nodes:
        nid = node["id"]
        for e in node.get("entities", []):
            key = _to_str(e).lower()
            entity_index.setdefault(key, []).append(nid)
        for t in node.get("topics", []):
            key = _to_str(t).lower()
            topic_index.setdefault(key, []).append(nid)

    pairs: set[tuple[str, str]] = set()
    for ids in (*entity_index.values(), *topic_index.values()):
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.add((ids[i], ids[j]))

    node_map = {n["id"]: n for n in nodes}
    relationships = []

    for id_a, id_b in pairs:
        a = node_map.get(id_a)
        b = node_map.get(id_b)
        if not a or not b:
            continue

        shared_e = (
            set(_to_str(e).lower() for e in a.get("entities", []))
            & set(_to_str(e).lower() for e in b.get("entities", []))
        )
        shared_t = (
            set(_to_str(t).lower() for t in a.get("topics", []))
            & set(_to_str(t).lower() for t in b.get("topics", []))
        )
        overlap = len(shared_e) + len(shared_t)
        max_possible = max(len(a.get("entities", [])) + len(a.get("topics", [])), 1)
        weight = min(overlap / max_possible, 1.0)

        if weight <= 0.1:
            continue

        if a.get("domain") == "feedback" or b.get("domain") == "feedback":
            rel_type = "extends"
        elif "correction" in a.get("source_file", "").lower() or "correction" in b.get("source_file", "").lower():
            rel_type = "contradicts"
        else:
            rel_type = "co-occurs"

        relationships.append({
            "source": id_a,
            "target": id_b,
            "type": rel_type,
            "weight": round(weight, 3),
            "shared": list(shared_e | shared_t),
        })

    return relationships


# MemoryGraph

class MemoryGraph:
    """Entity relationship graph over a set of memory files.

    Parameters
    ----------
    graph_path:
        Where to persist the graph JSON.
        Defaults to ``~/.agent-memory-core/memory_graph.json``.
    ollama_url:
        Ollama generate endpoint for entity extraction.
    ollama_model:
        Local model to use for enrichment. Default: mistral:latest.
    gemini_api_key:
        Gemini API key for fallback if Ollama is unavailable.
    """

    def __init__(
        self,
        graph_path: Optional[str | Path] = None,
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "mistral:latest",
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self._graph_path = (
            Path(graph_path)
            if graph_path
            else Path.home() / ".agent-memory-core" / "memory_graph.json"
        )
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._gemini_key = gemini_api_key
        self._graph: Optional[dict] = None

    # LLM enrichment

    def _enrich_node(self, node: dict) -> dict:
        """Use local LLM (with Gemini fallback) to extract entities and topics."""
        prompt = (
            f"Analyze this memory file and extract entities and topics.\n\n"
            f"Title: {node['title']}\n"
            f"Content: {node['_content_preview']}\n\n"
            'Return JSON: {"entities": ["list of named entities: people, projects, tools, companies"], '
            '"topics": ["list of 3-5 key topics"]}'
        )

        result = _call_ollama_json(prompt, self._ollama_url, self._ollama_model)
        if result is None and self._gemini_key:
            result = _call_gemini_json(prompt, self._gemini_key)

        if result:
            node["entities"] = result.get("entities", [])
            node["topics"] = result.get("topics", [])
        return node

    # Build / load

    def build(self, source_paths: list[str | Path], verbose: bool = True) -> dict:
        """Scan ``source_paths`` for markdown files, extract nodes, find relationships.

        Parameters
        ----------
        source_paths:
            List of file paths or directories to scan. Directories are searched
            recursively for ``*.md`` files.
        verbose:
            Print progress to stdout.

        Returns
        -------
        dict: The full graph with keys: version, built_at, node_count, edge_count, nodes, edges.
        """
        def _log(msg: str) -> None:
            if verbose:
                print(msg)

        # Collect files
        files: list[Path] = []
        for p in source_paths:
            p = Path(p)
            if p.is_file() and p.suffix == ".md":
                files.append(p)
            elif p.is_dir():
                for f in p.rglob("*.md"):
                    if not f.name.startswith(".") and "node_modules" not in str(f):
                        files.append(f)

        _log(f"Found {len(files)} markdown files")

        # Extract nodes
        nodes = [_extract_node(f) for f in files]
        _log(f"Extracting entities from {len(nodes)} nodes...")

        for i, node in enumerate(nodes):
            node = self._enrich_node(node)
            nodes[i] = node
            if verbose and (i + 1) % 10 == 0:
                _log(f"  {i + 1}/{len(nodes)} enriched")

        # Find relationships
        relationships = _find_relationships(nodes)
        _log(f"Found {len(relationships)} relationships")

        # Assign relationships to nodes
        for rel in relationships:
            for node in nodes:
                if node["id"] == rel["source"]:
                    node["relationships"].append({
                        "target_id": rel["target"],
                        "type": rel["type"],
                        "weight": rel["weight"],
                    })
                elif node["id"] == rel["target"]:
                    node["relationships"].append({
                        "target_id": rel["source"],
                        "type": rel["type"],
                        "weight": rel["weight"],
                    })

        # Strip content preview before saving (size reduction)
        for node in nodes:
            node.pop("_content_preview", None)

        graph = {
            "version": "1.0",
            "built_at": datetime.now().isoformat(),
            "node_count": len(nodes),
            "edge_count": len(relationships),
            "nodes": {n["id"]: n for n in nodes},
            "edges": relationships,
        }

        self._graph_path.write_text(json.dumps(graph, indent=2, default=str))
        self._graph = graph
        _log(f"Graph saved to {self._graph_path}")
        return graph

    def load(self) -> Optional[dict]:
        """Load graph from disk. Returns None if not yet built."""
        if self._graph is not None:
            return self._graph
        if not self._graph_path.exists():
            return None
        try:
            self._graph = json.loads(self._graph_path.read_text())
            return self._graph
        except Exception:
            return None

    # Search

    def search(self, query: str, top_n: int = 5) -> dict:
        """Hybrid keyword + 2-hop graph neighbor search.

        Returns a dict with keys:
          direct: list of (score, node) for direct keyword matches
          neighbors: list of neighbor nodes (1-2 hops from direct matches)
        """
        graph = self.load()
        if not graph:
            return {"direct": [], "neighbors": []}

        query_terms = set(query.lower().split())
        scored = []

        for nid, node in graph["nodes"].items():
            text = " ".join([
                node.get("title", ""),
                node.get("summary", ""),
                " ".join(_to_str(e) for e in node.get("entities", [])),
                " ".join(_to_str(t) for t in node.get("topics", [])),
            ]).lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, nid, node))

        scored.sort(key=lambda x: -x[0])
        direct = scored[:top_n]

        expanded: set[str] = set()
        for _, nid, _ in direct:
            node = graph["nodes"][nid]
            for rel in node.get("relationships", []):
                expanded.add(rel["target_id"])
                neighbor = graph["nodes"].get(rel["target_id"])
                if neighbor:
                    for rel2 in neighbor.get("relationships", []):
                        expanded.add(rel2["target_id"])

        direct_ids = {nid for _, nid, _ in direct}
        expanded -= direct_ids

        neighbor_nodes = [
            graph["nodes"][eid]
            for eid in list(expanded)[:top_n]
            if eid in graph["nodes"]
        ]

        return {
            "direct": [(score, node) for score, _, node in direct],
            "neighbors": neighbor_nodes,
        }

    # Stats & maintenance

    def stats(self) -> dict:
        """Return summary statistics about the graph."""
        graph = self.load()
        if not graph:
            return {"built": False}

        nodes = graph["nodes"]
        edges = graph["edges"]

        type_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        for n in nodes.values():
            t = n.get("type", "?")
            d = n.get("domain", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
            domain_counts[d] = domain_counts.get(d, 0) + 1

        rel_types: dict[str, int] = {}
        for e in edges:
            rt = e.get("type", "?")
            rel_types[rt] = rel_types.get(rt, 0) + 1

        connections = {
            nid: len(n.get("relationships", []))
            for nid, n in nodes.items()
        }
        top_connected = sorted(connections.items(), key=lambda x: -x[1])[:5]

        confs = [n.get("confidence", 0) for n in nodes.values()]
        avg_confidence = sum(confs) / len(confs) if confs else 0.0

        return {
            "built": True,
            "built_at": graph.get("built_at"),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "by_type": type_counts,
            "by_domain": domain_counts,
            "relationship_types": rel_types,
            "top_connected": [
                {"id": nid, "title": nodes[nid].get("title", "?"), "connections": count}
                for nid, count in top_connected
            ],
            "avg_confidence": round(avg_confidence, 2),
        }

    def contradictions(self) -> list[dict]:
        """Return all contradiction edges in the graph."""
        graph = self.load()
        if not graph:
            return []
        return [e for e in graph["edges"] if e.get("type") == "contradicts"]

    def entity_map(self) -> dict[str, set]:
        """Return {source_file: set_of_entity_strings} for use in Consolidator clustering."""
        graph = self.load()
        if not graph:
            return {}
        result: dict[str, set] = {}
        for node in graph["nodes"].values():
            src = node.get("source_file", "")
            if src:
                entities = set(_to_str(e).lower() for e in node.get("entities", []))
                result[src] = result.get(src, set()) | entities
        return result
