"""Memory retrieval eval harness: recall, precision, answer completeness."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .store import MemoryStore
from .types import EvalResult


# Default query suite (30 stratified queries from archon-memory-eval)

DEFAULT_EVAL_QUERIES: list[dict] = [
    # CREDENTIALS (5) — must never fail
    {"query": "Where is the database password stored", "expected_facts": ["keychain", "vault"], "type": "credential"},
    {"query": "What is the payment provider API key", "expected_facts": ["sk_live", "stripe"], "type": "credential"},
    {"query": "What is the cloud provider account ID", "expected_facts": ["account", "cloud"], "type": "credential"},
    {"query": "What is the project lead phone number", "expected_facts": ["555", "contact"], "type": "credential"},
    {"query": "What is the support team email address", "expected_facts": ["support@", "example.com"], "type": "credential"},

    # PROJECT STATUS (5) — should return recent info
    {"query": "What is the current status of the mobile app", "expected_facts": ["app store", "live"], "type": "project_status"},
    {"query": "What is the router validation result", "expected_facts": ["embedding", "93"], "type": "project_status"},
    {"query": "What is the benchmark ceiling result", "expected_facts": ["8.8", "ceiling"], "type": "project_status"},
    {"query": "Is the GPU node available for compute right now", "expected_facts": ["available", "node"], "type": "project_status"},
    {"query": "How many dataset records do we have", "expected_facts": ["54", "50"], "type": "project_status"},

    # LESSONS (5) — should return corrections/mistakes
    {"query": "What went wrong with the validation experiment judge", "expected_facts": ["mixed", "judge", "contaminated"], "type": "lesson"},
    {"query": "Why did the pipeline have a high failure rate", "expected_facts": ["install", "404"], "type": "lesson"},
    {"query": "What is the rule about judge selection for experiments", "expected_facts": ["single judge", "no mix"], "type": "lesson"},
    {"query": "What should happen before any data experiment", "expected_facts": ["preflight"], "type": "lesson"},
    {"query": "Should the agent ask permission for clearly autonomous work", "expected_facts": ["don't ask", "just do"], "type": "lesson"},

    # PERSONAL (5) — should know the user
    {"query": "How many people are on the core team", "expected_facts": ["6", "six"], "type": "personal"},
    {"query": "What are the names of the founding team members", "expected_facts": ["alice", "bob", "carol"], "type": "personal"},
    {"query": "Where does the project lead work and what is their title", "expected_facts": ["acme", "director"], "type": "personal"},
    {"query": "Where is the company headquarters located", "expected_facts": ["portland", "oregon"], "type": "personal"},
    {"query": "What is the engineering lead name", "expected_facts": ["jordan"], "type": "personal"},

    # TECHNICAL (5) — infrastructure knowledge
    {"query": "What is the secondary node IP address", "expected_facts": ["10.0.0.", "192.168."], "type": "technical"},
    {"query": "What does the Pipeline Sentinel do", "expected_facts": ["monitor", "pipeline", "anomaly"], "type": "technical"},
    {"query": "How many workers are in the agent fleet", "expected_facts": ["25", "24", "worker"], "type": "technical"},
    {"query": "What LLM does the summarizer use", "expected_facts": ["mistral", "ollama"], "type": "technical"},
    {"query": "What nodes are in the agent fleet", "expected_facts": ["primary", "secondary", "worker"], "type": "technical"},

    # SESSION (5) — should prioritize new over old
    {"query": "What are the two product tracks we decided on", "expected_facts": ["control tower", "edge ai"], "type": "session"},
    {"query": "What is the voting format for the model council", "expected_facts": ["score", "rubric", "json"], "type": "session"},
    {"query": "What is the party game app concept", "expected_facts": ["world cup", "party game", "funnel"], "type": "session"},
    {"query": "What new models should the inference node install", "expected_facts": ["qwen", "llama"], "type": "session"},
    {"query": "What is the memory system rated out of 10", "expected_facts": ["5"], "type": "session"},
]


# MemoryEval

class MemoryEval:
    """Retrieval quality evaluation harness.

    Parameters
    ----------
    store:
        The MemoryStore to query.
    history_path:
        Where to persist eval history as a JSON array.
        Defaults to ``~/.agent-memory-core/eval-history.json``.
    queries:
        Initial query list. Defaults to ``DEFAULT_EVAL_QUERIES``.
        Each query dict needs: query (str), expected_facts (list[str]), type (str).
    """

    def __init__(
        self,
        store: MemoryStore,
        history_path: Optional[str | Path] = None,
        queries: Optional[list[dict]] = None,
    ) -> None:
        self._store = store
        self._history_path = (
            Path(history_path)
            if history_path
            else Path.home() / ".agent-memory-core" / "eval-history.json"
        )
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._queries: list[dict] = list(queries) if queries is not None else list(DEFAULT_EVAL_QUERIES)

    # Query management

    def add_query(
        self,
        query: str,
        expected_facts: list[str],
        type: str = "fact",
    ) -> None:
        """Add a custom query to the evaluation suite."""
        self._queries.append({
            "query": query,
            "expected_facts": expected_facts,
            "type": type,
        })

    def clear_queries(self) -> None:
        """Remove all queries (use before adding a fully custom suite)."""
        self._queries.clear()

    # Scoring

    @staticmethod
    def score_query(query_def: dict, results: list) -> EvalResult:
        """Score a single query against a list of MemoryResult objects.

        Parameters
        ----------
        query_def:
            Dict with keys: query, expected_facts, type.
        results:
            List of MemoryResult instances (or dicts with a 'text' key).

        Returns
        -------
        EvalResult
        """
        expected = query_def["expected_facts"]

        def _text(r) -> str:
            return r.text if hasattr(r, "text") else r.get("text", "")

        if not results:
            return EvalResult(
                query=query_def["query"],
                query_type=query_def.get("type", "unknown"),
                recall=False,
                precision=0.0,
                answer=False,
                results_count=0,
            )

        recall = any(
            any(f.lower() in _text(r).lower() for f in expected)
            for r in results
        )

        relevant = sum(
            1 for r in results
            if any(f.lower() in _text(r).lower() for f in expected)
        )
        precision = relevant / len(results)

        all_text = " ".join(_text(r) for r in results).lower()
        answer = all(f.lower() in all_text for f in expected)

        return EvalResult(
            query=query_def["query"],
            query_type=query_def.get("type", "unknown"),
            recall=recall,
            precision=precision,
            answer=answer,
            results_count=len(results),
        )

    # Run

    def run(
        self,
        n: int = 5,
        version: str = "",
        verbose: bool = True,
        save: bool = True,
    ) -> dict:
        """Run the full evaluation suite.

        Parameters
        ----------
        n:         Number of results to retrieve per query.
        version:   Optional label for this eval run (e.g. "v0.2-reranker").
        verbose:   Print per-query status to stdout.
        save:      Persist results to history file.

        Returns
        -------
        dict with keys:
          date, version, recall (%), precision (%), answer (%),
          composite (/10), by_type (dict), failures (list[str]),
          results (list[EvalResult])
        """
        results_by_type: dict[str, list[EvalResult]] = defaultdict(list)
        all_results: list[EvalResult] = []
        failures: list[str] = []

        for q in self._queries:
            search_results = self._store.search(q["query"], n=n)
            result = self.score_query(q, search_results)
            all_results.append(result)
            results_by_type[q["type"]].append(result)

            if verbose:
                status = "[PASS]" if result.recall else "[FAIL]"
                print(f"  {status} [{q['type'][:8]:>8s}] {q['query'][:55]}", end="")
                if not result.recall:
                    print(" <- MISS", end="")
                    failures.append(q["query"])
                print()

        total = len(all_results)
        recall_pct = sum(1 for r in all_results if r.recall) / total * 100
        precision_avg = sum(r.precision for r in all_results) / total * 100
        answer_pct = sum(1 for r in all_results if r.answer) / total * 100
        composite = (0.4 * recall_pct + 0.3 * precision_avg + 0.3 * answer_pct) / 10

        type_scores: dict[str, float] = {}
        for qtype, scores in results_by_type.items():
            type_recall = sum(1 for s in scores if s.recall) / len(scores) * 100
            type_scores[qtype] = round(type_recall, 1)

        if verbose:
            print(f"\n  Recall@{n}:    {recall_pct:.0f}% ({sum(1 for r in all_results if r.recall)}/{total})")
            print(f"  Precision@{n}: {precision_avg:.0f}%")
            print(f"  Answer:      {answer_pct:.0f}% ({sum(1 for r in all_results if r.answer)}/{total})")
            print(f"\n  COMPOSITE:   {composite:.1f}/10")

            if type_scores:
                weakest = min(type_scores, key=lambda k: type_scores[k])
                print(f"\n  Weakest type: {weakest} ({type_scores[weakest]:.0f}%)")

        entry = {
            "date": datetime.now().isoformat(),
            "version": version or "unnamed",
            "recall": round(recall_pct, 1),
            "precision": round(precision_avg, 1),
            "answer": round(answer_pct, 1),
            "composite": round(composite, 1),
            "by_type": type_scores,
            "failures": failures,
            "results": all_results,
        }

        if save:
            self._append_history({k: v for k, v in entry.items() if k != "results"})

        return entry

    # History

    def _append_history(self, entry: dict) -> None:
        history = []
        if self._history_path.exists():
            try:
                history = json.loads(self._history_path.read_text())
            except Exception:
                history = []
        history.append(entry)
        self._history_path.write_text(json.dumps(history, indent=2))

    def history(self) -> list[dict]:
        """Return the full eval history sorted by date ascending."""
        if not self._history_path.exists():
            return []
        try:
            return json.loads(self._history_path.read_text())
        except Exception:
            return []

    def best_score(self) -> Optional[dict]:
        """Return the history entry with the highest composite score."""
        h = self.history()
        if not h:
            return None
        return max(h, key=lambda x: x.get("composite", 0))

    def score_delta(self) -> Optional[float]:
        """Return composite score improvement from first to last run. None if < 2 runs."""
        h = self.history()
        if len(h) < 2:
            return None
        return round(h[-1].get("composite", 0) - h[0].get("composite", 0), 1)
