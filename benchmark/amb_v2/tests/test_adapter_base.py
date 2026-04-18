"""T-15, T-16 — DecayAdapter Protocol + smoke test framework."""
from __future__ import annotations

import pytest

from benchmark.amb_v2.adapters.base import (
    DecayAdapter,
    METADATA_REQUIRED_KEYS,
    validate_metadata,
)
from benchmark.amb_v2.adapters.naive import NaiveAppendOnlyAdapter
from benchmark.amb_v2.chunks import Chunk


def test_protocol_required_methods():
    assert "ingest" in dir(DecayAdapter)
    assert "consolidate" in dir(DecayAdapter)
    assert "query" in dir(DecayAdapter)
    assert "metadata" in dir(DecayAdapter)


def test_naive_satisfies_protocol():
    a = NaiveAppendOnlyAdapter()
    assert isinstance(a, DecayAdapter)


def test_metadata_required_keys():
    assert METADATA_REQUIRED_KEYS == frozenset({"name", "version", "implements_consolidation"})


def test_validate_metadata_accepts_valid():
    validate_metadata({"name": "x", "version": "0.1", "implements_consolidation": False})


def test_validate_metadata_rejects_missing_key():
    with pytest.raises(ValueError, match="missing"):
        validate_metadata({"name": "x", "version": "0.1"})


def test_validate_metadata_rejects_wrong_type():
    with pytest.raises(ValueError, match="bool"):
        validate_metadata({"name": "x", "version": "0.1", "implements_consolidation": "yes"})


def test_naive_metadata_valid():
    a = NaiveAppendOnlyAdapter()
    validate_metadata(a.metadata)
