"""Tests for confana/cache.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confana.cache import embed_meta, matches, read_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_npz(path: Path, meta: dict) -> Path:
    """Write a minimal NPZ with embedded cache metadata."""
    payload: dict = {"dummy": np.array([1, 2, 3])}
    embed_meta(payload, meta)
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# read_meta / matches
# ---------------------------------------------------------------------------


def test_read_meta_returns_none_for_missing_file(tmp_path):
    assert read_meta(tmp_path / "nonexistent.npz") is None


def test_read_meta_returns_none_for_npz_without_meta(tmp_path):
    npz_path = tmp_path / "no_meta.npz"
    np.savez(npz_path, data=np.array([1]))
    assert read_meta(npz_path) is None


def test_read_meta_round_trips_meta(tmp_path):
    meta = {"version": 1, "key": "value", "nested": {"a": 42}}
    npz_path = _write_npz(tmp_path / "test.npz", meta)
    assert read_meta(npz_path) == meta


def test_matches_returns_true_on_same_meta(tmp_path):
    meta = {"version": 1, "trajectory_id": "traj0"}
    npz_path = _write_npz(tmp_path / "cache.npz", meta)
    assert matches(npz_path, meta) is True


def test_matches_returns_false_on_changed_meta(tmp_path):
    meta = {"version": 1, "trajectory_id": "traj0"}
    npz_path = _write_npz(tmp_path / "cache.npz", meta)
    different = {"version": 1, "trajectory_id": "traj1"}
    assert matches(npz_path, different) is False


def test_matches_returns_false_missing_file(tmp_path):
    assert matches(tmp_path / "no_such_file.npz", {"version": 1}) is False


def test_matches_returns_false_for_extra_key(tmp_path):
    meta_written = {"version": 1}
    npz_path = _write_npz(tmp_path / "cache.npz", meta_written)
    assert matches(npz_path, {"version": 1, "extra": "unexpected"}) is False


def test_matches_returns_false_for_missing_key(tmp_path):
    meta_written = {"version": 1, "key": "value"}
    npz_path = _write_npz(tmp_path / "cache.npz", meta_written)
    assert matches(npz_path, {"version": 1}) is False
