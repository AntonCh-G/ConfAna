"""Tests for confana/bond_check.py — bond-break detection on coordinate arrays.

How a break cuts a trajectory (frame range, beads) is tested through the
pipeline in tests/test_coordinate_table.py.
"""

from __future__ import annotations

import numpy as np

from confana.bond_check import bonded_pairs, first_broken_frame


def _cc_frames(break_at: int, n_frames: int = 10) -> np.ndarray:
    """A C–C pair 1.4 Å apart (bonded) that is 5.0 Å apart from *break_at* on."""
    coords = np.zeros((n_frames, 2, 3), dtype=np.float32)
    coords[:, 1, 0] = [1.4 if f < break_at else 5.0 for f in range(n_frames)]
    return coords


def test_bonded_pairs_of_a_bonded_frame():
    i, j = bonded_pairs(_cc_frames(break_at=5)[0], ["C", "C"])
    assert (i.tolist(), j.tolist()) == ([0], [1])


def test_bonded_pairs_of_a_broken_frame_is_empty():
    i, j = bonded_pairs(_cc_frames(break_at=0)[0], ["C", "C"])
    assert (i.tolist(), j.tolist()) == ([], [])


def test_first_broken_frame_finds_the_break():
    frames = _cc_frames(break_at=5)
    assert first_broken_frame(frames, bonded_pairs(frames[0], ["C", "C"]), cutoff=2.0) == 5


def test_first_broken_frame_without_a_break_is_none():
    frames = _cc_frames(break_at=999, n_frames=5)
    assert first_broken_frame(frames, bonded_pairs(frames[0], ["C", "C"]), cutoff=2.0) is None


def test_first_broken_frame_without_bonds_is_none():
    frames = _cc_frames(break_at=0)
    assert first_broken_frame(frames, bonded_pairs(frames[0], ["C", "C"]), cutoff=2.0) is None


def test_first_broken_frame_respects_the_cutoff():
    frames = _cc_frames(break_at=5)
    bonds = bonded_pairs(frames[0], ["C", "C"])
    assert first_broken_frame(frames, bonds, cutoff=6.0) is None
