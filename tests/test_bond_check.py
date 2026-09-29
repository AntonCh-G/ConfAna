"""Tests for confana/bond_check.py — bond-break detection with frame range."""

from __future__ import annotations

from pathlib import Path

import pytest

from confana.bond_check import find_bond_break_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_break_xyz(tmp_path: Path, break_at: int, n_frames: int = 10) -> Path:
    """Write a 2-atom C-C xyz trajectory where the bond breaks at *break_at*.

    Frames 0 … break_at-1: C-C distance 1.4 Å (bonded).
    Frames break_at … n_frames-1: C-C distance 5.0 Å (broken).
    """
    path = tmp_path / "break.xyz"
    lines: list[str] = []
    for f in range(n_frames):
        dist = 1.4 if f < break_at else 5.0
        lines.append("2\n")
        lines.append(f"frame {f}\n")
        lines.append(f"C  0.0  0.0  0.0\n")
        lines.append(f"C  {dist:.1f}  0.0  0.0\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_range_detects_break(tmp_path: Path):
    path = _make_break_xyz(tmp_path, break_at=5)
    assert find_bond_break_frame(path) == 5


def test_start_frame_before_break_still_detects(tmp_path: Path):
    """start_frame before break_at: break should still be found."""
    path = _make_break_xyz(tmp_path, break_at=5)
    assert find_bond_break_frame(path, start_frame=2) == 5


def test_start_frame_after_break_no_detection(tmp_path: Path):
    """start_frame after break: reference bond graph built from already-broken frame.

    No transition is observable so the function must return None.
    """
    path = _make_break_xyz(tmp_path, break_at=3, n_frames=10)
    # At frame 5 the pair is already at 5.0 Å — covalent radius check will find
    # no bonds in the reference frame → return None immediately.
    result = find_bond_break_frame(path, start_frame=5)
    assert result is None


def test_end_frame_before_break_no_detection(tmp_path: Path):
    """end_frame before break: scan stops before reaching the break."""
    path = _make_break_xyz(tmp_path, break_at=7, n_frames=10)
    assert find_bond_break_frame(path, end_frame=5) is None


def test_range_brackets_break(tmp_path: Path):
    """A range that includes the break must detect it."""
    path = _make_break_xyz(tmp_path, break_at=5, n_frames=10)
    assert find_bond_break_frame(path, start_frame=3, end_frame=7) == 5


def test_no_break_returns_none(tmp_path: Path):
    """File with no bond breaking must return None."""
    path = _make_break_xyz(tmp_path, break_at=999, n_frames=5)
    assert find_bond_break_frame(path) is None
