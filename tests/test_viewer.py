"""Tests for src/viewer.py (Phase 12)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.viewer import align_xyz_to_reference, build_bin_xyz_payloads, read_xyz_frame_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_xyz(path: Path, frames: list[list[str]]) -> list[int]:
    """Write multi-frame xyz to *path*; return list of byte offsets."""
    offsets: list[int] = []
    with open(path, "w", encoding="utf-8") as fh:
        for lines in frames:
            offsets.append(fh.tell() if hasattr(fh, "tell") else 0)
            fh.write("\n".join(lines) + "\n")

    # Re-derive offsets in binary mode (consistent with io_xyz.py)
    offsets_bytes: list[int] = []
    with open(path, "rb") as fh:
        for frame_lines in frames:
            offsets_bytes.append(fh.tell())
            for _ in frame_lines:
                fh.readline()
    return offsets_bytes


def _minimal_xyz_frame(n_atoms: int = 2) -> list[str]:
    lines = [str(n_atoms), "comment"]
    for i in range(n_atoms):
        lines.append(f"C  {i:.3f}  0.000  0.000")
    return lines


def _xyz_text_from_symbols_coords(
    symbols: list[str],
    coords: np.ndarray,
    comment: str = "comment",
) -> str:
    lines = [str(len(symbols)), comment]
    for symbol, (x, y, z) in zip(symbols, coords):
        lines.append(f"{symbol:<2s}  {x: .8f}  {y: .8f}  {z: .8f}")
    return "\n".join(lines) + "\n"


def _parse_xyz_symbols_coords(xyz_text: str) -> tuple[list[str], np.ndarray]:
    lines = xyz_text.strip().splitlines()
    atom_count = int(lines[0])
    symbols: list[str] = []
    coords = np.empty((atom_count, 3), dtype=np.float64)
    for i, line in enumerate(lines[2: 2 + atom_count]):
        parts = line.split()
        symbols.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return symbols, coords


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    diff = left - right
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# ---------------------------------------------------------------------------
# read_xyz_frame_text
# ---------------------------------------------------------------------------


def test_read_xyz_frame_text_round_trip(tmp_path):
    """Read back the exact text written to the file."""
    frame_lines = _minimal_xyz_frame(3)
    xyz_file = tmp_path / "test.xyz"
    offsets = _write_xyz(xyz_file, [frame_lines])

    text = read_xyz_frame_text(xyz_file, offsets[0], atom_count=3)
    returned_lines = [l.strip() for l in text.strip().splitlines()]

    # Header line should be atom count
    assert returned_lines[0] == "3"
    # Comment line
    assert returned_lines[1] == "comment"
    # Atom lines
    assert len(returned_lines) == 3 + 2


def test_read_xyz_frame_text_second_frame(tmp_path):
    """Correctly seeks to the second frame, not the first."""
    frame0 = _minimal_xyz_frame(2)
    frame1 = [str(2), "frame1_comment", "C  1.0  0.0  0.0", "O  2.0  0.0  0.0"]
    xyz_file = tmp_path / "multi.xyz"
    offsets = _write_xyz(xyz_file, [frame0, frame1])

    text = read_xyz_frame_text(xyz_file, offsets[1], atom_count=2)
    assert "frame1_comment" in text


def test_read_xyz_frame_text_bad_offset_raises(tmp_path):
    """Seeking past EOF should raise ValueError."""
    xyz_file = tmp_path / "tiny.xyz"
    xyz_file.write_text("2\ncomment\nC 0 0 0\nO 1 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_xyz_frame_text(xyz_file, byte_offset=10_000, atom_count=2)


def test_read_xyz_frame_text_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        read_xyz_frame_text(tmp_path / "nonexistent.xyz", 0, 2)


# ---------------------------------------------------------------------------
# align_xyz_to_reference
# ---------------------------------------------------------------------------


def test_align_xyz_to_reference_aligns_heavy_atoms_and_hydrogens():
    """Rigid alignment uses heavy atoms for the fit and moves hydrogens too."""
    symbols = ["C", "C", "O", "H", "H"]
    reference_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.3, 0.2, 0.1],
            [0.4, 1.1, -0.2],
            [-0.5, -0.2, 0.8],
            [1.2, 0.9, 0.7],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation = np.asarray([3.5, -2.0, 1.25], dtype=np.float64)
    target_coords = reference_coords @ rotation.T + translation

    aligned_text = align_xyz_to_reference(
        _xyz_text_from_symbols_coords(symbols, target_coords, comment="target"),
        _xyz_text_from_symbols_coords(symbols, reference_coords, comment="reference"),
        atom_selection="heavy",
    )

    aligned_symbols, aligned_coords = _parse_xyz_symbols_coords(aligned_text)
    heavy_idx = np.asarray([0, 1, 2], dtype=np.int64)

    assert aligned_symbols == symbols
    assert _rmsd(aligned_coords[heavy_idx], reference_coords[heavy_idx]) < 1e-6
    assert _rmsd(aligned_coords, reference_coords) < 1e-6


def test_align_xyz_to_reference_atom_count_mismatch_raises():
    ref_text = _xyz_text_from_symbols_coords(["C", "O"], np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float))
    target_text = _xyz_text_from_symbols_coords(["C"], np.asarray([[0, 0, 0]], dtype=float))

    with pytest.raises(ValueError, match="different atom counts"):
        align_xyz_to_reference(target_text, ref_text, atom_selection="heavy")


def test_align_xyz_to_reference_heavy_atom_sequence_mismatch_raises():
    ref_text = _xyz_text_from_symbols_coords(
        ["C", "O", "H"],
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
    )
    target_text = _xyz_text_from_symbols_coords(
        ["C", "N", "H"],
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
    )

    with pytest.raises(ValueError, match="heavy-atom element order"):
        align_xyz_to_reference(target_text, ref_text, atom_selection="heavy")


# ---------------------------------------------------------------------------
# build_bin_xyz_payloads
# ---------------------------------------------------------------------------


def _make_df(n: int, tmp_xyz: Path, offsets: list[int], n_atoms: int = 2) -> pd.DataFrame:
    """Return a minimal coordinate DataFrame pointing to *tmp_xyz*."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "carboxyl_plane": rng.uniform(0, 180, n).astype(np.float32),
            "ester_plane": rng.uniform(0, 180, n).astype(np.float32),
            "source_file": [str(tmp_xyz)] * n,
            "byte_offset": [offsets[i % len(offsets)] for i in range(n)],
            "atom_count": [n_atoms] * n,
        }
    )


def test_build_bin_xyz_payloads_returns_dict(tmp_path):
    """Payloads dict is non-empty when source files are readable."""
    n_atoms = 2
    frames = [_minimal_xyz_frame(n_atoms) for _ in range(10)]
    xyz_file = tmp_path / "traj.xyz"
    offsets = _write_xyz(xyz_file, frames)

    df = _make_df(10, xyz_file, offsets, n_atoms=n_atoms)
    x_edges = np.linspace(0, 180, 7)
    y_edges = np.linspace(0, 180, 7)

    payloads = build_bin_xyz_payloads(df, "carboxyl_plane", "ester_plane", x_edges, y_edges)

    assert isinstance(payloads, dict)
    assert len(payloads) > 0
    # All values should be non-empty strings
    for key, text in payloads.items():
        assert "_" in key
        assert len(text) > 0


def test_build_bin_xyz_payloads_empty_df():
    """Empty DataFrame returns empty dict without error."""
    df = pd.DataFrame(
        columns=["carboxyl_plane", "ester_plane", "source_file", "byte_offset", "atom_count"]
    )
    x_edges = np.linspace(0, 180, 5)
    y_edges = np.linspace(0, 180, 5)

    result = build_bin_xyz_payloads(df, "carboxyl_plane", "ester_plane", x_edges, y_edges)
    assert result == {}


def test_build_bin_xyz_payloads_skips_missing_offset(tmp_path):
    """Rows with NA byte_offset are silently skipped."""
    n_atoms = 2
    frames = [_minimal_xyz_frame(n_atoms)]
    xyz_file = tmp_path / "traj.xyz"
    offsets = _write_xyz(xyz_file, frames)

    df = pd.DataFrame(
        {
            "carboxyl_plane": [10.0, 90.0],
            "ester_plane": [10.0, 90.0],
            "source_file": [str(xyz_file), str(xyz_file)],
            "byte_offset": [offsets[0], None],   # second row has no offset
            "atom_count": [n_atoms, n_atoms],
        }
    )
    x_edges = np.linspace(0, 180, 5)
    y_edges = np.linspace(0, 180, 5)

    payloads = build_bin_xyz_payloads(df, "carboxyl_plane", "ester_plane", x_edges, y_edges)
    # Should still produce at least one payload (from the valid row)
    assert isinstance(payloads, dict)


def test_build_bin_xyz_payloads_missing_columns_raises():
    """Raise ValueError when required columns are absent."""
    df = pd.DataFrame({"carboxyl_plane": [1.0], "ester_plane": [1.0]})
    x_edges = np.linspace(0, 180, 5)
    y_edges = np.linspace(0, 180, 5)

    with pytest.raises(ValueError, match="Missing columns"):
        build_bin_xyz_payloads(df, "carboxyl_plane", "ester_plane", x_edges, y_edges)


def test_build_bin_xyz_payloads_unreadable_skipped(tmp_path):
    """Unreadable source files are skipped without raising."""
    df = pd.DataFrame(
        {
            "carboxyl_plane": [45.0],
            "ester_plane": [45.0],
            "source_file": [str(tmp_path / "does_not_exist.xyz")],
            "byte_offset": [0],
            "atom_count": [2],
        }
    )
    x_edges = np.linspace(0, 180, 5)
    y_edges = np.linspace(0, 180, 5)

    result = build_bin_xyz_payloads(df, "carboxyl_plane", "ester_plane", x_edges, y_edges)
    assert result == {}


def test_build_bin_xyz_payloads_alignment_uses_earliest_reference(tmp_path):
    """Aligned payloads are transformed into the earliest-frame reference orientation."""
    symbols = ["C", "C", "O", "H", "H"]
    reference_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.5, 1.0, -0.1],
            [-0.4, -0.3, 0.7],
            [1.0, 0.8, 0.6],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    translation = np.asarray([2.0, -1.5, 0.5], dtype=np.float64)
    target_coords = reference_coords @ rotation.T + translation

    xyz_file = tmp_path / "traj.xyz"
    offsets = _write_xyz(
        xyz_file,
        [
            _xyz_text_from_symbols_coords(symbols, reference_coords, comment="ref").strip().splitlines(),
            _xyz_text_from_symbols_coords(symbols, target_coords, comment="target").strip().splitlines(),
        ],
    )

    df = pd.DataFrame(
        {
            "carboxyl_plane": [10.0, 100.0],
            "ester_plane": [10.0, 100.0],
            "source_file": [str(xyz_file), str(xyz_file)],
            "byte_offset": offsets,
            "atom_count": [len(symbols), len(symbols)],
        }
    )
    x_edges = np.asarray([0.0, 60.0, 120.0, 180.0], dtype=np.float64)
    y_edges = np.asarray([0.0, 60.0, 120.0, 180.0], dtype=np.float64)

    payloads = build_bin_xyz_payloads(
        df,
        "carboxyl_plane",
        "ester_plane",
        x_edges,
        y_edges,
        alignment={"enabled": True, "reference": "earliest_frame", "atom_selection": "heavy"},
    )

    _, aligned_coords = _parse_xyz_symbols_coords(payloads["1_1"])
    assert _rmsd(aligned_coords, reference_coords) < 1e-6
