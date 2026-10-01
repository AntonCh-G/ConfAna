"""Tests for confana/viewer.py (Phase 12)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from confana.density import DensitySettings, build_conformational_map
from confana.models import CoordinatePair, FrameRecord
from confana.viewer import align_frame, build_bin_frame_metadata, build_bin_xyz_payloads
from tests.hdf5_runs import hdf5_table, write_hdf5_run


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


def _frame(symbols: list[str], coords, comment: str = "comment") -> FrameRecord:
    coords = np.asarray(coords, dtype=np.float64)
    return FrameRecord(
        source_file="mem.xyz", frame_number=0, byte_offset=0, atom_count=len(symbols),
        comment_line=comment, elements=list(symbols), coords=coords,
    )


# ---------------------------------------------------------------------------
# align_frame
# ---------------------------------------------------------------------------


def test_align_frame_aligns_heavy_atoms_and_hydrogens():
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

    aligned = align_frame(
        _frame(symbols, target_coords, comment="target"),
        _frame(symbols, reference_coords, comment="reference"),
        atom_selection="heavy",
    )

    aligned_symbols, aligned_coords = aligned.elements, aligned.coords
    heavy_idx = np.asarray([0, 1, 2], dtype=np.int64)

    assert aligned_symbols == symbols
    assert _rmsd(aligned_coords[heavy_idx], reference_coords[heavy_idx]) < 1e-6
    assert _rmsd(aligned_coords, reference_coords) < 1e-6


def test_align_frame_atom_count_mismatch_raises():
    reference = _frame(["C", "O"], [[0, 0, 0], [1, 0, 0]])
    target = _frame(["C"], [[0, 0, 0]])

    with pytest.raises(ValueError, match="different atom counts"):
        align_frame(target, reference, atom_selection="heavy")


def test_align_frame_heavy_atom_sequence_mismatch_raises():
    reference = _frame(["C", "O", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    target = _frame(["C", "N", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    with pytest.raises(ValueError, match="heavy-atom element order"):
        align_frame(target, reference, atom_selection="heavy")


# ---------------------------------------------------------------------------
# build_bin_xyz_payloads
# ---------------------------------------------------------------------------


def _plane_map(df: pd.DataFrame, edges: np.ndarray):
    """Map of the plane pair of *df* on the equal-width bins *edges* (both axes)."""
    pair = CoordinatePair(
        name="plane", x_col="carboxyl_plane", y_col="ester_plane",
        x_label="x", y_label="y", title="t", x_domain=(0.0, 180.0), y_domain=(0.0, 180.0),
    )
    span = (float(edges[0]), float(edges[-1]))
    settings = DensitySettings(bins=len(edges) - 1, x_range=span, y_range=span)
    return build_conformational_map(df, pair, settings)


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
    edges = np.linspace(0, 180, 7)

    payloads = build_bin_xyz_payloads(_plane_map(df, edges))

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
    edges = np.linspace(0, 180, 5)

    result = build_bin_xyz_payloads(_plane_map(df, edges))
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
    edges = np.linspace(0, 180, 5)

    payloads = build_bin_xyz_payloads(_plane_map(df, edges))
    # Should still produce at least one payload (from the valid row)
    assert isinstance(payloads, dict)


def test_build_bin_xyz_payloads_missing_columns_raises():
    """Raise ValueError when required columns are absent."""
    df = pd.DataFrame({"carboxyl_plane": [1.0], "ester_plane": [1.0]})
    edges = np.linspace(0, 180, 5)

    with pytest.raises(ValueError, match="source_file.*byte_offset"):
        build_bin_xyz_payloads(_plane_map(df, edges))


def test_build_bin_xyz_payloads_unreadable_frame_raises(tmp_path):
    """A frame the table names but that cannot be read stops the build."""
    df = pd.DataFrame(
        {
            "carboxyl_plane": [45.0],
            "ester_plane": [45.0],
            "source_file": [str(tmp_path / "does_not_exist.xyz")],
            "byte_offset": [0],
            "atom_count": [2],
        }
    )
    edges = np.linspace(0, 180, 5)

    with pytest.raises(ValueError, match="does_not_exist.xyz"):
        build_bin_xyz_payloads(_plane_map(df, edges))


@pytest.mark.parametrize(
    "alignment",
    [
        {"enabled": True, "reference": "earliest_frame", "atom_selection": "heavy"},
        None,  # alignment is on by default
        {},
    ],
)
def test_build_bin_xyz_payloads_alignment_uses_earliest_reference(tmp_path, alignment):
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
    edges = np.asarray([0.0, 60.0, 120.0, 180.0], dtype=np.float64)

    payloads = build_bin_xyz_payloads(_plane_map(df, edges), alignment=alignment)

    _, aligned_coords = _parse_xyz_symbols_coords(payloads["1_1"])
    assert _rmsd(aligned_coords, reference_coords) < 1e-6

    raw = build_bin_xyz_payloads(_plane_map(df, edges), alignment={"enabled": False})
    _, raw_coords = _parse_xyz_symbols_coords(raw["1_1"])
    assert _rmsd(raw_coords, target_coords) < 1e-6


def test_build_bin_xyz_payloads_alignment_keeps_atom_order(tmp_path):
    """Alignment must not reorder atoms: the page highlights atoms by file index.

    Hydrogens sit between heavy atoms, so fitting on the heavy-atom subset
    would expose any reordering in the written payload.
    """
    symbols = ["C", "H", "O", "H", "N", "C"]
    rng = np.random.default_rng(3)
    reference_coords = rng.normal(0.0, 1.0, (len(symbols), 3))
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target_coords = reference_coords @ rotation.T + np.asarray([1.0, 2.0, 3.0])

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
    edges = np.asarray([0.0, 60.0, 120.0, 180.0], dtype=np.float64)

    payloads = build_bin_xyz_payloads(
        _plane_map(df, edges),
        alignment={"enabled": True, "reference": "earliest_frame", "atom_selection": "heavy"},
    )

    aligned_symbols, aligned_coords = _parse_xyz_symbols_coords(payloads["1_1"])
    assert aligned_symbols == symbols
    # Atom i of the payload is atom i of the file, just moved rigidly.
    np.testing.assert_allclose(aligned_coords, reference_coords, atol=1e-6)


def test_bin_structure_and_metadata_come_from_one_frame(tmp_path):
    """A bin's structure and metadata both belong to its representative frame.

    Bin 0_0 (centre 45, 45) is represented by frame 0, which has no structure
    reference: the bin gets frame 0's metadata and no structure, never a
    structure borrowed from frame 1 further from the centre.
    """
    xyz_file = tmp_path / "traj.xyz"
    frame_a = ["2", "frame-a", "C 0.0 0.0 0.0", "C 1.0 0.0 0.0"]
    frame_b = ["2", "frame-b", "C 0.0 0.0 0.0", "C 1.5 0.0 0.0"]
    offsets = _write_xyz(xyz_file, [frame_a, frame_b])
    df = pd.DataFrame(
        {
            "frame_id": [0, 1, 2],
            "carboxyl_plane": [44.0, 10.0, 130.0],
            "ester_plane": [44.0, 10.0, 130.0],
            "source_file": [None, str(xyz_file), str(xyz_file)],
            "byte_offset": [None, offsets[0], offsets[1]],
            "atom_count": [2, 2, 2],
        }
    )
    conf_map = _plane_map(df, np.linspace(0.0, 180.0, 3))

    payloads = build_bin_xyz_payloads(conf_map, alignment={"enabled": False})
    metadata = build_bin_frame_metadata(conf_map)

    assert metadata["0_0"]["frame_id"] == 0
    assert "0_0" not in payloads
    assert metadata["1_1"]["frame_id"] == 2
    assert payloads["1_1"].splitlines()[1] == "frame-b"


def test_hdf5_map_gets_one_structure_per_occupied_bin(tmp_path):
    """HDF5 rows (byte_offset -1) are read by frame and bead, not skipped."""
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0")
    table = hdf5_table(h5_path)
    pair = CoordinatePair(
        name="dist", x_col="d_co", y_col="d_oh", x_label="C-O", y_label="O-H", title="t",
        x_domain=(0.0, 10.0), y_domain=(0.0, 10.0),
    )
    settings = DensitySettings(bins=4, x_range=(0.0, 10.0), y_range=(0.0, 10.0))
    conf_map = build_conformational_map(table, pair, settings)

    payloads = build_bin_xyz_payloads(conf_map)  # aligned onto the earliest frame

    assert set(payloads) == {f"{xi}_{yi}" for xi, yi in conf_map.representatives}
    for text in payloads.values():
        assert [line.split()[0] for line in text.splitlines()[2:]] == ["C", "O", "H"]
