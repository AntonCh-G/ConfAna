"""Tests for confana/io_hdf5.py — HDF5 PIMD trajectory reader.

Uses synthetic HDF5 fixtures (no real trajectory files required).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from confana.models import DoFDefinition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_FRAMES = 10
N_BEADS = 4
N_ATOMS = 21


def _make_dof_defs() -> list[DoFDefinition]:
    return [
        DoFDefinition(
            name="carboxyl_dihedral",
            type="dihedral",
            atoms=[6, 5, 10, 7],
            label="Carboxyl",
            domain=[-180, 180],
            enabled=True,
        ),
        DoFDefinition(
            name="ester_dihedral",
            type="dihedral",
            atoms=[5, 6, 12, 11],
            label="Ester",
            domain=[-180, 180],
            enabled=True,
        ),
    ]


def _write_fake_hdf5(sim_dir: Path, *, n_frames=N_FRAMES, n_beads=N_BEADS, n_atoms=N_ATOMS) -> Path:
    """Create a minimal synthetic HDF5 trajectory and matching input.xyz."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    hdf5_dir = sim_dir / "hdf5"
    hdf5_dir.mkdir(parents=True)
    h5_path = hdf5_dir / "trajectory.hdf5"

    rng = np.random.default_rng(42)
    bead_pos = rng.standard_normal((n_frames, n_beads, n_atoms, 3)).astype(np.float64)
    centroid_pos = bead_pos.mean(axis=1)
    potential = rng.standard_normal(n_frames).astype(np.float64) * 10 - 100.0

    with h5py.File(h5_path, "w") as fh:
        fh.create_dataset("bead_positions", data=bead_pos)
        fh.create_dataset("positions", data=centroid_pos)
        fh.create_dataset("potential", data=potential)

    # Aspirin-like element list (21 atoms)
    elements = ["C"] * 7 + ["O"] * 3 + ["C"] * 2 + ["O"] + ["H"] * 8
    assert len(elements) == n_atoms
    input_xyz = sim_dir / "input.xyz"
    lines = [f"{n_atoms}\n", "comment\n"]
    for el in elements:
        lines.append(f"{el}  0.0  0.0  0.0\n")
    input_xyz.write_text("".join(lines))

    return h5_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_coordinate_table_bead_mode(tmp_path):
    """Bead mode produces n_frames * n_beads rows with correct metadata."""
    from confana.io_hdf5 import build_coordinate_table_from_hdf5

    sim_dir = tmp_path / "s1"
    h5_path = _write_fake_hdf5(sim_dir)
    dof_defs = _make_dof_defs()

    df = build_coordinate_table_from_hdf5(h5_path, dof_defs, positions_source="bead")

    assert len(df) == N_FRAMES * N_BEADS
    assert set(df["bead_id"].dropna().unique()) == {f"bead_{i:02d}" for i in range(N_BEADS)}
    assert df["trajectory_id"].unique().tolist() == ["s1"]
    assert (df["byte_offset"] == -1).all()
    assert "carboxyl_dihedral" in df.columns
    assert "ester_dihedral" in df.columns
    assert "energy" in df.columns
    assert not df["energy"].isna().any()


def test_build_coordinate_table_centroid_mode(tmp_path):
    """Centroid mode produces exactly n_frames rows with bead_id = NA."""
    from confana.io_hdf5 import build_coordinate_table_from_hdf5

    sim_dir = tmp_path / "s0"
    h5_path = _write_fake_hdf5(sim_dir)
    dof_defs = _make_dof_defs()

    df = build_coordinate_table_from_hdf5(h5_path, dof_defs, positions_source="centroid")

    assert len(df) == N_FRAMES
    assert df["bead_id"].isna().all()
    assert df["trajectory_id"].unique().tolist() == ["s0"]


def test_frame_number_and_local_frame_index(tmp_path):
    """frame_number and local_frame_index are 0-based per bead."""
    from confana.io_hdf5 import build_coordinate_table_from_hdf5

    sim_dir = tmp_path / "s2"
    h5_path = _write_fake_hdf5(sim_dir)
    dof_defs = _make_dof_defs()

    df = build_coordinate_table_from_hdf5(h5_path, dof_defs, positions_source="bead")
    bead0 = df[df["bead_id"] == "bead_00"]

    assert list(bead0["frame_number"]) == list(range(N_FRAMES))
    assert list(bead0["local_frame_index"]) == list(range(N_FRAMES))


def test_multi_file_global_frame_index(tmp_path):
    """global_frame_index is sequential across multiple files."""
    from confana.io_hdf5 import build_coordinate_table_from_hdf5_files

    paths = []
    for name in ["s0", "s1"]:
        sim_dir = tmp_path / name
        paths.append(_write_fake_hdf5(sim_dir))

    dof_defs = _make_dof_defs()
    df = build_coordinate_table_from_hdf5_files(paths, dof_defs, positions_source="bead")

    total = N_FRAMES * N_BEADS * 2
    assert len(df) == total
    assert list(df["frame_id"]) == list(range(total))
    assert set(df["trajectory_id"].unique()) == {"s0", "s1"}


def test_atom_count_mismatch_raises(tmp_path):
    """Mismatched atom count between HDF5 and input.xyz raises ValueError."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    from confana.io_hdf5 import build_coordinate_table_from_hdf5

    sim_dir = tmp_path / "s_bad"
    hdf5_dir = sim_dir / "hdf5"
    hdf5_dir.mkdir(parents=True)
    h5_path = hdf5_dir / "trajectory.hdf5"

    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as fh:
        fh.create_dataset("bead_positions", data=rng.standard_normal((5, 2, 21, 3)))
        fh.create_dataset("positions", data=rng.standard_normal((5, 21, 3)))
        fh.create_dataset("potential", data=rng.standard_normal(5))

    # Write input.xyz with wrong atom count (10 instead of 21)
    (sim_dir / "input.xyz").write_text("10\ncomment\n" + "C  0 0 0\n" * 10)

    with pytest.raises(ValueError, match="atom"):
        build_coordinate_table_from_hdf5(h5_path, _make_dof_defs())


def test_missing_input_xyz_raises(tmp_path):
    """Missing input.xyz raises FileNotFoundError."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    from confana.io_hdf5 import build_coordinate_table_from_hdf5

    sim_dir = tmp_path / "s_noinput"
    hdf5_dir = sim_dir / "hdf5"
    hdf5_dir.mkdir(parents=True)
    h5_path = hdf5_dir / "trajectory.hdf5"

    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as fh:
        fh.create_dataset("bead_positions", data=rng.standard_normal((5, 2, 21, 3)))
        fh.create_dataset("positions", data=rng.standard_normal((5, 21, 3)))
        fh.create_dataset("potential", data=rng.standard_normal(5))

    with pytest.raises(FileNotFoundError, match="input.xyz"):
        build_coordinate_table_from_hdf5(h5_path, _make_dof_defs())


def test_discover_hdf5_files(tmp_path):
    """discover_hdf5_files returns sorted .hdf5 paths matching the pattern."""
    from confana.io_hdf5 import discover_hdf5_files

    for name in ["s0", "s1", "s2"]:
        d = tmp_path / name / "hdf5"
        d.mkdir(parents=True)
        (d / "trajectory.hdf5").touch()

    pattern = str(tmp_path / "*/hdf5/trajectory.hdf5")
    found = discover_hdf5_files(pattern)
    assert len(found) == 3
    assert all(p.suffix == ".hdf5" for p in found)


def test_cache_round_trip(tmp_path):
    """_load_or_build_hdf5_coordinate_table writes and reloads NPZ correctly."""
    from confana.io_coordinates import _load_or_build_hdf5_coordinate_table

    sim_dir = tmp_path / "s1"
    h5_path = _write_fake_hdf5(sim_dir)
    dof_defs = _make_dof_defs()

    config = {
        "run_dir": str(tmp_path / "outputs"),
        "data": {
            "path_pattern": str(h5_path),
            "format": "hdf5",
            "positions_source": "bead",
        },
        "dof": [
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "label": "C", "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "label": "E", "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [],
    }

    df1, hit1 = _load_or_build_hdf5_coordinate_table(config, dof_defs)
    assert not hit1
    assert len(df1) == N_FRAMES * N_BEADS

    df2, hit2 = _load_or_build_hdf5_coordinate_table(config, dof_defs)
    assert hit2
    assert len(df2) == len(df1)
