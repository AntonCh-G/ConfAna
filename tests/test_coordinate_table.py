"""Tests for confana/coordinate_table.py — the one coordinate-table build.

Every test drives the pipeline through
``load_or_build_coordinate_table_from_config``, the entry point the CLI runs,
with tiny xyz files in ``tmp_path`` or tiny HDF5 runs from ``tests/hdf5_runs``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confana.coordinate_table import load_or_build_coordinate_table_from_config
from confana.geometry import dihedral_angle, distance
from confana.io_coordinates import REQUIRED_COLUMNS
from confana.io_xyz import iter_xyz_frames, scan_xyz_frame_offsets
from tests.hdf5_runs import write_hdf5_run

_N_ATOMS = 13
_DOFS = [
    {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
     "label": "Carboxyl", "domain": [-180, 180]},
    {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
     "label": "Ester", "domain": [-180, 180]},
]
_PIMD_IDS = {"trajectory_id_pattern": r"^(.+)_\d+$", "bead_id_pattern": r"_(\d+)$"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _helix(n_atoms: int = _N_ATOMS) -> np.ndarray:
    """A helical carbon chain: neighbours 1.51 Å apart (bonded), all others >= 1.8 Å.

    The C–C bond threshold is 1.67 Å (``bond_check.build_bond_graph``).
    """
    i = np.arange(n_atoms)
    phi = np.deg2rad(120.0) * i
    return np.column_stack([0.8 * np.cos(phi), 0.8 * np.sin(phi), 0.6 * i])


def _write_xyz(
    path: Path,
    n_frames: int = 4,
    *,
    seed: int = 0,
    break_at: int | None = None,
    atom_counts: list[int] | None = None,
) -> Path:
    """Write a multi-frame xyz file of a jittered helix with ``Step:`` comments.

    From frame *break_at* on, the last atom sits 5 Å away (a broken bond).
    *atom_counts* overrides the atom count of each frame.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame in range(n_frames):
        n_atoms = atom_counts[frame] if atom_counts else _N_ATOMS
        coords = _helix(n_atoms) + rng.normal(scale=0.02, size=(n_atoms, 3))
        if break_at is not None and frame >= break_at:
            coords[-1, 0] += 5.0
        lines.append(f"{n_atoms}\n")
        lines.append(f"# Step: {frame * 10} Bead: 0\n")
        lines.extend(f"C {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in coords)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _config(tmp_path: Path, path_pattern: str | Path, **sections) -> dict:
    """A run config over *path_pattern*; keyword sections override or extend it.

    ``data=`` entries are merged into the ``data`` section.
    """
    config: dict = {
        "run_dir": str(tmp_path / "run"),
        "data": {"path_pattern": str(path_pattern)},
        "dof": copy.deepcopy(_DOFS),
        "coordinate_pairs": [
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
    }
    config["data"].update(sections.pop("data", {}))
    config.update(sections)
    return config


def _hdf5_config(tmp_path: Path, path_pattern: str | Path, **sections) -> dict:
    data = {"format": "hdf5", **sections.pop("data", {})}
    return _config(tmp_path, path_pattern, data=data, **sections)


def _cache_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "run" / ".cache").glob("*__coordinates.npz"))


def _load(config: dict, **kwargs) -> tuple[pd.DataFrame, bool]:
    return load_or_build_coordinate_table_from_config(config, **kwargs)


# ---------------------------------------------------------------------------
# xyz: table contents
# ---------------------------------------------------------------------------


def test_xyz_table_has_the_standard_schema_in_order(tmp_path):
    xyz = _write_xyz(tmp_path / "data" / "traj.xyz", n_frames=3)

    df, cache_hit = _load(_config(tmp_path, xyz))

    assert not cache_hit
    assert list(df.columns) == REQUIRED_COLUMNS + [
        "carboxyl_dihedral", "ester_dihedral", "energy", "step_number",
    ]
    assert df["frame_id"].tolist() == [0, 1, 2]
    assert df["global_frame_index"].tolist() == [0, 1, 2]
    assert df["frame_number"].tolist() == df["local_frame_index"].tolist() == [0, 1, 2]
    assert df["step_number"].tolist() == [0, 10, 20]
    assert df["byte_offset"].tolist() == [
        e.byte_offset for e in scan_xyz_frame_offsets(xyz).entries
    ]
    assert set(df["source_file"]) == {str(xyz.resolve())}
    assert (df["atom_count"] == _N_ATOMS).all()
    assert (df["comment_line"] == "").all()
    assert df["energy"].isna().all()
    assert df["bead_id"].isna().all()
    assert str(df["carboxyl_dihedral"].dtype) == "float32"


def test_xyz_dof_values_match_the_geometry_of_each_frame(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=5)

    df, _ = _load(_config(tmp_path, xyz))

    for row, frame in zip(df.itertuples(), iter_xyz_frames(xyz)):
        carboxyl = dihedral_angle(frame.coords, [6, 5, 10, 7])
        ester = dihedral_angle(frame.coords, [5, 6, 12, 11])
        assert row.carboxyl_dihedral == pytest.approx(carboxyl, abs=1e-4)
        assert row.ester_dihedral == pytest.approx(ester, abs=1e-4)


def test_cache_hit_returns_the_same_table_as_the_build(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz", seed=0)
    _write_xyz(tmp_path / "run.pos_01.xyz", seed=1)
    config = _config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS)

    built, hit1 = _load(config)
    cached, hit2 = _load(config)

    assert (hit1, hit2) == (False, True)
    pd.testing.assert_frame_equal(built, cached)


def test_force_rebuild_ignores_the_cache(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz)

    _load(config)
    _, hit = _load(config, force_rebuild_cache=True)

    assert not hit


def test_pimd_bead_files_take_ids_from_their_names(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz", n_frames=2, seed=0)
    _write_xyz(tmp_path / "run.pos_01.xyz", n_frames=2, seed=1)

    df, _ = _load(_config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS))

    assert df["trajectory_id"].tolist() == ["run.pos"] * 4
    assert df["bead_id"].tolist() == ["00", "00", "01", "01"]
    assert df["frame_id"].tolist() == [0, 1, 2, 3]
    assert len(_cache_files(tmp_path)) == 1  # one trajectory, one cache file


def test_coordinate_transforms_add_shifted_columns(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")

    df, _ = _load(_config(tmp_path, xyz, coordinate_transforms={"carboxyl_dihedral": 90.0}))

    raw = df["carboxyl_dihedral"].to_numpy(dtype=float)
    expected = (raw + 90.0 + 180.0) % 360.0 - 180.0
    shifted = df["carboxyl_dihedral_shifted"].to_numpy(dtype=float)
    np.testing.assert_allclose(shifted, expected, atol=1e-4)
    assert list(df.columns)[-1] == "carboxyl_dihedral_shifted"


def test_parallel_build_gives_the_same_table(tmp_path):
    for bead in range(3):
        _write_xyz(tmp_path / "data" / f"run.pos_0{bead}.xyz", seed=bead)
    pattern = tmp_path / "data" / "*.xyz"

    serial, _ = _load(_config(tmp_path / "serial", pattern, data=_PIMD_IDS))
    parallel, _ = _load(
        _config(tmp_path / "parallel", pattern, data=_PIMD_IDS, cache={"n_jobs": 2})
    )

    pd.testing.assert_frame_equal(serial, parallel)


# ---------------------------------------------------------------------------
# xyz: trajectory_id / bead_id rule
# ---------------------------------------------------------------------------


def test_without_a_pattern_the_folder_is_the_trajectory(tmp_path):
    _write_xyz(tmp_path / "sim_a" / "part1.xyz", n_frames=2)
    _write_xyz(tmp_path / "sim_a" / "part2.xyz", n_frames=2)
    _write_xyz(tmp_path / "sim_b" / "part1.xyz", n_frames=2)

    df, _ = _load(_config(tmp_path, tmp_path / "sim_*" / "*.xyz"))

    assert df["trajectory_id"].tolist() == ["sim_a"] * 4 + ["sim_b"] * 2
    assert df["bead_id"].isna().all()
    assert len(_cache_files(tmp_path)) == 2  # cache groups follow trajectory_id


def test_trajectory_id_pattern_is_searched_anywhere_in_the_name(tmp_path):
    xyz = _write_xyz(tmp_path / "prefix_run7.pos_00.xyz", n_frames=1)

    df, _ = _load(_config(tmp_path, xyz, data={"trajectory_id_pattern": r"(run\d+)"}))

    assert df["trajectory_id"].tolist() == ["run7"]


@pytest.mark.parametrize("key", ["trajectory_id_pattern", "bead_id_pattern"])
def test_a_pattern_without_a_capture_group_raises(tmp_path, key):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=1)

    with pytest.raises(ValueError, match=f"{key}.*capture group"):
        _load(_config(tmp_path, xyz, data={key: r"traj"}))


def test_a_trajectory_pattern_that_does_not_match_raises_naming_the_file(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz", n_frames=1)
    _write_xyz(tmp_path / "other.xyz", n_frames=1)

    with pytest.raises(ValueError, match=r"trajectory_id_pattern.*other\.xyz"):
        _load(_config(tmp_path, tmp_path / "*.xyz", data={"trajectory_id_pattern": r"^(.+)_\d+$"}))


def test_a_bead_pattern_that_does_not_match_raises_naming_the_file(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=1)

    with pytest.raises(ValueError, match=r"bead_id_pattern.*traj\.xyz"):
        _load(_config(tmp_path, xyz, data={"bead_id_pattern": r"_(\d+)$"}))


# ---------------------------------------------------------------------------
# xyz: cache invalidation
# ---------------------------------------------------------------------------


def _hits_after(tmp_path: Path, config: dict, change) -> bool:
    """Build *config*, apply *change* to a copy, and return the second load's cache_hit."""
    _load(config)
    changed = copy.deepcopy(config)
    change(changed)
    _, hit = _load(changed)
    return hit


def test_unchanged_settings_hit_the_cache(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    assert _hits_after(tmp_path, _config(tmp_path, xyz), lambda c: None)


def test_changing_bond_break_rebuilds(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz)

    def enable(c):
        c["bond_break"] = {"enabled": True, "cutoff": 2.0}

    def new_cutoff(c):
        c["bond_break"]["cutoff"] = 2.5

    assert not _hits_after(tmp_path, config, enable)
    enabled = copy.deepcopy(config)
    enable(enabled)
    assert not _hits_after(tmp_path, enabled, new_cutoff)


def test_changing_the_bead_pattern_rebuilds(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz")
    config = _config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS)

    def change(c):
        c["data"]["bead_id_pattern"] = r"(pos_\d+)$"

    assert not _hits_after(tmp_path, config, change)


def test_changing_the_trajectory_pattern_rebuilds(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz")
    config = _config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS)

    def change(c):
        c["data"]["trajectory_id_pattern"] = r"^(.+)\.pos_\d+$"

    assert not _hits_after(tmp_path, config, change)


def test_a_pattern_edit_that_gives_the_same_ids_keeps_the_cache(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz")
    config = _config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS)

    def change(c):
        c["data"]["trajectory_id_pattern"] = r"^(.+)_\d\d$"

    assert _hits_after(tmp_path, config, change)


def test_changing_frame_range_rebuilds(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=5)

    def change(c):
        c["frame_range"] = {"start_frame": 1, "end_frame": 3}

    assert not _hits_after(tmp_path, _config(tmp_path, xyz), change)


def test_changing_a_dof_rebuilds(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")

    def new_atoms(c):
        c["dof"][0]["atoms"] = [5, 4, 9, 7]

    def new_dof(c):
        c["dof"].append({"name": "d01", "type": "distance", "atoms": [0, 1],
                         "label": "d01", "domain": [0, 5]})

    assert not _hits_after(tmp_path, _config(tmp_path, xyz), new_atoms)
    assert not _hits_after(tmp_path / "b", _config(tmp_path / "b", xyz), new_dof)


def test_a_changed_source_file_rebuilds_only_its_trajectory(tmp_path):
    changed = _write_xyz(tmp_path / "sim_a" / "traj.xyz", n_frames=2)
    _write_xyz(tmp_path / "sim_b" / "traj.xyz", n_frames=2)
    config = _config(tmp_path, tmp_path / "sim_*" / "*.xyz")
    _load(config)
    stamps = {p.name: p.stat().st_mtime_ns for p in _cache_files(tmp_path)}

    _write_xyz(changed, n_frames=3)  # one more frame
    df, hit = _load(config)

    assert not hit
    assert len(df) == 5
    rebuilt = [p.name for p in _cache_files(tmp_path) if p.stat().st_mtime_ns != stamps[p.name]]
    assert len(rebuilt) == 1 and rebuilt[0].startswith("sim_a")


# ---------------------------------------------------------------------------
# xyz: frame range and bond break
# ---------------------------------------------------------------------------


def test_frame_range_keeps_the_inclusive_range(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=6)

    df, _ = _load(_config(tmp_path, xyz, frame_range={"start_frame": 1, "end_frame": 3}))

    assert df["frame_number"].tolist() == [1, 2, 3]
    assert df["frame_id"].tolist() == [0, 1, 2]


def test_frame_range_end_before_start_raises(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")

    with pytest.raises(ValueError, match="end_frame"):
        _load(_config(tmp_path, xyz, frame_range={"start_frame": 3, "end_frame": 1}))


def test_bond_break_cuts_every_bead_at_the_earliest_break(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz", n_frames=10, break_at=6)
    _write_xyz(tmp_path / "run.pos_01.xyz", n_frames=10, break_at=8, seed=1)
    config = _config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS,
                     bond_break={"enabled": True, "cutoff": 2.0})

    df, _ = _load(config)

    for bead in ("00", "01"):
        assert df.loc[df["bead_id"] == bead, "frame_number"].tolist() == [0, 1, 2, 3, 4, 5]


def test_bond_break_and_frame_range_combine(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=10, break_at=6)
    config = _config(tmp_path, xyz, bond_break={"enabled": True, "cutoff": 2.0},
                     frame_range={"start_frame": 2, "end_frame": 8})

    df, _ = _load(config)

    assert df["frame_number"].tolist() == [2, 3, 4, 5]


def test_bond_break_reference_is_the_first_frame_in_range(tmp_path):
    # From frame 3 on the last atom is already away: the reference at frame 5
    # has no bond to it, so nothing breaks later.
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=10, break_at=3)
    config = _config(tmp_path, xyz, bond_break={"enabled": True, "cutoff": 2.0},
                     frame_range={"start_frame": 5})

    df, _ = _load(config)

    assert df["frame_number"].tolist() == [5, 6, 7, 8, 9]


def test_disabled_bond_break_keeps_every_frame(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=10, break_at=6)

    df, _ = _load(_config(tmp_path, xyz, bond_break={"enabled": False, "cutoff": 2.0}))

    assert len(df) == 10


def test_without_bond_break_files_keep_their_own_length(tmp_path):
    _write_xyz(tmp_path / "run.pos_00.xyz", n_frames=3)
    _write_xyz(tmp_path / "run.pos_01.xyz", n_frames=5)

    df, _ = _load(_config(tmp_path, tmp_path / "*.xyz", data=_PIMD_IDS))

    assert df.groupby("bead_id")["frame_number"].count().to_dict() == {"00": 3, "01": 5}


# ---------------------------------------------------------------------------
# xyz: frame-index sidecars
# ---------------------------------------------------------------------------


def test_same_named_files_get_separate_frame_indices(tmp_path):
    _write_xyz(tmp_path / "sim_a" / "traj.xyz", n_frames=2)
    _write_xyz(tmp_path / "sim_b" / "traj.xyz", n_frames=3)
    index_dir = tmp_path / "indices"
    config = _config(tmp_path, tmp_path / "sim_*" / "traj.xyz",
                     cache={"index_cache_dir": str(index_dir)})

    df, _ = _load(config)

    assert len(list(index_dir.glob("traj.xyz.*.frameindex.npz"))) == 2
    assert df.groupby("trajectory_id")["frame_number"].count().to_dict() == {"sim_a": 2, "sim_b": 3}


# ---------------------------------------------------------------------------
# xyz: invalid input fails loudly
# ---------------------------------------------------------------------------


def test_an_atom_index_past_the_atom_count_raises(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz)
    config["dof"][0]["atoms"] = [6, 5, 10, 20]

    with pytest.raises(ValueError, match=r"carboxyl_dihedral.*20.*traj\.xyz"):
        _load(config)


def test_frames_with_different_atom_counts_raise(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz", n_frames=3, atom_counts=[13, 13, 14])

    with pytest.raises(ValueError, match=r"traj\.xyz.*atom"):
        _load(_config(tmp_path, xyz))


def test_files_of_one_trajectory_with_different_atom_counts_raise(tmp_path):
    _write_xyz(tmp_path / "sim" / "part1.xyz", n_frames=2)
    _write_xyz(tmp_path / "sim" / "part2.xyz", n_frames=2, atom_counts=[14, 14])

    with pytest.raises(ValueError, match=r"'sim'.*different atom counts"):
        _load(_config(tmp_path, tmp_path / "sim" / "*.xyz"))


def test_an_unimplemented_dof_type_raises(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz)
    config["dof"].append({"name": "pc1", "type": "collective", "label": "PC1",
                          "domain": [-1, 1], "input_dof": ["carboxyl_dihedral"]})

    with pytest.raises(ValueError, match="pc1.*collective"):
        _load(config)


def test_a_shift_on_an_unknown_dof_raises(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")

    with pytest.raises(ValueError, match="coordinate_transforms.*nope"):
        _load(_config(tmp_path, xyz, coordinate_transforms={"nope": 10.0}))


def test_coordinate_table_path_is_rejected(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz, cache={"coordinate_table_path": str(tmp_path / "t.npz")})

    with pytest.raises(ValueError, match="coordinate_table_path.*trajectory_cache_dir"):
        _load(config)


def test_a_missing_cache_location_raises(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz)
    del config["run_dir"]

    with pytest.raises(ValueError, match="run_dir.*trajectory_cache_dir"):
        _load(config)


def test_trajectory_cache_dir_overrides_run_dir(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")
    config = _config(tmp_path, xyz, cache={"trajectory_cache_dir": str(tmp_path / "elsewhere")})

    _load(config)

    assert len(list((tmp_path / "elsewhere").glob("*__coordinates.npz"))) == 1
    assert not _cache_files(tmp_path)


def test_positions_source_with_xyz_raises(tmp_path):
    xyz = _write_xyz(tmp_path / "traj.xyz")

    with pytest.raises(ValueError, match="positions_source"):
        _load(_config(tmp_path, xyz, data={"positions_source": "centroid"}))


def test_no_matching_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="path_pattern"):
        _load(_config(tmp_path, tmp_path / "*.xyz"))


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------


def test_hdf5_bead_mode_has_one_row_per_frame_and_bead(tmp_path):
    h5_path, beads, _ = write_hdf5_run(tmp_path / "s1", n_frames=4, n_beads=3, n_atoms=_N_ATOMS)

    df, _ = _load(_hdf5_config(tmp_path, h5_path))

    assert list(df.columns) == REQUIRED_COLUMNS + [
        "carboxyl_dihedral", "ester_dihedral", "energy", "step_number",
    ]
    assert df["bead_id"].tolist() == [f"bead_{b:02d}" for b in range(3) for _ in range(4)]
    assert df["frame_number"].tolist() == [0, 1, 2, 3] * 3
    assert df["step_number"].tolist() == [0, 1, 2, 3] * 3
    assert df["frame_id"].tolist() == list(range(12))
    assert set(df["trajectory_id"]) == {"s1"}
    assert (df["byte_offset"] == -1).all()
    assert df["energy"].tolist() == [0.0, -1.0, -2.0, -3.0] * 3
    bead2_frame1 = beads[1, 2].astype(np.float32)
    assert df["carboxyl_dihedral"].iloc[4 * 2 + 1] == pytest.approx(
        dihedral_angle(bead2_frame1, [6, 5, 10, 7]), abs=1e-4
    )


def test_hdf5_centroid_mode_has_one_row_per_frame(tmp_path):
    h5_path, _, centroid = write_hdf5_run(tmp_path / "s0", n_frames=4)
    config = _hdf5_config(tmp_path, h5_path, data={"positions_source": "centroid"})
    config["dof"] = [{"name": "d_co", "type": "distance", "atoms": [0, 1],
                      "label": "d", "domain": [0, 10]}]
    config["coordinate_pairs"] = []

    df, _ = _load(config)

    assert len(df) == 4
    assert df["bead_id"].isna().all()
    np.testing.assert_allclose(
        df["d_co"].to_numpy(dtype=float),
        [distance(c.astype(np.float32), [0, 1]) for c in centroid],
        atol=1e-5,
    )


def test_hdf5_coordinate_transforms_add_shifted_columns(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)
    config = _hdf5_config(tmp_path, h5_path, coordinate_transforms={"ester_dihedral": -45.0})

    df, _ = _load(config)

    raw = df["ester_dihedral"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        df["ester_dihedral_shifted"].to_numpy(dtype=float),
        (raw - 45.0 + 180.0) % 360.0 - 180.0,
        atol=1e-4,
    )


def test_hdf5_frame_range_cuts_every_bead_alike(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_frames=6, n_atoms=_N_ATOMS)
    config = _hdf5_config(tmp_path, h5_path, frame_range={"start_frame": 2, "end_frame": 4})

    df, _ = _load(config)

    assert df["frame_number"].tolist() == [2, 3, 4, 2, 3, 4]
    assert df["bead_id"].tolist() == ["bead_00"] * 3 + ["bead_01"] * 3


def test_hdf5_runs_are_separate_trajectories_with_separate_caches(tmp_path):
    for run in ("s0", "s1"):
        write_hdf5_run(tmp_path / run, n_frames=3, n_atoms=_N_ATOMS)
    config = _hdf5_config(tmp_path, tmp_path / "s*" / "hdf5" / "trajectory.hdf5")

    df, hit1 = _load(config)
    again, hit2 = _load(config)

    assert df["trajectory_id"].tolist() == ["s0"] * 6 + ["s1"] * 6
    assert df["frame_id"].tolist() == list(range(12))
    assert len(_cache_files(tmp_path)) == 2
    assert (hit1, hit2) == (False, True)
    pd.testing.assert_frame_equal(df, again)


def test_hdf5_changing_positions_source_rebuilds(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)

    def change(c):
        c["data"]["positions_source"] = "centroid"

    assert not _hits_after(tmp_path, _hdf5_config(tmp_path, h5_path), change)


def test_hdf5_bond_break_raises(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)
    config = _hdf5_config(tmp_path, h5_path, bond_break={"enabled": True, "cutoff": 2.0})

    with pytest.raises(ValueError, match="bond_break.*hdf5"):
        _load(config)


@pytest.mark.parametrize("key", ["trajectory_id_pattern", "bead_id_pattern"])
def test_hdf5_id_patterns_raise(tmp_path, key):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)

    with pytest.raises(ValueError, match=f"{key}.*hdf5"):
        _load(_hdf5_config(tmp_path, h5_path, data={key: r"(.+)"}))


def test_hdf5_without_input_xyz_raises(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)
    (tmp_path / "s0" / "input.xyz").unlink()

    with pytest.raises(FileNotFoundError, match="input.xyz"):
        _load(_hdf5_config(tmp_path, h5_path))


def test_hdf5_atom_count_unlike_input_xyz_raises(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)
    (tmp_path / "s0" / "input.xyz").write_text("2\ninput\nC 0 0 0\nO 0 0 0\n")

    with pytest.raises(ValueError, match="atom"):
        _load(_hdf5_config(tmp_path, h5_path))


def test_hdf5_without_a_dataset_raises_naming_it(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_atoms=_N_ATOMS)
    with h5py.File(h5_path, "a") as fh:
        del fh["potential"]

    with pytest.raises(ValueError, match="potential"):
        _load(_hdf5_config(tmp_path, h5_path))


# ---------------------------------------------------------------------------
# Reading in blocks
# ---------------------------------------------------------------------------


def test_small_blocks_give_the_same_tables(tmp_path, monkeypatch):
    _write_xyz(tmp_path / "data" / "run.pos_00.xyz", n_frames=9, break_at=7)
    _write_xyz(tmp_path / "data" / "run.pos_01.xyz", n_frames=9, seed=1)
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0", n_frames=7, n_beads=3, n_atoms=_N_ATOMS)
    configs = {
        "xyz": lambda root: _config(root, tmp_path / "data" / "*.xyz", data=_PIMD_IDS,
                                    bond_break={"enabled": True, "cutoff": 2.0},
                                    frame_range={"start_frame": 1}),
        "hdf5": lambda root: _hdf5_config(root, h5_path, frame_range={"start_frame": 1}),
    }
    whole = {name: _load(make(tmp_path / "whole" / name))[0] for name, make in configs.items()}

    monkeypatch.setattr("confana.io_xyz._BLOCK_FRAMES", 2)
    frame_bytes = 3 * _N_ATOMS * 3 * 8
    monkeypatch.setattr("confana.io_hdf5._BLOCK_BYTES", 2 * frame_bytes)
    blocked = {name: _load(make(tmp_path / "blocked" / name))[0] for name, make in configs.items()}

    assert whole["xyz"]["frame_number"].tolist() == [1, 2, 3, 4, 5, 6] * 2  # cut at the break
    for name in configs:
        pd.testing.assert_frame_equal(
            whole[name].drop(columns="source_file"), blocked[name].drop(columns="source_file")
        )
