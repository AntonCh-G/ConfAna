"""Tests for confana/io_coordinates.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from confana.coordinate_config import resolve_dof_definitions
from confana.coordinates import build_coordinate_table_from_xyz
from confana.io_coordinates import (
    REQUIRED_COLUMNS,
    _validate_coordinate_table,
    build_frame_metadata,
    load_or_build_coordinate_table_cache,
    load_or_build_coordinate_table_from_config,
    load_coordinate_table,
    save_coordinate_table,
)
from confana.io_xyz import iter_xyz_frames
from confana.models import DoFDefinition, FrameRecord
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_df(extra_cols: dict | None = None) -> pd.DataFrame:
    """Build a minimal valid coordinate table with all required columns."""
    row = {col: None for col in REQUIRED_COLUMNS}
    row.update(
        {
            "frame_id": 0,
            "source_file": "/tmp/test.xyz",
            "trajectory_id": "my_run.pos",
            "bead_id": "00",
            "frame_number": 0,
            "byte_offset": 0,
            "atom_count": 21,
            "comment_line": "# Step: 0",
            "local_frame_index": 0,
            "global_frame_index": 0,
            "carboxyl_dihedral": -120.0,
            "ester_dihedral": 60.0,
        }
    )
    if extra_cols:
        row.update(extra_cols)
    return pd.DataFrame([row])


def _make_frame_record() -> FrameRecord:
    return FrameRecord(
        source_file="/tmp/test.xyz",
        frame_number=5,
        byte_offset=1234,
        atom_count=21,
        comment_line="# Step: 50 Bead: 2",
        elements=["C"] * 21,
        coords=np.zeros((21, 3)),
        energy=None,
        step_number=50,
        bead_comment=2,
        trajectory_id="my_run.pos",
        bead_id="02",
        local_frame_index=5,
        global_frame_index=10,
    )


_DOF_DEFS = [
    DoFDefinition(
        name="carboxyl_dihedral",
        type="dihedral",
        label="Carboxyl dihedral (°)",
        domain=(-180.0, 180.0),
        atoms=(6, 5, 10, 7),
    ),
    DoFDefinition(
        name="ester_dihedral",
        type="dihedral",
        label="Ester dihedral (°)",
        domain=(-180.0, 180.0),
        atoms=(5, 6, 12, 11),
    ),
]

_DOF_DEFS_EXTENDED = _DOF_DEFS + [
    DoFDefinition(
        name="igor1_dihedral",
        type="dihedral",
        label="Igor1 dihedral (°)",
        domain=(-180.0, 180.0),
        atoms=(6, 12, 11, 8),
    ),
]


def _make_streaming_xyz(
    path: Path,
    *,
    bead: int,
    n_frames: int = 3,
    atom_count: int = 21,
) -> Path:
    """Write a small but geometrically valid multi-frame xyz file."""
    rng = np.random.default_rng(1234 + bead)
    lines: list[str] = []
    for frame_idx in range(n_frames):
        coords = rng.standard_normal((atom_count, 3)) + frame_idx * 0.05
        lines.append(f"{atom_count}\n")
        lines.append(
            "# CELL(abcABC):  10.0   10.0   10.0  90.0  90.0  90.0  "
            f"Step:     {frame_idx * 10}  Bead:     {bead}  positions{{angstrom}}  "
            "cell{angstrom}\n"
        )
        for atom_idx in range(atom_count):
            x, y, z = coords[atom_idx]
            lines.append(f"C {x:.8f} {y:.8f} {z:.8f}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _validate_coordinate_table
# ---------------------------------------------------------------------------


def test_validate_passes_with_all_columns():
    df = _make_minimal_df()
    _validate_coordinate_table(df)  # should not raise


def test_validate_raises_on_missing_columns():
    df = _make_minimal_df()
    df = df.drop(columns=["frame_id", "byte_offset"])
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_coordinate_table(df)


def test_validate_error_lists_missing():
    df = _make_minimal_df()
    df = df.drop(columns=["frame_id"])
    with pytest.raises(ValueError) as exc_info:
        _validate_coordinate_table(df)
    assert "frame_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_coordinate_table
# ---------------------------------------------------------------------------


def test_load_csv_valid(tmp_path):
    df = _make_minimal_df()
    csv_path = tmp_path / "coords.csv"
    df.to_csv(csv_path, index=False)
    loaded = load_coordinate_table(csv_path)
    assert list(loaded.columns[:5]) == list(df.columns[:5]) or set(REQUIRED_COLUMNS).issubset(loaded.columns)


def test_load_csv_missing_column_raises(tmp_path):
    df = _make_minimal_df()
    df = df.drop(columns=["frame_id"])
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_coordinate_table(csv_path)


def test_load_unsupported_format_raises(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("nothing")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_coordinate_table(p)


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_coordinate_table("/nonexistent/path/coords.csv")


def test_load_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow", reason="pyarrow not installed; skipping parquet test")
    df = _make_minimal_df()
    df["carboxyl_dihedral"] = df["carboxyl_dihedral"].astype(float)
    df["ester_dihedral"] = df["ester_dihedral"].astype(float)
    parquet_path = tmp_path / "coords.parquet"
    df.to_parquet(parquet_path, index=False)
    loaded = load_coordinate_table(parquet_path)
    assert set(REQUIRED_COLUMNS).issubset(loaded.columns)


def test_save_and_load_npz_round_trip(tmp_path):
    df = _make_minimal_df({"state_dihedral": "0", "step_number": 10})
    npz_path = tmp_path / "coords.npz"
    save_coordinate_table(df, npz_path)
    loaded = load_coordinate_table(npz_path)
    assert set(REQUIRED_COLUMNS).issubset(loaded.columns)
    assert float(loaded["carboxyl_dihedral"].iloc[0]) == pytest.approx(-120.0)
    assert str(loaded["state_dihedral"].iloc[0]) == "0"
    assert int(loaded["step_number"].iloc[0]) == 10


def test_load_coerces_float_angles(tmp_path):
    df = _make_minimal_df()
    csv_path = tmp_path / "coords.csv"
    df.to_csv(csv_path, index=False)
    loaded = load_coordinate_table(csv_path)
    assert str(loaded["carboxyl_dihedral"].dtype) == "float32"
    assert str(loaded["ester_dihedral"].dtype) == "float32"


# ---------------------------------------------------------------------------
# build_frame_metadata
# ---------------------------------------------------------------------------


def test_build_frame_metadata_keys():
    fr = _make_frame_record()
    meta_list = build_frame_metadata([fr])
    assert len(meta_list) == 1
    meta = meta_list[0]
    expected_keys = {
        "source_file", "frame_number", "byte_offset", "atom_count",
        "comment_line", "trajectory_id", "bead_id",
        "local_frame_index", "global_frame_index",
        "step_number", "bead_comment", "energy",
    }
    assert expected_keys <= set(meta.keys())


def test_build_frame_metadata_values():
    fr = _make_frame_record()
    meta = build_frame_metadata([fr])[0]
    assert meta["frame_number"] == 5
    assert meta["byte_offset"] == 1234
    assert meta["trajectory_id"] == "my_run.pos"
    assert meta["bead_id"] == "02"
    assert meta["step_number"] == 50
    assert "coords" not in meta  # coords must not be embedded


def test_build_frame_metadata_empty():
    assert build_frame_metadata([]) == []


def test_build_frame_metadata_multiple():
    frames = [_make_frame_record() for _ in range(5)]
    for i, fr in enumerate(frames):
        fr.global_frame_index = i
    meta_list = build_frame_metadata(frames)
    assert len(meta_list) == 5


# ---------------------------------------------------------------------------
# load_or_build_coordinate_table_cache
# ---------------------------------------------------------------------------


def test_coordinate_cache_builds_then_hits(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    _make_streaming_xyz(tmp_path / "my_run.pos_01.xyz", bead=1, n_frames=2)
    cache_path = tmp_path / "angles.npz"

    df1, hit1 = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    df2, hit2 = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )

    assert not hit1
    assert hit2
    assert cache_path.exists()
    assert len(df1) == len(df2) == 4
    assert set(REQUIRED_COLUMNS).issubset(df2.columns)


def test_coordinate_cache_rebuilds_when_embedded_metadata_missing(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    cache_path = tmp_path / "angles.npz"
    save_coordinate_table(_make_minimal_df(), cache_path)

    _, cache_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not cache_hit


def test_coordinate_cache_rebuilds_when_source_changes(tmp_path: Path):
    xyz_path = _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    cache_path = tmp_path / "angles.npz"

    _, first_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not first_hit

    with open(xyz_path, "a", encoding="utf-8") as fh:
        rng = np.random.default_rng(9001)
        fh.write(
            "21\n"
            "# CELL(abcABC):  10.0   10.0   10.0  90.0  90.0  90.0  "
            "Step:     999  Bead:     0  positions{angstrom}  cell{angstrom}\n"
        )
        for atom_idx in range(21):
            x, y, z = rng.standard_normal(3)
            fh.write(f"C {x:.8f} {y:.8f} {z:.8f}\n")

    df2, second_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not second_hit
    assert len(df2) == 3


def test_coordinate_cache_rebuilds_when_mapping_changes(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    cache_path = tmp_path / "angles.npz"

    _, first_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not first_hit

    # Different atom indices → different dof_fingerprint → cache miss
    updated_dof_defs = [
        DoFDefinition(
            name="carboxyl_dihedral",
            type="dihedral",
            label="Carboxyl dihedral (°)",
            domain=(-180.0, 180.0),
            atoms=(5, 4, 9, 7),  # changed
        ),
        _DOF_DEFS[1],
    ]
    _, second_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=updated_dof_defs,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not second_hit


def test_coordinate_cache_rebuilds_when_dof_defs_change(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    cache_path = tmp_path / "angles.npz"

    _, first_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not first_hit

    # Adding a new DoF changes the fingerprint → cache miss
    _, second_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS_EXTENDED,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
    )
    assert not second_hit


def test_coordinate_cache_streaming_matches_frame_builder(tmp_path: Path):
    xyz_path = _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=3)
    cache_path = tmp_path / "angles.npz"

    cached_df, _ = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,
        cache_path=cache_path,
        trajectory_id_pattern=r"^(.+)_\d+$",
        bead_id_pattern=r"_(\d+)$",
        force_rebuild=True,
    )

    frames = list(
        iter_xyz_frames(
            xyz_path,
            trajectory_id_pattern=r"^(.+)_\d+$",
            bead_id_pattern=r"_(\d+)$",
        )
    )
    for global_idx, frame in enumerate(frames):
        frame.global_frame_index = global_idx
    built_df = build_coordinate_table_from_xyz(frames, _DOF_DEFS)

    assert len(cached_df) == len(built_df) == 3
    # Batch (float64) and scalar (float32) paths agree within float32 precision.
    np.testing.assert_allclose(
        cached_df["carboxyl_dihedral"].to_numpy(dtype=float),
        built_df["carboxyl_dihedral"].to_numpy(dtype=float),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        cached_df["ester_dihedral"].to_numpy(dtype=float),
        built_df["ester_dihedral"].to_numpy(dtype=float),
        atol=1e-4,
    )
    assert cached_df["frame_number"].tolist() == built_df["frame_number"].tolist()
    assert cached_df["byte_offset"].tolist() == built_df["byte_offset"].tolist()


def test_load_or_build_coordinate_table_from_config_uses_cache_on_second_call(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    _make_streaming_xyz(tmp_path / "my_run.pos_01.xyz", bead=1, n_frames=2)
    config = {
        "data": {
            "path_pattern": str(tmp_path / "*.xyz"),
            "trajectory_id_pattern": r"^(.+)_\d+$",
            "bead_id_pattern": r"_(\d+)$",
        },
        "cache": {
            "index_cache_dir": None,
            "coordinate_table_path": str(tmp_path / "outputs" / "coordinates_angles.npz"),
        },
        "dof": [
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "label": "Carboxyl dihedral (°)", "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "label": "Ester dihedral (°)", "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
    }

    first, hit1 = load_or_build_coordinate_table_from_config(config)
    second, hit2 = load_or_build_coordinate_table_from_config(config)

    assert not hit1
    assert hit2
    assert len(first) == len(second) == 4
    assert "carboxyl_dihedral" in first.columns


def test_load_or_build_coordinate_table_from_config_force_rebuilds(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    config = {
        "data": {
            "path_pattern": str(tmp_path / "*.xyz"),
            "trajectory_id_pattern": r"^(.+)_\d+$",
            "bead_id_pattern": r"_(\d+)$",
        },
        "cache": {
            "index_cache_dir": None,
            "coordinate_table_path": str(tmp_path / "outputs" / "coordinates_angles.npz"),
        },
        "dof": [
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "label": "Carboxyl dihedral (°)", "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "label": "Ester dihedral (°)", "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
    }

    _, hit1 = load_or_build_coordinate_table_from_config(config)
    _, hit2 = load_or_build_coordinate_table_from_config(
        config,
        force_rebuild_cache=True,
    )
    assert not hit1
    assert not hit2


def test_load_or_build_coordinate_table_from_config_preserves_extra_dofs(tmp_path: Path):
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=2)
    config = {
        "data": {
            "path_pattern": str(tmp_path / "*.xyz"),
            "trajectory_id_pattern": r"^(.+)_\d+$",
            "bead_id_pattern": r"_(\d+)$",
        },
        "cache": {
            "index_cache_dir": None,
            "coordinate_table_path": str(tmp_path / "outputs" / "coordinates_angles.npz"),
        },
        "dof": [
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "label": "Carboxyl dihedral (°)", "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "label": "Ester dihedral (°)", "domain": [-180, 180], "enabled": True},
            {"name": "igor1_dihedral", "type": "dihedral", "atoms": [6, 12, 11, 8],
             "label": "Igor1 dihedral (°)", "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
    }

    df, _ = load_or_build_coordinate_table_from_config(config)
    assert "igor1_dihedral" in df.columns


# ---------------------------------------------------------------------------
# frame_range support
# ---------------------------------------------------------------------------


def test_frame_range_restricts_output_rows(tmp_path: Path):
    """frame_range.start_frame/end_frame must limit rows in output table."""
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=5)
    cache_path = tmp_path / "angles.npz"

    df_full, _ = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,

        cache_path=cache_path,
    )
    assert len(df_full) == 5

    cache_path.unlink()  # force rebuild with range
    df_range, _ = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,

        cache_path=cache_path,
        frame_range_cfg={"start_frame": 1, "end_frame": 3},
    )
    assert len(df_range) == 3


def test_frame_range_cache_invalidation(tmp_path: Path):
    """Changing frame_range must invalidate the cache (cache_hit=False)."""
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=5)
    cache_path = tmp_path / "angles.npz"

    _, hit1 = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,

        cache_path=cache_path,
        frame_range_cfg={"start_frame": 0, "end_frame": 4},
    )
    assert not hit1

    _, hit2 = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,

        cache_path=cache_path,
        frame_range_cfg={"start_frame": 0, "end_frame": 4},
    )
    assert hit2  # same range → cache hit

    _, hit3 = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,

        cache_path=cache_path,
        frame_range_cfg={"start_frame": 1, "end_frame": 3},  # different range
    )
    assert not hit3  # different range → must rebuild


def test_no_frame_range_processes_all_frames(tmp_path: Path):
    """Omitting frame_range must not change existing row-count behaviour."""
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=4)
    cache_path = tmp_path / "angles.npz"

    df, _ = load_or_build_coordinate_table_cache(
        path_pattern=str(tmp_path / "*.xyz"),
        dof_defs=_DOF_DEFS,

        cache_path=cache_path,
    )
    assert len(df) == 4


def test_frame_range_invalid_raises(tmp_path: Path):
    """end_frame < start_frame must raise ValueError."""
    _make_streaming_xyz(tmp_path / "my_run.pos_00.xyz", bead=0, n_frames=5)
    cache_path = tmp_path / "angles.npz"

    with pytest.raises(ValueError, match="end_frame"):
        load_or_build_coordinate_table_cache(
            path_pattern=str(tmp_path / "*.xyz"),
            dof_defs=_DOF_DEFS,
    
            cache_path=cache_path,
            frame_range_cfg={"start_frame": 4, "end_frame": 1},
        )
