"""Tests for confana/io_coordinates.py — coordinate-table files.

Building the table from trajectories is tested in tests/test_coordinate_table.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confana.cache import read_meta
from confana.io_coordinates import (
    REQUIRED_COLUMNS,
    build_frame_metadata,
    load_coordinate_table,
    save_coordinate_table,
    validate_coordinate_table,
)
from confana.models import FrameRecord


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


# ---------------------------------------------------------------------------
# validate_coordinate_table
# ---------------------------------------------------------------------------


def test_validate_passes_with_all_columns():
    df = _make_minimal_df()
    validate_coordinate_table(df)  # should not raise


def test_validate_raises_on_missing_columns():
    df = _make_minimal_df()
    df = df.drop(columns=["frame_id", "byte_offset"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_coordinate_table(df)


def test_validate_error_lists_missing():
    df = _make_minimal_df()
    df = df.drop(columns=["frame_id"])
    with pytest.raises(ValueError) as exc_info:
        validate_coordinate_table(df)
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
# validate_coordinate_table value columns / embedded cache metadata
# ---------------------------------------------------------------------------


def test_validate_raises_on_a_missing_value_column():
    with pytest.raises(ValueError, match="ester_dihedral_shifted"):
        validate_coordinate_table(_make_minimal_df(), value_columns=["ester_dihedral_shifted"])


def test_npz_embeds_cache_metadata(tmp_path):
    path = tmp_path / "coords.npz"
    save_coordinate_table(_make_minimal_df(), path, cache_metadata={"version": 2, "files": []})
    assert read_meta(path) == {"version": 2, "files": []}
    assert len(load_coordinate_table(path)) == 1


def test_cache_metadata_needs_npz(tmp_path):
    with pytest.raises(ValueError, match="npz"):
        save_coordinate_table(_make_minimal_df(), tmp_path / "coords.csv", cache_metadata={})
