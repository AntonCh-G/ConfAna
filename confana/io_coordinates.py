"""Coordinate-table files: CSV, Parquet and NPZ, plus schema checks.

The coordinate table is built from trajectories by
:mod:`confana.coordinate_table`, which caches each trajectory's table as an
NPZ written and read here. Downstream analysis (density, clustering,
transitions) can also start from a table file without parsing any xyz.

Public API
----------
- ``REQUIRED_COLUMNS``
- ``load_coordinate_table``
- ``save_coordinate_table``
- ``validate_coordinate_table``
- ``build_frame_metadata``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from confana.cache import embed_meta
from confana.models import FrameRecord

# ---------------------------------------------------------------------------
# Required columns for the standard coordinate table schema.
# Additional dihedral columns may be present beyond this backwards-compatible
# baseline and are preserved transparently in caches / I/O helpers.
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "frame_id",
    "source_file",
    "trajectory_id",
    "bead_id",
    "frame_number",
    "byte_offset",
    "atom_count",
    "comment_line",
    "local_frame_index",
    "global_frame_index",
]

# Optional columns that may be present but are not required
OPTIONAL_COLUMNS: list[str] = [
    "energy",
    "step_number",
    "has_structure",
]

_NPZ_TABLE_META_KEY = "coordinate_table_meta_json"
_TABLE_META_VERSION = 1


# ---------------------------------------------------------------------------
# Validation and dtype coercion
# ---------------------------------------------------------------------------


def validate_coordinate_table(df: pd.DataFrame, *, value_columns: Sequence[str] = ()) -> None:
    """Raise ValueError if a required column, or one of *value_columns*, is missing."""
    missing = [col for col in [*REQUIRED_COLUMNS, *value_columns] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Coordinate table is missing required columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce coordinate-table columns to canonical nullable pandas dtypes."""
    int_cols = [
        "frame_id",
        "frame_number",
        "byte_offset",
        "atom_count",
        "local_frame_index",
        "global_frame_index",
        "step_number",
    ]
    str_cols = ["bead_id"]
    _non_float = set(int_cols) | set(str_cols) | {
        "source_file", "trajectory_id", "comment_line",
    }

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Coerce any float-dtype column (DoF values, energy, etc.) to float32.
    for col in df.columns:
        if col not in _non_float and pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")

    return df


# ---------------------------------------------------------------------------
# NPZ serialization helpers
# ---------------------------------------------------------------------------


def _encode_string_categories(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return integer codes and a string lookup table for one column."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        cat = series.astype("category")
        categories = np.asarray([str(v) for v in cat.cat.categories], dtype=str)
        return cat.cat.codes.to_numpy(dtype=np.int32), categories

    values = series.astype("string")
    mask = values.isna().to_numpy()
    codes = np.full(len(values), -1, dtype=np.int32)
    if mask.all():
        return codes, np.asarray([], dtype=str)

    valid = values[~mask].astype(str)
    cat = pd.Categorical(valid)
    codes[~mask] = cat.codes.astype(np.int32)
    categories = np.asarray(cat.categories.to_list(), dtype=str)
    return codes, categories


def _decode_string_categories(
    codes: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    """Decode integer codes and category values to an object array."""
    values = np.empty(len(codes), dtype=object)
    values[:] = None
    valid = codes >= 0
    if valid.any():
        values[valid] = categories[codes[valid]]
    return values


def _column_kind(series: pd.Series) -> str:
    """Map a coordinate-table column to a serialization strategy."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "int"
    if pd.api.types.is_float_dtype(series.dtype):
        return "float"
    return "string_category"


def _write_coordinate_npz(
    df: pd.DataFrame,
    path: Path,
    *,
    cache_metadata: dict[str, Any] | None = None,
) -> None:
    """Write a coordinate table to an NPZ archive."""
    payload: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "version": _TABLE_META_VERSION,
        "columns": [],
    }

    for col in df.columns:
        series = df[col]
        kind = _column_kind(series)
        meta["columns"].append({"name": col, "kind": kind})

        if kind == "int":
            values = series.astype("Int64")
            payload[f"{col}__values"] = values.fillna(0).astype(np.int64).to_numpy()
            payload[f"{col}__mask"] = values.isna().to_numpy(dtype=bool)
        elif kind == "float":
            payload[f"{col}__values"] = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=np.float32
            )
        elif kind == "bool":
            values = series.astype("boolean")
            payload[f"{col}__values"] = values.fillna(False).to_numpy(dtype=bool)
            payload[f"{col}__mask"] = values.isna().to_numpy(dtype=bool)
        else:
            codes, categories = _encode_string_categories(series)
            payload[f"{col}__codes"] = codes
            payload[f"{col}__categories"] = categories

    payload[_NPZ_TABLE_META_KEY] = np.asarray(json.dumps(meta), dtype=str)
    if cache_metadata is not None:
        embed_meta(payload, cache_metadata)

    np.savez(path, **payload)


def _read_coordinate_npz(path: Path) -> pd.DataFrame:
    """Read a coordinate table from an NPZ archive."""
    with np.load(path, allow_pickle=False) as data:
        try:
            meta_raw = data[_NPZ_TABLE_META_KEY].item()
        except KeyError as exc:
            raise ValueError(
                f"NPZ coordinate cache at {path} is missing {_NPZ_TABLE_META_KEY!r}."
            ) from exc

        meta = json.loads(str(meta_raw))
        if meta.get("version") != _TABLE_META_VERSION:
            raise ValueError(
                f"Unsupported NPZ coordinate-table version: {meta.get('version')!r} "
                f"(expected {_TABLE_META_VERSION})."
            )

        columns: dict[str, Any] = {}
        for entry in meta["columns"]:
            col = entry["name"]
            kind = entry["kind"]

            if kind == "int":
                values = data[f"{col}__values"].astype(np.int64, copy=False)
                mask = data[f"{col}__mask"].astype(bool, copy=False)
                arr = pd.array(values, dtype="Int64")
                arr[mask] = pd.NA
                columns[col] = arr
            elif kind == "float":
                columns[col] = data[f"{col}__values"].astype(np.float32, copy=False)
            elif kind == "bool":
                values = data[f"{col}__values"].astype(bool, copy=False)
                mask = data[f"{col}__mask"].astype(bool, copy=False)
                arr = pd.array(values, dtype="boolean")
                arr[mask] = pd.NA
                columns[col] = arr
            elif kind == "string_category":
                codes = data[f"{col}__codes"].astype(np.int32, copy=False)
                categories = data[f"{col}__categories"].astype(str, copy=False)
                columns[col] = _decode_string_categories(codes, categories)
            else:
                raise ValueError(f"Unsupported serialized column kind {kind!r} in {path}.")

    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# Public load/save API
# ---------------------------------------------------------------------------


def load_coordinate_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load a pre-computed coordinate table from CSV, Parquet, or NPZ."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".npz":
        df = _read_coordinate_npz(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. "
            "Expected '.csv', '.parquet', or '.npz'."
        )

    validate_coordinate_table(df)
    return _coerce_dtypes(df)


def save_coordinate_table(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    cache_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a coordinate table to CSV, Parquet, or NPZ.

    *cache_metadata* (NPZ only) is embedded for cache validation; see
    :func:`confana.cache.matches`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if cache_metadata is not None and suffix != ".npz":
        raise ValueError(f"cache_metadata can only be embedded in an '.npz' file; got {path!s}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".npz":
        _write_coordinate_npz(df, path, cache_metadata=cache_metadata)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. "
            "Expected '.csv', '.parquet', or '.npz'."
        )

    return path


# ---------------------------------------------------------------------------
# Metadata extraction for interactive use
# ---------------------------------------------------------------------------


def build_frame_metadata(frames: list[FrameRecord]) -> list[dict]:
    """Convert a list of FrameRecord objects to lightweight metadata dicts."""
    return [
        {
            "source_file": fr.source_file,
            "frame_number": fr.frame_number,
            "byte_offset": fr.byte_offset,
            "atom_count": fr.atom_count,
            "comment_line": fr.comment_line,
            "trajectory_id": fr.trajectory_id,
            "bead_id": fr.bead_id,
            "local_frame_index": fr.local_frame_index,
            "global_frame_index": fr.global_frame_index,
            "step_number": fr.step_number,
            "bead_comment": fr.bead_comment,
            "energy": fr.energy,
        }
        for fr in frames
    ]
