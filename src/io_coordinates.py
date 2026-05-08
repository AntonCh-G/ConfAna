"""Coordinate-table ingestion and cached precomputed-angle utilities.

This module allows downstream analysis (density, clustering, transitions) to
run without xyz parsing when a coordinate table or precomputed-angle cache is
already available.

Public API
----------
- ``load_coordinate_table``
- ``save_coordinate_table``
- ``load_or_build_coordinate_table_cache``
- ``load_or_build_coordinate_table_from_config``
- ``load_or_build_trajectory_coordinates``
- ``load_or_build_all_coordinates``
- ``build_frame_metadata``
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from src.coordinate_config import (
    list_coordinate_pairs,
    resolve_coordinate_transforms,
    resolve_dof_definitions,
)
from src.models import DoFDefinition, FrameRecord

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
_NPZ_CACHE_META_KEY = "cache_manifest_json"
_TABLE_META_VERSION = 1
_CACHE_META_VERSION = 1


# ---------------------------------------------------------------------------
# Validation and dtype coercion
# ---------------------------------------------------------------------------


def _validate_coordinate_table(df: pd.DataFrame) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
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
        payload[_NPZ_CACHE_META_KEY] = np.asarray(json.dumps(cache_metadata), dtype=str)

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


def _read_embedded_cache_metadata(path: Path) -> dict[str, Any] | None:
    """Return embedded cache metadata from an NPZ archive, if present."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if _NPZ_CACHE_META_KEY not in data:
                return None
            return json.loads(str(data[_NPZ_CACHE_META_KEY].item()))
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None


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

    _validate_coordinate_table(df)
    return _coerce_dtypes(df)


def save_coordinate_table(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Persist a coordinate table to CSV, Parquet, or NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".npz":
        _write_coordinate_npz(df, path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. "
            "Expected '.csv', '.parquet', or '.npz'."
        )

    return path


# ---------------------------------------------------------------------------
# Cached streaming build of precomputed-angle tables
# ---------------------------------------------------------------------------


def _resolve_index_cache_path(
    source_file: Path,
    index_cache_dir: str | Path | None,
) -> Path | None:
    """Return the configured frame-index cache path for one source file."""
    if index_cache_dir is None:
        return None
    return Path(index_cache_dir) / f"{source_file.name}.frameindex.npz"


def _discover_source_files(path_pattern: str) -> list[Path]:
    """Return the sorted list of absolute xyz paths for one path pattern."""
    all_paths = sorted(Path(p).resolve() for p in glob.glob(path_pattern, recursive=True))
    # Keep only .xyz files; the glob may match side-car files (e.g. .frameindex.npz)
    # that share the same stem prefix when no extension is specified in the pattern.
    paths = [p for p in all_paths if p.suffix == ".xyz"]
    if not paths:
        # If the strict xyz filter leaves nothing, fall back to the original set so
        # the error message is still informative (e.g. pattern already ends in .xyz).
        paths = all_paths
    if not paths:
        raise FileNotFoundError(f"No files matched data.path_pattern={path_pattern!r}")
    return paths


def _build_cache_metadata(
    source_files: list[Path],
    *,
    path_pattern: str,
    trajectory_id_pattern: str | None,
    bead_id_pattern: str | None,
    dof_defs: list[DoFDefinition],
    frame_range_cfg: dict | None = None,
) -> dict[str, Any]:
    """Build cache-validation metadata for the current source set."""
    return {
        "version": _CACHE_META_VERSION,
        "path_pattern": path_pattern,
        "trajectory_id_pattern": trajectory_id_pattern,
        "bead_id_pattern": bead_id_pattern,
        "dof_fingerprint": [
            {
                "name": d.name,
                "type": d.type,
                "atoms": list(d.atoms) if d.atoms is not None else None,
                "domain": list(d.domain),
            }
            for d in dof_defs
        ],
        "frame_range": frame_range_cfg,
        "files": [
            {
                "path": str(path),
                "size": os.stat(path).st_size,
                "mtime": os.stat(path).st_mtime,
            }
            for path in source_files
        ],
    }


def _reconstruct_series_from_codes(
    codes: np.ndarray,
    values: list[str],
    *,
    dtype: str | None = None,
) -> pd.Series:
    """Reconstruct a Series from integer category codes."""
    cat = pd.Categorical.from_codes(codes, categories=values)
    if dtype is None:
        return pd.Series(cat)
    return pd.Series(pd.Series(cat.astype(object)).astype(dtype))


def _process_one_file(
    source_file: Path,
    trajectory_id_pattern: str | None,
    bead_id_pattern: str | None,
    index_cache_path: Path | None,
    n_file: int,
    file_n_atoms: int,
    dof_defs: list[DoFDefinition],
    max_frames: int | None = None,
    start_frame: int = 0,
) -> dict:
    """Stream one xyz file, batch-compute geometry, and return result arrays.

    Top-level function (picklable by joblib loky). Stateless.

    Returns
    -------
    dict
        Keys: ``source_file``, ``trajectory_id``, ``bead_id`` (scalars),
        plus ``frame_number``, ``byte_offset``, ``atom_count``,
        ``local_frame_index``, ``step_number``, ``step_missing`` (arrays),
        plus ``dof_values`` dict of float32 arrays keyed by DoF name.
    """
    from src.coordinates import batch_extract_geometry_dof  # noqa: PLC0415
    from src.io_xyz import iter_xyz_frames  # noqa: PLC0415

    file_coords = np.empty((n_file, file_n_atoms, 3), dtype=np.float32)
    frame_number_arr = np.empty(n_file, dtype=np.int64)
    byte_offset_arr = np.empty(n_file, dtype=np.int64)
    atom_count_arr = np.empty(n_file, dtype=np.int64)
    local_frame_idx = np.empty(n_file, dtype=np.int64)
    step_number_arr = np.empty(n_file, dtype=np.int64)
    step_missing_arr = np.zeros(n_file, dtype=bool)

    trajectory_id: str | None = None
    bead_id: str | None = None

    for local_i, frame in enumerate(iter_xyz_frames(
        source_file,
        trajectory_id_pattern=trajectory_id_pattern,
        bead_id_pattern=bead_id_pattern,
        max_frames=max_frames,
        start_frame=start_frame,
    )):
        file_coords[local_i] = frame.coords
        frame_number_arr[local_i] = frame.frame_number
        byte_offset_arr[local_i] = frame.byte_offset
        atom_count_arr[local_i] = frame.atom_count
        local_frame_idx[local_i] = frame.local_frame_index
        if frame.step_number is None:
            step_number_arr[local_i] = 0
            step_missing_arr[local_i] = True
        else:
            step_number_arr[local_i] = frame.step_number
        if local_i == 0:
            trajectory_id = frame.trajectory_id
            bead_id = frame.bead_id

    dof_values = batch_extract_geometry_dof(file_coords, dof_defs)
    del file_coords

    return {
        "source_file": str(source_file),
        "trajectory_id": trajectory_id,
        "bead_id": bead_id,
        "frame_number": frame_number_arr,
        "byte_offset": byte_offset_arr,
        "atom_count": atom_count_arr,
        "local_frame_index": local_frame_idx,
        "step_number": step_number_arr,
        "step_missing": step_missing_arr,
        "dof_values": dof_values,
    }


def _build_coordinate_table_cache(
    source_files: list[Path],
    *,
    dof_defs: list[DoFDefinition],
    trajectory_id_pattern: str | None,
    bead_id_pattern: str | None,
    index_cache_dir: str | Path | None,
    n_jobs: int = 1,
    bond_break_cfg: dict | None = None,
    frame_range_cfg: dict | None = None,
    _timer: "Any | None" = None,
) -> pd.DataFrame:
    """Stream xyz files and build a standard coordinate table in memory.

    Parameters
    ----------
    n_jobs:
        Number of parallel worker processes (passed to ``joblib.Parallel``).
        ``1`` = serial (default, timer-compatible).
        ``-1`` = use all available cores.
        Per-stage timer breakdowns are only reported in serial mode (n_jobs=1).
    _timer:
        Optional :class:`src.bench.StageTimer` instance.  When provided and
        ``n_jobs=1``, fine-grained wall-time measurements are recorded.
        Ignored when ``n_jobs != 1``.
    """
    from src.io_xyz import load_or_build_xyz_index  # noqa: PLC0415

    dof_names = [d.name for d in dof_defs if d.enabled]

    # ------------------------------------------------------------------
    # Frame range — optional user restriction [start_frame, end_frame]
    # ------------------------------------------------------------------
    user_start: int = 0
    user_end: Optional[int] = None
    if frame_range_cfg:
        user_start = int(frame_range_cfg.get("start_frame") or 0)
        user_end_raw = frame_range_cfg.get("end_frame")
        user_end = int(user_end_raw) if user_end_raw is not None else None
    if user_end is not None and user_end < user_start:
        raise ValueError(
            f"frame_range.end_frame ({user_end}) must be >= "
            f"frame_range.start_frame ({user_start})"
        )

    # ------------------------------------------------------------------
    # Pass 1: build/load frame indices; collect frame counts and atom counts
    # ------------------------------------------------------------------
    frame_counts: list[int] = []
    atom_counts_per_file: list[int] = []
    if _timer is not None:
        _timer.start("index_loading")
    for source_file in source_files:
        _index_label = f"index:{source_file.name}" if _timer is not None else ""
        if _timer is not None:
            _timer.start(_index_label)
        frame_index = load_or_build_xyz_index(
            source_file,
            cache_path=_resolve_index_cache_path(source_file, index_cache_dir),
        )
        if _timer is not None:
            _timer.stop(_index_label)
        frame_counts.append(len(frame_index))
        # Read atom count from the first frame entry (uniform across trajectory)
        file_atom_count = frame_index.entries[0].atom_count if frame_index.entries else 0
        atom_counts_per_file.append(file_atom_count)
    if _timer is not None:
        _timer.stop("index_loading")

    # ------------------------------------------------------------------
    # Pass 1.5: bond-break detection — per-trajectory minimum frame limit
    # ------------------------------------------------------------------
    import re as _re  # noqa: PLC0415

    effective_limits: list[int] = list(frame_counts)

    # Apply user end_frame cap before bond-break scan
    if user_end is not None:
        for i in range(len(effective_limits)):
            effective_limits[i] = min(effective_limits[i], user_end + 1)

    if bond_break_cfg and bond_break_cfg.get("enabled", False):
        from src.bond_check import find_bond_break_frame  # noqa: PLC0415

        bond_cutoff = float(bond_break_cfg.get("cutoff", 2.0))

        def _traj_id(path: Path) -> str:
            stem = path.stem
            if trajectory_id_pattern:
                m = _re.match(trajectory_id_pattern, stem)
                if m and m.lastindex:
                    return m.group(1)
            return path.parent.name or stem

        orig_counts = list(frame_counts)

        # Scan files in parallel (same n_jobs as geometry pass) or serially.
        if n_jobs != 1 and len(source_files) > 1:
            from joblib import Parallel, delayed  # noqa: PLC0415
            raw_or_none = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(find_bond_break_frame)(sf, bond_cutoff, user_start, user_end)
                for sf in source_files
            )
        else:
            raw_or_none = [
                find_bond_break_frame(sf, bond_cutoff, user_start, user_end)
                for sf in source_files
            ]

        raw_break: list[int] = [
            br if br is not None else fc
            for br, fc in zip(raw_or_none, orig_counts)
        ]

        traj_ids = [_traj_id(sf) for sf in source_files]
        traj_min: dict[str, int] = {}
        for tid, br in zip(traj_ids, raw_break):
            traj_min[tid] = min(traj_min.get(tid, br), br)

        for i, (sf, orig_fc) in enumerate(zip(source_files, orig_counts)):
            eff = traj_min[_traj_id(sf)]
            # Re-apply user end_frame cap (traj_min may have used orig_counts as fallback)
            if user_end is not None:
                eff = min(eff, user_end + 1)
            effective_limits[i] = eff
            if eff < orig_fc:
                print(f"    bond-break: {sf.name} truncated at frame {eff} (of {orig_fc})")

    # Compute per-file frame counts accounting for both effective_limits and user_start
    for i in range(len(frame_counts)):
        frame_counts[i] = max(0, effective_limits[i] - user_start)

    total_frames = sum(frame_counts)
    if total_frames == 0:
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS + dof_names + OPTIONAL_COLUMNS)
        return _coerce_dtypes(empty)

    # ------------------------------------------------------------------
    # Pass 1b: allocate global output arrays
    # ------------------------------------------------------------------
    if _timer is not None:
        _timer.start("array_alloc")
    frame_number = np.empty(total_frames, dtype=np.int64)
    byte_offset = np.empty(total_frames, dtype=np.int64)
    atom_count = np.empty(total_frames, dtype=np.int64)
    local_frame_index = np.empty(total_frames, dtype=np.int64)
    global_frame_index = np.arange(total_frames, dtype=np.int64)

    dof_arrays = {name: np.empty(total_frames, dtype=np.float32) for name in dof_names}

    step_number = np.empty(total_frames, dtype=np.int64)
    step_missing = np.zeros(total_frames, dtype=bool)

    source_codes = np.empty(total_frames, dtype=np.int32)
    trajectory_codes = np.empty(total_frames, dtype=np.int32)
    bead_codes = np.empty(total_frames, dtype=np.int32)

    source_values: list[str] = []
    trajectory_values: list[str] = []
    bead_values: list[str] = []

    source_map: dict[str, int] = {}
    trajectory_map: dict[str, int] = {}
    bead_map: dict[str, int] = {}
    if _timer is not None:
        _timer.stop("array_alloc")

    def intern(mapping_dict: dict[str, int], values: list[str], raw_value: str | None) -> int:
        if raw_value is None:
            return -1
        value = str(raw_value)
        code = mapping_dict.get(value)
        if code is None:
            code = len(values)
            mapping_dict[value] = code
            values.append(value)
        return code

    # ------------------------------------------------------------------
    # Pass 2: per-file geometry — serial (n_jobs=1) or parallel (n_jobs≠1)
    # ------------------------------------------------------------------
    if n_jobs == 1:
        # Serial path: preserves _timer stage breakdown
        from src.coordinates import batch_extract_geometry_dof  # noqa: PLC0415
        from src.io_xyz import iter_xyz_frames  # noqa: PLC0415

        file_start = 0
        for file_idx, source_file in enumerate(source_files):
            n_file = frame_counts[file_idx]
            file_n_atoms = atom_counts_per_file[file_idx]
            file_end = file_start + n_file

            file_coords = np.empty((n_file, file_n_atoms, 3), dtype=np.float32)

            if _timer is not None:
                _timer.start("frame_iter")

            local_i = 0
            for frame in iter_xyz_frames(
                source_file,
                trajectory_id_pattern=trajectory_id_pattern,
                bead_id_pattern=bead_id_pattern,
                max_frames=effective_limits[file_idx],
                start_frame=user_start,
            ):
                file_coords[local_i] = frame.coords
                pos = file_start + local_i
                frame_number[pos] = frame.frame_number
                byte_offset[pos] = frame.byte_offset
                atom_count[pos] = frame.atom_count
                local_frame_index[pos] = frame.local_frame_index
                if frame.step_number is None:
                    step_number[pos] = 0
                    step_missing[pos] = True
                else:
                    step_number[pos] = frame.step_number
                source_codes[pos] = intern(source_map, source_values, frame.source_file)
                trajectory_codes[pos] = intern(
                    trajectory_map, trajectory_values, frame.trajectory_id
                )
                bead_codes[pos] = intern(bead_map, bead_values, frame.bead_id)
                local_i += 1

            if _timer is not None:
                _timer.stop("frame_iter")
                _timer.start("batch_geometry")

            dof_res = batch_extract_geometry_dof(file_coords, dof_defs)

            if _timer is not None:
                _timer.stop("batch_geometry")
                _timer.start("array_write")

            for name, values in dof_res.items():
                if name in dof_arrays:
                    dof_arrays[name][file_start:file_end] = values

            if _timer is not None:
                _timer.stop("array_write")

            del file_coords
            file_start = file_end

    else:
        # Parallel path: one loky worker process per file
        from joblib import Parallel, delayed  # noqa: PLC0415

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_process_one_file)(
                source_file,
                trajectory_id_pattern,
                bead_id_pattern,
                _resolve_index_cache_path(source_file, index_cache_dir),
                frame_counts[file_idx],
                atom_counts_per_file[file_idx],
                dof_defs,
                effective_limits[file_idx],
                user_start,
            )
            for file_idx, source_file in enumerate(source_files)
        )

        # Merge results (preserved in source_files order by joblib)
        file_start = 0
        for file_idx, result in enumerate(results):
            n_file = frame_counts[file_idx]
            file_end = file_start + n_file

            frame_number[file_start:file_end] = result["frame_number"]
            byte_offset[file_start:file_end] = result["byte_offset"]
            atom_count[file_start:file_end] = result["atom_count"]
            local_frame_index[file_start:file_end] = result["local_frame_index"]
            step_number[file_start:file_end] = result["step_number"]
            step_missing[file_start:file_end] = result["step_missing"]

            src_code = intern(source_map, source_values, result["source_file"])
            traj_code = intern(trajectory_map, trajectory_values, result["trajectory_id"])
            bead_code = intern(bead_map, bead_values, result["bead_id"])
            source_codes[file_start:file_end] = src_code
            trajectory_codes[file_start:file_end] = traj_code
            bead_codes[file_start:file_end] = bead_code

            for name, values in result["dof_values"].items():
                if name in dof_arrays:
                    dof_arrays[name][file_start:file_end] = values

            file_start = file_end

    if _timer is not None:
        _timer.start("df_assembly")
    source_series = _reconstruct_series_from_codes(source_codes, source_values)
    trajectory_series = _reconstruct_series_from_codes(trajectory_codes, trajectory_values)
    bead_series = _reconstruct_series_from_codes(bead_codes, bead_values, dtype="string")
    comment_series = pd.Series(pd.Categorical.from_codes(
        np.zeros(total_frames, dtype=np.int8),
        categories=[""],
    ))

    data: dict[str, Any] = {
        "frame_id": pd.Series(global_frame_index, dtype="Int64"),
        "source_file": source_series,
        "trajectory_id": trajectory_series,
        "bead_id": bead_series,
        "frame_number": pd.Series(frame_number, dtype="Int64"),
        "byte_offset": pd.Series(byte_offset, dtype="Int64"),
        "atom_count": pd.Series(atom_count, dtype="Int64"),
        "comment_line": comment_series,
        "local_frame_index": pd.Series(local_frame_index, dtype="Int64"),
        "global_frame_index": pd.Series(global_frame_index, dtype="Int64"),
    }
    for name, arr in dof_arrays.items():
        data[name] = pd.Series(arr.astype(np.float32), dtype="float32")
    data.update(
        {
            "energy": pd.Series(np.full(total_frames, np.nan), dtype="float32"),
            "step_number": pd.Series(step_number, dtype="Int64"),
        }
    )
    df = pd.DataFrame(data)
    df.loc[step_missing, "step_number"] = pd.NA
    if _timer is not None:
        _timer.stop("df_assembly")
    return df


def load_or_build_coordinate_table_cache(
    *,
    path_pattern: str,
    dof_defs: list[DoFDefinition],
    cache_path: str | Path,
    trajectory_id_pattern: str | None = None,
    bead_id_pattern: str | None = None,
    index_cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    n_jobs: int = 1,
    bond_break_cfg: dict | None = None,
    frame_range_cfg: dict | None = None,
    _timer: "Any | None" = None,
) -> tuple[pd.DataFrame, bool]:
    """Load a cached NPZ coordinate table or stream-build it from xyz files.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(df, cache_hit)`` where ``cache_hit`` is ``True`` when the existing
        cache was reused without rebuilding.
    """
    cache_path = Path(cache_path)
    if cache_path.suffix.lower() != ".npz":
        raise ValueError(
            f"Coordinate-table cache must use a '.npz' path; got {cache_path!s}"
        )

    source_files = _discover_source_files(path_pattern)
    expected_metadata = _build_cache_metadata(
        source_files,
        path_pattern=path_pattern,
        trajectory_id_pattern=trajectory_id_pattern,
        bead_id_pattern=bead_id_pattern,
        dof_defs=dof_defs,
        frame_range_cfg=frame_range_cfg,
    )

    if not force_rebuild and cache_path.exists():
        cached_metadata = _read_embedded_cache_metadata(cache_path)
        if cached_metadata == expected_metadata:
            return load_coordinate_table(cache_path), True

    df = _build_coordinate_table_cache(
        source_files,
        dof_defs=dof_defs,
        trajectory_id_pattern=trajectory_id_pattern,
        bead_id_pattern=bead_id_pattern,
        index_cache_dir=index_cache_dir,
        n_jobs=n_jobs,
        bond_break_cfg=bond_break_cfg,
        frame_range_cfg=frame_range_cfg,
        _timer=_timer,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_coordinate_npz(df, cache_path, cache_metadata=expected_metadata)
    return df, False


def _reset_state_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Reset state columns for all configured coordinate pairs to NA (in-place)."""
    for _name, pair in list_coordinate_pairs(config):
        col = pair.state_col
        if col in df.columns:
            df[col] = pd.NA


def load_or_build_coordinate_table_from_config(
    config: dict[str, Any],
    *,
    force_rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Load or build the standard coordinate table using the project config.

    This is the normal entrypoint for the xyz-backed workflow: on the first
    run it streams the configured xyz files, computes the angles, and writes
    the NPZ cache. On later runs it reloads the same standard coordinate table
    from cache and clears Phase 8 state columns so clustering is always
    recomputed from the current settings.

    Parameters
    ----------
    config:
        Parsed `configs/default.yaml` dictionary.
    force_rebuild_cache:
        When True, bypass any existing NPZ cache and rebuild it from xyz files.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(df, cache_hit)`` where ``cache_hit`` is ``True`` when the NPZ cache
        was reused without rebuilding.
    """
    cache_cfg = config.get("cache", {})

    dof_defs = resolve_dof_definitions(config)

    trajectory_cache_dir = cache_cfg.get("trajectory_cache_dir")
    if trajectory_cache_dir:
        # Per-trajectory path: one NPZ per trajectory, supports bond-break truncation.
        df, cache_hit = load_or_build_all_coordinates(config, force_rebuild=force_rebuild_cache)
        _reset_state_columns(df, config)
        return df, cache_hit

    # Legacy: single global coordinate table cache.
    data_cfg = config.get("data", {})
    cache_path = cache_cfg.get("coordinate_table_path")
    if not cache_path:
        raise ValueError(
            "cache.coordinate_table_path must be set in config to use the "
            "automatic precomputed-angle cache."
        )

    df, cache_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(data_cfg["path_pattern"]),
        dof_defs=dof_defs,
        cache_path=cache_path,
        trajectory_id_pattern=data_cfg.get("trajectory_id_pattern"),
        bead_id_pattern=data_cfg.get("bead_id_pattern"),
        index_cache_dir=cache_cfg.get("index_cache_dir"),
        force_rebuild=force_rebuild_cache,
        n_jobs=int(cache_cfg.get("n_jobs", 1)),
        bond_break_cfg=config.get("bond_break"),
        frame_range_cfg=config.get("frame_range"),
    )

    # Reset state columns in-place; avoid a full df.copy() for a 1–2 GB table.
    _reset_state_columns(df, config)

    # Apply any configured angular shifts as post-processing (not cached).
    transforms = resolve_coordinate_transforms(config)
    if transforms:
        from src.coordinates import apply_coordinate_shifts  # avoid circular import at module level

        df = apply_coordinate_shifts(df, transforms)

    return df, cache_hit


# ---------------------------------------------------------------------------
# Per-trajectory coordinate caching
# ---------------------------------------------------------------------------


def _group_source_files_by_trajectory(
    source_files: list[Path],
    trajectory_id_pattern: str | None,
    bead_id_pattern: str | None,  # noqa: ARG001 — reserved for future per-bead grouping
) -> dict[str, list[Path]]:
    """Group source xyz files by trajectory_id.

    Returns
    -------
    dict[str, list[Path]]
        Insertion-ordered dict mapping trajectory_id → sorted list of paths.
    """
    import re

    groups: dict[str, list[Path]] = {}
    for path in source_files:
        stem = path.stem
        if trajectory_id_pattern:
            m = re.match(trajectory_id_pattern, stem)
            traj_id = m.group(1) if m else stem
        else:
            traj_id = stem
        groups.setdefault(traj_id, []).append(path)

    return {traj_id: sorted(files) for traj_id, files in groups.items()}


def _per_trajectory_cache_meta(
    trajectory_id: str,
    source_files: list[Path],
    dof_defs: list[DoFDefinition],
    frame_range_cfg: dict | None = None,
) -> dict[str, Any]:
    """Build the cache-validation fingerprint for one trajectory.

    Includes the trajectory_id, DoF definitions, and per-file stats
    (path, size, mtime) for every source file.
    """
    from src.cache import fingerprint_files, _CACHE_META_VERSION  # noqa: PLC0415

    return {
        "version": _CACHE_META_VERSION,
        "trajectory_id": trajectory_id,
        "dof_fingerprint": [
            {
                "name": d.name,
                "type": d.type,
                "atoms": list(d.atoms) if d.atoms is not None else None,
                "domain": list(d.domain),
            }
            for d in dof_defs
        ],
        "frame_range": frame_range_cfg,
        "files": fingerprint_files(source_files),
    }


def load_or_build_trajectory_coordinates(
    trajectory_id: str,
    source_files: list[Path],
    *,
    dof_defs: list[DoFDefinition],
    cache_dir: str | Path,
    trajectory_id_pattern: str | None = None,
    bead_id_pattern: str | None = None,
    index_cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    n_jobs: int = 1,
    bond_break_cfg: dict | None = None,
    frame_range_cfg: dict | None = None,
    _timer: "Any | None" = None,
) -> tuple[pd.DataFrame, bool]:
    """Load or build the coordinate cache for one trajectory.

    The cache file is stored at::

        {cache_dir}/{trajectory_id}__coordinates.npz

    The fingerprint covers the source file stats and DoF definitions.
    If any source file changes (size or mtime), only this
    trajectory's cache is rebuilt.

    Parameters
    ----------
    trajectory_id:
        Trajectory identifier; used to name the cache file.
    source_files:
        Ordered list of xyz file paths that belong to this trajectory
        (e.g. all bead files for one PIMD run).
    dof_defs:
        Enabled DoF definitions (from ``resolve_dof_definitions(config)``).
    cache_dir:
        Directory where the per-trajectory NPZ is stored.
    trajectory_id_pattern:
        Regex passed to ``iter_xyz_frames`` for trajectory_id extraction.
    bead_id_pattern:
        Regex passed to ``iter_xyz_frames`` for bead_id extraction.
    index_cache_dir:
        Directory for frame-index JSON caches (passed to xyz loader).
    force_rebuild:
        When True, bypass any existing cache.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(df, cache_hit)`` where ``cache_hit`` is True when the existing
        NPZ was reused without rebuilding.
    """
    from src.cache import matches  # noqa: PLC0415

    cache_dir = Path(cache_dir)
    safe_id = trajectory_id.replace("/", "_").replace(" ", "_")
    cache_path = cache_dir / f"{safe_id}__coordinates.npz"

    expected_meta = _per_trajectory_cache_meta(
        trajectory_id, source_files, dof_defs,
        frame_range_cfg=frame_range_cfg,
    )

    if not force_rebuild and cache_path.exists() and matches(cache_path, expected_meta):
        df = _read_coordinate_npz(cache_path)
        return _coerce_dtypes(df), True

    df = _build_coordinate_table_cache(
        source_files,
        dof_defs=dof_defs,
        trajectory_id_pattern=trajectory_id_pattern,
        bead_id_pattern=bead_id_pattern,
        index_cache_dir=index_cache_dir,
        n_jobs=n_jobs,
        bond_break_cfg=bond_break_cfg,
        frame_range_cfg=frame_range_cfg,
        _timer=_timer,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_coordinate_npz(df, cache_path, cache_metadata=expected_meta)
    return df, False


def load_or_build_all_coordinates(
    config: dict[str, Any],
    *,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Load or build per-trajectory coordinate caches and return the full table.

    One NPZ file is written per trajectory_id under
    ``cache.trajectory_cache_dir``.  On each run only trajectories whose
    source files have changed are recomputed; all others are read from cache.

    Parameters
    ----------
    config:
        Parsed ``configs/default.yaml`` dictionary.
    force_rebuild:
        When True, bypass all existing caches and rebuild from xyz files.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(df, all_cache_hit)`` where ``all_cache_hit`` is True only when
        every trajectory was served from cache.

    Raises
    ------
    ValueError
        If ``cache.trajectory_cache_dir`` is not set in config.
    """
    data_cfg = config.get("data", {})
    cache_cfg = config.get("cache", {})

    trajectory_cache_dir = cache_cfg.get("trajectory_cache_dir")
    if not trajectory_cache_dir:
        raise ValueError(
            "cache.trajectory_cache_dir must be set in config to use "
            "per-trajectory coordinate caching."
        )

    source_files = _discover_source_files(str(data_cfg["path_pattern"]))
    traj_groups = _group_source_files_by_trajectory(
        source_files,
        data_cfg.get("trajectory_id_pattern"),
        data_cfg.get("bead_id_pattern"),
    )

    frames: list[pd.DataFrame] = []
    all_cache_hit = True

    dof_defs = resolve_dof_definitions(config)
    bond_break_cfg = config.get("bond_break")
    frame_range_cfg = config.get("frame_range")
    for traj_id, files in traj_groups.items():
        df_traj, hit = load_or_build_trajectory_coordinates(
            traj_id,
            files,
            dof_defs=dof_defs,
            cache_dir=trajectory_cache_dir,
            trajectory_id_pattern=data_cfg.get("trajectory_id_pattern"),
            bead_id_pattern=data_cfg.get("bead_id_pattern"),
            index_cache_dir=cache_cfg.get("index_cache_dir"),
            force_rebuild=force_rebuild,
            n_jobs=int(cache_cfg.get("n_jobs", 1)),
            bond_break_cfg=bond_break_cfg,
            frame_range_cfg=frame_range_cfg,
        )
        frames.append(df_traj)
        if not hit:
            all_cache_hit = False

    if not frames:
        dof_names = [d.name for d in dof_defs if d.enabled]
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS + dof_names + OPTIONAL_COLUMNS)
        return _coerce_dtypes(empty), True

    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    df["frame_id"] = pd.array(range(n), dtype="Int64")
    df["global_frame_index"] = pd.array(range(n), dtype="Int64")

    # Apply any configured angular shifts as post-processing (not cached).
    transforms = resolve_coordinate_transforms(config)
    if transforms:
        from src.coordinates import apply_coordinate_shifts  # avoid circular import at module level

        df = apply_coordinate_shifts(df, transforms)

    return df, all_cache_hit


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
