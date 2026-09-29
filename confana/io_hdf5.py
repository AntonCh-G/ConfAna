"""HDF5 trajectory reader for PIMD trajectories.

Expected HDF5 layout (trajectory.hdf5):
    bead_positions : (n_frames, n_beads, n_atoms, 3)  float64  Å
    positions      : (n_frames, n_atoms, 3)            float64  Å  (centroid)
    potential      : (n_frames,)                       float64  eV

Atom types are read from input.xyz in the simulation directory (parent of hdf5/).
# TODO-hdf5-elements: when HDF5 format includes element symbols, read them
# from the file instead of from input.xyz.

Public API
----------
- ``build_coordinate_table_from_hdf5``
- ``build_coordinate_table_from_hdf5_files``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from confana.models import DoFDefinition

logger = logging.getLogger(__name__)

_SENTINEL_BYTE_OFFSET: int = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_atom_types_from_input_xyz(h5_path: Path) -> list[str]:
    """Read element symbols from input.xyz beside the HDF5 file's parent dir.

    Layout on disk::

        <sim_dir>/hdf5/trajectory.hdf5   ← h5_path
        <sim_dir>/input.xyz                   ← read from here
    """
    input_xyz = h5_path.parent.parent / "input.xyz"
    if not input_xyz.exists():
        raise FileNotFoundError(
            f"Cannot find atom-type source at {input_xyz}. "
            "Each simulation directory must contain input.xyz alongside hdf5/."
        )
    elements: list[str] = []
    with input_xyz.open("rb") as fh:
        n_atoms = int(fh.readline().strip())
        fh.readline()  # skip comment
        for _ in range(n_atoms):
            raw = fh.readline().decode(errors="replace").strip()
            if not raw:
                raise ValueError(
                    f"Unexpected end of {input_xyz} while reading atom types"
                )
            elements.append(raw.split()[0])
    if len(elements) != n_atoms:
        raise ValueError(
            f"{input_xyz}: expected {n_atoms} atom lines, got {len(elements)}"
        )
    return elements


def _derive_trajectory_id(h5_path: Path) -> str:
    """Derive trajectory_id from the simulation directory name (e.g. 's1').

    For path ``/some/root/s1/hdf5/trajectory.hdf5`` returns ``'s1'``.
    """
    return h5_path.parent.parent.name


# ---------------------------------------------------------------------------
# Per-file builder
# ---------------------------------------------------------------------------


def build_coordinate_table_from_hdf5(
    h5_path: Path | str,
    dof_defs: list[DoFDefinition],
    *,
    positions_source: Literal["bead", "centroid"] = "bead",
    global_frame_offset: int = 0,
) -> pd.DataFrame:
    """Load one HDF5 file and return a standard coordinate table.

    The full position array is loaded into RAM once, DoF are computed
    per-bead (or for the centroid), then the array is released.

    Parameters
    ----------
    h5_path:
        Path to ``trajectory.hdf5``.
    dof_defs:
        DoF definitions (from config).
    positions_source:
        ``"bead"`` — one row per (frame × bead); ``"centroid"`` — one row per frame.
    global_frame_offset:
        Start value for ``global_frame_index``; used when concatenating
        multiple files.

    Returns
    -------
    pd.DataFrame
        Standard coordinate table (same schema as the xyz-backed table).
        ``byte_offset`` is set to ``-1`` (HDF5 frames are accessed by index).
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 trajectories. Install with: uv add h5py"
        ) from exc

    from confana.coordinates import batch_extract_geometry_dof  # noqa: PLC0415

    h5_path = Path(h5_path)
    trajectory_id = _derive_trajectory_id(h5_path)
    atom_types = _read_atom_types_from_input_xyz(h5_path)

    with h5py.File(h5_path, "r") as fh:
        bead_pos_raw: np.ndarray = fh["bead_positions"][()].astype(np.float32)
        centroid_pos_raw: np.ndarray = fh["positions"][()].astype(np.float32)
        potential: np.ndarray = fh["potential"][()].astype(np.float32)

    n_frames, n_beads_total, n_atoms, _ = bead_pos_raw.shape

    if n_atoms != len(atom_types):
        raise ValueError(
            f"{h5_path.name}: HDF5 has {n_atoms} atoms per frame but "
            f"input.xyz has {len(atom_types)} atom-type entries."
        )

    logger.info(
        "HDF5 %s: %d frames, %d beads, %d atoms (trajectory_id=%r, source=%s)",
        h5_path.name, n_frames, n_beads_total, n_atoms, trajectory_id, positions_source,
    )

    dof_names = [d.name for d in dof_defs if d.enabled]

    if positions_source == "bead":
        bead_labels: list[str | None] = [f"bead_{i:02d}" for i in range(n_beads_total)]
    else:
        bead_labels = [None]  # centroid: single pass

    n_slices = len(bead_labels)
    n_rows = n_frames * n_slices

    # Pre-allocate output arrays
    frame_number_out = np.empty(n_rows, dtype=np.int64)
    local_frame_index_out = np.empty(n_rows, dtype=np.int64)
    global_frame_index_out = np.empty(n_rows, dtype=np.int64)
    step_number_out = np.empty(n_rows, dtype=np.int64)
    byte_offset_out = np.full(n_rows, _SENTINEL_BYTE_OFFSET, dtype=np.int64)
    atom_count_out = np.full(n_rows, n_atoms, dtype=np.int64)
    energy_out = np.empty(n_rows, dtype=np.float32)

    source_file_codes = np.zeros(n_rows, dtype=np.int32)
    trajectory_codes = np.zeros(n_rows, dtype=np.int32)
    bead_codes = np.empty(n_rows, dtype=np.int32)

    dof_arrays: dict[str, np.ndarray] = {
        name: np.empty(n_rows, dtype=np.float32) for name in dof_names
    }

    frame_idx_arr = np.arange(n_frames, dtype=np.int64)

    for slice_idx, bead_label in enumerate(bead_labels):
        row_start = slice_idx * n_frames
        row_end = row_start + n_frames

        if positions_source == "bead":
            coords = bead_pos_raw[:, slice_idx, :, :]  # (n_frames, n_atoms, 3)
        else:
            coords = centroid_pos_raw  # (n_frames, n_atoms, 3)

        dof_values = batch_extract_geometry_dof(coords, dof_defs)
        for name in dof_names:
            if name in dof_values:
                dof_arrays[name][row_start:row_end] = dof_values[name]

        frame_number_out[row_start:row_end] = frame_idx_arr
        local_frame_index_out[row_start:row_end] = frame_idx_arr
        global_frame_index_out[row_start:row_end] = (
            frame_idx_arr + global_frame_offset + slice_idx * n_frames
        )
        step_number_out[row_start:row_end] = frame_idx_arr
        energy_out[row_start:row_end] = potential
        bead_codes[row_start:row_end] = slice_idx if bead_label is not None else -1

    del bead_pos_raw, centroid_pos_raw

    source_file_str = str(h5_path)
    bead_cat_values = bead_labels  # one value per slice

    # Build categorical bead_id column
    bead_id_raw: list[str | None] = []
    for bead_label in bead_labels:
        bead_id_raw.extend([bead_label] * n_frames)

    data: dict = {
        "frame_id": pd.array(
            range(global_frame_offset, global_frame_offset + n_rows), dtype="Int64"
        ),
        "source_file": pd.Categorical([source_file_str] * n_rows),
        "trajectory_id": pd.Categorical([trajectory_id] * n_rows),
        "bead_id": pd.array(bead_id_raw, dtype="string"),
        "frame_number": pd.array(frame_number_out, dtype="Int64"),
        "byte_offset": pd.array(byte_offset_out, dtype="Int64"),
        "atom_count": pd.array(atom_count_out, dtype="Int64"),
        "comment_line": pd.Categorical([""] * n_rows, categories=[""]),
        "local_frame_index": pd.array(local_frame_index_out, dtype="Int64"),
        "global_frame_index": pd.array(global_frame_index_out, dtype="Int64"),
    }
    for name in dof_names:
        data[name] = pd.array(dof_arrays[name], dtype="float32")
    data["energy"] = pd.array(energy_out, dtype="float32")
    data["step_number"] = pd.array(step_number_out, dtype="Int64")

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Multi-file entry point
# ---------------------------------------------------------------------------


def build_coordinate_table_from_hdf5_files(
    h5_paths: list[Path | str],
    dof_defs: list[DoFDefinition],
    *,
    positions_source: Literal["bead", "centroid"] = "bead",
) -> pd.DataFrame:
    """Process a list of HDF5 trajectory files and return one coordinate table.

    Files are processed sequentially (each is loaded fully into RAM, then
    released before the next file is opened). The resulting tables are
    concatenated in file order.

    Parameters
    ----------
    h5_paths:
        Ordered list of HDF5 file paths (e.g. s0, s1, s2, …).
    dof_defs:
        DoF definitions (from config).
    positions_source:
        Forwarded to :func:`build_coordinate_table_from_hdf5`.

    Returns
    -------
    pd.DataFrame
        Combined coordinate table with sequential ``frame_id`` and
        ``global_frame_index``.
    """
    if not h5_paths:
        raise ValueError("h5_paths is empty — check data.path_pattern in config.")

    frames: list[pd.DataFrame] = []
    global_offset = 0
    for h5_path in h5_paths:
        df = build_coordinate_table_from_hdf5(
            h5_path,
            dof_defs,
            positions_source=positions_source,
            global_frame_offset=global_offset,
        )
        global_offset += len(df)
        frames.append(df)
        logger.info("Processed %s: %d rows", Path(h5_path).name, len(df))

    result = pd.concat(frames, ignore_index=True)
    # Re-assign frame_id to be strictly sequential 0..N-1
    result["frame_id"] = pd.array(range(len(result)), dtype="Int64")
    return result


# ---------------------------------------------------------------------------
# Utilities for cache integration
# ---------------------------------------------------------------------------


def discover_hdf5_files(path_pattern: str) -> list[Path]:
    """Glob for HDF5 files matching ``path_pattern``.

    Only files with ``.hdf5`` or ``.h5`` suffix are returned.
    Raises :class:`FileNotFoundError` if no files match.
    """
    import glob as _glob

    all_paths = sorted(
        Path(p).resolve()
        for p in _glob.glob(path_pattern, recursive=True)
    )
    paths = [p for p in all_paths if p.suffix.lower() in {".hdf5", ".h5"}]
    if not paths:
        paths = all_paths  # keep for informative error
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files matched data.path_pattern={path_pattern!r}"
        )
    return paths


def build_hdf5_cache_metadata(
    h5_paths: list[Path],
    *,
    path_pattern: str,
    positions_source: str,
    dof_defs: list[DoFDefinition],
) -> dict:
    """Build cache-validation metadata for an HDF5 source set."""
    return {
        "format": "hdf5",
        "version": 1,
        "path_pattern": path_pattern,
        "positions_source": positions_source,
        "dof_fingerprint": [
            {
                "name": d.name,
                "type": d.type,
                "atoms": list(d.atoms) if d.atoms is not None else None,
                "domain": list(d.domain),
            }
            for d in dof_defs
            if d.enabled
        ],
        "files": [
            {
                "path": str(p),
                "size": os.stat(p).st_size,
                "mtime": os.stat(p).st_mtime,
            }
            for p in h5_paths
        ],
    }
