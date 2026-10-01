"""Structure viewer support for standalone interactive HTML.

Provides the per-bin structures and metadata that ``make_density_interactive``
embeds when ``embed_xyz_payload: true`` is set in the config.

Design notes
------------
* ``build_bin_xyz_payloads`` and ``build_bin_frame_metadata`` both describe
  each bin's representative frame, taken from
  ``ConformationalMap.representatives``, so a bin's structure and metadata
  are always the same frame.  The JavaScript click handler then only needs an
  O(1) key lookup instead of scanning all frames.
* Structures are read through :mod:`confana.frame_source` (xyz or HDF5) and
  aligned as arrays with ``align_frame``; the page receives them as XYZ text.
  A frame that cannot be read stops the build; a representative that names no
  structure leaves its bin without one, never borrowing another frame's.

Public API
----------
- ``align_frame``
- ``build_bin_xyz_payloads``
- ``build_bin_frame_metadata``
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from confana.density import ConformationalMap
from confana.frame_source import names_frame, read_frames
from confana.models import FrameRecord


# ---------------------------------------------------------------------------
# Frame alignment and text
# ---------------------------------------------------------------------------


_HYDROGEN_SYMBOLS = {"H", "D", "T"}


def _xyz_text(frame: FrameRecord) -> str:
    """The frame as XYZ text, coordinates to 8 decimals: the form the page embeds."""
    lines = [str(frame.atom_count), frame.comment_line]
    for symbol, (x, y, z) in zip(frame.elements, frame.coords):
        lines.append(f"{symbol:<2s}  {x: .8f}  {y: .8f}  {z: .8f}")
    return "\n".join(lines) + "\n"


def _resolve_alignment_indices(symbols: tuple[str, ...], atom_selection: str) -> np.ndarray:
    """Return atom indices used for alignment."""
    if atom_selection == "heavy":
        indices = [
            i for i, symbol in enumerate(symbols)
            if symbol.strip().upper() not in _HYDROGEN_SYMBOLS
        ]
    elif atom_selection == "all":
        indices = list(range(len(symbols)))
    else:
        raise ValueError(
            f"Unsupported alignment atom_selection '{atom_selection}'. "
            "Valid choices: ['heavy', 'all']"
        )

    if not indices:
        raise ValueError(f"No atoms available for alignment with atom_selection='{atom_selection}'.")

    return np.asarray(indices, dtype=np.int64)


def _kabsch_rotation(reference_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Return the rigid rotation that aligns target_coords onto reference_coords."""
    ref_centroid = reference_coords.mean(axis=0)
    tgt_centroid = target_coords.mean(axis=0)
    ref_centered = reference_coords - ref_centroid
    tgt_centered = target_coords - tgt_centroid

    covariance = tgt_centered.T @ ref_centered
    v_mat, _, wt_mat = np.linalg.svd(covariance)
    det_sign = np.sign(np.linalg.det(v_mat @ wt_mat))
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = det_sign if det_sign != 0 else 1.0
    return v_mat @ correction @ wt_mat


def align_frame(
    frame: FrameRecord,
    reference: FrameRecord,
    atom_selection: str = "heavy",
) -> FrameRecord:
    """Return *frame* rigidly moved onto *reference* (Kabsch fit on the selected atoms).

    Every atom moves with the fit, in its original order.

    Raises
    ------
    ValueError
        If the frames' atom counts, selected-atom counts or selected element
        orders differ, or *atom_selection* is not ``"heavy"`` / ``"all"``.
    """
    if reference.atom_count != frame.atom_count:
        raise ValueError("Cannot align frames with different atom counts.")

    reference_symbols = tuple(reference.elements)
    target_symbols = tuple(frame.elements)
    reference_selection = _resolve_alignment_indices(reference_symbols, atom_selection)
    target_selection = _resolve_alignment_indices(target_symbols, atom_selection)
    if reference_selection.shape != target_selection.shape:
        raise ValueError(
            "Cannot align frames with different atom counts in the selected atom set."
        )

    if [reference_symbols[i].strip().upper() for i in reference_selection] != [
        target_symbols[i].strip().upper() for i in target_selection
    ]:
        descriptor = "heavy-atom" if atom_selection == "heavy" else "selected-atom"
        raise ValueError(
            f"Cannot align frames with different {descriptor} element order."
        )

    reference_coords = np.asarray(reference.coords, dtype=np.float64)
    target_coords = np.asarray(frame.coords, dtype=np.float64)
    rotation = _kabsch_rotation(
        reference_coords[reference_selection],
        target_coords[target_selection],
    )
    ref_centroid = reference_coords[reference_selection].mean(axis=0)
    tgt_centroid = target_coords[target_selection].mean(axis=0)
    transformed = (target_coords - tgt_centroid) @ rotation + ref_centroid
    return replace(frame, coords=transformed)


def _alignment_reference(df: pd.DataFrame, reference: str) -> FrameRecord:
    """Return the deterministic frame every structure is aligned onto."""
    if reference != "earliest_frame":
        raise ValueError(
            f"Unsupported alignment reference '{reference}'. "
            "Valid choices: ['earliest_frame']"
        )
    eligible = np.flatnonzero(names_frame(df))
    if len(eligible) == 0:
        raise ValueError("No eligible structures available for alignment reference.")
    (frame,) = read_frames(df, [int(eligible[0])])
    if frame is None:  # unreachable: eligible rows name a frame
        raise ValueError("The alignment reference row names no frame.")
    return frame


# ---------------------------------------------------------------------------
# Bin representative selection
# ---------------------------------------------------------------------------

# Metadata fields extracted per bin when building the bin-frame-metadata dict.
_BIN_META_FIELDS = [
    "frame_id",
    "source_file",
    "frame_number",
    "byte_offset",
    "trajectory_id",
    "bead_id",
    "local_frame_index",
    "global_frame_index",
    "carboxyl_plane",
    "ester_plane",
    "carboxyl_dihedral",
    "ester_dihedral",
    "state_plane",
    "state_dihedral",
    "energy",
]


def build_bin_xyz_payloads(
    conf_map: ConformationalMap,
    alignment: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return the XYZ structure of each occupied bin's representative frame.

    The representative is :attr:`ConformationalMap.representatives`, the same
    frame :func:`build_bin_frame_metadata` describes. Frames are read through
    :func:`confana.frame_source.read_frames`, so xyz and HDF5 rows both work.
    A representative that names no structure (coordinate-only input) leaves
    its bin without one.

    Parameters
    ----------
    conf_map:
        Map of the coordinate pair; its table must have ``source_file`` and
        ``byte_offset`` columns.
    alignment:
        Optional alignment config dict.  Alignment is on by default: payloads
        are rigidly aligned to the reference structure using the configured
        reference / atom-selection settings.  ``{"enabled": false}`` keeps
        each structure in its raw trajectory orientation.

    Returns
    -------
    dict[str, str]
        Mapping of ``"xi_yi"`` bin keys to XYZ text strings.

    Raises
    ------
    ValueError
        If a representative frame cannot be read (see
        :func:`~confana.frame_source.read_frames`), or alignment fails.
    """
    df = conf_map.table
    bins = conf_map.representatives
    frames = read_frames(df, list(bins.values())) if bins else []
    by_key = {
        f"{xi}_{yi}": frame for (xi, yi), frame in zip(bins, frames) if frame is not None
    }

    alignment_cfg = alignment or {}
    if bool(alignment_cfg.get("enabled", True)) and by_key:
        reference = _alignment_reference(
            df, reference=str(alignment_cfg.get("reference", "earliest_frame"))
        )
        atom_selection = str(alignment_cfg.get("atom_selection", "heavy"))
        by_key = {
            key: align_frame(frame, reference, atom_selection=atom_selection)
            for key, frame in by_key.items()
        }

    return {key: _xyz_text(frame) for key, frame in by_key.items()}


def build_bin_frame_metadata(
    conf_map: ConformationalMap,
    extra_fields: list[str] | None = None,
) -> dict[str, dict]:
    """Return the metadata record of each occupied bin's representative frame.

    The representative is :attr:`ConformationalMap.representatives`, the same
    frame whose structure :func:`build_bin_xyz_payloads` reads. The result is
    embedded in the interactive HTML as ``bin-frame-metadata`` instead of the
    entire per-frame coordinate table (which can reach GB scale for large
    PIMD datasets).

    Parameters
    ----------
    conf_map:
        Map of the coordinate pair.
    extra_fields:
        Optional additional columns to include in each metadata record.

    Returns
    -------
    dict[str, dict]
        Mapping of ``"xi_yi"`` bin keys to metadata dicts.  Each dict
        contains whichever fields from ``_BIN_META_FIELDS`` are present
        in *df*, with NA values replaced by ``None``.
    """
    df = conf_map.table
    bin_best = conf_map.representatives
    if not bin_best:
        return {}

    present_fields = [f for f in _BIN_META_FIELDS if f in df.columns]
    for field in extra_fields or []:
        if field in df.columns and field not in present_fields:
            present_fields.append(field)
    result: dict[str, dict] = {}
    for (xi, yi), i in bin_best.items():
        row = df.iloc[i]
        record: dict[str, Any] = {}
        for field in present_fields:
            val = row[field]
            try:
                if pd.isna(val):
                    val = None
            except (TypeError, ValueError):
                pass
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            record[field] = val
        result[f"{xi}_{yi}"] = record

    return result
