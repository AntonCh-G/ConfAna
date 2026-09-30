"""Structure viewer support for standalone interactive HTML.

Provides Python-side helpers that read raw XYZ frame text from disk and
pre-compute one representative structure per histogram bin.  These are
consumed by ``make_density_interactive`` when ``embed_xyz_payload: true``
is set in the config.

Design notes
------------
* ``read_xyz_frame_text`` returns the raw multi-line text of a single XYZ
  frame, suitable for direct embedding in the HTML or passing to 3Dmol.js
  ``viewer.addModel(text, 'xyz')``.
* ``build_bin_xyz_payloads`` and ``build_bin_frame_metadata`` both describe
  each bin's representative frame, taken from
  ``ConformationalMap.representatives``, so a bin's structure and metadata
  are always the same frame.  The JavaScript click handler then only needs an
  O(1) key lookup instead of scanning all frames.
* A representative without a readable structure leaves its bin without one;
  no structure is borrowed from another frame.

Public API
----------
- ``read_xyz_frame_text``
- ``align_xyz_to_reference``
- ``build_bin_xyz_payloads``
- ``build_bin_frame_metadata``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from confana.density import ConformationalMap


# ---------------------------------------------------------------------------
# Frame text retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedXYZ:
    """Parsed XYZ payload."""

    atom_count: int
    comment: str
    symbols: tuple[str, ...]
    coords: np.ndarray


_HYDROGEN_SYMBOLS = {"H", "D", "T"}


def read_xyz_frame_text(
    source_file: str | Path,
    byte_offset: int,
    atom_count: int,
) -> str:
    """Read one XYZ frame from *source_file* starting at *byte_offset*.

    Opens the file in binary mode, seeks to the given byte position, reads
    ``atom_count + 2`` lines (the atom-count header, the comment line, and
    all atom records), and returns the decoded text.

    Parameters
    ----------
    source_file:
        Path to the xyz file.
    byte_offset:
        Byte position of the atom-count line for the desired frame (as
        stored in the coordinate table's ``byte_offset`` column).
    atom_count:
        Number of atoms in the frame (used to determine how many lines to
        read).

    Returns
    -------
    str
        Raw text of the frame (atom_count + 2 lines).

    Raises
    ------
    ValueError
        If no data is found at *byte_offset* or the header line cannot be
        parsed as an integer.
    IOError
        If the file cannot be opened.
    """
    path = Path(source_file).resolve()
    with open(path, "rb") as fh:
        fh.seek(int(byte_offset))
        lines = []
        for _ in range(atom_count + 2):
            line = fh.readline()
            if not line:
                break
            lines.append(line)

    if not lines:
        raise ValueError(
            f"No data at byte_offset={byte_offset} in {path}"
        )

    return b"".join(lines).decode("utf-8", errors="replace")


def _parse_xyz_text(xyz_text: str) -> _ParsedXYZ:
    """Parse raw XYZ text into symbols and coordinates."""
    lines = xyz_text.splitlines()
    if len(lines) < 2:
        raise ValueError("Malformed XYZ payload: expected atom-count and comment lines.")

    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("Malformed XYZ payload: invalid atom count line.") from exc

    atom_lines = lines[2: 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"Malformed XYZ payload: expected {atom_count} atom lines, found {len(atom_lines)}."
        )

    symbols: list[str] = []
    coords = np.empty((atom_count, 3), dtype=np.float64)
    for i, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ payload: atom line {i + 1} has fewer than 4 fields.")
        symbols.append(parts[0])
        try:
            coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError as exc:
            raise ValueError(
                f"Malformed XYZ payload: atom line {i + 1} contains invalid coordinates."
            ) from exc

    return _ParsedXYZ(
        atom_count=atom_count,
        comment=lines[1],
        symbols=tuple(symbols),
        coords=coords,
    )


def _format_xyz_text(parsed: _ParsedXYZ, coords: np.ndarray) -> str:
    """Serialise XYZ data using the original symbols and comment line."""
    lines = [str(parsed.atom_count), parsed.comment]
    for symbol, (x, y, z) in zip(parsed.symbols, coords):
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


def align_xyz_to_reference(
    xyz_text: str,
    reference_xyz_text: str,
    atom_selection: str = "heavy",
) -> str:
    """Rigidly align one XYZ payload to a reference XYZ payload."""
    reference = _parse_xyz_text(reference_xyz_text)
    target = _parse_xyz_text(xyz_text)

    if reference.atom_count != target.atom_count:
        raise ValueError("Cannot align XYZ payloads with different atom counts.")

    reference_selection = _resolve_alignment_indices(reference.symbols, atom_selection)
    target_selection = _resolve_alignment_indices(target.symbols, atom_selection)
    if reference_selection.shape != target_selection.shape:
        raise ValueError(
            "Cannot align XYZ payloads with different atom counts in the selected atom set."
        )

    reference_symbols = [reference.symbols[i].strip().upper() for i in reference_selection]
    target_symbols = [target.symbols[i].strip().upper() for i in target_selection]
    if reference_symbols != target_symbols:
        descriptor = "heavy-atom" if atom_selection == "heavy" else "selected-atom"
        raise ValueError(
            f"Cannot align XYZ payloads with different {descriptor} element order."
        )

    rotation = _kabsch_rotation(
        reference.coords[reference_selection],
        target.coords[target_selection],
    )
    ref_centroid = reference.coords[reference_selection].mean(axis=0)
    tgt_centroid = target.coords[target_selection].mean(axis=0)
    transformed = (target.coords - tgt_centroid) @ rotation + ref_centroid

    return _format_xyz_text(target, transformed)


def _resolve_alignment_reference_text(
    df: pd.DataFrame,
    reference: str,
) -> str:
    """Return the deterministic XYZ payload used as the alignment reference."""
    if reference != "earliest_frame":
        raise ValueError(
            f"Unsupported alignment reference '{reference}'. "
            "Valid choices: ['earliest_frame']"
        )

    required = ["source_file", "byte_offset", "atom_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    mask = pd.Series(True, index=df.index)
    for column in required:
        mask = mask & df[column].notna()

    eligible = df.loc[mask]
    if eligible.empty:
        raise ValueError("No eligible structures available for alignment reference.")

    row = eligible.iloc[0]
    return read_xyz_frame_text(
        row["source_file"],
        int(row["byte_offset"]),
        int(row["atom_count"]),
    )


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
    """Read the XYZ structure of each occupied bin's representative frame.

    The representative is :attr:`ConformationalMap.representatives`, the same
    frame :func:`build_bin_frame_metadata` describes. A representative without
    a readable structure (no ``source_file`` / ``byte_offset`` / ``atom_count``
    value, or a file that cannot be read) leaves its bin without a structure.

    Parameters
    ----------
    conf_map:
        Map of the coordinate pair; its table must have ``source_file``,
        ``byte_offset`` and ``atom_count`` columns.
    alignment:
        Optional alignment config dict.  Alignment is on by default: payloads
        are rigidly aligned to the reference structure using the configured
        reference / atom-selection settings.  ``{"enabled": false}`` keeps
        each structure in its raw trajectory orientation.

    Returns
    -------
    dict[str, str]
        Mapping of ``"xi_yi"`` bin keys to XYZ text strings.
    """
    df = conf_map.table
    required = ["source_file", "byte_offset", "atom_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    bin_best = conf_map.representatives
    if not bin_best:
        return {}

    alignment_cfg = alignment or {}
    align_enabled = bool(alignment_cfg.get("enabled", True))
    alignment_reference = str(alignment_cfg.get("reference", "earliest_frame"))
    atom_selection = str(alignment_cfg.get("atom_selection", "heavy"))

    source_files = df["source_file"].to_numpy(dtype=object)
    byte_offsets = pd.to_numeric(df["byte_offset"], errors="coerce").to_numpy(dtype=float)
    atom_counts = pd.to_numeric(df["atom_count"], errors="coerce").to_numpy(dtype=float)

    payloads: dict[str, str] = {}
    for (xi, yi), i in bin_best.items():
        sf = source_files[i]
        bo = byte_offsets[i]
        ac = atom_counts[i]
        if pd.isna(sf) or pd.isna(bo) or pd.isna(ac):
            continue
        try:
            text = read_xyz_frame_text(sf, int(bo), int(ac))
        except (ValueError, IOError, OSError):
            continue
        payloads[f"{xi}_{yi}"] = text

    # The reference is read only when there is something to align; if it
    # cannot be read while bin structures could, that is a real error.
    if align_enabled and payloads:
        reference_xyz_text = _resolve_alignment_reference_text(
            df,
            reference=alignment_reference,
        )
        payloads = {
            key: align_xyz_to_reference(text, reference_xyz_text, atom_selection=atom_selection)
            for key, text in payloads.items()
        }

    return payloads


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
