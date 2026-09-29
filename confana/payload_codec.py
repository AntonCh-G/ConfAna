"""Compact encoding of the interactive page's embedded structures and metadata.

The interactive HTML carries one representative structure and one metadata
record per occupied density bin.  As plain JSON those two blocks are about
98 % of the file.  This module stores them far more compactly:

- **Structures** (:func:`encode_structures`): the element sequence is stored
  once per page instead of once per structure, coordinates become 16-bit
  integers in fixed steps of ``coordinate_step`` ångström (default 0.001 Å,
  so at most 0.0005 Å of rounding), and the resulting array is gzipped and
  base64-encoded.  Before gzip the integers are laid out so that gzip finds
  more repeats (see :func:`_coordinate_planes`), which is lossless.  This is
  a display copy only — exact coordinates stay in the trajectory files the
  metadata points to.
- **Metadata** (:func:`encode_columns`): records are stored column by column
  (one array per field) instead of one object per record, repeated strings
  are replaced by indices into a per-column lookup list, and the whole block
  is gzipped and base64-encoded.
- **Count grid** (:func:`encode_count_grid`): the density histogram as raw
  little-endian unsigned integers (2 bytes each when every count fits,
  else 4), base64-encoded without gzip so the page can read it at once,
  before its compressed blocks are unpacked.  The page derives the other
  colour-scale grids (log counts, free energy) from it.

The encoders are deterministic: the same input always yields the same block,
so ``render_density_page`` stays byte-stable.  The matching decoders exist so
tests can round-trip the blocks in Python; the browser decodes the same
formats in ``viewer.js`` with the built-in ``DecompressionStream('gzip')``.

Public API
----------
- ``encode_structures`` / ``decode_structures``
- ``encode_columns`` / ``decode_columns``
- ``encode_count_grid`` / ``decode_count_grid``
"""

from __future__ import annotations

import base64
import gzip
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np

# Block format markers, also checked by the JavaScript decoder.
STRUCTURES_FORMAT = "confana-structures-v2"
COLUMNS_FORMAT = "confana-columns-v1"
COUNT_GRID_FORMAT = "confana-count-grid-v1"

# 16-bit signed integers hold ±32767 coordinate steps.
_INT16_LIMIT = 32767

# gzip level 6 is the usual speed/size compromise; the level does not change
# what the decoders read.
_GZIP_LEVEL = 6

# A string column is stored as a lookup list plus indices when it has at most
# this share of distinct values; below that, the lookup costs more than it saves.
_LOOKUP_MAX_DISTINCT_RATIO = 0.5


# ---------------------------------------------------------------------------
# gzip + base64 helpers
# ---------------------------------------------------------------------------


def _pack(payload: bytes) -> str:
    """gzip *payload* deterministically (no timestamp) and base64-encode it."""
    return base64.b64encode(gzip.compress(payload, compresslevel=_GZIP_LEVEL, mtime=0)).decode(
        "ascii"
    )


def _unpack(blob: str) -> bytes:
    """Inverse of :func:`_pack`."""
    return gzip.decompress(base64.b64decode(blob))


def _pack_json(obj: object) -> str:
    """gzip + base64 the compact JSON form of *obj*."""
    return _pack(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _unpack_json(blob: str) -> dict:
    """Inverse of :func:`_pack_json`."""
    return json.loads(_unpack(blob).decode("utf-8"))


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def coordinate_decimals(step: float) -> int:
    """Decimals a coordinate needs to show one *step* (mirrors viewer.js)."""
    return max(0, min(9, int(math.ceil(-math.log10(step)))))


def _parse_xyz(text: str, key: str) -> tuple[int, str, tuple[str, ...], np.ndarray]:
    """Split one XYZ payload into ``(atom_count, comment, symbols, coords)``.

    Only the first four fields of each atom record are read, so extended-XYZ
    files with extra per-atom columns parse too.

    Raises
    ------
    ValueError
        If the text is not a complete XYZ frame; *key* names the structure.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError(f"Structure {key!r}: XYZ payload has no atom-count and comment line.")
    try:
        atom_count = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Structure {key!r}: first line is not an atom count: {lines[0]!r}") from exc
    records = lines[2 : 2 + atom_count]
    if len(records) != atom_count:
        raise ValueError(
            f"Structure {key!r}: XYZ payload declares {atom_count} atoms but has {len(records)} "
            "atom records."
        )
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line_number, record in enumerate(records, start=3):
        fields = record.split()
        if len(fields) < 4:
            raise ValueError(
                f"Structure {key!r}: atom record on line {line_number} is not "
                f"'symbol x y z': {record!r}"
            )
        symbols.append(fields[0])
        try:
            coords.append([float(v) for v in fields[1:4]])
        except ValueError as exc:
            raise ValueError(
                f"Structure {key!r}: atom record on line {line_number} has "
                f"non-numeric coordinates: {record!r}"
            ) from exc
    return atom_count, lines[1], tuple(symbols), np.asarray(coords, dtype=float)


def _coordinate_planes(codes: np.ndarray) -> bytes:
    """Lay out ``(structure, atom, axis)`` int16 *codes* for gzip.

    Two lossless reorderings, which shrink the gzipped block by about 30 %:

    1. atom-major order ``(atom, axis, structure)``: the same coordinate of
       every structure sits side by side, and aligned structures have
       similar values there;
    2. byte planes: all low bytes first, then all high bytes, so the slowly
       varying high bytes form long repeats.
    """
    atom_major = np.ascontiguousarray(codes.transpose(1, 2, 0)).astype("<i2")
    return np.ascontiguousarray(atom_major.reshape(-1).view(np.uint8).reshape(-1, 2).T).tobytes()


def _coordinates_from_planes(data: bytes, count: int, atom_count: int) -> np.ndarray:
    """Inverse of :func:`_coordinate_planes`: ``(structure, atom, axis)`` int16 codes."""
    planes = np.frombuffer(data, dtype=np.uint8).reshape(2, -1)
    atom_major = np.ascontiguousarray(planes.T).view("<i2").reshape(atom_count, 3, count)
    return atom_major.transpose(2, 0, 1)


def encode_structures(payloads: Mapping[str, str], coordinate_step: float = 0.001) -> dict:
    """Return the compact block for the per-bin XYZ *payloads*.

    Parameters
    ----------
    payloads:
        Mapping of bin key (``"xi_yi"``) to XYZ text.  Insertion order is the
        order of the encoded arrays and is preserved by the decoder.
    coordinate_step:
        Ångström per integer step.  Coordinates are rounded to this step, so
        the largest error is half a step (0.0005 Å by default).

    Returns
    -------
    dict
        JSON-embeddable block: ``format``, ``step``, ``atom_count``,
        ``count``, ``elements`` (the shared element sequence), ``index``
        (gzipped keys and comment lines) and ``coords`` (gzipped 16-bit
        little-endian coordinates in the layout of :func:`_coordinate_planes`).

    Raises
    ------
    ValueError
        If *coordinate_step* is not positive, a payload is not a complete XYZ
        frame, the structures do not all share one element sequence, or a
        coordinate needs more than 16 bits at this step.
    """
    if not isinstance(coordinate_step, (int, float)) or isinstance(coordinate_step, bool):
        raise ValueError(f"coordinate_step must be a positive number, got {coordinate_step!r}.")
    step = float(coordinate_step)
    if not step > 0 or not math.isfinite(step):
        raise ValueError(f"coordinate_step must be a positive number, got {coordinate_step!r}.")

    keys: list[str] = []
    comments: list[str] = []
    frames: list[np.ndarray] = []
    elements: tuple[str, ...] | None = None
    atom_count = 0
    for key, text in payloads.items():
        n_atoms, comment, symbols, coords = _parse_xyz(text, key)
        if elements is None:
            elements, atom_count = symbols, n_atoms
        elif symbols != elements:
            raise ValueError(
                f"Structure {key!r} has a different element sequence than the first structure "
                f"({list(symbols)} vs {list(elements)}). Compressed payloads store the elements "
                "once per page; set plots.interactive.compress_payloads: false to embed these "
                "structures as plain JSON."
            )
        keys.append(key)
        comments.append(comment)
        frames.append(coords)

    if frames:
        codes = np.rint(np.stack(frames) / step)
        if np.abs(codes).max() > _INT16_LIMIT:
            worst = float(np.abs(np.stack(frames)).max())
            raise ValueError(
                f"Coordinate {worst:.3f} Å needs more than {_INT16_LIMIT} steps of "
                f"{step} Å, which does not fit a 16-bit integer. Increase "
                "plots.interactive.coordinate_step (or set compress_payloads: false)."
            )
        coord_bytes = _coordinate_planes(codes.astype("<i2"))
    else:
        coord_bytes = b""

    return {
        "format": STRUCTURES_FORMAT,
        "step": step,
        "atom_count": atom_count,
        "count": len(keys),
        "elements": list(elements or ()),
        "index": _pack_json({"keys": keys, "comments": comments}),
        "coords": _pack(coord_bytes),
    }


def decode_structures(block: Mapping) -> dict[str, str]:
    """Rebuild ``{bin key: XYZ text}`` from a :func:`encode_structures` block.

    Coordinates come back rounded to the block's step, and the text is written
    in one canonical layout (the same one ``viewer.js`` builds), so it is not
    byte-identical to the encoded payload.

    Raises
    ------
    ValueError
        If *block* is not a structures block of a known format version.
    """
    if block.get("format") != STRUCTURES_FORMAT:
        raise ValueError(
            f"Not a {STRUCTURES_FORMAT} block: format={block.get('format')!r}."
        )
    index = _unpack_json(block["index"])
    keys: list[str] = index["keys"]
    comments: list[str] = index["comments"]
    elements: list[str] = list(block["elements"])
    atom_count = int(block["atom_count"])
    step = float(block["step"])
    decimals = coordinate_decimals(step)

    if keys:
        coords = _coordinates_from_planes(_unpack(block["coords"]), len(keys), atom_count) * step
    else:
        coords = np.empty((0, 0, 3))

    out: dict[str, str] = {}
    for i, key in enumerate(keys):
        lines = [str(atom_count), comments[i]]
        for symbol, (x, y, z) in zip(elements, coords[i]):
            lines.append(f"{symbol} {x:.{decimals}f} {y:.{decimals}f} {z:.{decimals}f}")
        out[key] = "\n".join(lines) + "\n"
    return out


# ---------------------------------------------------------------------------
# Metadata columns
# ---------------------------------------------------------------------------


def _column_block(values: list) -> dict:
    """Store one column either directly or as a lookup list plus indices.

    A column of repeated strings (``source_file``, ``trajectory_id``,
    ``bead_id``, …) is stored once per distinct value; every other column is
    stored as is.  Lookup order is first appearance, so the block is
    deterministic.
    """
    present = [v for v in values if v is not None]
    strings = bool(present) and all(isinstance(v, str) for v in present)
    if strings:
        lookup: list[str] = []
        seen: dict[str, int] = {}
        for value in present:
            if value not in seen:
                seen[value] = len(lookup)
                lookup.append(value)
        if len(lookup) <= _LOOKUP_MAX_DISTINCT_RATIO * len(values):
            return {
                "lookup": lookup,
                "codes": [None if v is None else seen[v] for v in values],
            }
    return {"values": values}


def encode_columns(records: Mapping[str, Mapping] | Sequence[Mapping]) -> dict:
    """Return the compact block for per-bin or per-frame metadata *records*.

    Parameters
    ----------
    records:
        Either a mapping of bin key to record (the per-bin metadata) or a
        sequence of records (the per-frame table).  Order is preserved, and a
        mapping decodes back to a mapping.

    Returns
    -------
    dict
        JSON-embeddable block with ``format``, ``count``, ``fields`` (union of
        the record fields, in first-appearance order) and ``data`` (the
        gzipped columns).

    Raises
    ------
    ValueError
        If *records* is neither a mapping nor a sequence of mappings.
    """
    if isinstance(records, Mapping):
        keys: list[str] | None = [str(k) for k in records]
        rows: list[Mapping] = list(records.values())
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        keys, rows = None, list(records)
    else:
        raise ValueError(
            f"encode_columns expects a mapping or a sequence of records, got {type(records).__name__}."
        )
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("encode_columns expects every record to be a mapping.")

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    columns = {field: _column_block([row.get(field) for row in rows]) for field in fields}
    return {
        "format": COLUMNS_FORMAT,
        "count": len(rows),
        "fields": fields,
        "data": _pack_json({"keys": keys, "columns": columns}),
    }


def decode_columns(block: Mapping) -> dict[str, dict] | list[dict]:
    """Rebuild the records of a :func:`encode_columns` block.

    Returns a mapping when the block was encoded from one, otherwise a list.
    Every record carries every field of the block: a field one record lacked
    comes back as ``None``, as do encoded ``None`` values.

    Raises
    ------
    ValueError
        If *block* is not a columns block of a known format version.
    """
    if block.get("format") != COLUMNS_FORMAT:
        raise ValueError(f"Not a {COLUMNS_FORMAT} block: format={block.get('format')!r}.")
    payload = _unpack_json(block["data"])
    columns = payload["columns"]
    count = int(block["count"])

    rows: list[dict] = [{} for _ in range(count)]
    for field in block["fields"]:
        column = columns[field]
        if "lookup" in column:
            lookup = column["lookup"]
            values = [None if code is None else lookup[code] for code in column["codes"]]
        else:
            values = column["values"]
        for row, value in zip(rows, values):
            row[field] = value

    keys = payload.get("keys")
    if keys is None:
        return rows
    return dict(zip(keys, rows))


# ---------------------------------------------------------------------------
# Count grid
# ---------------------------------------------------------------------------


def encode_count_grid(counts: np.ndarray) -> dict:
    """Return the compact block for a 2D histogram of frame *counts*.

    Parameters
    ----------
    counts:
        2D array of non-negative whole numbers, rows = y bins.  0 means an
        unsampled bin.

    Returns
    -------
    dict
        JSON-embeddable block: ``format``, ``shape`` (``[rows, columns]``),
        ``dtype`` (``"u2"`` or ``"u4"``) and ``data`` (base64 of the
        little-endian values in row order).

    Raises
    ------
    ValueError
        If *counts* is not 2D, holds negative or fractional values, or a
        count does not fit 32 bits.
    """
    grid = np.asarray(counts, dtype=float)
    if grid.ndim != 2:
        raise ValueError(f"Count grid must be 2D, got shape {grid.shape}.")
    if not np.all(np.isfinite(grid)) or np.any(grid < 0) or np.any(grid != np.rint(grid)):
        raise ValueError("Count grid must hold non-negative whole numbers.")
    top = float(grid.max()) if grid.size else 0.0
    if top > np.iinfo(np.uint32).max:
        raise ValueError(f"Count {top:.0f} does not fit a 32-bit unsigned integer.")
    dtype = "u2" if top <= np.iinfo(np.uint16).max else "u4"
    return {
        "format": COUNT_GRID_FORMAT,
        "shape": [int(grid.shape[0]), int(grid.shape[1])],
        "dtype": dtype,
        "data": base64.b64encode(grid.astype("<" + dtype).tobytes()).decode("ascii"),
    }


def decode_count_grid(block: Mapping) -> np.ndarray:
    """Rebuild the integer count array of an :func:`encode_count_grid` block.

    Raises
    ------
    ValueError
        If *block* is not a count-grid block of a known format version.
    """
    if block.get("format") != COUNT_GRID_FORMAT:
        raise ValueError(f"Not a {COUNT_GRID_FORMAT} block: format={block.get('format')!r}.")
    rows, cols = (int(n) for n in block["shape"])
    values = np.frombuffer(base64.b64decode(block["data"]), dtype="<" + block["dtype"])
    return values.reshape(rows, cols).astype(np.int64)
