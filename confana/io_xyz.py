"""Streaming xyz parser with cached byte-offset frame indexing.

Design principles
-----------------
* All file opens use **binary mode** (``rb``).  Python text-mode universal
  newline handling adjusts ``tell()`` on some platforms, making seek-based
  retrieval unreliable.  Binary mode is the only safe choice.
* The frame scanner (``scan_xyz_frame_offsets``) never stores coordinate data
  in memory; it only records byte offsets and lightweight metadata.
* Random-access retrieval (``read_xyz_frame``, ``read_xyz_frame_by_offset``)
  reads exactly one frame by seeking to the stored byte offset.
* The frame index is cached to disk as JSON alongside the source file (or in a
  configured cache directory) and is invalidated when file path, size, or
  modification time changes.

Public API
----------
- ``iter_xyz_frames``
- ``load_xyz_files``
- ``scan_xyz_frame_offsets``
- ``load_or_build_xyz_index``
- ``read_xyz_frame``
- ``read_xyz_frame_by_offset``
- ``extract_frame_metadata``
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from confana.models import FrameIndex, FrameIndexEntry, FrameRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comment-line parsing
# ---------------------------------------------------------------------------

_RE_STEP = re.compile(r"Step:\s+(\d+)")
_RE_BEAD = re.compile(r"Bead:\s+(\d+)")
_RE_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_comment_line(comment: str) -> dict:
    """Parse the CP2K-style xyz comment line into a metadata dict.

    Known format::

        # CELL(abcABC): ... Step: N  Bead: N positions{angstrom} cell{angstrom}

    Returns a dict with keys:
    - ``step_number`` (int | None)
    - ``bead_comment`` (int | None)
    - ``energy`` (float | None) — not present in current dataset; always None

    The function is intentionally lenient: unknown comment formats return all
    None values rather than raising.
    """
    step: Optional[int] = None
    bead: Optional[int] = None
    energy: Optional[float] = None

    m = _RE_STEP.search(comment)
    if m:
        step = int(m.group(1))

    m = _RE_BEAD.search(comment)
    if m:
        bead = int(m.group(1))

    # Energy is not present in the current dataset.
    # TODO-energy: if a future dataset encodes energy in the comment line,
    # implement format-specific parsing here rather than guessing.

    return {"step_number": step, "bead_comment": bead, "energy": energy}


# ---------------------------------------------------------------------------
# Low-level frame block parser
# ---------------------------------------------------------------------------


def _parse_xyz_block(
    lines: list[bytes],
    source_file: str,
    byte_offset: int,
    frame_number: int,
    dtype: npt.DTypeLike = np.float32,
) -> FrameRecord:
    """Parse ``atom_count + 2`` raw binary lines into a FrameRecord.

    ``lines[0]`` must be the atom-count line, ``lines[1]`` the comment line,
    and ``lines[2:]`` the atom records. Coordinates are stored as *dtype*.

    Raises
    ------
    ValueError
        On malformed atom count, wrong number of coordinate fields, or
        non-numeric coordinate values.
    """
    try:
        atom_count = int(lines[0].decode("utf-8").strip())
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Cannot parse atom count from line: {lines[0]!r} "
            f"(file={source_file}, byte_offset={byte_offset})"
        ) from exc

    comment_line = lines[1].decode("utf-8").rstrip("\n\r")

    if len(lines) != atom_count + 2:
        raise ValueError(
            f"Expected {atom_count + 2} lines for frame {frame_number} "
            f"but got {len(lines)} "
            f"(file={source_file}, byte_offset={byte_offset})"
        )

    elements: list[str] = []
    coords_list: list[list[float]] = []

    for i, raw in enumerate(lines[2:], start=1):
        try:
            parts = raw.decode("utf-8").split()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Non-UTF-8 atom line {i} in frame {frame_number} "
                f"(file={source_file})"
            ) from exc

        if len(parts) < 4:
            raise ValueError(
                f"Atom line {i} in frame {frame_number} has fewer than 4 fields: "
                f"{raw!r} (file={source_file})"
            )

        elements.append(parts[0])
        try:
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric coordinates on atom line {i} in frame {frame_number}: "
                f"{raw!r} (file={source_file})"
            ) from exc

    meta = _parse_comment_line(comment_line)

    return FrameRecord(
        source_file=source_file,
        frame_number=frame_number,
        byte_offset=byte_offset,
        atom_count=atom_count,
        comment_line=comment_line,
        elements=elements,
        coords=np.array(coords_list, dtype=dtype),
        energy=meta["energy"],
        step_number=meta["step_number"],
        bead_comment=meta["bead_comment"],
    )


# ---------------------------------------------------------------------------
# Frame scanner — builds index without loading coordinates
# ---------------------------------------------------------------------------


def scan_xyz_frame_offsets(path: str | Path) -> FrameIndex:
    """One-pass scan of an xyz file; returns a FrameIndex with no coordinates.

    Opens the file in binary mode and records ``byte_offset``, ``atom_count``,
    and ``comment_line`` for every frame without materialising coordinate data.

    Parameters
    ----------
    path:
        Path to the xyz file.

    Returns
    -------
    FrameIndex
        Contains one ``FrameIndexEntry`` per frame found in the file.

    Raises
    ------
    ValueError
        If any frame has a non-integer atom-count line or if the file ends
        prematurely within a frame.
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path).resolve()
    stat = os.stat(path)
    index = FrameIndex(
        source_file=str(path),
        file_size=stat.st_size,
        file_mtime=stat.st_mtime,
    )

    frame_number = 0
    with open(path, "rb") as fh:
        while True:
            byte_offset = fh.tell()
            header = fh.readline()
            if not header:
                break  # clean EOF between frames

            header_str = header.decode("utf-8").strip()
            if not header_str:
                # Skip blank lines between frames (lenient)
                continue

            try:
                atom_count = int(header_str)
            except ValueError as exc:
                raise ValueError(
                    f"Expected integer atom count at byte {byte_offset} "
                    f"(frame {frame_number}) in {path}: got {header_str!r}"
                ) from exc

            comment_raw = fh.readline()
            if not comment_raw:
                raise ValueError(
                    f"Premature EOF reading comment line of frame {frame_number} "
                    f"at byte {byte_offset} in {path}"
                )
            comment_line = comment_raw.decode("utf-8").rstrip("\n\r")

            # Skip atom_count coordinate lines without storing them
            for atom_idx in range(atom_count):
                line = fh.readline()
                if not line:
                    raise ValueError(
                        f"Premature EOF reading atom {atom_idx} of frame "
                        f"{frame_number} at byte {byte_offset} in {path}"
                    )

            index.entries.append(
                FrameIndexEntry(
                    frame_number=frame_number,
                    byte_offset=byte_offset,
                    atom_count=atom_count,
                    comment_line=comment_line,
                )
            )
            frame_number += 1

    logger.debug("Scanned %d frames from %s", len(index), path)
    return index


# ---------------------------------------------------------------------------
# Disk-cached index
# ---------------------------------------------------------------------------


def _cache_is_valid(npz: "np.lib.npyio.NpzFile", path: Path) -> bool:
    """Return True iff the NPZ cache header matches the current file's stat."""
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return False
    try:
        return (
            str(npz["_source_file"]) == str(path)
            and int(npz["_file_size"]) == stat.st_size
            and abs(float(npz["_file_mtime"]) - stat.st_mtime) < 1e-3
        )
    except KeyError:
        return False


def _write_index_cache(index: FrameIndex, cache_path: Path) -> None:
    """Serialise a FrameIndex to a binary NPZ cache file.

    Numeric fields (frame_number, byte_offset, atom_count) are stored as
    typed numpy arrays for fast binary I/O.  Comment lines are stored as a
    single newline-joined UTF-8 byte array to avoid pickle and fixed-length
    unicode padding overhead.
    """
    entries = index.entries
    n = len(entries)
    frame_number = np.empty(n, dtype=np.int64)
    byte_offset_arr = np.empty(n, dtype=np.int64)
    atom_count_arr = np.empty(n, dtype=np.int32)
    for i, e in enumerate(entries):
        frame_number[i] = e.frame_number
        byte_offset_arr[i] = e.byte_offset
        atom_count_arr[i] = e.atom_count
    comment_raw = "\n".join(e.comment_line for e in entries).encode("utf-8")
    np.savez(
        cache_path,
        _source_file=np.array(index.source_file),
        _file_size=np.array(index.file_size, dtype=np.int64),
        _file_mtime=np.array(index.file_mtime, dtype=np.float64),
        frame_number=frame_number,
        byte_offset=byte_offset_arr,
        atom_count=atom_count_arr,
        comment_raw=np.frombuffer(comment_raw, dtype=np.uint8),
    )
    logger.debug("Wrote frame index cache to %s", cache_path)


def _read_index_cache(npz: "np.lib.npyio.NpzFile") -> FrameIndex:
    """Deserialise a FrameIndex from an open NpzFile."""
    frame_numbers = npz["frame_number"]
    byte_offsets = npz["byte_offset"]
    atom_counts = npz["atom_count"]
    comment_lines = npz["comment_raw"].tobytes().decode("utf-8").split("\n")
    n = len(frame_numbers)
    entries = [
        FrameIndexEntry(
            frame_number=int(frame_numbers[i]),
            byte_offset=int(byte_offsets[i]),
            atom_count=int(atom_counts[i]),
            comment_line=comment_lines[i],
        )
        for i in range(n)
    ]
    return FrameIndex(
        source_file=str(npz["_source_file"]),
        file_size=int(npz["_file_size"]),
        file_mtime=float(npz["_file_mtime"]),
        entries=entries,
    )


def load_or_build_xyz_index(
    path: str | Path,
    cache_path: Optional[str | Path] = None,
) -> FrameIndex:
    """Load a cached FrameIndex or build one by scanning the file.

    The cache is stored as a binary NPZ file.  If ``cache_path`` is None, the
    cache is placed alongside the source file as ``{source}.frameindex.npz``.

    The cache is considered stale and discarded when the source file's absolute
    path, size, or modification time has changed.  On a ``PermissionError``
    when writing the cache, a warning is logged and the in-memory index is
    returned without caching.

    Parameters
    ----------
    path:
        Path to the xyz file.
    cache_path:
        Optional explicit path for the NPZ cache file.

    Returns
    -------
    FrameIndex
    """
    path = Path(path).resolve()

    if cache_path is None:
        cache_path = path.parent / (path.name + ".frameindex.npz")
    else:
        cache_path = Path(cache_path)

    # Try loading from cache
    if cache_path.exists():
        try:
            npz = np.load(cache_path)
            if _cache_is_valid(npz, path):
                logger.debug("Using cached frame index from %s", cache_path)
                return _read_index_cache(npz)
            else:
                logger.debug("Cache stale for %s; rescanning", path)
        except (KeyError, OSError, Exception):
            logger.debug("Cache unreadable for %s; rescanning", path)

    # Build fresh index
    index = scan_xyz_frame_offsets(path)

    # Persist cache
    try:
        _write_index_cache(index, cache_path)
    except PermissionError:
        warnings.warn(
            f"Cannot write frame-index cache to {cache_path} (PermissionError). "
            "Analysis will continue without caching.",
            RuntimeWarning,
            stacklevel=2,
        )

    return index


# ---------------------------------------------------------------------------
# Sequential frame iteration
# ---------------------------------------------------------------------------


def iter_xyz_frames(
    path: str | Path,
    trajectory_id_pattern: Optional[str] = None,
    bead_id_pattern: Optional[str] = None,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> Iterator[FrameRecord]:
    """Yield frames from one xyz file sequentially using a streaming parser.

    This is the public sequential-access counterpart to the random-access
    index API. Frames are parsed directly from disk without loading the full
    file into memory.

    Parameters
    ----------
    path:
        Path to one multi-frame xyz file.
    trajectory_id_pattern:
        Optional regex applied to the filename stem; group 1 becomes the
        per-frame ``trajectory_id``.
    bead_id_pattern:
        Optional regex applied to the filename stem; group 1 becomes the
        per-frame ``bead_id``.
    max_frames:
        Stop after yielding frames up to (but not including) this frame
        number.  ``None`` means no upper limit.
    start_frame:
        First frame number to yield (0-based, inclusive).  Frames before
        this index are read and discarded.  Default 0 (no skipping).

    Yields
    ------
    FrameRecord
        Sequentially parsed frames with ``local_frame_index`` populated and
        filename-derived ``trajectory_id`` / ``bead_id`` attached.
    """
    path = Path(path).resolve()
    traj_id = _derive_trajectory_id(path, trajectory_id_pattern)
    bead_id = _derive_bead_id(path, bead_id_pattern)

    frame_number = 0
    with open(path, "rb") as fh:
        while True:
            byte_offset = fh.tell()
            header = fh.readline()
            if not header:
                break

            header_str = header.decode("utf-8").strip()
            if not header_str:
                continue

            try:
                atom_count = int(header_str)
            except ValueError as exc:
                raise ValueError(
                    f"Expected integer atom count at byte {byte_offset} "
                    f"(frame {frame_number}) in {path}: got {header_str!r}"
                ) from exc

            lines = [header]
            comment = fh.readline()
            if not comment:
                raise ValueError(
                    f"Premature EOF reading comment line of frame {frame_number} "
                    f"at byte {byte_offset} in {path}"
                )
            lines.append(comment)

            for atom_idx in range(atom_count):
                atom_line = fh.readline()
                if not atom_line:
                    raise ValueError(
                        f"Premature EOF reading atom {atom_idx} of frame "
                        f"{frame_number} at byte {byte_offset} in {path}"
                    )
                lines.append(atom_line)

            record = _parse_xyz_block(lines, str(path), byte_offset, frame_number)
            record.trajectory_id = traj_id
            record.bead_id = bead_id
            record.local_frame_index = frame_number
            frame_number += 1
            if frame_number - 1 < start_frame:
                if max_frames is not None and frame_number >= max_frames:
                    return
                continue
            yield record
            if max_frames is not None and frame_number >= max_frames:
                return


# ---------------------------------------------------------------------------
# Random-access frame retrieval
# ---------------------------------------------------------------------------


def read_xyz_frame_by_offset(
    path: str | Path,
    byte_offset: int,
    *,
    dtype: npt.DTypeLike = np.float32,
) -> FrameRecord:
    """Open the file, seek to ``byte_offset``, and parse one frame.

    Parameters
    ----------
    path:
        Path to the xyz file.
    byte_offset:
        Byte position of the atom-count line for the desired frame.
    dtype:
        Coordinate dtype. float32 matches the streamed frames; float64 keeps
        every digit the file holds.

    Returns
    -------
    FrameRecord
        With coordinates populated.

    Raises
    ------
    ValueError
        On malformed frame data.
    """
    path = Path(path).resolve()
    with open(path, "rb") as fh:
        fh.seek(byte_offset)
        header = fh.readline()
        if not header:
            raise ValueError(
                f"No data at byte_offset={byte_offset} in {path}"
            )
        try:
            atom_count = int(header.decode("utf-8").strip())
        except ValueError as exc:
            raise ValueError(
                f"Cannot parse atom count at byte_offset={byte_offset} "
                f"in {path}: {header!r}"
            ) from exc

        lines = [header, fh.readline()]
        for _ in range(atom_count):
            lines.append(fh.readline())

    # Determine frame_number from byte_offset — unknown here, use -1 as sentinel
    return _parse_xyz_block(lines, str(path), byte_offset, frame_number=-1, dtype=dtype)


def read_xyz_frame(
    path: str | Path,
    frame_number: int,
    frame_index: FrameIndex,
) -> FrameRecord:
    """Retrieve a single frame by its 0-based frame number.

    Uses the ``FrameIndex`` to look up the byte offset and then reads exactly
    ``atom_count + 2`` lines from that position.

    Parameters
    ----------
    path:
        Path to the xyz file.
    frame_number:
        0-based frame index within the file.
    frame_index:
        Pre-built ``FrameIndex`` for the file.

    Returns
    -------
    FrameRecord
        With coordinates populated and ``frame_number`` set correctly.

    Raises
    ------
    IndexError
        If ``frame_number`` is outside [0, len(frame_index)).
    ValueError
        On malformed frame data.
    """
    entry = frame_index[frame_number]  # raises IndexError if out of range
    path = Path(path).resolve()

    with open(path, "rb") as fh:
        fh.seek(entry.byte_offset)
        lines = [fh.readline()]          # atom count line
        lines.append(fh.readline())      # comment line
        for _ in range(entry.atom_count):
            lines.append(fh.readline())

    record = _parse_xyz_block(lines, str(path), entry.byte_offset, frame_number)
    record.local_frame_index = frame_number
    return record


def extract_frame_metadata(
    path: str | Path,
    frame_number: int,
    frame_index: FrameIndex,
) -> dict:
    """Return metadata for one frame without loading coordinates.

    Parameters
    ----------
    path:
        Path to the xyz file.
    frame_number:
        0-based frame number.
    frame_index:
        Pre-built ``FrameIndex`` for the file.

    Returns
    -------
    dict
        Keys: ``source_file``, ``frame_number``, ``byte_offset``,
        ``atom_count``, ``comment_line``, ``step_number``, ``bead_comment``,
        ``energy``.
    """
    entry = frame_index[frame_number]
    meta = _parse_comment_line(entry.comment_line)
    return {
        "source_file": str(Path(path).resolve()),
        "frame_number": frame_number,
        "byte_offset": entry.byte_offset,
        "atom_count": entry.atom_count,
        "comment_line": entry.comment_line,
        "step_number": meta["step_number"],
        "bead_comment": meta["bead_comment"],
        "energy": meta["energy"],
    }


# ---------------------------------------------------------------------------
# Filename-based ID derivation
# ---------------------------------------------------------------------------


def _derive_trajectory_id(path: Path, pattern: Optional[str]) -> str:
    """Derive trajectory_id from a filename stem using a regex pattern.

    The regex is applied to the filename stem (name without extension).
    Group 1 of the match is captured as the trajectory_id.

    If ``pattern`` is None or produces no match, falls back to the parent
    directory name (if the file is not at the root of a drive) or the stem.

    # TODO-trajectory_id: confirm regex convention with user if filenames
    # differ from the my_run.pos_NN pattern.
    """
    stem = path.stem  # e.g. "my_run.pos_00"
    if pattern:
        m = re.search(pattern, stem)
        if m and m.lastindex and m.lastindex >= 1:
            return m.group(1)
        logger.warning(
            "trajectory_id_pattern %r did not match stem %r; "
            "falling back to parent directory name.",
            pattern,
            stem,
        )
    # Fallback: parent directory name if meaningful, else stem
    parent = path.parent.name
    return parent if parent else stem


def _derive_bead_id(path: Path, pattern: Optional[str]) -> Optional[str]:
    """Derive bead_id from a filename stem using a regex pattern.

    Group 1 of the match is captured as the bead_id string.
    Returns None if ``pattern`` is None or produces no match.

    # TODO-bead_id: confirm regex convention with user if filenames differ
    # from the my_run.pos_NN pattern.
    """
    if not pattern:
        return None
    stem = path.stem  # e.g. "my_run.pos_00"
    m = re.search(pattern, stem)
    if m and m.lastindex and m.lastindex >= 1:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Multi-file loading
# ---------------------------------------------------------------------------


def load_xyz_files(
    path_pattern: str,
    trajectory_id_pattern: Optional[str] = None,
    bead_id_pattern: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
) -> list[FrameRecord]:
    """Load all frames from all xyz files matching ``path_pattern``.

    Files are processed in sorted order.  For each file, a byte-offset index
    is built (or loaded from cache) and then every frame is read sequentially.

    Trajectory IDs and bead IDs are derived from filenames using the supplied
    regex patterns (or the defaults documented in ``_derive_trajectory_id`` and
    ``_derive_bead_id``).

    ``local_frame_index`` is the 0-based frame position within each file.
    ``global_frame_index`` is the 0-based position across all files combined
    (in sorted file order).

    Parameters
    ----------
    path_pattern:
        Glob pattern, e.g. ``"./data/*.xyz"``.
    trajectory_id_pattern:
        Regex applied to filename stem; group 1 → trajectory_id.
    bead_id_pattern:
        Regex applied to filename stem; group 1 → bead_id.
    cache_dir:
        Directory for frame-index JSON caches.  None = alongside source files.

    Returns
    -------
    list[FrameRecord]
        All frames from all matched files, in sorted-file then frame order.
    """
    import glob

    paths = sorted(Path(p).resolve() for p in glob.glob(path_pattern, recursive=True))
    if not paths:
        warnings.warn(
            f"No files matched path_pattern={path_pattern!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    all_records: list[FrameRecord] = []
    global_idx = 0

    for file_path in paths:
        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / (file_path.name + ".frameindex.npz")

        traj_id = _derive_trajectory_id(file_path, trajectory_id_pattern)
        bead_id = _derive_bead_id(file_path, bead_id_pattern)

        frame_index = load_or_build_xyz_index(file_path, cache_path=cache_path)

        for local_idx, entry in enumerate(frame_index.entries):
            record = read_xyz_frame(file_path, entry.frame_number, frame_index)
            record.trajectory_id = traj_id
            record.bead_id = bead_id
            record.local_frame_index = local_idx
            record.global_frame_index = global_idx
            all_records.append(record)
            global_idx += 1

        logger.info(
            "Loaded %d frames from %s (traj=%s, bead=%s)",
            len(frame_index),
            file_path.name,
            traj_id,
            bead_id,
        )

    return all_records
