"""Read the frames that coordinate-table rows name, whatever the trajectory format.

A coordinate-table row points at one frame of one trajectory file. This module
turns rows into :class:`~confana.models.FrameRecord` objects with float64
coordinates, so callers never open trajectory files or parse xyz text
themselves. Two readers sit behind it, chosen per row:

- **xyz** rows are read by seeking to their ``byte_offset``
  (:func:`confana.io_xyz.read_xyz_frame_by_offset`);
- **HDF5** rows carry ``byte_offset == HDF5_BYTE_OFFSET`` and are read by
  ``frame_number`` and ``bead_id`` (:class:`confana.io_hdf5.HDF5FrameReader`),
  each file opened once per call.

Public API
----------
- ``read_frames``
- ``names_frame``
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pandas as pd

from confana.io_hdf5 import HDF5_BYTE_OFFSET, HDF5FrameReader
from confana.io_xyz import read_xyz_frame_by_offset
from confana.models import FrameRecord


_REFERENCE_COLUMNS = ("source_file", "byte_offset")


def names_frame(table: pd.DataFrame) -> np.ndarray:
    """Return, per row of *table*, whether it names a frame to read.

    A row names a frame when it has both a ``source_file`` and a
    ``byte_offset`` value; coordinate-only input has neither.

    Raises
    ------
    ValueError
        If *table* has no ``source_file`` or ``byte_offset`` column.
    """
    missing = [c for c in _REFERENCE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"The coordinate table has no {missing} column(s), so it names no frames to read."
        )
    return (table["source_file"].notna() & table["byte_offset"].notna()).to_numpy(dtype=bool)


def read_frames(table: pd.DataFrame, rows: Sequence[int]) -> list[FrameRecord | None]:
    """Read the frames named by the rows at ``iloc`` positions *rows* of *table*.

    Returns one frame per requested row, in the order of *rows*, with float64
    coordinates and the row's ``frame_number``, ``trajectory_id`` and
    ``bead_id``; ``None`` for a row that names no frame (see
    :func:`names_frame`).

    Raises
    ------
    ValueError
        If a named frame cannot be read (missing file or dataset, malformed
        frame, frame or bead out of range) or its atom count differs from the
        row's ``atom_count``; the message names the frame, its file and the
        reason. Also if *table* lacks the ``source_file`` / ``byte_offset``
        columns.
    """
    named = names_frame(table)
    frames: list[FrameRecord | None] = []
    with ExitStack() as stack:
        hdf5_readers: dict[str, HDF5FrameReader] = {}
        for row in rows:
            if not named[row]:
                frames.append(None)
                continue
            record = table.iloc[row]
            where = _describe(record, row)
            source_file = str(record["source_file"])
            try:
                if _is_hdf5_row(record):
                    if source_file not in hdf5_readers:
                        hdf5_readers[source_file] = stack.enter_context(
                            HDF5FrameReader(source_file)
                        )
                    frame = hdf5_readers[source_file].read(
                        int(record["frame_number"]), _bead_id(record)
                    )
                else:
                    frame = _with_row_identity(
                        read_xyz_frame_by_offset(
                            source_file, int(record["byte_offset"]), dtype=np.float64
                        ),
                        record,
                    )
            except (OSError, TypeError, ValueError) as exc:
                raise ValueError(f"Cannot read the structure of {where}: {exc}") from exc
            _check_atom_count(frame, record, where)
            frames.append(frame)
    return frames


def _is_hdf5_row(record: pd.Series) -> bool:
    """Whether the row names an HDF5 frame (read by frame and bead, not by offset)."""
    return int(record["byte_offset"]) == HDF5_BYTE_OFFSET


def _bead_id(record: pd.Series) -> str | None:
    """The row's ``bead_id``, or ``None`` for a centroid / non-PIMD row."""
    value = record.get("bead_id")
    return None if value is None or pd.isna(value) else str(value)


def _with_row_identity(frame: FrameRecord, record: pd.Series) -> FrameRecord:
    """Give an xyz frame its row's frame number, trajectory and bead.

    The file alone knows only the byte offset; the table knows the rest.
    """
    frame_number = record.get("frame_number")
    trajectory_id = record.get("trajectory_id")
    return replace(
        frame,
        frame_number=frame.frame_number if frame_number is None or pd.isna(frame_number)
        else int(frame_number),
        trajectory_id=None if trajectory_id is None or pd.isna(trajectory_id)
        else str(trajectory_id),
        bead_id=_bead_id(record),
    )


def _describe(record: pd.Series, row: int) -> str:
    """Name a row's frame for an error message: its id, file and position in it."""
    name = f"frame_id {record['frame_id']}" if "frame_id" in record.index else f"row {row}"
    if _is_hdf5_row(record):
        bead = _bead_id(record)
        position = f"frame {record.get('frame_number')}" + (f", {bead}" if bead else "")
    else:
        position = f"byte offset {record['byte_offset']}"
    return f"{name} ({record['source_file']}, {position})"


def _check_atom_count(frame: FrameRecord, record: pd.Series, where: str) -> None:
    """Raise if the frame read does not have the atom count the table records."""
    expected = record.get("atom_count")
    if expected is not None and not pd.isna(expected) and int(expected) != frame.atom_count:
        raise ValueError(
            f"Cannot read the structure of {where}: the frame has {frame.atom_count} atoms "
            f"but the coordinate table says {int(expected)}."
        )
