"""Tests for confana/frame_source.py — reading the frames coordinate-table rows name."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confana.frame_source import names_frame, read_frames
from tests.hdf5_runs import hdf5_table, write_hdf5_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME_A = "3\nframe-a\nC 2.2393 -0.3791 0.263\nO 0.8424 1.9231 -0.4249\nH 0.0 0.0 1.0\n"
_FRAME_B = "3\nframe-b\nC 1.0 2.0 3.0\nO 4.0 5.0 6.0\nH 7.0 8.0 9.0\n"


def _write_xyz(path: Path, frames: list[str]) -> list[int]:
    """Write the xyz frame texts to *path*; return each frame's byte offset."""
    data = b""
    offsets = []
    for text in frames:
        offsets.append(len(data))
        data += text.encode("utf-8")
    path.write_bytes(data)
    return offsets


def _row(table: pd.DataFrame, frame_number: int, bead_id: str | None = None) -> int:
    bead = table["bead_id"].isna() if bead_id is None else table["bead_id"] == bead_id
    match = (table["frame_number"] == frame_number) & bead
    return int(np.flatnonzero(match.to_numpy(dtype=bool, na_value=False))[0])


def _xyz_table(path: Path, offsets: list, atom_count: int = 3) -> pd.DataFrame:
    n = len(offsets)
    return pd.DataFrame({
        "frame_id": range(n),
        "source_file": [str(path)] * n,
        "byte_offset": pd.array(offsets, dtype="Int64"),
        "atom_count": [atom_count] * n,
    })


# ---------------------------------------------------------------------------
# xyz rows
# ---------------------------------------------------------------------------


def test_xyz_rows_are_read_exactly_as_written_in_the_requested_order(tmp_path):
    path = tmp_path / "traj.xyz"
    table = _xyz_table(path, _write_xyz(path, [_FRAME_A, _FRAME_B]))

    frame_b, frame_a = read_frames(table, [1, 0])

    assert frame_a.elements == ["C", "O", "H"]
    assert (frame_a.comment_line, frame_b.comment_line) == ("frame-a", "frame-b")
    assert frame_a.coords.dtype == np.float64
    assert frame_a.coords[0, 0] == 2.2393  # float32 would give 2.2392999...
    np.testing.assert_array_equal(frame_b.coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


def test_xyz_frames_carry_their_rows_identity(tmp_path):
    # The file alone does not know which trajectory, bead or frame number it is.
    path = tmp_path / "traj.xyz"
    table = _xyz_table(path, _write_xyz(path, [_FRAME_A, _FRAME_B]))
    table["frame_number"] = [0, 1]
    table["trajectory_id"] = ["run"] * 2
    table["bead_id"] = ["03"] * 2

    (frame,) = read_frames(table, [1])

    assert (frame.frame_number, frame.trajectory_id, frame.bead_id) == (1, "run", "03")


def test_rows_that_name_no_structure_give_none(tmp_path):
    # Coordinate-only input has no file or no offset to read from.
    path = tmp_path / "traj.xyz"
    offset = _write_xyz(path, [_FRAME_A])[0]
    table = _xyz_table(path, [offset, None, offset])
    table.loc[2, "source_file"] = None

    np.testing.assert_array_equal(names_frame(table), [True, False, False])
    first, no_offset, no_file = read_frames(table, [0, 1, 2])
    assert first.comment_line == "frame-a"
    assert no_offset is None
    assert no_file is None


@pytest.mark.parametrize(
    ("content", "atom_count", "reason"),
    [
        (None, 3, "No such file"),  # moved or deleted since the table was built
        ("3\nbad\nC 1.0 2.0\nO 4.0 5.0 6.0\nH 7.0 8.0 9.0\n", 3, "fewer than 4 fields"),
        (_FRAME_A, 4, "3 atoms but the coordinate table says 4"),
    ],
    ids=["missing-file", "malformed-frame", "atom-count-mismatch"],
)
def test_unreadable_frames_stop_with_an_error_naming_the_frame(tmp_path, content, atom_count, reason):
    path = tmp_path / "traj.xyz"
    if content is not None:
        path.write_text(content)
    table = _xyz_table(path, [0], atom_count=atom_count)

    with pytest.raises(ValueError, match=rf"frame_id 0 .*traj\.xyz.*{reason}"):
        read_frames(table, [0])


def test_table_without_structure_columns_raises():
    with pytest.raises(ValueError, match="byte_offset"):
        read_frames(pd.DataFrame({"frame_id": [0], "source_file": ["a.xyz"]}), [0])


# ---------------------------------------------------------------------------
# HDF5 rows
# ---------------------------------------------------------------------------


def test_hdf5_bead_rows_read_their_bead_positions(tmp_path):
    h5_path, beads, _ = write_hdf5_run(tmp_path / "s0")
    table = hdf5_table(h5_path)

    (frame,) = read_frames(table, [_row(table, 3, "bead_01")])

    assert frame.elements == ["C", "O", "H"]  # from input.xyz
    np.testing.assert_array_equal(frame.coords, beads[3, 1])
    assert (frame.frame_number, frame.bead_id) == (3, "bead_01")


def test_hdf5_centroid_rows_read_the_centroid(tmp_path):
    h5_path, _, centroid = write_hdf5_run(tmp_path / "s0")
    table = hdf5_table(h5_path, positions_source="centroid")

    (frame,) = read_frames(table, [_row(table, 2)])

    np.testing.assert_array_equal(frame.coords, centroid[2])


def test_hdf5_frame_out_of_range_raises(tmp_path):
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0")
    table = hdf5_table(h5_path)
    table.loc[0, "frame_number"] = 99

    with pytest.raises(ValueError, match=r"frame_id 0 .*trajectory\.hdf5.*frame 99"):
        read_frames(table, [0])


def test_a_table_can_mix_xyz_and_hdf5_rows(tmp_path):
    h5_path, beads, _ = write_hdf5_run(tmp_path / "s0")
    xyz_path = tmp_path / "traj.xyz"
    xyz = _xyz_table(xyz_path, _write_xyz(xyz_path, [_FRAME_A]))
    table = pd.concat([hdf5_table(h5_path), xyz], ignore_index=True)

    hdf5_frame, xyz_frame = read_frames(table, [_row(table, 0, "bead_00"), len(table) - 1])

    np.testing.assert_array_equal(hdf5_frame.coords, beads[0, 0])
    assert xyz_frame.comment_line == "frame-a"


def test_hdf5_file_without_its_datasets_raises_naming_the_frame(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path, _, _ = write_hdf5_run(tmp_path / "s0")
    table = hdf5_table(h5_path)
    with h5py.File(h5_path, "a") as fh:
        del fh["bead_positions"]

    with pytest.raises(ValueError, match=r"frame_id 0 .*trajectory\.hdf5.*bead_positions"):
        read_frames(table, [0])
