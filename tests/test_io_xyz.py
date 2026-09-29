"""Tests for confana/io_xyz.py — streaming parser and byte-offset indexing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from confana.io_xyz import (
    _derive_bead_id,
    _derive_trajectory_id,
    _parse_comment_line,
    extract_frame_metadata,
    iter_xyz_frames,
    load_or_build_xyz_index,
    read_xyz_frame,
    read_xyz_frame_by_offset,
    scan_xyz_frame_offsets,
)
from confana.models import FrameIndex, FrameIndexEntry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Three-frame xyz with CP2K-style comment lines (21 atoms each, matching real data)
_ATOM_COUNT = 4  # keep small for speed; real data uses 21

def _make_xyz_bytes(n_frames: int = 3, atom_count: int = _ATOM_COUNT) -> bytes:
    """Create a valid multi-frame xyz file as bytes."""
    lines = []
    for f in range(n_frames):
        lines.append(f"{atom_count}\n".encode())
        lines.append(
            f"# CELL(abcABC):  10.0   10.0   10.0  90.0  90.0  90.0  "
            f"Step:     {f * 10}  Bead:     {0}  positions{{angstrom}}  cell{{angstrom}}\n"
            .encode()
        )
        for i in range(atom_count):
            x = float(f * 10 + i)
            lines.append(f"  C  {x:.5e}  {0.0:.5e}  {0.0:.5e}\n".encode())
    return b"".join(lines)


@pytest.fixture()
def xyz_file(tmp_path: Path) -> Path:
    """Write a 3-frame xyz file and return its path."""
    p = tmp_path / "test.xyz"
    p.write_bytes(_make_xyz_bytes(n_frames=3))
    return p


@pytest.fixture()
def xyz_index(xyz_file: Path) -> FrameIndex:
    return scan_xyz_frame_offsets(xyz_file)


# ---------------------------------------------------------------------------
# scan_xyz_frame_offsets
# ---------------------------------------------------------------------------


def test_scan_frame_count(xyz_file: Path):
    idx = scan_xyz_frame_offsets(xyz_file)
    assert len(idx) == 3


def test_scan_frame_numbers(xyz_file: Path):
    idx = scan_xyz_frame_offsets(xyz_file)
    for i, entry in enumerate(idx.entries):
        assert entry.frame_number == i


def test_scan_byte_offsets_are_nonnegative(xyz_index: FrameIndex):
    for entry in xyz_index.entries:
        assert entry.byte_offset >= 0


def test_scan_byte_offsets_strictly_increasing(xyz_index: FrameIndex):
    offsets = [e.byte_offset for e in xyz_index.entries]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets), "Byte offsets must be unique"


def test_scan_atom_count(xyz_index: FrameIndex):
    for entry in xyz_index.entries:
        assert entry.atom_count == _ATOM_COUNT


def test_scan_comment_line_step(xyz_index: FrameIndex):
    for i, entry in enumerate(xyz_index.entries):
        assert f"Step:     {i * 10}" in entry.comment_line


def test_scan_first_byte_offset_is_zero(xyz_file: Path):
    idx = scan_xyz_frame_offsets(xyz_file)
    assert idx.entries[0].byte_offset == 0


def test_scan_malformed_atom_count(tmp_path: Path):
    bad = tmp_path / "bad.xyz"
    bad.write_bytes(b"notanumber\n# comment\nC 0 0 0\n")
    with pytest.raises(ValueError, match="integer atom count"):
        scan_xyz_frame_offsets(bad)


def test_scan_premature_eof_in_atoms(tmp_path: Path):
    # Frame starts normally but ends before all atom lines
    bad = tmp_path / "truncated.xyz"
    bad.write_bytes(b"4\n# comment\nC 0 0 0\n")  # only 1 of 4 atom lines
    with pytest.raises(ValueError, match="Premature EOF"):
        scan_xyz_frame_offsets(bad)


def test_scan_file_metadata(xyz_file: Path):
    idx = scan_xyz_frame_offsets(xyz_file)
    import os
    stat = os.stat(xyz_file)
    assert idx.source_file == str(xyz_file)
    assert idx.file_size == stat.st_size
    assert abs(idx.file_mtime - stat.st_mtime) < 1e-3


# ---------------------------------------------------------------------------
# read_xyz_frame — random access
# ---------------------------------------------------------------------------


def test_read_frame_first(xyz_file: Path, xyz_index: FrameIndex):
    frame = read_xyz_frame(xyz_file, 0, xyz_index)
    assert frame.frame_number == 0
    assert frame.atom_count == _ATOM_COUNT
    assert len(frame.elements) == _ATOM_COUNT
    assert frame.coords.shape == (_ATOM_COUNT, 3)


def test_read_frame_middle(xyz_file: Path, xyz_index: FrameIndex):
    frame = read_xyz_frame(xyz_file, 1, xyz_index)
    assert frame.frame_number == 1


def test_read_frame_last(xyz_file: Path, xyz_index: FrameIndex):
    frame = read_xyz_frame(xyz_file, 2, xyz_index)
    assert frame.frame_number == 2


def test_read_frame_coords_distinct(xyz_file: Path, xyz_index: FrameIndex):
    """Frames 0 and 1 must have different first-atom x coordinates."""
    f0 = read_xyz_frame(xyz_file, 0, xyz_index)
    f1 = read_xyz_frame(xyz_file, 1, xyz_index)
    assert f0.coords[0, 0] != f1.coords[0, 0]


def test_read_frame_out_of_range(xyz_file: Path, xyz_index: FrameIndex):
    with pytest.raises(IndexError):
        read_xyz_frame(xyz_file, 99, xyz_index)


def test_read_frame_step_number(xyz_file: Path, xyz_index: FrameIndex):
    for i in range(3):
        frame = read_xyz_frame(xyz_file, i, xyz_index)
        assert frame.step_number == i * 10


def test_read_frame_bead_comment(xyz_file: Path, xyz_index: FrameIndex):
    frame = read_xyz_frame(xyz_file, 0, xyz_index)
    assert frame.bead_comment == 0


def test_read_frame_elements_are_carbon(xyz_file: Path, xyz_index: FrameIndex):
    frame = read_xyz_frame(xyz_file, 0, xyz_index)
    assert all(e == "C" for e in frame.elements)


# ---------------------------------------------------------------------------
# iter_xyz_frames — sequential streaming access
# ---------------------------------------------------------------------------


def test_iter_xyz_frames_yields_all_frames(xyz_file: Path):
    frames = list(iter_xyz_frames(xyz_file))
    assert len(frames) == 3
    assert [frame.frame_number for frame in frames] == [0, 1, 2]


def test_iter_xyz_frames_sets_local_indices_and_filename_ids(tmp_path: Path):
    path = tmp_path / "my_run.pos_03.xyz"
    path.write_bytes(_make_xyz_bytes(n_frames=2))
    frames = list(
        iter_xyz_frames(
            path,
            trajectory_id_pattern=r"^(.+)_\d+$",
            bead_id_pattern=r"_(\d+)$",
        )
    )
    assert [frame.local_frame_index for frame in frames] == [0, 1]
    assert all(frame.trajectory_id == "my_run.pos" for frame in frames)
    assert all(frame.bead_id == "03" for frame in frames)


# ---------------------------------------------------------------------------
# read_xyz_frame_by_offset
# ---------------------------------------------------------------------------


def test_read_by_offset_agrees_with_read_frame(xyz_file: Path, xyz_index: FrameIndex):
    for i in range(3):
        entry = xyz_index.entries[i]
        fr_by_num = read_xyz_frame(xyz_file, i, xyz_index)
        fr_by_off = read_xyz_frame_by_offset(xyz_file, entry.byte_offset)
        np.testing.assert_array_almost_equal(fr_by_num.coords, fr_by_off.coords)
        assert fr_by_num.comment_line == fr_by_off.comment_line


# ---------------------------------------------------------------------------
# extract_frame_metadata
# ---------------------------------------------------------------------------


def test_extract_metadata_keys(xyz_file: Path, xyz_index: FrameIndex):
    meta = extract_frame_metadata(xyz_file, 1, xyz_index)
    expected_keys = {
        "source_file", "frame_number", "byte_offset",
        "atom_count", "comment_line", "step_number", "bead_comment", "energy",
    }
    assert expected_keys <= set(meta.keys())


def test_extract_metadata_values(xyz_file: Path, xyz_index: FrameIndex):
    meta = extract_frame_metadata(xyz_file, 1, xyz_index)
    assert meta["frame_number"] == 1
    assert meta["atom_count"] == _ATOM_COUNT
    assert meta["step_number"] == 10  # frame 1 → Step 10


# ---------------------------------------------------------------------------
# load_or_build_xyz_index — caching
# ---------------------------------------------------------------------------


def test_cache_created(xyz_file: Path):
    cache_path = xyz_file.parent / (xyz_file.name + ".frameindex.npz")
    assert not cache_path.exists()
    load_or_build_xyz_index(xyz_file)
    assert cache_path.exists()


def test_cache_round_trip(xyz_file: Path):
    idx1 = load_or_build_xyz_index(xyz_file)
    idx2 = load_or_build_xyz_index(xyz_file)
    assert len(idx1) == len(idx2)
    for e1, e2 in zip(idx1.entries, idx2.entries):
        assert e1.byte_offset == e2.byte_offset
        assert e1.atom_count == e2.atom_count


def test_cache_invalidated_on_size_change(xyz_file: Path):
    """Appending a frame changes file size → cache must be discarded."""
    idx1 = load_or_build_xyz_index(xyz_file)
    original_count = len(idx1)

    # Append another frame
    extra = _make_xyz_bytes(n_frames=1)
    with open(xyz_file, "ab") as fh:
        fh.write(extra)

    idx2 = load_or_build_xyz_index(xyz_file)
    assert len(idx2) == original_count + 1, "Cache should have been invalidated"


def test_cache_explicit_path(xyz_file: Path, tmp_path: Path):
    custom_cache = tmp_path / "custom.cache.npz"
    load_or_build_xyz_index(xyz_file, cache_path=custom_cache)
    assert custom_cache.exists()
    # Second load should use the custom cache
    idx = load_or_build_xyz_index(xyz_file, cache_path=custom_cache)
    assert len(idx) == 3


# ---------------------------------------------------------------------------
# _parse_comment_line
# ---------------------------------------------------------------------------


def test_parse_comment_step_bead():
    comment = (
        "# CELL(abcABC):  10.0  10.0  10.0  90.0  90.0  90.0  "
        "Step:     42  Bead:     3  positions{angstrom}  cell{angstrom}"
    )
    meta = _parse_comment_line(comment)
    assert meta["step_number"] == 42
    assert meta["bead_comment"] == 3
    assert meta["energy"] is None


def test_parse_comment_no_step():
    meta = _parse_comment_line("# just a random comment")
    assert meta["step_number"] is None
    assert meta["bead_comment"] is None
    assert meta["energy"] is None


# ---------------------------------------------------------------------------
# _derive_trajectory_id / _derive_bead_id
# ---------------------------------------------------------------------------


def test_derive_trajectory_id_pattern(tmp_path: Path):
    p = tmp_path / "my_run.pos_07.xyz"
    traj = _derive_trajectory_id(p, pattern=r"^(.+)_\d+$")
    assert traj == "my_run.pos"


def test_derive_trajectory_id_fallback(tmp_path: Path):
    """No pattern match → fallback to parent directory name."""
    p = tmp_path / "single_file.xyz"
    traj = _derive_trajectory_id(p, pattern=None)
    assert traj == tmp_path.name or traj == "single_file"


def test_derive_bead_id_pattern(tmp_path: Path):
    p = tmp_path / "my_run.pos_07.xyz"
    bead = _derive_bead_id(p, pattern=r"_(\d+)$")
    assert bead == "07"


def test_derive_bead_id_no_pattern(tmp_path: Path):
    p = tmp_path / "my_run.pos_07.xyz"
    bead = _derive_bead_id(p, pattern=None)
    assert bead is None


def test_derive_bead_id_no_match(tmp_path: Path):
    p = tmp_path / "no_digits_here.xyz"
    bead = _derive_bead_id(p, pattern=r"_(\d+)$")
    assert bead is None


# ---------------------------------------------------------------------------
# iter_xyz_frames — start_frame
# ---------------------------------------------------------------------------


def test_iter_start_frame_skips_early_frames(tmp_path: Path):
    p = tmp_path / "t.xyz"
    p.write_bytes(_make_xyz_bytes(n_frames=5))
    frames = list(iter_xyz_frames(p, start_frame=2))
    assert len(frames) == 3
    assert all(f.local_frame_index >= 2 for f in frames)


def test_iter_start_frame_combined_with_max_frames(tmp_path: Path):
    p = tmp_path / "t.xyz"
    p.write_bytes(_make_xyz_bytes(n_frames=8))
    frames = list(iter_xyz_frames(p, start_frame=2, max_frames=5))
    assert [f.local_frame_index for f in frames] == [2, 3, 4]


def test_iter_start_frame_beyond_end_yields_nothing(tmp_path: Path):
    p = tmp_path / "t.xyz"
    p.write_bytes(_make_xyz_bytes(n_frames=3))
    frames = list(iter_xyz_frames(p, start_frame=10))
    assert frames == []


def test_iter_start_frame_zero_same_as_default(tmp_path: Path):
    p = tmp_path / "t.xyz"
    p.write_bytes(_make_xyz_bytes(n_frames=3))
    frames_default = list(iter_xyz_frames(p))
    frames_explicit = list(iter_xyz_frames(p, start_frame=0))
    assert len(frames_default) == len(frames_explicit) == 3
    for a, b in zip(frames_default, frames_explicit):
        assert a.local_frame_index == b.local_frame_index
