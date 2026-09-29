"""Tests for scripts/md17_to_xyz.py (MD17 npz -> multi-frame xyz)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.io_xyz import read_xyz_frame, scan_xyz_frame_offsets

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "md17_to_xyz.py"
_spec = importlib.util.spec_from_file_location("md17_to_xyz", _SCRIPT)
md17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md17)


def _write_npz(path: Path, n_frames: int = 5, **overrides) -> dict:
    rng = np.random.default_rng(0)
    arrays = {
        "z": np.array([6, 8, 1], dtype=np.uint8),
        "R": rng.normal(size=(n_frames, 3, 3)),
        "E": rng.normal(size=(n_frames, 1)) - 1000.0,
        "name": np.bytes_(b"toy"),
    }
    arrays.update(overrides)
    np.savez(path, **arrays)
    return arrays


def test_round_trip_through_xyz_reader(tmp_path):
    arrays = _write_npz(tmp_path / "toy.npz")
    out = tmp_path / "toy.xyz"

    assert md17.md17_npz_to_xyz(tmp_path / "toy.npz", out) == 5

    index = scan_xyz_frame_offsets(out)
    assert len(index.entries) == 5
    for i in (0, 2, 4):
        frame = read_xyz_frame(out, i, index)
        assert frame.elements == ["C", "O", "H"]
        np.testing.assert_allclose(frame.coords, arrays["R"][i], atol=1e-6)
        assert frame.step_number == i
        assert f"energy_kcal_mol={arrays['E'][i, 0]:.6f}" in frame.comment_line


def test_stride_keeps_original_step_numbers(tmp_path):
    _write_npz(tmp_path / "toy.npz", n_frames=7)
    out = tmp_path / "toy.xyz"

    assert md17.md17_npz_to_xyz(tmp_path / "toy.npz", out, stride=3) == 3

    index = scan_xyz_frame_offsets(out)
    assert [read_xyz_frame(out, i, index).step_number for i in range(3)] == [0, 3, 6]


def test_unknown_element_fails(tmp_path):
    _write_npz(tmp_path / "bad.npz", z=np.array([6, 99, 1]))
    with pytest.raises(ValueError, match="unknown atomic numbers"):
        md17.md17_npz_to_xyz(tmp_path / "bad.npz", tmp_path / "bad.xyz")


def test_shape_mismatch_fails(tmp_path):
    _write_npz(tmp_path / "bad.npz", R=np.zeros((4, 2, 3)))
    with pytest.raises(ValueError, match="expected"):
        md17.md17_npz_to_xyz(tmp_path / "bad.npz", tmp_path / "bad.xyz")


def test_checksum_mismatch_fails(tmp_path):
    path = tmp_path / "md17.npz"
    path.write_bytes(b"not the real file")
    with pytest.raises(ValueError, match="SHA-256"):
        md17.download_md17(path, url="unused://", sha256="0" * 64)
