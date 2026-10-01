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
- ``describe_hdf5_source``, ``read_hdf5_blocks``, ``run_trajectory_id``: the
  HDF5 trajectory reader of the coordinate-table build
  (``confana.coordinate_table``)
- ``HDF5FrameReader``: single frames, for the interactive page
- ``HDF5_BYTE_OFFSET``
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np

from confana.io_xyz import read_xyz_frame_by_offset
from confana.models import FrameBlock, FrameRecord, TrajectorySource

HDF5_BYTE_OFFSET: int = -1
"""``byte_offset`` of every HDF5-sourced row: HDF5 frames have no byte position
and are read by ``frame_number`` and ``bead_id`` instead (see :class:`HDF5FrameReader`)."""

_BEAD_LABEL = re.compile(r"^bead_(\d+)$")


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
    return read_xyz_frame_by_offset(input_xyz, 0).elements


def _bead_label(bead: int) -> str:
    """``bead_id`` of bead number *bead* in an HDF5 run, e.g. ``"bead_03"``."""
    return f"bead_{bead:02d}"


def _bead_index(bead_id: str) -> int:
    """Bead number of an HDF5 ``bead_id`` (inverse of :func:`_bead_label`).

    Raises
    ------
    ValueError
        If *bead_id* is not of the form ``bead_<number>``.
    """
    match = _BEAD_LABEL.match(str(bead_id))
    if match is None:
        raise ValueError(f"bead_id {bead_id!r} is not an HDF5 bead label (bead_<number>).")
    return int(match.group(1))


def _import_h5py():
    """Return the ``h5py`` module, or raise with an install hint."""
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 trajectories. Install with: uv add h5py"
        ) from exc
    return h5py


def _check_atom_types(h5_path: Path, n_atoms: int, atom_types: list[str]) -> None:
    """Raise if the HDF5 frames and ``input.xyz`` disagree on the atom count."""
    if n_atoms != len(atom_types):
        raise ValueError(
            f"{h5_path.name}: HDF5 has {n_atoms} atoms per frame but "
            f"input.xyz has {len(atom_types)} atom-type entries."
        )


def run_trajectory_id(h5_path: Path) -> str:
    """Return the ``trajectory_id`` of an HDF5 run: its simulation folder name.

    For ``/some/root/s1/hdf5/trajectory.hdf5`` this is ``'s1'``.
    """
    return Path(h5_path).parent.parent.name


# ---------------------------------------------------------------------------
# Trajectory reader for the coordinate-table build
# ---------------------------------------------------------------------------

HDF5_SUFFIXES: tuple[str, ...] = (".hdf5", ".h5")
"""File suffixes the coordinate-table build looks for in ``data.path_pattern``."""

_DATASETS = ("bead_positions", "positions", "potential")

_BLOCK_BYTES = 64 * 2**20
"""float64 bytes read per :class:`~confana.models.FrameBlock` (all streams)."""


def describe_hdf5_source(
    path: Path | str,
    *,
    positions_source: Literal["bead", "centroid"],
) -> TrajectorySource:
    """Describe one HDF5 run: one stream per bead, or one centroid stream.

    Raises
    ------
    FileNotFoundError
        If the run's ``input.xyz`` is missing.
    ValueError
        If a dataset is missing, the datasets disagree on frames or atoms, or
        the atom count differs from ``input.xyz``.
    """
    h5py = _import_h5py()
    path = Path(path).resolve()
    atom_types = _read_atom_types_from_input_xyz(path)
    with h5py.File(path, "r") as fh:
        missing = [name for name in _DATASETS if name not in fh]
        if missing:
            raise ValueError(
                f"{path}: no {', '.join(missing)} dataset; an HDF5 trajectory needs "
                f"{', '.join(_DATASETS)}."
            )
        beads, centroid, potential = (fh[name].shape for name in _DATASETS)
    if len(beads) != 4 or beads[3] != 3:
        raise ValueError(
            f"{path}: bead_positions has shape {beads}; expected (frames, beads, atoms, 3)."
        )
    n_frames, n_beads, n_atoms = beads[:3]
    if tuple(centroid) != (n_frames, n_atoms, 3) or tuple(potential) != (n_frames,):
        raise ValueError(
            f"{path}: positions {centroid} and potential {potential} do not match "
            f"bead_positions {beads}."
        )
    _check_atom_types(path, n_atoms, atom_types)
    bead_ids: tuple[str | None, ...] = (
        tuple(_bead_label(b) for b in range(n_beads)) if positions_source == "bead" else (None,)
    )
    return TrajectorySource(path=path, bead_ids=bead_ids, n_frames=n_frames, atom_count=n_atoms)


def read_hdf5_blocks(source: TrajectorySource, start: int, stop: int) -> Iterator[FrameBlock]:
    """Read frames ``start … stop-1`` of every stream of an HDF5 source in blocks.

    Each block reads one contiguous frame range of ``bead_positions`` (all
    beads), or of ``positions`` for a centroid source (one ``None`` stream),
    and turns it into float32. ``step_number`` is the frame index and
    ``energy`` the ``potential`` dataset.
    """
    h5py = _import_h5py()
    stop = min(stop, source.n_frames)
    if stop <= start:
        return
    elements = _read_atom_types_from_input_xyz(source.path)
    centroid = source.bead_ids == (None,)
    frame_bytes = len(source.bead_ids) * source.atom_count * 3 * 8
    block_frames = max(1, _BLOCK_BYTES // max(frame_bytes, 1))
    with h5py.File(source.path, "r") as fh:
        for first in range(start, stop, block_frames):
            end = min(first + block_frames, stop)
            if centroid:
                coords = fh["positions"][first:end][:, None].astype(np.float32)
            else:
                coords = fh["bead_positions"][first:end].astype(np.float32)
            frame_number = np.arange(first, end, dtype=np.int64)
            yield FrameBlock(
                coords=coords,
                frame_number=frame_number,
                byte_offset=np.full(end - first, HDF5_BYTE_OFFSET, dtype=np.int64),
                step_number=frame_number.copy(),
                step_missing=np.zeros(end - first, dtype=bool),
                energy=fh["potential"][first:end].astype(np.float32),
                elements=list(elements),
            )


# ---------------------------------------------------------------------------
# Single-frame reading
# ---------------------------------------------------------------------------


class HDF5FrameReader:
    """Reads single frames of one HDF5 run; the file stays open until closed.

    Use as a context manager. Elements come from the run's ``input.xyz``
    (see the module docstring); coordinates are float64 as stored.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file or its ``input.xyz`` is missing.
    ValueError
        If the file lacks ``bead_positions`` / ``positions`` or its atom count
        differs from ``input.xyz``.
    """

    def __init__(self, h5_path: Path | str) -> None:
        h5py = _import_h5py()
        self.path = Path(h5_path)
        self._elements = _read_atom_types_from_input_xyz(self.path)
        self._trajectory_id = run_trajectory_id(self.path)
        self._file = h5py.File(self.path, "r")
        try:
            missing = [name for name in ("bead_positions", "positions") if name not in self._file]
            if missing:
                raise ValueError(f"{self.path.name} has no {' or '.join(missing)} dataset.")
            self._beads = self._file["bead_positions"]
            self._centroid = self._file["positions"]
            self._n_frames, self._n_beads, self._n_atoms = self._beads.shape[:3]
            _check_atom_types(self.path, self._n_atoms, self._elements)
        except Exception:
            self._file.close()
            raise

    def read(self, frame_number: int, bead_id: str | None) -> FrameRecord:
        """Return frame *frame_number* of bead *bead_id*, or of the centroid when ``None``.

        Raises
        ------
        ValueError
            If the frame or bead is out of range, or *bead_id* is not a bead label.
        """
        if not 0 <= frame_number < self._n_frames:
            raise ValueError(
                f"frame {frame_number} is out of range (the file has {self._n_frames} frames)."
            )
        if bead_id is None:
            coords = self._centroid[frame_number]
        else:
            bead = _bead_index(bead_id)
            if not 0 <= bead < self._n_beads:
                raise ValueError(
                    f"{bead_id} is out of range (the file has {self._n_beads} beads)."
                )
            coords = self._beads[frame_number, bead]
        return FrameRecord(
            source_file=str(self.path),
            frame_number=frame_number,
            byte_offset=HDF5_BYTE_OFFSET,
            atom_count=self._n_atoms,
            comment_line="",
            elements=list(self._elements),
            coords=np.asarray(coords, dtype=np.float64),
            trajectory_id=self._trajectory_id,
            bead_id=bead_id,
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> HDF5FrameReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
