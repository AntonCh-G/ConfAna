"""Tiny HDF5 PIMD runs for tests, laid out as the HDF5 input adapter expects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confana.io_hdf5 import build_coordinate_table_from_hdf5
from confana.models import DoFDefinition


def write_hdf5_run(sim_dir: Path, n_frames: int = 5, n_beads: int = 2):
    """Write ``<sim_dir>/hdf5/trajectory.hdf5`` and its ``input.xyz`` (C, O, H)."""
    h5py = pytest.importorskip("h5py")
    (sim_dir / "hdf5").mkdir(parents=True)
    h5_path = sim_dir / "hdf5" / "trajectory.hdf5"
    beads = np.random.default_rng(0).standard_normal((n_frames, n_beads, 3, 3))
    centroid = beads.mean(axis=1)
    with h5py.File(h5_path, "w") as fh:
        fh.create_dataset("bead_positions", data=beads)
        fh.create_dataset("positions", data=centroid)
        fh.create_dataset("potential", data=np.zeros(n_frames))
    (sim_dir / "input.xyz").write_text("3\ninput\nC 0 0 0\nO 0 0 0\nH 0 0 0\n")
    return h5_path, beads, centroid


def hdf5_table(h5_path: Path, positions_source: str = "bead") -> pd.DataFrame:
    """Coordinate table of the run, exactly as the HDF5 input adapter builds it."""
    dofs = [
        DoFDefinition(name=name, type="distance", atoms=atoms, label=name, domain=[0.0, 10.0],
                      enabled=True)
        for name, atoms in (("d_co", [0, 1]), ("d_oh", [1, 2]))
    ]
    return build_coordinate_table_from_hdf5(h5_path, dofs, positions_source=positions_source)
