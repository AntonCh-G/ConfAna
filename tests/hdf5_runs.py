"""Tiny HDF5 PIMD runs for tests, laid out as the HDF5 input adapter expects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ELEMENTS = ("C", "O", "H")


def write_hdf5_run(sim_dir: Path, n_frames: int = 5, n_beads: int = 2, n_atoms: int = 3):
    """Write ``<sim_dir>/hdf5/trajectory.hdf5`` and its ``input.xyz``.

    Atoms cycle through C, O, H; positions are random, the potential is
    ``-frame`` eV. Returns ``(h5_path, bead_positions, centroid)``.
    """
    h5py = pytest.importorskip("h5py")
    (sim_dir / "hdf5").mkdir(parents=True)
    h5_path = sim_dir / "hdf5" / "trajectory.hdf5"
    beads = np.random.default_rng(0).standard_normal((n_frames, n_beads, n_atoms, 3))
    centroid = beads.mean(axis=1)
    with h5py.File(h5_path, "w") as fh:
        fh.create_dataset("bead_positions", data=beads)
        fh.create_dataset("positions", data=centroid)
        fh.create_dataset("potential", data=-np.arange(n_frames, dtype=float))
    elements = [_ELEMENTS[i % len(_ELEMENTS)] for i in range(n_atoms)]
    (sim_dir / "input.xyz").write_text(
        f"{n_atoms}\ninput\n" + "".join(f"{el} 0 0 0\n" for el in elements)
    )
    return h5_path, beads, centroid


def hdf5_table(h5_path: Path, positions_source: str = "bead") -> pd.DataFrame:
    """Coordinate table of the run, built by the coordinate-table pipeline."""
    from confana.coordinate_table import load_or_build_coordinate_table_from_config

    config = {
        "run_dir": str(Path(h5_path).parents[2] / "hdf5_table_run"),
        "data": {
            "format": "hdf5",
            "path_pattern": str(h5_path),
            "positions_source": positions_source,
        },
        "dof": [
            {"name": name, "type": "distance", "atoms": atoms, "label": name,
             "domain": [0.0, 10.0]}
            for name, atoms in (("d_co", [0, 1]), ("d_oh", [1, 2]))
        ],
    }
    table, _ = load_or_build_coordinate_table_from_config(config)
    return table
