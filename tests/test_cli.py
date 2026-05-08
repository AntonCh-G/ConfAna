"""Tests for src/cli.py (Phase 13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from src.cli import cli


# ---------------------------------------------------------------------------
# Helpers — minimal config + fixture data
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal config YAML that points to tmp_path for all outputs."""
    cfg = {
        "data": {
            "path_pattern": str(tmp_path / "*.xyz"),
            "trajectory_id_pattern": None,
            "bead_id_pattern": None,
        },
        "dof": [
            {
                "name": "carboxyl_dihedral",
                "type": "dihedral",
                "atoms": [0, 1, 2, 3],
                "label": "Carboxyl dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
            {
                "name": "ester_dihedral",
                "type": "dihedral",
                "atoms": [1, 2, 3, 4],
                "label": "Ester dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
            {
                "name": "igor1_dihedral",
                "type": "dihedral",
                "atoms": [2, 3, 4, 5],
                "label": "Igor 1 dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
            {
                "name": "igor2_dihedral",
                "type": "dihedral",
                "atoms": [3, 4, 5, 6],
                "label": "Igor 2 dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
        ],
        "coordinate_pairs": {
            "dihedral": {
                "x": "carboxyl_dihedral",
                "y": "ester_dihedral",
            },
            "dihedrals_igor": {
                "x": "igor1_dihedral",
                "y": "igor2_dihedral",
            },
        },
        "conventions": {
            "plane_signed": False,
            "dihedral_signed": True,
        },
        "cache": {
            "index_cache_dir": None,   # store alongside xyz file (no separate dir needed)
            "trajectory_cache_dir": str(tmp_path / "cache"),
            "coordinate_table_path": str(tmp_path / "coords.npz"),
        },
        "density": {
            "bins": 10,
            "plane_bins": 10,
            "dihedral_bins": 10,
            "colormap": "viridis",
            "log_scale": True,
            "plane_x_range": [0, 180],
            "plane_y_range": [0, 180],
            "dihedral_x_range": [-180, 180],
            "dihedral_y_range": [-180, 180],
        },
        "clustering": {
            "algorithm": "dbscan",
            "groupby": None,
            "eps": None,
            "min_samples": None,
            "plane": {"eps": 30, "min_samples": 2},
            "dihedral": {"eps": 30, "min_samples": 2},
            "dihedrals_igor": {"eps": 30, "min_samples": 2},
        },
        "transitions": {"lag": 1, "dt": None},
        "pimd": {"enabled": False, "average_across_beads": False},
        "outputs": {
            "dir": str(tmp_path / "outputs"),
            "dpi": 72,
            "save_csv": True,
            "save_parquet": False,
        },
        "transitions_plot": {
            "colormap_counts": "Blues",
            "colormap_probs": "viridis",
            "colormap_rates": "plasma",
            "max_annotate_states": 10,
        },
        "interactive": {
            "embed_xyz_payload": False,
            "include_plotlyjs": "cdn",
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return config_path


def _write_minimal_xyz(path: Path, n_frames: int = 5, n_atoms: int = 9) -> None:
    """Write a minimal xyz file with *n_frames* frames of *n_atoms* atoms.

    Atoms are placed on a circle in the z=0 plane so that the first three
    atoms (ring_plane indices [0,1,2] in the test config) are not collinear.
    A small z-perturbation is added so the dihedral angle is well-defined.
    """
    import math
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n_frames):
            fh.write(f"{n_atoms}\n")
            fh.write(f"frame {i}\n")
            for j in range(n_atoms):
                angle = 2 * math.pi * j / n_atoms
                x = math.cos(angle)
                y = math.sin(angle)
                z = 0.1 * (j % 3)  # small non-zero z so planes/dihedrals are non-degenerate
                fh.write(f"C  {x:.4f}  {y:.4f}  {z:.4f}\n")


# ---------------------------------------------------------------------------
# Help / smoke tests
# ---------------------------------------------------------------------------


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ConfAna" in result.output


def test_run_all_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run-all", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_extract_coordinates_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["extract-coordinates", "--help"])
    assert result.exit_code == 0


def test_plot_densities_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["plot-densities", "--help"])
    assert result.exit_code == 0


def test_cluster_states_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["cluster-states", "--help"])
    assert result.exit_code == 0


def test_compute_transitions_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["compute-transitions", "--help"])
    assert result.exit_code == 0


def test_build_interactive_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["build-interactive", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# extract-coordinates: basic smoke test
# ---------------------------------------------------------------------------


def test_extract_coordinates_creates_csv(tmp_path):
    """extract-coordinates saves a CSV when save_csv is true."""
    _write_minimal_xyz(tmp_path / "traj.xyz", n_frames=5, n_atoms=9)
    config_path = _write_minimal_config(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["extract-coordinates", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    csv_path = tmp_path / "outputs" / "coordinates_angles.csv"
    assert csv_path.exists(), f"CSV not found; CLI output:\n{result.output}"


def test_extract_coordinates_missing_config():
    """Nonexistent config file causes a non-zero exit."""
    runner = CliRunner()
    result = runner.invoke(cli, ["extract-coordinates", "--config", "/no/such/file.yaml"])
    assert result.exit_code != 0


def test_plot_densities_creates_extra_pair_outputs(tmp_path):
    _write_minimal_xyz(tmp_path / "traj.xyz", n_frames=5, n_atoms=9)
    config_path = _write_minimal_config(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["plot-densities", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs" / "density_dihedral.png").exists()
    assert (tmp_path / "outputs" / "density_dihedrals_igor.png").exists()


def test_build_interactive_creates_extra_pair_outputs(tmp_path):
    _write_minimal_xyz(tmp_path / "traj.xyz", n_frames=5, n_atoms=9)
    config_path = _write_minimal_config(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["build-interactive", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs" / "density_dihedral.html").exists()
    assert (tmp_path / "outputs" / "density_dihedrals_igor.html").exists()
