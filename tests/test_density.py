"""Tests for confana/density.py and confana/plots_static.py (Phases 7 and 10)."""

from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confana.density import compute_2d_histogram
from confana.models import CoordinatePair
from confana.plots_static import (
    _compute_state_bin_com_positions,
    _prepare_density_colormap,
    _prepare_density_values,
    _state_com_grid_from_config,
    make_density_png,
    make_transition_png,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_angle_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame with dihedral angle columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "carboxyl_dihedral": rng.uniform(-180.0, 180.0, n),
            "ester_dihedral": rng.uniform(-180.0, 180.0, n),
        }
    )


def _dihedral_pair() -> CoordinatePair:
    return CoordinatePair(
        name="dihedral",
        x_col="carboxyl_dihedral",
        y_col="ester_dihedral",
        x_label="Carboxyl dihedral (°)",
        y_label="Ester dihedral (°)",
        title="Dihedral density",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )


def _toy_pair() -> CoordinatePair:
    return CoordinatePair(
        name="toy",
        x_col="x",
        y_col="y",
        x_label="x",
        y_label="y",
        title="toy density",
        x_domain=(0.0, 40.0),
        y_domain=(0.0, 40.0),
        periodic=False,
    )


# ---------------------------------------------------------------------------
# compute_2d_histogram
# ---------------------------------------------------------------------------


def test_compute_2d_histogram_shape():
    df = _make_angle_df()
    H, x_edges, y_edges = compute_2d_histogram(
        df, "carboxyl_dihedral", "ester_dihedral", bins=50,
        x_range=(-180, 180), y_range=(-180, 180),
    )
    assert H.shape == (50, 50)
    assert len(x_edges) == 51
    assert len(y_edges) == 51


def test_compute_2d_histogram_counts_total():
    df = _make_angle_df(n=200)
    H, _, _ = compute_2d_histogram(
        df, "carboxyl_dihedral", "ester_dihedral", bins=50,
        x_range=(-180, 180), y_range=(-180, 180),
    )
    assert int(H.sum()) == len(df)


def test_compute_2d_histogram_dihedral_range():
    df = _make_angle_df()
    H, x_edges, y_edges = compute_2d_histogram(
        df, "carboxyl_dihedral", "ester_dihedral", bins=20,
        x_range=(-180.0, 180.0), y_range=(-180.0, 180.0),
    )
    assert x_edges[0] == pytest.approx(-180.0)
    assert x_edges[-1] == pytest.approx(180.0)


def test_compute_2d_histogram_empty_df():
    empty = pd.DataFrame({"carboxyl_dihedral": pd.Series([], dtype=float),
                          "ester_dihedral": pd.Series([], dtype=float)})
    H, x_edges, y_edges = compute_2d_histogram(
        empty, "carboxyl_dihedral", "ester_dihedral", bins=10,
        x_range=(-180, 180), y_range=(-180, 180),
    )
    assert H.shape == (10, 10)
    assert H.sum() == 0


def test_compute_2d_histogram_na_rows_dropped():
    df = _make_angle_df(n=10)
    df.loc[0, "carboxyl_dihedral"] = float("nan")
    df.loc[3, "ester_dihedral"] = float("nan")
    H, _, _ = compute_2d_histogram(
        df, "carboxyl_dihedral", "ester_dihedral", bins=10,
        x_range=(-180, 180), y_range=(-180, 180),
    )
    assert int(H.sum()) == 8  # 10 - 2 NA rows


def test_compute_2d_histogram_missing_column_raises():
    df = _make_angle_df()
    with pytest.raises(ValueError, match="not found"):
        compute_2d_histogram(df, "nonexistent_col", "ester_dihedral", bins=10)


def test_compute_2d_histogram_default_range():
    """When x_range/y_range are None the data extent is used."""
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    H, x_edges, y_edges = compute_2d_histogram(df, "x", "y", bins=4)
    assert H.sum() == 3


# ---------------------------------------------------------------------------
# make_density_png
# ---------------------------------------------------------------------------


def test_make_density_png_dihedral(tmp_path):
    df = _make_angle_df()
    out = make_density_png(df, _dihedral_pair(), tmp_path / "dihedral.png", dpi=72)
    assert out.exists()
    assert out.suffix == ".png"


def test_make_density_png_supports_configured_nondefault_dihedral_pair(tmp_path):
    df = pd.DataFrame(
        {
            "igor1_dihedral": np.linspace(-180.0, 180.0, 100),
            "igor2_dihedral": np.linspace(180.0, -180.0, 100),
        }
    )
    igor_pair = CoordinatePair(
        name="dihedral",
        x_col="igor1_dihedral",
        y_col="igor2_dihedral",
        x_label="Igor 1 (°)",
        y_label="Igor 2 (°)",
        title="Igor torsion map",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )
    out = make_density_png(df, igor_pair, tmp_path / "igor_pair.png", dpi=72)
    assert out.exists()


def test_make_density_png_accepts_custom_pair_name(tmp_path):
    df = pd.DataFrame(
        {
            "igor1_dihedral": np.linspace(-180.0, 180.0, 50),
            "igor2_dihedral": np.linspace(180.0, -180.0, 50),
        }
    )
    igor_pair = CoordinatePair(
        name="dihedrals_igor",
        x_col="igor1_dihedral",
        y_col="igor2_dihedral",
        x_label="Igor 1 (°)",
        y_label="Igor 2 (°)",
        title="Igor torsion map",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )
    out = make_density_png(df, igor_pair, tmp_path / "igor_pair_named.png", dpi=72)
    assert out.exists()


def test_make_density_png_creates_file(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "density_dihedral.png"
    assert not outpath.exists()
    make_density_png(df, _dihedral_pair(), outpath, dpi=72)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_make_density_png_missing_column_raises(tmp_path):
    df = _make_angle_df().drop(columns=["carboxyl_dihedral"])
    with pytest.raises(ValueError):
        make_density_png(df, _dihedral_pair(), tmp_path / "out.png", dpi=72)


def test_make_density_png_with_config(tmp_path):
    df = _make_angle_df()
    cfg = {
        "density": {
            "bins": 20,
            "colormap": "plasma",
            "log_scale": False,
        }
    }
    out = make_density_png(df, _dihedral_pair(), tmp_path / "custom.png", dpi=72, config=cfg)
    assert out.exists()


def test_make_density_png_uses_definition_specific_bins(tmp_path):
    df = _make_angle_df()
    cfg = {
        "density": {
            "bins": 20,
            "dihedral_bins": 180,
            "colormap": "plasma",
            "log_scale": False,
        }
    }
    out = make_density_png(df, _dihedral_pair(), tmp_path / "dihedral_bins.png", dpi=72, config=cfg)
    assert out.exists()


def test_make_density_png_creates_parent_dirs(tmp_path):
    df = _make_angle_df()
    nested = tmp_path / "a" / "b" / "c" / "density.png"
    make_density_png(df, _dihedral_pair(), nested, dpi=72)
    assert nested.exists()


def test_state_bin_com_positions_use_population_weighted_bin_centres():
    df = pd.DataFrame(
        {
            "carboxyl_dihedral": [0.1] * 9 + [19.9] + [5.0],
            "ester_dihedral": [0.1] * 9 + [19.9] + [5.0],
            "state_dihedral": ["0"] * 10 + ["noise"],
        }
    )
    edges = np.array([0.0, 10.0, 20.0])

    positions = _compute_state_bin_com_positions(
        df,
        "state_dihedral",
        "carboxyl_dihedral",
        "ester_dihedral",
        edges,
        edges,
        (0.0, 20.0),
        (0.0, 20.0),
        periodic=False,
    )

    assert positions.loc["0", "carboxyl_dihedral"] == pytest.approx(6.0)
    assert positions.loc["0", "ester_dihedral"] == pytest.approx(6.0)


def test_state_bin_com_positions_use_circular_mean_for_periodic_axes():
    df = pd.DataFrame(
        {
            "carboxyl_dihedral": [-175.0, 175.0],
            "ester_dihedral": [0.0, 0.0],
            "state_dihedral": ["0", "0"],
        }
    )

    positions = _compute_state_bin_com_positions(
        df,
        "state_dihedral",
        "carboxyl_dihedral",
        "ester_dihedral",
        np.array([-180.0, -170.0, 170.0, 180.0]),
        np.array([-180.0, 180.0]),
        (-180.0, 180.0),
        (-180.0, 180.0),
        periodic=True,
    )

    assert positions.loc["0", "carboxyl_dihedral"] == pytest.approx(-180.0)
    assert positions.loc["0", "ester_dihedral"] == pytest.approx(0.0)


def test_state_com_grid_from_config_prefers_clustering_bins_over_density_bins():
    density_edges = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    x_edges, y_edges, x_range, y_range = _state_com_grid_from_config(
        _toy_pair(),
        density_edges,
        density_edges,
        (0.0, 40.0),
        (0.0, 40.0),
        {
            "clustering": {
                "algorithm": "grid",
                "default": {"bin_size": 20.0, "min_count": 1},
            }
        },
    )

    np.testing.assert_allclose(x_edges, np.array([0.0, 20.0, 40.0]))
    np.testing.assert_allclose(y_edges, np.array([0.0, 20.0, 40.0]))
    assert x_range == (0.0, 40.0)
    assert y_range == (0.0, 40.0)


def test_prepare_density_values_masks_unsampled_bins():
    H = np.array([[0, 2], [3, 0]], dtype=np.int64)
    values = _prepare_density_values(H, log_scale=False)
    assert np.isnan(values[0, 0])
    assert np.isnan(values[1, 1])
    assert values[1, 0] == pytest.approx(2.0)
    assert values[0, 1] == pytest.approx(3.0)


def test_prepare_density_colormap_renders_nan_white():
    cmap = _prepare_density_colormap("viridis")
    rgb = cmap(np.array([np.nan]))[0, :3]
    np.testing.assert_allclose(rgb, np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# make_transition_png — Phase 10
# ---------------------------------------------------------------------------


def _make_transitions_dict(pair_name: str = "dihedral", n_states: int = 3,
                            with_averaged: bool = False,
                            with_barriers: bool = False) -> dict:
    """Build a minimal transitions result dict with synthetic n×n matrices."""
    labels = [str(i) for i in range(n_states)]
    rng = np.random.default_rng(42)

    def _df():
        data = np.abs(rng.standard_normal((n_states, n_states)))
        return pd.DataFrame(data, index=labels, columns=labels)

    per_group = {("traj0",): _df()}
    averaged = {("traj0",): _df()} if with_averaged else {}
    barriers = {("traj0",): _df()} if with_barriers else {}

    return {
        "pair_name": pair_name,
        "state_column": f"state_{pair_name}",
        "groupby": ["trajectory_id"],
        "average_groupby": ["bead_id"],
        "barrier_energy_unit": "eV",
        "per_group_counts": per_group,
        "per_group_probabilities": per_group,
        "per_group_rates": {},
        "per_group_barriers": barriers,
        "averaged_counts": averaged,
        "averaged_probabilities": averaged,
        "averaged_rates": {},
        "averaged_barriers": {},
    }


def test_make_transition_png_creates_file(tmp_path):
    transitions = _make_transitions_dict("dihedral")
    outpath = tmp_path / "transition_dihedral.png"
    make_transition_png(transitions, outpath)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_make_transition_png_returns_path(tmp_path):
    transitions = _make_transitions_dict("dihedral")
    outpath = tmp_path / "transition_dihedral.png"
    result = make_transition_png(transitions, outpath)
    assert isinstance(result, Path)
    assert result == outpath.resolve() or result == outpath


def test_make_transition_png_with_averaged_matrices(tmp_path):
    """PIMD-like dict with averaged_probabilities uses averaged matrix (not per-group)."""
    transitions = _make_transitions_dict("dihedral", n_states=2, with_averaged=True)
    outpath = tmp_path / "transition_averaged.png"
    result = make_transition_png(transitions, outpath)
    assert result.exists()


def test_make_transition_png_with_barrier_panel(tmp_path):
    transitions = _make_transitions_dict("dihedral", n_states=2, with_barriers=True)
    outpath = tmp_path / "transition_barriers.png"
    result = make_transition_png(
        transitions,
        outpath,
        config={"transitions_plot": {"colormap_barriers": "YlOrRd"}},
    )
    assert result.exists()


# ---------------------------------------------------------------------------
# population_free_energy
# ---------------------------------------------------------------------------


def test_population_free_energy_properties():
    from confana.density import population_free_energy

    counts = np.array([[0, 1, 4], [16, 2, 0]])
    fe = population_free_energy(counts)

    assert fe.shape == counts.shape
    # Most-populated bin is 0 (and not -0.0).
    assert fe[1, 0] == 0.0 and not np.signbit(fe[1, 0])
    # Unsampled bins are NaN.
    assert np.isnan(fe[0, 0]) and np.isnan(fe[1, 2])
    sampled = counts > 0
    assert np.all(fe[sampled] >= 0.0)
    np.testing.assert_allclose(fe[sampled], -np.log(counts[sampled] / 16.0))
    # Monotone: more frames → lower value.
    order = np.argsort(counts[sampled])
    assert np.all(np.diff(fe[sampled][order]) <= 0.0)


def test_population_free_energy_all_empty_is_nan():
    from confana.density import population_free_energy

    assert np.all(np.isnan(population_free_energy(np.zeros((3, 3)))))


def test_population_free_energy_rejects_negative_counts():
    from confana.density import population_free_energy

    with pytest.raises(ValueError, match="non-negative"):
        population_free_energy(np.array([1, -1]))
