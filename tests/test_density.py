"""Tests for confana/density.py and confana/plots_static.py (Phases 7 and 10)."""

from __future__ import annotations


from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from confana.density import DensitySettings, build_conformational_map
from confana.models import CoordinatePair
from confana.plots_static import (
    _density_figure,
    _prepare_density_colormap,
    _prepare_density_values,
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
# Conformational map
# ---------------------------------------------------------------------------


def _map_of(xs, ys, bins: int = 4, span: tuple[float, float] = (0.0, 120.0), **columns):
    """Map of the toy pair over *span* on both axes (30° bins by default)."""
    table = pd.DataFrame({"x": xs, "y": ys, **columns})
    settings = DensitySettings(bins=bins, x_range=span, y_range=span)
    return build_conformational_map(table, _toy_pair(), settings)


def test_map_counts_only_frames_inside_the_map():
    # -170 is off the map and NaN has no value; 30 and 60 sit on interior
    # edges, which belong to the upper bin.
    conf_map = _map_of([-170.0, 30.0, 60.0, np.nan], [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(conf_map.x_edges, [0.0, 30.0, 60.0, 90.0, 120.0])
    np.testing.assert_array_equal(conf_map.counts[:, 0], [0, 1, 1, 0])
    assert conf_map.counts.sum() == 2


def test_map_bin_index_edges_follow_histogram_rule():
    conf_map = _map_of([], [])
    xi, yi = conf_map.bin_index(
        np.array([0.0, 29.9, 30.0, 119.9, 120.0, -0.1, 120.1, np.nan]),
        np.full(8, 45.0),
    )
    # Interior edge -> upper bin; right edge -> last bin; off map or NaN -> -1.
    np.testing.assert_array_equal(xi, [0, 0, 1, 3, 3, -1, -1, -1])
    np.testing.assert_array_equal(yi, [1, 1, 1, 1, 1, 1, 1, 1])


def test_map_counts_agree_with_bin_index():
    rng = np.random.default_rng(3)
    x = rng.uniform(-20.0, 140.0, 400).round()  # rounding puts many values on edges
    y = rng.uniform(-20.0, 140.0, 400).round()
    conf_map = _map_of(x, y)
    xi, yi = conf_map.bin_index(x, y)
    inside = (xi >= 0) & (yi >= 0)
    tally = np.zeros_like(conf_map.counts)
    np.add.at(tally, (xi[inside], yi[inside]), 1)
    np.testing.assert_array_equal(conf_map.counts, tally)


def test_map_representatives_are_the_counted_bins():
    # Frames at -170, 30 and 60 are counted in bins 1 and 2 only; an off-map
    # frame must never represent a bin, and every counted bin gets one.
    conf_map = _map_of([-170.0, 30.0, 60.0], [10.0, 10.0, 10.0])
    assert conf_map.representatives == {(1, 0): 1, (2, 0): 2}


def test_map_representative_is_nearest_to_bin_centre_ties_to_earliest_row():
    # Bin (0, 0) spans 0-30 on both axes, centre (15, 15). Rows 1 and 2 are
    # both 1 away from it; the earlier row wins, so the choice is deterministic.
    conf_map = _map_of([2.0, 14.0, 16.0, 15.0], [15.0, 15.0, 15.0, 29.0])
    assert conf_map.representatives == {(0, 0): 1}


def test_map_every_representative_lies_in_its_own_bin():
    rng = np.random.default_rng(4)
    x = rng.uniform(-20.0, 140.0, 300).round()
    y = rng.uniform(-20.0, 140.0, 300).round()
    conf_map = _map_of(x, y)
    reps = conf_map.representatives
    assert set(reps) == {(int(i), int(j)) for i, j in zip(*np.nonzero(conf_map.counts))}
    rows = np.array(list(reps.values()))
    xi, yi = conf_map.bin_index(x[rows], y[rows])
    assert list(zip(xi.tolist(), yi.tolist())) == list(reps)


def test_map_of_empty_table_has_no_frames():
    conf_map = _map_of([], [])
    assert conf_map.counts.shape == (4, 4)
    assert conf_map.counts.sum() == 0
    assert conf_map.representatives == {}


def test_density_settings_prefer_pair_then_config_then_domain():
    config = {"plots": {"density": {
        "bins": 20, "x_range": [0, 10], "y_range": [1, 39], "colormap": "magma", "log_scale": False,
    }}}
    pair = replace(_toy_pair(), bins=8, x_range=(5.0, 35.0))
    assert DensitySettings.for_pair(pair, config) == DensitySettings(
        bins=8, x_range=(5.0, 35.0), y_range=(1.0, 39.0), colormap="magma", log_scale=False,
    )
    styled = replace(_toy_pair(), colormap="cividis", log_scale=True)
    assert DensitySettings.for_pair(styled, config) == DensitySettings(
        bins=20, x_range=(0.0, 10.0), y_range=(1.0, 39.0), colormap="cividis", log_scale=True,
    )
    # A density-only section is read the same way; unset keys fall back to
    # the pair's domain and the defaults.
    assert DensitySettings.for_pair(_toy_pair(), {"bins": 20}).bins == 20
    assert DensitySettings.for_pair(_toy_pair()) == DensitySettings(
        bins=180, x_range=(0.0, 40.0), y_range=(0.0, 40.0), colormap="viridis", log_scale=True,
    )


@pytest.mark.parametrize(
    "density",
    [
        {"bins": 0}, {"bins": 2.5}, {"bins": True}, {"bins": None},  # never rounded or guessed
        {"x_range": [10, 10]}, {"y_range": [5, 1]}, {"x_range": [0, None]}, {"y_range": [1]},
    ],
)
def test_density_settings_reject_bad_bins_and_ranges(density):
    with pytest.raises(ValueError, match="pair 'toy'"):
        DensitySettings.for_pair(_toy_pair(), {"plots": {"density": density}})


def test_map_missing_column_raises():
    settings = DensitySettings(bins=4, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    with pytest.raises(ValueError, match=r"pair 'toy'.*'y'"):
        build_conformational_map(pd.DataFrame({"x": [0.5]}), _toy_pair(), settings)


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


def test_density_png_markers_are_the_first_groups_state_centres():
    # Label "0" is a different region in each bead: the PNG shows bead 00's
    # centre (the group the interactive page starts on), never a pooled one.
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    table = pd.DataFrame({
        "x": [5.0, 7.0, 35.0, 33.0],
        "y": [5.0, 7.0, 35.0, 33.0],
        "state_toy": ["0", "0", "0", "0"],
        "bead_id": ["00", "00", "01", "01"],
    })
    settings = DensitySettings.for_pair(_toy_pair(), {"bins": 4})
    fig = _density_figure(build_conformational_map(table, _toy_pair(), settings), groupby=["bead_id"])
    try:
        ax = fig.axes[0]
        (markers,) = [c for c in ax.collections if isinstance(c, PathCollection)]
        np.testing.assert_allclose(markers.get_offsets(), [[6.0, 6.0]])
        assert [t.get_text() for t in ax.texts] == ["0"]
        assert ax.get_title() == "toy density — bead 00"
    finally:
        plt.close(fig)


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
