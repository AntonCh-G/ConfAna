"""Tests for src/states.py (Phase 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import CoordinatePair
from src.states import (
    _resolve_pair_params,
    assign_conformer_states,
    assign_conformer_states_from_config,
    build_bin_state_overlay,
    resolve_state_groupby,
)


# ---------------------------------------------------------------------------
# CoordinatePair fixtures
# ---------------------------------------------------------------------------


def _plane_pair(signed: bool = False) -> CoordinatePair:
    domain = (-180.0, 180.0) if signed else (0.0, 180.0)
    return CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="Carboxyl plane (°)",
        y_label="Ester plane (°)",
        title="Plane pair",
        x_domain=domain,
        y_domain=domain,
        periodic=False,
    )


def _dihedral_pair() -> CoordinatePair:
    return CoordinatePair(
        name="dihedral",
        x_col="carboxyl_dihedral",
        y_col="ester_dihedral",
        x_label="Carboxyl dihedral (°)",
        y_label="Ester dihedral (°)",
        title="Dihedral pair",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    carboxyl_plane: list[float] | None = None,
    ester_plane: list[float] | None = None,
    carboxyl_dihedral: list[float] | None = None,
    ester_dihedral: list[float] | None = None,
    trajectory_id: list[str] | None = None,
    bead_id: list[str] | None = None,
) -> pd.DataFrame:
    n = len(carboxyl_plane or ester_plane or [])
    return pd.DataFrame(
        {
            "carboxyl_plane":    carboxyl_plane    or [0.0] * n,
            "ester_plane":       ester_plane       or [0.0] * n,
            "carboxyl_dihedral": carboxyl_dihedral or [0.0] * n,
            "ester_dihedral":    ester_dihedral    or [0.0] * n,
            "trajectory_id":     trajectory_id     or ["traj"] * n,
            "bead_id":           bead_id           or ["00"] * n,
        }
    )


def _two_cluster_df(n_per_cluster: int = 30, seed: int = 0) -> pd.DataFrame:
    """Two tight Gaussian blobs far apart in carboxyl/ester plane space."""
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=[20.0, 20.0], scale=1.0, size=(n_per_cluster, 2))
    cluster_b = rng.normal(loc=[160.0, 160.0], scale=1.0, size=(n_per_cluster, 2))
    pts = np.vstack([cluster_a, cluster_b])
    return pd.DataFrame(
        {
            "carboxyl_plane":    pts[:, 0],
            "ester_plane":       pts[:, 1],
            "carboxyl_dihedral": 0.0,
            "ester_dihedral":    0.0,
            "trajectory_id":     "traj",
            "bead_id":           "00",
        }
    )


_GOOD_PARAMS = {"eps": 5.0, "min_samples": 3}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_assign_states_null_eps_raises():
    df = _two_cluster_df()
    with pytest.raises(ValueError, match="eps is null"):
        assign_conformer_states(df, _plane_pair(), "dbscan", {"eps": None, "min_samples": 3})


def test_assign_states_null_min_samples_raises():
    df = _two_cluster_df()
    with pytest.raises(ValueError, match="min_samples is null"):
        assign_conformer_states(df, _plane_pair(), "dbscan", {"eps": 5.0, "min_samples": None})


def test_assign_states_unknown_scheme_raises():
    df = _two_cluster_df()
    with pytest.raises(ValueError, match="Unsupported clustering scheme"):
        assign_conformer_states(df, _plane_pair(), "kmeans", _GOOD_PARAMS)


# ---------------------------------------------------------------------------
# Correct clustering behaviour
# ---------------------------------------------------------------------------


def test_assign_states_two_clear_clusters():
    df = _two_cluster_df()
    result = assign_conformer_states(df, _plane_pair(), "dbscan", _GOOD_PARAMS)
    labels = result["state_plane"].dropna()
    valid = labels[labels != "noise"]
    assert valid.nunique() == 2, f"Expected 2 clusters, got {valid.unique()}"


def test_assign_states_noise_labeled_noise():
    """Isolated single point well away from any cluster → 'noise'."""
    df = _two_cluster_df(n_per_cluster=30)
    # Add an isolated outlier
    outlier = pd.DataFrame(
        {
            "carboxyl_plane": [90.0],
            "ester_plane": [90.0],
            "carboxyl_dihedral": [0.0],
            "ester_dihedral": [0.0],
            "trajectory_id": "traj",
            "bead_id": "00",
        }
    )
    df = pd.concat([df, outlier], ignore_index=True)
    result = assign_conformer_states(df, _plane_pair(), "dbscan", _GOOD_PARAMS)
    assert result["state_plane"].iloc[-1] == "noise"


def test_assign_states_state_plane_column_populated():
    df = _two_cluster_df()
    result = assign_conformer_states(df, _plane_pair(), "dbscan", _GOOD_PARAMS)
    assert "state_plane" in result.columns
    assert result["state_plane"].notna().all()


def test_assign_states_dihedral_pair():
    """'dihedral' pair writes state_dihedral, not state_plane."""
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=[-150.0, -150.0], scale=2.0, size=(30, 2))
    cluster_b = rng.normal(loc=[150.0, 150.0], scale=2.0, size=(30, 2))
    pts = np.vstack([cluster_a, cluster_b])
    df = pd.DataFrame(
        {
            "carboxyl_plane": 0.0,
            "ester_plane": 0.0,
            "carboxyl_dihedral": pts[:, 0],
            "ester_dihedral": pts[:, 1],
            "trajectory_id": "traj",
            "bead_id": "00",
        }
    )
    result = assign_conformer_states(df, _dihedral_pair(), "dbscan", _GOOD_PARAMS)
    assert "state_dihedral" in result.columns
    assert "state_plane" not in result.columns or result["state_plane"].isna().all()
    valid = result["state_dihedral"].dropna()
    valid_nonnoise = valid[valid != "noise"]
    assert valid_nonnoise.nunique() == 2


def test_assign_states_groupby_trajectory():
    """Clustering per trajectory_id produces independent labels."""
    rng = np.random.default_rng(7)
    # traj_A: two clusters
    a1 = rng.normal([10.0, 10.0], 0.5, (20, 2))
    a2 = rng.normal([170.0, 170.0], 0.5, (20, 2))
    pts_a = np.vstack([a1, a2])
    # traj_B: two clusters in a different region
    b1 = rng.normal([30.0, 30.0], 0.5, (20, 2))
    b2 = rng.normal([150.0, 150.0], 0.5, (20, 2))
    pts_b = np.vstack([b1, b2])

    df = pd.DataFrame(
        {
            "carboxyl_plane": np.concatenate([pts_a[:, 0], pts_b[:, 0]]),
            "ester_plane":    np.concatenate([pts_a[:, 1], pts_b[:, 1]]),
            "carboxyl_dihedral": 0.0,
            "ester_dihedral":    0.0,
            "trajectory_id": ["A"] * 40 + ["B"] * 40,
            "bead_id": "00",
        }
    )
    result = assign_conformer_states(
        df, _plane_pair(), "dbscan", _GOOD_PARAMS, groupby=["trajectory_id"]
    )
    # Each group should have exactly 2 non-noise clusters
    for traj, group in result.groupby("trajectory_id"):
        valid = group["state_plane"]
        valid_nonnoise = valid[valid != "noise"]
        assert valid_nonnoise.nunique() == 2, f"traj {traj}: {valid_nonnoise.unique()}"


def test_assign_states_returns_copy_not_mutating_input():
    df = _two_cluster_df()
    original_cols = set(df.columns)
    _ = assign_conformer_states(df, _plane_pair(), "dbscan", _GOOD_PARAMS)
    # Original df must not have gained state_plane column
    assert set(df.columns) == original_cols


def test_assign_states_result_dtype_is_string():
    df = _two_cluster_df()
    result = assign_conformer_states(df, _plane_pair(), "dbscan", _GOOD_PARAMS)
    assert str(result["state_plane"].dtype) == "string"


def test_resolve_pair_params_prefers_pair_config():
    cfg = {
        "algorithm": "dbscan",
        "groupby": ["trajectory_id"],
        "default": {"eps": 3.0, "min_samples": 5},
        "plane": {"eps": 8.0, "min_samples": 100},
        "dihedral": {"eps": 7.0, "min_samples": 100},
    }
    scheme, params, groupby = _resolve_pair_params(cfg, "plane")
    assert scheme == "dbscan"
    assert params["eps"] == 8.0
    assert params["min_samples"] == 100
    assert groupby == ["trajectory_id"]


def test_resolve_pair_params_falls_back_to_default():
    cfg = {
        "algorithm": "dbscan",
        "groupby": ["trajectory_id"],
        "default": {"eps": 6.0, "min_samples": 75},
    }
    scheme, params, groupby = _resolve_pair_params(cfg, "dihedral")
    assert scheme == "dbscan"
    assert params["eps"] == 6.0
    assert params["min_samples"] == 75
    assert groupby == ["trajectory_id"]


def _full_config_for_states(plane_eps=5.0, dihedral_eps=10.0) -> dict:
    """Build a minimal full config that assign_conformer_states_from_config accepts."""
    return {
        "dof": [
            {"name": "carboxyl_plane", "type": "angle", "atoms": [0, 1, 2],
             "domain": [0, 180], "enabled": True},
            {"name": "ester_plane", "type": "angle", "atoms": [3, 4, 5],
             "domain": [0, 180], "enabled": True},
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "plane", "x": "carboxyl_plane", "y": "ester_plane"},
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
        "clustering": {
            "algorithm": "dbscan",
            "groupby": ["trajectory_id"],
            "plane": {"eps": plane_eps, "min_samples": 3},
            "dihedral": {"eps": dihedral_eps, "min_samples": 3},
        },
    }


def test_assign_states_from_config_populates_both_state_columns():
    df = _two_cluster_df()
    df["carboxyl_dihedral"] = np.r_[
        np.linspace(-150.0, -140.0, len(df) // 2),
        np.linspace(140.0, 150.0, len(df) - len(df) // 2),
    ]
    df["ester_dihedral"] = df["carboxyl_dihedral"]

    cfg = _full_config_for_states()
    result = assign_conformer_states_from_config(df, cfg)
    assert "state_plane" in result.columns
    assert "state_dihedral" in result.columns
    assert result["state_plane"].notna().all()
    assert result["state_dihedral"].notna().all()


def test_assign_states_from_config_groups_by_trajectory_not_bead():
    df = pd.DataFrame(
        {
            "carboxyl_plane": [10.0, 10.2, 10.1, 10.3],
            "ester_plane": [20.0, 20.1, 20.2, 20.1],
            "carboxyl_dihedral": [30.0, 30.2, 30.1, 30.3],
            "ester_dihedral": [40.0, 40.2, 40.1, 40.3],
            "trajectory_id": ["traj"] * 4,
            "bead_id": ["00", "00", "01", "01"],
        }
    )
    cfg = _full_config_for_states(plane_eps=1.0, dihedral_eps=1.0)
    result = assign_conformer_states_from_config(df, cfg)
    assert set(result["state_plane"]) == {"0"}
    assert set(result["state_dihedral"]) == {"0"}


def test_assign_states_from_config_groupby_override_clusters_per_bead():
    df = pd.DataFrame(
        {
            "carboxyl_plane": [10.0, 10.2, 150.0, 150.2],
            "ester_plane": [20.0, 20.2, 160.0, 160.2],
            "carboxyl_dihedral": [30.0, 30.2, -130.0, -130.2],
            "ester_dihedral": [40.0, 40.2, 140.0, 140.2],
            "trajectory_id": ["traj"] * 4,
            "bead_id": ["00", "00", "01", "01"],
        }
    )
    cfg = _full_config_for_states(plane_eps=1.0, dihedral_eps=1.0)
    # Reduce min_samples to 2 so 2-point groups form clusters
    cfg["clustering"]["plane"]["min_samples"] = 2
    cfg["clustering"]["dihedral"]["min_samples"] = 2
    result = assign_conformer_states_from_config(
        df,
        cfg,
        groupby_override=["trajectory_id", "bead_id"],
    )
    assert set(result.loc[result["bead_id"] == "00", "state_plane"]) == {"0"}
    assert set(result.loc[result["bead_id"] == "01", "state_plane"]) == {"0"}
    assert set(result.loc[result["bead_id"] == "00", "state_dihedral"]) == {"0"}
    assert set(result.loc[result["bead_id"] == "01", "state_dihedral"]) == {"0"}


# ---------------------------------------------------------------------------
# Grid clustering
# ---------------------------------------------------------------------------


def _grid_two_blob_df(n_per_cluster: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two tight Gaussian blobs well-separated in plane angle space."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[20.0, 20.0], scale=0.5, size=(n_per_cluster, 2))
    b = rng.normal(loc=[160.0, 160.0], scale=0.5, size=(n_per_cluster, 2))
    pts = np.vstack([a, b])
    return pd.DataFrame(
        {
            "carboxyl_plane":    pts[:, 0],
            "ester_plane":       pts[:, 1],
            "carboxyl_dihedral": 0.0,
            "ester_dihedral":    0.0,
            "trajectory_id":     "traj",
            "bead_id":           "00",
        }
    )


_GRID_PARAMS = {"bin_size": 2.0, "min_count": 5}


def _signed_seam_df(x_values: list[float], y_values: list[float]) -> pd.DataFrame:
    n = len(x_values)
    return pd.DataFrame(
        {
            "carboxyl_plane": x_values,
            "ester_plane": y_values,
            "carboxyl_dihedral": [0.0] * n,
            "ester_dihedral": [0.0] * n,
            "trajectory_id": ["traj"] * n,
            "bead_id": ["00"] * n,
        }
    )


def test_grid_two_clear_clusters():
    df = _grid_two_blob_df()
    result = assign_conformer_states(df, _plane_pair(), "grid", _GRID_PARAMS)
    valid = result["state_plane"].dropna()
    valid_nonnoise = valid[valid != "noise"]
    assert valid_nonnoise.nunique() == 2, f"Expected 2 clusters, got {valid_nonnoise.unique()}"


def test_grid_noise_below_min_count():
    """Bins with fewer than min_count frames should be labeled 'noise'."""
    rng = np.random.default_rng(42)
    # Dense cluster
    dense = rng.normal(loc=[20.0, 20.0], scale=0.3, size=(200, 2))
    # A single isolated point far from the cluster
    isolated = np.array([[90.0, 90.0]])
    pts = np.vstack([dense, isolated])
    df = pd.DataFrame(
        {
            "carboxyl_plane":    pts[:, 0],
            "ester_plane":       pts[:, 1],
            "carboxyl_dihedral": 0.0,
            "ester_dihedral":    0.0,
            "trajectory_id":     "traj",
            "bead_id":           "00",
        }
    )
    result = assign_conformer_states(df, _plane_pair(), "grid", {"bin_size": 2.0, "min_count": 10})
    assert result["state_plane"].iloc[-1] == "noise"


def test_grid_dihedral_pair():
    """Grid scheme on dihedral pair writes state_dihedral, not state_plane."""
    rng = np.random.default_rng(3)
    a = rng.normal(loc=[-150.0, -150.0], scale=0.5, size=(200, 2))
    b = rng.normal(loc=[150.0, 150.0], scale=0.5, size=(200, 2))
    pts = np.vstack([a, b])
    df = pd.DataFrame(
        {
            "carboxyl_plane":    0.0,
            "ester_plane":       0.0,
            "carboxyl_dihedral": pts[:, 0],
            "ester_dihedral":    pts[:, 1],
            "trajectory_id":     "traj",
            "bead_id":           "00",
        }
    )
    result = assign_conformer_states(df, _dihedral_pair(), "grid", {"bin_size": 5.0, "min_count": 5})
    assert "state_dihedral" in result.columns
    valid = result["state_dihedral"].dropna()
    valid_nonnoise = valid[valid != "noise"]
    assert valid_nonnoise.nunique() == 2


def test_grid_signed_plane_pair_wraps_x_seam_into_one_cluster():
    df = _signed_seam_df(
        [-179.4, -179.2, 179.2, 179.4],
        [10.0, 10.2, 10.1, 10.3],
    )
    # Use signed plane pair so the domain is [-180, 180]
    # Grid does NOT apply periodic wrapping for plane pairs, so this will remain split
    # unless we use a dihedral pair type
    # For a signed plane pair with periodic=False, seam stays split:
    signed_pair = CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="x",
        y_label="y",
        title="Signed plane",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,  # enable wrapping for this test
    )
    result = assign_conformer_states(
        df, signed_pair, "grid", {"bin_size": 2.0, "min_count": 2},
    )
    assert set(result["state_plane"]) == {"0"}


def test_grid_signed_plane_wraps_y_seam_into_one_cluster():
    df = _signed_seam_df(
        [15.0, 15.2, 15.1, 15.3],
        [-179.4, -179.2, 179.2, 179.4],
    )
    signed_pair = CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="x",
        y_label="y",
        title="Signed plane",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )
    result = assign_conformer_states(
        df, signed_pair, "grid", {"bin_size": 2.0, "min_count": 2},
    )
    assert set(result["state_plane"]) == {"0"}


def test_dbscan_signed_plane_keeps_seam_clusters_separate():
    df = _signed_seam_df(
        [-179.4, -179.2, 179.2, 179.4],
        [10.0, 10.2, 10.1, 10.3],
    )
    # Non-periodic plane pair: seam points should remain split
    result = assign_conformer_states(
        df,
        _plane_pair(signed=True),
        "dbscan",
        {"eps": 1.0, "min_samples": 2, "periodic": False},
    )
    valid = result["state_plane"][result["state_plane"] != "noise"]
    assert valid.nunique() == 2


def test_grid_null_bin_size_raises():
    df = _grid_two_blob_df()
    with pytest.raises(ValueError, match="bin_size is null"):
        assign_conformer_states(df, _plane_pair(), "grid", {"bin_size": None, "min_count": 5})


def test_grid_null_min_count_raises():
    df = _grid_two_blob_df()
    with pytest.raises(ValueError, match="min_count is null"):
        assign_conformer_states(df, _plane_pair(), "grid", {"bin_size": 2.0, "min_count": None})


def test_resolve_params_grid_scheme():
    cfg = {
        "algorithm": "grid",
        "groupby": ["trajectory_id"],
        "plane": {"bin_size": 2.0, "min_count": 50},
        "dihedral": {"bin_size": 5.0, "min_count": 50},
    }
    scheme, params, groupby = _resolve_pair_params(cfg, "plane")
    assert scheme == "grid"
    assert params == {"bin_size": 2.0, "min_count": 50}
    assert groupby == ["trajectory_id"]

    scheme2, params2, _ = _resolve_pair_params(cfg, "dihedral")
    assert scheme2 == "grid"
    assert params2 == {"bin_size": 5.0, "min_count": 50}


def _full_config_grid(plane_bin=2.0, dihedral_bin=5.0) -> dict:
    return {
        "dof": [
            {"name": "carboxyl_plane", "type": "angle", "atoms": [0, 1, 2],
             "domain": [0, 180], "enabled": True},
            {"name": "ester_plane", "type": "angle", "atoms": [3, 4, 5],
             "domain": [0, 180], "enabled": True},
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "plane", "x": "carboxyl_plane", "y": "ester_plane"},
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
        "clustering": {
            "algorithm": "grid",
            "groupby": ["trajectory_id"],
            "plane": {"bin_size": plane_bin, "min_count": 5},
            "dihedral": {"bin_size": dihedral_bin, "min_count": 5},
        },
    }


def test_assign_from_config_grid_algorithm():
    df = _grid_two_blob_df()
    df["carboxyl_dihedral"] = np.r_[
        np.linspace(-150.0, -145.0, len(df) // 2),
        np.linspace(145.0, 150.0, len(df) - len(df) // 2),
    ]
    df["ester_dihedral"] = df["carboxyl_dihedral"]
    cfg = _full_config_grid()
    result = assign_conformer_states_from_config(df, cfg)
    assert "state_plane" in result.columns
    assert "state_dihedral" in result.columns
    assert result["state_plane"].notna().all()
    assert result["state_dihedral"].notna().all()


# ---------------------------------------------------------------------------
# Periodic boundary behaviour — DBSCAN
# ---------------------------------------------------------------------------


def _dihedral_seam_df(x_values: list[float], y_values: list[float]) -> pd.DataFrame:
    """Minimal DataFrame with dihedral columns at specified values."""
    n = len(x_values)
    return pd.DataFrame(
        {
            "carboxyl_plane":    [0.0] * n,
            "ester_plane":       [0.0] * n,
            "carboxyl_dihedral": x_values,
            "ester_dihedral":    y_values,
            "trajectory_id":     ["traj"] * n,
            "bead_id":           ["00"] * n,
        }
    )


def test_dbscan_dihedral_periodic_merges_seam_into_one_cluster():
    """Angles near -180° and +180° should cluster together with periodic=True."""
    rng = np.random.default_rng(42)
    # Two tight groups straddling the ±180° seam
    neg = rng.normal(loc=-179.0, scale=0.3, size=20).tolist()
    pos = rng.normal(loc=+179.0, scale=0.3, size=20).tolist()
    df = _dihedral_seam_df(neg + pos, [0.0] * 40)
    result = assign_conformer_states(
        df,
        _dihedral_pair(),
        "dbscan",
        {"eps": 5.0, "min_samples": 3, "periodic": True},
    )
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    assert valid.nunique() == 1, f"Expected 1 cluster, got {valid.unique()}"


def test_dbscan_dihedral_non_periodic_keeps_seam_split():
    """With periodic=False, the same seam data should NOT be merged."""
    rng = np.random.default_rng(42)
    neg = rng.normal(loc=-179.0, scale=0.3, size=20).tolist()
    pos = rng.normal(loc=+179.0, scale=0.3, size=20).tolist()
    df = _dihedral_seam_df(neg + pos, [0.0] * 40)
    result = assign_conformer_states(
        df,
        _dihedral_pair(),
        "dbscan",
        {"eps": 5.0, "min_samples": 3, "periodic": False},
    )
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    # Without periodic embedding the two groups appear 358° apart → 2 clusters
    assert valid.nunique() == 2, f"Expected 2 clusters (seam split), got {valid.unique()}"


def test_dbscan_dihedral_2d_cross_seam_merge():
    """One axis crosses the ±180° seam; the perpendicular axis stays interior."""
    rng = np.random.default_rng(7)
    neg = rng.normal(loc=-178.0, scale=0.3, size=20).tolist()
    pos = rng.normal(loc=+178.0, scale=0.3, size=20).tolist()
    y = rng.normal(loc=30.0, scale=0.3, size=40).tolist()
    df = _dihedral_seam_df(neg + pos, y)
    result = assign_conformer_states(
        df,
        _dihedral_pair(),
        "dbscan",
        {"eps": 5.0, "min_samples": 3, "periodic": True},
    )
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    assert valid.nunique() == 1, f"Expected 1 cluster across x-seam, got {valid.unique()}"


def test_dbscan_dihedral_interior_clusters_unaffected():
    """Clusters far from the ±180° boundary should still be found correctly."""
    rng = np.random.default_rng(11)
    a = rng.normal(loc=[-30.0, -30.0], scale=0.5, size=(30, 2))
    b = rng.normal(loc=[120.0, 120.0], scale=0.5, size=(30, 2))
    pts = np.vstack([a, b])
    df = _dihedral_seam_df(pts[:, 0].tolist(), pts[:, 1].tolist())
    result = assign_conformer_states(
        df,
        _dihedral_pair(),
        "dbscan",
        {"eps": 5.0, "min_samples": 3, "periodic": True},
    )
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    assert valid.nunique() == 2, f"Expected 2 interior clusters, got {valid.unique()}"


def test_dbscan_plane_seam_still_unmodified():
    """Plane DBSCAN with periodic=False must NOT merge seam clusters."""
    df = _signed_seam_df(
        [-179.4, -179.2, 179.2, 179.4],
        [10.0, 10.2, 10.1, 10.3],
    )
    result = assign_conformer_states(
        df,
        _plane_pair(signed=True),
        "dbscan",
        {"eps": 1.0, "min_samples": 2, "periodic": False},
    )
    valid = result["state_plane"][result["state_plane"] != "noise"]
    # Plane DBSCAN does not embed periodically, so the two groups remain split.
    assert valid.nunique() == 2


# ---------------------------------------------------------------------------
# Periodic boundary behaviour — Grid
# ---------------------------------------------------------------------------


def test_grid_dihedral_wraps_seam_into_one_cluster():
    """Grid clustering on dihedrals should merge bins across the ±180° seam."""
    rng = np.random.default_rng(13)
    neg = rng.normal(loc=-179.0, scale=0.4, size=30).tolist()
    pos = rng.normal(loc=+179.0, scale=0.4, size=30).tolist()
    df = _dihedral_seam_df(neg + pos, [0.0] * 60)
    result = assign_conformer_states(
        df,
        _dihedral_pair(),
        "grid",
        {"bin_size": 2.0, "min_count": 5},
    )
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    assert valid.nunique() == 1, f"Expected 1 cluster (seam merged), got {valid.unique()}"


def test_grid_dihedral_interior_clusters_unaffected():
    """Grid dihedral wrapping must not merge interior clusters incorrectly."""
    rng = np.random.default_rng(17)
    a = rng.normal(loc=[-30.0, -30.0], scale=0.4, size=(100, 2))
    b = rng.normal(loc=[120.0, 120.0], scale=0.4, size=(100, 2))
    pts = np.vstack([a, b])
    df = _dihedral_seam_df(pts[:, 0].tolist(), pts[:, 1].tolist())
    result = assign_conformer_states(
        df,
        _dihedral_pair(),
        "grid",
        {"bin_size": 2.0, "min_count": 5},
    )
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    assert valid.nunique() == 2, f"Expected 2 interior clusters, got {valid.unique()}"


def test_assign_from_full_config_uses_configured_pair():
    rng = np.random.default_rng(22)
    cluster_a = rng.normal(loc=[-140.0, -145.0], scale=1.0, size=(20, 2))
    cluster_b = rng.normal(loc=[140.0, 145.0], scale=1.0, size=(20, 2))
    pts = np.vstack([cluster_a, cluster_b])
    df = pd.DataFrame(
        {
            "carboxyl_plane": np.linspace(0.0, 10.0, len(pts)),
            "ester_plane": np.linspace(5.0, 15.0, len(pts)),
            "igor1_dihedral": pts[:, 0],
            "igor2_dihedral": pts[:, 1],
            "trajectory_id": "traj",
            "bead_id": "00",
        }
    )
    cfg = {
        "dof": [
            {"name": "carboxyl_plane", "type": "angle", "atoms": [0, 1, 2],
             "domain": [0, 180], "enabled": True},
            {"name": "ester_plane", "type": "angle", "atoms": [3, 4, 5],
             "domain": [0, 180], "enabled": True},
            {"name": "igor1_dihedral", "type": "dihedral", "atoms": [6, 12, 11, 8],
             "domain": [-180, 180], "enabled": True},
            {"name": "igor2_dihedral", "type": "dihedral", "atoms": [6, 12, 11, 4],
             "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "plane", "x": "carboxyl_plane", "y": "ester_plane"},
            {"name": "dihedral", "x": "igor1_dihedral", "y": "igor2_dihedral"},
        ],
        "clustering": {
            "algorithm": "dbscan",
            "groupby": None,
            "plane": {"eps": 30, "min_samples": 2},
            "dihedral": {"eps": 8, "min_samples": 2},
        },
    }
    result = assign_conformer_states_from_config(df, cfg)
    valid = result["state_dihedral"][result["state_dihedral"] != "noise"]
    assert valid.nunique() == 2


# ---------------------------------------------------------------------------
# Regression: non-PIMD data (bead_id = NA) must still be clustered
# ---------------------------------------------------------------------------


def test_clustering_with_na_bead_id():
    """groupby dropna=False: frames with NA bead_id must be clustered, not silently dropped."""
    rng = np.random.default_rng(42)
    n = 300
    coords_a = rng.normal([30.0, 30.0], 2.0, size=(n, 2))
    coords_b = rng.normal([150.0, 150.0], 2.0, size=(n, 2))
    coords = np.vstack([coords_a, coords_b])

    df = pd.DataFrame(
        {
            "trajectory_id": "traj0",
            "bead_id": pd.array([pd.NA] * (2 * n), dtype="string"),
            "carboxyl_plane": coords[:, 0].astype("float32"),
            "ester_plane": coords[:, 1].astype("float32"),
            "carboxyl_dihedral": coords[:, 0].astype("float32"),
            "ester_dihedral": coords[:, 1].astype("float32"),
            "state_plane": pd.array([pd.NA] * (2 * n), dtype="string"),
            "state_dihedral": pd.array([pd.NA] * (2 * n), dtype="string"),
        }
    )

    cfg = {
        "dof": [
            {"name": "carboxyl_plane", "type": "angle", "atoms": [0, 1, 2],
             "domain": [0, 180], "enabled": True},
            {"name": "ester_plane", "type": "angle", "atoms": [3, 4, 5],
             "domain": [0, 180], "enabled": True},
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "domain": [-180, 180], "enabled": True},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "domain": [-180, 180], "enabled": True},
        ],
        "coordinate_pairs": [
            {"name": "plane", "x": "carboxyl_plane", "y": "ester_plane"},
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
        "clustering": {
            "algorithm": "grid",
            "groupby": ["trajectory_id", "bead_id"],
            "plane": {"bin_size": 5.0, "min_count": 5},
            "dihedral": {"bin_size": 5.0, "min_count": 5},
        },
    }

    result = assign_conformer_states_from_config(df, cfg)

    for col in ("state_plane", "state_dihedral"):
        non_noise = result[col][result[col].notna() & (result[col] != "noise")]
        assert len(non_noise) > 0, f"{col}: no frames assigned — dropna bug likely"
        assert non_noise.nunique() >= 2, f"{col}: expected ≥2 clusters"


# ---------------------------------------------------------------------------
# build_bin_state_overlay / resolve_state_groupby
# ---------------------------------------------------------------------------


def _overlay_pair(periodic: bool = False) -> CoordinatePair:
    lo, hi = (-180.0, 180.0) if periodic else (0.0, 3.0)
    return CoordinatePair(
        name="p", x_col="x", y_col="y", x_label="x", y_label="y", title="t",
        x_domain=(lo, hi), y_domain=(lo, hi), periodic=periodic,
    )


def _overlay_df(rows: list[tuple]) -> pd.DataFrame:
    """rows = (x, y, state, bead_id)."""
    return pd.DataFrame({
        "x": [r[0] for r in rows],
        "y": [r[1] for r in rows],
        "state_p": pd.array([r[2] for r in rows], dtype="string"),
        "bead_id": [r[3] for r in rows],
    })


_EDGES = np.linspace(0.0, 3.0, 4)  # 3 bins per axis: flat index = yi * 3 + xi


def _bin_states(group: dict, labels: list[str]) -> dict[int, str]:
    return {b: labels[s] for b, s in zip(group["bins"], group["states"])}


def test_overlay_none_without_state_column():
    df = _overlay_df([(0.5, 0.5, "0", "00")]).drop(columns="state_p")
    assert build_bin_state_overlay(df, _overlay_pair(), _EDGES, _EDGES) is None


def test_overlay_majority_noise_and_natural_ties():
    df = _overlay_df([
        (0.5, 0.5, "1", "00"), (0.5, 0.5, "1", "00"), (0.5, 0.5, "noise", "00"),  # bin 0 → "1"
        (1.5, 0.5, "noise", "00"), (1.5, 0.5, "noise", "00"), (1.5, 0.5, "2", "00"),  # bin 1 → none
        (2.5, 0.5, "10", "00"), (2.5, 0.5, "2", "00"),  # bin 2: tie → "2" before "10"
        (0.5, 1.5, "2", "00"), (0.5, 1.5, None, "00"),  # bin 3: tie with NA → real state
        (0.5, 2.5, None, "00"),  # bin 6: NA only → none
    ])
    overlay = build_bin_state_overlay(df, _overlay_pair(), _EDGES, _EDGES)
    assert overlay["labels"] == ["1", "2", "10"]
    assert overlay["groupby"] == []
    (group,) = overlay["groups"]
    assert group["name"] == "all frames"
    assert _bin_states(group, overlay["labels"]) == {0: "1", 2: "2", 3: "2"}


def test_overlay_keeps_groups_separate():
    # Label "0" is a different region in each bead: never pooled.
    df = _overlay_df([
        (0.5, 0.5, "0", "00"), (0.5, 0.5, "0", "00"), (2.5, 2.5, "1", "00"),
        (2.5, 2.5, "0", "01"), (0.5, 0.5, "1", "01"), (0.5, 0.5, "1", "01"), (0.5, 0.5, "1", "01"),
    ])
    overlay = build_bin_state_overlay(df, _overlay_pair(), _EDGES, _EDGES, groupby=["bead_id"])
    labels = overlay["labels"]
    g00, g01 = overlay["groups"]
    assert (g00["name"], g00["keys"]) == ("bead 00", {"bead_id": "00"})
    assert (g01["name"], g01["keys"]) == ("bead 01", {"bead_id": "01"})
    assert _bin_states(g00, labels) == {0: "0", 8: "1"}
    assert _bin_states(g01, labels) == {0: "1", 8: "0"}


def test_overlay_bins_match_histogram2d():
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.5, 3.5, 500)
    y = rng.uniform(-0.5, 3.5, 500)
    x[:5] = 3.0  # right edge belongs to the last bin
    df = _overlay_df([(a, b, "0", "00") for a, b in zip(x, y)])
    overlay = build_bin_state_overlay(df, _overlay_pair(), _EDGES, _EDGES)
    H, _, _ = np.histogram2d(x, y, bins=[_EDGES, _EDGES])
    occupied = {int(yi * 3 + xi) for xi, yi in zip(*np.nonzero(H))}
    assert set(overlay["groups"][0]["bins"]) == occupied


def test_overlay_centres_are_population_weighted():
    df = _overlay_df([(0.5, 0.5, "0", "00"), (0.5, 0.5, "0", "00"), (2.0, 2.0, "0", "00")])
    overlay = build_bin_state_overlay(df, _overlay_pair(), _EDGES, _EDGES)
    (centre,) = overlay["groups"][0]["centres"]
    assert centre == {"state": 0, "x": pytest.approx(1.0), "y": pytest.approx(1.0), "frames": 3}


def test_overlay_centres_wrap_on_periodic_axes():
    edges = np.linspace(-180.0, 180.0, 13)
    df = _overlay_df([(170.0, 0.0, "0", "00"), (-170.0, 0.0, "0", "00")])
    overlay = build_bin_state_overlay(df, _overlay_pair(periodic=True), edges, edges)
    (centre,) = overlay["groups"][0]["centres"]
    assert abs(centre["x"]) == pytest.approx(180.0)


def test_overlay_missing_groupby_column_raises():
    df = _overlay_df([(0.5, 0.5, "0", "00")])
    with pytest.raises(ValueError, match="trajectory_id"):
        build_bin_state_overlay(df, _overlay_pair(), _EDGES, _EDGES, groupby=["trajectory_id"])


def test_resolve_state_groupby_reads_clustering_section():
    assert resolve_state_groupby({"clustering": {"groupby": ["trajectory_id", "bead_id"]}}) == [
        "trajectory_id", "bead_id",
    ]
    assert resolve_state_groupby({"groupby": ["bead_id"]}) == ["bead_id"]
    assert resolve_state_groupby({"clustering": {"groupby": []}}) is None
    assert resolve_state_groupby({}) is None
