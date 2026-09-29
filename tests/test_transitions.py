"""Tests for confana/transitions.py (Phase 9)."""

from __future__ import annotations



import numpy as np
import pandas as pd
import pytest

from confana.models import CoordinatePair
from confana.transitions import (
    analyze_grouped_transitions,
    aggregate_pimd_transitions,
    compute_activation_free_energy_barriers,
    compute_transition_counts,
    compute_transition_probabilities,
    compute_transition_rates,
    load_or_build_trajectory_transitions,
)

_K_B_EV_PER_K = 8.617333262145e-5
_K_B_J_PER_K = 1.380649e-23
_PLANCK_J_S = 6.62607015e-34


# ---------------------------------------------------------------------------
# CoordinatePair fixtures for grouped transition tests
# ---------------------------------------------------------------------------


def _plane_pair() -> CoordinatePair:
    return CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="Carboxyl plane (°)",
        y_label="Ester plane (°)",
        title="Plane pair",
        x_domain=(0.0, 180.0),
        y_domain=(0.0, 180.0),
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


def _series(*labels: str) -> pd.Series:
    return pd.Series(list(labels), dtype="string")


def _counts_from_labels(
    *labels: str,
    lag: int = 1,
    skip_noise_intermediates: bool = False,
) -> pd.DataFrame:
    # Default skip_noise_intermediates=False here so that existing tests that
    # explicitly test pair-level noise exclusion continue to pass unchanged.
    return compute_transition_counts(
        _series(*labels),
        lag=lag,
        skip_noise_intermediates=skip_noise_intermediates,
    )


def _transition_df(
    states: list[str | None],
    *,
    pair: CoordinatePair | None = None,
    trajectory_id: str = "traj",
    bead_id: str | None = "00",
    global_frame_index: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal coordinate-table slice for grouped transition tests."""
    if pair is None:
        pair = _plane_pair()
    n = len(states)
    if global_frame_index is None:
        global_frame_index = list(range(n))

    data = {
        "trajectory_id": [trajectory_id] * n,
        "global_frame_index": global_frame_index,
        pair.state_col: pd.Series(states, dtype="string"),
    }
    if bead_id is not None:
        data["bead_id"] = [bead_id] * n

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# compute_transition_counts
# ---------------------------------------------------------------------------


def test_compute_counts_simple_sequence():
    # 5 elements → 4 pairs: A→B, B→A, A→B, B→A  → A→B:2, B→A:2
    counts = _counts_from_labels("A", "B", "A", "B", "A")
    assert counts.loc["A", "B"] == 2
    assert counts.loc["B", "A"] == 2
    assert counts.loc["A", "A"] == 0
    assert counts.loc["B", "B"] == 0


def test_compute_counts_lag_2():
    # lag=2: pairs (0,2),(1,3),(2,4) = (A,A),(B,B),(A,A)
    counts = compute_transition_counts(_series("A", "B", "A", "B", "A"), lag=2)
    assert counts.loc["A", "A"] == 2
    assert counts.loc["B", "B"] == 1


def test_compute_counts_noise_excluded():
    counts = _counts_from_labels("A", "noise", "A", "B")
    # ("A","noise") and ("noise","A") are excluded; only ("A","B") counts
    assert "noise" not in counts.index
    assert "noise" not in counts.columns
    assert counts.loc["A", "B"] == 1


def test_compute_counts_na_excluded():
    states = pd.Series(["A", None, "A", "B"], dtype="string")
    counts = compute_transition_counts(states, lag=1)
    assert "noise" not in counts.index
    # Only valid pair is ("A","B")
    assert counts.loc["A", "B"] == 1


def test_compute_counts_empty_series():
    counts = compute_transition_counts(pd.Series([], dtype="string"))
    assert counts.empty


def test_compute_counts_single_element():
    counts = compute_transition_counts(_series("A"))
    assert counts.empty


def test_compute_counts_single_state():
    # All same state → only self-transitions
    counts = _counts_from_labels("X", "X", "X")
    assert list(counts.index) == ["X"]
    assert counts.loc["X", "X"] == 2


def test_compute_counts_all_noise():
    counts = _counts_from_labels("noise", "noise", "noise")
    assert counts.empty


def test_compute_counts_invalid_lag_raises():
    with pytest.raises(ValueError, match="lag must be >= 1"):
        compute_transition_counts(_series("A", "B"), lag=0)


def test_compute_counts_labels_sorted():
    counts = _counts_from_labels("2", "1", "0", "1")
    assert list(counts.index) == ["0", "1", "2"]
    assert list(counts.columns) == ["0", "1", "2"]


def test_compute_counts_symmetric_sequence():
    # A→B B→A A→B B→A: counts should be symmetric
    counts = _counts_from_labels("A", "B", "A", "B", "A")
    assert counts.loc["A", "B"] == counts.loc["B", "A"]


# ---------------------------------------------------------------------------
# compute_transition_probabilities
# ---------------------------------------------------------------------------


def test_compute_probabilities_row_sums_to_one():
    counts = _counts_from_labels("A", "B", "A", "B", "A")
    probs = compute_transition_probabilities(counts)
    for lbl in probs.index:
        row_sum = probs.loc[lbl].sum()
        if not np.isnan(row_sum):
            assert np.isclose(row_sum, 1.0, atol=1e-10), (
                f"Row '{lbl}' sums to {row_sum}"
            )


def test_compute_probabilities_known_result():
    # A→B: 3, A→C: 1  → P(A→B)=0.75, P(A→C)=0.25
    counts = pd.DataFrame(
        [[0, 3, 1], [0, 0, 0], [0, 0, 0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    probs = compute_transition_probabilities(counts)
    assert np.isclose(probs.loc["A", "B"], 0.75, atol=1e-10)
    assert np.isclose(probs.loc["A", "C"], 0.25, atol=1e-10)


def test_compute_probabilities_zero_row_is_nan():
    counts = pd.DataFrame(
        [[2, 1], [0, 0]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    probs = compute_transition_probabilities(counts)
    assert probs.loc["B"].isna().all()


def test_compute_probabilities_empty_input():
    probs = compute_transition_probabilities(pd.DataFrame())
    assert probs.empty


def test_compute_probabilities_preserves_labels():
    counts = _counts_from_labels("X", "Y", "X")
    probs = compute_transition_probabilities(counts)
    assert set(probs.index) == {"X", "Y"}
    assert set(probs.columns) == {"X", "Y"}


def test_compute_transition_rates_known_result():
    counts = pd.DataFrame(
        [[0, 3], [1, 1]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    rates = compute_transition_rates(counts, dt=0.5, lag=2)
    # lag * dt = 1.0, so rates equal probabilities in this case
    assert rates.loc["A", "B"] == pytest.approx(1.0, abs=1e-10)
    assert rates.loc["B", "A"] == pytest.approx(0.5, abs=1e-10)
    assert rates.loc["B", "B"] == pytest.approx(0.5, abs=1e-10)


def test_compute_activation_barriers_eyring_known_result():
    rates = pd.DataFrame(
        [[1.0, 2.0], [0.0, np.nan]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    barriers = compute_activation_free_energy_barriers(rates, temperature=300.0)
    prefactor = _K_B_J_PER_K * 300.0 / _PLANCK_J_S
    expected = -_K_B_EV_PER_K * 300.0 * np.log(2.0 / prefactor)

    assert barriers.loc["A", "B"] == pytest.approx(expected, abs=1e-12)
    assert np.isnan(barriers.loc["A", "A"])
    assert np.isnan(barriers.loc["B", "A"])
    assert np.isnan(barriers.loc["B", "B"])


def test_compute_activation_barriers_arrhenius_and_unit_conversion():
    rates = pd.DataFrame(
        [[np.nan, 1.0e6], [1.0e7, np.nan]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    barriers = compute_activation_free_energy_barriers(
        rates,
        temperature=300.0,
        model="arrhenius",
        attempt_frequency=1.0e12,
        energy_conv_factor=96.48533212,
    )
    expected_ev = -_K_B_EV_PER_K * 300.0 * np.log(1.0e6 / 1.0e12)
    assert barriers.loc["A", "B"] == pytest.approx(
        expected_ev * 96.48533212,
        abs=1e-10,
    )


def test_compute_activation_barriers_can_be_negative():
    rates = pd.DataFrame(
        [[np.nan, 1.0e9]],
        index=["A"],
        columns=["A", "B"],
    )
    barriers = compute_activation_free_energy_barriers(
        rates,
        temperature=300.0,
        model="arrhenius",
        attempt_frequency=1.0e6,
    )
    assert barriers.loc["A", "B"] < 0.0


def test_compute_activation_barriers_arrhenius_requires_attempt_frequency():
    rates = pd.DataFrame([[0.0, 1.0]], index=["A"], columns=["A", "B"])
    with pytest.raises(ValueError, match="attempt_frequency is required"):
        compute_activation_free_energy_barriers(
            rates,
            temperature=300.0,
            model="arrhenius",
        )


# ---------------------------------------------------------------------------
# aggregate_pimd_transitions
# ---------------------------------------------------------------------------


def test_aggregate_pimd_identical_beads():
    probs = pd.DataFrame(
        [[0.8, 0.2], [0.3, 0.7]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    avg = aggregate_pimd_transitions([probs, probs.copy()])
    assert avg.loc["A", "A"] == pytest.approx(0.8, abs=1e-10)
    assert avg.loc["B", "B"] == pytest.approx(0.7, abs=1e-10)


def test_aggregate_pimd_two_beads_average():
    p1 = pd.DataFrame([[0.6, 0.4], [0.2, 0.8]], index=["A", "B"], columns=["A", "B"])
    p2 = pd.DataFrame([[0.8, 0.2], [0.4, 0.6]], index=["A", "B"], columns=["A", "B"])
    avg = aggregate_pimd_transitions([p1, p2])
    assert avg.loc["A", "A"] == pytest.approx(0.7, abs=1e-10)
    assert avg.loc["A", "B"] == pytest.approx(0.3, abs=1e-10)
    assert avg.loc["B", "A"] == pytest.approx(0.3, abs=1e-10)
    assert avg.loc["B", "B"] == pytest.approx(0.7, abs=1e-10)


def test_aggregate_pimd_single_bead():
    probs = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"])
    avg = aggregate_pimd_transitions([probs])
    pd.testing.assert_frame_equal(avg, probs)


def test_aggregate_pimd_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        aggregate_pimd_transitions([])


def test_aggregate_pimd_nan_ignored():
    """NaN in one bead (state never visited) should not propagate to mean."""
    p1 = pd.DataFrame([[0.8, 0.2], [0.3, 0.7]], index=["A", "B"], columns=["A", "B"])
    p2 = pd.DataFrame([[float("nan"), float("nan")], [0.5, 0.5]],
                      index=["A", "B"], columns=["A", "B"])
    avg = aggregate_pimd_transitions([p1, p2])
    # Row A: only bead 1 has valid values → mean = bead 1 values
    assert avg.loc["A", "A"] == pytest.approx(0.8, abs=1e-10)
    # Row B: mean of both beads
    assert avg.loc["B", "A"] == pytest.approx(0.4, abs=1e-10)


def test_aggregate_pimd_aligns_union_of_labels():
    p1 = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["A", "B"])
    p2 = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "C"], columns=["A", "C"])
    avg = aggregate_pimd_transitions([p1, p2])
    assert list(avg.index) == ["A", "B", "C"]
    assert list(avg.columns) == ["A", "B", "C"]
    assert avg.loc["B", "B"] == pytest.approx(1.0, abs=1e-10)
    assert avg.loc["C", "C"] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# analyze_grouped_transitions
# ---------------------------------------------------------------------------


def test_analyze_grouped_transitions_sorts_by_global_frame_index():
    pair = _plane_pair()
    df = _transition_df(
        ["B", "A", "B"],
        pair=pair,
        global_frame_index=[2, 0, 1],
    )
    result = analyze_grouped_transitions(df, pair)
    counts = result["per_group_counts"][("traj", "00")]
    assert counts.loc["A", "B"] == 1
    assert counts.loc["B", "B"] == 1


def test_analyze_grouped_transitions_omits_rates_when_dt_missing():
    pair = _plane_pair()
    df = _transition_df(["A", "B", "A"], pair=pair)
    result = analyze_grouped_transitions(df, pair, dt=None)
    assert result["per_group_rates"] == {}
    assert result["averaged_rates"] == {}
    assert result["per_group_barriers"] == {}
    assert result["averaged_barriers"] == {}


def test_analyze_grouped_transitions_includes_rates_when_dt_present():
    pair = _plane_pair()
    df = _transition_df(["A", "B", "A"], pair=pair)
    result = analyze_grouped_transitions(df, pair, dt=0.5)
    rates = result["per_group_rates"][("traj", "00")]
    assert rates.loc["A", "B"] == pytest.approx(2.0, abs=1e-10)
    assert rates.loc["B", "A"] == pytest.approx(2.0, abs=1e-10)
    assert result["per_group_barriers"] == {}


def test_analyze_grouped_transitions_includes_barriers_when_temperature_present():
    pair = _plane_pair()
    df = _transition_df(["A", "B", "A"], pair=pair)
    result = analyze_grouped_transitions(
        df,
        pair,
        dt=0.5,
        temperature=300.0,
        barrier_model="eyring",
        energy_unit="eV",
    )
    barriers = result["per_group_barriers"][("traj", "00")]
    prefactor = _K_B_J_PER_K * 300.0 / _PLANCK_J_S
    expected = -_K_B_EV_PER_K * 300.0 * np.log(2.0 / prefactor)
    assert barriers.loc["A", "B"] == pytest.approx(expected, abs=1e-12)
    assert barriers.loc["B", "A"] == pytest.approx(expected, abs=1e-12)
    assert np.isnan(barriers.loc["A", "A"])
    assert result["barrier_energy_unit"] == "eV"


def test_analyze_grouped_transitions_excludes_noise_and_missing_states():
    pair = _plane_pair()
    df = _transition_df(["A", "noise", None, "A", "B"], pair=pair)
    result = analyze_grouped_transitions(df, pair)
    counts = result["per_group_counts"][("traj", "00")]
    assert "noise" not in counts.index
    assert counts.loc["A", "B"] == 1


def test_analyze_grouped_transitions_uses_trajectory_only_without_bead_id():
    pair = _plane_pair()
    df = _transition_df(["A", "B", "A"], pair=pair, bead_id=None)
    result = analyze_grouped_transitions(df, pair)
    assert result["groupby"] == ["trajectory_id"]
    assert ("traj",) in result["per_group_counts"]


def test_analyze_grouped_transitions_averages_beads_with_label_alignment():
    pair = _plane_pair()
    bead0 = _transition_df(["A", "B", "A"], pair=pair, bead_id="00")
    bead1 = _transition_df(["A", "C", "A"], pair=pair, bead_id="01", global_frame_index=[10, 11, 12])
    df = pd.concat([bead0, bead1], ignore_index=True)

    result = analyze_grouped_transitions(df, pair, dt=1.0)

    assert set(result["per_group_counts"].keys()) == {("traj", "00"), ("traj", "01")}
    avg_counts = result["averaged_counts"][("traj",)]
    avg_probs = result["averaged_probabilities"][("traj",)]
    avg_rates = result["averaged_rates"][("traj",)]

    assert list(avg_counts.index) == ["A", "B", "C"]
    assert avg_counts.loc["A", "B"] == pytest.approx(0.5, abs=1e-10)
    assert avg_counts.loc["A", "C"] == pytest.approx(0.5, abs=1e-10)
    assert avg_counts.loc["B", "A"] == pytest.approx(0.5, abs=1e-10)
    assert avg_counts.loc["C", "A"] == pytest.approx(0.5, abs=1e-10)

    assert avg_probs.loc["A", "B"] == pytest.approx(0.5, abs=1e-10)
    assert avg_probs.loc["A", "C"] == pytest.approx(0.5, abs=1e-10)
    assert avg_probs.loc["B", "A"] == pytest.approx(1.0, abs=1e-10)
    assert avg_probs.loc["C", "A"] == pytest.approx(1.0, abs=1e-10)

    assert avg_rates.loc["A", "B"] == pytest.approx(0.5, abs=1e-10)
    assert avg_rates.loc["A", "C"] == pytest.approx(0.5, abs=1e-10)


def test_analyze_grouped_transitions_averages_barriers_from_averaged_rates():
    pair = _plane_pair()
    bead0 = _transition_df(["A", "B", "A"], pair=pair, bead_id="00")
    bead1 = _transition_df(
        ["A", "C", "A"],
        pair=pair,
        bead_id="01",
        global_frame_index=[10, 11, 12],
    )
    df = pd.concat([bead0, bead1], ignore_index=True)

    result = analyze_grouped_transitions(df, pair, dt=1.0, temperature=300.0)

    avg_rates = result["averaged_rates"][("traj",)]
    avg_barriers = result["averaged_barriers"][("traj",)]
    expected = compute_activation_free_energy_barriers(
        avg_rates,
        temperature=300.0,
    )
    pd.testing.assert_frame_equal(avg_barriers, expected)


def test_load_or_build_trajectory_transitions_caches_barriers(tmp_path):
    cfg = {
        "dof": [
            {
                "name": "x",
                "type": "dihedral",
                "atoms": [0, 1, 2, 3],
                "label": "x",
                "domain": [-180, 180],
                "enabled": True,
            },
            {
                "name": "y",
                "type": "dihedral",
                "atoms": [1, 2, 3, 4],
                "label": "y",
                "domain": [-180, 180],
                "enabled": True,
            },
        ],
        "coordinate_pairs": [{"name": "dihedral", "x": "x", "y": "y"}],
    }
    pair = _dihedral_pair()
    df = _transition_df(["A", "B", "A"], pair=pair, bead_id=None)
    transitions_cfg = {
        "lag": 1,
        "dt": 1.0,
        "temperature": 300.0,
        "barrier_model": "arrhenius",
        "attempt_frequency": 1.0e12,
        "energy_conv_factor": 1.0,
        "energy_unit": "eV",
        "skip_noise_intermediates": True,
    }

    built, cache_hit = load_or_build_trajectory_transitions(
        df,
        "traj",
        transitions_cfg,
        {"average_across_beads": False},
        tmp_path,
        root_config=cfg,
    )
    cached, cached_hit = load_or_build_trajectory_transitions(
        df,
        "traj",
        transitions_cfg,
        {"average_across_beads": False},
        tmp_path,
        root_config=cfg,
    )

    assert cache_hit is False
    assert cached_hit is True
    assert ("traj",) in built["dihedral"]["per_group_barriers"]
    assert ("traj",) in cached["dihedral"]["per_group_barriers"]
    built_barrier = built["dihedral"]["per_group_barriers"][("traj",)]
    cached_barrier = cached["dihedral"]["per_group_barriers"][("traj",)]
    assert cached["dihedral"]["barrier_energy_unit"] == "eV"
    pd.testing.assert_frame_equal(
        cached_barrier,
        built_barrier.astype("float32"),
    )


# ---------------------------------------------------------------------------
# skip_noise_intermediates — noise-bridging behaviour
# ---------------------------------------------------------------------------


def test_skip_noise_single_noise_frame_bridged():
    """A single noise frame between two states should not hide the transition."""
    counts = compute_transition_counts(
        _series("A", "noise", "B"),
        lag=1,
        skip_noise_intermediates=True,
    )
    # After stripping: [A, B] → one A→B pair
    assert counts.loc["A", "B"] == 1
    assert "noise" not in counts.index


def test_skip_noise_multi_noise_frames_bridged():
    """Multiple consecutive noise frames between states should be bridged."""
    counts = compute_transition_counts(
        _series("A", "noise", "noise", "noise", "B"),
        lag=1,
        skip_noise_intermediates=True,
    )
    # After stripping: [A, B] → one A→B pair
    assert counts.loc["A", "B"] == 1


def test_skip_noise_false_hides_transition():
    """Without noise bridging the transition through noise is invisible at lag=1."""
    counts = compute_transition_counts(
        _series("A", "noise", "B"),
        lag=1,
        skip_noise_intermediates=False,
    )
    # (A, noise) and (noise, B) are both excluded; no valid pairs remain
    assert counts.empty or ("A" not in counts.index or "B" not in counts.columns
                             or counts.loc["A", "B"] == 0)


def test_skip_noise_all_noise_strips_to_empty():
    """All-noise sequence produces empty counts even with skip_noise_intermediates."""
    counts = compute_transition_counts(
        _series("noise", "noise", "noise"),
        lag=1,
        skip_noise_intermediates=True,
    )
    assert counts.empty


def test_skip_noise_mixed_sequence():
    """Realistic mixed sequence: noise-bridged transitions are counted correctly."""
    # Sequence: A, A, noise, B, B, noise, A
    # After stripping: A, A, B, B, A
    # lag=1 pairs: (A,A), (A,B), (B,B), (B,A) → 1 each
    counts = compute_transition_counts(
        _series("A", "A", "noise", "B", "B", "noise", "A"),
        lag=1,
        skip_noise_intermediates=True,
    )
    assert counts.loc["A", "A"] == 1
    assert counts.loc["A", "B"] == 1
    assert counts.loc["B", "B"] == 1
    assert counts.loc["B", "A"] == 1


def test_skip_noise_diagonal_becomes_off_diagonal():
    """The exact pattern seen in PIMD dihedral data: all same-state pairs become visible transitions."""
    # Simulate a bead with two conformers separated by noise
    n_per_state = 5
    seq = (
        ["0"] * n_per_state
        + ["noise"]
        + ["1"] * n_per_state
        + ["noise"]
        + ["0"] * n_per_state
    )
    counts = compute_transition_counts(
        pd.Series(seq, dtype="string"),
        lag=1,
        skip_noise_intermediates=True,
    )
    # Self-transitions
    assert counts.loc["0", "0"] == 2 * (n_per_state - 1)
    assert counts.loc["1", "1"] == n_per_state - 1
    # Cross-state transitions revealed by noise bridging
    assert counts.loc["0", "1"] == 1
    assert counts.loc["1", "0"] == 1


def test_analyze_grouped_transitions_skip_noise_default():
    """analyze_grouped_transitions should reveal cross-state transitions by default."""
    # One bead: two states separated by noise
    pair = _plane_pair()
    states_col = ["0", "0", "noise", "1", "1", "noise", "0"]
    df = _transition_df(
        states_col,
        pair=pair,
        bead_id=None,
    )
    result = analyze_grouped_transitions(df, pair, lag=1)
    counts = result["per_group_counts"][("traj",)]
    assert counts.loc["0", "1"] >= 1
    assert counts.loc["1", "0"] >= 1
