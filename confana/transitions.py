"""Transition analysis for conformational state sequences.

Computes state-to-state transition counts, row-normalised probability
matrices, optional rates, and optional activation free-energy barriers from
ordered state label sequences.  High-level grouped helpers operate on the
standard coordinate table and preserve bead identity for PIMD.

Design notes
------------
* Transitions are formed from consecutive pairs (states[i], states[i+lag]).
* Pairs where either label is ``"noise"`` or ``pd.NA`` are excluded.
* For PIMD workflows, call ``compute_transition_counts`` /
  ``compute_transition_probabilities`` per bead, then
  ``aggregate_pimd_transitions`` to average across beads.
* The caller is responsible for sorting frames by ``local_frame_index``
  within each (trajectory_id, bead_id) group before passing to
  ``compute_transition_counts``.

Public API
----------
- ``compute_transition_counts``
- ``compute_transition_probabilities``
- ``compute_transition_rates``
- ``compute_activation_free_energy_barriers``
- ``analyze_grouped_transitions``
- ``aggregate_pimd_transitions``
- ``load_or_build_trajectory_transitions``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from confana.coordinate_config import list_coordinate_pairs
from confana.models import CoordinatePair

_K_B_EV_PER_K = 8.617333262145e-5
_K_B_J_PER_K = 1.380649e-23
_PLANCK_J_S = 6.62607015e-34


# ---------------------------------------------------------------------------
# Transition counts
# ---------------------------------------------------------------------------


def compute_transition_counts(
    states: pd.Series,
    lag: int = 1,
    skip_noise_intermediates: bool = True,
) -> pd.DataFrame:
    """Count state-to-state transitions in an ordered label sequence.

    Parameters
    ----------
    states:
        Ordered Series of state label strings (e.g. ``"0"``, ``"1"``,
        ``"noise"``).  Must already be sorted by frame index.
    lag:
        Frame lag for transition pairs.  ``lag=1`` counts consecutive-frame
        transitions; ``lag=2`` uses pairs (i, i+2), etc.
    skip_noise_intermediates:
        When ``True`` (default), noise and NA frames are stripped from the
        sequence **before** forming pairs.  This means a run like
        ``[A, noise, noise, B]`` contributes one ``A→B`` transition.
        This is the recommended mode when conformational transitions pass
        through a noise/transition-state region that would otherwise make
        cross-state counts invisible.

        When ``False``, the original pair-based behaviour is used: pairs
        where either element is ``"noise"`` or NA are excluded, but no
        noise stripping occurs before pairing.

    Returns
    -------
    pd.DataFrame
        Square count matrix with state labels as both index and columns.
        Index and columns are sorted lexicographically.
        Returns an empty DataFrame if there are no valid transitions.

    Notes
    -----
    When ``skip_noise_intermediates=False``, pairs where either label is
    ``"noise"`` or NA are excluded.  When ``True``, noise/NA frames are
    removed before pairing so that transitions across noise regions are
    counted as direct state-to-state transitions.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")

    arr = states.reset_index(drop=True)

    if skip_noise_intermediates:
        # Strip noise and NA before forming pairs so that transitions
        # separated by any number of noise frames are still counted.
        valid_mask = arr.notna() & (arr != "noise")
        arr = arr[valid_mask].reset_index(drop=True)

    if len(arr) <= lag:
        return pd.DataFrame()

    src = arr.iloc[: len(arr) - lag]
    dst = arr.iloc[lag:]
    dst = dst.reset_index(drop=True)

    if skip_noise_intermediates:
        # Noise is already gone; accept all remaining pairs.
        src_valid = src.tolist()
        dst_valid = dst.tolist()
    else:
        # Legacy pair-level filter: exclude pairs where either is noise/NA.
        valid = (
            src.notna() & (src != "noise") &
            dst.notna() & (dst != "noise")
        )
        src_valid = src[valid].tolist()
        dst_valid = dst[valid].tolist()

    if not src_valid:
        return pd.DataFrame()

    # Collect all observed valid labels (sorted)
    all_labels = sorted(set(src_valid) | set(dst_valid))

    # Build count matrix
    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n = len(all_labels)
    counts = np.zeros((n, n), dtype=np.int64)
    for s, d in zip(src_valid, dst_valid):
        counts[label_idx[s], label_idx[d]] += 1

    return pd.DataFrame(counts, index=all_labels, columns=all_labels)


# ---------------------------------------------------------------------------
# Transition probabilities
# ---------------------------------------------------------------------------


def compute_transition_probabilities(
    counts: pd.DataFrame,
) -> pd.DataFrame:
    """Row-normalise a count matrix to produce a probability matrix.

    Parameters
    ----------
    counts:
        Square integer count matrix as returned by
        ``compute_transition_counts``.

    Returns
    -------
    pd.DataFrame
        Row-normalised probability matrix with the same index and columns.
        Rows whose total count is zero contain ``NaN`` (state never
        observed as a transition origin in the data).

    Notes
    -----
    ``P[i, j] = counts[i, j] / sum_j(counts[i, j])``
    """
    if counts.empty:
        return counts.copy().astype(float)

    row_sums = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(row_sums, axis=0)
    return probs


def compute_transition_rates(
    counts: pd.DataFrame,
    dt: float,
    lag: int = 1,
) -> pd.DataFrame:
    """Convert transition probabilities to discrete-time rates.

    Uses the Phase 9 convention::

        rate = probability / (lag * dt)
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")

    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0; got {dt}")

    probs = compute_transition_probabilities(counts)
    if probs.empty:
        return probs.copy()

    return probs / (lag * dt)


def compute_activation_free_energy_barriers(
    rates: pd.DataFrame,
    temperature: float,
    *,
    model: str = "eyring",
    transmission_coefficient: float = 1.0,
    attempt_frequency: float | None = None,
    energy_conv_factor: float = 1.0,
) -> pd.DataFrame:
    """Estimate activation free-energy barriers from transition rates.

    Barriers are computed in eV internally and multiplied by
    ``energy_conv_factor`` for reporting.  Diagonal entries are set to NaN
    because self-transitions are not barriers between conformational states.
    Off-diagonal entries with missing, zero, or negative rates are also NaN,
    meaning the barrier was not estimated from the available data.

    Parameters
    ----------
    rates:
        Square transition-rate matrix in s^-1.
    temperature:
        Absolute temperature in Kelvin.
    model:
        ``"eyring"`` uses ``transmission_coefficient * k_B * T / h`` as the
        prefactor.  ``"arrhenius"`` uses ``attempt_frequency``.
    transmission_coefficient:
        Dimensionless Eyring transmission coefficient.  Must be positive.
    attempt_frequency:
        Arrhenius prefactor in s^-1.  Required when ``model="arrhenius"``.
    energy_conv_factor:
        Multiplicative conversion factor applied to internal eV barriers.

    Returns
    -------
    pd.DataFrame
        Barrier matrix with the same labels as ``rates``.
    """
    temperature_value = float(temperature)
    if temperature_value <= 0.0:
        raise ValueError(
            f"temperature must be > 0 K when barriers are computed; "
            f"got {temperature_value}"
        )

    conv = float(energy_conv_factor)
    if conv <= 0.0 or not np.isfinite(conv):
        raise ValueError(f"energy_conv_factor must be finite and > 0; got {conv}")

    model_norm = str(model).strip().lower()
    if model_norm == "eyring":
        kappa = float(transmission_coefficient)
        if kappa <= 0.0 or not np.isfinite(kappa):
            raise ValueError(
                "transmission_coefficient must be finite and > 0 for "
                f"Eyring barriers; got {kappa}"
            )
        prefactor = kappa * _K_B_J_PER_K * temperature_value / _PLANCK_J_S
    elif model_norm == "arrhenius":
        if attempt_frequency is None:
            raise ValueError(
                "attempt_frequency is required when barrier_model is arrhenius."
            )
        prefactor = float(attempt_frequency)
        if prefactor <= 0.0 or not np.isfinite(prefactor):
            raise ValueError(
                f"attempt_frequency must be finite and > 0; got {prefactor}"
            )
    else:
        raise ValueError(
            "barrier_model must be 'eyring' or 'arrhenius'; "
            f"got {model!r}"
        )

    if rates.empty:
        return rates.copy().astype(float)

    rate_values = rates.to_numpy(dtype=float)
    valid = np.isfinite(rate_values) & (rate_values > 0.0)
    barriers = np.full(rate_values.shape, np.nan, dtype=float)
    barriers[valid] = (
        -_K_B_EV_PER_K
        * temperature_value
        * np.log(rate_values[valid] / prefactor)
        * conv
    )

    n = min(barriers.shape)
    barriers[np.arange(n), np.arange(n)] = np.nan
    return pd.DataFrame(barriers, index=rates.index.copy(), columns=rates.columns.copy())


def _normalize_group_key(key: Any) -> tuple:
    """Return group keys in a uniform tuple form."""
    if isinstance(key, tuple):
        return key
    return (key,)


def _align_transition_matrices(
    matrices: Sequence[pd.DataFrame],
    *,
    fill_value: float | None = None,
    preserve_missing_rows: bool = False,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Align matrices to the union of observed state labels."""
    if not matrices:
        raise ValueError("matrices is empty — nothing to align.")

    labels: set[str] = set()
    for matrix in matrices:
        labels.update(str(lbl) for lbl in matrix.index.tolist())
        labels.update(str(lbl) for lbl in matrix.columns.tolist())

    ordered_labels = sorted(labels)
    aligned: list[pd.DataFrame] = []
    for matrix in matrices:
        aligned_matrix = matrix.reindex(index=ordered_labels, columns=ordered_labels)
        if fill_value is not None:
            if preserve_missing_rows:
                present_rows = {str(lbl) for lbl in matrix.index.tolist()}
                added_cols = [
                    col_label
                    for col_label in ordered_labels
                    if col_label not in {str(lbl) for lbl in matrix.columns.tolist()}
                ]
                for row_label in ordered_labels:
                    if row_label in present_rows:
                        if added_cols:
                            aligned_matrix.loc[row_label, added_cols] = fill_value
                    else:
                        aligned_matrix.loc[row_label] = np.nan
            else:
                aligned_matrix = aligned_matrix.fillna(fill_value)
        aligned.append(aligned_matrix)
    return aligned, ordered_labels


def _average_transition_matrices(
    matrices: Sequence[pd.DataFrame],
    *,
    fill_value: float | None = None,
    preserve_missing_rows: bool = False,
) -> pd.DataFrame:
    """Average aligned transition matrices element-wise, ignoring NaN."""
    aligned, ordered_labels = _align_transition_matrices(
        matrices,
        fill_value=fill_value,
        preserve_missing_rows=preserve_missing_rows,
    )
    if len(aligned) == 1:
        return aligned[0].copy()

    stacked = pd.concat(aligned, keys=range(len(aligned)), axis=0)
    mean_matrix = stacked.groupby(level=1).mean()
    return mean_matrix.reindex(index=ordered_labels, columns=ordered_labels)


def _resolve_transition_groupby(
    df: pd.DataFrame,
    groupby: Sequence[str] | None = None,
) -> list[str]:
    """Resolve grouping fields for transition analysis."""
    if groupby is not None:
        groupby_list = [str(col) for col in groupby]
        if not groupby_list:
            raise ValueError("groupby cannot be an empty sequence.")
        return groupby_list

    if "trajectory_id" not in df.columns:
        raise ValueError(
            "Transition analysis requires a 'trajectory_id' column in the "
            "coordinate table."
        )

    if "bead_id" in df.columns and df["bead_id"].notna().any():
        return ["trajectory_id", "bead_id"]
    return ["trajectory_id"]


def analyze_grouped_transitions(
    df: pd.DataFrame,
    pair: CoordinatePair,
    lag: int = 1,
    groupby: Sequence[str] | None = None,
    dt: float | None = None,
    average_across_beads: bool = True,
    skip_noise_intermediates: bool = True,
    temperature: float | None = None,
    barrier_model: str = "eyring",
    transmission_coefficient: float = 1.0,
    attempt_frequency: float | None = None,
    energy_conv_factor: float = 1.0,
    energy_unit: str = "eV",
) -> dict[str, Any]:
    """Analyze grouped transitions for one coordinate pair.

    Returns per-group count/probability matrices and, for PIMD-like grouped
    data, averaged observables across bead groups sharing the same parent
    grouping fields.

    Parameters
    ----------
    df:
        Coordinate table with state column ``pair.state_col`` populated.
    pair:
        :class:`~confana.models.CoordinatePair` identifying the analysis space.
        ``pair.state_col`` names the state label column.
    skip_noise_intermediates:
        Forwarded to ``compute_transition_counts``.  When ``True`` (default),
        noise/NA frames are stripped before pairing so that transitions that
        route through a noise/transition-state region are still counted.
    """
    state_column = pair.state_col
    resolved_groupby = _resolve_transition_groupby(df, groupby=groupby)

    required = ["global_frame_index", state_column, *resolved_groupby]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Transition analysis requires columns {missing}. "
            f"Present columns: {list(df.columns)}"
        )

    dt_value = None if dt is None else float(dt)
    if dt_value is not None and dt_value <= 0.0:
        raise ValueError(f"dt must be > 0 when provided; got {dt_value}")

    temperature_value = None if temperature is None else float(temperature)
    if temperature_value is not None and temperature_value <= 0.0:
        raise ValueError(
            f"temperature must be > 0 K when provided; got {temperature_value}"
        )
    barriers_enabled = dt_value is not None and temperature_value is not None
    barrier_model_norm = str(barrier_model).strip().lower()
    energy_unit_value = str(energy_unit)

    per_group_counts: dict[tuple, pd.DataFrame] = {}
    per_group_probabilities: dict[tuple, pd.DataFrame] = {}
    per_group_rates: dict[tuple, pd.DataFrame] = {}
    per_group_barriers: dict[tuple, pd.DataFrame] = {}

    grouped = df.groupby(
        resolved_groupby if len(resolved_groupby) > 1 else resolved_groupby[0],
        dropna=False,
    )
    for raw_key, group_df in grouped:
        key = _normalize_group_key(raw_key)
        ordered = group_df.sort_values("global_frame_index")
        states = ordered[state_column].astype("string")

        counts = compute_transition_counts(
            states,
            lag=lag,
            skip_noise_intermediates=skip_noise_intermediates,
        )
        probs = compute_transition_probabilities(counts)

        per_group_counts[key] = counts
        per_group_probabilities[key] = probs
        if dt_value is not None:
            rates = compute_transition_rates(counts, dt=dt_value, lag=lag)
            per_group_rates[key] = rates
            if barriers_enabled:
                per_group_barriers[key] = compute_activation_free_energy_barriers(
                    rates,
                    temperature_value,
                    model=barrier_model_norm,
                    transmission_coefficient=transmission_coefficient,
                    attempt_frequency=attempt_frequency,
                    energy_conv_factor=energy_conv_factor,
                )

    averaged_counts: dict[tuple, pd.DataFrame] = {}
    averaged_probabilities: dict[tuple, pd.DataFrame] = {}
    averaged_rates: dict[tuple, pd.DataFrame] = {}
    averaged_barriers: dict[tuple, pd.DataFrame] = {}
    average_groupby: list[str] = []

    if average_across_beads and "bead_id" in resolved_groupby:
        average_groupby = [field for field in resolved_groupby if field != "bead_id"]

        count_buckets: dict[tuple, list[pd.DataFrame]] = {}
        prob_buckets: dict[tuple, list[pd.DataFrame]] = {}
        rate_buckets: dict[tuple, list[pd.DataFrame]] = {}

        bead_idx = resolved_groupby.index("bead_id")
        for key, counts in per_group_counts.items():
            parent_key = tuple(
                value for i, value in enumerate(key) if i != bead_idx
            )
            count_buckets.setdefault(parent_key, []).append(counts)
            prob_buckets.setdefault(parent_key, []).append(per_group_probabilities[key])
            if dt_value is not None:
                rate_buckets.setdefault(parent_key, []).append(per_group_rates[key])

        for parent_key, matrices in count_buckets.items():
            averaged_counts[parent_key] = _average_transition_matrices(
                matrices,
                fill_value=0.0,
            )
            averaged_probabilities[parent_key] = _average_transition_matrices(
                prob_buckets[parent_key],
                fill_value=0.0,
                preserve_missing_rows=True,
            )
            if dt_value is not None:
                averaged_rate_matrix = _average_transition_matrices(
                    rate_buckets[parent_key],
                    fill_value=0.0,
                    preserve_missing_rows=True,
                )
                averaged_rates[parent_key] = averaged_rate_matrix
                if barriers_enabled:
                    averaged_barriers[parent_key] = (
                        compute_activation_free_energy_barriers(
                            averaged_rate_matrix,
                            temperature_value,
                            model=barrier_model_norm,
                            transmission_coefficient=transmission_coefficient,
                            attempt_frequency=attempt_frequency,
                            energy_conv_factor=energy_conv_factor,
                        )
                    )

    return {
        "pair_name": pair.name,
        "state_column": state_column,
        "groupby": resolved_groupby,
        "average_groupby": average_groupby,
        "temperature": temperature_value,
        "barrier_model": barrier_model_norm,
        "transmission_coefficient": float(transmission_coefficient),
        "attempt_frequency": (
            None if attempt_frequency is None else float(attempt_frequency)
        ),
        "energy_conv_factor": float(energy_conv_factor),
        "barrier_energy_unit": energy_unit_value,
        "per_group_counts": per_group_counts,
        "per_group_probabilities": per_group_probabilities,
        "per_group_rates": per_group_rates,
        "per_group_barriers": per_group_barriers,
        "averaged_counts": averaged_counts,
        "averaged_probabilities": averaged_probabilities,
        "averaged_rates": averaged_rates,
        "averaged_barriers": averaged_barriers,
    }


# ---------------------------------------------------------------------------
# PIMD aggregation
# ---------------------------------------------------------------------------


def aggregate_pimd_transitions(
    per_bead_probs: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average per-bead probability matrices element-wise.

    Parameters
    ----------
    per_bead_probs:
        List of per-bead probability matrices (as returned by
        ``compute_transition_probabilities``).

    Returns
    -------
    pd.DataFrame
        Element-wise mean probability matrix. Matrices are first aligned to
        the union of observed state labels; ``NaN`` entries are ignored.

    Raises
    ------
    ValueError
        If ``per_bead_probs`` is empty.
    """
    if not per_bead_probs:
        raise ValueError(
            "per_bead_probs is empty — no bead probability matrices to aggregate."
        )

    if len(per_bead_probs) == 1:
        aligned, _ = _align_transition_matrices(
            per_bead_probs,
            fill_value=0.0,
            preserve_missing_rows=True,
        )
        return aligned[0]

    return _average_transition_matrices(
        per_bead_probs,
        fill_value=0.0,
        preserve_missing_rows=True,
    )


# ---------------------------------------------------------------------------
# Per-trajectory transition caching
# ---------------------------------------------------------------------------

_TRANSITION_SECTIONS = (
    "per_group_counts",
    "per_group_probabilities",
    "per_group_rates",
    "per_group_barriers",
    "averaged_counts",
    "averaged_probabilities",
    "averaged_rates",
    "averaged_barriers",
)


def _transitions_meta_key(section_name: str, definition: str) -> str:
    """Return the ``_SECTION_ABBR`` key scoped to a definition."""
    # Prefix section names with the definition so plane/dihedral results sit
    # in different NPZ arrays within the same archive.
    from confana.cache import _SECTION_ABBR  # noqa: PLC0415

    return f"{definition}_{_SECTION_ABBR[section_name]}"


def load_or_build_trajectory_transitions(
    df_traj: pd.DataFrame,
    trajectory_id: str,
    transitions_config: dict[str, Any],
    pimd_config: dict[str, Any],
    cache_dir: str | Path,
    *,
    root_config: dict[str, Any] | None = None,
    state_cache_meta: dict[str, Any] | None = None,
    force_rebuild: bool = False,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Load or build transition caches for all configured coordinate pairs.

    The cache file is stored at::

        {cache_dir}/{trajectory_id}__transitions.npz

    The invalidation fingerprint chains from the state cache: changing the
    state cache meta or any transition parameter invalidates this cache.

    Parameters
    ----------
    df_traj:
        Coordinate table with state columns already populated.
    trajectory_id:
        Trajectory identifier; used to name the cache file.
    transitions_config:
        Dict from ``configs/default.yaml`` under ``transitions:``.
    pimd_config:
        Dict from ``configs/default.yaml`` under ``pimd:``.
    cache_dir:
        Directory where the transition NPZ is stored.
    root_config:
        Full project config dict (used to resolve coordinate pairs).
        When ``None``, ``transitions_config`` is tried as the root config.
    state_cache_meta:
        Optional state-cache fingerprint dict for chained invalidation.
    force_rebuild:
        When True, bypass any existing cache.

    Returns
    -------
    tuple[dict[str, dict], bool]
        ``({pair_name: result, ...}, cache_hit)``
        where each inner dict is the output of ``analyze_grouped_transitions``.
    """
    import json as _json  # noqa: PLC0415

    from confana.cache import embed_meta, matches, read_matrix_section, write_matrix_section  # noqa: PLC0415

    root_cfg = root_config or transitions_config
    pairs = list_coordinate_pairs(root_cfg)

    cache_dir = Path(cache_dir)
    safe_id = trajectory_id.replace("/", "_").replace(" ", "_")
    cache_path = cache_dir / f"{safe_id}__transitions.npz"

    lag = int(transitions_config.get("lag", 1))
    dt = transitions_config.get("dt")
    temperature = transitions_config.get("temperature")
    barrier_model = str(transitions_config.get("barrier_model", "eyring"))
    transmission_coefficient = float(
        transitions_config.get("transmission_coefficient", 1.0)
    )
    attempt_frequency = transitions_config.get("attempt_frequency")
    energy_conv_factor = float(transitions_config.get("energy_conv_factor", 1.0))
    energy_unit = str(transitions_config.get("energy_unit", "eV"))
    average_across_beads = bool(pimd_config.get("average_across_beads", True))
    skip_noise_intermediates = bool(
        transitions_config.get("skip_noise_intermediates", True)
    )

    trans_meta: dict[str, Any] = {
        "version": 3,
        "trajectory_id": trajectory_id,
        "transitions": {
            "lag": lag,
            "dt": dt,
            "temperature": temperature,
            "barrier_model": barrier_model,
            "transmission_coefficient": transmission_coefficient,
            "attempt_frequency": attempt_frequency,
            "energy_conv_factor": energy_conv_factor,
            "energy_unit": energy_unit,
            "skip_noise_intermediates": skip_noise_intermediates,
        },
        "pimd": {"average_across_beads": average_across_beads},
        "pairs": [name for name, _ in pairs],
        "state_cache_meta": state_cache_meta,
    }

    if not force_rebuild and cache_path.exists() and matches(cache_path, trans_meta):
        results: dict[str, dict[str, Any]] = {}
        with np.load(cache_path, allow_pickle=False) as data:
            for pair_name, pair in pairs:
                prefix = f"{pair_name}_"
                pair_result: dict[str, Any] = {
                    "pair_name": pair_name,
                    "state_column": pair.state_col,
                    "groupby": None,
                    "average_groupby": [],
                    "temperature": (
                        None if temperature is None else float(temperature)
                    ),
                    "barrier_model": barrier_model.strip().lower(),
                    "transmission_coefficient": transmission_coefficient,
                    "attempt_frequency": (
                        None
                        if attempt_frequency is None
                        else float(attempt_frequency)
                    ),
                    "energy_conv_factor": energy_conv_factor,
                    "barrier_energy_unit": energy_unit,
                }
                for section in _TRANSITION_SECTIONS:
                    pair_result[section] = read_matrix_section(
                        data, section, key_prefix=prefix
                    )
                scalar_key = f"{pair_name}__scalar_meta"
                if scalar_key in data:
                    scalar = _json.loads(str(data[scalar_key].item()))
                    pair_result["groupby"] = scalar.get("groupby")
                    pair_result["average_groupby"] = scalar.get("average_groupby", [])
                    pair_result["temperature"] = scalar.get("temperature")
                    pair_result["barrier_model"] = scalar.get(
                        "barrier_model", pair_result["barrier_model"]
                    )
                    pair_result["transmission_coefficient"] = scalar.get(
                        "transmission_coefficient",
                        pair_result["transmission_coefficient"],
                    )
                    pair_result["attempt_frequency"] = scalar.get(
                        "attempt_frequency"
                    )
                    pair_result["energy_conv_factor"] = scalar.get(
                        "energy_conv_factor", pair_result["energy_conv_factor"]
                    )
                    pair_result["barrier_energy_unit"] = scalar.get(
                        "barrier_energy_unit", pair_result["barrier_energy_unit"]
                    )
                results[pair_name] = pair_result
        return results, True

    dt_value = float(dt) if dt is not None else None
    results = {}
    for pair_name, pair in pairs:
        results[pair_name] = analyze_grouped_transitions(
            df_traj,
            pair=pair,
            lag=lag,
            dt=dt_value,
            average_across_beads=average_across_beads,
            skip_noise_intermediates=skip_noise_intermediates,
            temperature=None if temperature is None else float(temperature),
            barrier_model=barrier_model,
            transmission_coefficient=transmission_coefficient,
            attempt_frequency=(
                None if attempt_frequency is None else float(attempt_frequency)
            ),
            energy_conv_factor=energy_conv_factor,
            energy_unit=energy_unit,
        )

    # Serialize to NPZ — each pair scoped with its name as key prefix
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for pair_name, result in results.items():
        prefix = f"{pair_name}_"
        for section in _TRANSITION_SECTIONS:
            write_matrix_section(
                payload, section, result.get(section, {}), key_prefix=prefix
            )
        scalar_meta = {
            "groupby": result.get("groupby"),
            "average_groupby": result.get("average_groupby", []),
            "temperature": result.get("temperature"),
            "barrier_model": result.get("barrier_model"),
            "transmission_coefficient": result.get("transmission_coefficient"),
            "attempt_frequency": result.get("attempt_frequency"),
            "energy_conv_factor": result.get("energy_conv_factor"),
            "barrier_energy_unit": result.get("barrier_energy_unit"),
        }
        payload[f"{pair_name}__scalar_meta"] = np.asarray(
            _json.dumps(scalar_meta), dtype=str
        )

    embed_meta(payload, trans_meta)
    np.savez(cache_path, **payload)

    return results, False
