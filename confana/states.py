"""Conformational state assignment via grid-based or DBSCAN clustering.

Clustering is performed independently per call (or per group when
``groupby`` is provided).  Cluster labels are stored as strings in a
``state_<pair_name>`` column of the returned DataFrame.

Design notes
------------
* Default algorithm is ``"grid"``: bins the 2D angle space, applies a
  minimum-count threshold, then finds connected components via
  ``scipy.ndimage.label``.  Fast (O(N) histogram + tiny grid labeling).
* ``"dbscan"`` is also supported.
* Noise points are labelled ``"noise"``; valid clusters receive string
  labels ``"0"``, ``"1"``, … sorted by component id.
* When ``groupby`` is supplied, clustering is done independently for each
  group.  Labels are NOT unified across groups.
* The input DataFrame is not mutated; a copy is returned.

Public API
----------
- ``assign_conformer_states``
- ``assign_conformer_states_from_config``
- ``load_or_build_trajectory_states``
- ``resolve_state_groupby``
- ``build_bin_state_overlay``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from confana.coordinate_config import list_coordinate_pairs
from confana.models import CoordinatePair


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dbscan_labels(
    X: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Run DBSCAN on a (N, 2) feature matrix and return integer labels."""
    from sklearn.cluster import DBSCAN  # noqa: PLC0415

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def _labels_to_strings(labels: np.ndarray) -> list[str]:
    """Convert integer cluster labels to strings; -1 → 'noise'."""
    return [str(lbl) if lbl >= 0 else "noise" for lbl in labels]


def _sincos_embedding(angles_deg: np.ndarray) -> np.ndarray:
    """Embed a 1D array of angles (degrees) as (N, 2) cos/sin columns."""
    rad = np.deg2rad(angles_deg)
    return np.column_stack([np.cos(rad), np.sin(rad)])


def _apply_periodic_embedding(X: np.ndarray) -> np.ndarray:
    """Embed an (N, k) raw-angle matrix into (N, 2k) via cos/sin."""
    parts = [_sincos_embedding(X[:, i]) for i in range(X.shape[1])]
    return np.hstack(parts)


def _eps_deg_to_sincos(eps_deg: float) -> float:
    """Convert eps in degrees to sin/cos chord equivalent."""
    return float(np.sqrt(2.0 - 2.0 * np.cos(np.deg2rad(eps_deg))))


def _merge_periodic_components(
    labeled_grid: np.ndarray,
    *,
    wrap_x: bool,
    wrap_y: bool,
) -> np.ndarray:
    """Merge connected-component labels across wrapped grid edges."""
    if not (wrap_x or wrap_y):
        return labeled_grid

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        if a <= 0 or b <= 0:
            return
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    if wrap_x and labeled_grid.shape[0] > 1:
        for yi in range(labeled_grid.shape[1]):
            union(int(labeled_grid[0, yi]), int(labeled_grid[-1, yi]))

    if wrap_y and labeled_grid.shape[1] > 1:
        for xi in range(labeled_grid.shape[0]):
            union(int(labeled_grid[xi, 0]), int(labeled_grid[xi, -1]))

    if not parent:
        return labeled_grid

    remapped = labeled_grid.copy()
    root_to_new: dict[int, int] = {}
    next_label = 1
    for label in np.unique(remapped[remapped > 0]):
        root = find(int(label))
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        remapped[remapped == label] = root_to_new[root]

    return remapped


def _grid_labels(
    df_group: pd.DataFrame,
    feature_cols: list[str],
    bin_size: float,
    min_count: int,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    wrap_x: bool = False,
    wrap_y: bool = False,
) -> np.ndarray:
    """Assign labels via 2D histogram + connected-component clustering."""
    from scipy.ndimage import label as ndimage_label  # noqa: PLC0415

    n = len(df_group)
    result = np.full(n, "noise", dtype=object)

    mask = df_group[feature_cols].notna().all(axis=1).to_numpy()
    if mask.sum() < 1:
        return result

    X = df_group.loc[mask, feature_cols].astype(float).to_numpy()
    x_vals = X[:, 0]
    y_vals = X[:, 1]

    x_lo, x_hi = x_range
    y_lo, y_hi = y_range

    x_edges = np.arange(x_lo, x_hi + bin_size, bin_size)
    y_edges = np.arange(y_lo, y_hi + bin_size, bin_size)

    hist, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])

    occupied = hist >= min_count
    labeled_grid, _ = ndimage_label(occupied)
    labeled_grid = _merge_periodic_components(
        labeled_grid,
        wrap_x=wrap_x,
        wrap_y=wrap_y,
    )

    xi = np.clip(np.digitize(x_vals, x_edges) - 1, 0, len(x_edges) - 2)
    yi = np.clip(np.digitize(y_vals, y_edges) - 1, 0, len(y_edges) - 2)
    component_ids = labeled_grid[xi, yi]

    str_labels = [str(c - 1) if c > 0 else "noise" for c in component_ids]

    valid_indices = np.where(mask)[0]
    for i, lbl in zip(valid_indices, str_labels):
        result[i] = lbl

    return result


def _cluster_group(
    df_group: pd.DataFrame,
    pair: CoordinatePair,
    feature_cols: list[str],
    scheme: str,
    params: dict,
) -> np.ndarray:
    """Cluster one group of rows; return an array of string labels."""
    n = len(df_group)
    result = np.full(n, "noise", dtype=object)

    mask = df_group[feature_cols].notna().all(axis=1).to_numpy()
    if mask.sum() < 2:
        return result

    if scheme == "grid":
        return _grid_labels(
            df_group,
            feature_cols,
            params["bin_size"],
            params["min_count"],
            x_range=pair.x_domain,
            y_range=pair.y_domain,
            wrap_x=pair.periodic,
            wrap_y=pair.periodic,
        )

    # dbscan
    X = df_group.loc[mask, feature_cols].astype(float).to_numpy()
    eps = params["eps"]
    if params.get("periodic", False):
        X = _apply_periodic_embedding(X)
        eps = _eps_deg_to_sincos(eps)
    int_labels = _dbscan_labels(X, eps=eps, min_samples=params["min_samples"])
    str_labels = _labels_to_strings(int_labels)

    valid_indices = np.where(mask)[0]
    for i, lbl in zip(valid_indices, str_labels):
        result[i] = lbl

    return result


def _coerce_groupby(groupby: Sequence[str] | None) -> list[str] | None:
    """Return groupby as a list, or None when no grouping is requested."""
    if groupby is None:
        return None
    groupby_list = [str(col) for col in groupby]
    return groupby_list or None


def _resolve_pair_params(
    clustering_config: dict[str, Any],
    pair_name: str,
) -> tuple[str, dict[str, Any], list[str] | None]:
    """Resolve clustering algorithm and params for one pair from config.

    Looks up ``clustering.<pair_name>`` first, then falls back to
    ``clustering.default``.  Supports both ``"grid"`` and ``"dbscan"``
    algorithms.

    Config shape::

        clustering:
          algorithm: grid
          groupby: [trajectory_id]
          default:
            bin_size: 30.0
            min_count: 100
            eps: 7
            min_samples: 10
            periodic: true
          # optional per-pair overrides
          dihedral:
            bin_size: 15.0

    Returns
    -------
    tuple[str, dict, list | None]
        ``(scheme, params, groupby)``
    """
    scheme = str(clustering_config.get("algorithm", "grid"))
    groupby = _coerce_groupby(clustering_config.get("groupby"))

    # Merge default + pair-specific config (pair-specific overrides default)
    default_cfg = dict(clustering_config.get("default", {}) or {})
    pair_cfg = dict(clustering_config.get(pair_name, {}) or {})
    merged = {**default_cfg, **pair_cfg}

    if scheme == "grid":
        params: dict[str, Any] = {
            "bin_size": merged.get("bin_size"),
            "min_count": merged.get("min_count"),
        }
    else:
        eps = merged.get("eps", clustering_config.get("eps"))
        min_samples = merged.get("min_samples", clustering_config.get("min_samples"))
        params = {
            "eps": eps,
            "min_samples": min_samples,
            "periodic": bool(merged.get("periodic", False)),
            "embedding": str(merged.get("embedding", "sin_cos")),
        }

    return scheme, params, groupby


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_conformer_states(
    df: pd.DataFrame,
    pair: CoordinatePair,
    scheme: str,
    params: dict,
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Assign conformational state labels via clustering for one pair.

    Parameters
    ----------
    df:
        Coordinate table with DoF columns.
    pair:
        :class:`~confana.models.CoordinatePair` specifying the 2D analysis space.
        Domains (``x_domain``, ``y_domain``) determine grid range.
        ``pair.periodic`` enables wrapped grid clustering.
        ``pair.state_col`` names the output column.
        ``pair.feature_columns`` names the input columns.
    scheme:
        Clustering algorithm: ``"grid"`` (default) or ``"dbscan"``.
    params:
        Algorithm parameters dict.

        * ``"grid"``: must contain ``"bin_size"`` (float) and
          ``"min_count"`` (int).
        * ``"dbscan"``: must contain ``"eps"`` (float) and
          ``"min_samples"`` (int); optional ``"periodic"`` (bool).
    groupby:
        Optional list of column names to group by before clustering.
        Cluster labels are NOT unified across groups.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``pair.state_col`` populated.
        Dtype is ``"string"`` (nullable).

    Raises
    ------
    ValueError
        If ``scheme`` is unsupported, required params are missing, or
        feature columns are absent.
    """
    _VALID_SCHEMES = ("grid", "dbscan")

    if scheme not in _VALID_SCHEMES:
        raise ValueError(
            f"Unsupported clustering scheme '{scheme}'. "
            f"Valid choices: {list(_VALID_SCHEMES)}"
        )

    if scheme == "dbscan":
        eps = params.get("eps")
        min_samples = params.get("min_samples")
        if eps is None:
            raise ValueError(
                "clustering.eps is null in config — "
                "set it after inspecting density plots."
            )
        if min_samples is None:
            raise ValueError(
                "clustering.min_samples is null in config — "
                "set it after inspecting density plots."
            )
        params = {
            "eps": float(eps),
            "min_samples": int(min_samples),
            "periodic": bool(params.get("periodic", pair.periodic)),
            "embedding": str(params.get("embedding", "sin_cos")),
        }
    elif scheme == "grid":
        bin_size = params.get("bin_size")
        min_count = params.get("min_count")
        if bin_size is None:
            raise ValueError(
                "clustering.bin_size is null in config — "
                "set it under clustering.default.bin_size."
            )
        if min_count is None:
            raise ValueError(
                "clustering.min_count is null in config — "
                "set it under clustering.default.min_count."
            )
        params = {"bin_size": float(bin_size), "min_count": int(min_count)}

    feature_cols = pair.feature_columns
    target_col = pair.state_col

    result = df.copy()

    if target_col not in result.columns:
        result[target_col] = pd.NA

    missing_features = [col for col in feature_cols if col not in result.columns]
    if missing_features:
        raise ValueError(
            f"Feature columns not found in DataFrame for pair '{pair.name}': "
            f"{missing_features}"
        )

    groupby = _coerce_groupby(groupby)
    if groupby:
        missing = [col for col in groupby if col not in result.columns]
        if missing:
            raise ValueError(
                f"Grouping columns not found in DataFrame: {missing}. "
                f"Present columns: {list(result.columns)}"
            )

    if groupby:
        for _, group_idx in result.groupby(
            groupby if len(groupby) > 1 else groupby[0],
            dropna=False,
        ).groups.items():
            sub = result.loc[group_idx]
            labels = _cluster_group(sub, pair, feature_cols, scheme, params)
            result.loc[group_idx, target_col] = labels
    else:
        labels = _cluster_group(result, pair, feature_cols, scheme, params)
        result[target_col] = labels

    result[target_col] = result[target_col].astype("string")
    return result


def assign_conformer_states_from_config(
    df: pd.DataFrame,
    clustering_config: dict[str, Any],
    groupby_override: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assign state columns for all configured coordinate pairs.

    Parameters
    ----------
    df:
        Standard coordinate table.
    clustering_config:
        Either the ``clustering:`` section or the full root config dict.
    groupby_override:
        Optional explicit grouping fields to use for all pairs instead of
        the configured ``clustering.groupby`` value.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a ``state_<pair_name>`` column per pair.
    """
    root_cfg = clustering_config if "clustering" in clustering_config else {}
    clustering_section = clustering_config.get("clustering", clustering_config)

    pairs = list_coordinate_pairs(root_cfg) if root_cfg else []
    if not pairs:
        # Fallback: no pairs configured — return unchanged
        return df.copy()

    result = df
    override = _coerce_groupby(groupby_override)

    for pair_name, pair in pairs:
        scheme, params, groupby = _resolve_pair_params(clustering_section, pair_name)
        if override is not None:
            groupby = override
        result = assign_conformer_states(result, pair, scheme, params, groupby=groupby)

        # Diagnostic summary
        col = pair.state_col
        if col in result.columns:
            counts = result[col].value_counts(dropna=False)
            n_noise = int(counts.get("noise", 0))
            n_na = int(result[col].isna().sum())
            n_assigned = int(len(result) - n_noise - n_na)
            n_clusters = int(
                sum(
                    1
                    for k in counts.index
                    if k not in ("noise",) and not (isinstance(k, float) and pd.isna(k))
                )
            )
            print(
                f"  [{pair_name}] {n_clusters} cluster(s), "
                f"{n_assigned:,} assigned / {n_noise:,} noise / {n_na:,} unset"
            )

    return result


def load_or_build_trajectory_states(
    df_traj: pd.DataFrame,
    trajectory_id: str,
    clustering_config: dict[str, Any],
    cache_dir: str | Path,
    *,
    coord_cache_meta: dict[str, Any] | None = None,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Load or build the state-label cache for one trajectory.

    The cache file is stored at::

        {cache_dir}/{trajectory_id}__states.npz

    It contains ``frame_id`` plus all ``state_<pair_name>`` columns.

    Parameters
    ----------
    df_traj:
        Coordinate table for one trajectory (all beads included).
    trajectory_id:
        Trajectory identifier; used to name the cache file.
    clustering_config:
        Dict from ``configs/default.yaml`` under ``clustering:`` or the
        full root config.
    cache_dir:
        Directory where the state NPZ is stored.
    coord_cache_meta:
        Optional coordinate-cache fingerprint dict to embed in the state
        cache for chained invalidation.
    force_rebuild:
        When True, bypass any existing cache.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(df_with_states, cache_hit)``
    """
    from confana.cache import matches  # noqa: PLC0415
    from confana.io_coordinates import _coerce_dtypes, _read_coordinate_npz, _write_coordinate_npz  # noqa: PLC0415

    cache_dir = Path(cache_dir)
    safe_id = trajectory_id.replace("/", "_").replace(" ", "_")
    cache_path = cache_dir / f"{safe_id}__states.npz"

    root_cfg = clustering_config if "clustering" in clustering_config else {}
    clustering_section = clustering_config.get("clustering", clustering_config)

    pairs = list_coordinate_pairs(root_cfg) if root_cfg else []
    state_cols = [pair.state_col for _, pair in pairs]

    state_meta: dict[str, Any] = {
        "version": 2,
        "trajectory_id": trajectory_id,
        "clustering": {
            "algorithm": clustering_section.get("algorithm", "grid"),
            "groupby": clustering_section.get("groupby"),
            "default": dict(clustering_section.get("default", {}) or {}),
            # Include per-pair overrides so any change invalidates cache
            "pairs": {
                name: dict(clustering_section.get(name, {}) or {})
                for name, _ in pairs
            },
        },
        "coordinate_pairs": [
            {"name": n} for n, _ in pairs
        ],
        "coord_cache_meta": coord_cache_meta,
    }

    if not force_rebuild and cache_path.exists() and matches(cache_path, state_meta):
        state_df = _read_coordinate_npz(cache_path)
        result = df_traj.copy()
        available_state_cols = [c for c in state_cols if c in state_df.columns]
        for col in available_state_cols:
            if col in result.columns:
                result = result.drop(columns=[col])
        join_cols = ["frame_id"] + available_state_cols
        result = result.merge(
            state_df[join_cols],
            on="frame_id",
            how="left",
        )
        return _coerce_dtypes(result), True

    result = assign_conformer_states_from_config(df_traj, clustering_section)

    save_cols = ["frame_id"] + [c for c in state_cols if c in result.columns]
    state_df_save = result[save_cols].copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_coordinate_npz(state_df_save, cache_path, cache_metadata=state_meta)

    return result, False


# ---------------------------------------------------------------------------
# Per-bin state overlay (for the interactive density page)
# ---------------------------------------------------------------------------

_NOISE_LABEL = "noise"


def resolve_state_groupby(config: dict[str, Any]) -> list[str] | None:
    """Return the grouping columns states were clustered by, from config.

    Reads ``clustering.groupby`` exactly as
    :func:`assign_conformer_states_from_config` does (from the full config
    or the ``clustering:`` section). ``None`` means one group of all frames.
    """
    section = config.get("clustering", config) or {}
    return _coerce_groupby(section.get("groupby"))


def _natural_label_key(label: str) -> tuple[int, int, str]:
    """Sort numeric labels by value ("2" before "10"), then others by text."""
    return (0, int(label), "") if label.isdigit() else (1, 0, label)


def _histogram_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value as ``np.histogram2d`` assigns it; -1 outside the edges.

    Bins are right-open except the last, which includes the right edge.
    """
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(edges, v, side="right") - 1
    idx[v == edges[-1]] = len(edges) - 2
    idx[~np.isfinite(v) | (v < edges[0]) | (v > edges[-1])] = -1
    return idx


def _axis_centre(values: np.ndarray, periodic: bool) -> float:
    """Mean of *values* (degrees); circular mean when the axis is periodic."""
    if periodic:
        rad = np.deg2rad(values)
        return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))
    return float(np.mean(values))


def _state_group_name(keys: dict[str, Any]) -> str:
    """Readable group name, e.g. ``"my_run.pos / bead 00"``."""
    parts = []
    for col, value in keys.items():
        text = "none" if value is None else str(value)
        parts.append(f"bead {text}" if col == "bead_id" else text)
    return " / ".join(parts) or "all frames"


def build_bin_state_overlay(
    df: pd.DataFrame,
    pair: CoordinatePair,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    groupby: Sequence[str] | None = None,
) -> dict | None:
    """Return the majority state of each histogram bin, one map per group.

    States are clustered separately per ``groupby`` group and their labels
    are not unified across groups ("0" in one bead may be a different region
    from "0" in another), so bins are never pooled across groups: each group
    gets its own map.

    Within a group, each bin takes the label held by most of its frames.
    Noise and unset (NA) frames take part; when they win, the bin has no
    state. Ties go to a real state first, then to the earlier label in
    natural order ("2" before "10"). Frames outside the edges are ignored,
    with bins assigned exactly as :func:`np.histogram2d` assigns them.

    Parameters
    ----------
    df:
        Coordinate table with ``pair.feature_columns`` and ``pair.state_col``.
    pair:
        Coordinate pair; ``pair.periodic`` makes the centres circular means.
    x_edges, y_edges:
        Histogram bin edges of the density map.
    groupby:
        Columns the states were clustered by (see :func:`resolve_state_groupby`).

    Returns
    -------
    dict | None
        ``None`` when ``pair.state_col`` is not in *df*. Otherwise::

            {"state_col": str,
             "groupby": [str, ...],
             "labels": [str, ...],          # every state label, natural order
             "groups": [{
                 "name": str,               # e.g. "my_run.pos / bead 00"
                 "keys": {col: str | None},
                 "bins": [int, ...],        # flat bin index yi * n_bins_x + xi
                 "states": [int, ...],      # index into labels, per bin
                 "centres": [{"state": int, "x": float, "y": float, "frames": int}]
             }, ...]}

        ``centres`` are each state's population-weighted centre in the
        group: the mean (circular for periodic pairs) of its frames' values.

    Raises
    ------
    ValueError
        If a ``groupby`` column or a feature column is missing from *df*.
    """
    state_col = pair.state_col
    if state_col not in df.columns:
        return None

    groupby_cols = _coerce_groupby(groupby) or []
    x_col, y_col = pair.feature_columns
    missing = [c for c in [x_col, y_col, *groupby_cols] if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_bin_state_overlay: columns {missing} not found for pair '{pair.name}'."
        )

    x_values = df[x_col].to_numpy(dtype=float)
    y_values = df[y_col].to_numpy(dtype=float)
    xi = _histogram_bin_indices(x_values, np.asarray(x_edges, dtype=float))
    yi = _histogram_bin_indices(y_values, np.asarray(y_edges, dtype=float))
    inside = (xi >= 0) & (yi >= 0)
    flat_bin = yi * (len(x_edges) - 1) + xi

    raw = df[state_col].astype("string")
    labelled = (raw.notna() & raw.ne(_NOISE_LABEL)).fillna(False).to_numpy(dtype=bool)
    usable = labelled & inside
    labels = sorted(set(raw[usable].tolist()), key=_natural_label_key)
    no_state = len(labels)
    codes = np.full(len(df), no_state, dtype=np.int64)
    if labels:
        codes[usable] = pd.Categorical(raw[usable], categories=labels).codes

    if groupby_cols:
        by = groupby_cols if len(groupby_cols) > 1 else groupby_cols[0]
        group_items = df.groupby(by, dropna=False, sort=True).indices.items()
    else:
        group_items = [((), np.arange(len(df)))]

    groups = []
    n_codes = no_state + 1
    for key, positions in group_items:
        key_tuple = key if isinstance(key, tuple) else (key,)
        keys = {
            col: None if pd.isna(value) else str(value)
            for col, value in zip(groupby_cols, key_tuple)
        }
        pos = np.asarray(positions)[inside[positions]]
        group_codes = codes[pos]

        combo, counts = np.unique(flat_bin[pos] * n_codes + group_codes, return_counts=True)
        bins, bin_codes = combo // n_codes, combo % n_codes
        # Per bin: most frames first, then the lowest code (real states before
        # the no-state code, then natural label order).
        order = np.lexsort((bin_codes, -counts, bins))
        bins, bin_codes = bins[order], bin_codes[order]
        first = np.ones(len(bins), dtype=bool)
        first[1:] = bins[1:] != bins[:-1]
        winner_bins, winner_codes = bins[first], bin_codes[first]
        has_state = winner_codes < no_state

        centres = []
        for code in np.unique(group_codes[group_codes < no_state]):
            rows = pos[group_codes == code]
            centres.append({
                "state": int(code),
                "x": _axis_centre(x_values[rows], pair.periodic),
                "y": _axis_centre(y_values[rows], pair.periodic),
                "frames": int(len(rows)),
            })

        groups.append({
            "name": _state_group_name(keys),
            "keys": keys,
            "bins": winner_bins[has_state].tolist(),
            "states": winner_codes[has_state].tolist(),
            "centres": centres,
        })

    return {
        "state_col": state_col,
        "groupby": groupby_cols,
        "labels": labels,
        "groups": groups,
    }
