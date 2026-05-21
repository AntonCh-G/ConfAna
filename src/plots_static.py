"""Static PNG plot generation for conformer density analysis.

All functions save figures to disk and return the output path.  They have
no side effects on global matplotlib state (figures are closed after saving).

Public API
----------
- ``make_density_png``
- ``make_transition_png``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.density import compute_2d_histogram_arrays
from src.models import CoordinatePair


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_density_config(config: dict | None) -> tuple[dict, dict]:
    """Return ``(root_cfg, density_cfg)`` for full-config or density-only input."""
    cfg = config or {}
    if "density" in cfg:
        return cfg, cfg.get("density", {}) or {}
    return {}, cfg


def _prepare_density_values(H: np.ndarray, log_scale: bool) -> np.ndarray:
    """Mask unsampled bins and transform populated bins for plotting."""
    values = H.T.astype(float)
    values[values == 0.0] = np.nan
    if log_scale:
        return np.log10(values + 1.0)
    return values


def _prepare_density_colormap(colormap: str):
    """Return a copy of the colormap with unsampled bins rendered in white."""
    import matplotlib  # noqa: PLC0415

    cmap = matplotlib.colormaps[colormap].copy()
    cmap.set_bad(color="white")
    return cmap


def _weighted_axis_com(
    centres: np.ndarray,
    weights: np.ndarray,
    value_range: tuple[float, float],
    *,
    periodic: bool,
) -> float:
    """Return the population-weighted centre of mass for one binned axis."""
    centres = np.asarray(centres, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(centres) == 0 or float(weights.sum()) <= 0.0:
        return float("nan")

    if not periodic:
        return float(np.average(centres, weights=weights))

    lo, hi = value_range
    width = hi - lo
    if width <= 0:
        raise ValueError(f"Invalid periodic range {value_range!r}; expected min < max.")

    theta = ((centres - lo) / width) * (2.0 * np.pi)
    sin_sum = float(np.sum(weights * np.sin(theta)))
    cos_sum = float(np.sum(weights * np.cos(theta)))
    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
        return float(np.average(centres, weights=weights))

    mean_theta = float(np.arctan2(sin_sum, cos_sum) % (2.0 * np.pi))
    value = lo + (mean_theta / (2.0 * np.pi)) * width
    if value >= hi:
        value -= width
    return float(value)


def _compute_state_bin_com_positions(
    df: pd.DataFrame,
    state_col: str,
    x_col: str,
    y_col: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    periodic: bool,
) -> pd.DataFrame:
    """Calculate state marker positions from bin-population centres of mass.

    Each populated bin contributes its centre as a position and its frame count
    as mass.  This keeps the marker tied to the plotted population landscape
    instead of to the raw per-frame median.
    """
    state_mask = df[state_col].notna() & (df[state_col] != "noise")
    state_df = df.loc[state_mask & df[x_col].notna() & df[y_col].notna()]
    if len(state_df) == 0:
        return pd.DataFrame(columns=[x_col, y_col])

    x_centres = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2.0

    rows: list[dict[str, float | str]] = []
    for label, group in state_df.groupby(state_col, dropna=False):
        x_values = group[x_col].astype(float).to_numpy()
        y_values = group[y_col].astype(float).to_numpy()
        hist, _, _ = np.histogram2d(x_values, y_values, bins=[x_edges, y_edges])
        xi, yi = np.nonzero(hist)
        if len(xi) == 0:
            continue

        weights = hist[xi, yi].astype(float)
        rows.append(
            {
                state_col: str(label),
                x_col: _weighted_axis_com(
                    x_centres[xi],
                    weights,
                    x_range,
                    periodic=periodic,
                ),
                y_col: _weighted_axis_com(
                    y_centres[yi],
                    weights,
                    y_range,
                    periodic=periodic,
                ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=[x_col, y_col])

    return pd.DataFrame(rows).set_index(state_col)


def _cluster_grid_edges(
    value_range: tuple[float, float],
    bin_size: float,
) -> np.ndarray:
    """Return grid-clustering bin edges for a configured domain."""
    lo, hi = value_range
    if bin_size <= 0:
        raise ValueError(f"clustering bin_size must be > 0, got {bin_size!r}.")
    return np.arange(lo, hi + bin_size, bin_size)


def _state_com_grid_from_config(
    pair: CoordinatePair,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    config: dict | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """Return binning used for state COM markers.

    Grid-clustered states are represented by the same grid that assigned the
    states.  Non-grid clustering or density-only config falls back to the
    visible density grid.
    """
    root_cfg = config or {}
    clustering_cfg = root_cfg.get("clustering", {}) if "clustering" in root_cfg else {}
    if str(clustering_cfg.get("algorithm", "grid")) != "grid":
        return x_edges, y_edges, x_range, y_range

    default_cfg = dict(clustering_cfg.get("default", {}) or {})
    pair_cfg = dict(clustering_cfg.get(pair.name, {}) or {})
    merged = {**default_cfg, **pair_cfg}
    bin_size = merged.get("bin_size")
    if bin_size is None:
        return x_edges, y_edges, x_range, y_range

    cluster_x_range = pair.x_domain
    cluster_y_range = pair.y_domain
    return (
        _cluster_grid_edges(cluster_x_range, float(bin_size)),
        _cluster_grid_edges(cluster_y_range, float(bin_size)),
        cluster_x_range,
        cluster_y_range,
    )


# ---------------------------------------------------------------------------
# Density PNG
# ---------------------------------------------------------------------------


def make_density_png(
    df: pd.DataFrame,
    pair: CoordinatePair,
    outpath: str | Path,
    dpi: int = 300,
    config: dict | None = None,
    overlays: list[dict] | None = None,
) -> Path:
    """Save a 2D coordinate-density PNG for the given coordinate pair.

    Parameters
    ----------
    df:
        Standard coordinate table (must contain the DoF columns for ``pair``).
    pair:
        :class:`~src.models.CoordinatePair` specifying which columns to plot.
        Domain, labels, and per-pair overrides (bins, colormap, log_scale,
        x_range, y_range) are read from the pair object.
    outpath:
        Destination file path (parent directory is created if needed).
    dpi:
        Output resolution in dots per inch.
    config:
        Optional dict with either the full project config or just the
        ``density:`` section from ``configs/default.yaml``.  Missing keys fall
        back to ``pair.x_domain`` / ``pair.y_domain``.
    overlays:
        Optional list of scatter overlay datasets rendered on top of the
        density heatmap.  Each entry is a dict with keys:

        - ``"df"`` — coordinate table (must contain the same DoF columns as
          ``pair.feature_columns``)
        - ``"label"`` — legend entry string
        - ``"color"`` — matplotlib color string

        Missing DoF columns for a given overlay are skipped with a warning.

    Returns
    -------
    Path
        Resolved path of the saved PNG.

    Raises
    ------
    ValueError
        If the feature columns are not found in ``df``.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    _, density_cfg = _split_density_config(config)

    # Per-pair overrides take precedence over global density config.
    bins: int = (
        pair.bins if pair.bins is not None
        else int(density_cfg.get("bins", 180))
    )
    colormap: str = (
        pair.colormap if pair.colormap is not None
        else str(density_cfg.get("colormap", "viridis"))
    )
    log_scale: bool = (
        pair.log_scale if pair.log_scale is not None
        else bool(density_cfg.get("log_scale", True))
    )

    # Axis ranges: per-pair override → global config → pair domain
    raw_x_range = density_cfg.get("x_range")
    raw_y_range = density_cfg.get("y_range")
    x_range: tuple[float, float] = (
        pair.x_range if pair.x_range is not None
        else (tuple(raw_x_range) if raw_x_range is not None else pair.x_domain)
    )
    y_range: tuple[float, float] = (
        pair.y_range if pair.y_range is not None
        else (tuple(raw_y_range) if raw_y_range is not None else pair.y_domain)
    )

    # Feature columns account for any *_shifted variants.
    x_col, y_col = pair.feature_columns
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(
                f"make_density_png: column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    mask = df[x_col].notna() & df[y_col].notna()
    x_data = df.loc[mask, x_col].astype(float).to_numpy()
    y_data = df.loc[mask, y_col].astype(float).to_numpy()

    H, x_edges, y_edges = compute_2d_histogram_arrays(
        x_data,
        y_data,
        bins=bins,
        x_range=x_range,
        y_range=y_range,
    )

    C = _prepare_density_values(H, log_scale=log_scale)
    cbar_label = "log\u2081\u2080(count + 1)" if log_scale else "count"

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = _prepare_density_colormap(colormap)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        C,
        cmap=cmap,
        shading="flat",
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(pair.x_label)
    ax.set_ylabel(pair.y_label)
    ax.set_title(pair.title)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Overlay state COM markers when the state column is present in df.
    state_col = pair.state_col
    if state_col in df.columns:
        marker_x_edges, marker_y_edges, marker_x_range, marker_y_range = (
            _state_com_grid_from_config(
                pair,
                x_edges,
                y_edges,
                x_range,
                y_range,
                config,
            )
        )
        state_positions = _compute_state_bin_com_positions(
            df,
            state_col,
            x_col,
            y_col,
            marker_x_edges,
            marker_y_edges,
            marker_x_range,
            marker_y_range,
            periodic=pair.periodic,
        )
        if len(state_positions) > 0:
            ax.scatter(
                state_positions[x_col].astype(float),
                state_positions[y_col].astype(float),
                c="red",
                s=80,
                zorder=5,
                marker="o",
                linewidths=0.5,
                edgecolors="white",
            )
            for lbl, row in state_positions.iterrows():
                ax.annotate(
                    str(lbl),
                    (float(row[x_col]), float(row[y_col])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color="red",
                )

    # Scatter overlays from external datasets.
    if overlays:
        import warnings  # noqa: PLC0415

        for ov in overlays:
            ov_df = ov["df"]
            ov_label = ov.get("label", "")
            ov_color = ov.get("color", "white")
            if x_col not in ov_df.columns or y_col not in ov_df.columns:
                warnings.warn(
                    f"Scatter overlay '{ov_label}': columns {x_col!r} / {y_col!r} "
                    "not found — skipping.",
                    stacklevel=2,
                )
                continue
            ov_mask = ov_df[x_col].notna() & ov_df[y_col].notna()
            ov_x = ov_df.loc[ov_mask, x_col].astype(float).to_numpy()
            ov_y = ov_df.loc[ov_mask, y_col].astype(float).to_numpy()
            if len(ov_x) == 0:
                continue
            ax.scatter(
                ov_x,
                ov_y,
                s=4,
                alpha=0.8,
                color=ov_color,
                label=ov_label,
                zorder=4,
                linewidths=0,
                rasterized=True,
            )
        ax.legend(loc="upper right", fontsize=7, markerscale=3,
                  framealpha=0.7, handletextpad=0.4)

    fig.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

    return outpath


# ---------------------------------------------------------------------------
# Transition heatmaps
# ---------------------------------------------------------------------------


def _select_transition_matrix(transitions: dict, key_averaged: str, key_per_group: str):
    """Return (matrix_df, group_label) preferring PIMD-averaged over per-group."""
    averaged = transitions.get(key_averaged, {})
    if averaged:
        key, df = next(iter(averaged.items()))
        return df, str(key)
    per_group = transitions.get(key_per_group, {})
    if per_group:
        key, df = next(iter(per_group.items()))
        return df, str(key)
    return None, ""


def _draw_transition_heatmap(ax, matrix: pd.DataFrame, colormap: str, label: str,
                              fig, max_annotate: int) -> None:
    """Draw one transition heatmap onto *ax*."""
    import matplotlib  # noqa: PLC0415

    values = matrix.to_numpy(dtype=float)
    n = len(matrix)

    im = ax.imshow(values, aspect="auto", cmap=matplotlib.colormaps[colormap],
                   origin="upper")
    fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    state_labels = matrix.columns.tolist()
    ax.set_xticklabels(state_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(state_labels, fontsize=8)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")

    if n <= max_annotate:
        for i in range(n):
            for j in range(n):
                v = values[i, j]
                if np.isfinite(v):
                    txt = f"{v:.2g}"
                    vmin = np.nanmin(values)
                    vmax = np.nanmax(values)
                    relative = (v - vmin) / (vmax - vmin + 1e-30)
                    text_color = "white" if relative > 0.6 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, color=text_color)


def make_transition_png(
    transitions: dict,
    outpath: str | Path,
    dpi: int = 300,
    config: dict | None = None,
) -> Path:
    """Save transition heatmap PNG(s) for the given transition result.

    Parameters
    ----------
    transitions:
        Result dict returned by ``analyze_grouped_transitions``.  Must contain
        the standard section keys (``per_group_counts``, ``averaged_probabilities``,
        etc.) and ``pair_name``.
    outpath:
        Destination file path.  Parent directory is created if needed.
    dpi:
        Output resolution in dots per inch.
    config:
        Optional dict with keys from ``transitions_plot:`` in
        ``configs/default.yaml``.  Missing keys fall back to defaults.

    Returns
    -------
    Path
        Resolved path of the saved PNG.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    pair_name = transitions.get("pair_name", transitions.get("definition", "unknown"))

    tp_cfg = (config or {}).get("transitions_plot", {})
    max_annotate: int = int(tp_cfg.get("max_annotate_states", 10))
    cmap_counts: str = str(tp_cfg.get("colormap_counts", "Blues"))
    cmap_probs: str = str(tp_cfg.get("colormap_probs", "viridis"))
    cmap_rates: str = str(tp_cfg.get("colormap_rates", "plasma"))
    cmap_barriers: str = str(tp_cfg.get("colormap_barriers", "YlOrRd"))
    barrier_unit: str = str(
        transitions.get(
            "barrier_energy_unit",
            (config or {}).get("transitions", {}).get("energy_unit", "eV"),
        )
    )

    counts_df, group_label = _select_transition_matrix(
        transitions, "averaged_counts", "per_group_counts"
    )
    probs_df, _ = _select_transition_matrix(
        transitions, "averaged_probabilities", "per_group_probabilities"
    )
    rates_df, _ = _select_transition_matrix(
        transitions, "averaged_rates", "per_group_rates"
    )
    barriers_df, _ = _select_transition_matrix(
        transitions, "averaged_barriers", "per_group_barriers"
    )

    panels = []
    if counts_df is not None:
        panels.append((counts_df, cmap_counts, "Count"))
    if probs_df is not None:
        panels.append((probs_df, cmap_probs, "Probability"))
    if rates_df is not None:
        panels.append((rates_df, cmap_rates, "Rate (s⁻¹)"))
    if barriers_df is not None:
        panels.append(
            (
                barriers_df,
                cmap_barriers,
                f"Activation barrier ({barrier_unit})",
            )
        )

    n_panels = max(len(panels), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (matrix, colormap, label) in zip(axes, panels):
        ax.set_title(label)
        _draw_transition_heatmap(ax, matrix, colormap, label, fig, max_annotate)

    title = f"{pair_name} transitions"
    if group_label:
        title += f" — {group_label}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)

    return outpath
