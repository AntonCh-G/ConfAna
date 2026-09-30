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
from typing import Sequence

import numpy as np
import pandas as pd

from confana.density import ConformationalMap, DensitySettings, build_conformational_map
from confana.models import CoordinatePair
from confana.states import build_bin_state_overlay, resolve_state_groupby


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
        :class:`~confana.models.CoordinatePair` specifying which columns to plot.
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
        If the feature columns are not found in ``df``, the density bins or
        ranges are invalid, or ``clustering.groupby`` names a column missing
        from ``df`` while the pair's state column is present.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    conf_map = build_conformational_map(df, pair, DensitySettings.for_pair(pair, config))
    fig = _density_figure(conf_map, groupby=resolve_state_groupby(config or {}), overlays=overlays)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


def density_png_bytes(
    conf_map: ConformationalMap,
    dpi: int,
    groupby: Sequence[str] | None = None,
) -> bytes:
    """Return the density figure of :func:`make_density_png` for *conf_map* as PNG bytes.

    Same figure (bins, ranges, colour map, state markers), rendered in memory
    at *dpi*; nothing is written to disk. *groupby* names the columns states
    were clustered by (see :func:`~confana.states.resolve_state_groupby`).
    """
    import io  # noqa: PLC0415

    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig = _density_figure(conf_map, groupby=groupby)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    return buffer.getvalue()


def _density_figure(
    conf_map: ConformationalMap,
    groupby: Sequence[str] | None = None,
    overlays: list[dict] | None = None,
):
    """Draw the density figure shared by :func:`make_density_png` and
    :func:`density_png_bytes`; the caller saves and closes it.

    State markers are the state centres of the first clustering group, the
    group the interactive page starts on; with several groups the title names
    it, since state labels are not shared between groups.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    pair, settings = conf_map.pair, conf_map.settings
    x_col, y_col = pair.feature_columns

    C = _prepare_density_values(conf_map.counts, log_scale=settings.log_scale)
    cbar_label = "log₁₀(count + 1)" if settings.log_scale else "count"

    fig, ax = plt.subplots(figsize=(6, 5))
    colormap = _prepare_density_colormap(settings.colormap)

    pcm = ax.pcolormesh(
        conf_map.x_edges,
        conf_map.y_edges,
        C,
        cmap=colormap,
        shading="flat",
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(pair.x_label)
    ax.set_ylabel(pair.y_label)
    ax.set_title(pair.title)
    ax.set_xlim(settings.x_range)
    ax.set_ylim(settings.y_range)

    states = build_bin_state_overlay(conf_map, groupby=groupby)
    if states is not None and states["groups"]:
        group = states["groups"][0]
        if len(states["groups"]) > 1:
            ax.set_title(f"{pair.title} — {group['name']}")
        centres = group["centres"]
        if centres:
            ax.scatter(
                [c["x"] for c in centres],
                [c["y"] for c in centres],
                c="red",
                s=80,
                zorder=5,
                marker="o",
                linewidths=0.5,
                edgecolors="white",
            )
            for centre in centres:
                ax.annotate(
                    states["labels"][centre["state"]],
                    (centre["x"], centre["y"]),
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
            ov_size = ov.get("size") or 4
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
                marker="x",
                s=ov_size,
                color=ov_color,
                label=ov_label,
                zorder=4,
                linewidths=0.5,
                rasterized=True,
            )
        ax.legend(loc="upper right", fontsize=7, markerscale=3,
                  framealpha=0.7, handletextpad=0.4)

    fig.tight_layout()
    return fig


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

    tp_cfg = ((config or {}).get("plots", {}) or {}).get("transitions", {})
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
