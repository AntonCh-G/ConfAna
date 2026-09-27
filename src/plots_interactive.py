"""Standalone interactive HTML output for conformer density analysis.

Generates self-contained HTML files using Plotly.  Each file embeds a 2D
density heatmap and a single JSON "page data" block (schema below) so that
clicking a bin displays frame metadata (and, optionally, a 3D structure via
3Dmol.js) directly in the browser without a Python server.

When ``interactive.embed_xyz_payload: true`` is set in the config, one
representative XYZ structure per occupied bin is embedded using
``src.viewer.build_bin_xyz_payloads``.  Clicking bins then opens aligned
structures in a browser-native side panel with one 3Dmol.js viewer per open
structure card.

By default the output is fully offline: Plotly is inlined
(``include_plotlyjs: true``) and 3Dmol.js is inlined from a vendored copy
shipped under ``src/interactive_assets/vendor/`` (``include_3dmol: inline``).
Both can be switched to ``"cdn"`` for a smaller file when internet access is
guaranteed.

Page-data JSON schema (embedded as ``<script id="page-data">``)
-----------------------------------------------------------------
::

    {
      "schema_version": 1,
      "pair": {"name", "x_col", "y_col", "x_label", "y_label", "title"},
      "header": {"frame_count", "bin_count_x", "bin_count_y", "scale_mode_label"},
      "axis_spec": {"x_col", "y_col"},
      "bin_geometry": {"x_min", "y_min", "bin_w", "bin_h",
                        "n_bins_x", "n_bins_y"} | null,
      "bin_frame_metadata": {"<xi>_<yi>": {...}} | null,
      "bin_xyz_payloads": {"<xi>_<yi>": "<xyz text>"} | null,
      "frame_metadata": [{...}, ...] | null,
      "ui_state": {
        "theme": "auto" | "light" | "dark",
        "scale_mode": null,               # seam for Slice 4
        "state_overlay_visible": false,   # seam for Slice 5
        "temperature": null,              # seam for Slice 4
        "unit": null,                     # seam for Slice 4
        "pinned_bins": []                 # seam for later slices
      }
    }

``bin_geometry``/``bin_frame_metadata``/``bin_xyz_payloads`` are populated
together (``embed_xyz_payload: true``) and mutually exclusive with
``frame_metadata`` (``embed_xyz_payload: false``); the client derives which
mode is active from ``bin_geometry !== null``.

Public API
----------
- ``make_density_interactive``
- ``render_density_page``
"""

from __future__ import annotations

import html
import importlib.resources
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd

from src.density import compute_2d_histogram
from src.models import CoordinatePair

# ---------------------------------------------------------------------------
# Metadata columns included in the embedded JSON
# ---------------------------------------------------------------------------

_BASE_META_COLUMNS = [
    "frame_id",
    "source_file",
    "trajectory_id",
    "bead_id",
    "frame_number",
    "byte_offset",
    "atom_count",
    "comment_line",
    "local_frame_index",
    "global_frame_index",
    "energy",
]

# ---------------------------------------------------------------------------
# Bundled page assets (template, CSS, JS, vendored 3Dmol.js)
# ---------------------------------------------------------------------------

_ASSETS_PACKAGE = "src.interactive_assets"

# Kept in sync with src/interactive_assets/vendor/README.txt.
_VENDORED_3DMOL_VERSION = "2.5.5"
_VENDORED_3DMOL_CDN_URL = (
    f"https://cdn.jsdelivr.net/npm/3dmol@{_VENDORED_3DMOL_VERSION}/build/3Dmol-min.js"
)

# Initial Plotly colours baked in at build time so the file is never blank
# before viewer.js's applyTheme() runs; mirrors the CSS tokens in viewer.css.
# "auto" bakes the light set (CSS's own unqueried default) — if the browser
# actually prefers dark, applyTheme() corrects it immediately after first
# paint. Only an explicit theme: dark config bakes the dark set up front.
_PLOTLY_THEME_COLORS = {
    "light": {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font_color": "#1a1d21",
        "grid_color": "#d8dce1",
    },
    "dark": {
        "paper_bgcolor": "#1c2024",
        "plot_bgcolor": "#1c2024",
        "font_color": "#e7e9ec",
        "grid_color": "#33393f",
    },
}


def _load_asset(name: str) -> str:
    """Read a bundled interactive-viewer asset (template/CSS/JS/vendor file) as text."""
    return importlib.resources.files(_ASSETS_PACKAGE).joinpath(name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _resolve_interactive_config(cfg: dict) -> dict:
    """Return ``plots.interactive`` settings with Slice-1 defaults applied."""
    interactive_cfg = (cfg.get("plots", {}) or {}).get("interactive", {}) or {}
    return {
        "include_plotlyjs": interactive_cfg.get("include_plotlyjs", True),
        "include_3dmol": interactive_cfg.get("include_3dmol", "inline"),
        "embed_xyz_payload": bool(interactive_cfg.get("embed_xyz_payload", False)),
        "theme": interactive_cfg.get("theme", "auto"),
        "alignment": interactive_cfg.get("alignment"),
    }


def _build_frame_metadata_records(
    df: pd.DataFrame,
    extra_columns: list[str] | None = None,
) -> list[dict]:
    """Return per-frame metadata records for the embedded page-data payload.

    Only columns in ``_BASE_META_COLUMNS`` plus any requested *extra_columns*
    that are present in *df* are included. Numpy/Pandas NA values become
    ``None`` so they serialise as JSON ``null``.
    """
    cols = [c for c in _BASE_META_COLUMNS if c in df.columns]
    # Automatically include any *_shifted companion columns added by
    # apply_coordinate_shifts so they appear in click-point metadata panels.
    for column_name in df.columns:
        if column_name.endswith("_shifted") and column_name not in cols:
            cols.append(column_name)
    for column_name in extra_columns or []:
        if column_name in df.columns and column_name not in cols:
            cols.append(column_name)
    records = []
    for row in df[cols].itertuples(index=False):
        rec: dict = {}
        for col, val in zip(cols, row):
            if pd.isna(val) if not isinstance(val, str) else False:
                rec[col] = None
            else:
                if isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    rec[col] = float(val)
                else:
                    rec[col] = val
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Page builder
# ---------------------------------------------------------------------------


def render_density_page(page_data: dict) -> str:
    """Assemble the standalone interactive HTML page from *page_data*.

    Pure function of its input: reads only this package's own bundled assets
    (via ``importlib.resources``) and the values inside *page_data* — no
    caller-supplied file I/O, no geometry, no randomness, no timestamps.
    Identical *page_data* always yields a byte-identical string.

    *page_data* holds the JSON-embeddable page state (schema documented in
    this module's docstring) plus two reserved, non-JSON keys:

    - ``"plot_html"``: the raw Plotly fragment from
      ``fig.to_html(full_html=False, ...)``.
    - ``"include_3dmol"``: ``"inline"`` or ``"cdn"`` — controls the 3Dmol
      script tag. Only consulted when ``page_data["bin_geometry"]`` is not
      ``None`` (3Dmol is never needed in per-frame metadata-only mode).
    """
    reserved = {"plot_html", "include_3dmol"}
    state = {k: v for k, v in page_data.items() if k not in reserved}
    page_data_json = json.dumps(state, sort_keys=True, allow_nan=False, default=str)

    threedmol_tag = ""
    if state.get("bin_geometry") is not None:
        include_3dmol = page_data.get("include_3dmol", "inline")
        if include_3dmol == "inline":
            threedmol_tag = f"<script>{_load_asset('vendor/3Dmol-min.js')}</script>"
        elif include_3dmol == "cdn":
            threedmol_tag = f'<script src="{_VENDORED_3DMOL_CDN_URL}"></script>'
        else:
            raise ValueError(
                f"Unsupported include_3dmol: {include_3dmol!r}. Valid: 'inline', 'cdn'."
            )

    pair_info = state.get("pair") or {}
    template = string.Template(_load_asset("page.html"))
    return template.substitute(
        html_title=html.escape(str(pair_info.get("title", "ConfAna"))),
        page_data_json=page_data_json,
        plot_fragment=page_data.get("plot_html", ""),
        threedmol_tag=threedmol_tag,
        inline_css=_load_asset("viewer.css"),
        inline_js=_load_asset("viewer.js"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_density_interactive(
    df: pd.DataFrame,
    pair: CoordinatePair,
    outpath: str | Path,
    config: dict | None = None,
) -> Path:
    """Save a standalone interactive HTML density plot.

    The output is a self-contained HTML file with an embedded Plotly heatmap
    and a JSON page-data block. Clicking bins opens one card per selected
    structure in a right-hand side panel; double-clicking clears the panel.
    No Python server is required, and (by default) no internet connection is
    required either.

    When ``interactive.embed_xyz_payload: true`` is set in the config, one
    representative XYZ structure per occupied bin is also embedded. Clicking a
    bin then renders the aligned structure in a dedicated 3Dmol.js viewer
    inside that card.

    Parameters
    ----------
    df:
        Standard coordinate table containing the DoF columns for ``pair``
        and any optional metadata columns (source_file, byte_offset, etc.).
    pair:
        :class:`~src.models.CoordinatePair` specifying which columns to plot.
        Domain, labels, and per-pair overrides (bins, colormap, log_scale,
        x_range, y_range) are read from the pair object.
    outpath:
        Destination file path (.html). Parent directory is created if needed.
    config:
        Optional config dict. Accepts either the full project config or just
        the ``density:`` sub-section. Controls bins, axis ranges, log_scale,
        ``interactive.include_plotlyjs``, ``interactive.include_3dmol``,
        ``interactive.embed_xyz_payload``, ``interactive.theme``, and
        ``interactive.alignment``.

    Returns
    -------
    Path
        Resolved path of the saved HTML file.

    Raises
    ------
    ValueError
        If the feature columns are not found in ``df``.
    """
    import plotly.graph_objects as go  # noqa: PLC0415

    cfg = config or {}
    plots_cfg = cfg.get("plots", {}) or {}
    _, density_cfg = (
        (cfg, plots_cfg.get("density", {}) or {}) if "density" in plots_cfg else ({}, cfg)
    )

    # Per-pair overrides take precedence over global density config.
    bins: int = pair.bins if pair.bins is not None else int(density_cfg.get("bins", 180))
    colorscale: str = (
        pair.colormap if pair.colormap is not None else str(density_cfg.get("colormap", "Viridis"))
    )
    log_scale: bool = (
        pair.log_scale if pair.log_scale is not None else bool(density_cfg.get("log_scale", True))
    )

    # Axis ranges: per-pair override → global config → pair domain
    raw_x_range = density_cfg.get("x_range")
    raw_y_range = density_cfg.get("y_range")
    x_range: tuple[float, float] = (
        pair.x_range if pair.x_range is not None
        else (tuple(raw_x_range) if raw_x_range is not None else pair.x_domain)  # type: ignore[assignment]
    )
    y_range: tuple[float, float] = (
        pair.y_range if pair.y_range is not None
        else (tuple(raw_y_range) if raw_y_range is not None else pair.y_domain)  # type: ignore[assignment]
    )

    x_col, y_col = pair.feature_columns
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(
                f"make_density_interactive: column '{col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    H, x_edges, y_edges = compute_2d_histogram(
        df,
        x_col=x_col,
        y_col=y_col,
        bins=bins,
        x_range=x_range,
        y_range=y_range,
    )

    # Convert counts to display values (log or raw); mask zeros as NaN
    Z = H.T.astype(float)
    Z[Z == 0.0] = np.nan
    if log_scale:
        Z = np.log10(Z + 1.0)
    colorbar_title = "log₁₀(count+1)" if log_scale else "count"

    # Bin centres for hover / click coordinates
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1

    heatmap = go.Heatmap(
        z=Z,
        x=x_centres,
        y=y_centres,
        colorscale=colorscale,
        colorbar={"title": colorbar_title},
        hovertemplate=(
            f"{pair.x_label}: %{{x:.1f}}<br>"
            f"{pair.y_label}: %{{y:.1f}}<br>"
            f"{colorbar_title}: %{{z:.3f}}<extra></extra>"
        ),
    )

    interactive_cfg = _resolve_interactive_config(cfg)
    theme = interactive_cfg["theme"]
    plotly_theme = _PLOTLY_THEME_COLORS["dark" if theme == "dark" else "light"]

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=pair.title,
        xaxis_title=pair.x_label,
        yaxis_title=pair.y_label,
        xaxis={"range": list(x_range), "gridcolor": plotly_theme["grid_color"]},
        yaxis={"range": list(y_range), "gridcolor": plotly_theme["grid_color"]},
        width=700,
        height=600,
        paper_bgcolor=plotly_theme["paper_bgcolor"],
        plot_bgcolor=plotly_theme["plot_bgcolor"],
        font={"color": plotly_theme["font_color"]},
    )

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=interactive_cfg["include_plotlyjs"],
    )

    embed_xyz = interactive_cfg["embed_xyz_payload"]
    bin_geometry: dict | None = None
    bin_frame_metadata: dict | None = None
    bin_xyz_payloads: dict | None = None
    frame_metadata: list[dict] | None = None

    if embed_xyz:
        # Per-bin mode: embed one metadata record and one XYZ payload per occupied
        # bin instead of the full per-frame table.
        from src.viewer import build_bin_frame_metadata, build_bin_xyz_payloads  # noqa: PLC0415

        bin_xyz_payloads = build_bin_xyz_payloads(
            df, x_col=x_col, y_col=y_col,
            x_edges=x_edges, y_edges=y_edges,
            alignment=interactive_cfg["alignment"],
        )
        bin_frame_metadata = build_bin_frame_metadata(
            df, x_col=x_col, y_col=y_col,
            x_edges=x_edges, y_edges=y_edges,
            extra_fields=[x_col, y_col],
        )
        bin_geometry = {
            "x_min": float(x_edges[0]),
            "y_min": float(y_edges[0]),
            "bin_w": float(x_edges[1] - x_edges[0]),
            "bin_h": float(y_edges[1] - y_edges[0]),
            "n_bins_x": n_bins_x,
            "n_bins_y": n_bins_y,
        }
    else:
        # Per-frame mode: embed the full coordinate table so the JavaScript can
        # find the nearest frame to any click point. Practical for small datasets.
        frame_metadata = _build_frame_metadata_records(df, extra_columns=[x_col, y_col])

    page_data = {
        "schema_version": 1,
        "pair": {
            "name": pair.name,
            "x_col": x_col,
            "y_col": y_col,
            "x_label": pair.x_label,
            "y_label": pair.y_label,
            "title": pair.title,
        },
        "header": {
            "frame_count": int(len(df)),
            "bin_count_x": n_bins_x,
            "bin_count_y": n_bins_y,
            "scale_mode_label": colorbar_title,
        },
        "axis_spec": {"x_col": x_col, "y_col": y_col},
        "bin_geometry": bin_geometry,
        "bin_frame_metadata": bin_frame_metadata,
        "bin_xyz_payloads": bin_xyz_payloads,
        "frame_metadata": frame_metadata,
        "ui_state": {
            "theme": theme,
            "scale_mode": None,
            "state_overlay_visible": False,
            "temperature": None,
            "unit": None,
            "pinned_bins": [],
        },
        "plot_html": plot_html,
        "include_3dmol": interactive_cfg["include_3dmol"],
    }

    html_str = render_density_page(page_data)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html_str, encoding="utf-8")

    return outpath
