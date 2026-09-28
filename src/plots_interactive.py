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
      "scale": {                          # colour-scale modes (Slice 4)
        "grids": {                        # rows = y bins, columns = x bins;
          "counts": [[int | null]],       # null = unsampled bin (blank)
          "log_counts": [[float | null]], # log10(count + 1)
          "free_energy": [[float | null]] # dimensionless -ln(P / P_max)
        },
        "modes": {"<mode>": {"label", "value_label", "z_format", "subtitle"}},
        "colorscales": {"light": [[pos, color], ...], "dark": [...]},
        "energy_units": [{"key", "label", "k_B"}, ...]   # from src.units
      } | null,
      "axis_spec": {"x_col", "y_col"},
      "bin_geometry": {"x_min", "y_min", "bin_w", "bin_h",
                        "n_bins_x", "n_bins_y"} | null,
      "bin_frame_metadata": {"<xi>_<yi>": {...}} | null,
      "bin_xyz_payloads": {"<xi>_<yi>": "<xyz text>"} | null,
      "frame_metadata": [{...}, ...] | null,
      "states": {                         # null when the state column is absent
        "state_col", "groupby": [str, ...], "labels": [str, ...],
        "colors": [str, ...],             # one per label
        "groups": [{"name", "keys", "bins": [int], "states": [int],
                    "centres": [{"state", "x", "y", "frames"}]}, ...]
      } | null,
      "axis_ticks": {"steps": [float, ...], "max_intervals": int,
                     "x": bool, "y": bool},  # true = degree ticks on that axis
      "axis_atoms": {                     # null when highlight_dof_atoms: false
        "x": {"name", "type", "atoms": [int, ...] | null},
        "y": {"name", "type", "atoms": [int, ...] | null}
      } | null,
      "settings": {
        "hover_preview": true | false,    # live hover preview in the side panel
        "max_pinned": 15                  # pinned cards; oldest evicted beyond this
      },
      "ui_state": {
        "theme": "auto" | "light" | "dark",
        "scale_mode": "log_counts" | "counts" | "free_energy",
        "state_overlay_visible": true | false,
        "state_group": 0,                 # index into states.groups
        "temperature": float | null,      # K; null = none known yet
        "unit": "kT" | "kJ/mol" | ...,    # key of scale.energy_units
        "pinned_bins": []                 # seam for later slices
      }
    }

``bin_geometry``/``bin_frame_metadata``/``bin_xyz_payloads`` are populated
together (``embed_xyz_payload: true``) and mutually exclusive with
``frame_metadata`` (``embed_xyz_payload: false``); the client derives which
mode is active from ``bin_geometry !== null``.

``axis_atoms`` lists the atoms that define each axis, taken from the pair's
DoF definitions (``CoordinatePair.x_atoms`` / ``y_atoms``). Indices are
0-based file indices, as in config; ``name`` is the DoF column and ``type``
the DoF type. ``atoms`` is null for DoF types without atoms (collective,
external). The 3D views colour these atoms and the axis titles to match.

``scale`` holds every colour-scale mode, so the page switches modes, the
temperature and the energy unit without re-running the pipeline. In
free-energy mode the map shows ``F = k_B T · free_energy`` in
``ui_state.unit`` (``kT`` = dimensionless, ignores the temperature).
``modes.free_energy.value_label`` is null: the page builds it from the unit.
``colorscales`` holds the map's colour scale for each theme; the page swaps
them when the theme changes (see ``_theme_colorscale``).

``axis_ticks`` marks the axes measured in degrees. Those get ticks on
multiples of the smallest step in ``steps`` that leaves at most
``max_intervals`` intervals across the visible range; the page re-picks the
step after each zoom or pan (see ``_degree_tick_step``).

``states`` is the per-bin majority state from ``src.states.build_bin_state_overlay``,
one map per clustering group (``clustering.groupby``): labels are not
unified across groups, so the page shows one group at a time. ``bins`` are
flat indices ``yi * n_bins_x + xi``; ``states`` index ``labels``.

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

from collections.abc import Iterable

from src.density import compute_2d_histogram, population_free_energy
from src.models import CoordinatePair
from src.states import build_bin_state_overlay, resolve_state_groupby
from src.units import (
    energy_unit_table,
    thermal_energy,
    unit_label,
    validate_energy_unit,
    validate_temperature,
)

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
# Colour-scale modes
# ---------------------------------------------------------------------------

_DENSITY_SUBTITLE = "Coordinate-density landscape — not a potential energy surface"

# Embedded as page data "scale.modes", so the page's labels come from here.
# The free-energy value label depends on the unit and is built by
# _scale_value_label (and the same way in viewer.js).
_SCALE_MODES: dict[str, dict] = {
    "log_counts": {
        "label": "log counts",
        "value_label": "log₁₀(count+1)",
        "z_format": ".3f",
        "subtitle": _DENSITY_SUBTITLE,
    },
    "counts": {
        "label": "counts",
        "value_label": "count",
        "z_format": ".0f",
        "subtitle": _DENSITY_SUBTITLE,
    },
    "free_energy": {
        "label": "free-energy-like",
        "value_label": None,
        "z_format": ".3f",
        "subtitle": "Population-derived free-energy-like surface, not a potential energy surface",
    },
}

# Decimals kept in the embedded grids (counts are exact integers).
_GRID_DECIMALS = 4


def _scale_value_label(mode: str, unit: str) -> str:
    """Colourbar / read-out label for *mode* (mirrors valueLabel in viewer.js)."""
    if mode == "free_energy":
        return f"F ({unit_label(unit)})"
    return _SCALE_MODES[mode]["value_label"]


def _hovertemplate(x_label: str, y_label: str, value_label: str, z_format: str) -> str:
    """Heatmap hover template (mirrors hoverTemplate in viewer.js)."""
    return (
        f"{x_label}: %{{x:.1f}}<br>"
        f"{y_label}: %{{y:.1f}}<br>"
        f"{value_label}: %{{z:{z_format}}}<extra></extra>"
    )


def _scale_grids(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Return the display grids for every scale mode; unsampled bins are NaN.

    *counts* is oriented like the heatmap's z (rows = y bins). Values are
    rounded as embedded, so the initial figure and the page use equal numbers.
    """
    counts = np.asarray(counts, dtype=float)
    sampled = counts > 0
    log_counts = np.full(counts.shape, np.nan)
    log_counts[sampled] = np.log10(counts[sampled] + 1.0)
    return {
        "counts": np.where(sampled, counts, np.nan),
        "log_counts": np.round(log_counts, _GRID_DECIMALS),
        "free_energy": np.round(population_free_energy(counts), _GRID_DECIMALS),
    }


def _grid_to_json(grid: np.ndarray, integer: bool = False) -> list[list]:
    """Nested lists for JSON: NaN → None (JSON null), optionally as ints."""
    cast = int if integer else float
    return [[None if np.isnan(v) else cast(v) for v in row] for row in grid.tolist()]


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
        "axis_x_color": "#d55e00",
        "axis_y_color": "#0072b2",
        "state_label_bg": "rgba(255, 255, 255, 0.75)",
    },
    "dark": {
        "paper_bgcolor": "#1c2024",
        "plot_bgcolor": "#1c2024",
        "font_color": "#e7e9ec",
        "grid_color": "#33393f",
        "axis_x_color": "#f0894a",
        "axis_y_color": "#56b4e9",
        "state_label_bg": "rgba(28, 32, 36, 0.75)",
    },
}


# ---------------------------------------------------------------------------
# Theme colour scales for the density map
# ---------------------------------------------------------------------------

# Minimum contrast (WCAG ratio) between each end of the map's colour scale and
# the theme's plot background. Light: only near-white ends are trimmed, so
# standard viridis stays exactly viridis. Dark: ends that sink into the dark
# background are trimmed. The direction is kept, so bright means the same in
# both themes.
_THEME_MIN_CONTRAST = {"light": 1.25, "dark": 2.0}
_TRIMMED_COLORSCALE_STOPS = 21


def _rgb(color: str) -> tuple[float, float, float]:
    """``#rrggbb`` or ``rgb(r, g, b)`` → 0–255 floats."""
    import plotly.colors as pc  # noqa: PLC0415

    if color.startswith("#"):
        return tuple(float(v) for v in pc.hex_to_rgb(color))  # type: ignore[return-value]
    return tuple(float(v) for v in pc.unlabel_rgb(color))  # type: ignore[return-value]


def _contrast_ratio(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """WCAG contrast ratio of two 0–255 RGB colours (1 = identical, 21 = max)."""

    def luminance(rgb: tuple[float, float, float]) -> float:
        channels = [v / 255.0 for v in rgb]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _named_colorscale(name: object, source: str) -> list[list]:
    """Plotly colour scale *name* (``_r`` = reversed) as ``[[pos, color], ...]``.

    Raises
    ------
    ValueError
        If Plotly does not know *name*; *source* names the config key.
    """
    import plotly.colors as pc  # noqa: PLC0415
    from _plotly_utils.exceptions import PlotlyError  # noqa: PLC0415

    try:
        scale = pc.get_colorscale(str(name))
    except PlotlyError as exc:
        raise ValueError(f"{source}: unknown Plotly colour scale {name!r}.") from exc
    return [[float(pos), color] for pos, color in scale]


def _theme_colorscale(name: str, theme: str) -> list[list]:
    """Colour scale *name* with any end that blends into *theme*'s background trimmed.

    Only the two ends are trimmed, to the first / last point (of 101 samples)
    that reaches ``_THEME_MIN_CONTRAST[theme]`` against the plot background.
    A scale that needs no trimming is returned unchanged.
    """
    import plotly.colors as pc  # noqa: PLC0415

    scale = _named_colorscale(name, "density colormap")
    background = _rgb(_PLOTLY_THEME_COLORS[theme]["plot_bgcolor"])
    samples = np.linspace(0.0, 1.0, 101)
    visible = [
        _contrast_ratio(_rgb(color), background) >= _THEME_MIN_CONTRAST[theme]
        for color in pc.sample_colorscale(scale, samples)
    ]
    if not any(visible):
        return scale
    # Only the ends matter: a pale middle (e.g. jet's yellow) is kept.
    first = visible.index(True)
    last = len(visible) - 1 - visible[::-1].index(True)
    if (first, last) == (0, len(visible) - 1) or first >= last:
        return scale
    kept = np.linspace(samples[first], samples[last], _TRIMMED_COLORSCALE_STOPS)
    colors = pc.sample_colorscale(scale, kept)
    n = _TRIMMED_COLORSCALE_STOPS - 1
    return [[i / n, color] for i, color in enumerate(colors)]


# ---------------------------------------------------------------------------
# Degree axis ticks
# ---------------------------------------------------------------------------

# Tick steps that read naturally in degrees; embedded so the page picks
# steps the same way after a zoom.
_DEGREE_TICK_STEPS = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0)
_DEGREE_TICK_MAX_INTERVALS = 6
_DEGREE_DOF_TYPES = frozenset({"dihedral", "angle"})


def _is_degree_axis(dof_type: str | None, label: str) -> bool:
    """True for dihedral / angle DoFs; without a DoF type, when the label says (°)."""
    if dof_type is not None:
        return dof_type in _DEGREE_DOF_TYPES
    return "(°)" in label


def _degree_tick_step(span: float) -> float:
    """Smallest degree-friendly step giving at most _DEGREE_TICK_MAX_INTERVALS
    intervals over *span* (mirrors degreeTickStep in viewer.js)."""
    for step in _DEGREE_TICK_STEPS:
        if abs(span) / step <= _DEGREE_TICK_MAX_INTERVALS:
            return step
    return _DEGREE_TICK_STEPS[-1]


def _degree_axis_ticks(axis_range: tuple[float, float]) -> dict:
    """Plotly axis settings for degree ticks over *axis_range*."""
    return {
        "tickmode": "linear",
        "tick0": 0,
        "dtick": _degree_tick_step(axis_range[1] - axis_range[0]),
        "ticksuffix": "°",
    }


# ---------------------------------------------------------------------------
# State overlay
# ---------------------------------------------------------------------------

# Categorical state colours (Plotly's D3 set); they repeat after ten states.
_STATE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
# A tint over the density, not a replacement for it.
_STATE_OPACITY = 0.35

_STATE_CONTROLS_HTML = (
    '<span class="ca-state-controls">'
    '<button id="states-toggle" type="button" class="ca-btn" aria-pressed="false">States</button>'
    '<select id="state-group" aria-label="State group" hidden></select>'
    "</span>"
)


def _state_colorscale(n_labels: int) -> list[list]:
    """Stepped colourscale: z = label index i maps to _STATE_COLORS[i]."""
    if n_labels == 0:
        return [[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,0)"]]
    stops = []
    for i in range(n_labels):
        color = _STATE_COLORS[i % len(_STATE_COLORS)]
        stops += [[i / n_labels, color], [(i + 1) / n_labels, color]]
    return stops


def _state_grid(group: dict, n_bins_y: int, n_bins_x: int) -> np.ndarray:
    """Dense label-index grid of one group (rows = y bins); NaN = no state."""
    grid = np.full(n_bins_y * n_bins_x, np.nan)
    grid[np.asarray(group["bins"], dtype=np.int64)] = group["states"]
    return grid.reshape(n_bins_y, n_bins_x)


def _state_annotations(states: dict, group: dict, visible: bool, bgcolor: str) -> list[dict]:
    """State-name labels at each state's centre (mirrors stateAnnotations in viewer.js)."""
    return [
        {
            "x": centre["x"],
            "y": centre["y"],
            "text": states["labels"][centre["state"]],
            "showarrow": False,
            "font": {"size": 12},
            "bgcolor": bgcolor,
            "bordercolor": states["colors"][centre["state"]],
            "borderwidth": 1,
            "borderpad": 2,
            "visible": visible,
        }
        for centre in group["centres"]
    ]


def _load_asset(name: str) -> str:
    """Read a bundled interactive-viewer asset (template/CSS/JS/vendor file) as text."""
    return importlib.resources.files(_ASSETS_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _inline_asset(name: str, element: str) -> str:
    """Load a bundled asset to be inlined inside a ``<script>`` or ``<style>`` element.

    Browsers end the element at the first closing tag, even inside a JS
    comment or string, so such an asset would silently cut the page's code off.

    Raises
    ------
    ValueError
        If the asset contains ``</script`` / ``</style`` (matching *element*).
    """
    text = _load_asset(name)
    if f"</{element}" in text.lower():
        raise ValueError(
            f"Bundled asset {name} contains '</{element}', which would end its inline "
            f"<{element}> element early; remove it from the file."
        )
    return text


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _resolve_interactive_config(cfg: dict) -> dict:
    """Return ``plots.interactive`` settings with defaults applied.

    Raises
    ------
    ValueError
        If ``hover_preview``, ``highlight_dof_atoms`` or ``show_states`` is not a bool,
        ``max_pinned`` is not an integer >= 1, ``default_scale`` is not
        a known scale mode, or ``free_energy`` / ``theme_colorscales`` is
        not a mapping.
    """
    interactive_cfg = (cfg.get("plots", {}) or {}).get("interactive", {}) or {}

    flags = {}
    for key, default in (("hover_preview", True), ("highlight_dof_atoms", True), ("show_states", False)):
        value = interactive_cfg.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"plots.interactive.{key} must be true or false, got {value!r}.")
        flags[key] = value

    max_pinned = interactive_cfg.get("max_pinned", 15)
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(max_pinned, bool) or not isinstance(max_pinned, int) or max_pinned < 1:
        raise ValueError(
            f"plots.interactive.max_pinned must be an integer >= 1, got {max_pinned!r}."
        )

    # None = follow the density log_scale setting (log_counts or counts).
    default_scale = interactive_cfg.get("default_scale")
    if default_scale is not None and default_scale not in _SCALE_MODES:
        raise ValueError(
            f"plots.interactive.default_scale must be one of {list(_SCALE_MODES)}, "
            f"got {default_scale!r}."
        )

    free_energy = interactive_cfg.get("free_energy") or {}
    if not isinstance(free_energy, dict):
        raise ValueError(
            "plots.interactive.free_energy must be a mapping with 'temperature' and/or "
            f"'unit', got {free_energy!r}."
        )

    theme_colorscales = interactive_cfg.get("theme_colorscales") or {}
    if not isinstance(theme_colorscales, dict) or set(theme_colorscales) - {"light", "dark"}:
        raise ValueError(
            "plots.interactive.theme_colorscales must be a mapping with 'light' and/or "
            f"'dark' keys, got {theme_colorscales!r}."
        )

    return {
        "include_plotlyjs": interactive_cfg.get("include_plotlyjs", True),
        "include_3dmol": interactive_cfg.get("include_3dmol", "inline"),
        "embed_xyz_payload": bool(interactive_cfg.get("embed_xyz_payload", False)),
        "theme": interactive_cfg.get("theme", "auto"),
        "alignment": interactive_cfg.get("alignment"),
        "hover_preview": flags["hover_preview"],
        "highlight_dof_atoms": flags["highlight_dof_atoms"],
        "show_states": flags["show_states"],
        "max_pinned": max_pinned,
        "default_scale": default_scale,
        "free_energy": free_energy,
        "theme_colorscales": theme_colorscales,
    }


def _resolve_free_energy_defaults(cfg: dict, free_energy_cfg: dict) -> tuple[float | None, str]:
    """Return the page's initial ``(temperature, unit)`` for free-energy mode.

    Each falls back from ``plots.interactive.free_energy.*`` to
    ``transitions.temperature`` / ``transitions.energy_unit``. With no
    temperature from either, the page opens in ``kT`` (dimensionless) with an
    empty temperature field; a configured unit is still validated.

    Raises
    ------
    ValueError
        If the chosen temperature is not a positive number of kelvin, or the
        chosen unit is not one of :data:`src.units.ENERGY_UNITS`.
    """
    transitions_cfg = cfg.get("transitions", {}) or {}

    temperature = free_energy_cfg.get("temperature")
    t_source = "plots.interactive.free_energy.temperature"
    if temperature is None:
        temperature = transitions_cfg.get("temperature")
        t_source = "transitions.temperature"
    if temperature is not None:
        temperature = validate_temperature(temperature, source=t_source)

    unit = free_energy_cfg.get("unit")
    u_source = "plots.interactive.free_energy.unit"
    if unit is None:
        unit = transitions_cfg.get("energy_unit")
        u_source = "transitions.energy_unit"
    if unit is None:
        return temperature, "kT"
    unit = validate_energy_unit(unit, source=u_source)
    return temperature, unit if temperature is not None else "kT"


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


def _known_atom_counts(df: pd.DataFrame, bin_xyz_payloads: dict | None) -> set[int]:
    """Return the atom counts of the frames the page describes.

    Taken from the ``atom_count`` column (absent in coordinate-only input) and
    from the first line of each embedded XYZ payload.
    """
    counts: set[int] = set()
    if "atom_count" in df.columns:
        counts.update(int(c) for c in pd.unique(df["atom_count"].dropna()))
    for xyz_text in (bin_xyz_payloads or {}).values():
        counts.add(int(xyz_text.split("\n", 1)[0]))
    return counts


def _build_axis_atoms(pair: CoordinatePair, atom_counts: Iterable[int]) -> dict:
    """Return the ``axis_atoms`` page-data block for *pair*.

    Parameters
    ----------
    pair:
        Coordinate pair; its ``x_atoms`` / ``y_atoms`` are 0-based file indices.
    atom_counts:
        Atom counts of the frames on the page. Every atom index must be
        below the smallest of them. Empty when unknown (coordinate-only
        input without structures); then only negative indices are rejected.

    Raises
    ------
    ValueError
        If any atom index is negative or not below the atom count.
    """
    n_atoms = min(atom_counts, default=None)
    axes: dict = {}
    for axis, name, dof_type, atoms in (
        ("x", pair.x_col, pair.x_dof_type, pair.x_atoms),
        ("y", pair.y_col, pair.y_dof_type, pair.y_atoms),
    ):
        atom_list = [int(a) for a in atoms] if atoms is not None else None
        bad = [
            a for a in atom_list or []
            if a < 0 or (n_atoms is not None and a >= n_atoms)
        ]
        if bad:
            limit = f"0..{n_atoms - 1}" if n_atoms is not None else ">= 0"
            raise ValueError(
                f"Coordinate pair '{pair.name}': {axis}-axis DoF '{name}' uses atom "
                f"indices {bad} outside {limit}. Atom indices are 0-based file indices"
                + (f" and the structures have {n_atoms} atoms." if n_atoms is not None else ".")
            )
        axes[axis] = {"name": name, "type": dof_type, "atoms": atom_list}
    return axes


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
    # "<" only occurs inside JSON strings, so escaping it keeps the JSON valid and
    # stops data such as an xyz comment line from closing the <script> element.
    page_data_json = json.dumps(state, sort_keys=True, allow_nan=False, default=str).replace(
        "<", "\\u003c"
    )

    threedmol_tag = ""
    if state.get("bin_geometry") is not None:
        include_3dmol = page_data.get("include_3dmol", "inline")
        if include_3dmol == "inline":
            threedmol_tag = f"<script>{_inline_asset('vendor/3Dmol-min.js', 'script')}</script>"
        elif include_3dmol == "cdn":
            threedmol_tag = f'<script src="{_VENDORED_3DMOL_CDN_URL}"></script>'
        else:
            raise ValueError(
                f"Unsupported include_3dmol: {include_3dmol!r}. Valid: 'inline', 'cdn'."
            )

    pair_info = state.get("pair") or {}
    scale_modes = (state.get("scale") or {}).get("modes") or {}
    scale_mode = (state.get("ui_state") or {}).get("scale_mode")
    subtitle = (scale_modes.get(scale_mode) or {}).get("subtitle", _DENSITY_SUBTITLE)
    template = string.Template(_load_asset("page.html"))
    return template.substitute(
        html_title=html.escape(str(pair_info.get("title", "ConfAna"))),
        subtitle=html.escape(subtitle),
        page_data_json=page_data_json,
        plot_fragment=page_data.get("plot_html", ""),
        threedmol_tag=threedmol_tag,
        state_controls=_STATE_CONTROLS_HTML if state.get("states") else "",
        inline_css=_inline_asset("viewer.css", "style"),
        inline_js=_inline_asset("viewer.js", "script"),
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
        ``interactive.embed_xyz_payload``, ``interactive.theme``,
        ``interactive.alignment``, ``interactive.hover_preview``,
        ``interactive.highlight_dof_atoms``, ``interactive.max_pinned``,
        ``interactive.default_scale``, ``interactive.free_energy``
        (``temperature`` / ``unit``, falling back to ``transitions.*``) and
        ``interactive.show_states`` and ``interactive.theme_colorscales``
        (``light`` / ``dark`` Plotly scale names; unset = the density
        colormap trimmed per theme). ``clustering.groupby`` names the groups
        the state overlay is split into (one map per group).

    Returns
    -------
    Path
        Resolved path of the saved HTML file.

    Raises
    ------
    ValueError
        If the feature columns are not found in ``df``,
        ``hover_preview`` / ``highlight_dof_atoms`` / ``max_pinned`` are
        invalid, an axis atom index is outside the structures' atom count,
        ``default_scale`` is unknown, the free-energy temperature / unit
        is invalid, ``show_states`` is not a bool, a colour-scale name is
        unknown, or a ``clustering.groupby``
        column is missing while the state column is present.
    """
    import plotly.graph_objects as go  # noqa: PLC0415

    cfg = config or {}
    interactive_cfg = _resolve_interactive_config(cfg)
    temperature, unit = _resolve_free_energy_defaults(cfg, interactive_cfg["free_energy"])
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

    # Every scale mode's grid is embedded; the figure starts in scale_mode.
    scale_mode = interactive_cfg["default_scale"] or ("log_counts" if log_scale else "counts")
    grids = _scale_grids(H.T)
    Z = grids[scale_mode]
    if scale_mode == "free_energy":
        Z = Z * thermal_energy(unit, temperature)
    colorbar_title = _scale_value_label(scale_mode, unit)

    # Bin centres for hover / click coordinates
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1

    # One colour scale per theme: explicit config names are used as given,
    # otherwise the density colormap is trimmed to suit each background.
    colorscales = {}
    for theme_name in ("light", "dark"):
        explicit = interactive_cfg["theme_colorscales"].get(theme_name)
        colorscales[theme_name] = (
            _named_colorscale(explicit, f"plots.interactive.theme_colorscales.{theme_name}")
            if explicit is not None
            else _theme_colorscale(colorscale, theme_name)
        )
    theme = interactive_cfg["theme"]

    heatmap = go.Heatmap(
        z=Z,
        x=x_centres,
        y=y_centres,
        colorscale=colorscales["dark" if theme == "dark" else "light"],
        colorbar={"title": colorbar_title},
        # Hover must fire over empty bins so the preview can say "no frames".
        hoverongaps=True,
        hovertemplate=_hovertemplate(
            pair.x_label, pair.y_label, colorbar_title, _SCALE_MODES[scale_mode]["z_format"]
        ),
    )

    plotly_theme = _PLOTLY_THEME_COLORS["dark" if theme == "dark" else "light"]

    # Axis titles take the colour of their highlighted atoms in the 3D views.
    highlight = interactive_cfg["highlight_dof_atoms"]
    axis_titles = {}
    for axis, label, atoms in (("x", pair.x_label, pair.x_atoms), ("y", pair.y_label, pair.y_atoms)):
        axis_titles[axis] = {"text": label}
        if highlight and atoms:
            axis_titles[axis]["font"] = {"color": plotly_theme[f"axis_{axis}_color"]}

    # State overlay: one map per clustering group; the page starts on the first.
    states = build_bin_state_overlay(
        df, pair, x_edges, y_edges, groupby=resolve_state_groupby(cfg)
    )
    traces = [heatmap]
    state_annotations: list[dict] = []
    state_group = 0
    show_states = interactive_cfg["show_states"]
    if states is not None:
        n_labels = len(states["labels"])
        states["colors"] = [_STATE_COLORS[i % len(_STATE_COLORS)] for i in range(n_labels)]
        group = states["groups"][state_group] if states["groups"] else {"bins": [], "states": [], "centres": []}
        traces.append(go.Heatmap(
            z=_state_grid(group, n_bins_y, n_bins_x),
            x=x_centres,
            y=y_centres,
            zmin=-0.5,
            zmax=max(n_labels, 1) - 0.5,
            colorscale=_state_colorscale(n_labels),
            showscale=False,
            opacity=_STATE_OPACITY,
            # Hover and clicks go to the density trace underneath.
            hoverinfo="skip",
            visible=show_states,
            name="states",
        ))
        state_annotations = _state_annotations(
            states, group, show_states, plotly_theme["state_label_bg"]
        )

    degree_axes = {
        "x": _is_degree_axis(pair.x_dof_type, pair.x_label),
        "y": _is_degree_axis(pair.y_dof_type, pair.y_label),
    }
    axis_ticks = {
        axis: _degree_axis_ticks(axis_range) if degree_axes[axis] else {}
        for axis, axis_range in (("x", x_range), ("y", y_range))
    }

    fig = go.Figure(data=traces)
    fig.update_layout(
        annotations=state_annotations,
        title=pair.title,
        xaxis={
            "title": axis_titles["x"],
            "range": list(x_range),
            "gridcolor": plotly_theme["grid_color"],
            **axis_ticks["x"],
        },
        yaxis={
            "title": axis_titles["y"],
            "range": list(y_range),
            "gridcolor": plotly_theme["grid_color"],
            **axis_ticks["y"],
        },
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

    axis_atoms = (
        _build_axis_atoms(pair, _known_atom_counts(df, bin_xyz_payloads)) if highlight else None
    )

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
        "scale": {
            "grids": {
                "counts": _grid_to_json(grids["counts"], integer=True),
                "log_counts": _grid_to_json(grids["log_counts"]),
                "free_energy": _grid_to_json(grids["free_energy"]),
            },
            "modes": _SCALE_MODES,
            "colorscales": colorscales,
            "energy_units": energy_unit_table(),
        },
        "axis_spec": {"x_col": x_col, "y_col": y_col},
        "bin_geometry": bin_geometry,
        "bin_frame_metadata": bin_frame_metadata,
        "bin_xyz_payloads": bin_xyz_payloads,
        "frame_metadata": frame_metadata,
        "states": states,
        "axis_ticks": {
            "steps": list(_DEGREE_TICK_STEPS),
            "max_intervals": _DEGREE_TICK_MAX_INTERVALS,
            **degree_axes,
        },
        "axis_atoms": axis_atoms,
        "settings": {
            "hover_preview": interactive_cfg["hover_preview"],
            "max_pinned": interactive_cfg["max_pinned"],
        },
        "ui_state": {
            "theme": theme,
            "scale_mode": scale_mode,
            "state_overlay_visible": show_states if states is not None else False,
            "state_group": state_group,
            "temperature": temperature,
            "unit": unit,
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
