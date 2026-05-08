"""Standalone interactive HTML output for conformer density analysis.

Generates self-contained HTML files using Plotly.  Each file embeds a 2D
density heatmap and a JSON metadata block so that clicking a bin displays
frame metadata (and, optionally, a 3D structure via 3Dmol.js) directly in
the browser without a Python server.

When ``interactive.embed_xyz_payload: true`` is set in the config, one
representative XYZ structure per occupied bin is embedded using
``src.viewer.build_bin_xyz_payloads``.  Clicking bins then opens aligned
structures in a browser-native comparison tray with one 3Dmol.js viewer
per open structure card.

Public API
----------
- ``make_density_interactive``
"""

from __future__ import annotations

import json
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
# Helpers
# ---------------------------------------------------------------------------


def _split_plot_config(config: dict | None) -> tuple[dict, dict]:
    """Return ``(root_cfg, density_cfg)`` for full-config or density-only input."""
    cfg = config or {}
    if "density" in cfg:
        return cfg, cfg.get("density", {}) or {}
    return cfg, cfg


def _build_frame_metadata_json(
    df: pd.DataFrame,
    extra_columns: list[str] | None = None,
) -> str:
    """Serialise the coordinate table rows to a JSON string.

    Only columns in ``_BASE_META_COLUMNS`` plus any requested *extra_columns*
    that are present in *df* are included.
    Numpy/Pandas NA values are converted to ``null``.
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
    return json.dumps(records, default=str)


# ---------------------------------------------------------------------------
# JavaScript click handler (injected as post_script)
# ---------------------------------------------------------------------------

# HTML element injected near the top of <body> (not inside a <script> context).
# Provides a comparison tray that can keep multiple structure cards open at once.
_COMPARISON_TRAY_HTML = """<div id="comparison-tray" style="
    position: fixed; left: 10px; right: 10px; bottom: 10px;
    background: rgba(255,255,255,0.98); border: 1px solid #ccc; border-radius: 8px;
    padding: 12px; font-family: monospace; font-size: 12px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.18);
    display: none; z-index: 9999; overflow: hidden; max-height: 48vh;
">
  <div style="
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; margin-bottom: 10px;
  ">
    <div><b>Structure comparison</b></div>
    <button id="comparison-clear" type="button" style="
      border: 1px solid #ccc; border-radius: 4px; background: #f7f7f7;
      padding: 4px 8px; cursor: pointer; font: inherit;
    ">Clear all</button>
  </div>
  <div id="comparison-cards" style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
    overflow-y: auto;
    max-height: calc(48vh - 42px);
    padding-right: 4px;
  "></div>
</div>"""

# Pure JavaScript injected via post_script (runs inside Plotly's .then() callback).
# Must contain no HTML tags.
#
# Two modes:
#   embed_xyz_payload: false  — 'frame-metadata' holds all frame records (per-frame
#                               nearest scan).  No 3D viewer.
#   embed_xyz_payload: true   — 'bin-frame-metadata' holds one metadata record per
#                               occupied bin (O(1) lookup).  'bin-xyz-payloads' and
#                               'bin-geometry' enable 3Dmol.js rendering.
#
# The metadata blocks and the 3Dmol.js <script> tag are injected near the top of
# <body> before the Plotly callback runs, so the click handler can resolve them
# immediately.  Individual 3Dmol viewers are created lazily per comparison card.
_CLICK_HANDLER_JS = """
(function () {
  var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
  if (!plotDiv) return;

  var trayEl = document.getElementById('comparison-tray');
  var cardsEl = document.getElementById('comparison-cards');
  var clearBtn = document.getElementById('comparison-clear');
  if (!trayEl || !cardsEl || !clearBtn) return;

  // Per-frame metadata (embed_xyz_payload: false path) — may be null/empty.
  var metaEl = document.getElementById('frame-metadata');
  var frames = (metaEl && metaEl.textContent.trim()) ? JSON.parse(metaEl.textContent) : null;
  var axisSpecEl = document.getElementById('plot-axis-spec');
  var axisSpec = axisSpecEl ? JSON.parse(axisSpecEl.textContent) : {
    xCol: 'x',
    yCol: 'y'
  };

  // Per-bin data (embed_xyz_payload: true path).
  var xyzPayloadsEl = document.getElementById('bin-xyz-payloads');
  var geoEl         = document.getElementById('bin-geometry');
  var binMetaEl     = document.getElementById('bin-frame-metadata');
  var binXyz      = xyzPayloadsEl ? JSON.parse(xyzPayloadsEl.textContent) : null;
  var binGeo      = geoEl         ? JSON.parse(geoEl.textContent)         : null;
  var binFrameMeta = binMetaEl    ? JSON.parse(binMetaEl.textContent)     : null;
  var openCards = Object.create(null);

  function updateTrayVisibility() {
    trayEl.style.display = cardsEl.children.length ? 'block' : 'none';
  }

  function formatValue(value) {
    if (typeof value === 'number') {
      return value.toPrecision ? value.toPrecision(6) : value;
    }
    return value;
  }

  function buildMetadataHtml(best) {
    var html = '';
    var order = [
      axisSpec.xCol, axisSpec.yCol,
      'source_file','frame_number','byte_offset','trajectory_id','bead_id',
      'local_frame_index','global_frame_index','energy'
    ];
    // Append any remaining keys not already listed
    for (var k in best) {
      if (order.indexOf(k) === -1) order.push(k);
    }
    var seen = Object.create(null);
    for (var j = 0; j < order.length; j++) {
      var key = order[j];
      if (seen[key]) continue;
      seen[key] = true;
      if (best[key] != null) {
        html += '<div><b>' + key + ':</b> ' + formatValue(best[key]) + '</div>';
      }
    }
    return html;
  }

  function getCardKey(best, xi, yi) {
    if (binGeo && xi !== null && yi !== null) return 'bin:' + xi + '_' + yi;
    if (best.frame_id != null) return 'frame:' + best.frame_id;
    var filePart = best.source_file != null ? best.source_file : 'unknown';
    var offsetPart = best.byte_offset != null ? best.byte_offset : 'na';
    var framePart = best.frame_number != null ? best.frame_number : 'na';
    return 'frame:' + filePart + '|' + offsetPart + '|' + framePart;
  }

  function getCardTitle(best, xi, yi) {
    if (binGeo && xi !== null && yi !== null) return 'Bin ' + xi + '_' + yi;
    if (best.frame_number != null) return 'Frame ' + best.frame_number;
    return 'Structure';
  }

  function focusCard(card) {
    if (!card) return;
    card.scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'nearest'});
    var previousShadow = card.style.boxShadow;
    card.style.boxShadow = '0 0 0 2px #26828e, 0 8px 18px rgba(0,0,0,0.12)';
    window.setTimeout(function () {
      card.style.boxShadow = previousShadow || '0 6px 14px rgba(0,0,0,0.10)';
    }, 800);
  }

  function getOrCreateViewer(card) {
    if (card._viewer3d) return card._viewer3d;
    if (typeof $3Dmol === 'undefined') return null;
    var viewerEl = card.querySelector('.comparison-viewer');
    if (!viewerEl) return null;
    card._viewer3d = $3Dmol.createViewer(viewerEl, {backgroundColor: 'white'});
    return card._viewer3d;
  }

  function renderCardStructure(card, xyzText) {
    if (!xyzText) return;
    var v3d = getOrCreateViewer(card);
    if (!v3d) return;
    v3d.clear();
    v3d.addModel(xyzText, 'xyz');
    v3d.setStyle({}, {stick: {}});
    v3d.zoomTo();
    v3d.render();
  }

  function removeCard(cardKey) {
    var card = openCards[cardKey];
    if (!card) return;
    if (card._viewer3d) card._viewer3d.clear();
    if (card.parentNode) card.parentNode.removeChild(card);
    delete openCards[cardKey];
    updateTrayVisibility();
  }

  function clearAllCards() {
    var keys = Object.keys(openCards);
    for (var i = 0; i < keys.length; i++) {
      removeCard(keys[i]);
    }
    updateTrayVisibility();
  }

  function createCard(cardKey, cardTitle, metadataHtml, xyzText) {
    var card = document.createElement('div');
    card.style.cssText = [
      'background:white',
      'border:1px solid #ddd',
      'border-radius:6px',
      'padding:10px',
      'min-height:140px',
      'box-shadow:0 6px 14px rgba(0,0,0,0.10)'
    ].join(';');

    var header = document.createElement('div');
    header.style.cssText = [
      'display:flex',
      'align-items:center',
      'justify-content:space-between',
      'gap:8px',
      'margin-bottom:8px'
    ].join(';');

    var titleEl = document.createElement('div');
    titleEl.innerHTML = '<b>' + cardTitle + '</b>';

    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.textContent = 'Close';
    closeBtn.style.cssText = [
      'border:1px solid #ccc',
      'border-radius:4px',
      'background:#f7f7f7',
      'padding:3px 8px',
      'cursor:pointer',
      'font:inherit'
    ].join(';');
    closeBtn.addEventListener('click', function () {
      removeCard(cardKey);
    });

    header.appendChild(titleEl);
    header.appendChild(closeBtn);
    card.appendChild(header);

    var metaEl = document.createElement('div');
    metaEl.style.cssText = 'max-height:132px; overflow-y:auto;';
    metaEl.innerHTML = metadataHtml;
    card.appendChild(metaEl);

    if (xyzText) {
      var viewerEl = document.createElement('div');
      viewerEl.className = 'comparison-viewer';
      viewerEl.style.cssText = [
        'width:100%',
        'height:280px',
        'position:relative',
        'margin-top:10px',
        'border:1px solid #eee',
        'border-radius:4px'
      ].join(';');
      card.appendChild(viewerEl);
    }

    cardsEl.prepend(card);
    openCards[cardKey] = card;
    updateTrayVisibility();
    renderCardStructure(card, xyzText);
    focusCard(card);
    return card;
  }

  clearBtn.addEventListener('click', function () {
    clearAllCards();
  });

  plotDiv.on('plotly_click', function (data) {
    var pt = data.points[0];
    var cx = pt.x;  // bin centre x
    var cy = pt.y;  // bin centre y

    // Resolve bin indices (used for both per-bin paths).
    var xi = null, yi = null;
    if (binGeo) {
      xi = Math.floor((cx - binGeo.xMin) / binGeo.binW);
      yi = Math.floor((cy - binGeo.yMin) / binGeo.binH);
      xi = Math.max(0, Math.min(xi, binGeo.nBinsX - 1));
      yi = Math.max(0, Math.min(yi, binGeo.nBinsY - 1));
    }

    // Find the representative frame record.
    var best = null;
    if (binFrameMeta && binGeo) {
      // O(1) per-bin lookup.
      best = binFrameMeta[xi + '_' + yi] || null;
    } else if (frames) {
      // O(N) fallback scan (embed_xyz_payload: false, small datasets).
      var bestDist = Infinity;
      for (var i = 0; i < frames.length; i++) {
        var f = frames[i];
        if (f[axisSpec.xCol] == null || f[axisSpec.yCol] == null) continue;
        var dx = f[axisSpec.xCol] - cx;
        var dy = f[axisSpec.yCol] - cy;
        var d = dx * dx + dy * dy;
        if (d < bestDist) { bestDist = d; best = f; }
      }
    }

    if (!best) return;

    var cardKey = getCardKey(best, xi, yi);
    if (openCards[cardKey]) {
      focusCard(openCards[cardKey]);
      return;
    }

    var xyzText = null;
    if (binXyz && binGeo && xi !== null && yi !== null) {
      xyzText = binXyz[xi + '_' + yi] || null;
    }

    createCard(
      cardKey,
      getCardTitle(best, xi, yi),
      buildMetadataHtml(best),
      xyzText
    );
  });

  // Double-click clears the comparison tray.
  plotDiv.on('plotly_doubleclick', function () {
    clearAllCards();
  });
})();
"""


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
    and a JSON metadata block.  Clicking bins opens one comparison card per
    selected structure; double-clicking clears the tray.  No Python server is
    required.

    When ``interactive.embed_xyz_payload: true`` is set in the config, one
    representative XYZ structure per occupied bin is also embedded.  Clicking a
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
        Destination file path (.html).  Parent directory is created if needed.
    config:
        Optional config dict.  Accepts either the full project config or just
        the ``density:`` sub-section.  Controls bins, axis ranges, log_scale,
        ``interactive.include_plotlyjs``, ``interactive.embed_xyz_payload``,
        and ``interactive.alignment``.

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
    _, density_cfg = (
        (cfg, cfg.get("density", {}) or {}) if "density" in cfg else ({}, cfg)
    )

    # Per-pair overrides take precedence over global density config.
    bins: int = (
        pair.bins if pair.bins is not None
        else int(density_cfg.get("bins", 180))
    )
    colorscale: str = (
        pair.colormap if pair.colormap is not None
        else str(density_cfg.get("colormap", "Viridis"))
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
    colorbar_title = "log\u2081\u2080(count+1)" if log_scale else "count"

    # Bin centres for hover / click coordinates
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

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

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=pair.title,
        xaxis_title=pair.x_label,
        yaxis_title=pair.y_label,
        xaxis={"range": list(x_range)},
        yaxis={"range": list(y_range)},
        width=700,
        height=600,
    )

    # include_plotlyjs: "cdn", True (inline), or "require"
    interactive_cfg = cfg.get("interactive", {})
    include_plotlyjs = interactive_cfg.get("include_plotlyjs", "cdn")
    embed_xyz = bool(interactive_cfg.get("embed_xyz_payload", False))

    html_str = fig.to_html(
        full_html=True,
        include_plotlyjs=include_plotlyjs,
        post_script=_CLICK_HANDLER_JS,
    )

    # Build the block of HTML elements to inject near the top of <body>.
    inject_parts = [_COMPARISON_TRAY_HTML]

    if embed_xyz:
        # Per-bin mode: embed one metadata record and one XYZ payload per occupied
        # bin instead of the full per-frame table.
        from src.viewer import build_bin_frame_metadata, build_bin_xyz_payloads  # noqa: PLC0415

        bin_payloads = build_bin_xyz_payloads(
            df, x_col=x_col, y_col=y_col,
            x_edges=x_edges, y_edges=y_edges,
            alignment=interactive_cfg.get("alignment"),
        )
        bin_metadata = build_bin_frame_metadata(
            df, x_col=x_col, y_col=y_col,
            x_edges=x_edges, y_edges=y_edges,
            extra_fields=[x_col, y_col],
        )
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1
        bin_geo = {
            "xMin": float(x_edges[0]),
            "yMin": float(y_edges[0]),
            "binW": float(x_edges[1] - x_edges[0]),
            "binH": float(y_edges[1] - y_edges[0]),
            "nBinsX": n_x,
            "nBinsY": n_y,
        }
        inject_parts.append('<script id="frame-metadata" type="application/json">null</script>')
        inject_parts.extend([
            (
                '<script id="plot-axis-spec" type="application/json">'
                + json.dumps({"xCol": x_col, "yCol": y_col})
                + "</script>"
            ),
            (
                '<script id="bin-geometry" type="application/json">'
                + json.dumps(bin_geo)
                + "</script>"
            ),
            (
                '<script id="bin-xyz-payloads" type="application/json">'
                + json.dumps(bin_payloads)
                + "</script>"
            ),
            (
                '<script id="bin-frame-metadata" type="application/json">'
                + json.dumps(bin_metadata)
                + "</script>"
            ),
            '<script src="https://3dmol.org/build/3Dmol-min.js"></script>',
        ])
    else:
        # Per-frame mode: embed the full coordinate table so the JavaScript can
        # find the nearest frame to any click point.  Practical for small datasets.
        frame_meta_json = _build_frame_metadata_json(
            df,
            extra_columns=[x_col, y_col],
        )
        inject_parts.append(
            '<script id="plot-axis-spec" type="application/json">'
            + json.dumps({"xCol": x_col, "yCol": y_col})
            + "</script>"
        )
        inject_parts.append(
            '<script id="frame-metadata" type="application/json">'
            + frame_meta_json
            + "</script>"
        )

    # Insert the comparison tray / metadata blocks immediately after <body> so
    # they already exist when Plotly runs the post_script click handler.
    html_str = html_str.replace(
        "<body>",
        "<body>\n" + "\n".join(inject_parts) + "\n",
        1,
    )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html_str, encoding="utf-8")

    return outpath
