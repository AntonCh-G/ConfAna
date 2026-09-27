/*
 * ConfAna interactive density viewer client script.
 *
 * Reads the single embedded #page-data JSON object and wires up:
 *  - header stats (frame count, bin count, scale mode)
 *  - light/dark theme toggle, synced into Plotly and any open 3Dmol viewers
 *  - the comparison tray: clicking a bin/frame opens a persistent card with
 *    its own 3Dmol viewer (multi-card design; no singleton #viewer3d).
 *
 * Placeholders left for later slices (inert in Slice 1):
 *  - #preview-3d / #preview-metadata (hover preview, Slice 2)
 *  - ui_state.scale_mode / state_overlay_visible / temperature / unit /
 *    pinned_bins (Slices 4-5)
 */
(function () {
  'use strict';

  var pageDataEl = document.getElementById('page-data');
  var pageData = pageDataEl ? JSON.parse(pageDataEl.textContent) : {};

  var pairInfo = pageData.pair || {};
  var headerInfo = pageData.header || {};
  var axisSpec = pageData.axis_spec || {x_col: 'x', y_col: 'y'};
  var binGeo = pageData.bin_geometry || null;
  var binFrameMeta = pageData.bin_frame_metadata || null;
  var binXyz = pageData.bin_xyz_payloads || null;
  var frames = pageData.frame_metadata || null;
  var uiState = pageData.ui_state || {};

  // -------------------------------------------------------------------
  // Header
  // -------------------------------------------------------------------
  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  setText('hdr-title', pairInfo.title || '');
  setText('hdr-frame-count', headerInfo.frame_count != null ? headerInfo.frame_count : '–');
  setText(
    'hdr-bin-count',
    headerInfo.bin_count_x != null && headerInfo.bin_count_y != null
      ? headerInfo.bin_count_x + '×' + headerInfo.bin_count_y
      : '–'
  );
  setText('hdr-scale-mode', headerInfo.scale_mode_label || '');

  // -------------------------------------------------------------------
  // Theme
  // -------------------------------------------------------------------
  var root = document.documentElement;
  if (uiState.theme === 'light' || uiState.theme === 'dark') {
    root.dataset.theme = uiState.theme;
  }

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function hexToInt(hex) {
    return parseInt(String(hex).replace('#', ''), 16);
  }

  // Only the colours that differ from gd.layout: Plotly.relayout never
  // compares values and always forces a full redraw.
  function plotThemeUpdate() {
    var bg = cssVar('--ca-plot-bg');
    var fg = cssVar('--ca-plot-fg');
    var grid = cssVar('--ca-plot-grid');
    var layout = gd.layout || {};
    var update = {};
    if (layout.paper_bgcolor !== bg) update.paper_bgcolor = bg;
    if (layout.plot_bgcolor !== bg) update.plot_bgcolor = bg;
    if ((layout.font || {}).color !== fg) update['font.color'] = fg;
    if ((layout.xaxis || {}).gridcolor !== grid) update['xaxis.gridcolor'] = grid;
    if ((layout.yaxis || {}).gridcolor !== grid) update['yaxis.gridcolor'] = grid;
    return update;
  }

  function applyTheme() {
    if (gd && window.Plotly) {
      var update = plotThemeUpdate();
      if (Object.keys(update).length) Plotly.relayout(gd, update);
    }
    var viewerBg = hexToInt(cssVar('--ca-viewer-bg'));
    Object.keys(openCards).forEach(function (key) {
      var v = openCards[key]._viewer3d;
      if (v) {
        v.setBackgroundColor(viewerBg);
        v.render();
      }
    });
  }

  var themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      var current =
        root.dataset.theme ||
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light');
      root.dataset.theme = current === 'dark' ? 'light' : 'dark';
      applyTheme();
    });
  }

  if (window.matchMedia) {
    var darkQuery = window.matchMedia('(prefers-color-scheme: dark)');
    if (darkQuery.addEventListener) darkQuery.addEventListener('change', applyTheme);
  }

  // -------------------------------------------------------------------
  // Comparison tray / cards (multi-card design; one 3Dmol viewer per card)
  // -------------------------------------------------------------------
  var trayEl = document.getElementById('comparison-tray');
  var cardsEl = document.getElementById('comparison-cards');
  var clearBtn = document.getElementById('comparison-clear');
  var openCards = Object.create(null);

  function updateTrayVisibility() {
    if (!trayEl || !cardsEl) return;
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
      axisSpec.x_col,
      axisSpec.y_col,
      'source_file',
      'frame_number',
      'byte_offset',
      'trajectory_id',
      'bead_id',
      'local_frame_index',
      'global_frame_index',
      'energy'
    ];
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
    card.classList.add('ca-card-focus');
    window.setTimeout(function () {
      card.classList.remove('ca-card-focus');
    }, 800);
  }

  function getOrCreateViewer(card) {
    if (card._viewer3d) return card._viewer3d;
    if (typeof $3Dmol === 'undefined') return null;
    var viewerEl = card.querySelector('.comparison-viewer');
    if (!viewerEl) return null;
    card._viewer3d = $3Dmol.createViewer(viewerEl, {
      backgroundColor: hexToInt(cssVar('--ca-viewer-bg'))
    });
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
    card.className = 'ca-card';

    var header = document.createElement('div');
    header.className = 'ca-card-header';

    var titleEl = document.createElement('div');
    titleEl.innerHTML = '<b>' + cardTitle + '</b>';

    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'ca-btn';
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', function () {
      removeCard(cardKey);
    });

    header.appendChild(titleEl);
    header.appendChild(closeBtn);
    card.appendChild(header);

    var metaEl = document.createElement('div');
    metaEl.className = 'ca-card-meta';
    metaEl.innerHTML = metadataHtml;
    card.appendChild(metaEl);

    if (xyzText) {
      var viewerEl = document.createElement('div');
      viewerEl.className = 'comparison-viewer';
      card.appendChild(viewerEl);
    }

    cardsEl.prepend(card);
    openCards[cardKey] = card;
    updateTrayVisibility();
    renderCardStructure(card, xyzText);
    focusCard(card);
    return card;
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      clearAllCards();
    });
  }

  // -------------------------------------------------------------------
  // Plotly click handling
  // -------------------------------------------------------------------
  // gd.on(...) is Plotly's own pub/sub attached to the graph div (not a
  // native DOM event); it is available synchronously once the plot
  // fragment's Plotly.newPlot(...) call, emitted just before this script,
  // has run. (Never write a closing script tag in this file, even in a
  // comment: this file is inlined, and the browser would end it there.)
  var gd = document.getElementsByClassName('plotly-graph-div')[0];

  if (gd) {
    gd.on('plotly_click', function (data) {
      var pt = data.points[0];
      var cx = pt.x;
      var cy = pt.y;

      var xi = null;
      var yi = null;
      if (binGeo) {
        xi = Math.floor((cx - binGeo.x_min) / binGeo.bin_w);
        yi = Math.floor((cy - binGeo.y_min) / binGeo.bin_h);
        xi = Math.max(0, Math.min(xi, binGeo.n_bins_x - 1));
        yi = Math.max(0, Math.min(yi, binGeo.n_bins_y - 1));
      }

      var best = null;
      if (binFrameMeta && binGeo) {
        best = binFrameMeta[xi + '_' + yi] || null;
      } else if (frames) {
        var bestDist = Infinity;
        for (var i = 0; i < frames.length; i++) {
          var f = frames[i];
          if (f[axisSpec.x_col] == null || f[axisSpec.y_col] == null) continue;
          var dx = f[axisSpec.x_col] - cx;
          var dy = f[axisSpec.y_col] - cy;
          var d = dx * dx + dy * dy;
          if (d < bestDist) {
            bestDist = d;
            best = f;
          }
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

      createCard(cardKey, getCardTitle(best, xi, yi), buildMetadataHtml(best), xyzText);
    });

    gd.on('plotly_doubleclick', function () {
      clearAllCards();
    });
  }

  applyTheme();
  updateTrayVisibility();
})();
