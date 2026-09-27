/*
 * ConfAna interactive density viewer client script.
 *
 * Reads the single embedded #page-data JSON object and wires up:
 *  - header stats (frame count, bin count, scale mode)
 *  - light/dark theme toggle, synced into Plotly and the 3Dmol viewers
 *  - live hover preview: one shared 3Dmol viewer plus a read-out and a
 *    metadata table for the bin under the cursor
 *  - pinned cards: clicking a bin/frame opens a persistent card with its own
 *    3Dmol viewer (multi-card design; no singleton #viewer3d), capped at
 *    settings.max_pinned with the oldest card evicted
 *  - axis-atom highlighting: the atoms defining the x and y coordinates
 *    (page data axis_atoms) are coloured in every 3D view, matching the
 *    axis titles, with a legend in the side panel
 *
 * Seams left for later slices (inert here): ui_state.scale_mode,
 * state_overlay_visible, temperature, unit, pinned_bins.
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
  var settings = pageData.settings || {};
  var maxPinned = settings.max_pinned || 15;
  var hoverPreview = settings.hover_preview !== false;
  var axisAtoms = pageData.axis_atoms || null;

  function axisAtomList(axis) {
    return (axisAtoms && axisAtoms[axis] && axisAtoms[axis].atoms) || [];
  }

  // Atoms in both axes get their own colour, so each atom has exactly one.
  var highlightGroups = (function () {
    var xs = axisAtomList('x');
    var ys = axisAtomList('y');
    if (!xs.length && !ys.length) return null;
    function notIn(list) {
      return function (a) { return list.indexOf(a) === -1; };
    }
    var both = xs.filter(function (a) { return ys.indexOf(a) !== -1; });
    return [
      {name: 'x', atoms: xs.filter(notIn(both)), colorVar: '--ca-axis-x'},
      {name: 'y', atoms: ys.filter(notIn(both)), colorVar: '--ca-axis-y'},
      {name: 'both', atoms: both, colorVar: '--ca-axis-both'}
    ].filter(function (g) { return g.atoms.length; });
  })();

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
    ['x', 'y'].forEach(function (axis) {
      if (!axisAtomList(axis).length) return;
      var color = cssVar('--ca-axis-' + axis);
      var title = (layout[axis + 'axis'] || {}).title || {};
      if ((title.font || {}).color !== color) update[axis + 'axis.title.font.color'] = color;
    });
    return update;
  }

  // Viewers currently on the page. Pooled (detached) viewers are skipped:
  // setBackgroundColor renders, and they are recoloured when reused.
  function attachedViewers() {
    var list = [];
    if (previewViewer) list.push(previewViewer);
    var boxes = cardsEl ? cardsEl.querySelectorAll('.comparison-viewer') : [];
    for (var i = 0; i < boxes.length; i++) {
      if (boxes[i]._viewer3d) list.push(boxes[i]._viewer3d);
    }
    return list;
  }

  function applyTheme() {
    if (gd && window.Plotly) {
      var update = plotThemeUpdate();
      if (Object.keys(update).length) Plotly.relayout(gd, update);
    }
    var viewerBg = hexToInt(cssVar('--ca-viewer-bg'));
    attachedViewers().forEach(function (v) {
      v.setBackgroundColor(viewerBg);
      styleStructure(v);
      v.render();
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
  // 3Dmol viewers
  // -------------------------------------------------------------------
  // Each 3Dmol viewer permanently registers a window resize listener and two
  // observers, so card viewers are reused from a pool instead of recreated.
  var viewerPool = [];
  var viewerFailed = false;

  function createViewerIn(el) {
    if (viewerFailed || typeof $3Dmol === 'undefined') return null;
    try {
      return $3Dmol.createViewer(el, {backgroundColor: hexToInt(cssVar('--ca-viewer-bg'))});
    } catch (err) {
      viewerFailed = true;
      return null;
    }
  }

  function showViewerError(el) {
    el.innerHTML = '';
    var msg = document.createElement('div');
    msg.className = 'ca-viewer-error';
    msg.textContent = '3D viewer could not load';
    el.appendChild(msg);
  }

  // The one place that maps config atom indices to 3Dmol atoms. 3Dmol's
  // `index` is the atom's 0-based position in its model (`serial` may come
  // from the file), and every viewer holds a single model (showStructure
  // clears first), so `index` equals the 0-based file index used in config.
  function atomSelection(indices) {
    return {index: indices};
  }

  // Neutral sticks, with each axis's atoms as coloured sphere + stick.
  // Without axis atoms, keep 3Dmol's element-coloured sticks.
  function styleStructure(v) {
    if (!highlightGroups) {
      v.setStyle({}, {stick: {}});
      return;
    }
    v.setStyle({}, {stick: {color: cssVar('--ca-atom-neutral')}});
    highlightGroups.forEach(function (group) {
      var color = cssVar(group.colorVar);
      v.setStyle(atomSelection(group.atoms), {
        stick: {color: color},
        sphere: {color: color, radius: 0.4}
      });
    });
  }

  function showStructure(v, xyzText) {
    v.clear();
    v.addModel(xyzText, 'xyz');
    styleStructure(v);
    v.zoomTo();
    v.render();
  }

  // The card must already be in the DOM: 3Dmol sizes against the visible box.
  function takeViewerBox(card) {
    var box = viewerPool.pop();
    if (box) {
      card.appendChild(box);
      box._viewer3d.setBackgroundColor(hexToInt(cssVar('--ca-viewer-bg')));
      box._viewer3d.resize();
      return box;
    }
    box = document.createElement('div');
    box.className = 'comparison-viewer';
    card.appendChild(box);
    box._viewer3d = createViewerIn(box);
    if (!box._viewer3d) showViewerError(box);
    return box;
  }

  function releaseViewerBox(card) {
    var box = card.querySelector('.comparison-viewer');
    if (!box) return;
    if (box._viewer3d) box._viewer3d.clear();
    box.parentNode.removeChild(box);
    if (box._viewer3d) viewerPool.push(box);
  }

  // -------------------------------------------------------------------
  // Shared helpers
  // -------------------------------------------------------------------
  // Plotly heatmap events report the bin as pointNumber = [row, col] = [yi, xi].
  function binFromPoint(pt) {
    var pn = pt.pointNumber;
    if (!Array.isArray(pn) || pn.length !== 2) return null;
    return {xi: pn[1], yi: pn[0], key: pn[1] + '_' + pn[0]};
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

  // -------------------------------------------------------------------
  // Pinned cards (multi-card design; one 3Dmol viewer per card)
  // -------------------------------------------------------------------
  var trayEl = document.getElementById('comparison-tray');
  var cardsEl = document.getElementById('comparison-cards');
  var clearBtn = document.getElementById('comparison-clear');
  var noticeEl = document.getElementById('panel-notice');
  var openCards = Object.create(null);
  var noticeTimer = null;

  function updateTrayVisibility() {
    if (!trayEl || !cardsEl) return;
    trayEl.style.display = cardsEl.children.length ? 'block' : 'none';
  }

  function showNotice(message) {
    if (!noticeEl) return;
    noticeEl.textContent = message;
    noticeEl.hidden = false;
    if (noticeTimer) window.clearTimeout(noticeTimer);
    noticeTimer = window.setTimeout(function () {
      noticeEl.hidden = true;
    }, 3000);
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

  // The DOM is the single record of pinned cards and their order; openCards
  // only maps keys to elements for de-duplication.
  function removeCardElement(card) {
    releaseViewerBox(card);
    if (card.parentNode) card.parentNode.removeChild(card);
    delete openCards[card.dataset.cardKey];
    updateTrayVisibility();
  }

  function removeCard(cardKey) {
    var card = openCards[cardKey];
    if (card) removeCardElement(card);
  }

  function clearAllCards() {
    while (cardsEl && cardsEl.firstElementChild) {
      removeCardElement(cardsEl.firstElementChild);
    }
    updateTrayVisibility();
  }

  function createCard(cardKey, cardTitle, metadataHtml, xyzText) {
    // Cards are prepended, so the last child is the oldest.
    var evicted = false;
    while (cardsEl.children.length >= maxPinned) {
      removeCardElement(cardsEl.lastElementChild);
      evicted = true;
    }
    if (evicted) {
      showNotice('Pin limit (' + maxPinned + ') reached — removed the oldest pinned structure.');
    }

    var card = document.createElement('div');
    card.className = 'ca-card';
    card.dataset.cardKey = cardKey;

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

    cardsEl.prepend(card);
    openCards[cardKey] = card;
    updateTrayVisibility();

    if (xyzText) {
      var box = takeViewerBox(card);
      if (box._viewer3d) showStructure(box._viewer3d, xyzText);
    }

    focusCard(card);
    return card;
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      clearAllCards();
    });
  }

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') clearAllCards();
  });

  // -------------------------------------------------------------------
  // Hover preview
  // -------------------------------------------------------------------
  var previewBox = document.getElementById('preview-3d');
  var previewMetaBox = document.getElementById('preview-metadata');
  var previewViewerEl = document.getElementById('preview-viewer');
  var previewStatusEl = document.getElementById('preview-status');
  var previewReadoutEl = document.getElementById('preview-readout');
  var previewMetaEl = document.getElementById('preview-meta');
  // In per-frame mode 3Dmol is never embedded, so there is no 3D preview.
  var hasStructures = !!(binGeo && binXyz);
  var previewViewer = null;
  var previewFailed = false;
  var lastShownKey = null;
  var currentHoverKey = null;
  var pendingPoint = null;
  var hoverFrameRequested = false;

  function setStatus(text) {
    if (previewStatusEl) previewStatusEl.textContent = text;
  }

  function showingSuffix() {
    return lastShownKey ? ' — showing bin ' + lastShownKey : '';
  }

  function updateReadout(pt) {
    if (!previewReadoutEl) return;
    previewReadoutEl.innerHTML = '';
    var rows = [
      [pairInfo.x_label || axisSpec.x_col, pt.x],
      [pairInfo.y_label || axisSpec.y_col, pt.y],
      [headerInfo.scale_mode_label || 'value', pt.z]
    ];
    rows.forEach(function (row) {
      var line = document.createElement('div');
      var label = document.createElement('b');
      label.textContent = row[0] + ': ';
      line.appendChild(label);
      line.appendChild(
        document.createTextNode(Number.isFinite(row[1]) ? formatValue(row[1]) : '—')
      );
      previewReadoutEl.appendChild(line);
    });
  }

  function ensurePreviewViewer() {
    if (previewViewer || previewFailed) return previewViewer;
    previewViewer = createViewerIn(previewViewerEl);
    // Same handle as the card boxes carry.
    previewViewerEl._viewer3d = previewViewer;
    if (!previewViewer) {
      previewFailed = true;
      showViewerError(previewViewerEl);
    }
    return previewViewer;
  }

  function processHover() {
    hoverFrameRequested = false;
    var pt = pendingPoint;
    if (!pt) return;
    var bin = binFromPoint(pt);
    var key = bin ? bin.key : pt.x + ',' + pt.y;
    if (key === currentHoverKey) return;
    currentHoverKey = key;

    // Read-out first, so it keeps working even if the 3D viewer fails.
    updateReadout(pt);

    if (!binGeo) {
      // Per-frame mode: never scan frame_metadata on hover (click only).
      setStatus(Number.isFinite(pt.z) ? 'Click to pin the nearest frame' : 'No frames in this bin');
      return;
    }

    var record = bin && binFrameMeta ? binFrameMeta[bin.key] : null;
    if (!record) {
      setStatus('No frames in this bin' + showingSuffix());
      return;
    }
    if (previewMetaEl) previewMetaEl.innerHTML = buildMetadataHtml(record);

    var xyzText = binXyz ? binXyz[bin.key] : null;
    if (!xyzText) {
      setStatus('No structure available for bin ' + bin.key + showingSuffix());
      return;
    }
    var v = ensurePreviewViewer();
    if (v) {
      showStructure(v, xyzText);
      lastShownKey = bin.key;
    }
    setStatus('Bin ' + bin.key);
  }

  function onHover(data) {
    if (!data || !data.points || !data.points.length) return;
    pendingPoint = data.points[0];
    if (!hoverFrameRequested) {
      hoverFrameRequested = true;
      window.requestAnimationFrame(processHover);
    }
  }

  // -------------------------------------------------------------------
  // Axis-atom legend
  // -------------------------------------------------------------------
  var legendEl = document.getElementById('axis-legend');

  function renderAxisLegend() {
    if (!legendEl || !highlightGroups || !hasStructures) return;
    // Axis rows list the full DoF atoms in definition order, to compare with
    // config; the shared row lists only the atoms drawn in the third colour.
    var rows = [
      {name: 'x', label: pairInfo.x_label || axisSpec.x_col, atoms: axisAtomList('x')},
      {name: 'y', label: pairInfo.y_label || axisSpec.y_col, atoms: axisAtomList('y')}
    ];
    highlightGroups.forEach(function (group) {
      if (group.name === 'both') rows.push({name: 'both', label: 'Both axes', atoms: group.atoms});
    });
    rows.forEach(function (entry) {
      if (!entry.atoms.length) return;
      var row = document.createElement('div');
      row.className = 'ca-legend-row';
      row.dataset.group = entry.name;
      var swatch = document.createElement('span');
      swatch.className = 'ca-legend-swatch';
      swatch.style.background = 'var(--ca-axis-' + entry.name + ')';
      var label = document.createElement('span');
      label.textContent = entry.label;
      var atoms = document.createElement('span');
      atoms.className = 'ca-legend-atoms';
      atoms.textContent = 'atoms ' + entry.atoms.join(', ');
      row.appendChild(swatch);
      row.appendChild(label);
      row.appendChild(atoms);
      legendEl.appendChild(row);
    });
    var note = document.createElement('div');
    note.className = 'ca-legend-note';
    note.textContent = '0-based atom indices in the xyz file';
    legendEl.appendChild(note);
    legendEl.hidden = false;
  }

  renderAxisLegend();

  // Un-hide before any viewer is created, so 3Dmol measures a visible box.
  if (hoverPreview) {
    if (previewMetaBox) previewMetaBox.hidden = false;
    if (previewBox && hasStructures) previewBox.hidden = false;
  }

  // -------------------------------------------------------------------
  // Plotly events
  // -------------------------------------------------------------------
  // gd.on(...) is Plotly's own pub/sub attached to the graph div (not a
  // native DOM event); it is available synchronously once the plot
  // fragment's Plotly.newPlot(...) call, emitted just before this script,
  // has run. (Never write a closing script tag in this file, even in a
  // comment: this file is inlined, and the browser would end it there.)
  var gd = document.getElementsByClassName('plotly-graph-div')[0];

  if (gd) {
    gd.on('plotly_click', function (data) {
      // Double-click clears the pins. Use the browser's click count, which
      // requires both clicks in the same spot: plotly_doubleclick fires for
      // any two clicks within 300 ms, even on different bins, and would wipe
      // the pins when clicking quickly.
      if (data.event && data.event.detail >= 2) {
        clearAllCards();
        return;
      }
      var pt = data.points[0];
      var cx = pt.x;
      var cy = pt.y;

      var bin = binGeo ? binFromPoint(pt) : null;
      var xi = bin ? bin.xi : null;
      var yi = bin ? bin.yi : null;

      var best = null;
      if (binFrameMeta && bin) {
        best = binFrameMeta[bin.key] || null;
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

      var xyzText = binXyz && bin ? binXyz[bin.key] || null : null;

      createCard(cardKey, getCardTitle(best, xi, yi), buildMetadataHtml(best), xyzText);
    });

    if (hoverPreview) gd.on('plotly_hover', onHover);
  }

  applyTheme();
  updateTrayVisibility();
})();
