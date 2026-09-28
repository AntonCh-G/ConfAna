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
 *  - colour scale: log counts | counts | free-energy-like, with an editable
 *    temperature and energy unit; grids and the unit table come from Python
 *    (page data scale), and ui_state is kept in step with the controls
 *  - state overlay: a tint of the majority state per bin (trace 1) with
 *    state-name labels, for one clustering group at a time (page data
 *    states); the side panel names the hovered bin's state
 *  - pair navigation: header links to the pages of the run's other
 *    coordinate pairs (page data navigation)
 *  - payload codec: structures and metadata arrive gzipped (page data
 *    *_encoded); they are unpacked once on load with the browser's own
 *    DecompressionStream, then read through the payload stores
 *
 * Seam left for later slices (inert here): ui_state.pinned_bins.
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
  var encodedBinXyz = pageData.bin_xyz_payloads_encoded || null;
  var encodedBinMeta = pageData.bin_frame_metadata_encoded || null;
  var encodedFrames = pageData.frame_metadata_encoded || null;
  var uiState = pageData.ui_state || {};
  var settings = pageData.settings || {};
  var maxPinned = settings.max_pinned || 15;
  var hoverPreview = settings.hover_preview !== false;
  var axisAtoms = pageData.axis_atoms || null;

  // -------------------------------------------------------------------
  // Payload codec — mirrors src/payload_codec.py
  // -------------------------------------------------------------------
  // Everything between the two markers is self-contained (no DOM, no page
  // data): tests/test_interactive_assets.py cuts it out and runs it in Node
  // against a fixture encoded by the Python side.
  // --- payload codec start ---
  function caBytes(b64) {
    var binary = atob(b64);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  // gzip is unpacked by the browser itself: no library, works offline.
  function caGunzip(b64) {
    if (typeof DecompressionStream === 'undefined') {
      return Promise.reject(new Error('This browser has no DecompressionStream(gzip).'));
    }
    var stream = new Blob([caBytes(b64)]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).arrayBuffer();
  }

  function caGunzipJson(b64) {
    return caGunzip(b64).then(function (buffer) {
      return JSON.parse(new TextDecoder().decode(buffer));
    });
  }

  // Decimals one coordinate step needs (mirrors coordinate_decimals).
  function caDecimals(step) {
    return Math.max(0, Math.min(9, Math.ceil(-Math.log10(step))));
  }

  function caDecodeStructures(block) {
    if (!block || block.format !== 'confana-structures-v1') {
      return Promise.reject(new Error('Not a confana-structures-v1 block.'));
    }
    return Promise.all([caGunzipJson(block.index), caGunzip(block.coords)]).then(function (parts) {
      var index = Object.create(null);
      parts[0].keys.forEach(function (key, i) { index[key] = i; });
      return {
        keys: parts[0].keys,
        comments: parts[0].comments,
        index: index,
        elements: block.elements,
        atomCount: block.atom_count,
        step: block.step,
        coords: new Int16Array(parts[1])
      };
    });
  }

  // One structure rebuilt from the shared arrays, in the same text layout
  // decode_structures writes.
  function caStructureText(decoded, key) {
    var i = decoded.index[key];
    if (i === undefined) return null;
    var n = decoded.atomCount;
    var step = decoded.step;
    var digits = caDecimals(step);
    var at = i * n * 3;
    var lines = [String(n), decoded.comments[i]];
    for (var a = 0; a < n; a++) {
      lines.push(
        decoded.elements[a] + ' ' +
        (decoded.coords[at + a * 3] * step).toFixed(digits) + ' ' +
        (decoded.coords[at + a * 3 + 1] * step).toFixed(digits) + ' ' +
        (decoded.coords[at + a * 3 + 2] * step).toFixed(digits)
      );
    }
    return lines.join('\n') + '\n';
  }

  function caDecodeColumns(block) {
    if (!block || block.format !== 'confana-columns-v1') {
      return Promise.reject(new Error('Not a confana-columns-v1 block.'));
    }
    return caGunzipJson(block.data).then(function (payload) {
      var index = null;
      if (payload.keys) {
        index = Object.create(null);
        payload.keys.forEach(function (key, i) { index[key] = i; });
      }
      return {
        count: block.count,
        fields: block.fields,
        columns: payload.columns,
        keys: payload.keys,
        index: index
      };
    });
  }

  function caColumnValue(column, i) {
    if (column.lookup) {
      var code = column.codes[i];
      return code === null || code === undefined ? null : column.lookup[code];
    }
    return column.values[i];
  }

  function caColumnsRecord(decoded, i) {
    if (i === undefined || i === null || i < 0 || i >= decoded.count) return null;
    var record = {};
    decoded.fields.forEach(function (field) {
      record[field] = caColumnValue(decoded.columns[field], i);
    });
    return record;
  }
  // --- payload codec end ---

  // -------------------------------------------------------------------
  // Payload stores
  // -------------------------------------------------------------------
  // The page reads structures and metadata only through these, so plain
  // JSON blocks and compressed ones behave the same everywhere else.
  var structureStore = null;
  var binMetaStore = null;
  var frameStore = null;
  var frameColumnCache = Object.create(null);
  var payloadsPending = !!(encodedBinXyz || encodedBinMeta || encodedFrames);
  var payloadError = null;

  function binMetaFor(key) {
    if (binFrameMeta) return binFrameMeta[key] || null;
    if (binMetaStore) return caColumnsRecord(binMetaStore, binMetaStore.index[key]);
    return null;
  }

  function binXyzFor(key) {
    if (binXyz) return binXyz[key] || null;
    if (structureStore) return caStructureText(structureStore, key);
    return null;
  }

  function frameCount() {
    if (frames) return frames.length;
    return frameStore ? frameStore.count : 0;
  }

  function frameAt(i) {
    if (frames) return frames[i] || null;
    return frameStore ? caColumnsRecord(frameStore, i) : null;
  }

  // One field of the per-frame table as a plain array, for the click scan.
  function frameColumn(name) {
    if (frameColumnCache[name]) return frameColumnCache[name];
    var values = [];
    if (frames) {
      values = frames.map(function (f) { return f[name]; });
    } else if (frameStore && frameStore.columns[name]) {
      var column = frameStore.columns[name];
      values = new Array(frameStore.count);
      for (var i = 0; i < frameStore.count; i++) values[i] = caColumnValue(column, i);
    }
    frameColumnCache[name] = values;
    return values;
  }

  // Unpack the compressed blocks once, on load.
  function loadPayloads() {
    if (!payloadsPending) return;
    var jobs = [];
    if (encodedBinXyz) {
      jobs.push(caDecodeStructures(encodedBinXyz).then(function (d) { structureStore = d; }));
    }
    if (encodedBinMeta) {
      jobs.push(caDecodeColumns(encodedBinMeta).then(function (d) { binMetaStore = d; }));
    }
    if (encodedFrames) {
      jobs.push(caDecodeColumns(encodedFrames).then(function (d) { frameStore = d; }));
    }
    Promise.all(jobs).then(
      function () {
        payloadsPending = false;
        onPayloadsReady();
      },
      function (err) {
        payloadsPending = false;
        payloadError = err && err.message ? err.message : String(err);
        onPayloadsFailed();
      }
    );
  }

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
    var labelBg = cssVar('--ca-state-label-bg');
    (layout.annotations || []).forEach(function (a, i) {
      if (a.bgcolor !== labelBg) update['annotations[' + i + '].bgcolor'] = labelBg;
    });
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

  function activeTheme() {
    if (root.dataset.theme === 'light' || root.dataset.theme === 'dark') return root.dataset.theme;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }

  // The map's colour scale for the active theme, when it is not shown yet.
  function colorscaleUpdate() {
    var scales = (pageData.scale || {}).colorscales;
    if (!scales || !gd.data || !gd.data[0]) return {};
    var wanted = scales[activeTheme()];
    return JSON.stringify(gd.data[0].colorscale) === JSON.stringify(wanted)
      ? {}
      : {colorscale: [wanted]};
  }

  function applyTheme() {
    if (gd && window.Plotly) {
      var layoutUpdate = plotThemeUpdate();
      var traceUpdate = colorscaleUpdate();
      if (Object.keys(layoutUpdate).length || Object.keys(traceUpdate).length) {
        Plotly.update(gd, traceUpdate, layoutUpdate, [0]);
      }
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
      root.dataset.theme = activeTheme() === 'dark' ? 'light' : 'dark';
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
    if (Number.isInteger(value)) return String(value);
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
  var hasStructures = !!(binGeo && (binXyz || encodedBinXyz));
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

  function loadingMessage() {
    return hasStructures ? 'Loading structures…' : 'Loading frame data…';
  }

  function failedMessage() {
    return 'Could not unpack the embedded data: ' + payloadError;
  }

  // Both run once, when the compressed blocks are unpacked (or fail to be).
  function onPayloadsReady() {
    setStatus(hasStructures ? 'Hover over the map to preview a bin' : 'Hover over the map');
    // The cursor may already sit on a bin: redo that hover.
    currentHoverKey = null;
    if (pendingPoint) processHover();
  }

  function onPayloadsFailed() {
    setStatus(failedMessage());
    showNotice('Rebuild with plots.interactive.compress_payloads: false for plain JSON.');
  }

  var lastReadout = null;

  // Value and count come from the grids, so a scale change can refresh them.
  function updateReadout(pt, bin) {
    lastReadout = {pt: pt, bin: bin};
    if (!previewReadoutEl) return;
    previewReadoutEl.innerHTML = '';
    var value = bin && currentGrid ? currentGrid[bin.yi][bin.xi] : pt.z;
    var rows = [
      [pairInfo.x_label || axisSpec.x_col, pt.x],
      [pairInfo.y_label || axisSpec.y_col, pt.y],
      [scale ? valueLabel(scaleMode()) : headerInfo.scale_mode_label || 'value', value]
    ];
    if (bin && scale) rows.push(['count', scale.grids.counts[bin.yi][bin.xi] || 0]);
    if (bin && states) rows.push(['state (' + currentStateGroup().name + ')', stateLabelAt(bin)]);
    rows.forEach(function (row) {
      var line = document.createElement('div');
      var label = document.createElement('b');
      label.textContent = row[0] + ': ';
      line.appendChild(label);
      var text = typeof row[1] === 'string' ? row[1] : Number.isFinite(row[1]) ? formatValue(row[1]) : '—';
      line.appendChild(document.createTextNode(text));
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
    updateReadout(pt, bin);

    if (payloadsPending) {
      setStatus(loadingMessage());
      return;
    }
    if (payloadError) {
      setStatus(failedMessage());
      return;
    }

    if (!binGeo) {
      // Per-frame mode: never scan the frame table on hover (click only).
      setStatus(Number.isFinite(pt.z) ? 'Click to pin the nearest frame' : 'No frames in this bin');
      return;
    }

    var record = bin ? binMetaFor(bin.key) : null;
    if (!record) {
      setStatus('No frames in this bin' + showingSuffix());
      return;
    }
    if (previewMetaEl) previewMetaEl.innerHTML = buildMetadataHtml(record);

    var xyzText = binXyzFor(bin.key);
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

  if (payloadsPending) setStatus(loadingMessage());
  loadPayloads();

  // -------------------------------------------------------------------
  // Colour scale
  // -------------------------------------------------------------------
  var scale = pageData.scale || null;
  var scaleModes = (scale && scale.modes) || {};
  var energyUnits = Object.create(null);
  ((scale && scale.energy_units) || []).forEach(function (u) {
    energyUnits[u.key] = u;
  });
  var scaleControlsEl = document.getElementById('scale-controls');
  var segmentEls = document.querySelectorAll('.ca-segment');
  var feControlsEl = document.getElementById('fe-controls');
  var tempInput = document.getElementById('fe-temperature');
  var unitSelect = document.getElementById('fe-unit');
  var feHintEl = document.getElementById('fe-hint');
  var subtitleEl = document.getElementById('hdr-subtitle');
  var currentGrid = null;
  var shownScaleKey = null;
  var scaleFrameRequested = false;

  function validTemperature(t) {
    return typeof t === 'number' && Number.isFinite(t) && t > 0;
  }

  function scaleMode() {
    return scaleModes[uiState.scale_mode] ? uiState.scale_mode : 'log_counts';
  }

  // A unit other than kT needs a temperature; without one the map shows kT.
  function shownUnit() {
    var unit = energyUnits[uiState.unit] ? uiState.unit : 'kT';
    return unit !== 'kT' && !validTemperature(uiState.temperature) ? 'kT' : unit;
  }

  // Mirrors src.units.thermal_energy: k_B T in the unit, 1 for kT.
  function thermalEnergy(unit) {
    var kB = energyUnits[unit].k_B;
    return kB == null ? 1 : kB * uiState.temperature;
  }

  // Mirrors _scale_value_label in plots_interactive.py.
  function valueLabel(mode) {
    if (mode === 'free_energy') return 'F (' + energyUnits[shownUnit()].label + ')';
    return scaleModes[mode].value_label;
  }

  // Mirrors _hovertemplate in plots_interactive.py.
  function hoverTemplate(label, zFormat) {
    return (
      pairInfo.x_label + ': %{x:.1f}<br>' +
      pairInfo.y_label + ': %{y:.1f}<br>' +
      label + ': %{z:' + zFormat + '}<extra></extra>'
    );
  }

  function displayGrid(mode) {
    var grid = scale.grids[mode];
    if (mode !== 'free_energy') return grid;
    var factor = thermalEnergy(shownUnit());
    return grid.map(function (row) {
      return row.map(function (v) { return v === null ? null : v * factor; });
    });
  }

  function scaleKey() {
    var mode = scaleMode();
    return mode === 'free_energy' ? mode + '|' + shownUnit() + '|' + uiState.temperature : mode;
  }

  function syncScaleControls() {
    var mode = scaleMode();
    for (var i = 0; i < segmentEls.length; i++) {
      segmentEls[i].setAttribute('aria-pressed', String(segmentEls[i].dataset.scale === mode));
    }
    if (feControlsEl) feControlsEl.hidden = mode !== 'free_energy';
    if (feHintEl) {
      var wanted = energyUnits[uiState.unit] ? uiState.unit : 'kT';
      var missing = wanted !== shownUnit();
      feHintEl.hidden = !missing;
      feHintEl.textContent = missing
        ? 'Enter a temperature to show ' + energyUnits[wanted].label + '; showing kT'
        : '';
    }
    setText('hdr-scale-mode', valueLabel(mode));
    if (subtitleEl && scaleModes[mode].subtitle) subtitleEl.textContent = scaleModes[mode].subtitle;
  }

  function applyScale() {
    scaleFrameRequested = false;
    var key = scaleKey();
    if (key !== shownScaleKey && gd && window.Plotly) {
      var mode = scaleMode();
      var label = valueLabel(mode);
      currentGrid = displayGrid(mode);
      Plotly.restyle(gd, {
        z: [currentGrid],
        hovertemplate: hoverTemplate(label, scaleModes[mode].z_format),
        'colorbar.title.text': label
      }, [0]);
      shownScaleKey = key;
    }
    syncScaleControls();
    if (lastReadout) updateReadout(lastReadout.pt, lastReadout.bin);
  }

  // Coalesce typing in the temperature field to one restyle per frame.
  function requestScaleUpdate() {
    if (scaleFrameRequested) return;
    scaleFrameRequested = true;
    window.requestAnimationFrame(applyScale);
  }

  if (scale) {
    // The figure was built in this state, so nothing is restyled on load.
    currentGrid = displayGrid(scaleMode());
    shownScaleKey = scaleKey();
    (scale.energy_units || []).forEach(function (u) {
      var option = document.createElement('option');
      option.value = u.key;
      option.textContent = u.label;
      unitSelect.appendChild(option);
    });
    unitSelect.value = shownUnit();
    uiState.unit = unitSelect.value;
    if (validTemperature(uiState.temperature)) tempInput.value = String(uiState.temperature);

    for (var si = 0; si < segmentEls.length; si++) {
      segmentEls[si].addEventListener('click', function (event) {
        uiState.scale_mode = event.currentTarget.dataset.scale;
        requestScaleUpdate();
      });
    }
    tempInput.addEventListener('input', function () {
      var t = parseFloat(tempInput.value);
      uiState.temperature = validTemperature(t) ? t : null;
      requestScaleUpdate();
    });
    unitSelect.addEventListener('change', function () {
      uiState.unit = unitSelect.value;
      requestScaleUpdate();
    });
    syncScaleControls();
    scaleControlsEl.hidden = false;
  }

  // -------------------------------------------------------------------
  // State overlay (trace 1 and the layout annotations)
  // -------------------------------------------------------------------
  // States are clustered per group and their labels are not unified across
  // groups, so the overlay shows one group at a time, never a mix.
  var states = pageData.states || null;
  var statesToggle = document.getElementById('states-toggle');
  var stateGroupSelect = document.getElementById('state-group');
  var currentStateGrid = null;
  var emptyStateGroup = {name: 'all frames', bins: [], states: [], centres: []};

  function currentStateGroup() {
    return states.groups[uiState.state_group] || states.groups[0] || emptyStateGroup;
  }

  function stateGrid(group) {
    var nx = headerInfo.bin_count_x;
    var grid = [];
    for (var yi = 0; yi < headerInfo.bin_count_y; yi++) {
      var row = new Array(nx);
      for (var xi = 0; xi < nx; xi++) row[xi] = null;
      grid.push(row);
    }
    for (var i = 0; i < group.bins.length; i++) {
      grid[Math.floor(group.bins[i] / nx)][group.bins[i] % nx] = group.states[i];
    }
    return grid;
  }

  function stateLabelAt(bin) {
    var code = currentStateGrid ? currentStateGrid[bin.yi][bin.xi] : null;
    return code === null ? 'none' : states.labels[code];
  }

  // Mirrors _state_annotations in plots_interactive.py.
  function stateAnnotations(group, visible) {
    var bg = cssVar('--ca-state-label-bg');
    return group.centres.map(function (c) {
      return {
        x: c.x,
        y: c.y,
        text: states.labels[c.state],
        showarrow: false,
        font: {size: 12},
        bgcolor: bg,
        bordercolor: states.colors[c.state],
        borderwidth: 1,
        borderpad: 2,
        visible: visible
      };
    });
  }

  function syncStateControls() {
    if (statesToggle) {
      statesToggle.setAttribute('aria-pressed', String(!!uiState.state_overlay_visible));
    }
  }

  function applyStates(regrid) {
    var visible = !!uiState.state_overlay_visible;
    var group = currentStateGroup();
    var traceUpdate = {visible: visible};
    if (regrid) {
      currentStateGrid = stateGrid(group);
      traceUpdate.z = [currentStateGrid];
    }
    Plotly.update(gd, traceUpdate, {annotations: stateAnnotations(group, visible)}, [1]);
    syncStateControls();
    if (lastReadout) updateReadout(lastReadout.pt, lastReadout.bin);
  }

  if (states) {
    // The figure was built in this state, so nothing is redrawn on load.
    currentStateGrid = stateGrid(currentStateGroup());
    if (stateGroupSelect) {
      states.groups.forEach(function (g, i) {
        var option = document.createElement('option');
        option.value = String(i);
        option.textContent = g.name;
        stateGroupSelect.appendChild(option);
      });
      stateGroupSelect.value = String(states.groups[uiState.state_group] ? uiState.state_group : 0);
      stateGroupSelect.hidden = states.groups.length < 2;
      stateGroupSelect.addEventListener('change', function () {
        uiState.state_group = Number(stateGroupSelect.value);
        applyStates(true);
      });
    }
    if (statesToggle) {
      statesToggle.addEventListener('click', function () {
        uiState.state_overlay_visible = !uiState.state_overlay_visible;
        applyStates(false);
      });
    }
    syncStateControls();
  }

  // -------------------------------------------------------------------
  // Coordinate-pair navigation
  // -------------------------------------------------------------------
  // Links to the sibling pages of the same run, which sit in this folder.
  var navEl = document.getElementById('pair-nav');
  var navPairs = (pageData.navigation || {}).pairs || [];

  function renderPairNav() {
    // One pair alone has nowhere to jump to, so the nav stays hidden.
    if (!navEl || navPairs.length < 2) return;
    navPairs.forEach(function (entry) {
      var link = document.createElement('a');
      link.className = 'ca-pair-link';
      // Plain file names, so a space or other special character still resolves.
      link.href = encodeURIComponent(entry.filename);
      link.textContent = entry.title || entry.name;
      link.dataset.pair = entry.name;
      if (entry.current) link.setAttribute('aria-current', 'page');
      navEl.appendChild(link);
    });
    navEl.hidden = false;
  }

  renderPairNav();

  // -------------------------------------------------------------------
  // Degree axis ticks
  // -------------------------------------------------------------------
  var axisTicks = pageData.axis_ticks || null;

  // Mirrors _degree_tick_step in plots_interactive.py.
  function degreeTickStep(span) {
    var steps = axisTicks.steps;
    for (var i = 0; i < steps.length; i++) {
      if (Math.abs(span) / steps[i] <= axisTicks.max_intervals) return steps[i];
    }
    return steps[steps.length - 1];
  }

  // After a zoom or pan, re-pick the step for the visible range. Our own
  // update fires plotly_relayout again, but then nothing differs, so it stops.
  function updateDegreeTicks() {
    if (!axisTicks || !gd._fullLayout) return;
    var update = {};
    ['x', 'y'].forEach(function (axis) {
      if (!axisTicks[axis]) return;
      var range = gd._fullLayout[axis + 'axis'].range;
      var step = degreeTickStep(range[1] - range[0]);
      if ((gd.layout[axis + 'axis'] || {}).dtick !== step) update[axis + 'axis.dtick'] = step;
    });
    if (Object.keys(update).length) Plotly.relayout(gd, update);
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

      if (payloadsPending || payloadError) {
        showNotice(payloadError ? failedMessage() : loadingMessage());
        return;
      }

      var best = null;
      if (bin) {
        best = binMetaFor(bin.key);
      } else if (frameCount()) {
        var xs = frameColumn(axisSpec.x_col);
        var ys = frameColumn(axisSpec.y_col);
        var bestDist = Infinity;
        var bestIndex = -1;
        for (var i = 0; i < xs.length; i++) {
          if (xs[i] == null || ys[i] == null) continue;
          var dx = xs[i] - cx;
          var dy = ys[i] - cy;
          var d = dx * dx + dy * dy;
          if (d < bestDist) {
            bestDist = d;
            bestIndex = i;
          }
        }
        if (bestIndex >= 0) best = frameAt(bestIndex);
      }

      if (!best) return;

      var cardKey = getCardKey(best, xi, yi);
      if (openCards[cardKey]) {
        focusCard(openCards[cardKey]);
        return;
      }

      var xyzText = bin ? binXyzFor(bin.key) : null;

      createCard(cardKey, getCardTitle(best, xi, yi), buildMetadataHtml(best), xyzText);
    });

    if (hoverPreview) gd.on('plotly_hover', onHover);
    gd.on('plotly_relayout', updateDegreeTicks);
  }

  applyTheme();
  updateTrayVisibility();
})();
