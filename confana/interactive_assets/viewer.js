/*
 * ConfAna interactive density viewer client script.
 *
 * Reads the single embedded #page-data JSON object and wires up:
 *  - header stats (frame count, bin count, scale mode)
 *  - light/dark theme toggle, synced into Plotly and the 3Dmol viewers
 *  - live hover preview: one shared 3Dmol viewer plus a read-out and a
 *    metadata table for the bin under the cursor
 *  - pins (docs/adr/0002): clicking a bin pins its representative frame (in
 *    per-frame mode, the nearest frame). Each pin has a number, a card with
 *    its own 3Dmol viewer (no singleton #viewer3d) and a map marker: badge
 *    + arrow to the frame's spot, bin outline + dot (ring in per-frame mode).
 *    Card hover fades the other markers; map hover lights the cards of the
 *    pins in that bin; badge click shows the card. A new pin takes the lowest
 *    free number; at settings.max_pinned the oldest pin is removed. Pins
 *    carry all pairs' coordinates, and follow the pair links in the hash
 *  - axis-atom highlighting: atoms keep element colours; the atoms defining
 *    the x and y coordinates (page data axis_atoms) get ball-and-stick and a
 *    translucent halo matching the axis titles in every 3D view, with a
 *    legend in the side panel and a toggle for atom-index labels
 *  - shared camera: the preview and every card share one rotation and zoom,
 *    so turning any 3D view turns them all and new pins open at that angle
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
  // Touch behaviour follows the input, not the screen width: it is on when
  // the main input cannot hover (a finger). Then a tap previews a bin, and
  // pinning is a separate step. The layout follows the width (viewer.css).
  var touchInput = !!(window.matchMedia && window.matchMedia('(hover: none)').matches);
  document.documentElement.classList.toggle('ca-touch', touchInput);

  // -------------------------------------------------------------------
  // Payload codec — mirrors confana/payload_codec.py
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

  // Coordinates arrive atom-major, (atom, axis, structure), as a plane of
  // low bytes then a plane of high bytes (mirrors _coordinate_planes).
  // Rebuilt once into structure-major int16, (structure, atom, axis).
  function caCoordinatesFromPlanes(buffer, count, atomCount) {
    var bytes = new Uint8Array(buffer);
    var total = count * atomCount * 3;
    var coords = new Int16Array(total);
    for (var j = 0; j < total; j++) {
      var value = bytes[j] | (bytes[total + j] << 8);
      var s = j % count;
      var slot = (j - s) / count;  // atom * 3 + axis
      coords[s * atomCount * 3 + slot] = value > 32767 ? value - 65536 : value;
    }
    return coords;
  }

  function caDecodeStructures(block) {
    if (!block || block.format !== 'confana-structures-v2') {
      return Promise.reject(new Error('Not a confana-structures-v2 block.'));
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
        coords: caCoordinatesFromPlanes(parts[1], parts[0].keys.length, block.atom_count)
      };
    });
  }

  // Count grid (mirrors decode_count_grid): rows of counts, null for an
  // unsampled bin. Not gzipped, so it is read at once, on load. A plain
  // nested list (compress_payloads: false) is already in that form.
  function caCountRows(block) {
    if (Array.isArray(block)) return block;
    if (!block || block.format !== 'confana-count-grid-v1') {
      throw new Error('Not a confana-count-grid-v1 block.');
    }
    var bytes = caBytes(block.data);
    var view = new DataView(bytes.buffer);
    var width = block.dtype === 'u2' ? 2 : 4;
    var rows = [];
    for (var r = 0, at = 0; r < block.shape[0]; r++) {
      var row = new Array(block.shape[1]);
      for (var c = 0; c < block.shape[1]; c++, at += width) {
        var v = width === 2 ? view.getUint16(at, true) : view.getUint32(at, true);
        row[c] = v > 0 ? v : null;
      }
      rows.push(row);
    }
    return rows;
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
    // Plotly's default zero line is white; draw it as one more grid line.
    if ((layout.xaxis || {}).zerolinecolor !== grid) update['xaxis.zerolinecolor'] = grid;
    if ((layout.yaxis || {}).zerolinecolor !== grid) update['yaxis.zerolinecolor'] = grid;
    var labelBg = cssVar('--ca-state-label-bg');
    // Pin badges are recoloured by drawMapMarkers instead.
    (layout.annotations || []).forEach(function (a, i) {
      if (String(a.name || '').indexOf('pin-') === 0) return;
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
      if (pins.length || previewedBin) drawMapMarkers();
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
    var v;
    try {
      v = $3Dmol.createViewer(el, {backgroundColor: hexToInt(cssVar('--ca-viewer-bg'))});
    } catch (err) {
      viewerFailed = true;
      return null;
    }
    v.setViewChangeCallback(function () { onViewChange(v); });
    if (touchInput) addTouchLock(el);
    return v;
  }

  // Touch: 3Dmol claims every touch on its canvas, so a page of full-width
  // views could hardly be scrolled. Each view starts locked under a
  // see-through cover, which a swipe scrolls past. A tap on the cover
  // unlocks that one view; its Done button, or a tap anywhere else, locks it.
  var unlockedViewEl = null;

  function addTouchLock(el) {
    var cover = document.createElement('button');
    cover.type = 'button';
    cover.className = 'ca-3d-cover';
    var label = document.createElement('span');
    label.textContent = 'Tap to rotate';
    cover.appendChild(label);
    cover.addEventListener('click', function () { unlockView(el); });
    var done = document.createElement('button');
    done.type = 'button';
    done.className = 'ca-btn ca-3d-done';
    done.textContent = 'Done';
    done.addEventListener('click', lockView);
    el.appendChild(cover);
    el.appendChild(done);
  }

  function unlockView(el) {
    lockView();
    el.classList.add('ca-3d-unlocked');
    unlockedViewEl = el;
  }

  function lockView() {
    if (unlockedViewEl) unlockedViewEl.classList.remove('ca-3d-unlocked');
    unlockedViewEl = null;
  }

  // Capture phase: it runs before the tapped control's own handler.
  document.addEventListener('click', function (event) {
    if (unlockedViewEl && !unlockedViewEl.contains(event.target)) lockView();
  }, true);

  // One camera for every 3D view: turning or zooming the preview or any card
  // turns and zooms them all, and a new structure opens in that camera, so
  // pins compare with the preview and with each other. Only rotation and
  // zoom are shared (getView entries 3-7); each viewer centres its own
  // structure, which also holds when alignment is off and molecules drift.
  // A viewer joins once showStructure has put the shared camera on it
  // (_caSynced): until then its renders (creation, resize, reuse from the
  // pool) carry a stale camera and are ignored.
  var sharedView = null;
  var syncingViews = false;

  function cameraOf(v) {
    return v.getView().slice(3, 8);
  }

  function sameCamera(a, b) {
    for (var i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > 1e-9) return false;
    }
    return true;
  }

  function applySharedView(v) {
    var view = v.getView();
    for (var i = 0; i < sharedView.length; i++) view[3 + i] = sharedView[i];
    v.setView(view);
  }

  // setView re-renders, which calls back here: syncingViews stops the echo.
  function onViewChange(v) {
    if (syncingViews || !v._caSynced) return;
    var camera = cameraOf(v);
    if (sharedView && sameCamera(camera, sharedView)) return;
    sharedView = camera;
    syncingViews = true;
    try {
      attachedViewers().forEach(function (other) {
        if (other !== v && other._caSynced) applySharedView(other);
      });
    } finally {
      syncingViews = false;
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

  // Colour means element everywhere; the axis is shown by a translucent
  // halo in its colour around each axis atom, which is drawn as
  // ball-and-stick with thicker bonds. Halos are shapes, not atom styles:
  // 3Dmol shares sphere/stick opacity across a whole model.
  var ELEMENT_SCHEME = 'Jmol';
  var STICK_RADIUS = 0.12;
  var AXIS_STICK_RADIUS = 0.2;
  var AXIS_SPHERE_RADIUS = 0.32;
  var HALO_RADIUS = 0.62;
  var HALO_OPACITY = 0.6;
  var showAtomIndices = false;

  function styleStructure(v) {
    v.removeAllShapes();
    v.removeAllLabels();
    v.setStyle({}, {stick: {colorscheme: ELEMENT_SCHEME, radius: STICK_RADIUS}});
    // Test hook: which halo each atom carries, since shapes hide their spec.
    v._caHalos = [];
    (highlightGroups || []).forEach(function (group) {
      var color = cssVar(group.colorVar);
      var sel = atomSelection(group.atoms);
      v.setStyle(sel, {
        stick: {colorscheme: ELEMENT_SCHEME, radius: AXIS_STICK_RADIUS},
        sphere: {colorscheme: ELEMENT_SCHEME, radius: AXIS_SPHERE_RADIUS}
      });
      v.selectedAtoms(sel).forEach(function (atom) {
        v.addSphere({
          center: {x: atom.x, y: atom.y, z: atom.z},
          radius: HALO_RADIUS,
          color: color,
          opacity: HALO_OPACITY
        });
        v._caHalos.push({index: atom.index, color: color});
      });
    });
    if (showAtomIndices) labelAtoms(v);
  }

  // Every atom, not only axis atoms: a wrong mapping is fixed by reading
  // off the index of the atom that should have been used.
  function labelAtoms(v) {
    var fg = cssVar('--ca-fg');
    var bg = cssVar('--ca-bg-elevated');
    v.selectedAtoms({}).forEach(function (atom) {
      v.addLabel(String(atom.index), {
        position: {x: atom.x, y: atom.y, z: atom.z},
        fontSize: 11,
        fontColor: fg,
        backgroundColor: bg,
        backgroundOpacity: 0.7,
        borderThickness: 0,
        inFront: true,
        alignment: 'center'
      });
    });
  }

  // zoomTo centres the new structure; the first structure shown on the page
  // also sets the starting zoom, and every later one takes the shared camera.
  function showStructure(v, xyzText) {
    syncingViews = true;
    try {
      v.clear();
      v.addModel(xyzText, 'xyz');
      styleStructure(v);
      v.zoomTo();
      if (sharedView) applySharedView(v);
      else sharedView = cameraOf(v);
      v._caSynced = true;
    } finally {
      syncingViews = false;
    }
    v.render();
  }

  // The card must already be in the DOM: 3Dmol sizes against the visible box.
  function takeViewerBox(card) {
    var box = viewerPool.pop();
    if (box) {
      card.appendChild(box);
      // Resize before anything renders: off the page the canvas shrank to
      // 0x0, and Firefox throws when 3Dmol renders into a 0x0 canvas.
      box._viewer3d.resize();
      box._viewer3d.setBackgroundColor(hexToInt(cssVar('--ca-viewer-bg')));
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
    if (box === unlockedViewEl) lockView();
    if (box._viewer3d) {
      box._viewer3d._caSynced = false;
      box._viewer3d.clear();
    }
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
  // Pins (docs/adr/0002). A pin is one frame: a numbered card in the panel
  // and a numbered marker on the map at the frame's own coordinates. Pins
  // travel to the run's other pair pages in the link hash.
  // -------------------------------------------------------------------
  var trayEl = document.getElementById('comparison-tray');
  var cardsEl = document.getElementById('comparison-cards');
  var clearBtn = document.getElementById('comparison-clear');
  var noticeEl = document.getElementById('panel-notice');
  var noticeTimer = null;
  var grid = pageData.grid || binGeo || null;
  // Creation order, oldest first: the pin limit removes pins[0].
  var pins = [];
  var focusedPinNumber = null;
  var hoveredBinKey = null;
  // Touch only: the bin of the last tap ({xi, yi, key}), which the map marks.
  var previewedBin = null;
  var PIN_OFFSET = 22;  // px from the frame's spot to its badge
  var PIN_EDGE = 0.85;  // past this share of the visible range the badge flips inward
  var PIN_FADED = 0.3;

  function updateTrayVisibility() {
    if (!trayEl) return;
    trayEl.style.display = pins.length ? 'block' : 'none';
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

  function focusCard(card) {
    if (!card) return;
    card.scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'nearest'});
    card.classList.add('ca-card-focus');
    window.setTimeout(function () {
      card.classList.remove('ca-card-focus');
    }, 800);
  }

  // The same on every page of a run: they all embed the same frame_id.
  function frameKey(record) {
    if (record.frame_id != null) return 'frame:' + record.frame_id;
    return 'frame:' + [record.source_file, record.bead_id, record.frame_number, record.byte_offset]
      .map(function (v) { return v == null ? 'na' : v; })
      .join('|');
  }

  function findPin(key) {
    for (var i = 0; i < pins.length; i++) if (pins[i].key === key) return pins[i];
    return null;
  }

  function pinByNumber(number) {
    for (var i = 0; i < pins.length; i++) if (pins[i].number === number) return pins[i];
    return null;
  }

  function lowestFreeNumber() {
    var n = 1;
    while (pinByNumber(n)) n++;
    return n;
  }

  function byNumber(a, b) {
    return a.number - b.number;
  }

  // Bins follow numpy.histogram2d: the upper edge belongs to the last bin.
  function binIndex(value, min, width, count) {
    var i = Math.floor((value - min) / width);
    if (i === count && Math.abs(value - (min + count * width)) <= 1e-9 * width) i = count - 1;
    return i >= 0 && i < count ? i : null;
  }

  // Where the frame sits on this map, or a note saying why it is not shown.
  function pinPlace(meta) {
    var x = meta[axisSpec.x_col];
    var y = meta[axisSpec.y_col];
    var missing = [];
    if (!Number.isFinite(x)) missing.push(axisSpec.x_col);
    if (!Number.isFinite(y)) missing.push(axisSpec.y_col);
    if (missing.length) return {note: 'Not on this map (no ' + missing.join(' or ') + ' value)'};
    if (!grid) return {x: x, y: y, bin: null};
    var xi = binIndex(x, grid.x_min, grid.bin_w, grid.n_bins_x);
    var yi = binIndex(y, grid.y_min, grid.bin_h, grid.n_bins_y);
    if (xi === null || yi === null) return {note: 'Not on this map (outside its range)'};
    return {x: x, y: y, bin: {xi: xi, yi: yi, key: xi + '_' + yi}};
  }

  function pinTitle(pin) {
    var meta = pin.meta;
    var frame = meta.frame_id != null ? 'frame ' + meta.frame_id
      : meta.frame_number != null ? 'frame ' + meta.frame_number
      : 'structure';
    var title = pin.place.bin ? 'Bin ' + pin.place.bin.key + ' · ' + frame : frame;
    return title.charAt(0).toUpperCase() + title.slice(1);
  }

  function buildPinCard(pin) {
    var card = document.createElement('div');
    card.className = 'ca-card';
    card.dataset.cardKey = pin.key;
    card.dataset.pin = String(pin.number);
    card.dataset.bin = pin.place.bin ? pin.place.bin.key : '';

    var header = document.createElement('div');
    header.className = 'ca-card-header';

    var titleEl = document.createElement('div');
    titleEl.className = 'ca-card-title';
    var badge = document.createElement('span');
    badge.className = 'ca-pin-badge';
    badge.textContent = String(pin.number);
    var name = document.createElement('b');
    name.textContent = pinTitle(pin);
    titleEl.appendChild(badge);
    titleEl.appendChild(name);

    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'ca-btn';
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', function () {
      removePin(pin.key);
    });

    header.appendChild(titleEl);
    header.appendChild(closeBtn);
    card.appendChild(header);

    if (pin.place.note) {
      var note = document.createElement('div');
      note.className = 'ca-pin-note';
      note.textContent = pin.place.note;
      card.appendChild(note);
    }

    // Frame metadata is secondary: collapsed until the reader asks for it.
    var more = document.createElement('details');
    more.className = 'ca-more';
    var summary = document.createElement('summary');
    summary.textContent = 'More info';
    var metaEl = document.createElement('div');
    metaEl.className = 'ca-card-meta';
    metaEl.innerHTML = buildMetadataHtml(pin.meta);
    more.appendChild(summary);
    more.appendChild(metaEl);
    card.appendChild(more);

    // Hover effects need a mouse: on a touch screen a tap fires mouseenter too.
    if (!touchInput) {
      card.addEventListener('mouseenter', function () { setFocusedPin(pin.number); });
      card.addEventListener('mouseleave', function () { setFocusedPin(null); });
    }
    return card;
  }

  // Cards are kept in pin-number order.
  function insertCard(card, number) {
    var next = null;
    for (var c = cardsEl.firstElementChild; c; c = c.nextElementSibling) {
      if (Number(c.dataset.pin) > number) {
        next = c;
        break;
      }
    }
    cardsEl.insertBefore(card, next);
  }

  // Adds a pin without redrawing; the caller runs pinsChanged() once after.
  // `number` keeps a pin's number when it arrives from another page; `quiet`
  // skips the scroll-and-flash.
  function addPin(meta, xyzText, number, quiet) {
    var evicted = false;
    while (pins.length >= maxPinned) {
      dropPin(pins[0]);
      evicted = true;
    }
    if (evicted) {
      showNotice('Pin limit (' + maxPinned + ') reached — removed the oldest pinned structure.');
    }
    var pin = {
      key: frameKey(meta),
      number: number && !pinByNumber(number) ? number : lowestFreeNumber(),
      meta: meta,
      xyz: xyzText || null,
      place: pinPlace(meta)
    };
    pin.card = buildPinCard(pin);
    insertCard(pin.card, pin.number);
    pins.push(pin);
    updateTrayVisibility();
    // The card must be visible first: 3Dmol sizes against the visible box.
    // A 3D failure must not leave the pin half-made (card without marker).
    if (pin.xyz) {
      var box = null;
      try {
        box = takeViewerBox(pin.card);
        if (box._viewer3d) showStructure(box._viewer3d, pin.xyz);
      } catch (err) {
        if (box) {
          box._viewer3d = null;
          showViewerError(box);
        }
      }
    }
    // Only the wide layout scrolls to the card: there the side panel scrolls.
    // In the narrow layout it would move the whole page away from the map.
    if (!quiet && !isNarrow()) focusCard(pin.card);
    return pin;
  }

  function dropPin(pin) {
    releaseViewerBox(pin.card);
    if (pin.card.parentNode) pin.card.parentNode.removeChild(pin.card);
    pins.splice(pins.indexOf(pin), 1);
    if (focusedPinNumber === pin.number) focusedPinNumber = null;
  }

  function pinsChanged() {
    updateTrayVisibility();
    updatePinButton();
    // The mouse may rest on the bin just pinned: no new hover event comes.
    linkCardsToBin(hoveredBinKey);
    drawMapMarkers();
    syncPinHash();
  }

  function removePin(key) {
    var pin = findPin(key);
    if (!pin) return;
    dropPin(pin);
    pinsChanged();
  }

  function clearAllPins() {
    if (!pins.length) return;
    while (pins.length) dropPin(pins[0]);
    pinsChanged();
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      clearAllPins();
    });
  }

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') clearAllPins();
  });

  // The frame a point on the map pins: its bin's representative frame, or
  // in per-frame mode the frame nearest the point. {meta, xyz} or null.
  function pinCandidate(pt) {
    if (binGeo) {
      var bin = binFromPoint(pt);
      var meta = bin ? binMetaFor(bin.key) : null;
      return meta ? {meta: meta, xyz: binXyzFor(bin.key)} : null;
    }
    if (!frameCount()) return null;
    var xs = frameColumn(axisSpec.x_col);
    var ys = frameColumn(axisSpec.y_col);
    var bestDist = Infinity;
    var bestIndex = -1;
    for (var i = 0; i < xs.length; i++) {
      if (xs[i] == null || ys[i] == null) continue;
      var dx = xs[i] - pt.x;
      var dy = ys[i] - pt.y;
      var d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        bestIndex = i;
      }
    }
    return bestIndex >= 0 ? {meta: frameAt(bestIndex), xyz: null} : null;
  }

  // --- Pin markers on the map ---
  function pinOpacity(pin) {
    return focusedPinNumber === null || focusedPinNumber === pin.number ? 1 : PIN_FADED;
  }

  function pinFontSize(pin) {
    return focusedPinNumber === pin.number ? 13 : 11;
  }

  // Badges sit up-right of their spot and flip inward near the right or top
  // edge of the visible range, so they stay inside the plot.
  function badgeOffset(place) {
    var fl = gd._fullLayout || {};
    function share(value, axis) {
      var range = fl[axis] && fl[axis].range;
      return range ? (value - range[0]) / (range[1] - range[0]) : 0.5;
    }
    return {
      ax: share(place.x, 'xaxis') > PIN_EDGE ? -PIN_OFFSET : PIN_OFFSET,
      ay: share(place.y, 'yaxis') > PIN_EDGE ? PIN_OFFSET : -PIN_OFFSET
    };
  }

  function pixelCircle(x, y, radius, style) {
    var shape = {
      type: 'circle', xref: 'x', yref: 'y', xsizemode: 'pixel', ysizemode: 'pixel',
      xanchor: x, yanchor: y, x0: -radius, x1: radius, y0: -radius, y1: radius, layer: 'above'
    };
    for (var k in style) shape[k] = style[k];
    return shape;
  }

  // Per placed pin: a halo annotation (a wide arrow in the halo colour, so
  // the thin arrow shows on any map colour), the badge annotation, and the
  // shapes: bin outline + dot at the frame's spot, or in per-frame mode a
  // ring. They follow the state labels, whose count sets the first index.
  function pinLayout(firstAnnotation) {
    var mark = cssVar('--ca-pin');
    var text = cssVar('--ca-pin-contrast');
    var halo = cssVar('--ca-pin-halo');
    var clear = 'rgba(0,0,0,0)';
    var annotations = [];
    var shapes = [];
    pins.slice().sort(byNumber).forEach(function (pin) {
      var place = pin.place;
      pin.annotationIndices = [];
      pin.shapeIndices = [];
      pin.offset = null;
      if (place.x === undefined) return;
      var offset = badgeOffset(place);
      var opacity = pinOpacity(pin);
      pin.offset = offset;
      function annotation(extra) {
        var a = {
          x: place.x, y: place.y, xref: 'x', yref: 'y', ax: offset.ax, ay: offset.ay,
          text: '<b>' + pin.number + '</b>', showarrow: true, arrowhead: 0, standoff: 4,
          borderpad: 2, borderwidth: 1, opacity: opacity
        };
        for (var k in extra) a[k] = extra[k];
        pin.annotationIndices.push(firstAnnotation + annotations.length);
        annotations.push(a);
      }
      annotation({
        name: 'pin-halo-' + pin.number,
        font: {size: pinFontSize(pin), color: clear},
        bgcolor: clear, bordercolor: clear, arrowcolor: halo, arrowwidth: 4
      });
      annotation({
        name: 'pin-' + pin.number,
        font: {size: pinFontSize(pin), color: text},
        bgcolor: mark, bordercolor: halo, arrowcolor: mark, arrowwidth: 1.5,
        captureevents: true, hovertext: 'Pin ' + pin.number + ' — click to show its card'
      });
      function shape(s) {
        s.opacity = opacity;
        s.name = 'pin-' + pin.number;
        pin.shapeIndices.push(shapes.length);
        shapes.push(s);
      }
      if (binGeo && place.bin) {
        var x0 = grid.x_min + place.bin.xi * grid.bin_w;
        var y0 = grid.y_min + place.bin.yi * grid.bin_h;
        shape({
          type: 'rect', xref: 'x', yref: 'y', x0: x0, x1: x0 + grid.bin_w, y0: y0, y1: y0 + grid.bin_h,
          line: {color: mark, width: 1.5}, fillcolor: clear, layer: 'above'
        });
        shape(pixelCircle(place.x, place.y, 3, {fillcolor: mark, line: {color: halo, width: 1}}));
      } else {
        shape(pixelCircle(place.x, place.y, 6, {fillcolor: clear, line: {color: halo, width: 4}}));
        shape(pixelCircle(place.x, place.y, 6, {fillcolor: clear, line: {color: mark, width: 2}}));
      }
    });
    return {annotations: annotations, shapes: shapes};
  }

  // The map's annotations: state labels first, then the pin badges.
  function mapAnnotations() {
    var stateLabels = states ? stateAnnotations(currentStateGroup(), !!uiState.state_overlay_visible) : [];
    return {stateLabels: stateLabels, pins: pinLayout(stateLabels.length)};
  }

  // Touch only: the outline of the bin the last tap landed on, so the reader
  // sees where the finger hit. Dashed, unlike a pin's solid outline, with the
  // same halo so it shows on any map colour. After the pin shapes, whose
  // indices setFocusedPin relies on.
  function previewMarkerShapes() {
    if (!previewedBin || !grid) return [];
    var x0 = grid.x_min + previewedBin.xi * grid.bin_w;
    var y0 = grid.y_min + previewedBin.yi * grid.bin_h;
    function outline(line) {
      return {
        type: 'rect', name: 'preview', xref: 'x', yref: 'y', layer: 'above',
        x0: x0, x1: x0 + grid.bin_w, y0: y0, y1: y0 + grid.bin_h,
        fillcolor: 'rgba(0,0,0,0)', line: line
      };
    }
    return [
      outline({color: cssVar('--ca-pin-halo'), width: 4}),
      outline({color: cssVar('--ca-pin'), width: 2, dash: 'dash'})
    ];
  }

  function drawMapMarkers() {
    if (!gd || !window.Plotly) return;
    var layout = mapAnnotations();
    Plotly.relayout(gd, {
      annotations: layout.stateLabels.concat(layout.pins.annotations),
      shapes: layout.pins.shapes.concat(previewMarkerShapes())
    });
  }

  // Card hover: bold badge for this pin, the others faded. Only the changed
  // items are edited, which is much cheaper than redrawing every marker.
  function setFocusedPin(number) {
    if (number === focusedPinNumber || !gd || !window.Plotly) return;
    focusedPinNumber = number;
    var update = {};
    pins.forEach(function (pin) {
      var opacity = pinOpacity(pin);
      (pin.annotationIndices || []).forEach(function (i) {
        update['annotations[' + i + '].opacity'] = opacity;
        update['annotations[' + i + '].font.size'] = pinFontSize(pin);
      });
      (pin.shapeIndices || []).forEach(function (i) {
        update['shapes[' + i + '].opacity'] = opacity;
      });
    });
    if (Object.keys(update).length) Plotly.relayout(gd, update);
  }

  // Map hover: the cards of the pins in the hovered bin glow.
  function linkCardsToBin(binKey) {
    // A tap is not a hover: touch screens have no hover links.
    if (touchInput) return;
    hoveredBinKey = binKey;
    pins.forEach(function (pin) {
      var linked = !!binKey && !!pin.place.bin && pin.place.bin.key === binKey;
      pin.card.classList.toggle('ca-card-linked', linked);
    });
  }

  // After a zoom or pan, redraw only if some badge has to flip.
  function onPinRelayout(event) {
    if (!event || !pins.length) return;
    var ranged = Object.keys(event).some(function (k) { return /^[xy]axis\.(range|autorange)/.test(k); });
    if (!ranged) return;
    var flipped = pins.some(function (pin) {
      if (!pin.offset) return false;
      var offset = badgeOffset(pin.place);
      return offset.ax !== pin.offset.ax || offset.ay !== pin.offset.ay;
    });
    if (flipped) drawMapMarkers();
  }

  // --- Pins in the link hash ---
  // '#pins=' + base64url(gzip(JSON)); the JSON lists the pins oldest first,
  // each with its number, metadata record (all pairs' coordinates) and
  // structure text. Kept in step with the pins, so reload keeps them.
  var PIN_HASH = '#pins=';
  var hashJob = null;
  var navLinks = [];

  function base64UrlFromBytes(bytes) {
    var binary = '';
    for (var i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  function base64FromUrl(text) {
    var b64 = text.replace(/-/g, '+').replace(/_/g, '/');
    while (b64.length % 4) b64 += '=';
    return b64;
  }

  function gzipBase64Url(text) {
    if (typeof CompressionStream === 'undefined') {
      return Promise.reject(new Error('This browser has no CompressionStream(gzip).'));
    }
    var stream = new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'));
    return new Response(stream).arrayBuffer().then(function (buffer) {
      return base64UrlFromBytes(new Uint8Array(buffer));
    });
  }

  function pinHashPayload() {
    return JSON.stringify({
      v: 1,
      pins: pins.map(function (pin) { return {n: pin.number, m: pin.meta, x: pin.xyz}; })
    });
  }

  function setLocationHash(hash) {
    var url = window.location.href.split('#')[0] + hash;
    try {
      window.history.replaceState(window.history.state, '', url);
    } catch (err) {
      window.location.replace(url);
    }
  }

  function syncPinHash() {
    var job = pins.length
      ? gzipBase64Url(pinHashPayload()).then(function (s) { return PIN_HASH + s; })
      : Promise.resolve('');
    hashJob = job;
    job.then(
      function (hash) {
        if (hashJob !== job) return;
        setLocationHash(hash);
        navLinks.forEach(function (link) {
          link.href = encodeURIComponent(link.dataset.filename) + hash;
        });
        hashJob = null;
      },
      function () {
        if (hashJob !== job) return;
        hashJob = null;
        showNotice('Could not store the pins in the link; they will not follow you to other pages.');
      }
    );
  }

  // A pair link clicked while the hash is still being written waits for it.
  function followWhenHashReady(event) {
    if (!hashJob) return;
    event.preventDefault();
    var link = event.currentTarget;
    (function wait() {
      var job = hashJob;
      if (!job) {
        window.location.href = link.href;
        return;
      }
      job.then(wait, wait);
    })();
  }

  function restorePinsFromHash() {
    var hash = window.location.hash;
    if (hash.indexOf(PIN_HASH) !== 0) return;
    caGunzipJson(base64FromUrl(hash.slice(PIN_HASH.length))).then(function (data) {
      if (!data || data.v !== 1 || !Array.isArray(data.pins)) throw new Error('Unknown pin format.');
      data.pins.forEach(function (p) {
        if (p && p.m && !findPin(frameKey(p.m))) addPin(p.m, p.x, p.n, true);
      });
      pinsChanged();
    }).catch(function () {
      showNotice('Could not read the pins carried in the link.');
    });
  }

  // -------------------------------------------------------------------
  // Preview (hovering with a mouse, tapping on a touch screen)
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

  function idleStatus() {
    if (touchInput) return hasStructures ? 'Tap a bin to preview it' : 'Tap a bin to read its values';
    return hasStructures ? 'Hover over the map to preview a bin' : 'Hover over the map';
  }

  // Both run once, when the compressed blocks are unpacked (or fail to be).
  function onPayloadsReady() {
    setStatus(idleStatus());
    // The cursor may already sit on a bin, or a tap came early: redo it.
    currentHoverKey = null;
    if (pendingPoint) {
      if (touchInput) previewedFrame = tappedFrame(pendingPoint);
      processHover();
      updatePinButton();
    }
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
    linkCardsToBin(bin ? bin.key : null);
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
      var pinHint = touchInput ? 'Press Pin to pin the nearest frame' : 'Click to pin the nearest frame';
      setStatus(Number.isFinite(pt.z) ? pinHint : 'No frames in this bin');
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

  // Touch: a tap previews the bin it lands on, as hovering does with a mouse,
  // and the map marks that bin.
  function previewTap(pt) {
    pendingPoint = pt;
    previewedBin = binFromPoint(pt);
    previewedFrame = tappedFrame(pt);
    processHover();
    updatePinButton();
    drawMapMarkers();
  }

  // Touch: the Pin button pins the previewed frame, as a click does with a
  // mouse, then names the pin. The page stays where it is.
  var previewPinBtn = document.getElementById('preview-pin');
  // The frame the Pin button pins ({meta, xyz}), or null.
  var previewedFrame = null;

  // In per-frame mode an empty bin has nothing to pin, though some frame
  // elsewhere is nearest (a mouse click still pins that one).
  function tappedFrame(pt) {
    if (payloadsPending || payloadError) return null;
    if (!binGeo && !Number.isFinite(pt.z)) return null;
    return pinCandidate(pt);
  }

  function updatePinButton() {
    if (!previewPinBtn) return;
    previewPinBtn.hidden = !previewedFrame;
    if (!previewedFrame) return;
    var existing = findPin(frameKey(previewedFrame.meta));
    previewPinBtn.textContent = existing ? 'Pinned · ' + existing.number : 'Pin';
    previewPinBtn.disabled = !!existing;
  }

  if (previewPinBtn) {
    previewPinBtn.addEventListener('click', function () {
      if (previewedFrame) pinFrame(previewedFrame, true);
    });
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
    var toggle = document.createElement('label');
    toggle.className = 'ca-legend-toggle';
    var box = document.createElement('input');
    box.type = 'checkbox';
    box.id = 'atom-index-toggle';
    box.addEventListener('change', function () {
      showAtomIndices = box.checked;
      attachedViewers().forEach(function (v) {
        styleStructure(v);
        v.render();
      });
    });
    toggle.appendChild(box);
    toggle.appendChild(document.createTextNode(' Show atom indices in 3D'));
    legendEl.appendChild(toggle);
    legendEl.hidden = false;
  }

  renderAxisLegend();

  // Un-hide before any viewer is created, so 3Dmol measures a visible box.
  if (hoverPreview) {
    if (previewMetaBox) previewMetaBox.hidden = false;
    if (previewBox && hasStructures) previewBox.hidden = false;
  }

  setStatus(payloadsPending ? loadingMessage() : idleStatus());
  loadPayloads();

  // -------------------------------------------------------------------
  // Colour scale
  // -------------------------------------------------------------------
  var scale = pageData.scale || null;
  if (scale) scale.grids = scaleGrids(caCountRows(scale.counts), scale.grid_decimals);

  // The grid of every scale mode, from the counts (mirrors _scale_grids in
  // plots_interactive.py and population_free_energy in density.py).
  function scaleGrids(counts, decimals) {
    var factor = Math.pow(10, decimals);
    function rounded(v) { return Math.round(v * factor) / factor; }
    var top = 0;
    counts.forEach(function (row) {
      row.forEach(function (c) { if (c !== null && c > top) top = c; });
    });
    function derive(fn) {
      return counts.map(function (row) {
        return row.map(function (c) { return c === null ? null : rounded(fn(c)); });
      });
    }
    return {
      counts: counts,
      log_counts: derive(function (c) { return Math.log10(c + 1); }),
      free_energy: derive(function (c) { return Math.log(top / c); })
    };
  }
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

  // Mirrors confana.units.thermal_energy: k_B T in the unit, 1 for kT.
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
    syncControlsToggle();
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
    // Invisible copies of every subtitle reserve the width of the longest.
    if (subtitleEl && subtitleEl.parentNode) {
      Object.keys(scaleModes).forEach(function (m) {
        if (!scaleModes[m].subtitle) return;
        var sizer = document.createElement('span');
        sizer.className = 'ca-subtitle ca-subtitle-sizer';
        sizer.setAttribute('aria-hidden', 'true');
        sizer.textContent = scaleModes[m].subtitle;
        subtitleEl.parentNode.appendChild(sizer);
      });
    }
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
    syncControlsToggle();
  }

  function applyStates(regrid) {
    var visible = !!uiState.state_overlay_visible;
    var group = currentStateGroup();
    var traceUpdate = {visible: visible};
    if (regrid) {
      currentStateGrid = stateGrid(group);
      traceUpdate.z = [currentStateGrid];
    }
    // The pin badges follow the state labels, so both are rebuilt together.
    var annotations = mapAnnotations();
    Plotly.update(gd, traceUpdate, {
      annotations: annotations.stateLabels.concat(annotations.pins.annotations)
    }, [1]);
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
  // Header controls button (narrow layout)
  // -------------------------------------------------------------------
  // In the narrow layout the controls sit behind one button, which opens
  // them in place and names the current settings, so a visitor sees there
  // is something to change. Wide layouts never show it.
  var headerEl = document.querySelector('.ca-header');
  var controlsToggle = document.getElementById('controls-toggle');

  function syncControlsToggle() {
    if (!controlsToggle) return;
    var parts = ['Controls'];
    if (scale) parts.push(scaleModes[scaleMode()].label);
    if (states) parts.push(uiState.state_overlay_visible ? 'states on' : 'states off');
    controlsToggle.textContent = parts.join(' · ');
  }

  if (controlsToggle && headerEl) {
    controlsToggle.addEventListener('click', function () {
      var open = !headerEl.classList.contains('ca-controls-open');
      headerEl.classList.toggle('ca-controls-open', open);
      controlsToggle.setAttribute('aria-expanded', String(open));
    });
  }
  syncControlsToggle();

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
      // syncPinHash appends the pins to every link, so they follow the reader.
      link.dataset.filename = entry.filename;
      link.addEventListener('click', followWhenHashReady);
      navLinks.push(link);
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
  // Map size
  // -------------------------------------------------------------------
  // The figure is built at its desktop size. In the narrow layout it shrinks
  // to the column's width with the same shape, never growing past the
  // desktop size. The header already names the map, so the figure's own
  // title goes, and a thinner colour bar leaves the plot more of the width.
  // The right margin stays fixed, as on desktop, so a scale switch never
  // resizes the plot.
  var mapEl = document.querySelector('.ca-map');
  var narrowQuery = window.matchMedia ? window.matchMedia('(max-width: 900px)') : null;
  var NARROW_COLORBAR_PX = 14;
  var NARROW_MARGIN = {l: 60, r: 96, t: 16, b: 50};
  var wideFigure = null;
  var fitFrameRequested = false;

  function isNarrow() {
    return !!(narrowQuery && narrowQuery.matches);
  }

  function columnWidth() {
    var style = getComputedStyle(mapEl);
    return Math.floor(mapEl.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight));
  }

  function fitMap() {
    fitFrameRequested = false;
    if (!gd || !window.Plotly || !mapEl) return;
    if (!wideFigure) {
      wideFigure = {
        width: gd.layout.width,
        height: gd.layout.height,
        margin: gd.layout.margin || {},
        title: (gd.layout.title || {}).text || '',
        colorbar: (gd.data[0].colorbar || {}).thickness
      };
    }
    var available = isNarrow() ? columnWidth() : Infinity;
    var fit = available < wideFigure.width;
    var width = fit ? available : wideFigure.width;
    var height = fit ? Math.round(width * wideFigure.height / wideFigure.width) : wideFigure.height;
    if (gd.layout.width === width && gd.layout.height === height) return;
    // The exported figure carries the build size as inline styles too: on the
    // graph div itself (Plotly.py 6) or on a wrapper div around it (Plotly.py
    // 7). Resize every element up to the column that has one, or the old size
    // still sticks out and a phone zooms the whole page out to show it.
    for (var el = gd; el && el !== mapEl; el = el.parentElement) {
      if (el.style.width) el.style.width = width + 'px';
      if (el.style.height) el.style.height = height + 'px';
    }
    Plotly.update(gd, {
      'colorbar.thickness': fit ? NARROW_COLORBAR_PX : wideFigure.colorbar == null ? null : wideFigure.colorbar
    }, {
      width: width,
      height: height,
      margin: fit ? NARROW_MARGIN : wideFigure.margin,
      'title.text': fit ? '' : wideFigure.title
    }, [0]);
  }

  window.addEventListener('resize', function () {
    if (fitFrameRequested) return;
    fitFrameRequested = true;
    window.requestAnimationFrame(fitMap);
  });

  // -------------------------------------------------------------------
  // Clicks and taps on the map
  // -------------------------------------------------------------------
  // A click pins the frame at the point (with a mouse, or on a touch page
  // without a preview). A pin is a frame: the same frame again only shows
  // its card.
  function pinAt(pt) {
    if (payloadsPending || payloadError) {
      showNotice(payloadError ? failedMessage() : loadingMessage());
      return;
    }
    var frame = pinCandidate(pt);
    if (frame) pinFrame(frame, false);
  }

  // Pins a frame ({meta, xyz}). If it is pinned already, a click shows its
  // card; `quiet` (the Pin button) leaves the page where it is.
  function pinFrame(frame, quiet) {
    var existing = findPin(frameKey(frame.meta));
    if (existing) {
      if (!quiet) focusCard(existing.card);
      return;
    }
    addPin(frame.meta, frame.xyz, null, quiet);
    pinsChanged();
  }

  // Touch: the bin under a tap, read from the tap's own position, in the
  // form of a Plotly heatmap point (bin centre, value, [row, col]). Plotly's
  // click names its last hover point, which a tap does not move once the map
  // cannot be dragged, so it would name the previous tap's bin.
  function pointAtTap(event) {
    var fl = gd._fullLayout;
    var size = fl && fl._size;
    if (!size || !grid) return null;
    var rect = gd.getBoundingClientRect();
    var px = event.clientX - rect.left - size.l;
    var py = event.clientY - rect.top - size.t;
    if (px < 0 || py < 0 || px > size.w || py > size.h) return null;
    var xr = fl.xaxis.range;
    var yr = fl.yaxis.range;
    var xi = binIndex(xr[0] + px / size.w * (xr[1] - xr[0]), grid.x_min, grid.bin_w, grid.n_bins_x);
    var yi = binIndex(yr[1] - py / size.h * (yr[1] - yr[0]), grid.y_min, grid.bin_h, grid.n_bins_y);
    if (xi === null || yi === null) return null;
    var z = gd._fullData[0].z[yi][xi];
    return {
      x: grid.x_min + (xi + 0.5) * grid.bin_w,
      y: grid.y_min + (yi + 0.5) * grid.bin_h,
      z: z === null || Number.isNaN(z) ? null : z,
      pointNumber: [yi, xi]
    };
  }

  // Touch: a tap previews, never pins, and two taps are two previews (a
  // double tap clearing every pin would have no undo). A tap on a pin badge
  // is the badge's (plotly_clickannotation shows its card).
  function onTap(event) {
    // Plotly adds two synthetic clicks of its own to each tap (one at 0, 0);
    // only the browser's own click counts, or a tap would act three times.
    if (!event.isTrusted) return;
    if (event.target.closest && event.target.closest('.annotation')) return;
    var pt = pointAtTap(event);
    if (!pt) return;
    if (hoverPreview) previewTap(pt);
    else pinAt(pt);
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
      // Touch screens read taps in onTap instead.
      if (touchInput) return;
      // Double-click clears the pins. Use the browser's click count, which
      // requires both clicks in the same spot: plotly_doubleclick fires for
      // any two clicks within 300 ms, even on different bins, and would wipe
      // the pins when clicking quickly.
      if (data.event && data.event.detail >= 2) {
        clearAllPins();
        return;
      }
      pinAt(data.points[0]);
    });

    if (touchInput) gd.addEventListener('click', onTap);

    // A pin badge is clicked: show that pin's card.
    gd.on('plotly_clickannotation', function (data) {
      var match = /^pin-(\d+)$/.exec((data && data.annotation && data.annotation.name) || '');
      var pin = match ? pinByNumber(Number(match[1])) : null;
      if (pin) focusCard(pin.card);
    });

    if (hoverPreview && !touchInput) gd.on('plotly_hover', onHover);
    gd.on('plotly_unhover', function () { linkCardsToBin(null); });
    gd.on('plotly_relayout', function (event) {
      updateDegreeTicks();
      onPinRelayout(event);
    });
  }

  // Touch: a drag on the map scrolls the page, and the map never zooms or
  // pans. Plotly's default drag draws a zoom box and holds the page still;
  // its zoom tools are hidden too (viewer.css).
  if (touchInput && gd && window.Plotly) Plotly.relayout(gd, {dragmode: false});
  fitMap();
  applyTheme();
  updateTrayVisibility();
  restorePinsFromHash();
})();
