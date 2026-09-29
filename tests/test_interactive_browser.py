"""Browser tests for the interactive density page.

These open generated pages in headless Chromium, offline, and drive them with
real mouse and keyboard input. They catch what static HTML checks cannot:
inline scripts that never run, event-order bugs, and Plotly behaviour.

Needs the dev extra and a one-off browser install::

    python -m playwright install chromium --no-shell

The tests are skipped when Playwright or its Chromium is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from src.models import CoordinatePair  # noqa: E402
from src.plots_interactive import make_density_interactive  # noqa: E402

pytestmark = pytest.mark.browser

_WELLS = [(-120.0, 60.0), (60.0, -150.0), (150.0, 150.0), (-30.0, -60.0)]
_MAX_PINNED = 3
# Distinct elements, so the tests can tell which file atom 3Dmol styled.
_ELEMENTS = ["O", "C", "N", "S", "P"]
# x-only atom 0, shared atom 1, y-only atom 2, neutral atoms 3 and 4.
_X_ATOMS = (0, 1)
_Y_ATOMS = (1, 2)


# ---------------------------------------------------------------------------
# Fixtures: a small trajectory, three page variants, a browser
# ---------------------------------------------------------------------------


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _write_trajectory(
    path: Path, n_frames: int = 240, elements: list[str] | None = None
) -> pd.DataFrame:
    """Write a 5-atom xyz trajectory and return its coordinate table.

    The dihedral values come from four Gaussian wells, so the map has many
    populated bins; the geometry is a jittered chain that 3Dmol can draw.
    """
    rng = np.random.default_rng(0)
    chain = np.array([[0.0, 1.4, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                      [2.0, 1.4, 0.3], [3.5, 1.4, 0.3]])
    elements = elements or ["C"] * len(chain)
    rows = []
    with open(path, "wb") as fh:
        for i in range(n_frames):
            cx, cy = _WELLS[i % len(_WELLS)]
            coords = chain + rng.normal(0.0, 0.05, chain.shape)
            offset = fh.tell()
            lines = [f"{len(coords)}", f"frame {i}"]
            lines += [f"{el} {x:.5f} {y:.5f} {z:.5f}" for el, (x, y, z) in zip(elements, coords)]
            fh.write(("\n".join(lines) + "\n").encode("utf-8"))
            rows.append({
                "frame_id": i,
                "source_file": str(path),
                "trajectory_id": "traj0",
                "bead_id": None,
                "frame_number": i,
                "byte_offset": offset,
                "atom_count": len(coords),
                "comment_line": f"frame {i}",
                "local_frame_index": i,
                "global_frame_index": i,
                "carboxyl_dihedral": _wrap(rng.normal(cx, 15.0)),
                "ester_dihedral": _wrap(rng.normal(cy, 15.0)),
            })
    return pd.DataFrame(rows)


def _pair(**atoms) -> CoordinatePair:
    return CoordinatePair(
        name="dihedral",
        x_col="carboxyl_dihedral",
        y_col="ester_dihedral",
        x_label="Carboxyl dihedral (°)",
        y_label="Ester dihedral (°)",
        title="Dihedral density",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
        bins=12,
        **atoms,
    )


@pytest.fixture(scope="module")
def pages(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("browser_pages")
    df = _write_trajectory(root / "traj.xyz")
    variants = {
        "bin": {"embed_xyz_payload": True, "max_pinned": _MAX_PINNED},
        "frame": {"embed_xyz_payload": False},
        "cdn": {"embed_xyz_payload": True, "include_3dmol": "cdn"},
        "fe": {"embed_xyz_payload": True, "default_scale": "free_energy"},
        # Slice 7: the same page with the plain-JSON blocks instead.
        "plain": {"embed_xyz_payload": True, "compress_payloads": False},
        "plain_frame": {"embed_xyz_payload": False, "compress_payloads": False},
    }
    out = {}
    for name, interactive in variants.items():
        config = {"plots": {"interactive": interactive}}
        if name == "fe":
            config["transitions"] = {"temperature": 300.0, "energy_unit": "kJ/mol"}
        out[name] = make_density_interactive(df, _pair(), root / f"{name}.html", config=config)
    # Two beads, each with its own numbering of the same four wells.
    states_df = df.copy()
    wells = np.arange(len(df)) % len(_WELLS)
    states_df["bead_id"] = np.where(np.arange(len(df)) < len(df) // 2, "00", "01")
    states_df["state_dihedral"] = pd.array(
        [str(w if b == "00" else 3 - w) for w, b in zip(wells, states_df["bead_id"])],
        dtype="string",
    )
    for name, show in (("states", False), ("states_on", True)):
        out[name] = make_density_interactive(
            states_df, _pair(), root / f"{name}.html",
            config={
                "clustering": {"groupby": ["bead_id"]},
                "plots": {"interactive": {"embed_xyz_payload": True, "show_states": show}},
            },
        )
    # Two linked pages of one run, plus a run with a single pair.
    nav_siblings = [
        {"name": "dihedral", "title": "Dihedral density", "filename": "nav_a.html"},
        {"name": "swapped", "title": "Swapped density", "filename": "nav_b.html"},
    ]
    swapped = _pair()
    swapped = CoordinatePair(**{**swapped.__dict__, "name": "swapped", "title": "Swapped density"})
    no_payload = {"plots": {"interactive": {"embed_xyz_payload": False}}}
    out["nav_a"] = make_density_interactive(
        df, _pair(), root / "nav_a.html", config=no_payload, siblings=nav_siblings
    )
    out["nav_b"] = make_density_interactive(
        df, swapped, root / "nav_b.html", config=no_payload, siblings=nav_siblings
    )
    out["nav_single"] = make_density_interactive(
        df, _pair(), root / "nav_single.html", config=no_payload,
        siblings=[{"name": "dihedral", "title": "Dihedral density", "filename": "nav_single.html"}],
    )
    # Three linked pages with structures, for pins carried between pairs:
    # the second swaps the axes; the third has a DoF missing for every frame
    # with carboxyl_dihedral < -75 (the well at -120).
    carry_df = df.copy()
    carry_df["partial_dof"] = np.where(
        carry_df["carboxyl_dihedral"] < -75.0, np.nan, 1.0 + (np.arange(len(df)) % 8)
    )
    carry_pairs = {
        "carry_a": _pair(),
        "carry_b": CoordinatePair(**{
            **_pair().__dict__, "name": "swapped", "title": "Swapped density",
            "x_col": "ester_dihedral", "y_col": "carboxyl_dihedral",
        }),
        "carry_c": CoordinatePair(**{
            **_pair().__dict__, "name": "partial", "title": "Partial density",
            "y_col": "partial_dof", "y_label": "Partial", "y_domain": (0.0, 10.0),
            "periodic": False,
        }),
    }
    carry_siblings = [
        {"name": p.name, "title": p.title, "filename": f"{key}.html", "columns": [p.x_col, p.y_col]}
        for key, p in carry_pairs.items()
    ]
    for key, p in carry_pairs.items():
        out[key] = make_density_interactive(
            carry_df, p, root / f"{key}.html",
            config={"plots": {"interactive": {"embed_xyz_payload": True}}},
            siblings=carry_siblings,
        )
    out["highlight"] = make_density_interactive(
        _write_trajectory(root / "traj_elements.xyz", elements=_ELEMENTS),
        _pair(x_atoms=_X_ATOMS, y_atoms=_Y_ATOMS, x_dof_type="distance", y_dof_type="distance"),
        root / "highlight.html",
        config={"plots": {"interactive": {"embed_xyz_payload": True}}},
    )
    return out


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch(
                channel="chromium",
                headless=True,
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
        except sync_api.Error as exc:
            pytest.skip(
                "Playwright's Chromium is not installed "
                f"({str(exc).splitlines()[0]}); run "
                "`python -m playwright install chromium --no-shell`."
            )
        yield chromium
        chromium.close()


class _Page:
    """One offline page, with helpers to find bins and read the side panel."""

    def __init__(self, browser, path: Path, init_script: str | None = None):
        self.context = browser.new_context(viewport={"width": 1400, "height": 900}, offline=True)
        if init_script:
            self.context.add_init_script(init_script)
        self.page = self.context.new_page()
        self.errors: list[str] = []
        self.page.on("pageerror", lambda exc: self.errors.append(str(exc)))
        self.page.goto(path.as_uri())
        self.page.wait_for_function(
            "() => { const gd = document.getElementsByClassName('plotly-graph-div')[0];"
            " return gd && gd._fullLayout && gd._fullLayout._size; }"
        )

    def bins(self) -> dict[str, list[dict]]:
        """Screen positions of populated and empty bin centres."""
        return self.page.evaluate(
            """() => {
              const pd = JSON.parse(document.getElementById('page-data').textContent);
              const gd = document.getElementsByClassName('plotly-graph-div')[0];
              const fl = gd._fullLayout, rect = gd.getBoundingClientRect(), tr = gd._fullData[0];
              const out = {full: [], empty: []};
              for (let yi = 0; yi < tr.z.length; yi++) {
                for (let xi = 0; xi < tr.z[yi].length; xi++) {
                  const p = {key: xi + '_' + yi,
                             x: rect.left + fl._size.l + fl.xaxis.d2p(tr.x[xi]),
                             y: rect.top + fl._size.t + fl.yaxis.d2p(tr.y[yi])};
                  const full = pd.bin_frame_metadata ? !!pd.bin_frame_metadata[p.key]
                                                     : Number.isFinite(tr.z[yi][xi]);
                  (full ? out.full : out.empty).push(p);
                }
              }
              return out;
            }"""
        )

    def state(self) -> dict:
        return self.page.evaluate(
            """() => {
              const notice = document.getElementById('panel-notice');
              return {
                status: document.getElementById('preview-status').textContent,
                cards: [...document.getElementById('comparison-cards').children]
                  .map((c) => c.dataset.cardKey),
                // Per card, top to bottom: its pin number and its bin here.
                pins: [...document.getElementById('comparison-cards').children]
                  .map((c) => [Number(c.dataset.pin), c.dataset.bin]),
                notice: notice.hidden ? null : notice.textContent,
                preview3dHidden: document.getElementById('preview-3d').hidden,
                previewCanvas: !!document.querySelector('#preview-viewer canvas'),
                previewText: document.getElementById('preview-viewer').textContent,
                title: document.getElementById('hdr-title').textContent,
                theme: document.documentElement.dataset.theme || null,
                paper: document.getElementsByClassName('plotly-graph-div')[0]
                  ._fullLayout.paper_bgcolor,
              };
            }"""
        )

    def wait_until(self, condition: str, timeout_ms: int = 3000) -> None:
        """Wait for a JS condition; on timeout, the caller's assert reports state."""
        try:
            self.page.wait_for_function(condition, timeout=timeout_ms)
        except sync_api.TimeoutError:
            pass

    def hover(self, point: dict) -> None:
        self.page.mouse.move(point["x"], point["y"])

    def hover_settled(self, point: dict) -> None:
        """Hover, then wait until Plotly has registered this cell under the mouse.

        Plotly updates its hover point at most every 50 ms and reports clicks at
        that point, so an instant move-and-click would hit the previous cell.
        """
        self.hover(point)
        xi, yi = (int(v) for v in point["key"].split("_"))
        self.wait_until(
            "(() => { const h = document.getElementsByClassName('plotly-graph-div')[0]._hoverdata;"
            f" return !!(h && h[0] && h[0].pointNumber[0] === {yi}"
            f" && h[0].pointNumber[1] === {xi}); }})()"
        )

    def click(self, point: dict) -> None:
        """Click like a person: settle on the cell first, then press."""
        self.hover_settled(point)
        self.page.mouse.down()
        self.page.mouse.up()

    def card_count_is(self, n: int) -> None:
        self.wait_until(f"document.getElementById('comparison-cards').children.length === {n}")

    def css_var(self, name: str) -> str:
        return self.page.evaluate(
            f"getComputedStyle(document.documentElement).getPropertyValue('{name}').trim()"
        )

    def atom_styles(self, selector: str) -> list[dict]:
        """Element and stick/sphere colour of each atom in the 3Dmol viewer at *selector*."""
        return self.page.evaluate(
            """(selector) => {
              const v = document.querySelector(selector)._viewer3d;
              return v.getModel().selectedAtoms({}).map((a) => ({
                index: a.index,
                elem: a.elem,
                stick: a.style.stick ? String(a.style.stick.color) : null,
                sphere: a.style.sphere ? String(a.style.sphere.color) : null,
              }));
            }""",
            selector,
        )

    def scale_state(self) -> dict:
        """Heatmap z/labels and the scale controls, as the page shows them."""
        return self.page.evaluate(
            """() => {
              const gd = document.getElementsByClassName('plotly-graph-div')[0];
              const tr = gd._fullData[0];
              const z = Array.from(tr.z, (row) => Array.from(row,
                (v) => (v === null || Number.isNaN(v) ? null : v)));
              const pressed = [...document.querySelectorAll('.ca-segment')]
                .filter((b) => b.getAttribute('aria-pressed') === 'true')
                .map((b) => b.dataset.scale);
              const hint = document.getElementById('fe-hint');
              return {
                z,
                colorbar: tr.colorbar.title.text,
                hovertemplate: tr.hovertemplate,
                header: document.getElementById('hdr-scale-mode').textContent,
                subtitle: document.getElementById('hdr-subtitle').textContent,
                pressed,
                feHidden: document.getElementById('fe-controls').hidden,
                hint: hint.hidden ? null : hint.textContent,
                temperature: document.getElementById('fe-temperature').value,
                unit: document.getElementById('fe-unit').value,
              };
            }"""
        )

    def overlay_state(self) -> dict:
        """State trace, annotations and controls, as the page shows them."""
        return self.page.evaluate(
            """() => {
              const gd = document.getElementsByClassName('plotly-graph-div')[0];
              const tr = gd._fullData[1];
              // Plotly computes no z for a hidden trace.
              const z = tr.z ? Array.from(tr.z, (row) => Array.from(row,
                (v) => (v === null || Number.isNaN(v) ? null : v))) : null;
              return {
                visible: tr.visible,
                z,
                annotations: gd.layout.annotations || [],
                pressed: document.getElementById('states-toggle').getAttribute('aria-pressed'),
                groupHidden: document.getElementById('state-group').hidden,
                groups: [...document.getElementById('state-group').options].map((o) => o.text),
              };
            }"""
        )

    def pin_marks(self) -> list[dict]:
        """Pin badges on the map (halo annotations left out), in layout order."""
        return self.page.evaluate(
            """() => (document.getElementsByClassName('plotly-graph-div')[0].layout.annotations || [])
                 .filter((a) => /^pin-\\d+$/.test(a.name || ''))
                 .map((a) => ({name: a.name, x: a.x, y: a.y, opacity: a.opacity,
                               size: a.font.size, ax: a.ax, ay: a.ay}))"""
        )

    def pin_shapes(self) -> list[dict]:
        return self.page.evaluate(
            """() => (document.getElementsByClassName('plotly-graph-div')[0].layout.shapes || [])
                 .map((s) => ({name: s.name, type: s.type, x0: s.x0, x1: s.x1, y0: s.y0, y1: s.y1,
                               xanchor: s.xanchor, yanchor: s.yanchor, opacity: s.opacity}))"""
        )

    def card(self, number: int):
        return self.page.locator(f'#comparison-cards .ca-card[data-pin="{number}"]')

    def page_data(self) -> dict:
        return self.page.evaluate("JSON.parse(document.getElementById('page-data').textContent)")

    def axis_title_colors(self) -> dict:
        return self.page.evaluate(
            """() => {
              const l = document.getElementsByClassName('plotly-graph-div')[0].layout;
              const c = (ax) => ((l[ax].title || {}).font || {}).color || null;
              return {x: c('xaxis'), y: c('yaxis')};
            }"""
        )


@pytest.fixture
def open_page(browser):
    opened: list[_Page] = []

    def _open(path: Path, init_script: str | None = None) -> _Page:
        page = _Page(browser, path, init_script=init_script)
        opened.append(page)
        return page

    yield _open
    for page in opened:
        page.context.close()


# ---------------------------------------------------------------------------
# Bin mode (structures embedded)
# ---------------------------------------------------------------------------


def test_page_script_runs_without_errors(open_page, pages):
    page = open_page(pages["bin"])
    assert page.errors == []
    assert page.state()["title"] == "Dihedral density"


def test_hover_previews_bin_and_reports_empty_bins(open_page, pages):
    page = open_page(pages["bin"])
    bins = page.bins()
    first = bins["full"][0]

    page.hover(first)
    page.wait_until(f"document.getElementById('preview-status').textContent === 'Bin {first['key']}'")
    state = page.state()
    assert state["status"] == f"Bin {first['key']}"
    assert state["previewCanvas"]

    page.hover(bins["empty"][0])
    page.wait_until("document.getElementById('preview-status').textContent.startsWith('No frames')")
    assert page.state()["status"] == f"No frames in this bin — showing bin {first['key']}"
    assert page.errors == []


def test_click_pins_and_escape_clears(open_page, pages):
    page = open_page(pages["bin"])
    target = page.bins()["full"][0]
    page.click(target)
    page.card_count_is(1)
    assert page.state()["pins"] == [[1, target["key"]]]
    assert page.state()["cards"][0].startswith("frame:")

    page.page.keyboard.press("Escape")
    page.card_count_is(0)
    assert page.state()["cards"] == []


def test_pin_limit_evicts_oldest_with_notice(open_page, pages):
    page = open_page(pages["bin"])
    targets = page.bins()["full"][: _MAX_PINNED + 1]
    for point in targets:
        page.click(point)
    page.card_count_is(_MAX_PINNED)
    state = page.state()
    # The oldest pin (1) is removed and its number reused; cards go by number.
    assert state["pins"] == [[1, targets[3]["key"]], [2, targets[1]["key"]], [3, targets[2]["key"]]]
    assert state["notice"] and f"Pin limit ({_MAX_PINNED})" in state["notice"]


def test_double_click_clears_pins(open_page, pages):
    page = open_page(pages["bin"])
    bins = page.bins()["full"]
    page.click(bins[0])
    page.click(bins[1])
    page.card_count_is(2)

    page.hover_settled(bins[2])
    page.page.mouse.dblclick(bins[2]["x"], bins[2]["y"])
    page.card_count_is(0)
    assert page.state()["cards"] == []


def test_quick_clicks_on_different_bins_keep_both_pins(open_page, pages):
    # Plotly treats any two clicks within 300 ms as a double-click, even on
    # different bins; the page must not clear the pins for that. Uses the
    # per-frame page (same click handler): there, pinning creates no 3D viewer,
    # which in headless Chromium takes ~1 s and would stretch the gap past 300 ms.
    page = open_page(pages["frame"])
    page.page.evaluate(
        "() => { window.__plotlyDoubleClicks = 0;"
        " document.getElementsByClassName('plotly-graph-div')[0]"
        ".on('plotly_doubleclick', () => { window.__plotlyDoubleClicks++; }); }"
    )
    bins = page.bins()["full"]
    page.click(bins[0])
    page.click(bins[1])
    page.card_count_is(2)
    # Guard: the two clicks really were close enough for Plotly to call it a
    # double-click, so this test exercises the hazard.
    assert page.page.evaluate("window.__plotlyDoubleClicks") >= 1
    assert len(page.state()["cards"]) == 2


def test_theme_toggle_recolours_plot(open_page, pages):
    page = open_page(pages["bin"])
    page.page.click("#theme-toggle")
    page.wait_until(
        "document.getElementsByClassName('plotly-graph-div')[0]._fullLayout.paper_bgcolor"
        " === '#1c2024'"
    )
    state = page.state()
    assert state["theme"] == "dark"
    assert state["paper"] == "#1c2024"


@pytest.mark.parametrize("variant", ["bin", "highlight", "fe", "states_on"])
def test_idle_page_does_not_redraw(open_page, pages, variant):
    page = open_page(pages[variant])
    redraws = page.page.evaluate(
        """() => new Promise((resolve) => {
          const gd = document.getElementsByClassName('plotly-graph-div')[0];
          let n = 0;
          gd.on('plotly_afterplot', () => { n++; });
          setTimeout(() => resolve(n), 1500);
        })"""
    )
    assert redraws == 0


# ---------------------------------------------------------------------------
# Per-frame mode and 3Dmol from a CDN while offline
# ---------------------------------------------------------------------------


def test_per_frame_hover_is_readout_only(open_page, pages):
    page = open_page(pages["frame"])
    bins = page.bins()
    page.hover(bins["full"][0])
    page.wait_until("document.getElementById('preview-status').textContent.startsWith('Click')")
    state = page.state()
    assert state["preview3dHidden"]
    assert state["status"] == "Click to pin the nearest frame"

    page.hover(bins["empty"][0])
    page.wait_until("document.getElementById('preview-status').textContent.startsWith('No frames')")
    assert page.state()["status"] == "No frames in this bin"

    page.click(bins["full"][0])
    page.card_count_is(1)
    assert len(page.state()["cards"]) == 1
    assert page.errors == []


def test_cdn_3dmol_offline_shows_fallback(open_page, pages):
    page = open_page(pages["cdn"])
    first = page.bins()["full"][0]
    page.hover(first)
    page.wait_until("document.getElementById('preview-viewer').textContent.includes('could not load')")
    state = page.state()
    assert "3D viewer could not load" in state["previewText"]
    assert state["status"] == f"Bin {first['key']}"


# ---------------------------------------------------------------------------
# Axis-atom highlighting
# ---------------------------------------------------------------------------


def _expected_atom_colors(page: _Page) -> list[tuple[str | None, str]]:
    """(sphere colour, stick colour) per atom for _X_ATOMS / _Y_ATOMS."""
    x, y, both = (page.css_var(f"--ca-axis-{g}") for g in ("x", "y", "both"))
    neutral = page.css_var("--ca-atom-neutral")
    return [(x, x), (both, both), (y, y), (None, neutral), (None, neutral)]


def _atom_colors(styles: list[dict]) -> list[tuple[str | None, str]]:
    return [(a["sphere"], a["stick"]) for a in styles]


def test_axis_atoms_highlighted_by_file_index_in_preview(open_page, pages):
    page = open_page(pages["highlight"])
    first = page.bins()["full"][0]
    page.hover(first)
    page.wait_until(f"document.getElementById('preview-status').textContent === 'Bin {first['key']}'")

    styles = page.atom_styles("#preview-viewer")
    # 3Dmol's index is the file order: atom i carries element i of the xyz file.
    assert [a["index"] for a in styles] == list(range(len(_ELEMENTS)))
    assert [a["elem"] for a in styles] == _ELEMENTS
    assert _atom_colors(styles) == _expected_atom_colors(page)
    assert page.errors == []


def test_axis_atoms_highlighted_in_pinned_card(open_page, pages):
    page = open_page(pages["highlight"])
    page.click(page.bins()["full"][0])
    page.card_count_is(1)
    styles = page.atom_styles("#comparison-cards .comparison-viewer")
    assert [a["elem"] for a in styles] == _ELEMENTS
    assert _atom_colors(styles) == _expected_atom_colors(page)


def test_axis_titles_and_legend_use_highlight_colours(open_page, pages):
    page = open_page(pages["highlight"])
    assert page.axis_title_colors() == {
        "x": page.css_var("--ca-axis-x"),
        "y": page.css_var("--ca-axis-y"),
    }
    legend = page.page.evaluate(
        """() => {
          const el = document.getElementById('axis-legend');
          return {hidden: el.hidden,
                  rows: [...el.querySelectorAll('.ca-legend-row')]
                    .map((r) => [r.dataset.group, r.textContent])};
        }"""
    )
    assert legend["hidden"] is False
    assert legend["rows"] == [
        ["x", "Carboxyl dihedral (°)atoms 0, 1"],
        ["y", "Ester dihedral (°)atoms 1, 2"],
        ["both", "Both axesatoms 1"],
    ]


def test_theme_toggle_recolours_highlighted_atoms(open_page, pages):
    page = open_page(pages["highlight"])
    page.page.evaluate("document.documentElement.dataset.theme = 'light'")
    first = page.bins()["full"][0]
    page.hover(first)
    page.wait_until(f"document.getElementById('preview-status').textContent === 'Bin {first['key']}'")
    light_x = page.css_var("--ca-axis-x")

    page.page.click("#theme-toggle")
    page.wait_until("document.documentElement.dataset.theme === 'dark'")
    dark_x = page.css_var("--ca-axis-x")
    assert dark_x != light_x
    assert page.atom_styles("#preview-viewer")[0]["sphere"] == dark_x
    assert page.axis_title_colors()["x"] == dark_x


def test_page_without_axis_atoms_keeps_element_colours(open_page, pages):
    page = open_page(pages["bin"])
    first = page.bins()["full"][0]
    page.hover(first)
    page.wait_until(f"document.getElementById('preview-status').textContent === 'Bin {first['key']}'")
    styles = page.atom_styles("#preview-viewer")
    assert all(a["sphere"] is None and a["stick"] in ("None", "undefined") for a in styles)
    assert page.axis_title_colors() == {"x": None, "y": None}
    assert page.page.evaluate("document.getElementById('axis-legend').hidden") is True


# ---------------------------------------------------------------------------
# Colour scale, temperature and unit
# ---------------------------------------------------------------------------

_DENSITY_SUBTITLE = "Coordinate-density landscape — not a potential energy surface"
_FE_SUBTITLE = "Population-derived free-energy-like surface, not a potential energy surface"


def _scaled(grid: list[list], factor: float) -> list[list]:
    return [[None if v is None else v * factor for v in row] for row in grid]


def _k_b(data: dict, unit: str) -> float:
    return next(u["k_B"] for u in data["scale"]["energy_units"] if u["key"] == unit)


def test_scale_toggle_swaps_grids_and_labels(open_page, pages):
    page = open_page(pages["bin"])
    grids = page.page_data()["scale"]["grids"]
    initial = page.scale_state()
    assert initial["pressed"] == ["log_counts"]
    assert initial["z"] == grids["log_counts"]
    assert initial["feHidden"] is True

    page.page.click('[data-scale="counts"]')
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'count'")
    state = page.scale_state()
    assert state["z"] == grids["counts"]
    assert (state["colorbar"], state["header"], state["pressed"]) == ("count", "count", ["counts"])
    assert "count: %{z:.0f}" in state["hovertemplate"]
    assert state["subtitle"] == _DENSITY_SUBTITLE

    # No temperature configured: free-energy mode opens in kT.
    page.page.click('[data-scale="free_energy"]')
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'F (kT)'")
    state = page.scale_state()
    assert state["z"] == grids["free_energy"]
    assert state["colorbar"] == "F (kT)"
    assert state["feHidden"] is False
    assert state["subtitle"] == _FE_SUBTITLE

    # Back to the start: the page rebuilds exactly what Python built.
    page.page.click('[data-scale="log_counts"]')
    page.wait_until("document.getElementById('hdr-scale-mode').textContent.startsWith('log')")
    state = page.scale_state()
    for key in ("z", "colorbar", "hovertemplate", "header", "subtitle"):
        assert state[key] == initial[key], key
    assert page.errors == []


def test_temperature_and_unit_rescale_free_energy(open_page, pages):
    page = open_page(pages["bin"])
    data = page.page_data()
    fe = data["scale"]["grids"]["free_energy"]
    page.page.click('[data-scale="free_energy"]')

    page.page.select_option("#fe-unit", "kJ/mol")
    page.wait_until("!document.getElementById('fe-hint').hidden")
    state = page.scale_state()
    assert state["hint"] == "Enter a temperature to show kJ/mol; showing kT"
    assert state["z"] == fe and state["colorbar"] == "F (kT)"

    page.page.fill("#fe-temperature", "300")
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'F (kJ/mol)'")
    state = page.scale_state()
    assert state["hint"] is None
    assert state["colorbar"] == "F (kJ/mol)"
    assert state["z"] == _scaled(fe, _k_b(data, "kJ/mol") * 300.0)

    page.page.fill("#fe-temperature", "600")
    page.wait_until(
        "document.getElementsByClassName('plotly-graph-div')[0]._fullData[0].z"
        f".flat().some((v) => Math.abs(v - {max(v for r in fe for v in r if v) * _k_b(data, 'kJ/mol') * 600.0}) < 1e-9)"
    )
    assert page.scale_state()["z"] == _scaled(fe, _k_b(data, "kJ/mol") * 600.0)

    page.page.select_option("#fe-unit", "cm^-1")
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'F (cm⁻¹)'")
    assert page.scale_state()["z"] == _scaled(fe, _k_b(data, "cm^-1") * 600.0)

    # kT ignores the temperature.
    page.page.select_option("#fe-unit", "kT")
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'F (kT)'")
    assert page.scale_state()["z"] == fe
    assert page.errors == []


def test_free_energy_default_page_opens_as_configured(open_page, pages):
    page = open_page(pages["fe"])
    data = page.page_data()
    state = page.scale_state()
    assert state["pressed"] == ["free_energy"]
    assert (state["temperature"], state["unit"]) == ("300", "kJ/mol")
    assert state["colorbar"] == state["header"] == "F (kJ/mol)"
    assert state["subtitle"] == _FE_SUBTITLE
    assert state["z"] == _scaled(data["scale"]["grids"]["free_energy"], _k_b(data, "kJ/mol") * 300.0)

    # Switch away and back: the page's own formula must give exactly the
    # numbers Python baked into the figure.
    page.page.click('[data-scale="counts"]')
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'count'")
    page.page.click('[data-scale="free_energy"]')
    page.wait_until("document.getElementById('hdr-scale-mode').textContent === 'F (kJ/mol)'")
    rebuilt = page.scale_state()
    for key in ("z", "colorbar", "hovertemplate"):
        assert rebuilt[key] == state[key], key


def test_readout_shows_scaled_value_and_raw_count(open_page, pages):
    page = open_page(pages["fe"])
    data = page.page_data()
    first = page.bins()["full"][0]
    xi, yi = (int(v) for v in first["key"].split("_"))
    page.hover(first)
    page.wait_until(f"document.getElementById('preview-status').textContent === 'Bin {first['key']}'")
    text = page.page.evaluate("document.getElementById('preview-readout').innerText")
    count = data["scale"]["grids"]["counts"][yi][xi]
    value = data["scale"]["grids"]["free_energy"][yi][xi] * _k_b(data, "kJ/mol") * 300.0
    lines = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)
    assert float(lines["F (kJ/mol)"]) == pytest.approx(value, rel=1e-5)
    assert lines["count"] == str(count)

    # Changing the scale refreshes the read-out for the bin under the cursor.
    page.page.select_option("#fe-unit", "kT")
    page.wait_until("document.getElementById('preview-readout').innerText.includes('F (kT)')")
    assert f"count: {count}" in page.page.evaluate("document.getElementById('preview-readout').innerText")


# ---------------------------------------------------------------------------
# State overlay
# ---------------------------------------------------------------------------


def _expected_state_z(data: dict, group_index: int) -> list[list]:
    nx, ny = data["header"]["bin_count_x"], data["header"]["bin_count_y"]
    z = [[None] * nx for _ in range(ny)]
    group = data["states"]["groups"][group_index]
    for b, code in zip(group["bins"], group["states"]):
        z[b // nx][b % nx] = code
    return z


def test_states_toggle_shows_overlay_and_labels(open_page, pages):
    page = open_page(pages["states"])
    data = page.page_data()
    state = page.overlay_state()
    assert state["visible"] is False and state["pressed"] == "false"
    assert state["groups"] == ["bead 00", "bead 01"] and state["groupHidden"] is False
    assert state["annotations"] and all(a["visible"] is False for a in state["annotations"])

    page.page.click("#states-toggle")
    page.wait_until("document.getElementsByClassName('plotly-graph-div')[0]._fullData[1].visible === true")
    state = page.overlay_state()
    assert state["pressed"] == "true"
    assert state["z"] == _expected_state_z(data, 0)
    assert all(a["visible"] is True for a in state["annotations"])

    page.page.click("#states-toggle")
    page.wait_until("document.getElementsByClassName('plotly-graph-div')[0]._fullData[1].visible === false")
    assert page.overlay_state()["pressed"] == "false"
    assert page.errors == []


def test_state_group_switch_never_mixes_groups(open_page, pages):
    page = open_page(pages["states_on"])
    data = page.page_data()
    initial = page.overlay_state()

    page.page.select_option("#state-group", "1")
    page.wait_until(
        "document.getElementsByClassName('plotly-graph-div')[0].layout.annotations[0].text"
        f" === {data['states']['labels'][data['states']['groups'][1]['centres'][0]['state']]!r}"
    )
    state = page.overlay_state()
    assert state["z"] == _expected_state_z(data, 1)
    assert state["z"] != initial["z"]  # the beads number the wells differently

    # Back to the first group: the page rebuilds exactly what Python built.
    page.page.select_option("#state-group", "0")
    page.wait_until(
        "document.getElementsByClassName('plotly-graph-div')[0].layout.annotations[0].text"
        f" === {initial['annotations'][0]['text']!r}"
    )
    assert page.overlay_state() == initial
    assert page.errors == []


def test_overlay_does_not_capture_hover_or_clicks(open_page, pages):
    page = open_page(pages["states_on"])
    data = page.page_data()
    z0, z1 = _expected_state_z(data, 0), _expected_state_z(data, 1)

    def labelled_in_both(point):
        xi, yi = (int(v) for v in point["key"].split("_"))
        # Off-diagonal and asymmetric, so a swapped row/column index would show.
        return (z0[yi][xi] is not None and z1[yi][xi] is not None
                and z0[yi][xi] != z0[xi][yi] and z0[yi][xi] != z1[yi][xi])

    target = next(p for p in page.bins()["full"] if labelled_in_both(p))
    xi, yi = (int(v) for v in target["key"].split("_"))
    page.click(target)
    page.card_count_is(1)
    assert page.state()["pins"] == [[1, target["key"]]]
    assert page.state()["status"] == f"Bin {target['key']}"

    text = page.page.evaluate("document.getElementById('preview-readout').innerText")
    labels = data["states"]["labels"]
    assert f"state (bead 00): {labels[z0[yi][xi]]}" in text

    page.page.select_option("#state-group", "1")
    page.wait_until("document.getElementById('preview-readout').innerText.includes('bead 01')")
    assert f"state (bead 01): {labels[z1[yi][xi]]}" in page.page.evaluate(
        "document.getElementById('preview-readout').innerText"
    )


def test_theme_toggle_recolours_state_labels(open_page, pages):
    page = open_page(pages["states_on"])
    page.page.evaluate("document.documentElement.dataset.theme = 'light'")
    page.page.click("#theme-toggle")
    page.wait_until("document.documentElement.dataset.theme === 'dark'")
    page.wait_until(
        "document.getElementsByClassName('plotly-graph-div')[0].layout.annotations[0].bgcolor"
        " === 'rgba(28, 32, 36, 0.75)'"
    )
    assert {a["bgcolor"] for a in page.overlay_state()["annotations"]} == {"rgba(28, 32, 36, 0.75)"}


def test_page_without_states_has_no_state_controls(open_page, pages):
    page = open_page(pages["bin"])
    assert page.page.evaluate("document.getElementById('states-toggle')") is None
    assert page.page.evaluate(
        "document.getElementsByClassName('plotly-graph-div')[0]._fullData.length"
    ) == 1



def test_theme_toggle_swaps_map_colorscale(open_page, pages):
    page = open_page(pages["bin"])
    page.page.evaluate("document.documentElement.dataset.theme = 'light'")
    scales = page.page_data()["scale"]["colorscales"]
    assert scales["light"] != scales["dark"]
    shown = "document.getElementsByClassName('plotly-graph-div')[0].data[0].colorscale"
    assert page.page.evaluate(shown) == scales["light"]

    for theme in ("dark", "light"):
        page.page.click("#theme-toggle")
        page.wait_until(f"{shown}[0][1] === '{scales[theme][0][1]}'")
        assert page.page.evaluate(shown) == scales[theme], theme
    assert page.errors == []


def test_degree_ticks_follow_zoom(open_page, pages):
    page = open_page(pages["bin"])
    dtick = (
        "(() => { const l = document.getElementsByClassName('plotly-graph-div')[0].layout;"
        " return [l.xaxis.dtick, l.yaxis.dtick]; })()"
    )
    assert page.page.evaluate(dtick) == [60, 60]
    labels = page.page.evaluate(
        "[...document.querySelectorAll('.xtick text')].map((t) => t.textContent)"
    )
    assert labels == ["−180°", "−120°", "−60°", "0°", "60°", "120°", "180°"]

    page.page.evaluate(
        "Plotly.relayout(document.getElementsByClassName('plotly-graph-div')[0],"
        " {'xaxis.range': [-30, 0], 'yaxis.range': [100, 190]})"
    )
    page.wait_until(f"{dtick}[0] === 5")
    assert page.page.evaluate(dtick) == [5, 15]

    page.page.evaluate(
        "Plotly.relayout(document.getElementsByClassName('plotly-graph-div')[0],"
        " {'xaxis.range': [-180, 180], 'yaxis.range': [-180, 180]})"
    )
    page.wait_until(f"{dtick}[0] === 60")
    assert page.page.evaluate(dtick) == [60, 60]
    assert page.errors == []


# ---------------------------------------------------------------------------
# Pair navigation
# ---------------------------------------------------------------------------


def _nav_links(page) -> list[dict]:
    return page.page.evaluate(
        """() => [...document.querySelectorAll('#pair-nav a')].map((a) => ({
             text: a.textContent,
             href: a.getAttribute('href'),
             current: a.getAttribute('aria-current'),
           }))"""
    )


def test_pair_nav_lists_both_pages_and_marks_the_current_one(open_page, pages):
    page = open_page(pages["nav_a"])
    assert page.page.evaluate("document.getElementById('pair-nav').hidden") is False
    assert _nav_links(page) == [
        {"text": "Dihedral density", "href": "nav_a.html", "current": "page"},
        {"text": "Swapped density", "href": "nav_b.html", "current": None},
    ]
    assert page.errors == []


def test_pair_nav_link_opens_the_sibling_page(open_page, pages):
    page = open_page(pages["nav_a"])
    page.page.click('#pair-nav a[data-pair="swapped"]')
    page.page.wait_for_function(
        "() => document.getElementById('hdr-title').textContent === 'Swapped density'"
    )
    assert page.page.url.endswith("nav_b.html")
    assert _nav_links(page) == [
        {"text": "Dihedral density", "href": "nav_a.html", "current": None},
        {"text": "Swapped density", "href": "nav_b.html", "current": "page"},
    ]
    assert page.errors == []


def test_single_pair_page_hides_the_nav(open_page, pages):
    page = open_page(pages["nav_single"])
    assert page.page.evaluate("document.getElementById('pair-nav').hidden") is True
    assert _nav_links(page) == []
    assert page.errors == []


# ---------------------------------------------------------------------------
# Compressed embedded data (Slice 7)
# ---------------------------------------------------------------------------


def test_compressed_page_embeds_no_plain_blocks(open_page, pages):
    page = open_page(pages["bin"])
    data = page.page_data()
    assert data["bin_xyz_payloads"] is None and data["bin_frame_metadata"] is None
    assert data["bin_xyz_payloads_encoded"]["format"] == "confana-structures-v1"
    assert data["bin_frame_metadata_encoded"]["format"] == "confana-columns-v1"


def test_compressed_structures_unpack_and_render(open_page, pages):
    page = open_page(pages["bin"])
    target = page.bins()["full"][0]
    page.hover(target)
    page.wait_until(
        f"document.getElementById('preview-status').textContent === 'Bin {target['key']}'"
    )
    state = page.state()
    assert state["status"] == f"Bin {target['key']}"
    assert state["previewCanvas"]
    # The rebuilt structure has the same atoms as the source trajectory.
    assert len(page.atom_styles("#preview-viewer")) == 5
    assert page.errors == []


def test_compressed_metadata_matches_the_plain_page(open_page, pages):
    """The same bin shows the same metadata whether packed or plain."""
    meta = {}
    for variant in ("bin", "plain"):
        page = open_page(pages[variant])
        target = page.bins()["full"][0]
        page.hover(target)
        page.wait_until(
            f"document.getElementById('preview-status').textContent === 'Bin {target['key']}'"
        )
        meta[variant] = page.page.evaluate(
            "document.getElementById('preview-meta').textContent"
        )
        assert page.errors == []
    assert meta["bin"] == meta["plain"]


def test_plain_per_frame_page_still_pins(open_page, pages):
    page = open_page(pages["plain_frame"])
    assert page.page_data()["frame_metadata"] is not None
    target = page.bins()["full"][0]
    page.click(target)
    page.card_count_is(1)
    assert page.errors == []


def test_compressed_per_frame_page_pins_the_nearest_frame(open_page, pages):
    """The click scan reads the decoded columns, not a list of records."""
    page = open_page(pages["frame"])
    assert page.page_data()["frame_metadata_encoded"] is not None
    target = page.bins()["full"][0]
    page.click(target)
    page.card_count_is(1)
    assert page.state()["cards"][0].startswith("frame:")
    assert page.errors == []


def test_browser_without_decompression_stream_says_so(open_page, pages):
    page = open_page(pages["bin"], init_script="delete window.DecompressionStream;")
    page.wait_until(
        "document.getElementById('preview-status').textContent.indexOf('Could not unpack') === 0"
    )
    status = page.state()["status"]
    assert "DecompressionStream" in status
    # The map itself still works.
    assert page.page.evaluate(
        "document.getElementsByClassName('plotly-graph-div')[0]._fullData.length"
    ) == 1


# ---------------------------------------------------------------------------
# Pin markers, and pins carried between pair pages (docs/adr/0002)
# ---------------------------------------------------------------------------


def _bin_of(x: float, y: float, lo: float = -180.0, width: float = 30.0) -> str:
    """Bin key of a point on the 12-bin dihedral map of these fixtures."""
    return f"{int((x - lo) // width)}_{int((y - lo) // width)}"


_HAS_PIN_1 = (
    "(document.getElementsByClassName('plotly-graph-div')[0].layout.annotations || [])"
    ".some((a) => a.name === 'pin-1')"
)


def test_pin_marks_its_frame_and_bin_on_the_map(open_page, pages):
    page = open_page(pages["bin"])
    target = page.bins()["full"][0]
    page.click(target)
    page.card_count_is(1)
    page.wait_until(_HAS_PIN_1)

    marks = page.pin_marks()
    assert [m["name"] for m in marks] == ["pin-1"]
    x, y = marks[0]["x"], marks[0]["y"]
    assert _bin_of(x, y) == target["key"]

    shapes = {s["type"]: s for s in page.pin_shapes()}
    xi, yi = (int(v) for v in target["key"].split("_"))
    rect = shapes["rect"]
    assert (rect["x0"], rect["x1"]) == pytest.approx((-180 + 30 * xi, -150 + 30 * xi))
    assert (rect["y0"], rect["y1"]) == pytest.approx((-180 + 30 * yi, -150 + 30 * yi))
    assert (shapes["circle"]["xanchor"], shapes["circle"]["yanchor"]) == (x, y)

    assert page.card(1).locator(".ca-pin-badge").inner_text() == "1"
    assert page.card(1).locator(".ca-card-title b").inner_text().startswith(f"Bin {target['key']} · frame ")

    page.page.keyboard.press("Escape")
    page.card_count_is(0)
    page.wait_until(f"!{_HAS_PIN_1}")
    assert page.pin_marks() == [] and page.pin_shapes() == []
    assert page.errors == []


def test_per_frame_pin_is_a_ring_without_bin_outline(open_page, pages):
    page = open_page(pages["frame"])
    page.click(page.bins()["full"][0])
    page.card_count_is(1)
    page.wait_until(_HAS_PIN_1)
    shapes = page.pin_shapes()
    assert [s["type"] for s in shapes] == ["circle", "circle"]
    mark = page.pin_marks()[0]
    assert all((s["xanchor"], s["yanchor"]) == (mark["x"], mark["y"]) for s in shapes)
    assert page.errors == []


def test_closed_pin_number_is_reused_and_cards_go_by_number(open_page, pages):
    page = open_page(pages["bin"])
    bins = page.bins()["full"]
    for point in bins[:3]:
        page.click(point)
    page.card_count_is(3)

    page.card(2).locator("button").click()
    page.card_count_is(2)
    page.click(bins[3])
    page.card_count_is(3)
    assert page.state()["pins"] == [[1, bins[0]["key"]], [2, bins[3]["key"]], [3, bins[2]["key"]]]
    page.wait_until(
        "(document.getElementsByClassName('plotly-graph-div')[0].layout.annotations || [])"
        ".filter((a) => /^pin-\\d+$/.test(a.name)).length === 3"
    )
    assert sorted(m["name"] for m in page.pin_marks()) == ["pin-1", "pin-2", "pin-3"]
    assert page.errors == []


def test_card_hover_fades_the_other_markers(open_page, pages):
    page = open_page(pages["frame"])
    bins = page.bins()["full"]
    page.click(bins[0])
    page.click(bins[1])
    page.card_count_is(2)

    page.card(1).hover()
    page.wait_until(
        "(document.getElementsByClassName('plotly-graph-div')[0].layout.annotations || [])"
        ".some((a) => a.name === 'pin-2' && a.opacity < 1)"
    )
    marks = {m["name"]: m for m in page.pin_marks()}
    assert (marks["pin-1"]["opacity"], marks["pin-1"]["size"]) == (1, 13)
    assert (marks["pin-2"]["opacity"], marks["pin-2"]["size"]) == (0.3, 11)

    page.page.mouse.move(5, 5)
    page.wait_until(
        "(document.getElementsByClassName('plotly-graph-div')[0].layout.annotations || [])"
        ".every((a) => a.opacity === undefined || a.opacity === 1)"
    )
    assert {m["opacity"] for m in page.pin_marks()} == {1}
    assert page.errors == []


def test_map_hover_lights_the_card_of_a_pinned_bin(open_page, pages):
    page = open_page(pages["frame"])
    bins = page.bins()
    points = {p["key"]: p for p in bins["full"] + bins["empty"]}
    page.click(bins["full"][0])
    page.click(bins["full"][1])
    page.card_count_is(2)
    linked = (
        "[...document.querySelectorAll('#comparison-cards .ca-card-linked')]"
        ".map((c) => Number(c.dataset.pin))"
    )
    # The mouse still rests on the bin just clicked; its pin lights up at once
    # if the nearest frame lies in that bin.
    pin_bins = dict(page.state()["pins"])
    if pin_bins[2] == bins["full"][1]["key"]:
        page.wait_until(f"{linked}.length === 1")
        assert page.page.evaluate(linked) == [2]

    page.hover_settled(bins["empty"][0])
    page.wait_until(f"{linked}.length === 0")
    assert page.page.evaluate(linked) == []

    # The pinned frame's own bin, which in per-frame mode may differ from the click.
    page.hover_settled(points[pin_bins[1]])
    page.wait_until(f"{linked}.length > 0")
    assert 1 in page.page.evaluate(linked)
    assert page.errors == []


def test_badge_click_shows_its_card_without_a_new_pin(open_page, pages):
    page = open_page(pages["frame"])
    bins = page.bins()["full"]
    page.click(bins[0])
    page.click(bins[1])
    page.card_count_is(2)
    page.wait_until(_HAS_PIN_1)
    index = page.page.evaluate(
        "document.getElementsByClassName('plotly-graph-div')[0].layout.annotations"
        ".findIndex((a) => a.name === 'pin-1')"
    )
    page.page.locator(f'g.annotation[data-index="{index}"] .annotation-text-g').click()
    page.wait_until(
        "document.querySelector('#comparison-cards .ca-card[data-pin=\"1\"]')"
        ".classList.contains('ca-card-focus')"
    )
    assert page.page.evaluate(
        "document.querySelector('#comparison-cards .ca-card[data-pin=\"1\"]')"
        ".classList.contains('ca-card-focus')"
    )
    assert len(page.state()["cards"]) == 2
    assert page.errors == []


def _pin_and_follow(page, target: dict, pair: str, title: str) -> dict:
    """Pin *target*, follow the header link to *pair*; return the first page's badge."""
    page.click(target)
    page.card_count_is(1)
    page.wait_until(_HAS_PIN_1)
    page.wait_until("window.location.hash.startsWith('#pins=')")
    mark = page.pin_marks()[0]
    assert "#pins=" in page.page.get_attribute(f'#pair-nav a[data-pair="{pair}"]', "href")
    page.page.click(f'#pair-nav a[data-pair="{pair}"]')
    page.page.wait_for_function(
        f"() => document.getElementById('hdr-title').textContent === {title!r}"
    )
    page.card_count_is(1)
    return mark


def test_pin_follows_the_pair_link_to_its_own_coordinates(open_page, pages):
    page = open_page(pages["carry_a"])
    mark_a = _pin_and_follow(page, page.bins()["full"][0], "swapped", "Swapped density")
    assert page.page.url.split("#")[0].endswith("carry_b.html")

    # The axes are swapped on this page, so the frame sits at (y, x).
    page.wait_until(_HAS_PIN_1)
    mark_b = page.pin_marks()[0]
    assert (mark_b["x"], mark_b["y"]) == (mark_a["y"], mark_a["x"])
    assert page.state()["pins"] == [[1, _bin_of(mark_a["y"], mark_a["x"])]]
    # The carried structure is drawn, not this page's representative.
    page.wait_until("!!document.querySelector('#comparison-cards canvas')")
    assert page.card(1).locator("canvas").count() >= 1

    # The link keeps the pins through a reload.
    page.page.reload()
    page.page.wait_for_function(
        "() => { const gd = document.getElementsByClassName('plotly-graph-div')[0];"
        " return gd && gd._fullLayout && gd._fullLayout._size; }"
    )
    page.card_count_is(1)
    assert page.state()["pins"] == [[1, _bin_of(mark_a["y"], mark_a["x"])]]
    assert page.errors == []


def test_carried_pin_without_a_value_keeps_its_card_but_no_marker(open_page, pages):
    page = open_page(pages["carry_a"])
    # Bins left of -90°: every frame there lacks partial_dof.
    target = next(p for p in page.bins()["full"] if int(p["key"].split("_")[0]) <= 2)
    _pin_and_follow(page, target, "partial", "Partial density")

    assert page.state()["pins"] == [[1, ""]]
    assert page.card(1).locator(".ca-pin-note").inner_text() == (
        "Not on this map (no partial_dof value)"
    )
    assert page.pin_marks() == [] and page.pin_shapes() == []
    assert page.errors == []
