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


# ---------------------------------------------------------------------------
# Fixtures: a small trajectory, three page variants, a browser
# ---------------------------------------------------------------------------


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _write_trajectory(path: Path, n_frames: int = 240) -> pd.DataFrame:
    """Write a 5-atom xyz trajectory and return its coordinate table.

    The dihedral values come from four Gaussian wells, so the map has many
    populated bins; the geometry is a jittered chain that 3Dmol can draw.
    """
    rng = np.random.default_rng(0)
    chain = np.array([[0.0, 1.4, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                      [2.0, 1.4, 0.3], [3.5, 1.4, 0.3]])
    rows = []
    with open(path, "wb") as fh:
        for i in range(n_frames):
            cx, cy = _WELLS[i % len(_WELLS)]
            coords = chain + rng.normal(0.0, 0.05, chain.shape)
            offset = fh.tell()
            lines = [f"{len(coords)}", f"frame {i}"]
            lines += [f"C {x:.5f} {y:.5f} {z:.5f}" for x, y, z in coords]
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


def _pair() -> CoordinatePair:
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
    )


@pytest.fixture(scope="module")
def pages(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("browser_pages")
    df = _write_trajectory(root / "traj.xyz")
    variants = {
        "bin": {"embed_xyz_payload": True, "max_pinned": _MAX_PINNED},
        "frame": {"embed_xyz_payload": False},
        "cdn": {"embed_xyz_payload": True, "include_3dmol": "cdn"},
    }
    out = {}
    for name, interactive in variants.items():
        out[name] = make_density_interactive(
            df, _pair(), root / f"{name}.html", config={"plots": {"interactive": interactive}}
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

    def __init__(self, browser, path: Path):
        self.context = browser.new_context(viewport={"width": 1400, "height": 900}, offline=True)
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


@pytest.fixture
def open_page(browser):
    opened: list[_Page] = []

    def _open(path: Path) -> _Page:
        page = _Page(browser, path)
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
    assert page.state()["cards"] == [f"bin:{target['key']}"]

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
    assert state["cards"] == [f"bin:{p['key']}" for p in reversed(targets[1:])]
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


def test_idle_page_does_not_redraw(open_page, pages):
    page = open_page(pages["bin"])
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
