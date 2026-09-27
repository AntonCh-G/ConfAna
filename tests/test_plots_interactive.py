"""Tests for src/plots_interactive.py (Phase 11)."""

from __future__ import annotations

import importlib.resources
import json
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from src.models import CoordinatePair
from src.plots_interactive import make_density_interactive, render_density_page


class _ScriptCollector(HTMLParser):
    """Collect <script> elements the way a browser splits them.

    Like a browser, HTMLParser ends a script at the first ``</script``, even
    inside a JS comment or string.
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts: list[tuple[dict, str]] = []
        self._current: tuple[dict, list[str]] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._current = (dict(attrs), [])

    def handle_data(self, data):
        if self._current is not None:
            self._current[1].append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._current is not None:
            self.scripts.append((self._current[0], "".join(self._current[1])))
            self._current = None


def _scripts(content: str) -> list[tuple[dict, str]]:
    collector = _ScriptCollector()
    collector.feed(content)
    return collector.scripts


def _asset_text(name: str) -> str:
    return importlib.resources.files("src.interactive_assets").joinpath(name).read_text(
        encoding="utf-8"
    )


def _page_data(content: str) -> dict:
    """Extract and parse the embedded ``#page-data`` JSON block from *content*."""
    match = re.search(
        r'<script id="page-data" type="application/json">(.*?)</script>',
        content,
        re.DOTALL,
    )
    assert match, "no #page-data script block found in content"
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# CoordinatePair fixtures
# ---------------------------------------------------------------------------


def _plane_pair() -> CoordinatePair:
    return CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="Carboxyl plane (°)",
        y_label="Ester plane (°)",
        title="Plane density",
        x_domain=(0.0, 180.0),
        y_domain=(0.0, 180.0),
    )


def _dihedral_pair() -> CoordinatePair:
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
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_angle_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Return a minimal DataFrame with all standard coordinate table columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame_id": range(n),
            "source_file": ["/data/test.xyz"] * n,
            "trajectory_id": ["traj0"] * n,
            "bead_id": [None] * n,
            "frame_number": range(n),
            "byte_offset": rng.integers(0, 100_000, size=n).tolist(),
            "atom_count": [21] * n,
            "comment_line": [""] * n,
            "local_frame_index": range(n),
            "global_frame_index": range(n),
            "carboxyl_plane": rng.uniform(0, 180, n),
            "ester_plane": rng.uniform(0, 180, n),
            "carboxyl_dihedral": rng.uniform(-180, 180, n),
            "ester_dihedral": rng.uniform(-180, 180, n),
            "state_plane": ["0"] * n,
            "state_dihedral": ["0"] * n,
            "energy": rng.standard_normal(n),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_make_density_interactive_creates_file(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "density_plane.html"
    make_density_interactive(df, _plane_pair(), outpath)
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_make_density_interactive_returns_path(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "density_dihedral.html"
    result = make_density_interactive(df, _dihedral_pair(), outpath)
    assert result == outpath or result == outpath.resolve()


def test_make_density_interactive_contains_plotly(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "out.html"
    make_density_interactive(df, _plane_pair(), outpath)
    content = outpath.read_text(encoding="utf-8")
    assert "plotly" in content.lower()


def test_make_density_interactive_contains_metadata_script(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "out.html"
    make_density_interactive(df, _plane_pair(), outpath)
    content = outpath.read_text(encoding="utf-8")
    assert 'id="page-data"' in content


def test_make_density_interactive_metadata_has_frame_fields(tmp_path):
    """The embedded JSON contains expected field names."""
    df = _make_angle_df(n=5)
    outpath = tmp_path / "out.html"
    make_density_interactive(df, _dihedral_pair(), outpath)
    content = outpath.read_text(encoding="utf-8")
    for field in ("frame_id", "source_file", "carboxyl_dihedral", "ester_dihedral"):
        assert field in content, f"Expected field '{field}' not found in HTML"


def test_make_density_interactive_missing_feature_column_raises(tmp_path):
    """DataFrame missing required feature columns raises ValueError."""
    df = _make_angle_df()
    df = df.drop(columns=["carboxyl_plane"])
    with pytest.raises(ValueError):
        make_density_interactive(df, _plane_pair(), tmp_path / "out.html")


def test_make_density_interactive_creates_parent_dirs(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "nested" / "subdir" / "out.html"
    make_density_interactive(df, _plane_pair(), outpath)
    assert outpath.exists()


def test_make_density_interactive_dihedral_range(tmp_path):
    """Default dihedral domain [-180, 180] is embedded in the output."""
    df = _make_angle_df()
    outpath = tmp_path / "dihedral.html"
    make_density_interactive(df, _dihedral_pair(), outpath)
    content = outpath.read_text(encoding="utf-8")
    assert "-180" in content


def test_make_density_interactive_supports_custom_pair(tmp_path):
    df = pd.DataFrame(
        {
            "frame_id": range(8),
            "source_file": ["/data/test.xyz"] * 8,
            "trajectory_id": ["traj0"] * 8,
            "bead_id": [None] * 8,
            "frame_number": range(8),
            "byte_offset": range(8),
            "atom_count": [21] * 8,
            "comment_line": [""] * 8,
            "local_frame_index": range(8),
            "global_frame_index": range(8),
            "igor1_dihedral": np.linspace(-170.0, 170.0, 8),
            "igor2_dihedral": np.linspace(170.0, -170.0, 8),
            "state_dihedral": ["0"] * 8,
        }
    )
    igor_pair = CoordinatePair(
        name="dihedral",
        x_col="igor1_dihedral",
        y_col="igor2_dihedral",
        x_label="Igor 1 (°)",
        y_label="Igor 2 (°)",
        title="Igor pair",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )
    outpath = tmp_path / "igor_pair.html"
    make_density_interactive(
        df,
        igor_pair,
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": False, "include_plotlyjs": "cdn"}}},
    )
    content = outpath.read_text(encoding="utf-8")
    assert "igor1_dihedral" in content
    assert "igor2_dihedral" in content


def test_make_density_interactive_accepts_named_pair(tmp_path):
    df = pd.DataFrame(
        {
            "frame_id": range(8),
            "source_file": ["/data/test.xyz"] * 8,
            "trajectory_id": ["traj0"] * 8,
            "bead_id": [None] * 8,
            "frame_number": range(8),
            "byte_offset": range(8),
            "atom_count": [21] * 8,
            "comment_line": [""] * 8,
            "local_frame_index": range(8),
            "global_frame_index": range(8),
            "igor1_dihedral": np.linspace(-170.0, 170.0, 8),
            "igor2_dihedral": np.linspace(170.0, -170.0, 8),
            "state_dihedrals_igor": ["0"] * 8,
        }
    )
    igor_pair = CoordinatePair(
        name="dihedrals_igor",
        x_col="igor1_dihedral",
        y_col="igor2_dihedral",
        x_label="Igor 1 (°)",
        y_label="Igor 2 (°)",
        title="Igor pair",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
        periodic=True,
    )
    outpath = tmp_path / "igor_pair_named.html"
    make_density_interactive(
        df,
        igor_pair,
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": False, "include_plotlyjs": "cdn"}}},
    )
    content = outpath.read_text(encoding="utf-8")
    assert "igor1_dihedral" in content
    assert "igor2_dihedral" in content


def test_make_density_interactive_signed_plane_range(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "plane_signed.html"
    signed_pair = CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="Carboxyl plane (°)",
        y_label="Ester plane (°)",
        title="Signed plane",
        x_domain=(-180.0, 180.0),
        y_domain=(-180.0, 180.0),
    )
    make_density_interactive(
        df,
        signed_pair,
        outpath,
        config={
            "plots": {
                "density": {"plane_bins": 20},
                "interactive": {"embed_xyz_payload": False, "include_plotlyjs": "cdn"},
            }
        },
    )
    content = outpath.read_text(encoding="utf-8")
    assert "-180" in content


def test_make_density_interactive_metadata_precedes_click_handler(tmp_path):
    """Metadata script must exist before Plotly initialises the click handler."""
    df = _make_angle_df()
    outpath = tmp_path / "plane.html"

    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": False, "include_plotlyjs": "cdn"}}},
    )

    content = outpath.read_text(encoding="utf-8")
    page_data_idx = content.find('id="page-data"')
    plotly_click_idx = content.find("plotly_click")

    assert page_data_idx != -1
    assert plotly_click_idx != -1
    assert page_data_idx < plotly_click_idx


def test_make_density_interactive_bin_payloads_precede_click_handler(tmp_path):
    """Per-bin metadata / XYZ payload scripts must precede the click handler."""
    df = _make_angle_df()
    outpath = tmp_path / "plane_embed.html"

    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": True, "include_plotlyjs": "cdn"}}},
    )

    content = outpath.read_text(encoding="utf-8")
    data = _page_data(content)
    page_data_idx = content.find('id="page-data"')
    plotly_click_idx = content.find("plotly_click")

    assert data["bin_frame_metadata"] is not None
    assert data["bin_xyz_payloads"] is not None
    assert page_data_idx != -1
    assert plotly_click_idx != -1
    assert page_data_idx < plotly_click_idx


def test_make_density_interactive_contains_comparison_tray(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "comparison.html"

    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": False, "include_plotlyjs": "cdn"}}},
    )

    content = outpath.read_text(encoding="utf-8")
    assert 'id="comparison-tray"' in content
    assert 'id="comparison-clear"' in content
    assert 'id="comparison-cards"' in content


def test_make_density_interactive_uses_multi_card_js_and_no_singleton_viewer(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "multi_card.html"

    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": True, "include_plotlyjs": "cdn"}}},
    )

    content = outpath.read_text(encoding="utf-8")
    assert "createCard(" in content
    assert "comparison-viewer" in content
    assert "clearAllCards" in content
    assert 'id="viewer3d"' not in content
    assert 'id="info-panel"' not in content


def test_make_density_interactive_metadata_only_remains_valid_with_alignment_config(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "metadata_only.html"

    make_density_interactive(
        df,
        _dihedral_pair(),
        outpath,
        config={
            "plots": {
                "interactive": {
                    "embed_xyz_payload": False,
                    "include_plotlyjs": "cdn",
                    "alignment": {
                        "enabled": True,
                        "reference": "earliest_frame",
                        "atom_selection": "heavy",
                    },
                }
            }
        },
    )

    content = outpath.read_text(encoding="utf-8")
    data = _page_data(content)
    assert data["frame_metadata"] is not None
    assert "GLViewer" not in content


# ---------------------------------------------------------------------------
# Slice 1: page template, theme and offline-by-default
# ---------------------------------------------------------------------------


def test_make_density_interactive_no_google_fonts(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "out.html"
    make_density_interactive(df, _plane_pair(), outpath)
    content = outpath.read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in content
    assert "fonts.gstatic.com" not in content


def test_make_density_interactive_theme_tokens_present(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "out.html"
    make_density_interactive(df, _plane_pair(), outpath)
    content = outpath.read_text(encoding="utf-8")
    assert "--ca-bg" in content
    assert "--ca-accent" in content
    assert "prefers-color-scheme" in content


def test_make_density_interactive_default_output_is_fully_offline(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "offline.html"
    # embed_xyz_payload defaults to False when omitted entirely; set it
    # explicitly so this test also exercises the vendored-3Dmol inlining,
    # while include_plotlyjs/include_3dmol are left at their Slice-1 defaults.
    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={"plots": {"interactive": {"embed_xyz_payload": True}}},
    )
    content = outpath.read_text(encoding="utf-8")
    assert re.search(r'<script[^>]+src="https?://', content) is None
    assert re.search(r'<link[^>]+href="https?://', content) is None
    assert "Plotly.newPlot(" in content
    assert "GLViewer" in content


def test_make_density_interactive_include_3dmol_cdn_uses_cdn_tag(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "cdn_3dmol.html"
    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={
            "plots": {
                "interactive": {"embed_xyz_payload": True, "include_3dmol": "cdn"}
            }
        },
    )
    content = outpath.read_text(encoding="utf-8")
    assert re.search(r'<script src="https://cdn\.jsdelivr\.net/npm/3dmol@[^"]+"></script>', content)
    assert "GLViewer" not in content


def test_make_density_interactive_embeds_ui_state_from_config(tmp_path):
    df = _make_angle_df()
    outpath = tmp_path / "theme_dark.html"
    make_density_interactive(
        df,
        _plane_pair(),
        outpath,
        config={"plots": {"interactive": {"theme": "dark"}}},
    )
    content = outpath.read_text(encoding="utf-8")
    data = _page_data(content)
    assert data["ui_state"]["theme"] == "dark"


def test_render_density_page_is_deterministic():
    page_data = {
        "schema_version": 1,
        "pair": {"name": "plane", "x_col": "x", "y_col": "y",
                  "x_label": "X", "y_label": "Y", "title": "Plane"},
        "header": {"frame_count": 10, "bin_count_x": 5, "bin_count_y": 5,
                    "scale_mode_label": "count"},
        "axis_spec": {"x_col": "x", "y_col": "y"},
        "bin_geometry": None,
        "bin_frame_metadata": None,
        "bin_xyz_payloads": None,
        "frame_metadata": [{"frame_id": 0, "x": 1.0, "y": 2.0}],
        "ui_state": {
            "theme": "auto", "scale_mode": None, "state_overlay_visible": False,
            "temperature": None, "unit": None, "pinned_bins": [],
        },
        "plot_html": "<div>plot</div>",
        "include_3dmol": "inline",
    }
    first = render_density_page(page_data)
    second = render_density_page(page_data)
    assert first == second

    reordered = {
        "include_3dmol": "inline",
        "plot_html": "<div>plot</div>",
        "ui_state": {
            "pinned_bins": [], "unit": None, "temperature": None,
            "state_overlay_visible": False, "scale_mode": None, "theme": "auto",
        },
        "frame_metadata": [{"x": 1.0, "y": 2.0, "frame_id": 0}],
        "bin_xyz_payloads": None,
        "bin_frame_metadata": None,
        "bin_geometry": None,
        "axis_spec": {"y_col": "y", "x_col": "x"},
        "header": {"scale_mode_label": "count", "bin_count_y": 5,
                    "bin_count_x": 5, "frame_count": 10},
        "pair": {"title": "Plane", "y_label": "Y", "x_label": "X",
                  "y_col": "y", "x_col": "x", "name": "plane"},
        "schema_version": 1,
    }
    third = render_density_page(reordered)
    assert first == third


def _interactive_cfg(**interactive) -> dict:
    return {"plots": {"interactive": interactive}}


# ---------------------------------------------------------------------------
# Inline content must not end its <script>/<style> element early
# ---------------------------------------------------------------------------


def _minimal_page_data(frame_metadata: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "pair": {"name": "plane", "x_col": "x", "y_col": "y",
                  "x_label": "X", "y_label": "Y", "title": "Plane"},
        "header": {"frame_count": 1, "bin_count_x": 1, "bin_count_y": 1,
                    "scale_mode_label": "count"},
        "axis_spec": {"x_col": "x", "y_col": "y"},
        "bin_geometry": None,
        "bin_frame_metadata": None,
        "bin_xyz_payloads": None,
        "frame_metadata": frame_metadata or [{"frame_id": 0, "x": 1.0, "y": 2.0}],
        "ui_state": {"theme": "auto"},
        "plot_html": "<div>plot</div>",
        "include_3dmol": "inline",
    }


def test_rendered_page_inline_scripts_are_complete(tmp_path):
    outpath = tmp_path / "scripts.html"
    make_density_interactive(
        _make_angle_df(),
        _plane_pair(),
        outpath,
        config=_interactive_cfg(embed_xyz_payload=True, include_plotlyjs="cdn"),
    )
    inline = [text.strip() for attrs, text in _scripts(outpath.read_text(encoding="utf-8"))
              if "src" not in attrs]
    assert _asset_text("viewer.js").strip() in inline
    assert _asset_text("vendor/3Dmol-min.js").strip() in inline


def test_render_density_page_page_data_survives_script_close_text():
    comment = "</script><b>not markup</b>"
    html = render_density_page(_minimal_page_data([{"frame_id": 0, "comment_line": comment}]))
    blob = next(text for attrs, text in _scripts(html) if attrs.get("id") == "page-data")
    assert json.loads(blob)["frame_metadata"][0]["comment_line"] == comment


def test_render_density_page_rejects_asset_that_closes_its_element(monkeypatch):
    import src.plots_interactive as plots_interactive

    real_load = plots_interactive._load_asset
    monkeypatch.setattr(
        plots_interactive,
        "_load_asset",
        lambda name: "// see </script>\n" if name == "viewer.js" else real_load(name),
    )
    with pytest.raises(ValueError, match="viewer.js"):
        render_density_page(_minimal_page_data())
