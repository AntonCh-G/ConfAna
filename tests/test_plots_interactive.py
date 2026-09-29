"""Tests for src/plots_interactive.py (Phase 11)."""

from __future__ import annotations

import importlib.resources
import json
import re
from html.parser import HTMLParser
from pathlib import Path

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

    # Compressed by default (Slice 7); the plain blocks are the fallback.
    assert data["bin_frame_metadata_encoded"] is not None
    assert data["bin_xyz_payloads_encoded"] is not None
    assert data["bin_frame_metadata"] is None and data["bin_xyz_payloads"] is None
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
    assert "addPin(" in content
    assert "comparison-viewer" in content
    assert "clearAllPins" in content
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
    assert data["frame_metadata_encoded"] is not None
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


# ---------------------------------------------------------------------------
# Slice 2: hover preview and pin limit
# ---------------------------------------------------------------------------


def _interactive_cfg(**interactive) -> dict:
    return {"plots": {"interactive": interactive}}


def test_make_density_interactive_embeds_default_settings(tmp_path):
    outpath = tmp_path / "settings_default.html"
    make_density_interactive(_make_angle_df(), _plane_pair(), outpath)
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["settings"] == {"hover_preview": True, "max_pinned": 15}


def test_make_density_interactive_embeds_max_pinned_from_config(tmp_path):
    outpath = tmp_path / "settings_pinned.html"
    make_density_interactive(
        _make_angle_df(), _plane_pair(), outpath, config=_interactive_cfg(max_pinned=5)
    )
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["settings"]["max_pinned"] == 5


def test_make_density_interactive_embeds_hover_preview_disabled(tmp_path):
    outpath = tmp_path / "settings_no_hover.html"
    make_density_interactive(
        _make_angle_df(), _plane_pair(), outpath, config=_interactive_cfg(hover_preview=False)
    )
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["settings"]["hover_preview"] is False


@pytest.mark.parametrize("value", [0, -3, 2.5, "15", True])
def test_make_density_interactive_invalid_max_pinned_raises(tmp_path, value):
    with pytest.raises(ValueError, match="max_pinned"):
        make_density_interactive(
            _make_angle_df(),
            _plane_pair(),
            tmp_path / "bad.html",
            config=_interactive_cfg(max_pinned=value),
        )


@pytest.mark.parametrize("value", ["yes", 1])
def test_make_density_interactive_invalid_hover_preview_raises(tmp_path, value):
    with pytest.raises(ValueError, match="hover_preview"):
        make_density_interactive(
            _make_angle_df(),
            _plane_pair(),
            tmp_path / "bad.html",
            config=_interactive_cfg(hover_preview=value),
        )


def test_make_density_interactive_contains_hover_preview_markup(tmp_path):
    outpath = tmp_path / "hover_markup.html"
    # Plotly from CDN so the plotly_hover check can only match viewer.js.
    make_density_interactive(
        _make_angle_df(),
        _plane_pair(),
        outpath,
        config=_interactive_cfg(embed_xyz_payload=True, include_plotlyjs="cdn"),
    )
    content = outpath.read_text(encoding="utf-8")
    assert 'id="preview-viewer"' in content
    assert 'id="preview-readout"' in content
    assert 'id="panel-notice"' in content
    assert "plotly_hover" in content


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
        "settings": {"hover_preview": True, "max_pinned": 15},
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


# ---------------------------------------------------------------------------
# Slice 3: coordinate-defining atoms
# ---------------------------------------------------------------------------


def _atom_pair(x_atoms=(6, 5, 10, 7), y_atoms=(5, 6, 12, 11)) -> CoordinatePair:
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
        x_atoms=x_atoms,
        y_atoms=y_atoms,
        x_dof_type="dihedral",
        y_dof_type="dihedral",
    )


def test_make_density_interactive_embeds_axis_atoms_from_pair(tmp_path):
    outpath = tmp_path / "axis_atoms.html"
    make_density_interactive(_make_angle_df(), _atom_pair(), outpath)
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["axis_atoms"] == {
        "x": {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7]},
        "y": {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11]},
    }


def test_make_density_interactive_axis_atoms_null_for_dof_without_atoms(tmp_path):
    outpath = tmp_path / "axis_atoms_none.html"
    make_density_interactive(_make_angle_df(), _plane_pair(), outpath)
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["axis_atoms"]["x"]["atoms"] is None
    assert data["axis_atoms"]["y"]["atoms"] is None


@pytest.mark.parametrize(
    ("x_atoms", "bad"),
    [((6, 5, 10, 21), "[21]"), ((6, 5, 10, 30), "[30]"), ((-1, 5, 10, 7), "[-1]")],
)
def test_make_density_interactive_out_of_range_axis_atom_raises(tmp_path, x_atoms, bad):
    # _make_angle_df frames have 21 atoms, so valid indices are 0..20.
    with pytest.raises(ValueError, match=rf"x-axis DoF 'carboxyl_dihedral'.*{re.escape(bad)}"):
        make_density_interactive(
            _make_angle_df(), _atom_pair(x_atoms=x_atoms), tmp_path / "bad.html"
        )


def test_make_density_interactive_checks_axis_atoms_against_smallest_atom_count(tmp_path):
    df = _make_angle_df()
    df.loc[0, "atom_count"] = 12
    with pytest.raises(ValueError, match=r"y-axis DoF 'ester_dihedral'.*\[12\].*12 atoms"):
        make_density_interactive(df, _atom_pair(), tmp_path / "bad.html")


def test_make_density_interactive_highlight_disabled_omits_axis_atoms(tmp_path):
    outpath = tmp_path / "no_highlight.html"
    # Out-of-range atoms are not checked when nothing is highlighted.
    make_density_interactive(
        _make_angle_df(),
        _atom_pair(x_atoms=(6, 5, 10, 99)),
        outpath,
        config=_interactive_cfg(highlight_dof_atoms=False),
    )
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["axis_atoms"] is None


@pytest.mark.parametrize("value", ["no", 0])
def test_make_density_interactive_invalid_highlight_dof_atoms_raises(tmp_path, value):
    with pytest.raises(ValueError, match="highlight_dof_atoms"):
        make_density_interactive(
            _make_angle_df(),
            _atom_pair(),
            tmp_path / "bad.html",
            config=_interactive_cfg(highlight_dof_atoms=value),
        )


# ---------------------------------------------------------------------------
# Slice 4: colour-scale modes, temperature and energy unit
# ---------------------------------------------------------------------------


def _plot_figure(content: str) -> tuple[list, dict]:
    """Return (data, layout) of the Plotly.newPlot call in the page."""
    decoder = json.JSONDecoder()
    # The fragment's call passes the div id as a string literal; the inlined
    # Plotly library and viewer.js mention Plotly.newPlot( without one.
    match = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', content)
    assert match, "no Plotly.newPlot call for the figure"
    rest = content[match.end():]
    data, end = decoder.raw_decode(rest)
    rest = rest[end:].lstrip()[1:].lstrip()
    layout, _ = decoder.raw_decode(rest)
    return data, layout


def _plotly_array(value) -> np.ndarray:
    """Decode a Plotly figure array (plain list or base64 typed array)."""
    if isinstance(value, dict) and "bdata" in value:
        import base64

        arr = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        shape = value.get("shape")
        return arr.reshape([int(n) for n in str(shape).split(",")]) if shape else arr
    return np.array(value, dtype=float)


def _counts_df() -> pd.DataFrame:
    """Frames in three bins of a 3×3 grid on [0, 3): counts 4, 2, 1."""
    points = [(0.5, 0.5)] * 4 + [(1.5, 0.5)] * 2 + [(2.5, 2.5)]
    df = _make_angle_df(n=len(points))
    df["carboxyl_plane"] = [p[0] for p in points]
    df["ester_plane"] = [p[1] for p in points]
    return df


def _grid_pair() -> CoordinatePair:
    return CoordinatePair(
        name="plane",
        x_col="carboxyl_plane",
        y_col="ester_plane",
        x_label="X",
        y_label="Y",
        title="Grid",
        x_domain=(0.0, 3.0),
        y_domain=(0.0, 3.0),
        bins=3,
    )


def _build(tmp_path, config=None, df=None, name="scale.html") -> tuple[dict, str]:
    outpath = tmp_path / name
    make_density_interactive(df if df is not None else _counts_df(), _grid_pair(), outpath, config=config)
    content = outpath.read_text(encoding="utf-8")
    return _page_data(content), content


def test_scale_embeds_the_count_grid_once(tmp_path):
    """Only counts are embedded; the page derives the other modes' grids."""
    from src.payload_codec import decode_count_grid

    data, _ = _build(tmp_path)
    assert "grids" not in data["scale"]
    assert data["scale"]["grid_decimals"] == 4
    block = data["scale"]["counts"]
    assert (block["format"], block["dtype"], block["shape"]) == ("confana-count-grid-v1", "u2", [3, 3])
    # Rows are y bins, columns x bins; 0 = unsampled.
    assert decode_count_grid(block).tolist() == [[4, 2, 0], [0, 0, 0], [0, 0, 1]]


def test_scale_counts_are_plain_json_without_compression(tmp_path):
    data, _ = _build(tmp_path, config={"plots": {"interactive": {"compress_payloads": False}}})
    assert data["scale"]["counts"] == [[4, 2, None], [None, None, None], [None, None, 1]]


def test_scale_grids_cover_every_mode():
    from src.plots_interactive import _scale_grids

    grids = _scale_grids(np.array([[4, 2, 0], [0, 0, 0], [0, 0, 1]]))
    assert set(grids) == {"counts", "log_counts", "free_energy"}
    assert grids["log_counts"][0][0] == pytest.approx(np.log10(5.0), abs=1e-4)
    assert grids["free_energy"][0][:2].tolist() == [0.0, pytest.approx(np.log(2.0), abs=1e-4)]
    assert grids["free_energy"][2][2] == pytest.approx(np.log(4.0), abs=1e-4)
    for grid in grids.values():
        assert np.isnan(grid[1]).all() and np.isnan(grid[0][2])


def test_scale_embeds_unit_table_and_modes(tmp_path):
    from src.units import energy_unit_table

    data, _ = _build(tmp_path)
    assert data["scale"]["energy_units"] == energy_unit_table()
    assert set(data["scale"]["modes"]) == {"counts", "log_counts", "free_energy"}


def test_default_scale_follows_log_scale_when_unset(tmp_path):
    data, _ = _build(tmp_path)
    assert data["ui_state"]["scale_mode"] == "log_counts"
    data, _ = _build(tmp_path, config={"plots": {"density": {"log_scale": False}}}, name="c.html")
    assert data["ui_state"]["scale_mode"] == "counts"


def test_free_energy_defaults_fall_back_to_transitions(tmp_path):
    cfg = {"transitions": {"temperature": 310.0, "energy_unit": "eV"}}
    data, _ = _build(tmp_path, config=cfg)
    assert data["ui_state"]["temperature"] == 310.0
    assert data["ui_state"]["unit"] == "eV"


def test_free_energy_config_overrides_transitions(tmp_path):
    cfg = {
        "transitions": {"temperature": 310.0, "energy_unit": "eV"},
        "plots": {"interactive": {"free_energy": {"temperature": 250, "unit": "cm^-1"}}},
    }
    data, _ = _build(tmp_path, config=cfg)
    assert data["ui_state"]["temperature"] == 250.0
    assert data["ui_state"]["unit"] == "cm^-1"


def test_free_energy_without_temperature_opens_in_kt(tmp_path):
    cfg = {"plots": {"interactive": {"free_energy": {"unit": "kJ/mol"}}}}
    data, _ = _build(tmp_path, config=cfg)
    assert data["ui_state"]["temperature"] is None
    assert data["ui_state"]["unit"] == "kT"
    data, _ = _build(tmp_path, name="none.html")
    assert (data["ui_state"]["temperature"], data["ui_state"]["unit"]) == (None, "kT")


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"plots": {"interactive": {"free_energy": {"unit": "hartree"}}}},
         "plots.interactive.free_energy.unit"),
        ({"transitions": {"temperature": 300.0, "energy_unit": "kcal"}}, "transitions.energy_unit"),
        ({"plots": {"interactive": {"free_energy": {"temperature": 0}}}},
         "plots.interactive.free_energy.temperature"),
        ({"transitions": {"temperature": -5.0}}, "transitions.temperature"),
        ({"plots": {"interactive": {"default_scale": "free-energy"}}}, "default_scale"),
        ({"plots": {"interactive": {"free_energy": 300}}}, "plots.interactive.free_energy"),
    ],
)
def test_invalid_scale_config_raises(tmp_path, config, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        _build(tmp_path, config=config)


def test_initial_figure_matches_free_energy_default(tmp_path):
    from src.units import thermal_energy

    cfg = {
        "transitions": {"temperature": 300.0, "energy_unit": "kJ/mol"},
        "plots": {"interactive": {"default_scale": "free_energy"}},
    }
    data, content = _build(tmp_path, config=cfg)
    assert data["ui_state"]["scale_mode"] == "free_energy"
    fig_data, _ = _plot_figure(content)
    trace = fig_data[0]
    from src.payload_codec import decode_count_grid
    from src.plots_interactive import _scale_grids

    z = _plotly_array(trace["z"])
    assert z.dtype == np.float32
    grid = _scale_grids(decode_count_grid(data["scale"]["counts"]))["free_energy"]
    np.testing.assert_allclose(z, grid * thermal_energy("kJ/mol", 300.0), rtol=1e-6)
    assert trace["colorbar"]["title"]["text"] == "F (kJ/mol)"
    assert "F (kJ/mol): %{z:.3f}" in trace["hovertemplate"]
    assert data["header"]["scale_mode_label"] == "F (kJ/mol)"
    assert "Population-derived free-energy-like surface" in content


def test_density_modes_keep_density_subtitle(tmp_path):
    _, content = _build(tmp_path)
    assert "Coordinate-density landscape — not a potential energy surface" in content
    assert "Population-derived" not in content.split("<script")[0]


def test_scale_controls_markup_present(tmp_path):
    _, content = _build(tmp_path)
    for mode in ("log_counts", "counts", "free_energy"):
        assert f'data-scale="{mode}"' in content
    assert 'id="fe-temperature"' in content
    assert 'id="fe-unit"' in content


# ---------------------------------------------------------------------------
# Slice 5: state overlay
# ---------------------------------------------------------------------------


def _state_df() -> pd.DataFrame:
    """_counts_df with states; bead 00 and 01 number the same regions differently."""
    df = _counts_df()  # bins (0,0)×4, (1,0)×2, (2,2)×1
    df["bead_id"] = ["00", "00", "01", "01", "00", "01", "00"]
    df["state_plane"] = pd.array(["0", "0", "1", "1", "1", "0", "noise"], dtype="string")
    return df


def test_state_overlay_trace_and_toggle_present_with_state_column(tmp_path):
    cfg = {"clustering": {"groupby": ["bead_id"]}}
    data, content = _build(tmp_path, config=cfg, df=_state_df())
    states = data["states"]
    assert states["state_col"] == "state_plane"
    assert states["labels"] == ["0", "1"]
    assert states["colors"] == ["#1f77b4", "#ff7f0e"]
    assert [g["name"] for g in states["groups"]] == ["bead 00", "bead 01"]

    fig_data, layout = _plot_figure(content)
    assert len(fig_data) == 2
    overlay = fig_data[1]
    assert overlay["name"] == "states"
    assert overlay["visible"] is False
    assert overlay["opacity"] == pytest.approx(0.35)
    assert overlay["hoverinfo"] == "skip"
    # Starts on the first group: bead 00 has bin (0,0) → "0" (2 of 2) and
    # bin (1,0) → "1"; bin (2,2) is noise → no state.
    z = _plotly_array(overlay["z"])
    assert z[0, 0] == 0 and z[0, 1] == 1
    assert np.isnan(z[2, 2]) and np.isnan(z[1]).all()
    assert [a["text"] for a in layout["annotations"]] == ["0", "1"]
    assert all(a["visible"] is False for a in layout["annotations"])

    assert 'id="states-toggle"' in content
    assert 'id="state-group"' in content
    assert data["ui_state"]["state_overlay_visible"] is False
    assert data["ui_state"]["state_group"] == 0


def test_state_overlay_absent_without_state_column(tmp_path):
    df = _counts_df().drop(columns=["state_plane", "state_dihedral"])
    data, content = _build(tmp_path, df=df)
    assert data["states"] is None
    fig_data, layout = _plot_figure(content)
    assert len(fig_data) == 1
    assert not layout.get("annotations")
    assert 'id="states-toggle"' not in content
    assert 'id="state-group"' not in content
    assert data["ui_state"]["state_overlay_visible"] is False


def test_show_states_config_makes_overlay_visible(tmp_path):
    cfg = {"plots": {"interactive": {"show_states": True}}}
    data, content = _build(tmp_path, config=cfg, df=_state_df())
    fig_data, layout = _plot_figure(content)
    assert fig_data[1]["visible"] is True
    assert all(a["visible"] is True for a in layout["annotations"])
    assert data["ui_state"]["state_overlay_visible"] is True
    # No clustering.groupby: one group of all frames.
    assert [g["name"] for g in data["states"]["groups"]] == ["all frames"]


@pytest.mark.parametrize("value", ["yes", 1])
def test_invalid_show_states_raises(tmp_path, value):
    with pytest.raises(ValueError, match="show_states"):
        _build(tmp_path, config={"plots": {"interactive": {"show_states": value}}}, df=_state_df())


# ---------------------------------------------------------------------------
# Theme colour scales
# ---------------------------------------------------------------------------


def test_theme_colorscale_keeps_viridis_on_white_and_trims_it_on_dark():
    from src.plots_interactive import (
        _PLOTLY_THEME_COLORS,
        _contrast_ratio,
        _named_colorscale,
        _rgb,
        _theme_colorscale,
    )

    viridis = _named_colorscale("viridis", "test")
    assert _theme_colorscale("viridis", "light") == viridis
    dark = _theme_colorscale("viridis", "dark")
    dark_bg = _rgb(_PLOTLY_THEME_COLORS["dark"]["plot_bgcolor"])
    assert _contrast_ratio(_rgb(dark[0][1]), dark_bg) >= 2.0
    assert _contrast_ratio(_rgb(viridis[0][1]), dark_bg) < 2.0
    # Same direction: the bright end stays viridis yellow.
    assert _rgb(dark[-1][1]) == _rgb(viridis[-1][1])
    assert dark[0][0] == 0.0 and dark[-1][0] == 1.0


def test_theme_colorscale_trims_near_white_end_on_light():
    from src.plots_interactive import _contrast_ratio, _named_colorscale, _rgb, _theme_colorscale

    blues = _named_colorscale("Blues", "test")
    light = _theme_colorscale("Blues", "light")
    white = (255.0, 255.0, 255.0)
    assert _contrast_ratio(_rgb(blues[0][1]), white) < 1.25
    assert _contrast_ratio(_rgb(light[0][1]), white) >= 1.25
    assert _rgb(light[-1][1]) == _rgb(blues[-1][1])


def test_theme_colorscale_ignores_a_pale_middle():
    from src.plots_interactive import _named_colorscale, _theme_colorscale

    # jet's yellow middle is pale on white, but only the ends are trimmed.
    assert _theme_colorscale("jet", "light") == _named_colorscale("jet", "test")


def test_page_embeds_both_theme_colorscales_and_bakes_the_active_one(tmp_path):
    from src.plots_interactive import _theme_colorscale

    data, content = _build(tmp_path)
    scales = data["scale"]["colorscales"]
    assert scales == {
        "light": _theme_colorscale("Viridis", "light"),
        "dark": _theme_colorscale("Viridis", "dark"),
    }
    fig_data, _ = _plot_figure(content)
    assert fig_data[0]["colorscale"] == scales["light"]  # theme: auto bakes light

    data, content = _build(tmp_path, config=_interactive_cfg(theme="dark"), name="dark.html")
    fig_data, _ = _plot_figure(content)
    assert fig_data[0]["colorscale"] == data["scale"]["colorscales"]["dark"]


def test_explicit_theme_colorscales_are_used_unchanged(tmp_path):
    from src.plots_interactive import _named_colorscale

    cfg = _interactive_cfg(theme_colorscales={"dark": "plasma_r"})
    data, _ = _build(tmp_path, config=cfg)
    assert data["scale"]["colorscales"]["dark"] == _named_colorscale("plasma_r", "test")


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"dark": "not-a-scale"}, "plots.interactive.theme_colorscales.dark"),
        ({"dim": "viridis"}, "theme_colorscales"),
        ("viridis", "theme_colorscales"),
    ],
)
def test_invalid_theme_colorscales_raise(tmp_path, value, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        _build(tmp_path, config=_interactive_cfg(theme_colorscales=value))


# ---------------------------------------------------------------------------
# Degree axis ticks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("span", "step"),
    [(360, 60), (180, 30), (90, 15), (60, 10), (20, 5), (10, 2), (0.5, 1), (1000, 90), (-360, 60)],
)
def test_degree_tick_step(span, step):
    from src.plots_interactive import _degree_tick_step

    assert _degree_tick_step(span) == step


@pytest.mark.parametrize(
    ("dof_type", "label", "expected"),
    [
        ("dihedral", "Carboxyl dihedral (°)", True),
        ("angle", "Bond angle", True),
        ("distance", "C–O distance (°)", False),  # the DoF type wins over the label
        (None, "Carboxyl vs ring (°)", True),
        (None, "PC1", False),
    ],
)
def test_is_degree_axis(dof_type, label, expected):
    from src.plots_interactive import _is_degree_axis

    assert _is_degree_axis(dof_type, label) is expected


def test_degree_axes_get_degree_ticks(tmp_path):
    outpath = tmp_path / "ticks.html"
    pair = CoordinatePair(
        name="mixed", x_col="carboxyl_dihedral", y_col="carboxyl_plane",
        x_label="Carboxyl dihedral (°)", y_label="C–O distance (Å)", title="t",
        x_domain=(-180.0, 180.0), y_domain=(0.0, 180.0),
        x_dof_type="dihedral", y_dof_type="distance",
    )
    make_density_interactive(_make_angle_df(), pair, outpath)
    content = outpath.read_text(encoding="utf-8")
    _, layout = _plot_figure(content)
    assert layout["xaxis"]["tickmode"] == "linear"
    assert layout["xaxis"]["tick0"] == 0
    assert layout["xaxis"]["dtick"] == 60
    assert layout["xaxis"]["ticksuffix"] == "°"
    assert "dtick" not in layout["yaxis"] and "ticksuffix" not in layout["yaxis"]
    data = _page_data(content)
    assert data["axis_ticks"]["x"] is True and data["axis_ticks"]["y"] is False
    assert data["axis_ticks"]["steps"] == [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0]


def test_plane_axes_get_30_degree_ticks(tmp_path):
    outpath = tmp_path / "plane_ticks.html"
    make_density_interactive(_make_angle_df(), _plane_pair(), outpath)
    _, layout = _plot_figure(outpath.read_text(encoding="utf-8"))
    assert layout["xaxis"]["dtick"] == 30 and layout["yaxis"]["dtick"] == 30


# ---------------------------------------------------------------------------
# Slice 6 — navigation between coordinate pairs
# ---------------------------------------------------------------------------


def _siblings() -> list[dict]:
    return [
        {"name": "plane", "title": "Plane density", "filename": "density_plane.html"},
        {"name": "dihedral", "title": "Dihedral density", "filename": "density_dihedral.html"},
    ]


def test_make_density_interactive_embeds_sibling_pair_links(tmp_path):
    outpath = tmp_path / "density_plane.html"
    make_density_interactive(_make_angle_df(), _plane_pair(), outpath, siblings=_siblings())
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["navigation"]["pairs"] == [
        {
            "name": "plane",
            "title": "Plane density",
            "filename": "density_plane.html",
            "current": True,
        },
        {
            "name": "dihedral",
            "title": "Dihedral density",
            "filename": "density_dihedral.html",
            "current": False,
        },
    ]


def test_make_density_interactive_marks_the_current_pair(tmp_path):
    outpath = tmp_path / "density_dihedral.html"
    make_density_interactive(_make_angle_df(), _dihedral_pair(), outpath, siblings=_siblings())
    data = _page_data(outpath.read_text(encoding="utf-8"))
    current = [p["name"] for p in data["navigation"]["pairs"] if p["current"]]
    assert current == ["dihedral"]


def test_make_density_interactive_page_has_nav_container(tmp_path):
    outpath = tmp_path / "density_plane.html"
    make_density_interactive(_make_angle_df(), _plane_pair(), outpath, siblings=_siblings())
    assert 'id="pair-nav"' in outpath.read_text(encoding="utf-8")


def test_make_density_interactive_sibling_title_defaults_to_name(tmp_path):
    outpath = tmp_path / "density_plane.html"
    make_density_interactive(
        _make_angle_df(),
        _plane_pair(),
        outpath,
        siblings=[{"name": "plane", "filename": "density_plane.html"}],
    )
    data = _page_data(outpath.read_text(encoding="utf-8"))
    assert data["navigation"]["pairs"][0]["title"] == "plane"


@pytest.mark.parametrize("siblings", [None, []])
def test_make_density_interactive_navigation_absent_without_siblings(tmp_path, siblings):
    outpath = tmp_path / "density_plane.html"
    make_density_interactive(_make_angle_df(), _plane_pair(), outpath, siblings=siblings)
    assert _page_data(outpath.read_text(encoding="utf-8"))["navigation"] is None


@pytest.mark.parametrize(
    ("siblings", "message"),
    [
        ([{"name": "plane"}], "non-empty string 'filename'"),
        ([{"filename": "density_plane.html"}], "non-empty string 'name'"),
        ([{"name": "plane", "filename": ""}], "non-empty string 'filename'"),
        (["density_plane.html"], "must be a mapping"),
        ([{"name": "plane", "filename": "sub/density_plane.html"}], "plain file name"),
        ([{"name": "plane", "filename": "/abs/density_plane.html"}], "plain file name"),
        (
            [
                {"name": "plane", "filename": "density_plane.html"},
                {"name": "plane", "filename": "other.html"},
            ],
            "duplicate sibling page name",
        ),
        ([{"name": "dihedral", "filename": "density_dihedral.html"}], "is not among the sibling"),
    ],
)
def test_make_density_interactive_invalid_siblings_raise(tmp_path, siblings, message):
    with pytest.raises(ValueError, match=message):
        make_density_interactive(
            _make_angle_df(), _plane_pair(), tmp_path / "density_plane.html", siblings=siblings
        )


# ---------------------------------------------------------------------------
# Pins carried between pair pages (docs/adr/0002)
# ---------------------------------------------------------------------------


def _pin_siblings() -> list[dict]:
    return [
        {"name": "plane", "filename": "density_plane.html",
         "columns": ["carboxyl_plane", "ester_plane"]},
        {"name": "dihedral", "filename": "density_dihedral.html",
         "columns": ["carboxyl_dihedral", "ester_dihedral"]},
        # A pair whose column this table lacks: its pins show "not on this map".
        {"name": "absent", "filename": "density_absent.html", "columns": ["no_such_dof", "ester_plane"]},
    ]


@pytest.mark.parametrize("embed", [True, False])
def test_every_frame_record_carries_frame_id_and_all_pairs_columns(tmp_path, embed):
    from src.payload_codec import decode_columns

    df, _ = _bin_df()
    df["carboxyl_dihedral"] = np.linspace(-170.0, 170.0, len(df))
    df["ester_dihedral"] = np.linspace(170.0, -170.0, len(df))
    data = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "density_plane.html",
            config=_bin_cfg(embed_xyz_payload=embed), siblings=_pin_siblings(),
        ).read_text(encoding="utf-8")
    )
    block = "bin_frame_metadata_encoded" if embed else "frame_metadata_encoded"
    decoded = decode_columns(data[block])
    records = list(decoded.values()) if isinstance(decoded, dict) else decoded
    assert records
    by_id = df.set_index("frame_id")
    for record in records:
        row = by_id.loc[record["frame_id"]]
        for column in ("carboxyl_plane", "ester_plane", "carboxyl_dihedral", "ester_dihedral"):
            assert record[column] == pytest.approx(row[column])
        assert "no_such_dof" not in record


@pytest.mark.parametrize("embed", [True, False])
def test_frame_records_carry_the_shifted_columns_the_maps_use(tmp_path, embed):
    # coordinate_transforms map a pair onto '<dof>_shifted'; pins must carry that
    # column, or they are "not on this map" even on the page they came from.
    from src.cli import _pair_siblings
    from src.payload_codec import decode_columns

    df, _ = _bin_df()
    df["ester_plane_shifted"] = (df["ester_plane"] + 90.0) % 180.0
    pair = CoordinatePair(**{**_plane_pair().__dict__, "feature_y_col": "ester_plane_shifted"})
    siblings = _pair_siblings([("plane", pair)])
    assert siblings[0]["columns"] == ["carboxyl_plane", "ester_plane_shifted"]

    data = _page_data(
        make_density_interactive(
            df, pair, tmp_path / "density_plane.html",
            config=_bin_cfg(embed_xyz_payload=embed, compress_payloads=False),
            siblings=siblings,
        ).read_text(encoding="utf-8")
    )
    records = (
        list(data["bin_frame_metadata"].values()) if embed else data["frame_metadata"]
    )
    by_id = df.set_index("frame_id")
    for record in records:
        assert record["ester_plane_shifted"] == pytest.approx(
            by_id.loc[record["frame_id"], "ester_plane_shifted"]
        )


@pytest.mark.parametrize("embed", [True, False])
def test_grid_is_embedded_in_both_page_modes(tmp_path, embed):
    df, _ = _bin_df()
    data = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "grid.html", config=_bin_cfg(embed_xyz_payload=embed)
        ).read_text(encoding="utf-8")
    )
    assert data["grid"] == {
        "x_min": 0.0, "y_min": 0.0, "bin_w": 15.0, "bin_h": 15.0, "n_bins_x": 12, "n_bins_y": 12,
    }
    if embed:
        assert data["bin_geometry"] == data["grid"]


@pytest.mark.parametrize("columns", ["carboxyl_plane", [1, 2], {"x": "carboxyl_plane"}])
def test_invalid_sibling_columns_raise(tmp_path, columns):
    siblings = [{"name": "plane", "filename": "density_plane.html", "columns": columns}]
    with pytest.raises(ValueError, match="'columns' to be a list of column names"):
        make_density_interactive(
            _make_angle_df(), _plane_pair(), tmp_path / "density_plane.html", siblings=siblings
        )


# ---------------------------------------------------------------------------
# Slice 7 — compressed embedded data
# ---------------------------------------------------------------------------


def _bin_df(
    n: int = 400, seed: int = 5, atoms: int = 3, coord_range: float = 5.0
) -> tuple[pd.DataFrame, Path]:
    """Return a coordinate table whose frames point into a real xyz file."""
    import tempfile

    rng = np.random.default_rng(seed)
    path = Path(tempfile.mkdtemp()) / "traj.xyz"
    rows = []
    with open(path, "wb") as fh:
        for i in range(n):
            coords = rng.uniform(-coord_range, coord_range, (atoms, 3))
            offset = fh.tell()
            lines = [str(atoms), f"frame {i}"]
            lines += [f"C {x:.6f} {y:.6f} {z:.6f}" for x, y, z in coords]
            fh.write(("\n".join(lines) + "\n").encode("utf-8"))
            rows.append(
                {
                    "frame_id": i,
                    "source_file": str(path),
                    "trajectory_id": "traj0",
                    "bead_id": None,
                    "frame_number": i,
                    "byte_offset": offset,
                    "atom_count": atoms,
                    "comment_line": f"frame {i}",
                    "local_frame_index": i,
                    "global_frame_index": i,
                    "carboxyl_plane": rng.uniform(0, 180),
                    "ester_plane": rng.uniform(0, 180),
                }
            )
    return pd.DataFrame(rows), path


def _bin_cfg(**interactive) -> dict:
    base = {"embed_xyz_payload": True, "include_plotlyjs": "cdn", "alignment": None}
    return {"plots": {"density": {"bins": 12}, "interactive": {**base, **interactive}}}


def test_compressed_payloads_decode_back_to_the_plain_blocks(tmp_path):
    from src.payload_codec import decode_columns, decode_structures

    df, _ = _bin_df()
    plain = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "plain.html", config=_bin_cfg(compress_payloads=False)
        ).read_text(encoding="utf-8")
    )
    packed = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "packed.html", config=_bin_cfg()
        ).read_text(encoding="utf-8")
    )

    assert packed["bin_xyz_payloads"] is None and packed["bin_frame_metadata"] is None
    assert plain["bin_xyz_payloads_encoded"] is None
    assert plain["bin_frame_metadata_encoded"] is None

    structures = decode_structures(packed["bin_xyz_payloads_encoded"])
    # The embedded JSON sorts its keys, so compare the sets, not the order.
    assert sorted(structures) == sorted(plain["bin_xyz_payloads"])
    for key, text in plain["bin_xyz_payloads"].items():
        original = np.array([[float(v) for v in ln.split()[1:4]] for ln in text.splitlines()[2:]])
        rebuilt = np.array(
            [[float(v) for v in ln.split()[1:4]] for ln in structures[key].splitlines()[2:]]
        )
        assert np.abs(rebuilt - original).max() <= 0.0005 + 1e-12

    assert decode_columns(packed["bin_frame_metadata_encoded"]) == plain["bin_frame_metadata"]


def test_compressed_payloads_shrink_the_embedded_blocks(tmp_path):
    df, _ = _bin_df(n=2000, atoms=21)
    plain = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "plain.html", config=_bin_cfg(compress_payloads=False)
        ).read_text(encoding="utf-8")
    )
    packed = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "packed.html", config=_bin_cfg()
        ).read_text(encoding="utf-8")
    )

    def size(data, keys):
        return sum(len(json.dumps(data[k])) for k in keys)

    assert size(packed, ["bin_xyz_payloads_encoded", "bin_frame_metadata_encoded"]) < size(
        plain, ["bin_xyz_payloads", "bin_frame_metadata"]
    ) / 4


def test_compressed_per_frame_metadata_round_trips(tmp_path):
    from src.payload_codec import decode_columns

    df = _make_angle_df()
    packed = _page_data(
        make_density_interactive(
            df,
            _plane_pair(),
            tmp_path / "frames.html",
            config={"plots": {"interactive": {"embed_xyz_payload": False}}},
        ).read_text(encoding="utf-8")
    )
    plain = _page_data(
        make_density_interactive(
            df,
            _plane_pair(),
            tmp_path / "frames_plain.html",
            config={"plots": {"interactive": {"embed_xyz_payload": False, "compress_payloads": False}}},
        ).read_text(encoding="utf-8")
    )
    assert packed["frame_metadata"] is None
    assert decode_columns(packed["frame_metadata_encoded"]) == plain["frame_metadata"]


def test_coordinate_step_from_config_is_used(tmp_path):
    df, _ = _bin_df()
    data = _page_data(
        make_density_interactive(
            df, _plane_pair(), tmp_path / "step.html", config=_bin_cfg(coordinate_step=0.01)
        ).read_text(encoding="utf-8")
    )
    assert data["bin_xyz_payloads_encoded"]["step"] == 0.01


def test_compressed_blocks_are_identical_across_runs(tmp_path):
    """gzip carries no timestamp, so repeated builds embed the same bytes."""
    df, _ = _bin_df()
    blocks = []
    for name in ("a.html", "b.html"):
        data = _page_data(
            make_density_interactive(
                df, _plane_pair(), tmp_path / name, config=_bin_cfg()
            ).read_text(encoding="utf-8")
        )
        blocks.append((data["bin_xyz_payloads_encoded"], data["bin_frame_metadata_encoded"]))
    assert blocks[0] == blocks[1]


def test_coordinate_out_of_16_bit_range_fails_the_build(tmp_path):
    # ±100 Å is past the ±32.767 Å that 16-bit steps of 0.001 Å reach.
    df, _ = _bin_df(n=20, coord_range=100.0)
    with pytest.raises(ValueError, match="16-bit"):
        make_density_interactive(df, _plane_pair(), tmp_path / "far.html", config=_bin_cfg())


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_invalid_compress_payloads_raises(tmp_path, value):
    with pytest.raises(ValueError, match="compress_payloads"):
        make_density_interactive(
            _make_angle_df(), _plane_pair(), tmp_path / "bad.html",
            config={"plots": {"interactive": {"compress_payloads": value}}},
        )


@pytest.mark.parametrize("value", [0, -0.1, "0.001", True])
def test_invalid_coordinate_step_raises(tmp_path, value):
    with pytest.raises(ValueError, match="coordinate_step"):
        make_density_interactive(
            _make_angle_df(), _plane_pair(), tmp_path / "bad.html",
            config={"plots": {"interactive": {"coordinate_step": value}}},
        )
