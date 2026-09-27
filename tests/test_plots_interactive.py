"""Tests for src/plots_interactive.py (Phase 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import CoordinatePair
from src.plots_interactive import make_density_interactive


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
    assert 'id="frame-metadata"' in content


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
    frame_meta_idx = content.find('id="frame-metadata"')
    plotly_click_idx = content.find("plotly_click")

    assert frame_meta_idx != -1
    assert plotly_click_idx != -1
    assert frame_meta_idx < plotly_click_idx


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
    bin_meta_idx = content.find('id="bin-frame-metadata"')
    bin_xyz_idx = content.find('id="bin-xyz-payloads"')
    plotly_click_idx = content.find("plotly_click")

    assert bin_meta_idx != -1
    assert bin_xyz_idx != -1
    assert plotly_click_idx != -1
    assert bin_meta_idx < plotly_click_idx
    assert bin_xyz_idx < plotly_click_idx


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
    assert 'id="frame-metadata"' in content
    assert "3Dmol-min.js" not in content
