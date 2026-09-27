"""Tests for the bundled src/interactive_assets package."""

from __future__ import annotations

import importlib.resources
import shutil
import subprocess

import pytest


def _assets():
    return importlib.resources.files("src.interactive_assets")


def test_viewer_js_does_not_relayout_on_afterplot():
    # Plotly.relayout always forces a full redraw that re-emits plotly_afterplot,
    # so an afterplot -> relayout handler loops forever.
    text = _assets().joinpath("viewer.js").read_text(encoding="utf-8")
    assert "plotly_afterplot" not in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_viewer_js_parses_with_node():
    with importlib.resources.as_file(_assets().joinpath("viewer.js")) as path:
        result = subprocess.run(
            ["node", "--check", str(path)], capture_output=True, text=True
        )
    assert result.returncode == 0, result.stderr


def test_vendored_3dmol_is_packaged():
    path = _assets().joinpath("vendor/3Dmol-min.js")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert len(text) > 100_000
    assert "GLViewer" in text


def test_vendored_3dmol_license_is_packaged_and_bsd():
    path = _assets().joinpath("vendor/LICENSE-3dmol.txt")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "BSD" in text


def test_page_template_and_css_and_js_are_packaged():
    for name in ("page.html", "viewer.css", "viewer.js"):
        path = _assets().joinpath(name)
        assert path.is_file(), f"missing packaged asset: {name}"
        assert len(path.read_text(encoding="utf-8")) > 0
