"""Tests for the bundled src/interactive_assets package (Slice 1)."""

from __future__ import annotations

import importlib.resources


def _assets():
    return importlib.resources.files("src.interactive_assets")


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
