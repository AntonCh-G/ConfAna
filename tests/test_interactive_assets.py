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


# ---------------------------------------------------------------------------
# Payload codec: the JavaScript decoder must agree with the Python one
# ---------------------------------------------------------------------------

_CODEC_START = "// --- payload codec start ---"
_CODEC_END = "// --- payload codec end ---"


def _codec_js() -> str:
    """The self-contained decoder section of viewer.js."""
    text = _assets().joinpath("viewer.js").read_text(encoding="utf-8")
    start = text.index(_CODEC_START)
    end = text.index(_CODEC_END)
    return text[start:end]


def test_viewer_js_exposes_the_codec_section():
    codec = _codec_js()
    for name in ("caDecodeStructures", "caStructureText", "caDecodeColumns", "caColumnsRecord"):
        assert f"function {name}(" in codec


def _node_has_decompression_stream() -> bool:
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(
        ["node", "-e", "process.stdout.write(typeof DecompressionStream)"],
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip() == "function"


@pytest.mark.skipif(
    not _node_has_decompression_stream(),
    reason="node with DecompressionStream (>= 18) is not installed",
)
def test_js_decoder_matches_python_decoder(tmp_path):
    """Run viewer.js's decoder in Node on blocks encoded by src/payload_codec.py."""
    import json

    import numpy as np

    from src.payload_codec import (
        decode_columns,
        decode_structures,
        encode_columns,
        encode_structures,
    )

    rng = np.random.default_rng(0)
    elements = ["C", "O", "H", "H"]
    payloads = {}
    for i in range(5):
        coords = rng.uniform(-8.0, 8.0, (len(elements), 3))
        lines = [str(len(elements)), f"frame {i}"]
        lines += [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in zip(elements, coords)]
        payloads[f"{i}_{i + 2}"] = "\n".join(lines) + "\n"
    records = {
        key: {
            "source_file": "/data/traj.xyz",
            "frame_number": i,
            "bead_id": None if i % 2 else "00",
            "carboxyl_plane": 1.5 * i,
        }
        for i, key in enumerate(payloads)
    }

    fixture = tmp_path / "blocks.json"
    fixture.write_text(
        json.dumps(
            {"structures": encode_structures(payloads), "columns": encode_columns(records)}
        ),
        encoding="utf-8",
    )

    script = tmp_path / "decode.cjs"
    script.write_text(
        _codec_js()
        + """
const blocks = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf-8'));
Promise.all([
  caDecodeStructures(blocks.structures),
  caDecodeColumns(blocks.columns)
]).then(([structures, columns]) => {
  const out = {structures: {}, records: {}};
  structures.keys.forEach((key, i) => {
    out.structures[key] = caStructureText(structures, key);
    out.records[key] = caColumnsRecord(columns, columns.index[key]);
  });
  process.stdout.write(JSON.stringify(out));
}).catch((err) => { console.error(err); process.exit(1); });
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["node", str(script), str(fixture)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    from_js = json.loads(result.stdout)

    from_python = decode_structures(encode_structures(payloads))
    python_records = decode_columns(encode_columns(records))

    assert list(from_js["structures"]) == list(from_python)
    for key, text in from_python.items():
        js_lines = from_js["structures"][key].splitlines()
        py_lines = text.splitlines()
        assert js_lines[:2] == py_lines[:2]
        for js_line, py_line in zip(js_lines[2:], py_lines[2:]):
            assert js_line.split()[0] == py_line.split()[0]
            js_xyz = [float(v) for v in js_line.split()[1:4]]
            py_xyz = [float(v) for v in py_line.split()[1:4]]
            assert js_xyz == pytest.approx(py_xyz, abs=1e-9)

    assert from_js["records"] == python_records
