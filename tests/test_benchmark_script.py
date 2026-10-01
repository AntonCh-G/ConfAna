"""Smoke test for scripts/benchmark_coordinates.py on a tiny trajectory."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import yaml

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_coordinates.py"
_spec = importlib.util.spec_from_file_location("benchmark_coordinates", _SCRIPT)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

_N_ATOMS = 13
_N_FRAMES = 5


def _write_config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    lines = []
    for i in range(_N_FRAMES):
        lines += [str(_N_ATOMS), f"frame {i}"]
        lines += [f"C {x:.6f} {y:.6f} {z:.6f}" for x, y, z in rng.normal(size=(_N_ATOMS, 3))]
    xyz = tmp_path / "toy.xyz"
    xyz.write_text("\n".join(lines) + "\n")

    cfg = {
        "run_dir": str(tmp_path / "run"),
        "data": {"path_pattern": str(xyz)},
        "dof": [
            {"name": "carboxyl_dihedral", "type": "dihedral", "atoms": [6, 5, 10, 7],
             "label": "Carboxyl", "domain": [-180, 180]},
            {"name": "ester_dihedral", "type": "dihedral", "atoms": [5, 6, 12, 11],
             "label": "Ester", "domain": [-180, 180]},
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_all_phases_run_on_the_current_config_schema(tmp_path, capsys):
    config = _write_config(tmp_path)

    bench.main(["--config", str(config), "--micro"])

    out = capsys.readouterr().out
    # Cold build and warm load both report every frame.
    assert len(re.findall(rf"total_frames :\s+{_N_FRAMES}\n", out)) == 2
    assert "Stage breakdown:" in out
    assert "stale" not in out                                         # phase B was a cache hit
    assert "all 2 DoF" in out
    assert "Benchmark complete." in out
    assert len(list((tmp_path / "run" / ".bench").glob("*__coordinates.npz"))) == 1
