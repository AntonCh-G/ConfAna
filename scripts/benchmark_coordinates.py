"""Benchmark the load/build coordinate-table stage.

Run from the project root:
    python scripts/benchmark_coordinates.py                      # MD17 example config
    python scripts/benchmark_coordinates.py --config FILE.yaml
    python scripts/benchmark_coordinates.py --no-cold --micro
    python scripts/benchmark_coordinates.py --help

Phases
------
A  Cold build  — rebuilds the coordinate table from xyz; with --n-jobs 1
                 (the default) it also times every substep via StageTimer.
B  Warm load   — loads the same table again from its NPZ cache.
C  Micro       — times the configured DoF on synthetic coordinates, one
                 frame at a time and as one batch.

The benchmark writes its own cache (``<run_dir>/.bench/coordinates.npz`` or
``--cache``), so the run's real caches are never touched. Frame indices
(``.frameindex.npz``) are reused if present, as in a normal run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _sep(char: str = "-", width: int = 64) -> str:
    return char * width


def _header(label: str, width: int = 64) -> None:
    print()
    print(_sep("=", width))
    print(f"  {label}")
    print(_sep("=", width))


def _section(label: str, width: int = 64) -> None:
    print()
    print(f"[{label}]")
    print(_sep("-", width))


def _fmt_fps(fps: float) -> str:
    if fps >= 1_000_000:
        return f"{fps / 1e6:.2f} M frames/s"
    if fps >= 1_000:
        return f"{fps / 1e3:.1f} k frames/s"
    return f"{fps:.0f} frames/s"


def _build_kwargs(cfg: dict, cache_path: Path, n_jobs: int) -> dict:
    """Keyword arguments for load_or_build_coordinate_table_cache from a run config."""
    from confana.coordinate_config import resolve_dof_definitions

    data_cfg = cfg.get("data", {})
    if "path_pattern" not in data_cfg:
        raise SystemExit("ERROR: data.path_pattern is not set in the config.")
    return {
        "path_pattern": str(data_cfg["path_pattern"]),
        "dof_defs": resolve_dof_definitions(cfg),
        "cache_path": cache_path,
        "trajectory_id_pattern": data_cfg.get("trajectory_id_pattern"),
        "bead_id_pattern": data_cfg.get("bead_id_pattern"),
        "index_cache_dir": cfg.get("cache", {}).get("index_cache_dir"),
        "n_jobs": n_jobs,
        "bond_break_cfg": cfg.get("bond_break"),
        "frame_range_cfg": cfg.get("frame_range"),
    }


# ---------------------------------------------------------------------------
# Phase A: cold build
# ---------------------------------------------------------------------------

# What each StageTimer label in io_coordinates measures.
_STAGE_NOTES = {
    "index_loading": "frame-index cache check, or the one-pass file scan",
    "array_alloc": "pre-allocating the output arrays",
    "frame_iter": "reading and parsing xyz frames",
    "batch_geometry": "vectorised DoF geometry per chunk of frames",
    "array_write": "copying chunk results into the output arrays",
    "df_assembly": "building the pandas DataFrame",
}


def _phase_a(kwargs: dict) -> int:
    """Force-rebuild the coordinate table and collect fine-grained timing."""
    from confana.bench import StageTimer
    from confana.io_coordinates import load_or_build_coordinate_table_cache

    _section("Phase A — cold build (force_rebuild=True)")

    n_jobs = kwargs["n_jobs"]
    # Stage-level timer only works in serial mode; suppress it in parallel runs.
    timer = StageTimer() if n_jobs == 1 else None
    mode = "(serial — stage timer active)" if n_jobs == 1 else "(parallel — no stage breakdown)"
    print(f"  n_jobs       : {n_jobs}  {mode}")

    t0 = time.perf_counter()
    df, _ = load_or_build_coordinate_table_cache(**kwargs, force_rebuild=True, _timer=timer)
    t_total = time.perf_counter() - t0

    n = len(df)
    fps = _div(n, t_total)
    print(f"  total_frames : {n:>12,}")
    print(f"  total_time   : {t_total:>10.2f} s")
    print(f"  throughput   : {fps:>10,.0f} frames/s  ({_fmt_fps(fps)})")

    if timer is not None:
        print()
        print("  Stage breakdown:")
        print(timer.report(total_frames=n))
        _print_bottlenecks(timer, n)
    return n


def _print_bottlenecks(timer, total_frames: int) -> None:
    """Rank the stages by time, with what each one measures."""
    grand = timer.grand_total() or 1.0
    ranked = sorted(
        ((lbl, t) for lbl, t in timer._totals.items() if not lbl.startswith("index:")),
        key=lambda kv: -kv[1],
    )
    if not ranked:
        return
    print()
    print("  Stages by time:")
    for rank, (lbl, t) in enumerate(ranked, 1):
        pct = t / grand * 100.0
        us_per_frame = _div(t, total_frames) * 1e6
        note = _STAGE_NOTES.get(lbl, "")
        print(f"  {rank}. {lbl:<16}  {pct:5.1f}%  {us_per_frame:7.2f} µs/frame  {note}")


# ---------------------------------------------------------------------------
# Phase B: warm load
# ---------------------------------------------------------------------------


def _phase_b(kwargs: dict) -> None:
    """Load the coordinate table from the NPZ cache and measure throughput."""
    from confana.io_coordinates import load_or_build_coordinate_table_cache

    _section("Phase B — warm load (cache hit)")

    cache_path = Path(kwargs["cache_path"])
    if not cache_path.exists():
        print("  SKIP: cache not found — run Phase A first.")
        return

    t0 = time.perf_counter()
    df, cache_hit = load_or_build_coordinate_table_cache(**kwargs)
    t_total = time.perf_counter() - t0
    if not cache_hit:
        print("  NOTE: the cache was stale and has been rebuilt; this is not a warm load.")

    n = len(df)
    fps = _div(n, t_total)
    size_mb = cache_path.stat().st_size / 1e6
    print(f"  total_frames : {n:>12,}")
    print(f"  total_time   : {t_total:>10.3f} s")
    print(f"  throughput   : {fps:>10,.0f} frames/s  ({_fmt_fps(fps)})")
    print(f"  cache_size   : {size_mb:>10.1f} MB")
    print(f"  read_MB/s    : {_div(size_mb, t_total):>10.1f} MB/s")


# ---------------------------------------------------------------------------
# Phase C: geometry micro-benchmark
# ---------------------------------------------------------------------------


def _phase_c(dof_defs: list, n_scalar: int = 1_000, n_batch: int = 10_000) -> None:
    """Time the configured geometry DoF on synthetic coordinates."""
    import numpy as np

    from confana.coordinates import batch_extract_geometry_dof, extract_geometry_dof
    from confana.models import FrameRecord

    _section("Phase C — geometry micro-benchmark")

    geometry = [d for d in dof_defs if d.type in ("dihedral", "distance", "angle")]
    if not geometry:
        print("  SKIP: no dihedral, distance or angle DoF in the config.")
        return

    n_atoms = max(max(d.atoms) for d in geometry) + 1
    rng = np.random.default_rng(42)
    coords = rng.standard_normal((n_atoms, 3)).astype(np.float32)
    batch_coords = rng.standard_normal((n_batch, n_atoms, 3)).astype(np.float32)
    frame = FrameRecord(
        source_file="bench",
        frame_number=0,
        byte_offset=0,
        atom_count=n_atoms,
        comment_line="",
        elements=["C"] * n_atoms,
        coords=coords,
    )

    def per_frame_us(defs: list) -> float:
        t0 = time.perf_counter()
        for _ in range(n_scalar):
            extract_geometry_dof(frame, defs)
        return (time.perf_counter() - t0) / n_scalar * 1e6

    def batch_us(defs: list) -> float:
        t0 = time.perf_counter()
        batch_extract_geometry_dof(batch_coords, defs)
        return (time.perf_counter() - t0) / n_batch * 1e6

    rows = [(f"{d.name} ({d.type})", per_frame_us([d]), batch_us([d])) for d in geometry]
    rows.append((f"all {len(geometry)} DoF", per_frame_us(geometry), batch_us(geometry)))

    lw = max(len(r[0]) for r in rows) + 2
    print(f"  Synthetic {n_atoms}-atom frames. One frame: {n_scalar:,} calls. "
          f"Batch: 1 call × {n_batch:,} frames.")
    print()
    print(f"  {'DoF':<{lw}}  {'one frame':>11}  {'batch':>11}  {'speedup':>8}")
    print(f"  {'':<{lw}}  {'µs/frame':>11}  {'µs/frame':>11}")
    print(f"  {'-' * lw}  {'-' * 11}  {'-' * 11}  {'-' * 8}")
    for name, one, batch in rows:
        speedup = f"{_div(one, batch):.0f}×" if batch > 0 else "∞"
        print(f"  {name:<{lw}}  {one:>11.3f}  {batch:>11.3f}  {speedup:>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark the ConfAna coordinate-table build stage."
    )
    parser.add_argument(
        "--config",
        default="examples/md17_aspirin.yaml",
        help="Path to config YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="NPZ path for the benchmark's own cache (default: <run_dir>/.bench/coordinates.npz)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for the build; 1 (default) gives the stage breakdown.",
    )
    parser.add_argument("--no-cold", action="store_true", help="Skip Phase A (cold build).")
    parser.add_argument("--no-warm", action="store_true", help="Skip Phase B (warm load).")
    parser.add_argument("--micro", action="store_true", help="Run Phase C (geometry micro-benchmark).")
    args = parser.parse_args(argv)

    import yaml

    from confana.coordinate_config import resolve_dof_definitions

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())

    cache_path = args.cache
    if cache_path is None:
        cache_path = Path(cfg.get("run_dir") or "outputs") / ".bench" / "coordinates.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    _header("ConfAna coordinate-table benchmark")
    print(f"  config: {cfg_path}")
    print(f"  cache : {cache_path}")

    if not args.no_cold or not args.no_warm:
        kwargs = _build_kwargs(cfg, cache_path, args.n_jobs)
        if not args.no_cold:
            _phase_a(kwargs)
        if not args.no_warm:
            _phase_b(kwargs)

    if args.micro:
        _phase_c(resolve_dof_definitions(cfg))

    print()
    print(_sep("=", 64))
    print("  Benchmark complete.")
    print(_sep("=", 64))


if __name__ == "__main__":
    main()
