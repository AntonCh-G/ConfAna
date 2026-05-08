"""Benchmark the load/build coordinate-table stage.

Run from the project root:
    python scripts/benchmark_coordinates.py
    python scripts/benchmark_coordinates.py --config configs/default.yaml
    python scripts/benchmark_coordinates.py --no-cold --micro
    python scripts/benchmark_coordinates.py --help

Phases
------
A  Cold build  — forces cache rebuild; times all substeps via StageTimer.
B  Warm load   — reloads from NPZ cache; measures I/O-only throughput.
C  Micro       — times individual geometry primitives on a synthetic fixture.
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


# ---------------------------------------------------------------------------
# Phase A: cold build
# ---------------------------------------------------------------------------


def _phase_a(cfg: dict, config_path: str) -> int:
    """Force-rebuild the coordinate table and collect fine-grained timing."""
    import yaml
    from src.bench import StageTimer
    from src.io_coordinates import load_or_build_coordinate_table_cache
    from src.io_xyz import load_or_build_xyz_index

    _section("Phase A — cold build (force_rebuild=True)")

    data_cfg = cfg.get("data", {})
    cache_cfg = cfg.get("cache", {})
    cache_path = cache_cfg.get("coordinate_table_path")
    if not cache_path:
        print("  ERROR: cache.coordinate_table_path not set in config.")
        return 0

    n_jobs = int(cache_cfg.get("n_jobs", 1))
    # Stage-level timer only works in serial mode; suppress it in parallel runs.
    timer = StageTimer() if n_jobs == 1 else None

    print(f"  n_jobs       : {n_jobs}  {'(serial — stage timer active)' if n_jobs == 1 else '(parallel — no stage breakdown)'}")

    t_total_start = time.perf_counter()
    df, cache_hit = load_or_build_coordinate_table_cache(
        path_pattern=str(data_cfg["path_pattern"]),
        mapping=cfg["atom_mapping"],
        conventions=cfg["conventions"],
        cache_path=cache_path,
        trajectory_id_pattern=data_cfg.get("trajectory_id_pattern"),
        bead_id_pattern=data_cfg.get("bead_id_pattern"),
        index_cache_dir=cache_cfg.get("index_cache_dir"),
        force_rebuild=True,
        n_jobs=n_jobs,
        _timer=timer,
    )
    t_total = time.perf_counter() - t_total_start

    n = len(df)
    fps = _div(n, t_total)

    print(f"  total_frames : {n:>12,}")
    print(f"  cache_hit    : {cache_hit}")
    print(f"  total_time   : {t_total:>10.2f} s")
    print(f"  throughput   : {fps:>10,.0f} frames/s  ({_fmt_fps(fps)})")

    if timer is not None:
        print()
        print("  Stage breakdown:")
        print(timer.report(total_frames=n))

        # Per-file index timing
        per_file_labels = [k for k in timer._totals if k.startswith("index:")]
        if per_file_labels:
            print()
            print(f"  {'File':<40}  {'index_s':>8}")
            print(f"  {'-'*40}  {'-'*8}")
            for lbl in sorted(per_file_labels):
                fname = lbl[len("index:"):]
                print(f"  {fname:<40}  {timer.total(lbl):>8.4f}")

        # Bottleneck analysis
        _print_bottleneck_analysis(timer, n)

    return n


def _print_bottleneck_analysis(timer, total_frames: int) -> None:
    from src.bench import StageTimer

    grand = timer.grand_total() or 1.0
    ranked = sorted(
        [(lbl, t) for lbl, t in timer._totals.items() if not lbl.startswith("index:")],
        key=lambda kv: -kv[1],
    )
    if not ranked:
        return

    print()
    print("  TOP BOTTLENECKS:")
    explanations = {
        "frame_iter": "xyz line parsing in Python (I/O + coordinate parsing)",
        "compute_plane": "best_fit_plane SVD called 2x per frame (ring + func. group)",
        "compute_dihedral": "atan2 / cross-product geometry, 2x per frame",
        "array_write": "scalar numpy indexing overhead",
        "df_assembly": "pd.DataFrame construction + dtype coercion",
        "array_alloc": "np.empty pre-allocation",
        "index_loading": "frame-index cache check + JSON load (or scan)",
    }
    for rank, (lbl, t) in enumerate(ranked, 1):
        pct = t / grand * 100.0
        us_per_frame = _div(t, total_frames) * 1e6
        note = explanations.get(lbl, "")
        print(f"  {rank}. {lbl:<20}  {pct:5.1f}%  {us_per_frame:7.1f} µs/frame  — {note}")

    top_label = ranked[0][0] if ranked else ""
    print()
    print("  RECOMMENDED NEXT STEPS (based on measurement):")
    if top_label == "compute_plane":
        print("  > carboxyl and ester planes have 3 atoms each — replace SVD")
        print("    (best_fit_plane) with a direct cross-product for those planes.")
        print("    Only the 6-atom ring plane needs SVD. Expected speedup: 40-60%.")
    elif top_label == "frame_iter":
        print("  > xyz parsing is the bottleneck. Consider:")
        print("    - Parsing coordinate lines with numpy.fromstring instead of split()")
        print("    - Writing a C extension or using ASE's faster reader")
        print("    - Pre-computing and caching coordinates as binary (NPZ/HDF5)")
    elif top_label == "compute_dihedral":
        print("  > Vectorise dihedral computation across all frames using numpy")
        print("    batch operations instead of calling once per frame.")
    elif top_label == "df_assembly":
        print("  > DataFrame assembly is slow. Consider building the DataFrame")
        print("    after all frames are processed rather than using pd.Categorical.")
    else:
        print(f"  > Profile {top_label!r} further with --micro or cProfile.")


# ---------------------------------------------------------------------------
# Phase B: warm load
# ---------------------------------------------------------------------------


def _phase_b(cfg: dict) -> None:
    """Load the coordinate table from the NPZ cache and measure throughput."""
    _section("Phase B — warm load (cache hit)")

    cache_cfg = cfg.get("cache", {})
    cache_path = cache_cfg.get("coordinate_table_path")
    if not cache_path or not Path(cache_path).exists():
        print("  SKIP: cache not found — run Phase A first.")
        return

    from src.io_coordinates import load_coordinate_table

    t0 = time.perf_counter()
    df = load_coordinate_table(cache_path)
    t_total = time.perf_counter() - t0

    n = len(df)
    fps = _div(n, t_total)
    size_mb = Path(cache_path).stat().st_size / 1e6

    print(f"  total_frames : {n:>12,}")
    print(f"  total_time   : {t_total:>10.3f} s")
    print(f"  throughput   : {fps:>10,.0f} frames/s  ({_fmt_fps(fps)})")
    print(f"  cache_size   : {size_mb:>10.1f} MB")
    print(f"  read_MB/s    : {_div(size_mb, t_total):>10.1f} MB/s")


# ---------------------------------------------------------------------------
# Phase C: geometry micro-benchmark
# ---------------------------------------------------------------------------


def _phase_c(cfg: dict) -> None:
    """Time individual geometry primitives on a synthetic FrameRecord fixture."""
    _section("Phase C — geometry micro-benchmark")

    import numpy as np
    from src.geometry import best_fit_plane, dihedral_angle, plane_plane_angle
    from src.coordinates import compute_angles_plane, compute_angles_dihedral
    from src.models import FrameRecord

    mapping = cfg.get("atom_mapping", {})
    conventions = cfg.get("conventions", {})

    rng = np.random.default_rng(42)
    # Typical molecule has ~15 atoms; use 15 atoms
    n_atoms = 15
    coords = rng.standard_normal((n_atoms, 3)).astype(np.float32)

    # Build a minimal FrameRecord
    frame = FrameRecord(
        source_file="bench",
        frame_number=0,
        byte_offset=0,
        atom_count=n_atoms,
        comment_line="",
        elements=["C"] * n_atoms,
        coords=coords,
        trajectory_id="bench",
        bead_id=None,
        local_frame_index=0,
        global_frame_index=0,
        energy=None,
        step_number=None,
        bead_comment=None,
    )

    ring_ids = mapping.get("ring_plane", [0, 1, 2, 3, 5, 6])
    carboxyl_ids = mapping.get("carboxyl_plane", [9, 10, 7])
    ester_ids = mapping.get("ester_plane", [12, 11, 8])
    carboxyl_dih_ids = mapping.get("carboxyl_dihedral", [6, 5, 10, 7])
    ester_dih_ids = mapping.get("ester_dihedral", [5, 6, 12, 11])

    # Clamp indices to available atoms
    def clamp(ids: list[int]) -> list[int]:
        return [i % n_atoms for i in ids]

    ring_ids = clamp(ring_ids)
    carboxyl_ids = clamp(carboxyl_ids)
    ester_ids = clamp(ester_ids)
    carboxyl_dih_ids = clamp(carboxyl_dih_ids)
    ester_dih_ids = clamp(ester_dih_ids)

    from src.coordinates import (
        batch_compute_angles_dihedral,
        batch_compute_angles_plane,
    )
    from src.geometry import batch_best_fit_plane, batch_dihedral_angle, batch_plane_plane_angle

    N_SCALAR = 1_000   # scalar loop iterations
    N_BATCH  = 10_000  # frames per batch call

    def _bench_scalar(fn, *args) -> float:
        t0 = time.perf_counter()
        for _ in range(N_SCALAR):
            fn(*args)
        return (time.perf_counter() - t0) / N_SCALAR * 1e6

    # Batch fixture: N_BATCH frames of n_atoms atoms
    batch_coords = rng.standard_normal((N_BATCH, n_atoms, 3)).astype(np.float32)

    def _bench_batch(fn, *args) -> float:
        """Return µs/frame for a batch call over N_BATCH frames."""
        t0 = time.perf_counter()
        fn(*args)
        return (time.perf_counter() - t0) / N_BATCH * 1e6

    # ---- Scalar primitives ----
    scalar_results: list[tuple[str, float]] = []
    scalar_results.append(("scalar: best_fit_plane (ring 6)", _bench_scalar(best_fit_plane, coords, ring_ids)))
    scalar_results.append(("scalar: best_fit_plane (carboxyl 3)", _bench_scalar(best_fit_plane, coords, carboxyl_ids)))
    scalar_results.append(("scalar: plane_plane_angle", _bench_scalar(plane_plane_angle, coords, ring_ids, carboxyl_ids)))
    scalar_results.append(("scalar: dihedral_angle", _bench_scalar(dihedral_angle, coords, carboxyl_dih_ids)))
    scalar_results.append(("scalar: compute_angles_plane", _bench_scalar(compute_angles_plane, frame, mapping, conventions)))
    scalar_results.append(("scalar: compute_angles_dihedral", _bench_scalar(compute_angles_dihedral, frame, mapping, conventions)))

    # ---- Batch primitives ----
    batch_results: list[tuple[str, float]] = []
    batch_results.append(("batch:  best_fit_plane (ring 6)", _bench_batch(batch_best_fit_plane, batch_coords, ring_ids)))
    batch_results.append(("batch:  best_fit_plane (carboxyl 3)", _bench_batch(batch_best_fit_plane, batch_coords, carboxyl_ids)))
    batch_results.append(("batch:  plane_plane_angle", _bench_batch(batch_plane_plane_angle, batch_coords, ring_ids, carboxyl_ids)))
    batch_results.append(("batch:  dihedral_angle", _bench_batch(batch_dihedral_angle, batch_coords, carboxyl_dih_ids)))
    batch_results.append(("batch:  compute_angles_plane", _bench_batch(batch_compute_angles_plane, batch_coords, mapping, conventions)))
    batch_results.append(("batch:  compute_angles_dihedral", _bench_batch(batch_compute_angles_dihedral, batch_coords, mapping, conventions)))

    all_results = scalar_results + batch_results
    lw = max(len(r[0]) for r in all_results) + 2

    print(f"  Scalar: {N_SCALAR:,} iterations × 1 frame.  Batch: 1 call × {N_BATCH:,} frames.")
    print(f"  All times in µs/frame.")
    print()
    print(f"  {'function':<{lw}}  {'µs/frame':>10}")
    print(f"  {'-'*lw}  {'-'*10}")
    for name, us in sorted(all_results, key=lambda r: -r[1]):
        print(f"  {name:<{lw}}  {us:>10.3f}")

    # Speedup summary
    print()
    print("  Speedup (scalar ÷ batch):")
    pairs = [
        ("compute_angles_plane", "scalar: compute_angles_plane", "batch:  compute_angles_plane"),
        ("compute_angles_dihedral", "scalar: compute_angles_dihedral", "batch:  compute_angles_dihedral"),
        ("best_fit_plane (ring 6)", "scalar: best_fit_plane (ring 6)", "batch:  best_fit_plane (ring 6)"),
        ("dihedral_angle", "scalar: dihedral_angle", "batch:  dihedral_angle"),
    ]
    scalar_map = dict(scalar_results)
    batch_map  = dict(batch_results)
    for label, sk, bk in pairs:
        sv = scalar_map.get(sk, 0.0)
        bv = batch_map.get(bk, 0.0)
        speedup = _div(sv, bv) if bv > 0 else float("inf")
        print(f"  {label:<35}  {speedup:>6.1f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the ConfAna coordinate-table build stage."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--no-cold",
        action="store_true",
        help="Skip Phase A (cold build).",
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="Skip Phase B (warm load).",
    )
    parser.add_argument(
        "--micro",
        action="store_true",
        help="Run Phase C (geometry micro-benchmark).",
    )
    args = parser.parse_args()

    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())

    _header("ConfAna coordinate-table benchmark")
    print(f"  config: {cfg_path}")

    total_frames = 0
    if not args.no_cold:
        total_frames = _phase_a(cfg, str(cfg_path))

    if not args.no_warm:
        _phase_b(cfg)

    if args.micro:
        _phase_c(cfg)

    print()
    print(_sep("=", 64))
    print("  Benchmark complete.")
    print(_sep("=", 64))


if __name__ == "__main__":
    main()
