"""Command-line interface for the ConfAna workflow.

Provides six commands that correspond to the major analysis phases.  Each
command loads the project configuration from a YAML file (default:
``configs/default.yaml``) and delegates to the appropriate public API
functions from the other ``src`` modules.

Commands
--------
extract-coordinates
    Load or build the coordinate table (Phases 0–6).
plot-densities
    Generate 2D density PNG plots (Phase 7).
cluster-states
    Assign conformational states with DBSCAN (Phase 8).
compute-transitions
    Compute transition counts/probabilities and save heatmap PNGs (Phases 9–10).
build-interactive
    Build standalone interactive HTML density plots (Phases 11–12).
run-all
    Run the full pipeline sequentially (Phases 0–12).

Usage example
-------------
    confana run-all
    confana run-all --config configs/my_config.yaml
    confana extract-coordinates --config configs/default.yaml
    confana build-interactive
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_cfg(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _out_dir(cfg: dict) -> Path:
    out = Path(cfg["outputs"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def _coordinate_pairs(cfg: dict):
    """Return list of (name, CoordinatePair) from config."""
    from src.coordinate_config import list_coordinate_pairs  # noqa: PLC0415

    return list_coordinate_pairs(cfg)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

_CONFIG_OPTION = click.option(
    "--config",
    default="configs/default.yaml",
    show_default=True,
    help="Path to the YAML configuration file.",
)


@click.group()
def cli() -> None:
    """ConfAna: conformer analysis workflow for PIMD/MD trajectories."""


# ---------------------------------------------------------------------------
# extract-coordinates
# ---------------------------------------------------------------------------


@cli.command("extract-coordinates")
@_CONFIG_OPTION
def extract_coordinates(config: str) -> None:
    """Load or build the coordinate table (Phases 0–6).

    Reads xyz trajectories (or a cached NPZ), computes DoF angles / distances
    for every frame, and optionally saves a CSV summary to the output directory.
    """
    from src.io_coordinates import load_or_build_coordinate_table_from_config  # noqa: PLC0415

    cfg = _load_cfg(config)
    out = _out_dir(cfg)

    click.echo("Loading / building coordinate table …")
    df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
    click.echo(f"  frames={len(df):,}  cache_hit={cache_hit}")

    if cfg.get("outputs", {}).get("save_csv", False):
        csv_path = out / "coordinates_angles.csv"
        df.to_csv(csv_path, index=False)
        click.echo(f"  Saved {csv_path}")

    click.echo("extract-coordinates done.")


# ---------------------------------------------------------------------------
# plot-densities
# ---------------------------------------------------------------------------


@cli.command("plot-densities")
@_CONFIG_OPTION
def plot_densities(config: str) -> None:
    """Generate 2D density PNG plots for all configured coordinate pairs (Phase 7)."""
    from src.io_coordinates import load_or_build_coordinate_table_from_config  # noqa: PLC0415
    from src.plots_static import make_density_png  # noqa: PLC0415

    cfg = _load_cfg(config)
    out = _out_dir(cfg)

    click.echo("Loading coordinate table …")
    df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
    click.echo(f"  frames={len(df):,}  cache_hit={cache_hit}")

    click.echo("Plotting density PNGs …")
    written: list[str] = []
    for pair_name, pair in _coordinate_pairs(cfg):
        filename = f"density_{pair_name}.png"
        make_density_png(df, pair, out / filename, dpi=cfg["outputs"]["dpi"], config=cfg)
        written.append(filename)
    click.echo(f"  Saved {', '.join(written)} → {out}")
    click.echo("plot-densities done.")


# ---------------------------------------------------------------------------
# cluster-states
# ---------------------------------------------------------------------------


@cli.command("cluster-states")
@_CONFIG_OPTION
def cluster_states(config: str) -> None:
    """Assign conformational state labels with DBSCAN (Phase 8)."""
    from src.io_coordinates import load_or_build_coordinate_table_from_config  # noqa: PLC0415
    from src.states import assign_conformer_states_from_config  # noqa: PLC0415

    cfg = _load_cfg(config)

    click.echo("Loading coordinate table …")
    df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
    click.echo(f"  frames={len(df):,}  cache_hit={cache_hit}")

    click.echo("Assigning conformational states …")
    df = assign_conformer_states_from_config(df, cfg)

    for pair_name, pair in _coordinate_pairs(cfg):
        state_col = pair.state_col
        if state_col in df.columns:
            n_states = df[state_col].nunique()
            click.echo(f"  {state_col} unique={n_states}")

    click.echo("cluster-states done.")


# ---------------------------------------------------------------------------
# compute-transitions
# ---------------------------------------------------------------------------


@cli.command("compute-transitions")
@_CONFIG_OPTION
def compute_transitions(config: str) -> None:
    """Compute transition counts/probabilities and save heatmap PNGs (Phases 9–10)."""
    from src.io_coordinates import load_or_build_coordinate_table_from_config  # noqa: PLC0415
    from src.plots_static import make_transition_png  # noqa: PLC0415
    from src.states import assign_conformer_states_from_config  # noqa: PLC0415
    from src.transitions import analyze_grouped_transitions  # noqa: PLC0415

    cfg = _load_cfg(config)
    out = _out_dir(cfg)
    transitions_cfg = cfg.get("transitions", {})
    lag: int = transitions_cfg.get("lag", 1)
    dt = transitions_cfg.get("dt")

    click.echo("Loading coordinate table …")
    df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
    click.echo(f"  frames={len(df):,}  cache_hit={cache_hit}")

    click.echo("Assigning states …")
    df = assign_conformer_states_from_config(df, cfg)

    click.echo("Analyzing transitions …")
    written: list[str] = []
    for pair_name, pair in _coordinate_pairs(cfg):
        result = analyze_grouped_transitions(
            df,
            pair,
            lag=lag,
            dt=dt,
            skip_noise_intermediates=transitions_cfg.get(
                "skip_noise_intermediates", True
            ),
            temperature=transitions_cfg.get("temperature"),
            barrier_model=transitions_cfg.get("barrier_model", "eyring"),
            transmission_coefficient=transitions_cfg.get(
                "transmission_coefficient", 1.0
            ),
            attempt_frequency=transitions_cfg.get("attempt_frequency"),
            energy_conv_factor=transitions_cfg.get("energy_conv_factor", 1.0),
            energy_unit=transitions_cfg.get("energy_unit", "eV"),
        )
        n_groups = len(result.get("per_group_counts", {}))
        click.echo(f"  {pair_name}: groups={n_groups}")
        filename = f"transition_{pair_name}.png"
        make_transition_png(result, out / filename, dpi=cfg["outputs"]["dpi"], config=cfg)
        written.append(filename)
    click.echo(f"  Saved {', '.join(written)} → {out}")
    click.echo("compute-transitions done.")


# ---------------------------------------------------------------------------
# build-interactive
# ---------------------------------------------------------------------------


@cli.command("build-interactive")
@_CONFIG_OPTION
def build_interactive(config: str) -> None:
    """Build standalone interactive HTML density plots (Phases 11–12)."""
    from src.io_coordinates import load_or_build_coordinate_table_from_config  # noqa: PLC0415
    from src.plots_interactive import make_density_interactive  # noqa: PLC0415
    from src.states import assign_conformer_states_from_config  # noqa: PLC0415

    cfg = _load_cfg(config)
    out = _out_dir(cfg)

    click.echo("Loading coordinate table …")
    df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
    click.echo(f"  frames={len(df):,}  cache_hit={cache_hit}")

    click.echo("Assigning states …")
    df = assign_conformer_states_from_config(df, cfg)

    click.echo("Building interactive HTML …")
    written: list[str] = []
    for pair_name, pair in _coordinate_pairs(cfg):
        filename = f"density_{pair_name}.html"
        make_density_interactive(df, pair, out / filename, config=cfg)
        written.append(filename)
    click.echo(f"  Saved {', '.join(written)} → {out}")
    click.echo("build-interactive done.")


# ---------------------------------------------------------------------------
# run-all
# ---------------------------------------------------------------------------


@cli.command("run-all")
@_CONFIG_OPTION
@click.option("--skip-transitions", is_flag=True, default=False, help="Skip the compute-transitions phase.")
def run_all(config: str, skip_transitions: bool) -> None:
    """Run the full analysis pipeline sequentially (Phases 0–12)."""
    from src.io_coordinates import load_or_build_coordinate_table_from_config  # noqa: PLC0415
    from src.plots_interactive import make_density_interactive  # noqa: PLC0415
    from src.plots_static import make_density_png, make_transition_png  # noqa: PLC0415
    from src.states import assign_conformer_states_from_config  # noqa: PLC0415

    cfg = _load_cfg(config)
    out = _out_dir(cfg)
    transitions_cfg = cfg.get("transitions", {})
    lag: int = transitions_cfg.get("lag", 1)
    dt = transitions_cfg.get("dt")
    pairs = _coordinate_pairs(cfg)

    # Phases 0–6 — coordinate table
    click.echo("[1/4] Loading / building coordinate table …")
    df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
    click.echo(f"      frames={len(df):,}  cache_hit={cache_hit}")

    # Phase 7 — density PNGs (without states)
    click.echo("[2/4] Plotting density PNGs …")
    for pair_name, pair in pairs:
        make_density_png(
            df,
            pair,
            out / f"density_{pair_name}.png",
            dpi=cfg["outputs"]["dpi"],
            config=cfg,
        )

    # Phase 8 — state assignment
    click.echo("[3/4] Assigning conformational states …")
    df = assign_conformer_states_from_config(df, cfg)

    # Phase 8b — density PNGs with state COM markers
    for pair_name, pair in pairs:
        make_density_png(
            df,
            pair,
            out / f"density_{pair_name}_states.png",
            dpi=cfg["outputs"]["dpi"],
            config=cfg,
        )

    # Phases 9–10 — transitions
    if not skip_transitions:
        from src.transitions import analyze_grouped_transitions  # noqa: PLC0415

        click.echo("[+] Computing transitions …")
        for pair_name, pair in pairs:
            result = analyze_grouped_transitions(
                df,
                pair,
                lag=lag,
                dt=dt,
                skip_noise_intermediates=transitions_cfg.get(
                    "skip_noise_intermediates", True
                ),
                temperature=transitions_cfg.get("temperature"),
                barrier_model=transitions_cfg.get("barrier_model", "eyring"),
                transmission_coefficient=transitions_cfg.get(
                    "transmission_coefficient", 1.0
                ),
                attempt_frequency=transitions_cfg.get("attempt_frequency"),
                energy_conv_factor=transitions_cfg.get("energy_conv_factor", 1.0),
                energy_unit=transitions_cfg.get("energy_unit", "eV"),
            )
            make_transition_png(
                result,
                out / f"transition_{pair_name}.png",
                dpi=cfg["outputs"]["dpi"],
                config=cfg,
            )

    # Phases 11–12 — interactive HTML
    click.echo("[4/4] Building interactive HTML …")
    for pair_name, pair in pairs:
        make_density_interactive(
            df,
            pair,
            out / f"density_{pair_name}.html",
            config=cfg,
        )

    click.echo(f"\nAll outputs saved to: {out}")
    click.echo("run-all done.")
