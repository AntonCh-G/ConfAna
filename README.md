# ConfAna

ConfAna is a Python workflow for conformer analysis from ordered multi-frame XYZ trajectories and precomputed coordinate tables. It computes plane-based and dihedral-based conformational coordinates, builds density maps, assigns conformational states, analyzes transitions, and produces both static PNG figures and standalone interactive HTML outputs.

The project is designed around a coordinate-table boundary so the downstream workflow can operate on either:

- full-structure XYZ trajectories
- coordinate-only tables with precomputed carboxyl and ester values

## What The Workflow Does

For each frame, ConfAna computes two coordinate systems:

- Plane-to-plane definition
  - carboxyl = angle between the benzene best-fit plane and the carboxyl plane
  - ester = angle between the benzene best-fit plane and the ester plane
- Dihedral definition
  - carboxyl dihedral
  - ester dihedral

Then, for both definitions, the workflow can:

- build 2D density plots
- assign states with clustering
- compute transition counts
- compute transition probabilities
- compute physical rates only when `dt` is explicitly provided in config
- estimate activation free-energy barriers only when both `dt` and `temperature` are explicitly provided in config

## Scientific Scope And Guardrails

- Treat outputs as coordinate-density landscapes unless energies are explicitly available.
- Do not interpret the result as a true PES just from XYZ structures alone.
- Do not report physical transition rates unless frame ordering and time spacing are known.
- Treat activation barriers as rate-derived free-energy estimates, not true potential-energy barriers from a PES.
- PIMD trajectories preserve bead identity during transition analysis and can be averaged across beads afterward.
- Current datasets may share one atom ordering, but the code is intended to keep parsing and downstream analysis decoupled for future datasets.

## Current Atom Mapping

All indices below are 0-based Python indices.

- Ring plane: `[0, 1, 2, 3, 5, 6]`
- Carboxyl plane: `[9, 10, 7]`
- Ester plane: `[12, 11, 8]`
- Carboxyl dihedral: `[6, 5, 10, 7]`
- Ester dihedral: `[5, 6, 12, 11]`

`plane(C1-C6)` is implemented as a best-fit plane through all six ring atoms.

## Repository Layout

- `src/geometry.py`: reusable geometry primitives
- `src/io_xyz.py`: streaming XYZ ingestion and indexed random access
- `src/io_coordinates.py`: coordinate-table ingestion and cache-aware loading
- `src/coordinates.py`: plane/dihedral coordinate extraction
- `src/states.py`: conformational state assignment
- `src/transitions.py`: transition counts, probabilities, optional rates, and optional activation barriers
- `src/plots_static.py`: PNG density and transition plots
- `src/plots_interactive.py`: standalone interactive HTML outputs
- `src/viewer.py`: nearest-structure lookup and structure rendering helpers
- `src/cli.py`: command-line entrypoints
- `configs/`: local workflow configurations ignored by git

## Installation

ConfAna targets Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Configuration

The workflow is driven by YAML config. Create or update a local config under `configs/`, for example [configs/default.yaml](configs/default.yaml).

Important sections include:

- `data`: input path pattern plus trajectory and bead ID extraction rules
- `atom_mapping`: named atom groups for plane-based coordinates
- `dihedrals`: named dihedral definitions
- `coordinate_pairs`: which coordinate pairs get plotted
- `conventions`: signed/unsigned angle settings
- `cache`: frame-index and coordinate-cache settings
- `density`: histogram bins, ranges, and coloring
- `clustering`: state-assignment settings
- `transitions`: lag, optional `dt`, and optional activation-barrier settings
- `interactive`: interactive HTML behavior
- `outputs`: output directory and file settings

Local configs are intentionally ignored by git because they often contain machine-specific data and output paths.

## CLI Usage

After installation, the CLI is available as `confana`.

```bash
confana extract-coordinates --config configs/default.yaml
confana plot-densities --config configs/default.yaml
confana cluster-states --config configs/default.yaml
confana compute-transitions --config configs/default.yaml
confana build-interactive --config configs/default.yaml
confana run-all --config configs/default.yaml
```

## Outputs

Depending on config, a run can produce:

- coordinate table as CSV and/or cached binary data
- density PNGs for configured coordinate pairs
- transition PNGs for plane and dihedral state models
- standalone interactive HTML density plots
- embedded structure metadata for nearest-frame lookup
- transition count/probability/rate/barrier matrices in per-trajectory caches when applicable

Typical output filenames include:

- `coordinates_angles.csv`
- `density_plane.png`
- `density_dihedral.png`
- `density_<pair_name>.png`
- `density_<pair_name>.html`
- `transition_plane.png`
- `transition_dihedral.png`

## Dihedral Configuration

Dihedrals are configured as a list of named definitions instead of hard-coded keys under `atom_mapping`.

```yaml
atom_mapping:
  ring_plane: [0, 1, 2, 3, 5, 6]
  carboxyl_plane: [9, 10, 7]
  ester_plane: [12, 11, 8]

dihedrals:
  - name: carboxyl_dihedral
    atoms: [6, 5, 10, 7]
    label: Carboxyl dihedral
    convention: signed
    group: core
    enabled: true
  - name: ester_dihedral
    atoms: [5, 6, 12, 11]
    label: Ester dihedral
    convention: signed
    group: core
    enabled: true
  - name: igor1_dihedral
    atoms: [6, 12, 11, 8]
    label: Igor 1 dihedral
    convention: signed
    group: igor
    enabled: true

coordinate_pairs:
  dihedral:
    x: carboxyl_dihedral
    y: ester_dihedral
    x_label: Carboxyl dihedral (degrees)
    y_label: Ester dihedral (degrees)
    title: Dihedral-angle density
  dihedrals_igor:
    x: igor1_dihedral
    y: igor2_dihedral
    x_label: Igor 1 dihedral (degrees)
    y_label: Igor 2 dihedral (degrees)
    title: Igor dihedral density
```

To add a new dihedral:

1. Add a new item under `dihedrals` with a unique `name` and four 0-based atom indices.
2. Set `enabled: true`.
3. Update `coordinate_pairs` if you want density and interactive plots for that pair.

Extra named entries under `coordinate_pairs` generate additional density PNG and HTML outputs such as `density_dihedrals_igor.png` and `density_dihedrals_igor.html`.

## Transition Rates And Barriers

Physical rates are computed only when `transitions.dt` is set. Activation free-energy barriers are computed only when both `transitions.dt` and `transitions.temperature` are set.

The default barrier model is Eyring transition-state theory:

```yaml
transitions:
  dt: 1e-14
  temperature: 300.0
  barrier_model: eyring
  transmission_coefficient: 1.0
  attempt_frequency: null
  energy_conv_factor: 1.0
  energy_unit: eV
```

Eyring uses `transmission_coefficient * k_B*T/h` as the prefactor. Arrhenius mode is also supported with `barrier_model: arrhenius`, but it requires an explicit `attempt_frequency` in `s^-1`.

Barriers are computed internally in eV and reported as:

```text
reported_barrier = barrier_eV * energy_conv_factor
```

For `kJ/mol`, use `energy_conv_factor: 96.48533212` and `energy_unit: kJ/mol`. Self-transitions and zero/missing-rate off-diagonal entries are reported as `NaN`. Negative barriers are not clamped; they indicate that the configured kinetic model or prefactor is inconsistent with the observed discrete-time rate estimate.

## Backwards Compatibility

Legacy configs that still define `carboxyl_dihedral`, `ester_dihedral`, or other `*_dihedral` keys directly under `atom_mapping` continue to work. The list-based `dihedrals:` format is the recommended configuration style.

## Testing

Run the test suite with:

```bash
pytest
```

The repository includes tests for geometry, XYZ ingestion, coordinate-table loading, clustering, transitions, caching, viewer helpers, and interactive plot generation.

## Notes

- Interactive structure rendering is intended to use browser-native HTML output.
- Standalone HTML can use Plotly from CDN or inline it, depending on config.
- XYZ ingestion is designed around streaming parsing and cached byte-offset frame indices for fast random access.
