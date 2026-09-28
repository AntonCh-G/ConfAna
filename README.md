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
- `src/payload_codec.py`: compact encoding of the structures and metadata embedded in the HTML
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
- `interactive`: interactive HTML behavior (Plotly/3Dmol inlining, theme, alignment, XYZ payload embedding)
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

`tests/test_interactive_browser.py` opens generated HTML pages in headless Chromium,
offline, and drives them with real mouse and keyboard input (hover preview, pinning,
double-click, Esc, theme toggle). It needs a one-off browser install:

```bash
python -m playwright install chromium --no-shell
```

Without it these tests are skipped. To skip them explicitly: `pytest -m "not browser"`.

## Notes

- Interactive structure rendering is intended to use browser-native HTML output.
- Interactive HTML is fully offline by default: Plotly and 3Dmol.js are both inlined into
  the single output file (`include_plotlyjs: true`, `include_3dmol: inline`). Set either
  to `cdn` to load from a CDN instead (requires internet, produces a smaller file).
- Theme follows the browser's OS-level light/dark preference by default
  (`interactive.theme: auto`); set to `light` or `dark` to force it, or use the in-page toggle.
- The map's colour scale suits each theme. Its direction never changes (bright means
  the same in both), but an end that would blend into the background is trimmed: on
  white, a near-white end (contrast below 1.25:1); on dark, an end below 2:1 contrast
  with the dark background. Standard viridis on white is unchanged; on dark its darkest
  purples are dropped. Set `interactive.theme_colorscales: {light: <name>, dark: <name>}`
  to use given Plotly scales (e.g. `plasma`, `viridis_r`) unchanged instead.
- Axes measured in degrees (dihedral or angle DoFs; for pairs without a DoF type, axes
  whose label contains `(°)`) get ticks on multiples of 1, 2, 5, 10, 15, 30, 45, 60 or 90°,
  whichever gives at most six intervals, e.g. 60° steps across −180…180° and 30° across
  0…180°. Zooming or panning re-picks the step for the visible range.
- The header links to the pages of the run's other coordinate pairs, so you can jump
  between e.g. `density_dihedral.html` and `density_plane.html`. The links are plain
  file names, so the pages must stay in one folder; keep them together when copying or
  emailing them. With a single coordinate pair no links are shown.
- The embedded structures and metadata are compressed by default
  (`interactive.compress_payloads: true`), which is most of the file: on a large run the page shrinks to about a third of its size. Element
  symbols are stored once per page, coordinates as 16-bit integers in steps of
  `interactive.coordinate_step` (default 0.001 Å, so at most 0.0005 Å of rounding — a
  display copy; exact coordinates stay in the trajectory files), metadata column by column
  with repeated strings stored once, and both blocks gzipped. The page unpacks them once
  on load with the browser's own `DecompressionStream`, needing no library and no internet;
  a browser without it says so in the side panel. Set `compress_payloads: false` to embed
  plain JSON for debugging. The build fails loudly if the structures do not share one
  element sequence, or if a coordinate needs more than 16 bits at the chosen step.
- The vendored 3Dmol.js copy and its BSD license live under `src/interactive_assets/vendor/`.
- Hovering the map previews the bin under the cursor in the side panel
  (`interactive.hover_preview`, default `true`). Clicking pins it as a card; Esc or
  double-click clears pins. At most `interactive.max_pinned` cards (default 15) are kept:
  pinning more removes the oldest.
- Every 3D view colours the atoms that define each axis: x-axis atoms in orange, y-axis
  atoms in blue, atoms shared by both in pink, all other atoms as grey sticks. The axis
  titles use the same colours, and a legend in the side panel lists the atom indices
  (0-based file indices, from the pair's `dof` entries). The build fails if an index is
  not below the structures' atom count. Turn it off with
  `interactive.highlight_dof_atoms: false`.
- The header switches the map's colour scale between `log counts`, `counts` and
  `free-energy-like` without re-running the pipeline. Free-energy-like values are
  `F = k_B·T·(−ln(P / P_max))`, where `P` is the bin population: the most-populated bin is
  0 and unsampled bins stay blank. This is derived from frame counts, not energies, so it
  is a population-derived free-energy-like surface, not a potential energy surface. The
  temperature (K) and the unit (`kT`, `kJ/mol`, `kcal/mol`, `eV`, `meV`, `cm⁻¹`) can be
  changed in the page; `kT` is dimensionless and ignores the temperature. The side panel
  shows the hovered bin's value in the current unit and its raw count.
  - `interactive.default_scale` sets the opening mode (`log_counts` | `counts` |
    `free_energy`); unset, it follows `density.log_scale`.
  - `interactive.free_energy.temperature` / `.unit` set the opening temperature and unit,
    falling back to `transitions.temperature` / `transitions.energy_unit`. With no
    temperature from either, the page opens in `kT` with an empty temperature field.
    An unknown unit or a temperature ≤ 0 stops the build with an error.
  - The `k_B` values per unit live in one table in `src/units.py`, which the page embeds.
- When the coordinate table has the pair's state column (`state_<pair>`), a `States`
  button tints each bin with its majority state (35 % opacity over the density) and puts
  each state's name at its population-weighted centre (circular mean on periodic axes).
  The side panel names the hovered bin's state. Noise or unset frames winning a bin leave
  it untinted. Without the state column the button is not shown.
  `interactive.show_states` (default `false`) sets whether the overlay is on at opening.
  - States are clustered separately per `clustering.groupby` group (e.g. per bead), and
    their labels are not matched across groups: state `0` of bead 00 can be a different
    region from state `0` of bead 01. The overlay therefore never pools groups. It shows
    one group at a time, chosen from a dropdown next to the button (hidden when there is
    one group).
- XYZ ingestion is designed around streaming parsing and cached byte-offset frame indices for fast random access.
