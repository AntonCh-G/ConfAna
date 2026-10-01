# Configuration

Full reference for ConfAna's YAML config. For a working minimal config see
[examples/md17_aspirin.yaml](../examples/md17_aspirin.yaml) (one trajectory) or
[examples/pimd_template.yaml](../examples/pimd_template.yaml) (one file per PIMD bead); for every option with comments, keep a
local copy such as `configs/default.yaml` (the `configs/` folder is git-ignored).

## Config sections

The workflow is driven by YAML config. Start from an example in [examples/](../examples/) and keep your own copies under `configs/` (git-ignored).

Important sections include:

- `data`: input format, path pattern, and trajectory and bead ID rules (see [Input data](#input-data))
- `frame_range`, `bond_break`: which frames of each file go into the table (see [Input data](#input-data))
- `dof`: named degrees of freedom (dihedral, distance, angle) with 0-based atom indices
- `coordinate_transforms`: optional angle shifts (adds `<name>_shifted` columns; each name must be an enabled DoF)
- `atom_mapping`: named atom groups for plane-based coordinates
- `dihedrals`: named dihedral definitions
- `coordinate_pairs`: which coordinate pairs get plotted
- `conventions`: signed/unsigned angle settings
- `cache`: coordinate-table and frame-index caches, worker processes (see [Input data](#input-data))
- `density`: histogram bins, ranges, and coloring
- `clustering`: state-assignment settings
- `transitions`: lag, optional `dt`, and optional activation-barrier settings
- `interactive`: interactive HTML behavior (Plotly/3Dmol inlining, theme, alignment, XYZ payload embedding)
- `outputs`: output directory and file settings

Local configs are intentionally ignored by git because they often contain machine-specific data and output paths.

## Input data

These settings decide which frames the coordinate table holds. Every input format
goes through the same steps (`confana/coordinate_table.py`); only reading the files
differs.

```yaml
run_dir: outputs/my_run           # caches go to {run_dir}/.cache/

data:
  format: xyz                     # xyz (default) or hdf5
  path_pattern: ./data/my_run.pos_*.xyz   # glob; ** matches nested folders
  trajectory_id_pattern: '^(.+)_\d+$'     # optional; "my_run.pos_03" -> "my_run.pos"
  bead_id_pattern: '_(\d+)$'              # optional; "my_run.pos_03" -> "03"
  # positions_source: bead        # hdf5 only: bead (default) or centroid

frame_range:                      # optional; 0-based frame numbers, both ends kept
  start_frame: 1000
  end_frame: null                 # null: to the last frame

bond_break:                       # optional; xyz only
  enabled: true
  cutoff: 2.0                     # Å

cache:
  trajectory_cache_dir: null      # default {run_dir}/.cache
  index_cache_dir: null           # default: beside each xyz file
  n_jobs: 1                       # processes for rebuilding; -1 = all cores
```

**Files.** `path_pattern` is a glob; the matches are sorted by path. Files ending in
`.xyz` (or `.hdf5` / `.h5` for `format: hdf5`) are kept; if none does, every matched
file is read. No match is an error.

**Trajectory and bead ids.** One rule decides both, for the table and for its cache:

- `trajectory_id` is group 1 of `trajectory_id_pattern`, searched in the file name
  without its extension. Without a pattern it is the name of the folder holding the
  file, so all files in one folder form one trajectory.
- `bead_id` is group 1 of `bead_id_pattern`; without a pattern a file has no bead.
- A pattern without a capture group, or one that gives no id for a matched file,
  stops the run with an error naming the file.

The files of one trajectory are concatenated in name order; transitions are counted
per trajectory and bead across them.

**HDF5.** With `format: hdf5` each file is one PIMD run laid out as
`<sim_dir>/hdf5/trajectory.hdf5` with `<sim_dir>/input.xyz` (see `CONTEXT.md`). Its
`trajectory_id` is `<sim_dir>`'s name and its beads are `bead_00`, `bead_01`, …, so the
two id patterns are errors with HDF5. `positions_source: centroid` gives one row per
frame from `positions` instead of one row per bead and frame.

**`frame_range`.** Keeps frames `start_frame` to `end_frame` of every file (with HDF5,
of every bead). The table keeps each file's own frame numbers.

**`bond_break`.** When enabled, the atoms bonded in the first frame of the range
(covalent radii × 1.1) are watched, and the first later frame in which one of these
bonds is longer than `cutoff` ends the file. Every file of the same trajectory is cut
at the earliest such break among them; a file without a break counts as breaking
after its last frame, so the trajectory's files are also cut to the shortest one. This
suits one file per PIMD bead. With HDF5 it is an error until it is decided whether
a break in one bead should cut every bead of the run in the same way.

**Coordinate cache.** Each trajectory's table is cached as
`{trajectory_id}__{hash}__coordinates.npz` in `trajectory_cache_dir`, and rebuilt
when one of these changes: its files (path, size, modification time), its
`trajectory_id` or a file's `bead_id`, `format`, `positions_source`, the enabled DoF
(name, type, atoms, domain), `frame_range` or `bond_break`. Changing an id pattern
rebuilds exactly the trajectories whose ids change; a new file in `path_pattern`
rebuilds only the trajectory it joins. `coordinate_transforms` are applied after loading
and never cached. The commands report `cache_hit=True` when no trajectory was
rebuilt. Caches from ConfAna versions before this layout are rebuilt once, and frame
indices kept in `index_cache_dir` are rescanned once (their names now carry a hash; the
old index files are no longer read and can be deleted). `cache.coordinate_table_path`
is no longer used and is an error.

**Frame index.** Each xyz file is scanned once into `{file}.frameindex.npz` beside
it, or `{file}.{hash of its path}.frameindex.npz` in `index_cache_dir`, and rescanned
when the file's path, size or modification time changes.

**DoF types.** The table computes `dihedral`, `distance` and `angle` DoF; a
`collective` or `external` DoF is an error until those types are implemented.

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
