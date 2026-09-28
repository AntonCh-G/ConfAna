# ConfAna Domain Glossary

## Core Concepts

**Conformational map**
A 2D population landscape derived from PIMD/MD sampling, showing how often each pair of coordinate values occurs. Not a true potential energy surface (PES) unless frame energies are provided. The primary result of this workflow.

**DoF (degree of freedom)**
A named scalar quantity computed per frame. Defined in the config `dof:` list with a `type:` field. Supported types:
- `dihedral` — signed torsion angle in `[-180, 180)`, defined by 4 atom indices
- `distance` — interatomic distance in Å, defined by 2 atom indices
- `angle` (bond angle) — angle in degrees, defined by 3 atom indices
- `collective` — linear combination or PCA projection of other DoF (stub — not yet implemented)
- `external` — value loaded from an external table by `frame_id` (stub — not yet implemented)

**CoordinatePair**
A named 2D analysis space pairing two DoF. Drives density plotting, clustering, transition analysis, and interactive output. Key properties:
- `x_col`, `y_col` — column names of the two DoF in the coordinate table
- `x_domain`, `y_domain` — axis extents `(min, max)`
- `state_col` — derived as `f"state_{pair.name}"`
- `periodic` — True when both DoF are dihedral (affects DBSCAN embedding)
- `feature_columns` — returns `[x_col, y_col]`

**Coordinate table**
The central data structure passed between all pipeline stages. A `pd.DataFrame` with one row per frame. Required metadata columns: `frame_id`, `source_file`, `trajectory_id`, `bead_id`, `frame_number`, `byte_offset`, `atom_count`, `comment_line`, `local_frame_index`, `global_frame_index`. DoF value columns are added dynamically from the config.

**State column**
A string column `state_{pair.name}` added by the clustering step (e.g., `state_dihedral`). Contains DBSCAN cluster labels. NA = noise / unassigned. Not present in the coordinate table cache — added by `assign_conformer_states`.

## Analysis Concepts

**Transition rate**
Computed only when `dt` is explicitly set in config. Otherwise the output contains transition counts and probabilities only. Do not assume ordered frames unless both file ordering and `dt` are confirmed.

**Activation free-energy barrier**
A rate-derived estimate of the free-energy barrier between two conformational states, computed from transition rates and temperature and reported in eV by default; not a true potential-energy barrier from a PES.

Activation free-energy barriers are derived for the same origin/destination conformational-state pairs as transition rates.
Barrier calculations use eV internally; reported barriers multiply the internal eV value by the configured `energy_conv_factor`.
Self-transitions do not define barriers between states; diagonal barrier entries are reported as missing values.
Off-diagonal state pairs with zero or missing transition rates have missing barrier entries, meaning the barrier was not estimated from the available data.
Barrier outputs are produced only when both `dt` and `temperature` are configured; otherwise transition analysis reports counts, probabilities, and any valid rates only.
Negative activation free-energy barriers are reported without clamping; they indicate that the configured kinetic model or prefactor is inconsistent with the observed discrete-time rate estimate.

**Eyring barrier estimate**
The default activation free-energy barrier model, using the transition-state-theory prefactor `k_B T / h` and an optional transmission coefficient.

**Arrhenius barrier estimate**
An activation barrier estimate that requires an explicit attempt frequency; avoid using it unless that prefactor is scientifically justified.

**Attempt frequency**
The Arrhenius prefactor for rate-derived barrier estimates, configured explicitly in s^-1 when using the Arrhenius barrier model.

**PIMD beads**
Each bead is a separate ordered xyz trajectory sharing the same trajectory_id. Transitions are analyzed per-bead and then averaged across beads. Bead identity is preserved throughout unless explicit averaging is configured.

**Frame index**
A byte-offset index built by scanning each xyz file once. Enables O(1) random access to any frame by seeking to `byte_offset`. Cached to disk as `.frameindex.npz` alongside the source file (or in `cache.index_cache_dir`).

## Config Schema

**`run_dir`** (required, top-level)
Single root directory for all outputs and caches for a given run. Missing or null = hard validation error at startup. Derived paths:
- plots → `{run_dir}/plots/`
- trajectory cache → `{run_dir}/.cache/` (overridable via `cache.trajectory_cache_dir`)
- coordinate table cache → `{run_dir}/coordinates_angles.npz` (overridable via `cache.coordinate_table_path`)

**`plots:`** block
All rendering settings in one place: `dpi`, `density`, `transitions`, `interactive`. Replaces the old scattered `density:`, `transitions_plot:`, and `interactive:` top-level keys plus `dpi` in `outputs:`.

**`outputs:`**
Retains only serialization flags: `save_csv`, `save_parquet`. No longer holds `dir` or `dpi`.

**`scatter_overlays:`** (top-level list)
Data definition for overlay datasets — which files to load and render as scatter points. Stays top-level because it is data input, not rendering settings.

**`dof:` list**
Replaces the old `dihedrals:` list and `atom_mapping:` section. Each entry defines one named scalar DoF:
```yaml
dof:
  - name: carboxyl_dihedral
    type: dihedral
    atoms: [6, 5, 10, 7]   # 0-based Python indices
    label: "Carboxyl dihedral (°)"
    domain: [-180, 180]
    enabled: true
```

**`coordinate_pairs:`**
Defines 2D analysis spaces. Can be a list (preferred) or dict (legacy):
```yaml
coordinate_pairs:
  - name: dihedral
    x: carboxyl_dihedral
    y: ester_dihedral
```
Domain, periodicity, and labels are inherited from the referenced DoF definitions.

**Scatter overlay**
A set of xyz trajectory files from a different dataset (e.g. a training set from another level of theory) whose dihedral coordinates are computed using the same geometry code and atom mapping as the main dataset, then rendered as semi-transparent scatter points on top of a density PNG. Purpose: compare what region of dihedral space a training set covers relative to the PIMD/MD population landscape. Defined under the top-level `scatter_overlays:` config key. Applied to all coordinate pairs by default.

**HDF5 trajectory**
A PIMD trajectory stored as a single HDF5 file (``trajectory.hdf5``) with datasets:
- `bead_positions` — `(n_frames, n_beads, n_atoms, 3)` float64 Å, one slice per PIMD bead
- `positions` — `(n_frames, n_atoms, 3)` float64 Å, ring-polymer centroid
- `potential` — `(n_frames,)` float64 eV

Multiple independent HDF5 runs (e.g. `s0`, `s1`, …) each become one `trajectory_id`.
All beads within one file share the same `trajectory_id`; bead identity is encoded as `bead_id = "bead_00"` … `"bead_NN"`. Atom types are read from `input.xyz` in the simulation directory (parent of `hdf5/`). `byte_offset` is set to `-1` (sentinel) in all HDF5-sourced rows; structure retrieval uses `source_file` + `frame_number` + `bead_id` instead.

Configured via `data.format: hdf5` and `data.positions_source: bead | centroid` (default `bead`). `trajectory_id` is derived from the parent directory name of each HDF5 file.

## Architecture Notes

The pipeline is layered with a data-model boundary:
1. **Input adapters** — `io_xyz.py` (xyz files), `io_hdf5.py` (HDF5 PIMD files), `io_coordinates.py` (precomputed tables)
2. **Geometry / coordinate engine** — `geometry.py`, `coordinates.py`
3. **State assignment** — `states.py`
4. **Transition analysis** — `transitions.py`
5. **Static plotting** — `plots_static.py`
6. **Interactive artifact** — `plots_interactive.py` (page assembly), `viewer.py` (per-bin structures and metadata), `payload_codec.py` (compact embedded encoding, decoded in the browser by `interactive_assets/viewer.js`)
7. **CLI** — `cli.py` wires together all stages via `list_coordinate_pairs(config)`

All downstream stages accept a `CoordinatePair` object, not a definition string. The pair carries all necessary domain, label, and column information.
