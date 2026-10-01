# ConfAna Domain Glossary

## Core Concepts

**Conformational map**
A 2D population landscape derived from PIMD/MD sampling, showing how often each pair of coordinate values occurs. Not a true potential energy surface (PES) unless frame energies are provided. The primary result of this workflow.
Each coordinate pair has one map, and every view of that pair (the static PNG, the interactive page, the state overlay) reads the same bins. A frame belongs to a bin exactly as `numpy.histogram2d` places it: a value on an interior edge goes to the upper bin, the right edge belongs to the last bin, and a frame outside the map's range is on no bin.

**DoF (degree of freedom)**
A named scalar quantity computed per frame. Defined in the config `dof:` list with a `type:` field. Supported types:
- `dihedral` — signed torsion angle in `[-180, 180)`, defined by 4 atom indices
- `distance` — interatomic distance in Å, defined by 2 atom indices
- `angle` (bond angle) — angle in degrees, defined by 3 atom indices
- `collective` — linear combination or PCA projection of other DoF (not implemented: a config that uses it is an error)
- `external` — value loaded from an external table by `frame_id` (not implemented: a config that uses it is an error)

**CoordinatePair**
A named 2D analysis space pairing two DoF. Drives density plotting, clustering, transition analysis, and interactive output. Key properties:
- `x_col`, `y_col` — column names of the two DoF in the coordinate table
- `x_domain`, `y_domain` — axis extents `(min, max)`
- `state_col` — derived as `f"state_{pair.name}"`
- `periodic` — True when both DoF are dihedral (affects DBSCAN embedding)
- `feature_columns` — returns `[x_col, y_col]`

**Coordinate table**
The central data structure passed between all pipeline stages. A `pd.DataFrame` with one row per frame. Required metadata columns: `frame_id`, `source_file`, `trajectory_id`, `bead_id`, `frame_number`, `byte_offset`, `atom_count`, `comment_line`, `local_frame_index`, `global_frame_index`. DoF value columns are added dynamically from the config.
Built by one pipeline for every input format (`coordinate_table.py`): discover the files, apply the trajectory id rule, check each trajectory's cache, cut to `frame_range` and `bond_break`, compute the DoF values, apply the coordinate shifts, reset the state columns, validate the schema. Only reading frames differs between xyz and HDF5. `frame_id` is the row number of the whole table, so it shifts when files or `frame_range` change.

**Trajectory**
The files that share one `trajectory_id`; their frames are concatenated in file-name order, and transitions run across them. The one id rule: group 1 of `data.trajectory_id_pattern`, searched in the file name without its extension; with no pattern, the folder holding the file (for HDF5, the simulation folder above `hdf5/`). `bead_id` works the same way with `data.bead_id_pattern`; no pattern means no bead. A pattern without a capture group, or one that gives no id for a matched file, is an error.

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
A byte-offset index built by scanning each xyz file once. Enables O(1) random access to any frame by seeking to `byte_offset`. Cached to disk as `{file}.frameindex.npz` alongside the source file, or as `{file}.{hash of its path}.frameindex.npz` in `cache.index_cache_dir`.

## Interactive Map Concepts

**Representative frame**
The one frame shown for a density bin when the page embeds structures (`interactive.embed_xyz_payload: true`): the sampled frame closest to the bin centre, ties going to the earlier frame in the coordinate table. It is chosen only among the frames the bin counts, so every bin with frames has one and an empty bin has none. The same frame gives the bin's structure, its metadata and its pin; a bin never shows another frame's structure. The structure is read from the frame's own trajectory file (xyz or HDF5); if a frame the table names cannot be read, the page build stops with an error naming it, while a frame that names no structure (coordinate-only input) simply has none. Each coordinate-pair page picks its own representatives, so the same bin region on two pages usually has different representative frames. Without embedded structures the page holds every frame and pins the frame nearest the click point instead.

**Preview**
The one bin currently shown in the side panel, with its representative frame and values. It is passing: the next bin pointed at replaces it. With a mouse, hovering a bin previews it; on a touch screen, tapping a bin previews it.
_Avoid_: hover preview (on a touch screen there is no hover)

**Pin**
One frame the researcher has selected on the interactive map to keep on screen. A pin is a frame, not a bin: pinning a bin pins that bin's representative frame. With a mouse, clicking a bin pins it; on a touch screen, pinning is a separate step from the preview. The same pin appears on every coordinate-pair page of the run, placed at that frame's own coordinates in each pair.
_Avoid_: pinned bin, card (the card is only how a pin is displayed)

**Pin number**
The small integer that names a pin on both the map and its card. A new pin takes the lowest number not in use, so a number freed by closing a pin is reused.

**Pin marker**
The on-map sign of a pin: a numbered badge with a short arrow pointing at the frame's exact coordinates, plus the outline of the bin the frame falls in.

**Preview marker**
The on-map sign of the preview on a touch screen: an outline of the previewed bin, kept until the next tap. With a mouse there is none; the pointer itself shows the bin.

## Config Schema

**`run_dir`** (required, top-level)
Single root directory for all outputs and caches for a given run. Missing or null = hard validation error at startup. Derived paths:
- plots → `{run_dir}/plots/`
- coordinate-table cache, one NPZ per trajectory → `{run_dir}/.cache/` (overridable via `cache.trajectory_cache_dir`; `cache.coordinate_table_path` is no longer used and is an error)

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
All beads within one file share the same `trajectory_id`; bead identity is encoded as `bead_id = "bead_00"`, `"bead_01"`, …. Atom types are read from `input.xyz` in the simulation directory (parent of `hdf5/`), which must be a complete xyz frame (atom count, comment, one `symbol x y z` line per atom). `byte_offset` is set to `-1` (sentinel, `io_hdf5.HDF5_BYTE_OFFSET`) in all HDF5-sourced rows; a frame's structure is read from `source_file` by `frame_number` + `bead_id` instead (`bead_positions[frame, bead]`, or `positions[frame]` for a centroid row with no `bead_id`).

Configured via `data.format: hdf5` and `data.positions_source: bead | centroid` (default `bead`). `trajectory_id` is the simulation folder (the parent of `hdf5/`), so `data.trajectory_id_pattern` and `data.bead_id_pattern` are errors with HDF5. `frame_range` cuts every bead to the same frames; `bond_break` is an error with HDF5 until it is decided whether a break in one bead should cut every bead of the run, as it cuts every bead file of an xyz trajectory.

## Architecture Notes

The pipeline is layered with a data-model boundary:
1. **Input adapters** — `io_xyz.py` (xyz files), `io_hdf5.py` (HDF5 PIMD files), `io_coordinates.py` (precomputed tables); `frame_source.py` reads the frame a coordinate-table row names, through the xyz or HDF5 reader
2. **Geometry / coordinate engine** — `geometry.py`, `coordinates.py`; `coordinate_table.py` builds the coordinate table from either trajectory reader
3. **State assignment** — `states.py`
4. **Transition analysis** — `transitions.py`
5. **Static plotting** — `plots_static.py`
6. **Interactive artifact** — `plots_interactive.py` (page assembly), `viewer.py` (per-bin structures and metadata), `payload_codec.py` (compact embedded encoding, decoded in the browser by `interactive_assets/viewer.js`)
7. **CLI** — `cli.py` wires together all stages via `list_coordinate_pairs(config)`

All downstream stages accept a `CoordinatePair` object, not a definition string. The pair carries all necessary domain, label, and column information.
