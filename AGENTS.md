# PES / conformer analysis project instructions

## Mission
Build a reproducible Python workflow that reads multi-frame xyz structures from `./data`,
computes carboxyl and ester coordinates using two definitions, generates density maps,
assigns conformational states, computes transitions between states, and produces both
static and standalone interactive visualizations.

The workflow must also be designed so that, in the future, it can accept precomputed
carboxyl/ester coordinate values instead of full xyz structures without major refactoring.

## Primary scientific scope
Compute two coordinate systems for every configuration:

1. Plane-to-plane definition
   - carboxyl = angle between plane(C1–C6) and plane(O1,C7,O2)
   - ester    = angle between plane(C1–C6) and plane(O3,C8,O4)

2. Dihedral definition
   - carboxyl = dihedral defined by atoms (6, 5, 10, 7) in the input numbering
   - ester    = dihedral defined by atoms (5, 6, 12, 11) in the input numbering

Then, for both definitions:
- build 2D density plots
- define conformational states
- compute state-to-state transition counts
- compute transition probabilities
- compute physical rates only if time spacing is explicitly available in config
- estimate activation free-energy barriers only if both time spacing and temperature are explicitly available in config

## Confirmed data assumptions
- xyz files are multi-frame.
- files represent ordered trajectories
- PIMD trajectories are represented by many ordered trajectory files, one per bead
- transitions are computed across concatenated files within a group
- currently all xyz files use the same atom ordering and indexing
- future datasets may not preserve the same ordering/indexing
- future datasets may provide only carboxyl and ester values instead of full xyz structures

## Angle conventions
### Plane-to-plane angles
- default implementation: unsigned angles in the range `[0, 180]`
- optional future implementation: signed angles in the range `[-180, 180)`

### Dihedral angles
- use signed dihedrals in the range `[-180, 180)`

## Confirmed atom indexing and mapping
- all atom indices below are **0-based Python file indices** (NOT 1-based chemical labels)
- the benzene ring skips file index 4 (a non-ring atom sits there), so no simple +1/-1
  conversion applies between chemical label and file index for C5 and C6
- current implementation may assume one global atom ordering for the dataset

### Plane and dihedral definitions in actual file indices (0-based)
- ring plane (C1–C6): `[0, 1, 2, 3, 5, 6]`
- carboxyl plane (O1, C7, O2): `[9, 10, 7]`
- ester plane (O3, C8, O4): `[12, 11, 8]`
- carboxyl dihedral (C6–C5–C7–O2): `[6, 5, 10, 7]`
- ester dihedral (C5–C6–O3–C8): `[5, 6, 12, 11]`

### Plane construction rule
- `plane(C1–C6)` means a best-fit plane through all six ring atoms

## State model
- use clustering-based state assignment
- clustering algorithm: DBSCAN
- clustering is performed separately for:
  - plane-based coordinates
  - dihedral-based coordinates
  - each trajectory family

## Transition policy
- transition counts and probabilities are computed across concatenated files within a group
- for PIMD:
  - analyze each bead separately
  - average transition observables across beads afterward
- if `dt` is provided in config, physical rates are required
- if both `dt` and `temperature` are provided in config, activation free-energy barriers are required
- if `dt` is not provided, report counts and probabilities only

## Interactive environment
- required interactive output: standalone HTML
- clicked structure behavior: show nearest structure
- standalone 3D structure rendering should use a browser-native HTML solution
- structure panel should include 3D rendering and metadata

## Interactive metadata schema
The interactive artifact must embed, at minimum:
- `source_file`
- `frame_number`
- `byte_offset`
- `atom_count`
- `comment_line`
- `trajectory_id`
- `bead_id`
- `carboxyl_plane`
- `ester_plane`
- `carboxyl_dihedral`
- `ester_dihedral`
- `state_plane`
- `state_dihedral`
- `energy` if available
- structure lookup token or direct xyz payload/path

## Viewer implementation note
- prefer 3Dmol.js for standalone HTML rendering
- use ASE for xyz parsing / frame extraction / structure handling
- do not make `nglview` the primary standalone viewer


## Architecture requirements
The code must be organized around a data-model boundary so that coordinate extraction
from xyz and direct ingestion of precomputed coordinates are separate adapters.

Recommended layered design:
1. input adapters
   - xyz trajectory adapter
   - future coordinate-table adapter
2. geometry / coordinate engine
3. state assignment
4. transition analysis
5. static plotting
6. standalone interactive artifact generation
7. structure lookup / display utilities

Do not couple plotting logic directly to xyz parsing logic.

## Fast xyz loading requirement
The xyz loader must be implemented as a streaming parser and must not require loading full
trajectory files into memory.

### Required behavior
- Parse xyz files directly.
- Use the first line of each frame to read the atom count `N`.
- Treat each frame as a block of `N + 2` lines:
  - line 1: atom count
  - line 2: comment / metadata
  - next `N` lines: atom records
- Build a per-file frame index that enables near-instant access to a selected frame.

### Random-access design
Because xyz frames are typically variable-width in bytes, frame access must not rely only on
line counts. The loader must build and cache a frame index containing at least:
- `source_file`
- `frame_number`
- `byte_offset`
- `atom_count`
- optional comment line / parsed metadata

The loader must then retrieve a selected frame by:
1. opening the file,
2. seeking to `byte_offset`,
3. reading exactly `N + 2` lines for that frame.

### Performance requirements
- Do not load full xyz files into memory just to retrieve one frame.
- The first pass through a file may build the frame index.
- Subsequent frame retrieval must use the cached byte-offset index.
- The indexing layer should be reusable by both analysis code and the interactive viewer.

### Caching requirements
- Support persistent caching of frame indices, so repeated runs do not need to rescan all files.
- Cache invalidation must occur if the source file path, size, or modification time changes.
- The cache format may be JSON, CSV, pickle, sqlite, or another lightweight local format.

### Note on xyz random access
The fact that each frame occupies `N + 2` lines defines the frame structure, but does not by
itself guarantee O(1) direct file access. Near-instant frame retrieval requires a byte-offset
index built during an initial scan or stored from a previous run.

## Non-negotiable engineering requirements
- Geometry functions must be reusable and easily extractable.
- Separate geometry, state assignment, transition logic, static plotting, and interactive plotting into different modules.
- Save all non-interactive figures as PNG with `dpi=300`.
- Provide standalone interactive HTML plots.
- Clicking a point/bin/region in the interactive artifact must show the nearest structure from that region.
- The interactive artifact must embed enough metadata to retrieve structure metadata and file paths quickly.
- All user-facing paths and analysis settings must come from config where practical.
- Do not hard-code dataset-specific assumptions unless they are documented in config or README.
- Fail loudly on invalid atom indexing, malformed xyz files, or inconsistent structure sizes.

## Important scientific constraints
- Do not assume that “PES” means a true potential energy surface unless energies are provided.
  If only xyz structures or coordinate values are available, describe the result as a
  coordinate-density landscape or population-derived free-energy-like surface.
- Do not report physical transition rates unless frame ordering and time spacing are known.
  If time information is missing, report transition counts and transition probabilities instead.
- Do not silently assume atom numbering conventions. Validate and document them.
- Transition analysis for PIMD must preserve bead identity unless an explicit aggregation step is configured.


## Current assumptions that are allowed for now
- all current xyz files use the same atom ordering and indexing
- atom mappings may therefore be configured once globally for the current dataset

## Future-compatibility requirements
The workflow must support both of these modes with minimal shared-core changes:

1. full-structure mode
   - input = xyz trajectories
   - output includes clickable structure lookup

2. coordinate-only mode
   - input = table containing carboxyl and ester values
   - same density/state/transition pipeline should still work
   - structure lookup becomes optional or disabled if no structure metadata exists

Design the internal API so that downstream analysis operates on a standard coordinate table,
not directly on xyz frame objects.

## Suggested repository structure
- `confana/io_xyz.py`
- `confana/io_coordinates.py`
- `confana/models.py`
- `confana/geometry.py`
- `confana/coordinates.py`
- `confana/density.py`
- `confana/states.py`
- `confana/transitions.py`
- `confana/plots_static.py`
- `confana/plots_interactive.py`
- `confana/viewer.py`
- `confana/cli.py`
- `configs/default.yaml`
- `tests/test_geometry.py`
- `tests/test_io_xyz.py`
- `tests/test_io_coordinates.py`
- `tests/test_coordinates.py`
- `tests/test_states.py`
- `tests/test_transitions.py`
- `outputs/`

## Required public API
Implement functions with stable signatures or close equivalents:

- `load_xyz_files(path_pattern)`
- `scan_xyz_frame_offsets(path) -> FrameIndex`
- `load_or_build_xyz_index(path, cache_path=None) -> FrameIndex`
- `read_xyz_frame(path, frame_number, frame_index) -> FrameRecord`
- `read_xyz_frame_by_offset(path, byte_offset) -> FrameRecord`
- `extract_frame_metadata(path, frame_number, frame_index) -> dict`
- `load_coordinate_table(path)`
- `build_frame_metadata(frames)`
- `plane_normal(coords, atom_ids)`
- `best_fit_plane(coords, atom_ids)`
- `plane_plane_angle(coords, plane1_atom_ids, plane2_atom_ids, signed=False)`
- `dihedral_angle(coords, atom_ids)`
- `compute_angles_plane(frame, mapping, convention)`
- `compute_angles_dihedral(frame, mapping, convention)`
- `build_coordinate_table_from_xyz(frames, mapping, conventions)`
- `build_coordinate_table_from_values(table, conventions)`
- `assign_conformer_states(df, definition, scheme, params)`
- `compute_transition_counts(states, lag=1, groupby=None)`
- `compute_transition_probabilities(counts)`
- `compute_transition_rates(counts, dt, lag=1)`  # only if dt exists
- `compute_activation_free_energy_barriers(rates, temperature, model="eyring")`  # only if dt and temperature exist
- `make_density_png(df, definition, outpath, dpi=300)`
- `make_transition_png(transitions, outpath, dpi=300)`
- `make_density_interactive(df, definition, metadata, state_scheme=None)`
- `make_transition_interactive(transitions, metadata, df, state_scheme=None)`
- `lookup_nearest_structure(selection_payload, metadata)`
- `render_structure_panel(structure_record)`

## Standard coordinate table schema
All downstream analysis must consume a standard coordinate table with fields such as:
- `frame_id`
- `source_file`
- `trajectory_id`
- `bead_id`          # nullable for non-PIMD
- `frame_number`
- `byte_offset`
- `atom_count`
- `comment_line`
- `local_frame_index`
- `global_frame_index`
- `carboxyl_plane`
- `ester_plane`
- `carboxyl_dihedral`
- `ester_dihedral`
- `state_plane`
- `state_dihedral`

Optional metadata fields:
- `xyz_path`
- `byte_offset` or frame lookup token
- `has_structure`
- `energy`           # optional future extension

## Output requirements
For every run, produce:
1. Per-configuration coordinate table with:
   - frame_id
   - source_file
   - trajectory_id
   - bead_id if applicable
   - carboxyl_plane
   - ester_plane
   - carboxyl_dihedral
   - ester_dihedral
   - state_plane if assigned
   - state_dihedral if assigned

2. Static plots:
   - plane density PNG
   - dihedral density PNG
   - plane transition PNG
   - dihedral transition PNG

3. Interactive outputs:
   - plane density standalone HTML
   - dihedral density standalone HTML
   - clickable nearest-structure inspection

4. Serialized analysis data:
   - CSV or parquet coordinate table
   - transition count matrices
   - transition probability matrices
   - transition rate matrices only if dt is available
   - activation free-energy barrier matrices only if both dt and temperature are available

## Acceptance criteria
### Geometry
- Unit tests verify best-fit plane, plane-plane angle, and dihedral angle on small synthetic fixtures.
- Index validation prevents off-by-one mistakes.
- Geometry functions are importable without plotting dependencies.

### Coordinate extraction
- All structures in `./data` are processed or clearly reported as failed.
- Coordinate extraction works for both plane and dihedral definitions.
- Coordinate-only input mode can run downstream analysis without xyz parsing.

### Density analysis
- 2D density maps exist for both definitions.
- Static density plots are saved as PNG with `dpi=300`.

### State assignment
- Every frame is assigned to a cluster label or explicitly marked unassigned if that is part of the configured method.
- State definitions are reproducible through config.

### Transition analysis
- Transition counts and probabilities are computed for both definitions.
- Physical rates are computed only if ordered frames and `dt` exist.
- PIMD trajectories preserve bead identity unless explicit aggregation is configured.

### Interactivity
- Clicking a region/bin/point selects the nearest associated frame.
- Associated structure metadata is displayed.
- Structure lookup is deterministic and reproducible.
- Standalone HTML works without a Python server.

## Coding rules
- Prefer clear, typed, testable functions.
- Keep I/O, geometry, clustering, and plotting separate.
- Keep plotting code independent from state assignment where possible.
- Avoid hidden global state.
- Add docstrings for all public functions.
- Use small fixtures in tests rather than relying on the full dataset.

## Execution rules for the agent
- Start by inspecting the repo and summarizing the current state before large edits.
- Implement in small, reviewable patches.
- Run tests after each major module is added.
- Update README as behavior or interfaces change.
- If a TODO item blocks scientifically correct output, stop and mark the task blocked rather than guessing.

## Implementation backlog

### Phase 0 — repo bootstrap
Create the initial project skeleton:
- `confana/`
- `tests/`
- `configs/`
- `outputs/`
- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `pyproject.toml`

Requirements:
- project installs cleanly
- test runner works
- basic package structure is in place

### Phase 1 — config contract
Create `configs/default.yaml` and config loading utilities.

Include:
- data paths
- atom indexing convention
- atom mappings
- angle conventions
- clustering settings
- transition settings
- PIMD settings
- interactive settings
- output paths

Requirements:
- all important analysis settings come from config
- missing required config values fail clearly

### Phase 2 — core data model
Create `confana/models.py` and define the standard internal schema.

Include:
- frame metadata model
- coordinate-table schema
- trajectory grouping fields
- PIMD bead fields
- interactive metadata fields

Requirements:
- downstream analysis must operate on the coordinate table, not raw xyz objects

### Phase 3 — xyz ingestion
Create `confana/io_xyz.py`:
- load multi-frame xyz files
- preserve file order and frame order
- parse each frame as:
  - line 1: atom count
  - line 2: comment / metadata
  - next N lines: atom records
- attach:
  - `source_file`
  - `trajectory_id`
  - `bead_id`
  - `local_frame_index`
  - `global_frame_index`
  - `atom_count`
  - `comment_line`

Requirements:
- support ordered multi-frame trajectories
- support PIMD trajectories as multiple ordered files
- do not assume the full trajectory must remain in memory after parsing
- keep bead and trajectory metadata explicit

Tests:
- valid multi-frame xyz file
- malformed xyz file
- ordered frame numbering
- PIMD-like multiple-file ingestion

TODO:
- exact rule for deriving `trajectory_id`
- exact rule for deriving `bead_id` from filenames
- whether energy is present in the comment line

### Phase 3A — fast random-access xyz indexing
Create indexed streaming access for multi-frame xyz trajectories.

Implement:
- one-pass scan of each xyz file
- build a per-file frame index with:
  - `source_file`
  - `frame_number`
  - `byte_offset`
  - `atom_count`
  - `comment_line`
- cached frame index on disk
- retrieval of a single frame via `seek()`

Required functions:
- `scan_xyz_frame_offsets(path) -> FrameIndex`
- `load_or_build_xyz_index(path, cache_path=None) -> FrameIndex`
- `read_xyz_frame(path, frame_number, frame_index) -> FrameRecord`
- `read_xyz_frame_by_offset(path, byte_offset) -> FrameRecord`
- `extract_frame_metadata(path, frame_number, frame_index) -> dict`

Requirements:
- frame scan must not store all frame coordinates in memory
- selected-frame retrieval must read only one frame
- interactive structure lookup must reuse the same index
- repeated runs should reuse the cached index if the source file has not changed

Tests:
- verify correct frame count
- verify byte offsets point to the correct frame
- verify retrieved frame matches sequential parse
- verify cache invalidates when source file changes
- verify random-access retrieval works for first, middle, and last frame

TODO:
- decide cache format
- decide cache invalidation policy
- decide whether energy or other metadata should be parsed from the comment line

### Phase 4 — coordinate-table ingestion
Create `confana/io_coordinates.py`:
- load precomputed coordinate tables
- validate required columns
- map them into the standard downstream schema

Requirements:
- downstream density/state/transition code must work without xyz parsing
- structure lookup becomes optional when only coordinate tables are provided

Tests:
- valid coordinate table input
- missing required columns
- compatibility with downstream schema

### Phase 5 — geometry primitives
Create `confana/geometry.py`:
- best-fit plane through arbitrary atom sets
- plane normal
- plane-plane angle
- dihedral angle
- index normalization and validation

Requirements:
- 1-based task indices are converted safely to internal indexing
- plane(C1–C6) uses a best-fit plane through all six atoms
- default plane angle is unsigned `[0,180]`
- dihedrals use `[-180,180)`

Tests:
- best-fit plane
- plane-plane angle
- dihedral angle
- invalid indices
- degenerate plane cases

### Phase 6 — coordinate extraction
Create `confana/coordinates.py`:
- compute plane-based coordinates
- compute dihedral-based coordinates
- build the standard coordinate table

Outputs:
- `carboxyl_plane`
- `ester_plane`
- `carboxyl_dihedral`
- `ester_dihedral`

Requirements:
- works from xyz input
- works from coordinate-only input where appropriate

### Phase 7 — static density analysis
Create:
- `confana/density.py`
- `confana/plots_static.py`

Implement:
- 2D histogram first
- optional KDE later

Outputs:
- plane density PNG
- dihedral density PNG

Requirements:
- save PNG with `dpi=300`
- consistent axis labels and units

TODO:
- binning parameters
- axis ranges
- colormap

### Phase 8 — clustering-based state assignment
Create `confana/states.py`.

Implement:
- DBSCAN-based clustering
- separate clustering for:
  - plane-based coordinates
  - dihedral-based coordinates
  - each trajectory family

Requirements:
- cluster labels written back to the coordinate table
- clustering parameters come from config

TODO:
- `eps`
- `min_samples`

### Phase 9 — transition analysis
Create `confana/transitions.py`.

Implement:
- transition counts
- transition probabilities
- physical rates only if `dt` is provided
- activation free-energy barriers only if both `dt` and `temperature` are provided

Requirements:
- transitions computed across concatenated files within a group
- PIMD beads analyzed separately
- transition observables averaged across beads afterward

TODO:
- exact averaging outputs and uncertainty handling

### Phase 10 — static transition plots
Extend `confana/plots_static.py`.

Outputs:
- transition count heatmaps
- transition probability heatmaps
- rate plots if valid

### Phase 11 — standalone interactive HTML
Create `confana/plots_interactive.py`.

Implement:
- standalone HTML output
- click selection of nearest structure
- embedded metadata for fast lookup
- reuse byte-offset frame index

Requirements:
- no Python server required
- interactive artifact contains enough metadata for structure retrieval

### Phase 12 — structure viewer
Create `confana/viewer.py`.

Implement:
- nearest-structure lookup
- 3D rendering in standalone HTML
- metadata panel

Requirements:
- 3D rendering plus metadata
- fast retrieval from indexed xyz files

TODO:
- exact embedded-structure strategy
- whether full xyz text is embedded or loaded by token/path

### Phase 13 — CLI
Create `confana/cli.py`.

Commands:
- `extract-coordinates`
- `plot-densities`
- `cluster-states`
- `compute-transitions`
- `build-interactive`
- `run-all`

### Phase 14 — tests and validation
Add:
- unit tests for geometry
- ingestion tests
- coordinate extraction tests
- clustering tests
- transition tests
- random-access frame retrieval tests

### Phase 15 — documentation
Update `README.md`:
- setup
- config
- usage
- outputs
- assumptions
- TODO items
- distinction between density landscape and true PES
- distinction between transition probabilities and physical rates

## Definition of done
Done means:
- code is modular
- tests pass
- README documents usage
- config file exists
- full-structure mode works
- coordinate-only mode skeleton exists
- all requested plots are generated
- unresolved assumptions remain explicitly marked as TODO

Use
```
source .venv/bin/activate
```
to activate the environment where the package is installed for testing and development.
