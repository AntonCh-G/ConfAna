# ConfAna agent instructions

Rules for coding agents working in this repo. Terms are defined in `CONTEXT.md`;
design decisions are in `docs/adr/`; code layout is in `docs/development.md`.

## Mission
A reproducible Python workflow that reads multi-frame xyz trajectories (or HDF5, or a
table of precomputed coordinates), computes carboxyl and ester coordinates, builds 2D
density maps, assigns conformational states, computes transitions between states, and
produces static PNG plots and standalone interactive HTML pages.

## Scientific scope
Two coordinate systems per frame:

1. Plane-to-plane
   - carboxyl = angle between plane(C1–C6) and plane(O1, C7, O2)
   - ester    = angle between plane(C1–C6) and plane(O3, C8, O4)
2. Dihedral
   - carboxyl = dihedral C6–C5–C7–O2
   - ester    = dihedral C5–C6–O3–C8

For both: density maps, states, transition counts and probabilities; physical rates only
if `dt` is set; activation free-energy barriers only if both `dt` and `temperature` are set.

## Data
- xyz files are multi-frame, ordered trajectories.
- PIMD: one ordered file per bead; transitions are computed per bead, then averaged.
- Transitions are computed across concatenated files within a group.
- Demo data: public MD17 aspirin (`scripts/md17_to_xyz.py`, `examples/md17_aspirin.yaml`),
  one classical trajectory, 0.5 fs between frames, same atom order as below.
- All current datasets share one atom ordering. Future datasets may not: never assume it.
- Energy is not parsed from comment lines yet (`TODO-energy` in `confana/io_xyz.py`).

## Angle conventions
- Plane-to-plane: unsigned, `[0, 180]` (signed `[-180, 180)` optional).
- Dihedrals: signed, `[-180, 180)`.

## Atom indexing (aspirin)
- All indices are **0-based file indices**, not 1-based chemical labels.
- The benzene ring skips file index 4 (a non-ring atom), so no simple ±1 conversion
  applies to C5 and C6.
- `plane(C1–C6)` is a best-fit plane through all six ring atoms.

| Definition | File indices |
|---|---|
| Ring plane (C1–C6) | `[0, 1, 2, 3, 5, 6]` |
| Carboxyl plane (O1, C7, O2) | `[9, 10, 7]` |
| Ester plane (O3, C8, O4) | `[12, 11, 8]` |
| Carboxyl dihedral (C6–C5–C7–O2) | `[6, 5, 10, 7]` |
| Ester dihedral (C5–C6–O3–C8) | `[5, 6, 12, 11]` |

## States and transitions
- Clustering: `grid` (default) or `dbscan`, set per pair in config; parameters come from
  config, never guessed.
- Clustering runs separately per coordinate pair and per `clustering.groupby` group.
  State labels are not unified across groups.
- PIMD bead identity is preserved unless an explicit aggregation step is configured.

## Architecture
Layers, each in its own module under `confana/`:
1. input adapters (`io_xyz`, `io_hdf5`, `io_coordinates`), and `frame_source`, which reads
   the frame a coordinate-table row names through `io_xyz` or `io_hdf5`
2. geometry and coordinate extraction (`geometry`, `coordinates`)
3. state assignment (`states`)
4. transition analysis (`transitions`)
5. static plots (`plots_static`)
6. interactive pages (`plots_interactive`, `viewer`, `payload_codec`, `interactive_assets/`)
7. CLI (`cli`)

- Downstream analysis works on the standard coordinate table (see `CONTEXT.md`), never
  on raw xyz frame objects.
- Plotting never depends on xyz parsing. Geometry imports no plotting code.
- Coordinate-only input (a table of angles) must run the same density, state and
  transition pipeline; structure lookup is then optional.

## xyz loading
- Streaming parser: never load a whole trajectory into memory.
- Each frame is `N + 2` lines: atom count, comment, `N` atom records.
- A one-pass scan builds a byte-offset frame index (`source_file`, `frame_number`,
  `byte_offset`, `atom_count`, `comment_line`), cached as `.frameindex.npz` and
  invalidated when the file path, size or modification time changes.
- Single-frame access seeks to `byte_offset` and reads exactly `N + 2` lines. The
  interactive viewer reuses the same index.

## Interactive pages
- Standalone HTML, no server. 3D rendering with 3Dmol.js (vendored); xyz parsing is the
  project's own streaming parser (no ASE).
- Clicking a bin shows its representative frame (nearest to the bin centre) with 3D
  structure and metadata; lookup is deterministic.
- Embedded metadata includes at least `source_file`, `frame_number`, `byte_offset`,
  `atom_count`, `trajectory_id`, `bead_id`, the coordinate values and the state labels.

## Scientific constraints
- Density maps are coordinate-density landscapes or population-derived free-energy-like
  surfaces, not potential energy surfaces, unless energies are provided.
- No physical rates without known frame ordering and `dt`; otherwise report counts and
  probabilities.
- Validate and document atom numbering; never assume a convention silently.

## Engineering rules
- Separate modules for I/O, geometry, clustering, transitions and plotting.
- Static figures: PNG at `dpi=300`.
- User-facing paths and analysis settings come from config. Dataset-specific assumptions
  are documented in config or README.
- Fail loudly on invalid atom indices, malformed xyz files and inconsistent structure sizes.
- Clear, typed, testable functions with docstrings; no hidden global state.
- Tests use small fixtures, not the full dataset.

## Working in this repo
- Inspect and summarise the current state before large edits.
- Small, reviewable patches; run the targeted tests after each module change.
- Update README and `docs/` when behaviour or interfaces change.
- If an open TODO blocks scientifically correct output, stop and mark the task blocked
  rather than guessing.

## Open TODOs
- Transition averaging across beads: exact outputs and uncertainty handling.
- Energy parsing from comment lines, and carrying energy through the fast loading path.
- Element symbols for HDF5 input (`TODO-hdf5-elements` in `confana/io_hdf5.py`).
- MD17 simulation temperature (not in the paper's main text): needed before barriers.

## Environment
```
source .venv/bin/activate
```
activates the environment where the package is installed for testing and development.
