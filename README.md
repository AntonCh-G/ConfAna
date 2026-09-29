# ConfAna

Conformer analysis for molecular trajectories. ConfAna reads multi-frame xyz files
(or a table of precomputed angles), computes carboxyl and ester angles for every frame,
and produces density maps, conformational states, state-to-state transitions, and
standalone interactive HTML pages with clickable 3D structures.

## Install

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Demo

**[docs/demo/density_dihedral.html](docs/demo/density_dihedral.html)** (8 MB): download it
(on GitHub, the **Download raw file** button) and open it in any browser. It works offline.

It shows the public MD17 aspirin trajectory (211,762 frames). Hover the map to see structures,
and click a bin to pin one.

## Quick start: MD17 aspirin

[MD17](http://www.sgdml.org/#datasets) is a public set of ab initio molecular dynamics
trajectories. Its aspirin trajectory has 211,762 time-ordered frames, 0.5 fs apart.

```bash
python scripts/md17_to_xyz.py                                  # download (193 MB) and convert to xyz
confana build-interactive --config examples/md17_aspirin.yaml
```

Result (about 10 seconds after the download):

```text
outputs/md17_aspirin/plots/density_dihedral.html   carboxyl vs ester dihedral
```

Open it in a browser. No server or internet needed. This is the page in `docs/demo/`.

The script checks the download against a fixed SHA-256 checksum and writes
`data/md17/md17_aspirin.xyz`. The `data/` folder is git-ignored; MD17 is not redistributed here.

The key parts of [examples/md17_aspirin.yaml](examples/md17_aspirin.yaml):

```yaml
run_dir: outputs/md17_aspirin
data:
  path_pattern: ./data/md17/md17_aspirin.xyz

dof:                                       # angles to compute, 0-based atom indices
  - {name: carboxyl_dihedral, type: dihedral, atoms: [6, 5, 10, 7]}
  - {name: ester_dihedral,    type: dihedral, atoms: [5, 6, 12, 11]}

coordinate_pairs:                          # one density map + HTML page per pair
  - {name: dihedral, x: carboxyl_dihedral, y: ester_dihedral}
```

## PIMD trajectories

For path-integral MD, give one xyz file per bead (for example `my_run.pos_00.xyz` …
`my_run.pos_01.xyz`). ConfAna reads the trajectory and bead IDs from the file names and keeps
beads separate for transitions. See [examples/pimd_template.yaml](examples/pimd_template.yaml).

## Using the interactive page

| Action | What happens |
|---|---|
| Hover a bin | 3D structure preview; side panel shows angles, count, state |
| Click "More info" | Full frame metadata: source file, frame number, byte offset, indices |
| Click a bin | Pins that frame as a numbered card and map marker |
| Esc or double-click | Clears all pins |
| Drag any 3D view | Rotates all views together |
| Header links | Switch pair pages; pins come along |
| Scale buttons | `log counts`, `counts`, or `free-energy-like` |

Details: [docs/interactive-viewer.md](docs/interactive-viewer.md).

## Commands

| Command | Does |
|---|---|
| `confana extract-coordinates` | xyz → coordinate table (CSV) |
| `confana plot-densities` | density PNGs (300 dpi) |
| `confana cluster-states` | assigns a state to every frame |
| `confana compute-transitions` | transition counts and probabilities (rates if `dt` is set) |
| `confana build-interactive` | standalone HTML pages |
| `confana run-all` | all of the above |

All take `--config <file.yaml>`.

## Read the results correctly

- **Density, not energy.** Maps come from frame counts. "Free-energy-like" means
  `−kT·ln(population)`, not a potential energy surface.
- **Rates need time.** Physical rates appear only if `transitions.dt` is set.
  Barriers need both `dt` and `temperature`.
- **PIMD beads stay separate.** Transitions are computed per bead, then averaged.
- **Atom indices are 0-based file positions**, not chemical labels.

## Atom mapping (aspirin)

These indices follow the MD17 aspirin atom order.

| Coordinate | Atoms (0-based) |
|---|---|
| Ring plane (best fit, C1–C6) | `[0, 1, 2, 3, 5, 6]` |
| Carboxyl plane | `[9, 10, 7]` |
| Ester plane | `[12, 11, 8]` |
| Carboxyl dihedral | `[6, 5, 10, 7]` |
| Ester dihedral | `[5, 6, 12, 11]` |

## Tests

```bash
pytest                                              # all tests
python -m playwright install chromium --no-shell firefox   # once, for browser tests
```

## Data credit

The demo uses the MD17 aspirin trajectory. If you publish results made from it, cite:

> S. Chmiela, A. Tkatchenko, H. E. Sauceda, I. Poltavsky, K. T. Schütt, K.-R. Müller,
> "Machine learning of accurate energy-conserving molecular force fields",
> *Sci. Adv.* **3**, e1603015 (2017). https://doi.org/10.1126/sciadv.1603015

## More docs

- [docs/configuration.md](docs/configuration.md): every config section, rates and barriers
- [docs/interactive-viewer.md](docs/interactive-viewer.md): viewer behaviour and settings
- [docs/development.md](docs/development.md): code layout and testing
- [docs/adr/](docs/adr/): design decisions
