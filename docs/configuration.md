# Configuration

Full reference for ConfAna's YAML config. For a working minimal config see
[examples/md17_aspirin.yaml](../examples/md17_aspirin.yaml) (one trajectory) or
[examples/pimd_template.yaml](../examples/pimd_template.yaml) (one file per PIMD bead); for every option with comments, keep a
local copy such as `configs/default.yaml` (the `configs/` folder is git-ignored).

## Config sections

The workflow is driven by YAML config. Create or update a local config under `configs/`, for example [configs/default.yaml](../configs/default.yaml).

Important sections include:

- `data`: input path pattern plus trajectory and bead ID extraction rules
- `dof`: named degrees of freedom (dihedral, distance, angle) with 0-based atom indices
- `coordinate_transforms`: optional angle shifts (adds `<name>_shifted` columns)
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
