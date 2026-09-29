# Interactive viewer

How the standalone HTML pages behave and which `plots.interactive` settings control them.

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
  with repeated strings stored once, and both blocks gzipped. Before gzip, coordinates are
  reordered losslessly (each atom's x, y, z for all structures side by side, then low
  bytes before high bytes), which makes the structure block about 25 % smaller. The page unpacks them once
  on load with the browser's own `DecompressionStream`, needing no library and no internet;
  a browser without it says so in the side panel. Set `compress_payloads: false` to embed
  plain JSON for debugging. The build fails loudly if the structures do not share one
  element sequence, or if a coordinate needs more than 16 bits at the chosen step.
- The colour-scale data is embedded once, as the frame count per bin (raw 16- or 32-bit
  integers). The page computes the log-count and free-energy-like grids from it with the
  same formulas and 4-decimal rounding as Python. The heatmap inside the Plotly figure is
  stored as float32 (about 7 significant digits, far more than the hover shows).
- The vendored 3Dmol.js copy and its BSD license live under `confana/interactive_assets/vendor/`.
- Hovering the map previews the bin under the cursor in the side panel
  (`interactive.hover_preview`, default `true`). Clicking a bin pins its representative
  frame (the frame closest to the bin centre); Esc or double-click clears pins. At most
  `interactive.max_pinned` pins (default 15) are kept: pinning more removes the oldest.
  - Each pin has a number, shown both on its card and on the map. On the map, a numbered
    badge with a short arrow points at the frame's exact coordinates, and the bin it falls
    in is outlined (visible once you zoom in). In per-frame mode
    (`embed_xyz_payload: false`) the marker is a ring around the frame's point.
  - A new pin takes the lowest free number, so closing pin 2 frees number 2. Cards are
    listed by number.
  - The hover preview always shows the bin's coordinates, value, count and state. The
    frame's full metadata (source file, frame number, byte offset, trajectory and frame
    indices, every coordinate) sits behind a collapsed "More info" line, in the preview
    and on each card; click it to expand.
  - Hovering a card makes its badge bold and fades the others. Hovering a pinned bin on
    the map outlines its card. Clicking a badge scrolls to its card.
  - All 3D views share one camera. Turning or zooming the hover preview or any card
    turns and zooms every view, and a new pin opens at the current angle, so pinned
    structures line up with the preview and with each other. Each view keeps its own
    structure centred.
  - A pin is a frame, not a bin. When you switch pair pages with the header links, the
    pins come along, carried in the link (`#pins=…`). Each page places a pin at that
    frame's own coordinates in its pair, keeping its number and structure. A pin with no
    value for a page's pair keeps its card there, marked "Not on this map". A page opened
    on its own, not through a header link, starts with no pins. See
    `docs/adr/0002-pins-are-frames-carried-in-link-hash.md`.
- Every embedded structure is rigidly rotated onto one reference (the earliest frame,
  fitted on heavy atoms), so structures from different bins face the same way and can be
  compared. This is on by default; set `plots.interactive.alignment.enabled: false` to show
  each structure in its raw trajectory orientation. Only the display copy is rotated;
  angles are computed from the original coordinates.
- Every 3D view draws atoms in their element colours (Jmol scheme: C grey, O red, H
  white). The atoms that define each axis are drawn as balls with thicker bonds, inside a
  see-through halo: orange for the x axis, blue for the y axis, pink for atoms shared by
  both. The axis titles use the same colours, and a legend in the side panel lists the
  atom indices (0-based file indices, from the pair's `dof` entries). A checkbox in the
  legend labels every atom in the 3D views with its index, to check the mapping. The build fails if an index is
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
  - The `k_B` values per unit live in one table in `confana/units.py`, which the page embeds.
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
