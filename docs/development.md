# Development

## Repository Layout

- `confana/geometry.py`: reusable geometry primitives
- `confana/io_xyz.py`: streaming XYZ ingestion, indexed random access, and the xyz trajectory reader
- `confana/io_hdf5.py`: the HDF5 PIMD trajectory reader and single-frame reading (`HDF5FrameReader`)
- `confana/frame_source.py`: reads the frame a coordinate-table row names (xyz by byte offset, HDF5 by frame and bead)
- `confana/io_coordinates.py`: coordinate-table files (CSV, Parquet, NPZ) and schema checks
- `confana/coordinate_table.py`: builds the coordinate table from xyz or HDF5 trajectories, one pipeline and one per-trajectory cache for every format (`load_or_build_coordinate_table_from_config`)
- `confana/coordinates.py`: plane/dihedral coordinate extraction
- `confana/density.py`: the conformational map of a coordinate pair (bins, counts, bin lookup, representative frames) and the population free-energy surface
- `confana/states.py`: conformational state assignment
- `confana/transitions.py`: transition counts, probabilities, optional rates, and optional activation barriers
- `confana/plots_static.py`: PNG density and transition plots
- `confana/plots_interactive.py`: standalone interactive HTML outputs
- `confana/viewer.py`: reading and aligning the structures and metadata of each bin's representative frame
- `confana/payload_codec.py`: compact encoding of the structures and metadata embedded in the HTML
- `confana/cli.py`: command-line entrypoints
- `configs/`: local workflow configurations ignored by git

## Testing

Run the test suite with:

```bash
pytest
```

The repository includes tests for geometry, XYZ ingestion, coordinate-table building and loading, clustering, transitions, caching, viewer helpers, and interactive plot generation.

`tests/test_interactive_browser.py` opens generated HTML pages in headless Chromium,
offline, and drives them with real mouse and keyboard input (hover preview, pinning,
double-click, Esc, theme toggle). Others open the pages as an emulated phone (390 px wide,
touch, no hover) and tap and swipe. A few tests also run in Firefox, where 3Dmol behaves
differently. It needs a one-off browser install:

```bash
python -m playwright install chromium --no-shell firefox
```

Without it these tests are skipped. To skip them explicitly: `pytest -m "not browser"`.
