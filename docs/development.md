# Development

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

## Testing

Run the test suite with:

```bash
pytest
```

The repository includes tests for geometry, XYZ ingestion, coordinate-table loading, clustering, transitions, caching, viewer helpers, and interactive plot generation.

`tests/test_interactive_browser.py` opens generated HTML pages in headless Chromium,
offline, and drives them with real mouse and keyboard input (hover preview, pinning,
double-click, Esc, theme toggle). A few tests also run in Firefox, where 3Dmol behaves
differently. It needs a one-off browser install:

```bash
python -m playwright install chromium --no-shell firefox
```

Without it these tests are skipped. To skip them explicitly: `pytest -m "not browser"`.
