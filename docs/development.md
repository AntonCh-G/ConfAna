# Development

## Repository Layout

- `confana/geometry.py`: reusable geometry primitives
- `confana/io_xyz.py`: streaming XYZ ingestion and indexed random access
- `confana/io_coordinates.py`: coordinate-table ingestion and cache-aware loading
- `confana/coordinates.py`: plane/dihedral coordinate extraction
- `confana/states.py`: conformational state assignment
- `confana/transitions.py`: transition counts, probabilities, optional rates, and optional activation barriers
- `confana/plots_static.py`: PNG density and transition plots
- `confana/plots_interactive.py`: standalone interactive HTML outputs
- `confana/viewer.py`: nearest-structure lookup and structure rendering helpers
- `confana/payload_codec.py`: compact encoding of the structures and metadata embedded in the HTML
- `confana/cli.py`: command-line entrypoints
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
