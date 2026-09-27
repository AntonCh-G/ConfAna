# ADR 0001 — Scatter overlay coordinates computed by caller, not by `make_density_png`

**Status:** Accepted

## Context

When rendering scatter overlays from external xyz files onto a density PNG, two integration points were considered:

**Option A (chosen):** The caller (CLI) loads each overlay xyz file and computes the dihedral coordinate table using the same geometry pipeline as the main dataset. `make_density_png` receives a `overlays: list[dict]` parameter where each dict carries a pre-computed `pd.DataFrame`, a label, and a color.

**Option B (rejected):** `make_density_png` reads file paths from config and runs IO + geometry itself.

## Decision

Option A. Plotting functions accept coordinate data; they do not load files or invoke geometry code.

## Consequences

- `make_density_png` stays testable in isolation with synthetic DataFrames.
- The existing architecture boundary between input adapters / geometry (layers 1–2) and static plotting (layer 5) is preserved.
- CLI is responsible for the load → compute → plot wiring, consistent with how the main coordinate table is produced.
- Future coordinate-only input mode (precomputed tables) can supply overlay DataFrames directly without touching the plot function.
