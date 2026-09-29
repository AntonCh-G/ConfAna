Vendored third-party library
============================

File:    3Dmol-min.js
Library: 3Dmol.js
Version: 2.5.5
Source:  https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js
         (published to npm as package "3dmol"; upstream project
         https://github.com/3dmol/3Dmol.js)
Fetched: 2026-09-27
License: BSD-3-Clause, see LICENSE-3dmol.txt in this directory
         (fetched from https://raw.githubusercontent.com/3dmol/3Dmol.js/2.5.5/LICENSE)

Used by src/plots_interactive.py (render_density_page) to inline a
standalone copy of 3Dmol.js into generated interactive HTML files when
plots.interactive.include_3dmol is "inline" (the default), so the
generated page has no runtime dependency on an internet connection.
Set include_3dmol: cdn to load this same pinned version from jsDelivr
instead (see _VENDORED_3DMOL_CDN_URL in src/plots_interactive.py).

To update: bump the version number in the two URLs above, re-download
both files, and update _VENDORED_3DMOL_VERSION in src/plots_interactive.py
to match.
