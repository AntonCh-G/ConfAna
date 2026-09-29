"""Bond-break detection for xyz trajectories.

Detects the first frame where an initially-bonded atom pair exceeds a distance
cutoff.  Uses ConfAna's own xyz I/O and numpy — no ASE required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Covalent radii (Å) — Alvarez 2008, Dalton Trans. 2832.
_COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
    "Br": 1.20, "I": 1.39, "Se": 1.20, "Te": 1.38,
}
_FALLBACK_RADIUS = 0.77


def build_bond_graph(
    coords: np.ndarray,
    elements: list[str],
    mult: float = 1.1,
) -> list[tuple[int, int]]:
    """Return bonded pairs (i, j) from a single frame.

    A pair is bonded if their distance is less than
    (covalent_radius_i + covalent_radius_j) * mult.
    """
    n = len(elements)
    bonds: list[tuple[int, int]] = []
    for i in range(n):
        r_i = _COVALENT_RADII.get(elements[i], _FALLBACK_RADIUS)
        for j in range(i + 1, n):
            r_j = _COVALENT_RADII.get(elements[j], _FALLBACK_RADIUS)
            threshold = (r_i + r_j) * mult
            d = coords[i] - coords[j]
            if float(np.sqrt(float(np.dot(d, d)))) < threshold:
                bonds.append((i, j))
    return bonds


def find_bond_break_frame(
    source_file: Path,
    bond_cutoff: float = 2.0,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Optional[int]:
    """Return the local frame index of the first bond break, or None.

    Streams the file sequentially (one pass, no seeks) for efficiency on
    network/parallel file systems.  The bond graph is built from the first
    yielded frame (``start_frame``); each subsequent frame within the range
    is checked against that fixed graph.

    Parameters
    ----------
    source_file:
        Path to the xyz trajectory file.
    bond_cutoff:
        Distance threshold in Å above which a bond is considered broken.
    start_frame:
        First frame to include in the scan (0-based, inclusive).  The
        reference bond graph is built from this frame.  Default 0.
    end_frame:
        Last frame to include in the scan (0-based, inclusive).  ``None``
        means scan to end of file.
    """
    from confana.io_xyz import iter_xyz_frames  # noqa: PLC0415

    max_frames = (end_frame + 1) if end_frame is not None else None

    idx_i: Optional[np.ndarray] = None
    idx_j: Optional[np.ndarray] = None
    reference_frame_set = False

    for frame in iter_xyz_frames(source_file, start_frame=start_frame, max_frames=max_frames):
        if not reference_frame_set:
            bonds = build_bond_graph(frame.coords, frame.elements)
            if not bonds:
                return None
            idx_i = np.array([b[0] for b in bonds], dtype=np.int32)
            idx_j = np.array([b[1] for b in bonds], dtype=np.int32)
            reference_frame_set = True
            continue

        coords = frame.coords.astype(np.float32)
        diffs = coords[idx_i] - coords[idx_j]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        if float(dists.max()) > bond_cutoff:
            return frame.local_frame_index

    return None
