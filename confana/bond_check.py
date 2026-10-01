"""Bond-break detection on coordinate arrays.

Finds the first frame where an atom pair bonded in a reference frame is
farther apart than a distance cutoff. Pure numpy — no ASE, no file I/O; the
coordinate-table build (``confana.coordinate_table``) feeds it the frames.
"""
from __future__ import annotations

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


def bonded_pairs(
    coords: np.ndarray,
    elements: list[str],
    mult: float = 1.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bonded atom pairs of one reference frame as index arrays ``(i, j)``.

    Same bond rule as :func:`build_bond_graph`.
    """
    bonds = build_bond_graph(coords, elements, mult)
    idx_i = np.array([b[0] for b in bonds], dtype=np.int32)
    idx_j = np.array([b[1] for b in bonds], dtype=np.int32)
    return idx_i, idx_j


def first_broken_frame(
    coords: np.ndarray,
    bonds: tuple[np.ndarray, np.ndarray],
    cutoff: float,
) -> Optional[int]:
    """Return the position of the first frame in which a bond is broken, or None.

    A bond is broken when its two atoms are more than *cutoff* Å apart.

    Parameters
    ----------
    coords:
        float32 coordinates, shape ``(n_frames, n_atoms, 3)``.
    bonds:
        Bonded pairs of the reference frame, from :func:`bonded_pairs`. With
        no bonds nothing can break, so the result is None.
    cutoff:
        Distance threshold in Å.
    """
    idx_i, idx_j = bonds
    if len(idx_i) == 0 or len(coords) == 0:
        return None
    coords = coords.astype(np.float32, copy=False)
    diffs = coords[:, idx_i] - coords[:, idx_j]
    max_dists = np.sqrt((diffs * diffs).sum(axis=-1)).max(axis=1)
    broken = np.flatnonzero(max_dists > cutoff)
    return int(broken[0]) if len(broken) else None
