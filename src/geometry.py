"""Geometry primitives for conformer analysis.

All functions accept a full coordinate array (shape N×3) plus a list of
0-based atom indices, so they compose naturally with the FrameRecord.coords
array without requiring index translation at call sites.

Design notes
------------
* SVD-based best-fit plane is numerically stable for any M ≥ 3 atoms.
* Plane normals are oriented deterministically from the ordered atom list.
* Dihedral angles use atan2 for correct quadrant and produce signed values
  in [-180, 180).
* Bond angles are returned in [0, 180] degrees.
* All public functions raise ``ValueError`` on invalid atom indices or
  degenerate geometry so that errors are caught early rather than silently
  propagating as NaN.

Public API
----------
- ``best_fit_plane``
- ``plane_normal``
- ``dihedral_angle``
- ``distance``
- ``bond_angle``
- ``batch_best_fit_plane``
- ``batch_dihedral_angle``
- ``batch_distance``
- ``batch_bond_angle``
- ``shift_angle``
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_atom_ids(
    coords: np.ndarray,
    atom_ids: Sequence[int],
    min_count: int,
    label: str = "atom_ids",
) -> None:
    """Raise ValueError if atom_ids are invalid for the given coords array.

    Checks:
    - ``len(atom_ids) >= min_count``
    - All ids are in [0, N-1] where N = coords.shape[0]

    Parameters
    ----------
    coords:
        Coordinate array, shape (N, 3).
    atom_ids:
        Sequence of 0-based indices.
    min_count:
        Minimum required number of indices.
    label:
        Name used in error messages.
    """
    if len(atom_ids) < min_count:
        raise ValueError(
            f"{label} must contain at least {min_count} indices; "
            f"got {len(atom_ids)}: {list(atom_ids)}"
        )
    n = coords.shape[0]
    bad = [i for i in atom_ids if not (0 <= i < n)]
    if bad:
        raise ValueError(
            f"{label} contains out-of-range indices {bad} "
            f"for coords with {n} atoms (valid range 0..{n - 1})"
        )


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v.

    Raises
    ------
    ValueError
        If ``||v|| < 1e-10`` (degenerate / zero vector), which indicates
        collinear or coincident atoms in the calling context.
    """
    norm = float(np.linalg.norm(v))
    if norm < 1e-10:
        raise ValueError(
            "Degenerate geometry: vector has near-zero length. "
            "Check for collinear or coincident atoms."
        )
    return v / norm


def _wrap_signed_degrees(values: "np.ndarray | float") -> np.ndarray:
    """Wrap degree values into the canonical ``[-180, 180)`` interval."""
    return ((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0


_SHIFT_MODES = frozenset({"signed", "dihedral"})


def shift_angle(
    values: "np.ndarray | float",
    shift: float,
    mode: str = "signed",
) -> np.ndarray:
    """Apply an angular shift and wrap back into the signed ``[-180, 180)`` domain.

    Parameters
    ----------
    values:
        Raw angle(s) in degrees.
    shift:
        Degrees to add (positive = right-hand rotation on number line).
    mode:
        ``"signed"`` or ``"dihedral"`` → wrap to ``[-180, 180)``.
        Both modes behave identically; ``"dihedral"`` is accepted as an alias
        for backward compatibility.

    Returns
    -------
    np.ndarray of the same shape as *values* (dtype float64).

    Raises
    ------
    ValueError
        If *mode* is not one of the supported strings.
    """
    if mode not in _SHIFT_MODES:
        raise ValueError(
            f"shift_angle: unsupported mode {mode!r}. "
            f"Valid choices: {sorted(_SHIFT_MODES)}"
        )
    shifted = np.asarray(values, dtype=float) + float(shift)
    return _wrap_signed_degrees(shifted)


def _plane_reference_normal(
    coords: np.ndarray,
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Return a deterministic reference normal from the ordered atom list.

    The first non-collinear ordered triple is used so the orientation remains
    stable even when the leading three atoms are nearly collinear.
    """
    _validate_atom_ids(coords, atom_ids, min_count=3, label="atom_ids")

    ids = list(atom_ids)
    n_ids = len(ids)
    for i in range(n_ids - 2):
        a = coords[ids[i]]
        for j in range(i + 1, n_ids - 1):
            b = coords[ids[j]]
            for k in range(j + 1, n_ids):
                c = coords[ids[k]]
                cross = np.cross(b - a, c - a)
                norm = float(np.linalg.norm(cross))
                if norm >= 1e-10:
                    return cross / norm

    raise ValueError(
        "Cannot build a reference plane normal from the ordered atom list: "
        f"all candidate triples are collinear for atom_ids={ids}."
    )


def _oriented_plane_normal(
    coords: np.ndarray,
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Return the best-fit plane normal oriented by the ordered atom list."""
    normal = best_fit_plane(coords, atom_ids)
    ref_normal = _plane_reference_normal(coords, atom_ids)
    if float(np.dot(normal, ref_normal)) < 0.0:
        normal = -normal
    return normal


# ---------------------------------------------------------------------------
# Best-fit plane
# ---------------------------------------------------------------------------


def best_fit_plane(coords: np.ndarray, atom_ids: Sequence[int]) -> np.ndarray:
    """Return the unit normal of the best-fit plane through the selected atoms.

    The plane is computed by SVD of the centroid-subtracted coordinate matrix.
    The normal corresponds to the right singular vector with the smallest
    singular value (least-squares fit minimising orthogonal distances).

    Parameters
    ----------
    coords:
        Full coordinate array, shape (N, 3), dtype float32.
    atom_ids:
        0-based indices of the atoms to use (at least 3).

    Returns
    -------
    np.ndarray
        Unit normal vector, shape (3,).

    Raises
    ------
    ValueError
        If fewer than 3 atoms are given, any index is out of range, or the
        selected atoms are collinear (degenerate plane).
    """
    _validate_atom_ids(coords, atom_ids, min_count=3, label="atom_ids")

    pts = coords[list(atom_ids)]            # shape (M, 3)
    centroid = pts.mean(axis=0)
    centred = pts - centroid                # shape (M, 3)

    # SVD: centred = U @ diag(s) @ Vh
    # The row of Vh with the smallest singular value is the plane normal.
    _, s, Vh = np.linalg.svd(centred, full_matrices=False)
    normal = Vh[-1]                         # shape (3,)

    # Collinearity check: if the second-largest singular value is near zero,
    # the atoms are (nearly) collinear and the plane is degenerate.
    s_max = float(s[0]) if len(s) > 0 else 0.0
    s_second = float(s[-2]) if len(s) >= 2 else 0.0
    if s_second < 1e-10 or (s_max > 0 and s_second < 1e-8 * s_max):
        raise ValueError(
            "Degenerate plane: atoms appear to be collinear or coincident "
            f"(singular values: {s}). Check atom_ids={list(atom_ids)}."
        )

    return _normalize(normal)


def plane_normal(coords: np.ndarray, atom_ids: Sequence[int]) -> np.ndarray:
    """Alias for ``best_fit_plane``.  Provided for API consistency."""
    return best_fit_plane(coords, atom_ids)


# ---------------------------------------------------------------------------
# Dihedral angle
# ---------------------------------------------------------------------------


def dihedral_angle(coords: np.ndarray, atom_ids: Sequence[int]) -> float:
    """Return the signed dihedral angle for four atoms.

    The dihedral is defined as the angle between plane(A, B, C) and
    plane(B, C, D) where the four atoms are given in order A, B, C, D.

    Implementation uses the atan2-based formula which gives the correct
    quadrant and stays in **[-180, 180)** by construction.

    Parameters
    ----------
    coords:
        Full coordinate array, shape (N, 3).
    atom_ids:
        Exactly 4 0-based indices: [A, B, C, D].

    Returns
    -------
    float
        Dihedral angle in degrees, range [-180, 180).

    Raises
    ------
    ValueError
        If ``len(atom_ids) != 4``, any index is out of range, or any bond
        vector has near-zero length (coincident atoms).
    """
    if len(atom_ids) != 4:
        raise ValueError(
            f"dihedral_angle requires exactly 4 atom indices; "
            f"got {len(atom_ids)}: {list(atom_ids)}"
        )
    _validate_atom_ids(coords, atom_ids, min_count=4, label="atom_ids")

    a, b, c, d = (coords[i] for i in atom_ids)

    b1 = b - a   # A→B
    b2 = c - b   # B→C
    b3 = d - c   # C→D

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 = _normalize(n1)
    n2 = _normalize(n2)
    b2_hat = _normalize(b2)

    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, b2_hat), n2)

    angle = np.degrees(np.arctan2(y, x))

    # atan2 returns (-180, 180]; clamp the +180 edge to keep [-180, 180).
    if angle >= 180.0:
        angle -= 360.0

    return float(angle)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def distance(coords: np.ndarray, atom_ids: Sequence[int]) -> float:
    """Return the Euclidean distance between two atoms in angstrom.

    Parameters
    ----------
    coords:
        Full coordinate array, shape (N, 3).
    atom_ids:
        Exactly 2 0-based indices: [A, B].

    Returns
    -------
    float
        Distance in the same units as *coords* (typically angstrom).

    Raises
    ------
    ValueError
        If ``len(atom_ids) != 2`` or any index is out of range.
    """
    if len(atom_ids) != 2:
        raise ValueError(
            f"distance requires exactly 2 atom indices; got {list(atom_ids)}"
        )
    _validate_atom_ids(coords, atom_ids, min_count=2, label="atom_ids")
    a, b = coords[atom_ids[0]], coords[atom_ids[1]]
    return float(np.linalg.norm(b - a))


# ---------------------------------------------------------------------------
# Bond angle
# ---------------------------------------------------------------------------


def bond_angle(coords: np.ndarray, atom_ids: Sequence[int]) -> float:
    """Return the bond angle A–B–C in degrees, range [0, 180].

    Parameters
    ----------
    coords:
        Full coordinate array, shape (N, 3).
    atom_ids:
        Exactly 3 0-based indices: [A, B, C].  B is the central atom.

    Returns
    -------
    float
        Bond angle in degrees, range [0, 180].

    Raises
    ------
    ValueError
        If ``len(atom_ids) != 3``, any index is out of range, or either
        bond vector has near-zero length (coincident atoms).
    """
    if len(atom_ids) != 3:
        raise ValueError(
            f"bond_angle requires exactly 3 atom indices; got {list(atom_ids)}"
        )
    _validate_atom_ids(coords, atom_ids, min_count=3, label="atom_ids")

    a, b, c = coords[atom_ids[0]], coords[atom_ids[1]], coords[atom_ids[2]]
    ba = _normalize(a - b)
    bc = _normalize(c - b)
    cos_angle = float(np.clip(np.dot(ba, bc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Batch geometry (vectorised over N frames)
# ---------------------------------------------------------------------------


def batch_best_fit_plane(
    coords: np.ndarray,
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Return best-fit plane unit normals for N frames simultaneously.

    Parameters
    ----------
    coords:
        Float array of shape ``(N, A, 3)`` where N is the number of frames
        and A is the total atom count.
    atom_ids:
        0-based indices selecting the atoms to use (at least 3 required).

    Returns
    -------
    np.ndarray
        Unit normal vectors, shape ``(N, 3)``, oriented deterministically by
        the ordered-atom-list rule (same as the scalar version).

    Raises
    ------
    ValueError
        If fewer than 3 atoms are given, any index is out of range, or any
        frame has collinear atoms (degenerate plane).
    """
    ids = list(atom_ids)
    if len(ids) < 3:
        raise ValueError(
            f"batch_best_fit_plane requires at least 3 atom indices; got {ids}"
        )
    n_atoms = coords.shape[1]
    bad = [i for i in ids if not (0 <= i < n_atoms)]
    if bad:
        raise ValueError(
            f"atom_ids contains out-of-range indices {bad} "
            f"for coords with {n_atoms} atoms (valid range 0..{n_atoms - 1})"
        )

    pts = coords[:, ids, :].astype(np.float64)            # (N, M, 3)
    centroid = pts.mean(axis=1, keepdims=True)             # (N, 1, 3)
    centered = pts - centroid                              # (N, M, 3)

    _, s, Vh = np.linalg.svd(centered, full_matrices=False)  # Vh: (N, min(M,3), 3)
    normals = Vh[:, -1, :]                                 # (N, 3)

    s_max = s[:, 0]
    s_second = s[:, -2] if s.shape[1] >= 2 else np.zeros(len(s))
    degenerate = (s_second < 1e-10) | ((s_max > 0) & (s_second < 1e-8 * s_max))
    if degenerate.any():
        bad_frames = np.where(degenerate)[0].tolist()
        raise ValueError(
            f"batch_best_fit_plane: degenerate (collinear) atoms in frames "
            f"{bad_frames[:5]}{'...' if len(bad_frames) > 5 else ''}. "
            f"atom_ids={ids}"
        )

    # Deterministic orientation via first non-collinear ordered triple
    ref = None
    n_ids = len(ids)
    for i in range(n_ids - 2):
        a = pts[:, i]
        for j in range(i + 1, n_ids - 1):
            b = pts[:, j]
            for k in range(j + 1, n_ids):
                c = pts[:, k]
                cross = np.cross(b - a, c - a)        # (N, 3)
                norms = np.linalg.norm(cross, axis=1)  # (N,)
                if (norms >= 1e-10).all():
                    ref = cross / norms[:, np.newaxis]
                    break
            if ref is not None:
                break
        if ref is not None:
            break

    if ref is None:
        raise ValueError(
            f"batch_best_fit_plane: all candidate triples are collinear "
            f"for atom_ids={ids}."
        )

    dots = (normals * ref).sum(axis=1)               # (N,)
    normals[dots < 0] *= -1

    return normals.astype(np.float32)


def batch_dihedral_angle(
    coords: np.ndarray,
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Return signed dihedral angles for N frames simultaneously.

    Parameters
    ----------
    coords:
        Float array of shape ``(N, A, 3)``.
    atom_ids:
        Exactly 4 0-based indices ``[a, b, c, d]``.

    Returns
    -------
    np.ndarray
        Signed dihedral angles in degrees, shape ``(N,)``, range ``[-180, 180)``.

    Raises
    ------
    ValueError
        If ``len(atom_ids) != 4``, any index is out of range, or any frame
        has a near-zero bond vector (coincident atoms).
    """
    if len(atom_ids) != 4:
        raise ValueError(
            f"batch_dihedral_angle requires exactly 4 atom indices; "
            f"got {len(atom_ids)}: {list(atom_ids)}"
        )
    ids = list(atom_ids)
    n_atoms = coords.shape[1]
    bad = [i for i in ids if not (0 <= i < n_atoms)]
    if bad:
        raise ValueError(
            f"atom_ids contains out-of-range indices {bad} "
            f"for coords with {n_atoms} atoms"
        )

    a = coords[:, ids[0]].astype(np.float64)   # (N, 3)
    b = coords[:, ids[1]].astype(np.float64)
    c = coords[:, ids[2]].astype(np.float64)
    d = coords[:, ids[3]].astype(np.float64)

    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)   # (N, 3)
    n2 = np.cross(b2, b3)

    n1_norms = np.linalg.norm(n1, axis=1)
    n2_norms = np.linalg.norm(n2, axis=1)
    b2_norms = np.linalg.norm(b2, axis=1)

    degenerate = (n1_norms < 1e-10) | (n2_norms < 1e-10) | (b2_norms < 1e-10)
    if degenerate.any():
        bad_frames = np.where(degenerate)[0].tolist()
        raise ValueError(
            f"batch_dihedral_angle: near-zero bond vector (coincident atoms) "
            f"in frames {bad_frames[:5]}{'...' if len(bad_frames) > 5 else ''}. "
            f"atom_ids={ids}"
        )

    n1 /= n1_norms[:, np.newaxis]
    n2 /= n2_norms[:, np.newaxis]
    b2_hat = b2 / b2_norms[:, np.newaxis]

    x = (n1 * n2).sum(axis=1)
    y = (np.cross(n1, b2_hat) * n2).sum(axis=1)

    angles = np.degrees(np.arctan2(y, x))
    angles[angles >= 180.0] -= 360.0
    return angles.astype(np.float32)


def batch_distance(
    coords: np.ndarray,
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Return Euclidean distances between two atoms for N frames.

    Parameters
    ----------
    coords:
        Float array of shape ``(N, A, 3)``.
    atom_ids:
        Exactly 2 0-based indices ``[A, B]``.

    Returns
    -------
    np.ndarray
        Distances, shape ``(N,)``, same units as *coords*.

    Raises
    ------
    ValueError
        If ``len(atom_ids) != 2`` or any index is out of range.
    """
    if len(atom_ids) != 2:
        raise ValueError(
            f"batch_distance requires exactly 2 atom indices; got {list(atom_ids)}"
        )
    ids = list(atom_ids)
    n_atoms = coords.shape[1]
    bad = [i for i in ids if not (0 <= i < n_atoms)]
    if bad:
        raise ValueError(
            f"atom_ids contains out-of-range indices {bad} "
            f"for coords with {n_atoms} atoms"
        )

    a = coords[:, ids[0]].astype(np.float64)   # (N, 3)
    b = coords[:, ids[1]].astype(np.float64)
    return np.linalg.norm(b - a, axis=1).astype(np.float32)   # (N,)


def batch_bond_angle(
    coords: np.ndarray,
    atom_ids: Sequence[int],
) -> np.ndarray:
    """Return bond angles A–B–C in degrees for N frames.

    Parameters
    ----------
    coords:
        Float array of shape ``(N, A, 3)``.
    atom_ids:
        Exactly 3 0-based indices ``[A, B, C]``.  B is the central atom.

    Returns
    -------
    np.ndarray
        Bond angles in degrees, shape ``(N,)``, range ``[0, 180]``.

    Raises
    ------
    ValueError
        If ``len(atom_ids) != 3``, any index is out of range, or any bond
        vector has near-zero length (coincident atoms).
    """
    if len(atom_ids) != 3:
        raise ValueError(
            f"batch_bond_angle requires exactly 3 atom indices; got {list(atom_ids)}"
        )
    ids = list(atom_ids)
    n_atoms = coords.shape[1]
    bad = [i for i in ids if not (0 <= i < n_atoms)]
    if bad:
        raise ValueError(
            f"atom_ids contains out-of-range indices {bad} "
            f"for coords with {n_atoms} atoms"
        )

    a = coords[:, ids[0]].astype(np.float64)   # (N, 3)
    b = coords[:, ids[1]].astype(np.float64)
    c = coords[:, ids[2]].astype(np.float64)

    ba = a - b   # (N, 3)
    bc = c - b

    ba_norms = np.linalg.norm(ba, axis=1)   # (N,)
    bc_norms = np.linalg.norm(bc, axis=1)

    degenerate = (ba_norms < 1e-10) | (bc_norms < 1e-10)
    if degenerate.any():
        bad_frames = np.where(degenerate)[0].tolist()
        raise ValueError(
            f"batch_bond_angle: near-zero bond vector (coincident atoms) "
            f"in frames {bad_frames[:5]}{'...' if len(bad_frames) > 5 else ''}. "
            f"atom_ids={ids}"
        )

    ba_hat = ba / ba_norms[:, np.newaxis]
    bc_hat = bc / bc_norms[:, np.newaxis]

    cos_angles = np.clip((ba_hat * bc_hat).sum(axis=1), -1.0, 1.0)   # (N,)
    return np.degrees(np.arccos(cos_angles)).astype(np.float32)       # (N,)
