"""Tests for src/geometry.py."""

from __future__ import annotations



import numpy as np
import pytest

from src.geometry import (
    _normalize,
    _validate_atom_ids,
    batch_best_fit_plane,
    batch_bond_angle,
    batch_dihedral_angle,
    batch_distance,
    best_fit_plane,
    bond_angle,
    dihedral_angle,
    distance,
    plane_normal,
    shift_angle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_coords(*points: tuple) -> np.ndarray:
    """Build an (N, 3) float32 array from a sequence of (x, y, z) tuples."""
    return np.array(points, dtype=np.float32)


def _build_dihedral_pair(
    a: np.ndarray,
    b: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return points C and D so dihedral(A, B, C, D) equals angle_deg."""
    u = a - b
    u = u / np.linalg.norm(u)

    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(u, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)

    v = np.cross(u, helper)
    v = v / np.linalg.norm(v)
    w = np.cross(v, u)
    w = w / np.linalg.norm(w)

    theta = np.radians(angle_deg)
    c = b + v
    d = c + np.cos(theta) * u - np.sin(theta) * w
    return c, d


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_unit_vector():
    v = np.array([3.0, 4.0, 0.0])
    n = _normalize(v)
    assert np.isclose(np.linalg.norm(n), 1.0, rtol=1e-12)
    assert np.isclose(n[0], 0.6, rtol=1e-12)
    assert np.isclose(n[1], 0.8, rtol=1e-12)


def test_normalize_zero_raises():
    with pytest.raises(ValueError, match="near-zero"):
        _normalize(np.array([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# _validate_atom_ids
# ---------------------------------------------------------------------------


def test_validate_atom_ids_too_few():
    coords = make_coords((0, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="at least 4"):
        _validate_atom_ids(coords, [0, 1], min_count=4)


def test_validate_atom_ids_out_of_range():
    coords = make_coords((0, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="out-of-range"):
        _validate_atom_ids(coords, [0, 1, 5], min_count=3)


def test_validate_atom_ids_valid():
    coords = make_coords((0, 0, 0), (1, 0, 0), (0, 1, 0))
    # Should not raise
    _validate_atom_ids(coords, [0, 1, 2], min_count=3)


# ---------------------------------------------------------------------------
# best_fit_plane / plane_normal
# ---------------------------------------------------------------------------


def test_best_fit_plane_xy_plane():
    """Three points in the XY plane → normal should be ≈ [0, 0, ±1]."""
    coords = make_coords(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
    )
    n = best_fit_plane(coords, [0, 1, 2, 3])
    assert abs(abs(n[2]) - 1.0) < 1e-10, f"Normal z component should be ±1, got {n}"
    assert abs(n[0]) < 1e-10
    assert abs(n[1]) < 1e-10


def test_best_fit_plane_xz_plane():
    """Points in XZ plane → normal should be ≈ [0, ±1, 0]."""
    coords = make_coords(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    n = best_fit_plane(coords, [0, 1, 2])
    assert abs(abs(n[1]) - 1.0) < 1e-10, f"Normal y component should be ±1, got {n}"


def test_plane_normal_alias():
    coords = make_coords(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    n1 = best_fit_plane(coords, [0, 1, 2])
    n2 = plane_normal(coords, [0, 1, 2])
    np.testing.assert_array_almost_equal(np.abs(n1), np.abs(n2))


def test_best_fit_plane_fewer_than_3_raises():
    coords = make_coords((0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="at least 3"):
        best_fit_plane(coords, [0, 1])


def test_best_fit_plane_collinear_raises():
    """Collinear points → degenerate plane → should raise ValueError."""
    coords = make_coords(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        best_fit_plane(coords, [0, 1, 2])


def test_best_fit_plane_six_ring_atoms():
    """Six atoms roughly in a plane → normal approximately orthogonal to that plane."""
    # Regular hexagon in XY plane, slightly perturbed
    angles_deg = [0, 60, 120, 180, 240, 300]
    pts = [(np.cos(np.radians(a)), np.sin(np.radians(a)), 0.01 * i)
           for i, a in enumerate(angles_deg)]
    coords = make_coords(*pts)
    n = best_fit_plane(coords, list(range(6)))
    # z component should dominate (close to ±1)
    assert abs(n[2]) > 0.99, f"Expected normal close to z-axis, got {n}"


# ---------------------------------------------------------------------------
# dihedral_angle
# ---------------------------------------------------------------------------


def test_dihedral_angle_180():
    """Trans (anti) conformation → dihedral = ±180°."""
    # A–B–C–D in a straight line with the last atom flipped
    coords = make_coords(
        (0.0, 1.0, 0.0),   # A
        (0.0, 0.0, 0.0),   # B
        (1.0, 0.0, 0.0),   # C
        (1.0, -1.0, 0.0),  # D  (trans — all in the same plane, anti)
    )
    angle = dihedral_angle(coords, [0, 1, 2, 3])
    assert np.isclose(abs(angle), 180.0, atol=1e-8) or np.isclose(angle, 180.0, atol=1e-8)


def test_dihedral_angle_0():
    """Cis (eclipsed) conformation → dihedral = 0°."""
    coords = make_coords(
        (0.0, 1.0, 0.0),   # A
        (0.0, 0.0, 0.0),   # B
        (1.0, 0.0, 0.0),   # C
        (1.0, 1.0, 0.0),   # D  (cis — A and D on the same side)
    )
    angle = dihedral_angle(coords, [0, 1, 2, 3])
    assert np.isclose(angle, 0.0, atol=1e-8)


def test_dihedral_angle_90():
    """Gauche conformation → dihedral = ±90°."""
    coords = make_coords(
        (0.0, 1.0, 0.0),   # A (in XY plane)
        (0.0, 0.0, 0.0),   # B
        (1.0, 0.0, 0.0),   # C
        (1.0, 0.0, 1.0),   # D (perpendicular — 90° dihedral)
    )
    angle = dihedral_angle(coords, [0, 1, 2, 3])
    assert np.isclose(abs(angle), 90.0, atol=1e-8)


def test_dihedral_angle_wrong_count():
    coords = make_coords((0, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="exactly 4"):
        dihedral_angle(coords, [0, 1, 2])


def test_dihedral_angle_in_range():
    """Dihedral must be in [-180, 180)."""
    rng = np.random.default_rng(99)
    coords = rng.standard_normal((6, 3))
    for _ in range(30):
        ids = rng.choice(6, size=4, replace=False).tolist()
        try:
            angle = dihedral_angle(coords, ids)
            assert -180.0 <= angle < 180.0, f"Dihedral {angle} out of [-180, 180)"
        except ValueError:
            pass  # degenerate geometry — acceptable


def test_dihedral_angle_symmetry():
    """Reversing the atom order gives the same dihedral value (not negated).

    With the atan2 formula used here, dihedral(A,B,C,D) = dihedral(D,C,B,A).
    Both orderings describe the same four-atom geometry.
    """
    coords = make_coords(
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
    )
    fwd = dihedral_angle(coords, [0, 1, 2, 3])
    rev = dihedral_angle(coords, [3, 2, 1, 0])
    assert np.isclose(fwd, rev, atol=1e-8), f"fwd={fwd}, rev={rev}"


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------


def test_distance_known_value():
    """Distance between (0,0,0) and (3,4,0) is 5."""
    coords = make_coords((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
    assert np.isclose(distance(coords, [0, 1]), 5.0, atol=1e-6)


def test_distance_self_is_zero():
    """Distance from an atom to itself is 0."""
    coords = make_coords((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert np.isclose(distance(coords, [0, 0]), 0.0, atol=1e-10)


def test_distance_wrong_count_raises():
    coords = make_coords((0, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="exactly 2"):
        distance(coords, [0, 1, 2])


def test_distance_out_of_range_raises():
    coords = make_coords((0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="out-of-range"):
        distance(coords, [0, 5])


def test_distance_symmetric():
    """distance(A, B) == distance(B, A)."""
    coords = make_coords((1.0, 2.0, 3.0), (4.0, 6.0, 3.0))
    assert np.isclose(distance(coords, [0, 1]), distance(coords, [1, 0]), atol=1e-10)


def test_distance_returns_float():
    coords = make_coords((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    result = distance(coords, [0, 1])
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# bond_angle
# ---------------------------------------------------------------------------


def test_bond_angle_right_angle():
    """90° bond angle: A–B–C where A is along x, C is along y."""
    coords = make_coords(
        (1.0, 0.0, 0.0),  # A
        (0.0, 0.0, 0.0),  # B (central)
        (0.0, 1.0, 0.0),  # C
    )
    assert np.isclose(bond_angle(coords, [0, 1, 2]), 90.0, atol=1e-6)


def test_bond_angle_linear():
    """180° bond angle: A–B–C collinear."""
    coords = make_coords(
        (-1.0, 0.0, 0.0),  # A
        (0.0, 0.0, 0.0),   # B (central)
        (1.0, 0.0, 0.0),   # C
    )
    assert np.isclose(bond_angle(coords, [0, 1, 2]), 180.0, atol=1e-6)


def test_bond_angle_60_degrees():
    """60° bond angle for equilateral triangle vertex."""
    coords = make_coords(
        (0.0, 0.0, 0.0),                       # A
        (1.0, 0.0, 0.0),                        # B (central)
        (0.5, np.sqrt(3) / 2, 0.0),             # C
    )
    assert np.isclose(bond_angle(coords, [0, 1, 2]), 60.0, atol=1e-5)


def test_bond_angle_in_range():
    """Bond angles must be in [0, 180]."""
    rng = np.random.default_rng(77)
    coords = rng.standard_normal((6, 3))
    for _ in range(20):
        ids = rng.choice(6, size=3, replace=False).tolist()
        try:
            angle = bond_angle(coords, ids)
            assert 0.0 <= angle <= 180.0, f"Bond angle {angle} out of [0, 180]"
        except ValueError:
            pass  # degenerate geometry — acceptable


def test_bond_angle_wrong_count_raises():
    coords = make_coords((0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="exactly 3"):
        bond_angle(coords, [0, 1])


def test_bond_angle_out_of_range_raises():
    coords = make_coords((0, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="out-of-range"):
        bond_angle(coords, [0, 1, 99])


def test_bond_angle_returns_float():
    coords = make_coords(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    result = bond_angle(coords, [0, 1, 2])
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Batch geometry — regression tests against scalar versions
# ---------------------------------------------------------------------------


# Shared fixture: N random coordinate sets (N, A, 3)
_N_BATCH = 50
_N_ATOMS = 15
_RNG_BATCH = np.random.default_rng(2024)
_BATCH_COORDS = _RNG_BATCH.standard_normal((_N_BATCH, _N_ATOMS, 3)).astype(np.float32)

# Fixed atom index sets that are valid for 15 atoms
_RING_IDS = [0, 1, 2, 3, 5, 6]
_PLANE3_IDS = [7, 9, 10]
_DIHEDRAL_IDS = [6, 5, 10, 7]
_DIST_IDS = [0, 1]
_ANGLE_IDS = [0, 1, 2]


def test_batch_best_fit_plane_matches_scalar():
    """batch_best_fit_plane must agree with best_fit_plane for every frame."""
    batch = batch_best_fit_plane(_BATCH_COORDS, _RING_IDS)
    for i in range(_N_BATCH):
        scalar = best_fit_plane(_BATCH_COORDS[i], _RING_IDS)
        # normals may have opposite sign if scalar and batch orient differently;
        # compare by checking |dot| ≈ 1 (parallel or anti-parallel unit vectors)
        dot = float(np.dot(batch[i], scalar))
        assert abs(abs(dot) - 1.0) < 1e-4, (
            f"frame {i}: batch normal {batch[i]} not parallel to scalar normal {scalar}"
        )


def test_batch_dihedral_angle_matches_scalar():
    """batch_dihedral_angle must agree with dihedral_angle within float32 tolerance."""
    batch = batch_dihedral_angle(_BATCH_COORDS, _DIHEDRAL_IDS)
    for i in range(_N_BATCH):
        scalar = dihedral_angle(_BATCH_COORDS[i].astype(np.float32), _DIHEDRAL_IDS)
        assert np.isclose(batch[i], scalar, atol=1e-3), (
            f"frame {i}: batch={batch[i]:.4f} scalar={scalar:.4f}"
        )


def test_batch_dihedral_angle_range():
    """All batch dihedral results must be in [-180, 180)."""
    batch = batch_dihedral_angle(_BATCH_COORDS, _DIHEDRAL_IDS)
    assert (batch >= -180.0).all() and (batch < 180.0).all(), (
        f"Out-of-range values: {batch[~((batch >= -180) & (batch < 180))]}"
    )


def test_batch_distance_matches_scalar():
    """batch_distance must agree with distance for every frame."""
    batch = batch_distance(_BATCH_COORDS, _DIST_IDS)
    for i in range(_N_BATCH):
        scalar = distance(_BATCH_COORDS[i].astype(float), _DIST_IDS)
        assert np.isclose(float(batch[i]), scalar, atol=1e-4), (
            f"frame {i}: batch={batch[i]:.6f} scalar={scalar:.6f}"
        )


def test_batch_distance_non_negative():
    """Distances are always non-negative."""
    batch = batch_distance(_BATCH_COORDS, _DIST_IDS)
    assert (batch >= 0.0).all()


def test_batch_distance_wrong_count_raises():
    with pytest.raises(ValueError, match="exactly 2"):
        batch_distance(_BATCH_COORDS, [0, 1, 2])


def test_batch_bond_angle_matches_scalar():
    """batch_bond_angle must agree with bond_angle for every frame."""
    batch = batch_bond_angle(_BATCH_COORDS, _ANGLE_IDS)
    for i in range(_N_BATCH):
        scalar = bond_angle(_BATCH_COORDS[i].astype(float), _ANGLE_IDS)
        assert np.isclose(float(batch[i]), scalar, atol=1e-3), (
            f"frame {i}: batch={batch[i]:.4f} scalar={scalar:.4f}"
        )


def test_batch_bond_angle_in_range():
    """All batch bond angles must be in [0, 180]."""
    batch = batch_bond_angle(_BATCH_COORDS, _ANGLE_IDS)
    assert (batch >= 0.0).all() and (batch <= 180.0).all()


def test_batch_bond_angle_wrong_count_raises():
    with pytest.raises(ValueError, match="exactly 3"):
        batch_bond_angle(_BATCH_COORDS, [0, 1])


# ---------------------------------------------------------------------------
# shift_angle
# ---------------------------------------------------------------------------


def test_shift_angle_zero_dihedral():
    """shift=0 is an identity for dihedral mode."""
    vals = np.linspace(-180.0, 170.0, 36)
    np.testing.assert_allclose(shift_angle(vals, 0.0, "dihedral"), vals, atol=1e-10)


def test_shift_angle_zero_signed():
    """shift=0 is an identity for signed mode."""
    vals = np.linspace(-180.0, 170.0, 36)
    np.testing.assert_allclose(shift_angle(vals, 0.0, "signed"), vals, atol=1e-10)


def test_shift_angle_dihedral_wrap_positive():
    """160 + 40 = 200 → wraps to -160 in [-180, 180)."""
    result = float(shift_angle(160.0, 40.0, "dihedral"))
    assert np.isclose(result, -160.0, atol=1e-10), f"got {result}"


def test_shift_angle_dihedral_wrap_negative():
    """-170 - 20 = -190 → wraps to 170 in [-180, 180)."""
    result = float(shift_angle(-170.0, -20.0, "dihedral"))
    assert np.isclose(result, 170.0, atol=1e-10), f"got {result}"


def test_shift_angle_signed_wrap_positive():
    """160 + 40 = 200 → wraps to -160 for signed mode."""
    result = float(shift_angle(160.0, 40.0, "signed"))
    assert np.isclose(result, -160.0, atol=1e-10), f"got {result}"


def test_shift_angle_invalid_mode_raises():
    """Unknown mode raises ValueError."""
    with pytest.raises(ValueError, match="unsupported mode"):
        shift_angle(0.0, 10.0, "invalid_mode")
