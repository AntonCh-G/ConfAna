"""Tests for confana/coordinates.py."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from confana.coordinate_config import PairTransformSpec
from confana.coordinates import (
    apply_coordinate_shifts,
    apply_pair_transforms,
    build_coordinate_table_from_values,
    build_dof_long_table,
    extract_geometry_dof,
)
from confana.models import DoFDefinition, FrameRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(coords: np.ndarray, frame_number: int = 0) -> FrameRecord:
    n = coords.shape[0]
    return FrameRecord(
        source_file="/tmp/test.xyz",
        frame_number=frame_number,
        byte_offset=0,
        atom_count=n,
        comment_line="# Step: 0  Bead: 0",
        elements=["C"] * n,
        coords=coords,
        energy=None,
        step_number=0,
        bead_comment=0,
        trajectory_id="traj",
        bead_id="00",
        local_frame_index=frame_number,
        global_frame_index=frame_number,
    )


def _aspirin_like_coords(n_atoms: int = 21, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_atoms, 3))


# Standard dihedral DoF definitions (aspirin dataset, 0-based)
_DOF_DEFS = [
    DoFDefinition(
        name="carboxyl_dihedral",
        type="dihedral",
        atoms=(6, 5, 10, 7),
        label="Carboxyl dihedral (°)",
        domain=(-180.0, 180.0),
        enabled=True,
    ),
    DoFDefinition(
        name="ester_dihedral",
        type="dihedral",
        atoms=(5, 6, 12, 11),
        label="Ester dihedral (°)",
        domain=(-180.0, 180.0),
        enabled=True,
    ),
]

_DOF_DEFS_WITH_EXTRA = _DOF_DEFS + [
    DoFDefinition(
        name="igor1_dihedral",
        type="dihedral",
        atoms=(6, 12, 11, 8),
        label="Igor 1 dihedral (°)",
        domain=(-180.0, 180.0),
        enabled=True,
    ),
]

_BASE_METADATA_COLUMNS = [
    "frame_id",
    "source_file",
    "trajectory_id",
    "bead_id",
    "frame_number",
    "byte_offset",
    "atom_count",
    "comment_line",
    "local_frame_index",
    "global_frame_index",
]


def _build_dihedral_pair(
    a: np.ndarray,
    b: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return points C and D so dihedral(A, B, C, D) equals angle_deg."""
    u = a - b
    u = u / np.linalg.norm(u)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(u, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    v = np.cross(u, helper)
    v = v / np.linalg.norm(v)
    w = np.cross(v, u)
    w = w / np.linalg.norm(w)

    theta = np.radians(angle_deg)
    c = b + v
    d = c + np.cos(theta) * u - np.sin(theta) * w
    return c, d


def _make_dihedral_test_frame(
    carboxyl_angle: float,
    ester_angle: float,
    frame_number: int = 0,
) -> FrameRecord:
    """Build one synthetic frame with controllable carboxyl/ester dihedrals."""
    coords = np.zeros((21, 3), dtype=float)

    atom5 = np.array([0.0, 0.0, 0.0])
    atom6 = np.array([0.0, 1.0, 0.0])
    coords[5] = atom5
    coords[6] = atom6

    coords[10], coords[7] = _build_dihedral_pair(atom6, atom5, carboxyl_angle)
    coords[12], coords[11] = _build_dihedral_pair(atom5, atom6, ester_angle)

    return _make_frame(coords, frame_number=frame_number)


# ---------------------------------------------------------------------------
# extract_geometry_dof
# ---------------------------------------------------------------------------


def test_extract_geometry_dof_returns_expected_keys():
    coords = _aspirin_like_coords()
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, _DOF_DEFS)
    assert set(result.keys()) == {"carboxyl_dihedral", "ester_dihedral"}


def test_extract_geometry_dof_dihedral_in_range():
    coords = _aspirin_like_coords()
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, _DOF_DEFS)
    assert -180.0 <= result["carboxyl_dihedral"] < 180.0
    assert -180.0 <= result["ester_dihedral"] < 180.0


def test_extract_geometry_dof_values_are_float():
    coords = _aspirin_like_coords()
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, _DOF_DEFS)
    assert isinstance(result["carboxyl_dihedral"], float)
    assert isinstance(result["ester_dihedral"], float)


def test_extract_geometry_dof_preserves_sign_quadrants():
    targets = [
        (60.0, 60.0),
        (60.0, -60.0),
        (-60.0, 60.0),
        (-60.0, -60.0),
    ]

    observed: list[tuple[float, float]] = []
    for i, (carboxyl_target, ester_target) in enumerate(targets):
        frame = _make_dihedral_test_frame(carboxyl_target, ester_target, frame_number=i)
        result = extract_geometry_dof(frame, _DOF_DEFS)
        observed.append((result["carboxyl_dihedral"], result["ester_dihedral"]))

    observed_arr = np.array(observed)
    np.testing.assert_allclose(
        observed_arr,
        np.array(targets),
        atol=1e-8,
    )


def test_extract_geometry_dof_supports_multiple_definitions():
    coords = _aspirin_like_coords(seed=11)
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, _DOF_DEFS_WITH_EXTRA)
    assert {"carboxyl_dihedral", "ester_dihedral", "igor1_dihedral"} <= set(result)
    assert -180.0 <= result["igor1_dihedral"] < 180.0


def test_extract_geometry_dof_skips_disabled():
    dof_defs = [
        DoFDefinition(
            name="carboxyl_dihedral",
            type="dihedral",
            atoms=(6, 5, 10, 7),
            label="Carboxyl dihedral (°)",
            domain=(-180.0, 180.0),
            enabled=False,
        ),
        DoFDefinition(
            name="ester_dihedral",
            type="dihedral",
            atoms=(5, 6, 12, 11),
            label="Ester dihedral (°)",
            domain=(-180.0, 180.0),
            enabled=True,
        ),
    ]
    coords = _aspirin_like_coords()
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, dof_defs)
    assert "carboxyl_dihedral" not in result
    assert "ester_dihedral" in result


def test_extract_geometry_dof_distance_type():
    dof_defs = [
        DoFDefinition(
            name="bond_length",
            type="distance",
            atoms=(0, 1),
            label="Bond length (Å)",
            domain=(0.0, 10.0),
            enabled=True,
        ),
    ]
    coords = _aspirin_like_coords()
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, dof_defs)
    assert "bond_length" in result
    assert result["bond_length"] >= 0.0


def test_extract_geometry_dof_angle_type():
    dof_defs = [
        DoFDefinition(
            name="bond_angle",
            type="angle",
            atoms=(0, 1, 2),
            label="Bond angle (°)",
            domain=(0.0, 180.0),
            enabled=True,
        ),
    ]
    coords = _aspirin_like_coords()
    frame = _make_frame(coords)
    result = extract_geometry_dof(frame, dof_defs)
    assert "bond_angle" in result
    assert 0.0 <= result["bond_angle"] <= 180.0


# ---------------------------------------------------------------------------
# build_coordinate_table_from_values
# ---------------------------------------------------------------------------


def _make_values_df() -> pd.DataFrame:
    """Build a minimal DataFrame with pre-computed angle values."""
    row = {col: None for col in _BASE_METADATA_COLUMNS}
    row.update(
        {
            "frame_id": 0,
            "source_file": "/tmp/test.xyz",
            "trajectory_id": "traj",
            "bead_id": "00",
            "frame_number": 0,
            "byte_offset": 0,
            "atom_count": 21,
            "comment_line": "",
            "local_frame_index": 0,
            "global_frame_index": 0,
            "carboxyl_dihedral": -120.0,
            "ester_dihedral": 200.0,  # out of range → should be wrapped
        }
    )
    return pd.DataFrame([row])


def test_build_from_values_dihedral_wrapping():
    df = _make_values_df()
    result = build_coordinate_table_from_values(df, _DOF_DEFS)
    # 200° wraps to 200 - 360 = -160°
    assert np.isclose(float(result["ester_dihedral"].iloc[0]), -160.0, atol=1e-5)


def test_build_from_values_preserves_valid_dihedral():
    df = _make_values_df()
    df["carboxyl_dihedral"] = -120.0
    result = build_coordinate_table_from_values(df, _DOF_DEFS)
    assert np.isclose(float(result["carboxyl_dihedral"].iloc[0]), -120.0, atol=1e-5)


def test_build_from_values_missing_metadata_raises():
    df = _make_values_df()
    df = df.drop(columns=["frame_id"])
    with pytest.raises(ValueError, match="missing required"):
        build_coordinate_table_from_values(df, _DOF_DEFS)


def test_build_from_values_wraps_additional_dihedrals():
    df = _make_values_df()
    df["igor1_dihedral"] = 540.0
    result = build_coordinate_table_from_values(df, _DOF_DEFS_WITH_EXTRA)
    assert np.isclose(float(result["igor1_dihedral"].iloc[0]), -180.0, atol=1e-5)


# ---------------------------------------------------------------------------
# build_dof_long_table
# ---------------------------------------------------------------------------


def test_build_dof_long_table_returns_tidy_rows():
    df = _make_values_df()
    df["igor1_dihedral"] = 15.0
    long_df = build_dof_long_table(df, _DOF_DEFS_WITH_EXTRA, names=["ester_dihedral", "igor1_dihedral"])
    assert set(long_df["dof_name"]) == {"ester_dihedral", "igor1_dihedral"}


def test_build_dof_long_table_one_row_per_dof():
    df = _make_values_df()
    long_df = build_dof_long_table(df, _DOF_DEFS, names=["carboxyl_dihedral", "ester_dihedral"])
    assert len(long_df) == 2


# ---------------------------------------------------------------------------
# apply_coordinate_shifts
# ---------------------------------------------------------------------------


_CONV_UNSIGNED = {"plane_signed": False, "dihedral_signed": True}
_CONV_SIGNED_PLANE = {"plane_signed": True, "dihedral_signed": True}


def _make_shift_df(
    carboxyl_dihedral: float = 45.0,
    ester_dihedral: float = -90.0,
) -> pd.DataFrame:
    """Minimal single-row DataFrame with two dihedral columns."""
    return pd.DataFrame(
        {
            "carboxyl_dihedral": pd.array([carboxyl_dihedral], dtype="float32"),
            "ester_dihedral": pd.array([ester_dihedral], dtype="float32"),
        }
    )


def test_apply_shifts_empty_transforms_returns_unchanged():
    """Empty transforms dict → df returned as-is."""
    df = _make_shift_df()
    result = apply_coordinate_shifts(df, {})
    assert result is df
    assert list(result.columns) == list(df.columns)


def test_apply_shifts_adds_shifted_columns():
    """Configured columns gain a *_shifted companion; others are untouched."""
    df = _make_shift_df()
    transforms = {"carboxyl_dihedral": 10.0}
    result = apply_coordinate_shifts(df, transforms)
    assert "carboxyl_dihedral_shifted" in result.columns
    assert "ester_dihedral_shifted" not in result.columns


def test_apply_shifts_preserves_raw_columns():
    """Raw columns are not modified by the shift."""
    df = _make_shift_df(carboxyl_dihedral=45.0)
    result = apply_coordinate_shifts(df, {"carboxyl_dihedral": 30.0})
    assert float(result["carboxyl_dihedral"].iloc[0]) == pytest.approx(45.0, abs=1e-4)


def test_apply_shifts_zero_shift_matches_raw():
    """A shift of 0 must produce a shifted column equal to the raw column."""
    df = _make_shift_df(ester_dihedral=-90.0)
    result = apply_coordinate_shifts(df, {"ester_dihedral": 0.0})
    assert float(result["ester_dihedral_shifted"].iloc[0]) == pytest.approx(
        float(result["ester_dihedral"].iloc[0]), abs=1e-4
    )


def test_apply_shifts_dihedral_wraps_correctly():
    """160 + 40 = 200 → wraps to -160 in [-180, 180)."""
    df = _make_shift_df(carboxyl_dihedral=160.0)
    result = apply_coordinate_shifts(df, {"carboxyl_dihedral": 40.0})
    shifted = float(result["carboxyl_dihedral_shifted"].iloc[0])
    assert shifted == pytest.approx(-160.0, abs=1e-3)


def test_apply_shifts_dihedral_negative_wrap():
    """-170 - 20 = -190 → wraps to 170."""
    df = _make_shift_df(ester_dihedral=-170.0)
    result = apply_coordinate_shifts(df, {"ester_dihedral": -20.0})
    shifted = float(result["ester_dihedral_shifted"].iloc[0])
    assert shifted == pytest.approx(170.0, abs=1e-3)


def test_apply_shifts_shifted_column_dtype_is_float32():
    """Shifted columns are stored as float32 (matching raw column dtype)."""
    df = _make_shift_df()
    result = apply_coordinate_shifts(df, {"carboxyl_dihedral": 10.0})
    assert result["carboxyl_dihedral_shifted"].dtype == np.float32


def test_apply_shifts_unknown_column_silently_skipped():
    """A transform that names a non-existent column does not raise."""
    df = _make_shift_df()
    result = apply_coordinate_shifts(df, {"nonexistent_col": 90.0})
    assert "nonexistent_col_shifted" not in result.columns


def test_apply_shifts_multiple_columns():
    """Multiple transforms are all applied independently."""
    df = _make_shift_df(carboxyl_dihedral=10.0, ester_dihedral=-10.0)
    transforms = {"carboxyl_dihedral": 20.0, "ester_dihedral": -20.0}
    result = apply_coordinate_shifts(df, transforms)
    assert float(result["carboxyl_dihedral_shifted"].iloc[0]) == pytest.approx(30.0, abs=1e-3)
    assert float(result["ester_dihedral_shifted"].iloc[0]) == pytest.approx(-30.0, abs=1e-3)


# ---------------------------------------------------------------------------
# apply_pair_transforms
# ---------------------------------------------------------------------------


def _spec(type_: str, value: float = 0.0) -> PairTransformSpec:
    return PairTransformSpec(type=type_, value=value)


def test_pair_transform_center_x_at():
    """center_x_at shifts x so that value → 0, wraps to [-180, 180)."""
    x = np.array([90.0, 0.0, 180.0, -90.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    rx, ry = apply_pair_transforms(x, y, [_spec("center_x_at", 90.0)])
    # 90 - 90 = 0; 0 - 90 = -90; 180 - 90 = 90; -90 - 90 = -180
    assert np.allclose(rx, [0.0, -90.0, 90.0, -180.0], atol=1e-6)
    assert np.allclose(ry, y, atol=1e-6)


def test_pair_transform_fold_sign_symmetry_negative_x():
    """Negative x gets flipped; y is negated accordingly."""
    x = np.array([-45.0, -90.0])
    y = np.array([30.0, -60.0])
    rx, ry = apply_pair_transforms(x, y, [_spec("fold_sign_symmetry")])
    assert np.allclose(rx, [45.0, 90.0], atol=1e-9)
    assert np.allclose(ry, [-30.0, 60.0], atol=1e-9)


def test_pair_transform_fold_sign_symmetry_positive_x_unchanged():
    """Positive x is unaffected."""
    x = np.array([45.0, 90.0])
    y = np.array([30.0, -60.0])
    rx, ry = apply_pair_transforms(x, y, [_spec("fold_sign_symmetry")])
    assert np.allclose(rx, x, atol=1e-9)
    assert np.allclose(ry, y, atol=1e-9)


def test_pair_transform_fold_sign_symmetry_zero_x():
    """x ≈ 0 is kept at 0; y becomes |y|."""
    x = np.array([0.0, 1e-12])
    y = np.array([-5.0, -10.0])
    rx, ry = apply_pair_transforms(x, y, [_spec("fold_sign_symmetry")])
    assert np.all(ry >= 0.0)


def test_pair_transform_restrict_positive_y():
    """Points with y < 0 become NaN (to be filtered by caller)."""
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([5.0, -5.0, 0.0])
    rx, ry = apply_pair_transforms(x, y, [_spec("restrict_positive_y")])
    assert np.isfinite(rx[0]) and np.isfinite(ry[0])
    assert np.isnan(rx[1]) and np.isnan(ry[1])
    assert np.isfinite(rx[2]) and np.isfinite(ry[2])


def test_pair_transform_shift_x():
    """shift_x adds offset and wraps to [-180, 180)."""
    x = np.array([170.0, -170.0])
    y = np.array([0.0, 0.0])
    rx, ry = apply_pair_transforms(x, y, [_spec("shift_x", -90.0)])
    # 170 - 90 = 80; -170 - 90 = -260 → wrap → 100
    assert np.allclose(rx, [80.0, 100.0], atol=1e-6)


def test_pair_transform_chained_pipeline():
    """The full publication pipeline reduces data to x ∈ [-90, 90] and y ≥ 0."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-180, 180, 200)
    y = rng.uniform(-180, 180, 200)
    pipeline = [
        _spec("center_x_at", 90.0),
        _spec("fold_sign_symmetry"),
        _spec("restrict_positive_y"),
        _spec("shift_x", -90.0),
    ]
    rx, ry = apply_pair_transforms(x.copy(), y.copy(), pipeline)
    finite = np.isfinite(rx) & np.isfinite(ry)
    # All finite x values should be in [-90, 90] after the full pipeline
    assert np.all(rx[finite] >= -90.0 - 1e-6)
    assert np.all(rx[finite] <= 90.0 + 1e-6)
    # All finite y values should be >= 0
    assert np.all(ry[finite] >= -1e-9)


def test_pair_transform_unknown_type_raises():
    """An unknown transform type raises ValueError."""
    x = np.array([0.0])
    y = np.array([0.0])
    with pytest.raises(ValueError, match="unknown transform type"):
        apply_pair_transforms(x, y, [_spec("bad_type")])
