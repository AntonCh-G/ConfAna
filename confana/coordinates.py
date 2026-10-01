"""Coordinate extraction: geometry DoF values and angular shifts.

The coordinate-table build (``confana.coordinate_table``) computes every
geometry DoF (dihedral, distance, bond angle) for blocks of frames with
:func:`batch_extract_geometry_dof` and adds the ``*_shifted`` columns of
``coordinate_transforms`` with :func:`apply_coordinate_shifts`. A table of
pre-computed values can be normalised with
:func:`build_coordinate_table_from_values` instead.

Public API
----------
- ``extract_geometry_dof``
- ``batch_extract_geometry_dof``
- ``apply_pair_transforms``
- ``build_coordinate_table_from_values``
- ``apply_coordinate_shifts``
- ``build_dof_long_table``
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from confana.angle_domains import wrap_signed_degrees
from confana.coordinate_config import PairTransformSpec
from confana.geometry import (
    batch_bond_angle,
    batch_dihedral_angle,
    batch_distance,
    bond_angle,
    dihedral_angle,
    distance,
    shift_angle,
)
from confana.models import DoFDefinition, FrameRecord

# ---------------------------------------------------------------------------
# Base metadata columns (always present in coordinate table, before any DoF)
# ---------------------------------------------------------------------------

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

_GEOMETRY_DOF_TYPES = frozenset({"dihedral", "distance", "angle"})


# ---------------------------------------------------------------------------
# Per-frame geometry extraction
# ---------------------------------------------------------------------------


def extract_geometry_dof(
    frame: FrameRecord,
    dof_defs: list[DoFDefinition],
) -> dict[str, float]:
    """Compute geometry DoF for one frame.

    Handles types: ``"dihedral"``, ``"distance"``, ``"angle"``.
    Types ``"collective"`` and ``"external"`` are skipped (handled downstream).

    Parameters
    ----------
    frame:
        A ``FrameRecord`` with ``coords`` populated (shape N×3).
    dof_defs:
        Ordered list of enabled :class:`~confana.models.DoFDefinition` objects.

    Returns
    -------
    dict[str, float]
        ``{dof_name: value_in_degrees_or_angstrom}`` for every geometry DoF.

    Raises
    ------
    ValueError
        Propagated from geometry functions on invalid indices or degenerate
        geometry.
    """
    result: dict[str, float] = {}
    for dof in dof_defs:
        if not dof.enabled or dof.type not in _GEOMETRY_DOF_TYPES:
            continue
        if dof.type == "dihedral":
            result[dof.name] = dihedral_angle(frame.coords, dof.atoms)
        elif dof.type == "distance":
            result[dof.name] = distance(frame.coords, dof.atoms)
        elif dof.type == "angle":
            result[dof.name] = bond_angle(frame.coords, dof.atoms)
    return result


# ---------------------------------------------------------------------------
# Batch geometry extraction (vectorised over N frames)
# ---------------------------------------------------------------------------


def batch_extract_geometry_dof(
    coords: np.ndarray,
    dof_defs: list[DoFDefinition],
) -> dict[str, np.ndarray]:
    """Compute geometry DoF for N frames simultaneously.

    Equivalent to calling :func:`extract_geometry_dof` in a loop but uses
    vectorised batch geometry functions for efficiency.

    Parameters
    ----------
    coords:
        Float array of shape ``(N, A, 3)`` — N frames, A atoms each.
    dof_defs:
        Ordered list of enabled :class:`~confana.models.DoFDefinition` objects.

    Returns
    -------
    dict[str, np.ndarray]
        ``{dof_name: values}`` for every enabled geometry DoF.
        Each value array has shape ``(N,)``.

    Raises
    ------
    ValueError
        Propagated from batch geometry functions on invalid indices or
        degenerate geometry.
    """
    result: dict[str, np.ndarray] = {}
    for dof in dof_defs:
        if not dof.enabled or dof.type not in _GEOMETRY_DOF_TYPES:
            continue
        if dof.type == "dihedral":
            result[dof.name] = batch_dihedral_angle(coords, dof.atoms)
        elif dof.type == "distance":
            result[dof.name] = batch_distance(coords, dof.atoms)
        elif dof.type == "angle":
            result[dof.name] = batch_bond_angle(coords, dof.atoms)
    return result


# ---------------------------------------------------------------------------
# Pair transform application
# ---------------------------------------------------------------------------

_EPS_FOLD = 1e-9


def apply_pair_transforms(
    x: np.ndarray,
    y: np.ndarray,
    transforms: list[PairTransformSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an ordered pipeline of coupled (x, y) transforms.

    Transforms are applied in list order.

    Supported transform types
    -------------------------
    ``center_x_at``:
        Shift x so that ``value`` maps to 0°.
        ``x = ((x - value + 180) % 360) - 180``

    ``fold_sign_symmetry``:
        Exploit the physical ``(x, y) ≡ (-x, -y)`` symmetry.  Points with
        ``x < 0`` are remapped to ``(-x, -y)``.  Points with ``|x| ≤ eps``
        have their y set to ``|y|``.

    ``restrict_positive_y``:
        Points with ``y < 0`` are set to NaN.  Callers must filter them out.

    ``shift_x``:
        Add ``value`` to x and wrap to ``[-180, 180)``.

    Parameters
    ----------
    x, y:
        1-D float arrays of equal length.
    transforms:
        Ordered list of :class:`~confana.coordinate_config.PairTransformSpec`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed ``(x, y)`` as float64 arrays.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()

    for spec in transforms:
        if spec.type == "center_x_at":
            x = ((x - spec.value + 180.0) % 360.0) - 180.0

        elif spec.type == "fold_sign_symmetry":
            neg = x < -_EPS_FOLD
            x[neg] = -x[neg]
            y[neg] = -y[neg]
            zero = np.abs(x) <= _EPS_FOLD
            y[zero] = np.abs(y[zero])

        elif spec.type == "restrict_positive_y":
            discard = y < 0.0
            x[discard] = np.nan
            y[discard] = np.nan

        elif spec.type == "shift_x":
            x = ((x + spec.value + 180.0) % 360.0) - 180.0

        else:
            raise ValueError(
                f"apply_pair_transforms: unknown transform type {spec.type!r}"
            )

    return x, y


# ---------------------------------------------------------------------------
# Table builder from pre-computed values
# ---------------------------------------------------------------------------


def build_coordinate_table_from_values(
    table: pd.DataFrame,
    dof_defs: list[DoFDefinition],
) -> pd.DataFrame:
    """Normalise a DataFrame of pre-computed coordinate values.

    Applies domain normalisation to DoF columns that are present in *table*:
    - ``dihedral`` type: wrapped to ``[-180, 180)``
    - ``distance`` / ``angle`` types: left as-is (already non-periodic)
    - ``collective`` / ``external`` types: left as-is

    This function does **not** recompute any geometry.

    Parameters
    ----------
    table:
        Input DataFrame; must contain the base metadata columns and at least
        one DoF column.
    dof_defs:
        List of :class:`~confana.models.DoFDefinition` objects describing the
        expected DoF columns.  Only those present in *table* are normalised.

    Returns
    -------
    pd.DataFrame
        Validated and normalised copy of the input table.

    Raises
    ------
    ValueError
        If base metadata columns are missing from *table*.
    """
    df = table.copy()

    missing_meta = [c for c in _BASE_METADATA_COLUMNS if c not in df.columns]
    if missing_meta:
        raise ValueError(
            f"build_coordinate_table_from_values: table is missing required "
            f"metadata columns: {missing_meta}"
        )

    for dof in dof_defs:
        if dof.name not in df.columns:
            continue
        if dof.type == "dihedral":
            vals = df[dof.name].astype(float)
            df[dof.name] = wrap_signed_degrees(vals).astype("float32")

    return df


# ---------------------------------------------------------------------------
# Angular shift application
# ---------------------------------------------------------------------------


def apply_coordinate_shifts(
    df: pd.DataFrame,
    transforms: dict[str, float],
) -> pd.DataFrame:
    """Apply angular shifts to DoF columns and store as ``*_shifted`` companions.

    For each ``(col, shift_degrees)`` entry in *transforms* that names a column
    present in *df*:

    1. The raw column ``df[col]`` is left untouched.
    2. A new column ``df[f"{col}_shifted"]`` is added, containing
       ``df[col] + shift_degrees`` wrapped to ``[-180, 180)``.

    Parameters
    ----------
    df:
        Coordinate table with DoF columns already present.
    transforms:
        Mapping of ``{column_name: shift_degrees}``.  Entries referencing
        columns absent from *df* are silently skipped.

    Returns
    -------
    pd.DataFrame
        *df* with ``*_shifted`` columns appended (dtype ``float32``).
        If *transforms* is empty the original object is returned unchanged.
    """
    if not transforms:
        return df

    result = df.copy()
    for col, shift in transforms.items():
        if col not in result.columns:
            continue
        raw = result[col].to_numpy(dtype=float, na_value=float("nan"))
        shifted = shift_angle(raw, float(shift), mode="signed").astype("float32")
        result[f"{col}_shifted"] = shifted

    return result


# ---------------------------------------------------------------------------
# Long-form DoF table
# ---------------------------------------------------------------------------


def build_dof_long_table(
    table: pd.DataFrame,
    dof_defs: list[DoFDefinition] | None = None,
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a tidy long-form table by melting selected DoF columns.

    Parameters
    ----------
    table:
        Wide coordinate table.
    dof_defs:
        DoF definition list used to attach ``dof_label``, ``dof_type`` metadata.
        Pass ``None`` to skip metadata attachment.
    names:
        Subset of DoF column names to include.  Defaults to all DoF columns
        that appear in both *table* and *dof_defs*.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns ``dof_name``, ``dof_value``, plus
        optional ``dof_label`` and ``dof_type`` when *dof_defs* is provided.
    """
    dof_map = {d.name: d for d in (dof_defs or [])}

    if names is None:
        # Use all DoF columns present in both the table and dof_map
        if dof_map:
            names = [n for n in dof_map if n in table.columns]
        else:
            # Fallback: unknown DoF columns — nothing to melt
            names = []

    missing = [n for n in names if n not in table.columns]
    if missing:
        raise ValueError(
            f"build_dof_long_table: requested DoF columns not in table: {missing}"
        )

    id_columns = [col for col in table.columns if col not in names]
    long_df = table.melt(
        id_vars=id_columns,
        value_vars=list(names),
        var_name="dof_name",
        value_name="dof_value",
    )

    if dof_map:
        long_df["dof_label"] = long_df["dof_name"].map(
            {n: dof_map[n].label for n in names if n in dof_map}
        )
        long_df["dof_type"] = long_df["dof_name"].map(
            {n: dof_map[n].type for n in names if n in dof_map}
        )

    return long_df
