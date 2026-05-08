"""Coordinate extraction from xyz frames and coordinate table builder.

This module bridges the I/O layer (FrameRecord) and the analysis layer
(standard coordinate table DataFrame).  It can operate in two modes:

1. **Full-structure mode**: accepts a list of ``FrameRecord`` objects produced
   by ``io_xyz.load_xyz_files`` and computes geometry DoF (dihedrals, distances,
   bond angles) via a unified :class:`~src.models.DoFDefinition` list.
2. **Coordinate-only mode**: accepts a DataFrame that already contains the
   required DoF columns and normalises them without recomputing geometry.

The extraction pipeline is sequential:

    xyz frames
      → Geometry DoF (dihedral / distance / angle)
      → External DoF joined by frame_id  [stub — NotImplementedError]
      → Collective variables (PCA, linear)  [stub — NotImplementedError]
      → Standard coordinate table

Public API
----------
- ``extract_geometry_dof``
- ``batch_extract_geometry_dof``
- ``apply_external_dof``
- ``apply_collective_dof``
- ``apply_pair_transforms``
- ``build_coordinate_table_from_xyz``
- ``build_coordinate_table_from_values``
- ``apply_coordinate_shifts``
- ``build_dof_long_table``
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.angle_domains import wrap_signed_degrees
from src.coordinate_config import PairTransformSpec
from src.geometry import (
    batch_bond_angle,
    batch_dihedral_angle,
    batch_distance,
    bond_angle,
    dihedral_angle,
    distance,
    shift_angle,
)
from src.models import DoFDefinition, FrameRecord

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
# DoF validation helpers
# ---------------------------------------------------------------------------


def _validate_atom_count_for_dofs(
    frame: FrameRecord,
    dof_defs: list[DoFDefinition],
) -> None:
    """Raise ValueError if any DoF atom index exceeds frame.atom_count."""
    max_id = -1
    for dof in dof_defs:
        if dof.atoms is not None:
            local_max = max(dof.atoms)
            if local_max > max_id:
                max_id = local_max
    if max_id >= frame.atom_count:
        raise ValueError(
            f"DoF atom mapping references index {max_id} but frame "
            f"{frame.frame_number} in {frame.source_file} has only "
            f"{frame.atom_count} atoms (max 0-based index = {frame.atom_count - 1})."
        )


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
        Ordered list of enabled :class:`~src.models.DoFDefinition` objects.

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
        Ordered list of enabled :class:`~src.models.DoFDefinition` objects.

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
# External DoF (stub)
# ---------------------------------------------------------------------------


def apply_external_dof(
    df: pd.DataFrame,
    dof_defs: list[DoFDefinition],
    external_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join external DoF columns onto the coordinate table by ``frame_id``.

    Parameters
    ----------
    df:
        Coordinate table with at least a ``frame_id`` column.
    dof_defs:
        Full DoF definition list.  Only entries with ``type='external'`` are
        processed.
    external_df:
        DataFrame indexed on ``frame_id`` supplying external columns.  If
        ``None`` and external DoF are configured, :class:`NotImplementedError`
        is raised.

    Returns
    -------
    pd.DataFrame
        *df* with external columns appended.

    Raises
    ------
    NotImplementedError
        If any external DoF are configured and ``external_df`` is None.
    """
    external_dofs = [d for d in dof_defs if d.enabled and d.type == "external"]
    if not external_dofs:
        return df
    if external_df is None:
        names = [d.name for d in external_dofs]
        raise NotImplementedError(
            f"External DoF {names} require an external_df to be provided. "
            "This feature is not yet implemented — pass external_df to "
            "apply_external_dof once an external data source is available."
        )
    result = df.copy()
    for dof in external_dofs:
        col = dof.source_column
        if col not in external_df.columns:
            raise KeyError(
                f"External DoF '{dof.name}' references source_column '{col}' "
                f"which is not present in external_df."
            )
        result[dof.name] = external_df[col].values
    return result


# ---------------------------------------------------------------------------
# Collective variable DoF (stub)
# ---------------------------------------------------------------------------


def apply_collective_dof(
    df: pd.DataFrame,
    dof_defs: list[DoFDefinition],
) -> pd.DataFrame:
    """Compute collective variables and append them to the coordinate table.

    Parameters
    ----------
    df:
        Coordinate table with geometry DoF columns already populated.
    dof_defs:
        Full DoF definition list.  Only entries with ``type='collective'`` are
        processed.

    Returns
    -------
    pd.DataFrame
        *df* with collective variable columns appended.

    Raises
    ------
    NotImplementedError
        Always — collective variables (PCA, linear combinations) are not yet
        implemented.
    """
    collective_dofs = [d for d in dof_defs if d.enabled and d.type == "collective"]
    if not collective_dofs:
        return df
    names = [d.name for d in collective_dofs]
    raise NotImplementedError(
        f"Collective variable DoF {names} are not yet implemented. "
        "Register them in the dof: config list with type: collective and "
        "implement the computation in apply_collective_dof."
    )


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
        Ordered list of :class:`~src.coordinate_config.PairTransformSpec`.

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
# Table builder from xyz frames
# ---------------------------------------------------------------------------


def build_coordinate_table_from_xyz(
    frames: list[FrameRecord],
    dof_defs: list[DoFDefinition],
    transforms: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build the standard coordinate table from a list of FrameRecord objects.

    Sequential pipeline:
    1. Compute geometry DoF (dihedral, distance, angle) per frame.
    2. Apply external DoF (stub — skipped if none configured).
    3. Apply collective DoF (stub — skipped if none configured).
    4. Apply angular shifts to produce ``*_shifted`` companion columns.

    Parameters
    ----------
    frames:
        List of ``FrameRecord`` objects with coords populated.
    dof_defs:
        Ordered list of :class:`~src.models.DoFDefinition` objects (enabled
        and disabled); disabled ones are skipped.
    transforms:
        Optional ``{column_name: shift_degrees}`` map.  Applied after
        extraction to produce ``*_shifted`` columns.  Pass ``None`` or ``{}``
        to skip.

    Returns
    -------
    pd.DataFrame
        Standard coordinate table.  Contains base metadata columns plus one
        column per enabled geometry DoF, plus ``energy`` and ``step_number``.
        State columns are added later by the state-assignment step.

    Raises
    ------
    ValueError
        On the first frame that fails geometry or atom-count validation.
        The error message includes ``frame_number`` and ``source_file``.
    NotImplementedError
        If collective or external DoF are configured and not yet implemented.
    """
    enabled_dofs = [d for d in dof_defs if d.enabled]
    geometry_dofs = [d for d in enabled_dofs if d.type in _GEOMETRY_DOF_TYPES]
    dof_names = [d.name for d in geometry_dofs]

    rows: list[dict] = []

    for frame_id, frame in enumerate(frames):
        # Validate atom indices for all geometry DoF
        try:
            _validate_atom_count_for_dofs(frame, geometry_dofs)
        except ValueError as exc:
            raise ValueError(
                f"Atom-count validation failed for frame {frame.frame_number} "
                f"in {frame.source_file}: {exc}"
            ) from exc

        # Compute geometry DoF
        try:
            geom_values = extract_geometry_dof(frame, geometry_dofs)
        except ValueError as exc:
            raise ValueError(
                f"Geometry DoF computation failed for frame {frame.frame_number} "
                f"in {frame.source_file}: {exc}"
            ) from exc

        rows.append(
            {
                "frame_id": frame_id,
                "source_file": frame.source_file,
                "trajectory_id": frame.trajectory_id,
                "bead_id": frame.bead_id,
                "frame_number": frame.frame_number,
                "byte_offset": frame.byte_offset,
                "atom_count": frame.atom_count,
                "comment_line": frame.comment_line,
                "local_frame_index": frame.local_frame_index,
                "global_frame_index": frame.global_frame_index,
                "energy": frame.energy,
                "step_number": frame.step_number,
                **geom_values,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=_BASE_METADATA_COLUMNS + ["energy", "step_number"] + dof_names
        )

    df = pd.DataFrame(rows)

    # Apply canonical dtypes
    int_cols = [
        "frame_id", "frame_number", "byte_offset", "atom_count",
        "local_frame_index", "global_frame_index", "step_number",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    for col in dof_names:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if "energy" in df.columns:
        df["energy"] = df["energy"].astype("float32")

    for col in ("bead_id",):
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Pipeline stage 2: external DoF (stub — no-op if none configured)
    df = apply_external_dof(df, enabled_dofs)

    # Pipeline stage 3: collective DoF (stub — no-op if none configured)
    df = apply_collective_dof(df, enabled_dofs)

    # Pipeline stage 4: angular shifts
    if transforms:
        df = apply_coordinate_shifts(df, transforms)

    return df


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
        List of :class:`~src.models.DoFDefinition` objects describing the
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
