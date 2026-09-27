"""DoF-definition and coordinate-pair configuration helpers.

This module resolves analysis-level configuration (DoF definitions, coordinate
pairs, angular transforms) into concrete runtime objects consumed by the
geometry, coordinate extraction, plotting, and state-assignment layers.

Public API
----------
- ``DoFDefinition``          (re-exported from models)
- ``CoordinatePair``         (re-exported from models)
- ``PairTransformSpec``
- ``resolve_dof_definitions``
- ``resolve_coordinate_transforms``
- ``resolve_pair_transforms``
- ``resolve_coordinate_pair``
- ``list_coordinate_pairs``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import CoordinatePair, DoFDefinition

# Re-export for callers that import from this module
__all__ = [
    "DoFDefinition",
    "CoordinatePair",
    "PairTransformSpec",
    "resolve_dof_definitions",
    "resolve_coordinate_transforms",
    "resolve_pair_transforms",
    "resolve_coordinate_pair",
    "list_coordinate_pairs",
]


# ---------------------------------------------------------------------------
# Pair transform spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairTransformSpec:
    """A single step in a coordinate-pair transform pipeline.

    Supported transform types
    -------------------------
    ``center_x_at``:
        Shift x so that ``value`` maps to 0°.  Wraps result to ``[-180, 180)``.
        ``x = wrap_signed(x - value)``

    ``fold_sign_symmetry``:
        Exploit the physical ``(x, y) ≡ (-x, -y)`` symmetry by folding the
        negative-x half onto the positive-x half.
        ``x < 0  →  (x, y) = (-x, -y)``
        ``|x| ≈ 0 →  y = |y|``

    ``restrict_positive_y``:
        Discard points where ``y < 0``.  Implemented by setting discarded
        points to NaN; callers must filter them out.

    ``shift_x``:
        Add a fixed offset to x and wrap to ``[-180, 180)``.
        ``x = wrap_signed(x + value)``
    """

    type: str
    value: float = 0.0


_VALID_PAIR_TRANSFORM_TYPES = frozenset(
    {"center_x_at", "fold_sign_symmetry", "restrict_positive_y", "shift_x"}
)


# ---------------------------------------------------------------------------
# DoF definition resolution
# ---------------------------------------------------------------------------


def resolve_dof_definitions(
    config: dict[str, Any],
    *,
    include_disabled: bool = False,
) -> list[DoFDefinition]:
    """Parse ``config['dof']`` into a list of :class:`DoFDefinition` objects.

    Parameters
    ----------
    config:
        Parsed project config dict containing a ``"dof"`` key.
    include_disabled:
        When False (default), only enabled definitions are returned.

    Returns
    -------
    list[DoFDefinition]
        Ordered list of validated DoF definitions; empty list if none configured.

    Raises
    ------
    ValueError
        If any DoF entry is malformed or has an unsupported type.
    KeyError
        If a required field (``name``, ``type``) is missing.
    """
    raw_list = config.get("dof", []) or []
    result: list[DoFDefinition] = []
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            raise ValueError(f"dof entry {i} must be a dict; got {type(raw).__name__}")
        name = str(raw["name"])
        dof_type = str(raw["type"])

        atoms_raw = raw.get("atoms")
        atoms: tuple[int, ...] | None = (
            tuple(int(a) for a in atoms_raw) if atoms_raw is not None else None
        )

        domain_raw = raw.get("domain")
        if domain_raw is None:
            raise ValueError(f"DoF '{name}' is missing required 'domain' field.")
        domain: tuple[float, float] = (float(domain_raw[0]), float(domain_raw[1]))

        input_dof_raw = raw.get("input_dof")
        input_dof: tuple[str, ...] | None = (
            tuple(str(s) for s in input_dof_raw) if input_dof_raw is not None else None
        )

        dof = DoFDefinition(
            name=name,
            type=dof_type,  # type: ignore[arg-type]
            label=str(raw.get("label", name)),
            domain=domain,
            enabled=bool(raw.get("enabled", True)),
            atoms=atoms,
            convention=str(raw.get("convention", "signed")),
            method=raw.get("method"),
            input_dof=input_dof,
            component=raw.get("component"),
            source_column=raw.get("source_column"),
        )
        if not include_disabled and not dof.enabled:
            continue
        result.append(dof)
    return result


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------


def resolve_coordinate_transforms(config: dict[str, Any]) -> dict[str, float]:
    """Return angular-shift transforms from config as ``{column: shift_degrees}``.

    Reads ``config["coordinate_transforms"]``.  Each value may be a bare
    number or a dict with a ``"shift"`` key::

        coordinate_transforms:
          carboxyl_dihedral: 90.0
          ester_dihedral: {shift: 45.0}

    Parameters
    ----------
    config:
        Parsed project config dict.

    Returns
    -------
    dict[str, float]
        Mapping of column name → shift in degrees.  Empty dict when none configured.
    """
    raw = config.get("coordinate_transforms") or {}
    result: dict[str, float] = {}
    for col, val in raw.items():
        if val is None:
            continue
        if isinstance(val, dict):
            shift = val.get("shift")
            if shift is None:
                continue
            result[col] = float(shift)
        else:
            result[col] = float(val)
    return result


# ---------------------------------------------------------------------------
# Pair transform resolution
# ---------------------------------------------------------------------------


def resolve_pair_transforms(raw_list: list | None) -> list[PairTransformSpec]:
    """Parse a ``pair_transforms`` YAML list into :class:`PairTransformSpec` objects.

    Each entry in *raw_list* must be a dict with a ``"type"`` key and an
    optional ``"value"`` key::

        pair_transforms:
          - {type: center_x_at,      value: 90.0}
          - {type: fold_sign_symmetry}
          - {type: restrict_positive_y}
          - {type: shift_x,          value: -90.0}

    Raises
    ------
    ValueError
        When an entry is missing ``"type"`` or uses an unsupported type.
    """
    if not raw_list:
        return []
    result: list[PairTransformSpec] = []
    for i, entry in enumerate(raw_list):
        if not isinstance(entry, dict):
            raise ValueError(
                f"pair_transforms entry {i} must be a dict; got {type(entry).__name__}"
            )
        t = entry.get("type")
        if t is None:
            raise ValueError(f"pair_transforms entry {i} is missing the 'type' key.")
        if t not in _VALID_PAIR_TRANSFORM_TYPES:
            raise ValueError(
                f"pair_transforms entry {i} has unknown type {t!r}. "
                f"Valid types: {sorted(_VALID_PAIR_TRANSFORM_TYPES)}"
            )
        value = float(entry.get("value", 0.0))
        result.append(PairTransformSpec(type=str(t), value=value))
    return result


# ---------------------------------------------------------------------------
# Coordinate pair resolution
# ---------------------------------------------------------------------------


def _effective_feature_col(col: str, transforms: dict[str, float]) -> str:
    """Return the shifted column name if a transform is active, else *col*."""
    if col in transforms:
        return f"{col}_shifted"
    return col


def resolve_coordinate_pair(
    pair_cfg: dict[str, Any],
    dof_map: dict[str, DoFDefinition],
    config: dict[str, Any],
) -> CoordinatePair:
    """Build a :class:`CoordinatePair` from a pair config dict and DoF map.

    Domain and label are inherited from the referenced :class:`DoFDefinition`
    objects unless explicitly overridden in *pair_cfg*.

    Parameters
    ----------
    pair_cfg:
        One entry from ``config["coordinate_pairs"]``.  Must contain ``name``,
        ``x``, and ``y`` keys referencing DoF names.
    dof_map:
        ``{dof_name: DoFDefinition}`` mapping built from ``resolve_dof_definitions``.
    config:
        Full project config dict (used to resolve coordinate transforms).

    Returns
    -------
    CoordinatePair

    Raises
    ------
    KeyError
        If ``x`` or ``y`` DoF names are not found in *dof_map*.
    ValueError
        If ``name``, ``x``, or ``y`` keys are missing from *pair_cfg*.
    """
    name = str(pair_cfg["name"])
    x_dof_name = str(pair_cfg["x"])
    y_dof_name = str(pair_cfg["y"])

    if x_dof_name not in dof_map:
        raise KeyError(
            f"Coordinate pair '{name}': x DoF '{x_dof_name}' not found in dof list. "
            f"Available: {list(dof_map.keys())}"
        )
    if y_dof_name not in dof_map:
        raise KeyError(
            f"Coordinate pair '{name}': y DoF '{y_dof_name}' not found in dof list. "
            f"Available: {list(dof_map.keys())}"
        )

    x_dof = dof_map[x_dof_name]
    y_dof = dof_map[y_dof_name]

    x_col = x_dof_name
    y_col = y_dof_name
    x_label = str(pair_cfg.get("x_label", x_dof.label))
    y_label = str(pair_cfg.get("y_label", y_dof.label))
    title = str(pair_cfg.get("title", f"{x_label} vs {y_label}"))
    x_domain: tuple[float, float] = (
        tuple(float(v) for v in pair_cfg["x_domain"])  # type: ignore[assignment]
        if "x_domain" in pair_cfg
        else x_dof.domain
    )
    y_domain: tuple[float, float] = (
        tuple(float(v) for v in pair_cfg["y_domain"])  # type: ignore[assignment]
        if "y_domain" in pair_cfg
        else y_dof.domain
    )

    transforms = resolve_coordinate_transforms(config)
    feature_x = _effective_feature_col(x_col, transforms)
    feature_y = _effective_feature_col(y_col, transforms)

    bins: int | None = int(pair_cfg["bins"]) if pair_cfg.get("bins") is not None else None
    x_range_raw = pair_cfg.get("x_range")
    x_range: tuple[float, float] | None = (
        (float(x_range_raw[0]), float(x_range_raw[1])) if x_range_raw is not None else None
    )
    y_range_raw = pair_cfg.get("y_range")
    y_range: tuple[float, float] | None = (
        (float(y_range_raw[0]), float(y_range_raw[1])) if y_range_raw is not None else None
    )
    colormap = str(pair_cfg["colormap"]) if pair_cfg.get("colormap") is not None else None
    log_scale = bool(pair_cfg["log_scale"]) if pair_cfg.get("log_scale") is not None else None

    # Both axes periodic when both DoF are dihedral (angle DoF are not periodic)
    periodic = (x_dof.type == "dihedral") and (y_dof.type == "dihedral")

    return CoordinatePair(
        name=name,
        x_col=x_col,
        y_col=y_col,
        x_label=x_label,
        y_label=y_label,
        title=title,
        x_domain=x_domain,
        y_domain=y_domain,
        feature_x_col=feature_x if feature_x != x_col else None,
        feature_y_col=feature_y if feature_y != y_col else None,
        periodic=periodic,
        bins=bins,
        x_range=x_range,
        y_range=y_range,
        colormap=colormap,
        log_scale=log_scale,
        x_atoms=x_dof.atoms,
        y_atoms=y_dof.atoms,
        x_dof_type=x_dof.type,
        y_dof_type=y_dof.type,
    )


def list_coordinate_pairs(
    config: dict[str, Any],
) -> list[tuple[str, CoordinatePair]]:
    """Return all configured coordinate pairs as ``(name, CoordinatePair)`` tuples.

    Reads ``config["coordinate_pairs"]`` in order.  Only enabled DoF are
    included via the DoF map.

    Parameters
    ----------
    config:
        Full project config dict.

    Returns
    -------
    list[tuple[str, CoordinatePair]]
        Ordered list of ``(name, CoordinatePair)`` tuples.

    Raises
    ------
    KeyError
        If any pair references a DoF name not found in the DoF list.
    """
    dof_defs = resolve_dof_definitions(config, include_disabled=True)
    dof_map = {d.name: d for d in dof_defs}

    raw_pairs = config.get("coordinate_pairs") or []
    # Support both list and dict forms
    if isinstance(raw_pairs, dict):
        raw_pairs = [{"name": k, **v} for k, v in raw_pairs.items()]

    result: list[tuple[str, CoordinatePair]] = []
    for entry in raw_pairs:
        if not isinstance(entry, dict):
            raise ValueError(f"coordinate_pairs entry must be a dict; got {type(entry).__name__}")
        pair = resolve_coordinate_pair(entry, dof_map, config)
        result.append((pair.name, pair))
    return result
