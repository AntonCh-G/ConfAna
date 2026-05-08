"""Config helpers for named coordinates and 2D analysis axis selection."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from src.models import CoordinatePair, DihedralDefinition

DEFAULT_CARBOXYL_DIHEDRAL_IDS: tuple[int, int, int, int] = (6, 5, 10, 7)
DEFAULT_ESTER_DIHEDRAL_IDS: tuple[int, int, int, int] = (5, 6, 12, 11)

_SUPPORTED_DIHEDRAL_CONVENTIONS = {None, "signed"}

DEFAULT_DIHEDRAL_DEFINITIONS: tuple[DihedralDefinition, ...] = (
    DihedralDefinition(
        name="carboxyl_dihedral",
        atoms=DEFAULT_CARBOXYL_DIHEDRAL_IDS,
        label="Carboxyl dihedral",
        convention="signed",
        enabled=True,
        group="core",
    ),
    DihedralDefinition(
        name="ester_dihedral",
        atoms=DEFAULT_ESTER_DIHEDRAL_IDS,
        label="Ester dihedral",
        convention="signed",
        enabled=True,
        group="core",
    ),
)

DEFAULT_PLANE_PAIR = CoordinatePair(
    definition="plane",
    x_col="carboxyl_plane",
    y_col="ester_plane",
    x_label="Carboxyl plane angle (°)",
    y_label="Ester plane angle (°)",
    title="Plane-angle density",
)

DEFAULT_DIHEDRAL_PAIR = CoordinatePair(
    definition="dihedral",
    x_col="carboxyl_dihedral",
    y_col="ester_dihedral",
    x_label="Carboxyl dihedral (°)",
    y_label="Ester dihedral (°)",
    title="Dihedral-angle density",
)

_PLANE_COLUMNS = {
    DEFAULT_PLANE_PAIR.x_col,
    DEFAULT_PLANE_PAIR.y_col,
}

def _humanize_name(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def _coordinate_pairs_section(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the raw ``coordinate_pairs`` config section."""
    if config is None:
        return {}
    return config.get("coordinate_pairs", {}) or {}


def _legacy_dihedral_definitions(mapping: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if mapping is None:
        return []

    legacy_defs: list[dict[str, Any]] = []
    for name, atoms in mapping.items():
        if not isinstance(name, str) or not name.endswith("_dihedral"):
            continue
        legacy_defs.append(
            {
                "name": name,
                "atoms": atoms,
                "label": _humanize_name(name),
                "convention": "signed",
                "enabled": True,
            }
        )
    return legacy_defs


def _extract_raw_dihedral_definitions(source: Mapping[str, Any] | None) -> tuple[list[Any], bool]:
    """Return ``(raw_defs, explicit)`` from config/mapping-like input."""
    if source is None:
        return [], False

    if "dihedral_definitions" in source:
        return list(source.get("dihedral_definitions") or []), True

    raw = source.get("dihedrals")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return list(raw), True

    if "atom_mapping" in source:
        atom_mapping = source.get("atom_mapping", {}) or {}
        if "dihedral_definitions" in atom_mapping:
            return list(atom_mapping.get("dihedral_definitions") or []), True
        raw = atom_mapping.get("dihedrals")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return list(raw), True
        return _legacy_dihedral_definitions(atom_mapping), False

    return _legacy_dihedral_definitions(source), False


def _coerce_dihedral_definition(raw: Any, index: int) -> DihedralDefinition:
    if isinstance(raw, DihedralDefinition):
        definition = raw
    elif isinstance(raw, Mapping):
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Dihedral definition #{index} is missing a valid string 'name'."
            )

        atoms = raw.get("atoms")
        if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)):
            raise ValueError(
                f"Dihedral '{name}' must define 'atoms' as a sequence of four indices."
            )
        if len(atoms) != 4:
            raise ValueError(
                f"Dihedral '{name}' must define exactly 4 atom indices; got {len(atoms)}."
            )

        atom_ids: list[int] = []
        for atom in atoms:
            try:
                atom_id = int(atom)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dihedral '{name}' contains a non-integer atom index: {atom!r}."
                ) from exc
            if atom_id < 0:
                raise ValueError(
                    f"Dihedral '{name}' contains a negative atom index: {atom_id}."
                )
            atom_ids.append(atom_id)

        label = raw.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError(f"Dihedral '{name}' has a non-string label: {label!r}.")

        convention = raw.get("convention")
        if convention is not None:
            convention = str(convention)
        if convention not in _SUPPORTED_DIHEDRAL_CONVENTIONS:
            raise ValueError(
                f"Dihedral '{name}' uses unsupported convention {convention!r}. "
                f"Supported values: {sorted(v for v in _SUPPORTED_DIHEDRAL_CONVENTIONS if v is not None)}"
            )

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Dihedral '{name}' must define 'enabled' as true/false, got {enabled!r}."
            )

        group = raw.get("group")
        if group is not None and not isinstance(group, str):
            raise ValueError(f"Dihedral '{name}' has a non-string group: {group!r}.")

        definition = DihedralDefinition(
            name=name.strip(),
            atoms=tuple(atom_ids),  # type: ignore[arg-type]
            label=label,
            convention=convention,
            enabled=enabled,
            group=group,
        )
    else:
        raise ValueError(
            f"Dihedral definition #{index} must be a mapping, got {type(raw).__name__}."
        )

    return definition


def resolve_dihedral_definitions(
    source: Mapping[str, Any] | None,
    *,
    include_disabled: bool = False,
) -> list[DihedralDefinition]:
    """Return validated dihedral definitions from config or legacy mapping."""
    raw_defs, explicit = _extract_raw_dihedral_definitions(source)
    if not explicit and not raw_defs:
        raw_defs = list(DEFAULT_DIHEDRAL_DEFINITIONS)

    definitions = [_coerce_dihedral_definition(raw, i) for i, raw in enumerate(raw_defs)]

    seen: set[str] = set()
    for definition in definitions:
        if definition.name in seen:
            raise ValueError(f"Duplicate dihedral definition name: {definition.name!r}.")
        seen.add(definition.name)

    if include_disabled:
        selected = definitions
    else:
        selected = [definition for definition in definitions if definition.enabled]

    if explicit and not selected:
        raise ValueError("No enabled dihedral definitions were configured.")

    return selected


def serialize_dihedral_definitions(
    definitions: Sequence[DihedralDefinition],
) -> list[dict[str, Any]]:
    """Return JSON-serializable dictionaries for cache metadata/runtime mapping."""
    serialized: list[dict[str, Any]] = []
    for definition in definitions:
        payload = asdict(definition)
        payload["atoms"] = list(definition.atoms)
        serialized.append(payload)
    return serialized


def build_runtime_mapping(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return an atom-mapping dict augmented with normalized dihedral definitions."""
    if source is None:
        base_mapping: dict[str, Any] = {}
    elif "atom_mapping" in source:
        base_mapping = dict(source.get("atom_mapping", {}) or {})
    else:
        base_mapping = dict(source)

    base_mapping["dihedral_definitions"] = serialize_dihedral_definitions(
        resolve_dihedral_definitions(source, include_disabled=True)
    )
    return base_mapping


def resolve_dihedral_atom_ids(
    source: Mapping[str, Any] | None,
    name: str,
    default: Sequence[int] | None = None,
) -> list[int]:
    """Return the atom ids for one named dihedral, with an optional fallback."""
    for definition in resolve_dihedral_definitions(source, include_disabled=True):
        if definition.name == name:
            return list(definition.atoms)

    if default is not None:
        return [int(atom_id) for atom_id in default]

    raise KeyError(f"Unknown dihedral definition: {name!r}")


def _strip_shifted(col: str) -> str:
    """Return the raw (unshifted) column name for a possibly-shifted column."""
    return col[: -len("_shifted")] if col.endswith("_shifted") else col


def resolve_coordinate_pair(
    definition: str,
    config: Mapping[str, Any] | None = None,
) -> CoordinatePair:
    """Resolve one named 2D analysis pair from config.

    Supported names:
    - ``"plane"``: built-in plane pair
    - ``"dihedral"``: canonical dihedral pair
    - any extra key under ``coordinate_pairs``, e.g. ``"dihedrals_igor"``

    If ``coordinate_transforms`` is configured in *config*, the returned
    ``CoordinatePair`` will use the ``*_shifted`` column names for any axis
    that has a configured shift.  This is transparent to callers — plots and
    clustering code simply receive the correct column name to read.
    """
    transforms = resolve_coordinate_transforms(config)

    if definition == "plane":
        pair_cfg = _coordinate_pairs_section(config).get("plane", {}) or {}
        x_col = str(pair_cfg.get("x", DEFAULT_PLANE_PAIR.x_col))
        y_col = str(pair_cfg.get("y", DEFAULT_PLANE_PAIR.y_col))
        # Validate against raw plane column names (strip _shifted suffix first)
        missing = [
            column_name
            for column_name in (x_col, y_col)
            if _strip_shifted(column_name) not in _PLANE_COLUMNS
        ]
        if missing:
            raise ValueError(
                "coordinate_pairs.plane references unsupported plane column(s): "
                f"{missing}"
            )
        x_col = effective_column(x_col, transforms)
        y_col = effective_column(y_col, transforms)
        return CoordinatePair(
            definition="plane",
            x_col=x_col,
            y_col=y_col,
            x_label=str(pair_cfg.get("x_label") or DEFAULT_PLANE_PAIR.x_label),
            y_label=str(pair_cfg.get("y_label") or DEFAULT_PLANE_PAIR.y_label),
            title=str(pair_cfg.get("title") or DEFAULT_PLANE_PAIR.title),
        )

    pair_cfg = _coordinate_pairs_section(config).get(definition, {}) or {}
    if definition != "dihedral" and not pair_cfg:
        valid = ["plane", "dihedral", *_coordinate_pairs_section(config).keys()]
        raise ValueError(
            f"Unknown definition '{definition}'. Valid choices: {list(dict.fromkeys(valid))}"
        )

    x_col = str(pair_cfg.get("x", DEFAULT_DIHEDRAL_PAIR.x_col))
    y_col = str(pair_cfg.get("y", DEFAULT_DIHEDRAL_PAIR.y_col))

    pair_family = str(pair_cfg.get("definition") or pair_cfg.get("family") or "").strip().lower()
    if not pair_family:
        # Auto-detect: check raw (un-shifted) column names against plane set
        if {_strip_shifted(x_col), _strip_shifted(y_col)} <= _PLANE_COLUMNS:
            pair_family = "plane"
        else:
            pair_family = "dihedral"

    if pair_family == "plane":
        missing = [
            column_name
            for column_name in (x_col, y_col)
            if _strip_shifted(column_name) not in _PLANE_COLUMNS
        ]
        if missing:
            raise ValueError(
                f"coordinate_pairs.{definition} references unsupported plane column(s): {missing}"
            )
        x_col = effective_column(x_col, transforms)
        y_col = effective_column(y_col, transforms)
        default_title = (
            DEFAULT_PLANE_PAIR.title
            if (_strip_shifted(x_col), _strip_shifted(y_col))
            == (DEFAULT_PLANE_PAIR.x_col, DEFAULT_PLANE_PAIR.y_col)
            else f"{_humanize_name(_strip_shifted(x_col))} (°) vs {_humanize_name(_strip_shifted(y_col))} (°)"
        )
        return CoordinatePair(
            definition="plane",
            x_col=x_col,
            y_col=y_col,
            x_label=str(
                pair_cfg.get("x_label") or f"{_humanize_name(_strip_shifted(x_col))} (°)"
            ),
            y_label=str(
                pair_cfg.get("y_label") or f"{_humanize_name(_strip_shifted(y_col))} (°)"
            ),
            title=str(pair_cfg.get("title") or default_title),
        )

    if pair_family != "dihedral":
        raise ValueError(
            f"coordinate_pairs.{definition} uses unsupported definition {pair_family!r}."
        )

    definitions_by_name = {
        dihedral_definition.name: dihedral_definition
        for dihedral_definition in resolve_dihedral_definitions(config, include_disabled=False)
    }
    # Validate against raw (un-shifted) column names
    missing = [
        column_name
        for column_name in (x_col, y_col)
        if _strip_shifted(column_name) not in definitions_by_name
    ]
    if missing:
        raise ValueError(
            f"coordinate_pairs.{definition} references unknown or disabled dihedral column(s): "
            f"{missing}"
        )

    x_raw = _strip_shifted(x_col)
    y_raw = _strip_shifted(y_col)
    x_def = definitions_by_name[x_raw]
    y_def = definitions_by_name[y_raw]
    x_label = str(pair_cfg.get("x_label") or f"{x_def.label or _humanize_name(x_raw)} (°)")
    y_label = str(pair_cfg.get("y_label") or f"{y_def.label or _humanize_name(y_raw)} (°)")
    default_title = (
        DEFAULT_DIHEDRAL_PAIR.title
        if (x_raw, y_raw) == (DEFAULT_DIHEDRAL_PAIR.x_col, DEFAULT_DIHEDRAL_PAIR.y_col)
        else f"{x_label} vs {y_label}"
    )
    title = str(pair_cfg.get("title") or default_title)

    x_col = effective_column(x_col, transforms)
    y_col = effective_column(y_col, transforms)

    return CoordinatePair(
        definition="dihedral",
        x_col=x_col,
        y_col=y_col,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )


def resolve_coordinate_transforms(
    config: "Mapping[str, Any] | None",
) -> dict[str, float]:
    """Return a ``{column_name: shift_degrees}`` mapping from config.

    Reads the optional ``coordinate_transforms`` section.  Each entry may be:

    * a bare number: ``carboxyl_dihedral: 90.0``
    * a dict with a ``shift`` key: ``carboxyl_dihedral: {shift: 90.0}``

    Entries with a null value or a zero shift are included (callers that only
    want *non-trivial* transforms should filter on ``shift != 0`` themselves).
    Absent or empty sections return an empty dict.

    Parameters
    ----------
    config:
        Full config mapping (e.g. loaded from ``default.yaml``).

    Returns
    -------
    dict[str, float]
        Mapping of column name → shift in degrees.
    """
    if config is None:
        return {}
    section = config.get("coordinate_transforms") or {}
    if not isinstance(section, Mapping):
        return {}
    result: dict[str, float] = {}
    for col, spec in section.items():
        if not isinstance(col, str):
            continue
        if spec is None:
            continue
        if isinstance(spec, (int, float)):
            result[col] = float(spec)
        elif isinstance(spec, Mapping):
            raw_shift = spec.get("shift")
            if raw_shift is None:
                continue
            result[col] = float(raw_shift)
    return result


def effective_column(col: str, transforms: "dict[str, float]") -> str:
    """Return the effective column name, accounting for any configured shift.

    If *col* has a configured transform (regardless of whether the shift value
    is zero), the ``*_shifted`` companion column is returned.  Otherwise the
    raw column name is returned unchanged.

    Parameters
    ----------
    col:
        Raw column name, e.g. ``"carboxyl_dihedral"``.
    transforms:
        Mapping returned by :func:`resolve_coordinate_transforms`.

    Returns
    -------
    str
        ``f"{col}_shifted"`` when *col* is in *transforms*, else *col*.
    """
    return f"{col}_shifted" if col in transforms else col


def list_coordinate_pairs(
    config: Mapping[str, Any] | None = None,
) -> list[tuple[str, CoordinatePair]]:
    """Return all named coordinate pairs in output order.

    The canonical ``plane`` and ``dihedral`` pairs are always included.
    Additional configured pairs are appended in config order.
    """
    names = ["plane", "dihedral"]
    for pair_name in _coordinate_pairs_section(config):
        if pair_name not in names:
            names.append(str(pair_name))
    return [(pair_name, resolve_coordinate_pair(pair_name, config)) for pair_name in names]
