"""Angle-domain helpers shared across coordinate, plotting, and state code."""

from __future__ import annotations

from typing import Any, Mapping


def plane_is_signed(conventions: Mapping[str, Any] | None = None) -> bool:
    """Return True when plane coordinates should use the signed convention."""
    return bool((conventions or {}).get("plane_signed", False))


def plane_angle_domain(conventions: Mapping[str, Any] | None = None) -> tuple[float, float]:
    """Return the effective domain for plane coordinates.

    Signed plane angles use the full ``[-180, 180)`` domain so the
    plane-density landscape preserves the distinction between near-coplanar and
    near-antiparallel orientations.
    """
    if plane_is_signed(conventions):
        return (-180.0, 180.0)
    return (0.0, 180.0)


def definition_angle_domain(
    definition: str,
    conventions: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    """Return the effective angular domain for one coordinate definition."""
    if definition == "plane":
        return plane_angle_domain(conventions)
    if definition == "dihedral":
        return (-180.0, 180.0)
    raise ValueError(
        f"Unknown definition '{definition}'. Valid choices: ['plane', 'dihedral']"
    )


def wrap_signed_degrees(values):
    """Wrap degree values into the canonical [-180, 180) interval."""
    return ((values + 180.0) % 360.0) - 180.0
