"""Energy units for population-derived free-energy-like values.

One table of Boltzmann constants, defined here once. The interactive page
embeds it as JSON (``energy_unit_table``), so the values the page computes
in the browser and the values Python bakes into the initial figure use the
same constants.

A free-energy-like value is ``F = k_B · T · (−ln(P / P_max))``. The unit
``kT`` keeps it dimensionless (``F / k_B T``) and ignores the temperature.

Public API
----------
- ``ENERGY_UNITS``
- ``unit_label``
- ``validate_energy_unit``
- ``validate_temperature``
- ``thermal_energy``
- ``energy_unit_table``
"""

from __future__ import annotations

import math

# k_B per kelvin in each unit (CODATA 2018 exact k_B, h, c, N_A).
# kT has no constant: values stay in units of k_B T.
_BOLTZMANN_PER_KELVIN: dict[str, float | None] = {
    "kT": None,
    "kJ/mol": 8.314462618e-3,
    "kcal/mol": 8.314462618e-3 / 4.184,
    "eV": 8.617333262e-5,
    "meV": 8.617333262e-2,
    "cm^-1": 0.6950348005,
}

_UNIT_LABELS = {"cm^-1": "cm⁻¹"}

ENERGY_UNITS: tuple[str, ...] = tuple(_BOLTZMANN_PER_KELVIN)
"""Accepted unit keys, in display order."""


def unit_label(unit: str) -> str:
    """Return the display label for *unit* (e.g. ``"cm^-1"`` → ``"cm⁻¹"``)."""
    return _UNIT_LABELS.get(unit, unit)


def validate_energy_unit(unit: object, source: str = "unit") -> str:
    """Return *unit* if it is a known unit key.

    Raises
    ------
    ValueError
        If *unit* is not one of :data:`ENERGY_UNITS`. *source* names the
        config key in the message.
    """
    if not isinstance(unit, str) or unit not in _BOLTZMANN_PER_KELVIN:
        raise ValueError(
            f"{source} must be one of {list(ENERGY_UNITS)}, got {unit!r}."
        )
    return unit


def validate_temperature(temperature: object, source: str = "temperature") -> float:
    """Return *temperature* (K) as a float if it is a finite number > 0.

    Raises
    ------
    ValueError
        If *temperature* is not a number, not finite, or not positive.
        *source* names the config key in the message.
    """
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(f"{source} must be a number in kelvin, got {temperature!r}.")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{source} must be a positive temperature in kelvin, got {temperature!r}.")
    return value


def thermal_energy(unit: str, temperature: float | None) -> float:
    """Return ``k_B · T`` in *unit*; 1.0 for ``kT``, which ignores *temperature*.

    Raises
    ------
    ValueError
        If *unit* is unknown, or *unit* is not ``kT`` and *temperature* is
        missing or not positive.
    """
    k_b = _BOLTZMANN_PER_KELVIN[validate_energy_unit(unit)]
    if k_b is None:
        return 1.0
    if temperature is None:
        raise ValueError(f"A temperature is required to express energies in {unit}.")
    return k_b * validate_temperature(temperature)


def energy_unit_table() -> list[dict]:
    """Return the unit table for embedding as JSON.

    Each entry is ``{"key", "label", "k_B"}``; ``k_B`` is per kelvin in that
    unit, or ``None`` for ``kT``.
    """
    return [
        {"key": unit, "label": unit_label(unit), "k_B": k_b}
        for unit, k_b in _BOLTZMANN_PER_KELVIN.items()
    ]
