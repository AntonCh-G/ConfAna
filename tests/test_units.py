"""Tests for confana/units.py (energy units for free-energy-like values)."""

from __future__ import annotations

import math

import pytest

from confana.units import (
    ENERGY_UNITS,
    energy_unit_table,
    thermal_energy,
    unit_label,
    validate_energy_unit,
    validate_temperature,
)


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("kJ/mol", 2.494),
        ("kcal/mol", 0.5962),
        ("eV", 0.02585),
        ("meV", 25.85),
        ("cm^-1", 208.5),
    ],
)
def test_thermal_energy_at_300_k_matches_reference(unit, expected):
    assert thermal_energy(unit, 300.0) == pytest.approx(expected, rel=5e-4)


def test_thermal_energy_kt_ignores_temperature():
    assert thermal_energy("kT", None) == 1.0
    assert thermal_energy("kT", 500.0) == 1.0


def test_units_are_consistent_with_each_other():
    kj = thermal_energy("kJ/mol", 300.0)
    assert thermal_energy("kcal/mol", 300.0) == pytest.approx(kj / 4.184)
    assert thermal_energy("meV", 300.0) == pytest.approx(1000.0 * thermal_energy("eV", 300.0))
    # 1 eV = 96.485 kJ/mol
    assert kj / thermal_energy("eV", 300.0) == pytest.approx(96.485, rel=1e-4)


@pytest.mark.parametrize("unit", ["kj/mol", "Hartree", "", None, 1])
def test_unknown_unit_raises(unit):
    with pytest.raises(ValueError, match="must be one of"):
        validate_energy_unit(unit, source="some.key")
    if isinstance(unit, str):
        with pytest.raises(ValueError):
            thermal_energy(unit, 300.0)


@pytest.mark.parametrize("temperature", [0, -10.0, math.nan, math.inf, "300", True])
def test_invalid_temperature_raises(temperature):
    with pytest.raises(ValueError, match="some.key"):
        validate_temperature(temperature, source="some.key")
    with pytest.raises(ValueError):
        thermal_energy("kJ/mol", temperature)


def test_thermal_energy_needs_temperature_for_energy_units():
    with pytest.raises(ValueError, match="temperature is required"):
        thermal_energy("eV", None)


def test_energy_unit_table_lists_every_unit_in_order():
    table = energy_unit_table()
    assert [u["key"] for u in table] == list(ENERGY_UNITS)
    assert ENERGY_UNITS[0] == "kT"
    assert table[0]["k_B"] is None
    assert all(u["k_B"] > 0 for u in table[1:])
    assert unit_label("cm^-1") == "cm⁻¹"
    assert {u["key"]: u["label"] for u in table}["cm^-1"] == "cm⁻¹"
