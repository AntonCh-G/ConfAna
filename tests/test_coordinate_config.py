"""Tests for config-driven DoF definitions and pair selection."""

from __future__ import annotations

import pytest

from src.coordinate_config import (
    list_coordinate_pairs,
    resolve_coordinate_pair,
    resolve_dof_definitions,
)


def _multi_dof_config() -> dict:
    return {
        "dof": [
            {
                "name": "carboxyl_dihedral",
                "type": "dihedral",
                "atoms": [6, 5, 10, 7],
                "label": "Carboxyl dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
            {
                "name": "ester_dihedral",
                "type": "dihedral",
                "atoms": [5, 6, 12, 11],
                "label": "Ester dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
            {
                "name": "igor1_dihedral",
                "type": "dihedral",
                "atoms": [6, 12, 11, 8],
                "label": "Igor 1 dihedral (°)",
                "domain": [-180, 180],
                "enabled": True,
            },
        ],
        "coordinate_pairs": [
            {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"},
        ],
    }


def test_resolve_dof_definitions_supports_multiple_entries():
    definitions = resolve_dof_definitions(_multi_dof_config())
    assert [d.name for d in definitions] == [
        "carboxyl_dihedral",
        "ester_dihedral",
        "igor1_dihedral",
    ]


def test_resolve_dof_definitions_rejects_missing_domain():
    cfg = _multi_dof_config()
    # Remove domain from first entry
    cfg["dof"][0] = {k: v for k, v in cfg["dof"][0].items() if k != "domain"}
    with pytest.raises(ValueError, match="missing required 'domain'"):
        resolve_dof_definitions(cfg)


def test_resolve_dof_definitions_rejects_malformed_entry():
    cfg = {"dof": ["not_a_dict"]}
    with pytest.raises(ValueError, match="must be a dict"):
        resolve_dof_definitions(cfg)


def test_resolve_dof_definitions_filters_disabled():
    cfg = _multi_dof_config()
    cfg["dof"][2]["enabled"] = False
    definitions = resolve_dof_definitions(cfg)
    assert [d.name for d in definitions] == ["carboxyl_dihedral", "ester_dihedral"]


def test_resolve_dof_definitions_include_disabled():
    cfg = _multi_dof_config()
    cfg["dof"][2]["enabled"] = False
    definitions = resolve_dof_definitions(cfg, include_disabled=True)
    assert len(definitions) == 3


def test_resolve_dof_definitions_empty_list():
    definitions = resolve_dof_definitions({"dof": []})
    assert definitions == []


def test_resolve_dof_definitions_missing_key_returns_empty():
    definitions = resolve_dof_definitions({})
    assert definitions == []


def test_resolve_dof_definitions_domain_stored_correctly():
    definitions = resolve_dof_definitions(_multi_dof_config())
    assert definitions[0].domain == (-180.0, 180.0)


def test_resolve_dof_definitions_type_stored_correctly():
    definitions = resolve_dof_definitions(_multi_dof_config())
    assert definitions[0].type == "dihedral"


def test_resolve_coordinate_pair_uses_configured_pair():
    cfg = _multi_dof_config()
    dof_defs = resolve_dof_definitions(cfg, include_disabled=True)
    dof_map = {d.name: d for d in dof_defs}
    pair_cfg = {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"}
    pair = resolve_coordinate_pair(pair_cfg, dof_map, cfg)
    assert pair.x_col == "carboxyl_dihedral"
    assert pair.y_col == "ester_dihedral"


def test_resolve_coordinate_pair_custom_labels():
    cfg = _multi_dof_config()
    dof_defs = resolve_dof_definitions(cfg, include_disabled=True)
    dof_map = {d.name: d for d in dof_defs}
    pair_cfg = {
        "name": "custom",
        "x": "igor1_dihedral",
        "y": "ester_dihedral",
        "x_label": "Igor torsion (°)",
        "y_label": "Ester torsion (°)",
        "title": "Custom torsion map",
    }
    pair = resolve_coordinate_pair(pair_cfg, dof_map, cfg)
    assert pair.x_col == "igor1_dihedral"
    assert pair.y_col == "ester_dihedral"
    assert pair.title == "Custom torsion map"


def test_resolve_coordinate_pair_inherits_domain_from_dof():
    cfg = _multi_dof_config()
    dof_defs = resolve_dof_definitions(cfg, include_disabled=True)
    dof_map = {d.name: d for d in dof_defs}
    pair_cfg = {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"}
    pair = resolve_coordinate_pair(pair_cfg, dof_map, cfg)
    assert pair.x_domain == (-180.0, 180.0)
    assert pair.y_domain == (-180.0, 180.0)


def test_resolve_coordinate_pair_copies_dof_atoms_and_types():
    cfg = _multi_dof_config()
    dof_defs = resolve_dof_definitions(cfg, include_disabled=True)
    dof_map = {d.name: d for d in dof_defs}
    pair_cfg = {"name": "dihedral", "x": "carboxyl_dihedral", "y": "ester_dihedral"}
    pair = resolve_coordinate_pair(pair_cfg, dof_map, cfg)
    assert pair.x_atoms == (6, 5, 10, 7)
    assert pair.y_atoms == (5, 6, 12, 11)
    assert pair.x_dof_type == "dihedral"
    assert pair.y_dof_type == "dihedral"


def test_resolve_coordinate_pair_unknown_dof_raises():
    cfg = _multi_dof_config()
    dof_defs = resolve_dof_definitions(cfg, include_disabled=True)
    dof_map = {d.name: d for d in dof_defs}
    pair_cfg = {"name": "bad", "x": "nonexistent_dof", "y": "ester_dihedral"}
    with pytest.raises(KeyError):
        resolve_coordinate_pair(pair_cfg, dof_map, cfg)


def test_list_coordinate_pairs_returns_named_tuples():
    cfg = _multi_dof_config()
    pairs = list_coordinate_pairs(cfg)
    assert len(pairs) == 1
    name, pair = pairs[0]
    assert name == "dihedral"
    assert pair.x_col == "carboxyl_dihedral"


def test_list_coordinate_pairs_includes_custom_pairs():
    cfg = _multi_dof_config()
    cfg["coordinate_pairs"].append(
        {"name": "igor_pair", "x": "igor1_dihedral", "y": "ester_dihedral"}
    )
    names = [name for name, _ in list_coordinate_pairs(cfg)]
    assert names == ["dihedral", "igor_pair"]


def test_list_coordinate_pairs_dict_form():
    """coordinate_pairs as a dict (legacy form) is also accepted."""
    cfg = {
        "dof": _multi_dof_config()["dof"],
        "coordinate_pairs": {
            "dihedral": {"x": "carboxyl_dihedral", "y": "ester_dihedral"},
            "igor_pair": {"x": "igor1_dihedral", "y": "ester_dihedral"},
        },
    }
    names = [name for name, _ in list_coordinate_pairs(cfg)]
    assert set(names) == {"dihedral", "igor_pair"}


def test_dihedral_pair_is_periodic():
    """A pair of two dihedral DoF should have periodic=True."""
    cfg = _multi_dof_config()
    pairs = list_coordinate_pairs(cfg)
    _, pair = pairs[0]
    assert pair.periodic is True


def test_state_col_property():
    """state_col is derived from the pair name."""
    cfg = _multi_dof_config()
    pairs = list_coordinate_pairs(cfg)
    _, pair = pairs[0]
    assert pair.state_col == "state_dihedral"
