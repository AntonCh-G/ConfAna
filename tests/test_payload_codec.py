"""Tests for confana/payload_codec.py (Slice 7 — compressed embedded data)."""

from __future__ import annotations

import base64
import gzip
import json

import numpy as np
import pytest

from confana.payload_codec import (
    COLUMNS_FORMAT,
    STRUCTURES_FORMAT,
    decode_columns,
    decode_structures,
    encode_columns,
    encode_structures,
)

_ELEMENTS = ["C", "O", "H"]


def _xyz(coords: np.ndarray, comment: str = "frame 0") -> str:
    lines = [str(len(coords)), comment]
    lines += [
        f"{el:<2s}  {x: .8f}  {y: .8f}  {z: .8f}"
        for el, (x, y, z) in zip(_ELEMENTS, coords)
    ]
    return "\n".join(lines) + "\n"


def _payloads(n: int = 4, seed: int = 0) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    return {
        f"{i}_{i + 1}": _xyz(rng.uniform(-5.0, 5.0, (len(_ELEMENTS), 3)), comment=f"frame {i}")
        for i in range(n)
    }


def _coords_of(text: str) -> np.ndarray:
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in text.splitlines()[2:]]
    )


def _symbols_of(text: str) -> list[str]:
    return [line.split()[0] for line in text.splitlines()[2:]]


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def test_structures_round_trip_keeps_atoms_elements_and_bin_order():
    payloads = _payloads()
    decoded = decode_structures(encode_structures(payloads))

    assert list(decoded) == list(payloads)
    for key, text in payloads.items():
        assert int(decoded[key].splitlines()[0]) == len(_ELEMENTS)
        assert _symbols_of(decoded[key]) == _ELEMENTS
        assert decoded[key].splitlines()[1] == text.splitlines()[1]


@pytest.mark.parametrize("step", [0.001, 0.01, 0.1])
def test_structures_round_trip_error_is_at_most_half_a_step(step):
    payloads = _payloads(n=6, seed=1)
    decoded = decode_structures(encode_structures(payloads, coordinate_step=step))

    worst = max(
        float(np.abs(_coords_of(decoded[key]) - _coords_of(text)).max())
        for key, text in payloads.items()
    )
    assert worst <= step / 2 + 1e-12


def test_structures_store_elements_once_and_compress():
    payloads = _payloads(n=50, seed=2)
    block = encode_structures(payloads)

    assert block["format"] == STRUCTURES_FORMAT
    assert block["elements"] == _ELEMENTS
    assert block["count"] == 50
    assert block["atom_count"] == len(_ELEMENTS)
    plain = len(json.dumps(payloads))
    assert len(json.dumps(block)) < plain / 2


def test_structures_encoding_is_deterministic():
    payloads = _payloads(n=5, seed=3)
    assert encode_structures(payloads) == encode_structures(payloads)


def test_empty_structures_round_trip():
    assert decode_structures(encode_structures({})) == {}


def test_mixed_element_sequences_raise():
    payloads = _payloads(n=2)
    other = list(payloads.values())[1].splitlines()
    other[2] = other[2].replace("C ", "N ", 1)
    payloads["1_2"] = "\n".join(other) + "\n"

    with pytest.raises(ValueError, match="different element sequence"):
        encode_structures(payloads)


def test_coordinate_outside_16_bit_range_raises():
    payloads = {"0_0": _xyz(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [40.0, 0.0, 0.0]]))}

    with pytest.raises(ValueError, match="16-bit"):
        encode_structures(payloads)

    # A larger step brings the same structure back inside the range.
    decoded = decode_structures(encode_structures(payloads, coordinate_step=0.01))
    assert _coords_of(decoded["0_0"])[2][0] == pytest.approx(40.0, abs=0.005)


@pytest.mark.parametrize("step", [0, -0.001, "0.001", None])
def test_invalid_coordinate_step_raises(step):
    with pytest.raises(ValueError, match="coordinate_step"):
        encode_structures(_payloads(n=1), coordinate_step=step)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("3\ncomment\nC 0 0 0\n", "3 atoms but has 1"),
        ("x\ncomment\n", "not an atom count"),
        ("1\ncomment\nC 0 0\n", "not 'symbol x y z'"),
        ("1\ncomment\nC a b c\n", "non-numeric"),
        ("1\n", "no atom-count and comment line"),
    ],
)
def test_malformed_structure_payload_raises(text, message):
    with pytest.raises(ValueError, match=message):
        encode_structures({"0_0": text})


def test_decode_structures_rejects_unknown_format():
    block = encode_structures(_payloads(n=1))
    block["format"] = "something-else"
    with pytest.raises(ValueError, match=STRUCTURES_FORMAT):
        decode_structures(block)


# ---------------------------------------------------------------------------
# Metadata columns
# ---------------------------------------------------------------------------


def _records(n: int = 6) -> list[dict]:
    return [
        {
            "source_file": "/data/traj_00.xyz" if i % 2 else "/data/traj_01.xyz",
            "trajectory_id": "my_run.pos",
            "bead_id": None if i % 3 else "00",
            "frame_number": i,
            "carboxyl_dihedral": -12.5 + i,
            "comment_line": f"frame {i}",
        }
        for i in range(n)
    ]


def test_columns_round_trip_is_exact_for_a_list():
    records = _records()
    block = encode_columns(records)

    assert block["format"] == COLUMNS_FORMAT
    assert block["count"] == len(records)
    assert decode_columns(block) == records


def test_columns_round_trip_is_exact_for_a_mapping():
    records = {f"{i}_{i}": row for i, row in enumerate(_records())}
    assert decode_columns(encode_columns(records)) == records


def test_columns_round_trip_keeps_missing_and_null_values():
    records = [{"a": 1, "b": None}, {"a": None}, {"b": "x"}]
    decoded = decode_columns(encode_columns(records))
    assert decoded == [
        {"a": 1, "b": None},
        {"a": None, "b": None},
        {"a": None, "b": "x"},
    ]


def test_repeated_strings_are_stored_once():
    records = [{"source_file": "/data/traj.xyz", "i": i} for i in range(200)]
    packed = encode_columns(records)["data"]
    payload = json.loads(gzip.decompress(base64.b64decode(packed)).decode("utf-8"))
    assert payload["columns"]["source_file"] == {
        "lookup": ["/data/traj.xyz"],
        "codes": [0] * 200,
    }
    assert payload["columns"]["i"]["values"] == list(range(200))


def test_columns_encoding_is_deterministic_and_compresses():
    records = _records(n=500)
    assert encode_columns(records) == encode_columns(records)
    assert len(json.dumps(encode_columns(records))) < len(json.dumps(records)) / 2


def test_columns_reject_non_records():
    with pytest.raises(ValueError, match="mapping or a sequence"):
        encode_columns(42)
    with pytest.raises(ValueError, match="every record to be a mapping"):
        encode_columns([{"a": 1}, 7])


def test_decode_columns_rejects_unknown_format():
    block = encode_columns(_records(n=1))
    block["format"] = "something-else"
    with pytest.raises(ValueError, match=COLUMNS_FORMAT):
        decode_columns(block)
