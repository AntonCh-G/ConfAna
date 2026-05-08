"""Shared caching utilities for per-trajectory NPZ caches.

Provides helpers for:
- File fingerprinting (path, size, mtime)
- Embedding / reading cache metadata JSON in NPZ archives
- Serialising / deserialising dicts of square DataFrames (transition matrices,
  including optional rate-derived barrier matrices)

Public API
----------
- fingerprint_files
- read_meta
- matches
- embed_meta
- write_matrix_section
- read_matrix_section
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CACHE_META_KEY = "cache_manifest_json"
_CACHE_META_VERSION = 1

# Transition result dict sections and their short NPZ-key prefixes.
# Keys match the output of ``analyze_grouped_transitions``.
_SECTION_ABBR: dict[str, str] = {
    "per_group_counts":        "pcc",
    "per_group_probabilities": "pcp",
    "per_group_rates":         "pcr",
    "per_group_barriers":      "pgb",
    "averaged_counts":         "ac",
    "averaged_probabilities":  "ap",
    "averaged_rates":          "ar",
    "averaged_barriers":       "ab",
}


# ---------------------------------------------------------------------------
# File fingerprinting
# ---------------------------------------------------------------------------


def fingerprint_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Return a per-file fingerprint list for cache invalidation.

    Parameters
    ----------
    paths:
        File paths to fingerprint.

    Returns
    -------
    list[dict]
        Each entry: ``{"path": str, "size": int, "mtime": float}``.
    """
    result = []
    for path in paths:
        st = os.stat(path)
        result.append({
            "path": str(path),
            "size": st.st_size,
            "mtime": st.st_mtime,
        })
    return result


# ---------------------------------------------------------------------------
# NPZ metadata embedding / reading
# ---------------------------------------------------------------------------


def read_meta(npz_path: Path) -> dict[str, Any] | None:
    """Read the embedded cache metadata JSON from an NPZ file.

    Returns ``None`` if the key is absent, the file cannot be opened, or the
    JSON is malformed.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if _CACHE_META_KEY not in data:
                return None
            return json.loads(str(data[_CACHE_META_KEY].item()))
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None


def matches(npz_path: Path, expected_meta: dict[str, Any]) -> bool:
    """Return True if the NPZ's embedded metadata matches *expected_meta* exactly."""
    return read_meta(npz_path) == expected_meta


def embed_meta(payload: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """Write *meta* as a JSON string into *payload* under ``_CACHE_META_KEY``.

    Mutates *payload* in-place before the caller passes it to ``np.savez``.
    """
    payload[_CACHE_META_KEY] = np.asarray(json.dumps(meta), dtype=str)


# ---------------------------------------------------------------------------
# Transition matrix serialization
# ---------------------------------------------------------------------------


def write_matrix_section(
    payload: dict[str, np.ndarray],
    section_name: str,
    matrices: dict[tuple, pd.DataFrame],
    *,
    key_prefix: str = "",
) -> None:
    """Serialize one dict of transition matrices into an NPZ payload dict.

    The section abbreviation (from ``_SECTION_ABBR``) is used as a key
    prefix, optionally scoped by *key_prefix*.  A JSON manifest records the
    original tuple keys so they can be recovered on load.

    Parameters
    ----------
    payload:
        Mutable dict that will be passed to ``np.savez``.
    section_name:
        One of the standard transition result dict keys
        (e.g. ``"per_group_counts"``).
    matrices:
        Mapping from group-key tuple to count / probability / rate DataFrame.
    key_prefix:
        Optional string prepended to every NPZ key, e.g. ``"plane_"`` to
        scope plane and dihedral results in the same archive.
    """
    abbr = key_prefix + _SECTION_ABBR[section_name]
    manifest: list[dict] = []
    for idx, (key, df) in enumerate(matrices.items()):
        manifest.append({"index": idx, "key": list(key)})
        labels = np.asarray([str(lbl) for lbl in df.index.tolist()], dtype=str)
        values = df.to_numpy(dtype=np.float32, na_value=np.nan)
        payload[f"{abbr}__{idx}__labels"] = labels
        payload[f"{abbr}__{idx}__values"] = values
    payload[f"{abbr}__manifest"] = np.asarray(json.dumps(manifest), dtype=str)


def read_matrix_section(
    data: Any,
    section_name: str,
    *,
    key_prefix: str = "",
) -> dict[tuple, pd.DataFrame]:
    """Deserialize one dict of transition matrices from an open NPZ archive.

    Parameters
    ----------
    data:
        Open ``np.load(...)`` context supporting key access.
    section_name:
        Same section name used when writing.
    key_prefix:
        Must match the value used when writing.

    Returns
    -------
    dict[tuple, pd.DataFrame]
        Reconstructed mapping from group-key tuples to DataFrames.
        Returns an empty dict if the section is not present in the archive.
    """
    abbr = key_prefix + _SECTION_ABBR[section_name]
    manifest_key = f"{abbr}__manifest"
    if manifest_key not in data:
        return {}
    manifest = json.loads(str(data[manifest_key].item()))
    result: dict[tuple, pd.DataFrame] = {}
    for entry in manifest:
        idx = entry["index"]
        key = tuple(entry["key"])
        labels = data[f"{abbr}__{idx}__labels"].astype(str, copy=False).tolist()
        values = data[f"{abbr}__{idx}__values"].astype(np.float32, copy=False)
        result[key] = pd.DataFrame(values, index=labels, columns=labels)
    return result
