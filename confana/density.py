"""The conformational map of a coordinate pair, and its free-energy-like surface.

A :class:`ConformationalMap` is built once per pair and read by every view of
it: the static PNG, the interactive page and the state overlay. It holds the
bin edges, the counts, the one rule that places a frame in a bin, and the
representative frame of each bin. This module has no plotting dependency.

Public API
----------
- ``DensitySettings``
- ``ConformationalMap``
- ``build_conformational_map``
- ``population_free_energy``
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from confana.models import CoordinatePair


@dataclass(frozen=True)
class DensitySettings:
    """How one coordinate pair's map is binned and drawn."""

    bins: int
    """Number of bins along each axis."""

    x_range: tuple[float, float]
    """(min, max) of the x axis; frames outside it are not on the map."""

    y_range: tuple[float, float]
    """(min, max) of the y axis; frames outside it are not on the map."""

    colormap: str = "viridis"
    log_scale: bool = True

    def __post_init__(self) -> None:
        bins = self.bins
        if isinstance(bins, bool) or not isinstance(bins, numbers.Integral) or bins < 1:
            raise ValueError(f"density bins must be a positive integer, got {bins!r}.")
        object.__setattr__(self, "bins", int(bins))
        for axis, (lo, hi) in (("x", self.x_range), ("y", self.y_range)):
            if not lo < hi:
                raise ValueError(f"density {axis}_range must have min < max, got {(lo, hi)!r}.")

    @classmethod
    def for_pair(cls, pair: CoordinatePair, config: dict | None = None) -> DensitySettings:
        """Resolve the settings of *pair*'s map.

        Each value comes from the pair's own override if set, else from
        ``plots.density`` in *config* (a full config, or the density section
        alone), else from the pair's domain and the defaults (180 bins,
        ``viridis``, log scale).

        Raises
        ------
        ValueError
            If the resolved bins are not a positive integer (they are never
            rounded) or a range is not two numbers with min < max.
        """
        cfg = config or {}
        plots_cfg = cfg.get("plots", {}) or {}
        density_cfg = (plots_cfg.get("density", {}) or {}) if "density" in plots_cfg else cfg

        def axis_range(override, key, domain) -> tuple[float, float]:
            raw = override if override is not None else density_cfg.get(key)
            lo, hi = raw if raw is not None else domain
            if lo is None or hi is None:
                raise ValueError(f"density {key} must be two numbers, got {raw!r}.")
            return float(lo), float(hi)

        try:
            return cls(
                bins=pair.bins if pair.bins is not None else density_cfg.get("bins", 180),
                x_range=axis_range(pair.x_range, "x_range", pair.x_domain),
                y_range=axis_range(pair.y_range, "y_range", pair.y_domain),
                colormap=str(
                    pair.colormap if pair.colormap is not None
                    else density_cfg.get("colormap", "viridis")
                ),
                log_scale=bool(
                    pair.log_scale if pair.log_scale is not None
                    else density_cfg.get("log_scale", True)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Coordinate pair '{pair.name}': {exc}") from exc


@dataclass(frozen=True, eq=False)
class ConformationalMap:
    """The binned population of one coordinate pair, shared by every view of it."""

    table: pd.DataFrame
    """Coordinate table the map was built from."""

    pair: CoordinatePair
    settings: DensitySettings
    x_edges: np.ndarray
    y_edges: np.ndarray

    counts: np.ndarray
    """Frames per bin, ``counts[xi, yi]`` (the ``np.histogram2d`` layout)."""

    def bin_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the bin ``(xi, yi)`` of each point, exactly as :attr:`counts` counts it.

        Bins are right-open except the last, which includes the right edge.
        Points off the map or with a NaN value get ``-1``.
        """
        return _bin_indices(x, self.x_edges), _bin_indices(y, self.y_edges)

    def frame_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return each table row's ``(x, y)`` values as the map reads them.

        Floats in table order; a missing or non-numeric value is NaN, so that
        row is on no bin.
        """
        return _pair_values(self.table, self.pair)

    @cached_property
    def representatives(self) -> dict[tuple[int, int], int]:
        """Representative frame of each occupied bin: ``{(xi, yi): row position}``.

        The representative is the counted frame nearest the bin centre, so the
        keys are exactly the bins with a non-zero count; ties go to the earlier
        row. Positions index :attr:`table` with ``iloc``. Bins are in ``(xi, yi)``
        order. Computed on first use, then kept.
        """
        x, y = self.frame_values()
        xi, yi = self.bin_index(x, y)
        rows = np.flatnonzero((xi >= 0) & (yi >= 0))
        if len(rows) == 0:
            return {}
        xi, yi = xi[rows], yi[rows]
        x_centres = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_centres = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        dist = (x[rows] - x_centres[xi]) ** 2 + (y[rows] - y_centres[yi]) ** 2
        flat = xi * (len(self.y_edges) - 1) + yi
        # Sort by bin, then distance, then row; the first row of each bin wins.
        order = np.lexsort((rows, dist, flat))
        first = np.ones(len(order), dtype=bool)
        first[1:] = flat[order][1:] != flat[order][:-1]
        chosen = order[first]
        return {
            (int(i), int(j)): int(r)
            for i, j, r in zip(xi[chosen], yi[chosen], rows[chosen])
        }


def _pair_values(table: pd.DataFrame, pair: CoordinatePair) -> tuple[np.ndarray, np.ndarray]:
    """The pair's two feature columns as float arrays; missing values become NaN."""
    x_col, y_col = pair.feature_columns
    return (
        pd.to_numeric(table[x_col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(table[y_col], errors="coerce").to_numpy(dtype=float),
    )


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per value as ``np.histogram2d`` assigns it; -1 outside the edges."""
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(edges, v, side="right") - 1
    idx[v == edges[-1]] = len(edges) - 2
    idx[~np.isfinite(v) | (v < edges[0]) | (v > edges[-1])] = -1
    return idx


def build_conformational_map(
    table: pd.DataFrame,
    pair: CoordinatePair,
    settings: DensitySettings,
) -> ConformationalMap:
    """Bin *table*'s values of *pair* into the map described by *settings*.

    Raises
    ------
    ValueError
        If a feature column of *pair* is not in *table*.
    """
    for col in pair.feature_columns:
        if col not in table.columns:
            raise ValueError(
                f"Coordinate pair '{pair.name}': column '{col}' not found in the "
                f"coordinate table. Available columns: {list(table.columns)}"
            )
    x, y = _pair_values(table, pair)
    x_edges = np.linspace(*settings.x_range, settings.bins + 1)
    y_edges = np.linspace(*settings.y_range, settings.bins + 1)
    finite = np.isfinite(x) & np.isfinite(y)
    counts, _, _ = np.histogram2d(x[finite], y[finite], bins=[x_edges, y_edges])
    return ConformationalMap(
        table=table,
        pair=pair,
        settings=settings,
        x_edges=x_edges,
        y_edges=y_edges,
        counts=counts.astype(np.int64),
    )


def population_free_energy(counts: np.ndarray) -> np.ndarray:
    """Return the dimensionless free-energy-like surface ``−ln(P / P_max)``.

    ``P`` is the population of each bin, so ``P / P_max = counts / counts.max()``.
    The most-populated bin is 0, every other sampled bin is positive, and
    unsampled bins (count 0) are NaN. Multiply by ``k_B · T`` in some unit
    (see :mod:`confana.units`) to get an energy. This is derived from frame
    counts, not from energies: it is not a potential energy surface.

    Parameters
    ----------
    counts:
        Histogram counts of any shape (non-negative).

    Returns
    -------
    np.ndarray
        Float array of the same shape; all NaN when no bin is sampled.

    Raises
    ------
    ValueError
        If any count is negative or not finite.
    """
    c = np.asarray(counts, dtype=float)
    if not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("population_free_energy: counts must be finite and non-negative.")
    out = np.full(c.shape, np.nan)
    sampled = c > 0
    if np.any(sampled):
        # log(max / c) rather than -log(c / max): the top bin is +0.0, not -0.0.
        out[sampled] = np.log(c.max() / c[sampled])
    return out
