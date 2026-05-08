"""2D histogram computation for conformer coordinate data.

This module contains only data-computation logic; it has no dependency on
matplotlib and can be used independently for testing or downstream processing.

Public API
----------
- ``compute_2d_histogram``
- ``compute_2d_histogram_arrays``
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_2d_histogram(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 100,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D histogram from two coordinate columns of a DataFrame.

    Rows where either ``x_col`` or ``y_col`` is NA are silently dropped
    before binning.

    Parameters
    ----------
    df:
        DataFrame containing the coordinate columns.
    x_col:
        Name of the column to use for the x axis.
    y_col:
        Name of the column to use for the y axis.
    bins:
        Number of bins along each axis (square grid).
    x_range:
        (min, max) limits for the x axis.  If None, the data extent is used.
    y_range:
        (min, max) limits for the y axis.  If None, the data extent is used.

    Returns
    -------
    H : np.ndarray, shape (bins, bins)
        2D count array.  ``H[i, j]`` is the count in the bin at
        (x_edges[i], y_edges[j]).
    x_edges : np.ndarray, shape (bins + 1,)
        Bin edges along the x axis.
    y_edges : np.ndarray, shape (bins + 1,)
        Bin edges along the y axis.

    Raises
    ------
    ValueError
        If ``x_col`` or ``y_col`` is not present in ``df``.
    """
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")

    # Drop rows where either coordinate is NA
    mask = df[x_col].notna() & df[y_col].notna()
    x = df.loc[mask, x_col].astype(float).to_numpy()
    y = df.loc[mask, y_col].astype(float).to_numpy()

    return compute_2d_histogram_arrays(x, y, bins=bins, x_range=x_range, y_range=y_range)


def compute_2d_histogram_arrays(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 100,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2D histogram from pre-extracted coordinate arrays.

    Equivalent to :func:`compute_2d_histogram` but accepts raw numpy arrays
    instead of a DataFrame.  Non-finite values (NaN, Inf) are silently dropped.

    Parameters
    ----------
    x, y:
        1-D float arrays of equal length.
    bins:
        Number of bins along each axis.
    x_range, y_range:
        Axis limits; defaults to data extent when None.

    Returns
    -------
    H : np.ndarray, shape (bins, bins)
    x_edges : np.ndarray, shape (bins + 1,)
    y_edges : np.ndarray, shape (bins + 1,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Drop non-finite values (may be introduced by restrict_positive_y)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) == 0:
        x_lo, x_hi = x_range if x_range is not None else (0.0, 1.0)
        y_lo, y_hi = y_range if y_range is not None else (0.0, 1.0)
        x_edges = np.linspace(x_lo, x_hi, bins + 1)
        y_edges = np.linspace(y_lo, y_hi, bins + 1)
        H = np.zeros((bins, bins), dtype=np.int64)
        return H, x_edges, y_edges

    H, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[
            x_range if x_range is not None else (float(x.min()), float(x.max())),
            y_range if y_range is not None else (float(y.min()), float(y.max())),
        ],
    )

    return H.astype(np.int64), x_edges, y_edges
