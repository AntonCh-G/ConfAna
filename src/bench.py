"""Lightweight wall-time accumulator for benchmarking pipeline stages.

Usage
-----
::

    from src.bench import StageTimer

    timer = StageTimer()
    timer.start("my_stage")
    do_work()
    timer.stop("my_stage")

    print(timer.report(total_frames=n))

Design notes
------------
* Based solely on ``time.perf_counter()`` — no external dependencies.
* Thread-unsafe; intended for single-threaded benchmarking only.
* The ``_timer`` parameter in production functions defaults to ``None``;
  all guards are ``if _timer is not None:`` so overhead is zero in normal runs.
* This module has no scientific logic and can be deleted without affecting results.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class StageTimer:
    """Accumulate named wall-time intervals across multiple calls.

    Parameters
    ----------
    None

    Examples
    --------
    >>> t = StageTimer()
    >>> t.start("io")
    >>> time.sleep(0.01)
    >>> t.stop("io")
    >>> print(t.report())
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._starts: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def start(self, label: str) -> None:
        """Record the start time for *label*."""
        self._starts[label] = time.perf_counter()

    def stop(self, label: str) -> float:
        """Record the stop time for *label* and return elapsed seconds."""
        now = time.perf_counter()
        start = self._starts.pop(label)
        elapsed = now - start
        self._totals[label] = self._totals.get(label, 0.0) + elapsed
        self._counts[label] = self._counts.get(label, 0) + 1
        return elapsed

    @contextmanager
    def section(self, label: str) -> Generator[None, None, None]:
        """Context manager that calls ``start`` / ``stop`` automatically."""
        self.start(label)
        try:
            yield
        finally:
            self.stop(label)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def total(self, label: str) -> float:
        """Return accumulated seconds for *label* (0.0 if never recorded)."""
        return self._totals.get(label, 0.0)

    def count(self, label: str) -> int:
        """Return call count for *label*."""
        return self._counts.get(label, 0)

    def grand_total(self) -> float:
        """Return the sum of all accumulated times."""
        return sum(self._totals.values())

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, total_frames: int = 0) -> str:
        """Return a formatted timing table sorted by total time descending.

        Parameters
        ----------
        total_frames:
            If > 0, an extra column shows µs/frame for each stage.
        """
        if not self._totals:
            return "(no timings recorded)"

        grand = self.grand_total() or 1.0  # avoid division by zero

        # Build rows
        rows: list[tuple[str, int, float, float, float]] = []
        for label, total in sorted(self._totals.items(), key=lambda kv: -kv[1]):
            cnt = self._counts.get(label, 0)
            mean_ms = (total / cnt * 1000.0) if cnt else 0.0
            pct = total / grand * 100.0
            rows.append((label, cnt, total, mean_ms, pct))

        # Column widths
        lw = max(len(r[0]) for r in rows)
        lw = max(lw, 5)

        if total_frames > 0:
            header = (
                f"  {'label':<{lw}}  {'calls':>8}  {'total_s':>8}  "
                f"{'mean_ms':>9}  {'us/frame':>9}  {'%total':>7}"
            )
            sep = "  " + "-" * (lw + 8 + 9 + 10 + 10 + 8 + 10)
            lines = [header, sep]
            for label, cnt, total, mean_ms, pct in rows:
                us_per_frame = total / total_frames * 1e6
                lines.append(
                    f"  {label:<{lw}}  {cnt:>8,}  {total:>8.3f}  "
                    f"{mean_ms:>9.3f}  {us_per_frame:>9.1f}  {pct:>6.1f}%"
                )
        else:
            header = (
                f"  {'label':<{lw}}  {'calls':>8}  {'total_s':>8}  "
                f"{'mean_ms':>9}  {'%total':>7}"
            )
            sep = "  " + "-" * (lw + 8 + 9 + 10 + 8 + 5)
            lines = [header, sep]
            for label, cnt, total, mean_ms, pct in rows:
                lines.append(
                    f"  {label:<{lw}}  {cnt:>8,}  {total:>8.3f}  "
                    f"{mean_ms:>9.3f}  {pct:>6.1f}%"
                )

        lines.append(sep)
        lines.append(f"  {'TOTAL':<{lw}}  {'':>8}  {grand:>8.3f}")
        return "\n".join(lines)
