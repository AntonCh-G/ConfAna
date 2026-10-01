"""Build the coordinate table from trajectories: one pipeline for every input format.

Whatever ``data.format`` says, the table is built by the same steps in the
same order:

1. discover the trajectory files matched by ``data.path_pattern``;
2. give each file its ``trajectory_id`` and ``bead_id`` by the one id rule
   (:func:`_trajectory_id`, :func:`_bead_id`) and group the files into
   trajectories;
3. per trajectory, fingerprint every input setting and reuse its cached table
   when the fingerprint matches;
4. otherwise cut each file to ``frame_range`` and the ``bond_break`` limit,
5. compute the DoF values, and cache the trajectory's table;
6. apply the coordinate shifts (``coordinate_transforms``);
7. reset the state columns;
8. validate the table schema.

The cache is per trajectory, so the ids (step 2) come before the cache check
(step 3): changing one trajectory's files rebuilds only that trajectory. Files
are opened (frame index, atom count) and the DoF atom indices checked against
them only for a trajectory that is rebuilt: a cache hit needs neither, since its
key covers the files and the DoF definitions.

Only reading differs between formats, behind two readers: xyz
(:func:`confana.io_xyz.describe_xyz_source`, :func:`confana.io_xyz.read_xyz_blocks`)
and HDF5 (:func:`confana.io_hdf5.describe_hdf5_source`,
:func:`confana.io_hdf5.read_hdf5_blocks`).

Public API
----------
- ``InputSettings``, ``FrameRange``
- ``load_or_build_coordinate_table``
- ``load_or_build_coordinate_table_from_config``
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Literal

import numpy as np
import pandas as pd

from confana.bond_check import bonded_pairs, first_broken_frame
from confana.cache import fingerprint_files, matches
from confana.coordinate_config import (
    list_coordinate_pairs,
    resolve_coordinate_transforms,
    resolve_dof_definitions,
)
from confana.coordinates import apply_coordinate_shifts, batch_extract_geometry_dof
from confana.io_coordinates import (
    load_coordinate_table,
    save_coordinate_table,
    validate_coordinate_table,
)
from confana.io_hdf5 import HDF5_SUFFIXES, describe_hdf5_source, read_hdf5_blocks, run_trajectory_id
from confana.io_xyz import XYZ_SUFFIXES, describe_xyz_source, read_xyz_blocks
from confana.models import DoFDefinition, FrameBlock, TrajectorySource

if TYPE_CHECKING:
    from confana.bench import StageTimer

_CACHE_VERSION = 2
_GEOMETRY_DOF_TYPES = ("dihedral", "distance", "angle")
_FORMATS = ("xyz", "hdf5")
_POSITIONS_SOURCES = ("bead", "centroid")


# ---------------------------------------------------------------------------
# Input settings, parsed once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameRange:
    """Frames ``start … end`` (inclusive) of every file; ``end=None`` runs to the last frame."""

    start: int = 0
    end: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"frame_range.start_frame must be >= 0; got {self.start}")
        if self.end is not None and self.end < self.start:
            raise ValueError(
                f"frame_range.end_frame ({self.end}) must be >= "
                f"frame_range.start_frame ({self.start})"
            )

    def stop(self, n_frames: int) -> int:
        """First frame number past the range, in a file of *n_frames* frames."""
        return n_frames if self.end is None else min(n_frames, self.end + 1)


@dataclass(frozen=True)
class InputSettings:
    """Every setting that decides the coordinate table's rows and values, and its cache.

    Built once from config by :meth:`from_config`. The settings are checked on
    construction, so a copy made by :meth:`for_overlay` or
    :func:`dataclasses.replace` is checked again.

    Raises
    ------
    ValueError
        On any setting that is invalid, or that does nothing for the format.
    """

    path_pattern: str
    dofs: tuple[DoFDefinition, ...]
    cache_dir: Path
    format: Literal["xyz", "hdf5"] = "xyz"
    trajectory_id_pattern: re.Pattern[str] | None = None
    bead_id_pattern: re.Pattern[str] | None = None
    positions_source: Literal["bead", "centroid"] | None = None
    """HDF5 only: one row per bead and frame, or one centroid row per frame."""
    shifts: Mapping[str, float] = field(default_factory=dict)
    """``coordinate_transforms``: degrees added to a DoF in its ``*_shifted`` column."""
    frame_range: FrameRange = FrameRange()
    bond_cutoff: float | None = None
    """``bond_break.cutoff`` in Å when ``bond_break.enabled``, else None."""
    index_cache_dir: Path | None = None
    n_jobs: int = 1
    state_columns: tuple[str, ...] = ()
    """State columns of the configured coordinate pairs, reset on every load."""

    def __post_init__(self) -> None:
        if self.format not in _FORMATS:
            raise ValueError(f"data.format must be one of {_FORMATS}; got {self.format!r}")
        for key in ("trajectory_id_pattern", "bead_id_pattern"):
            pattern = getattr(self, key)
            if pattern is None:
                continue
            if self.format == "hdf5":
                raise ValueError(
                    f"data.{key} cannot be used with data.format: hdf5. An HDF5 run's "
                    "trajectory_id is its simulation folder and its beads are bead_00, "
                    "bead_01, …; remove the setting."
                )
            if pattern.groups < 1:
                raise ValueError(
                    f"data.{key} {pattern.pattern!r} has no capture group; "
                    "group 1 of its match is the id."
                )
        if self.format == "hdf5":
            if self.positions_source not in _POSITIONS_SOURCES:
                raise ValueError(
                    f"data.positions_source must be one of {_POSITIONS_SOURCES}; "
                    f"got {self.positions_source!r}"
                )
            if self.bond_cutoff is not None:
                raise ValueError(
                    "bond_break cannot be used with data.format: hdf5 yet: whether a bond "
                    "break in one bead should cut every bead of the run, as it cuts every "
                    "bead file of an xyz trajectory, is not decided. Disable it."
                )
        elif self.positions_source is not None:
            raise ValueError("data.positions_source only applies to data.format: hdf5.")
        for dof in self.dofs:
            if dof.type not in _GEOMETRY_DOF_TYPES:
                raise ValueError(
                    f"DoF {dof.name!r} has type {dof.type!r}, which is not implemented "
                    f"yet; the coordinate table computes {', '.join(_GEOMETRY_DOF_TYPES)} DoFs."
                )
        names = [dof.name for dof in self.dofs]
        unknown = [col for col in self.shifts if col not in names]
        if unknown:
            raise ValueError(
                f"coordinate_transforms names {unknown}, which are not enabled DoFs {names}."
            )
        if self.bond_cutoff is not None and self.bond_cutoff <= 0:
            raise ValueError(f"bond_break.cutoff must be > 0 Å; got {self.bond_cutoff}")
        if self.n_jobs == 0:
            raise ValueError("cache.n_jobs must not be 0 (1 = serial, -1 = all cores).")

    def for_overlay(self, path_pattern: str, cache_dir: Path) -> InputSettings:
        """Settings for a scatter overlay: other xyz files, read like this run's.

        The DoF, coordinate shifts and ``bond_break`` stay; the overlay keeps every
        frame (no ``frame_range``) and has no id patterns, so its files are grouped
        by folder. Its tables are cached in *cache_dir*.
        """
        return replace(
            self,
            format="xyz",
            path_pattern=path_pattern,
            trajectory_id_pattern=None,
            bead_id_pattern=None,
            positions_source=None,
            frame_range=FrameRange(),
            cache_dir=cache_dir,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> InputSettings:
        """Parse the input settings of a run config.

        Reads ``data``, ``dof``, ``coordinate_transforms``, ``frame_range``,
        ``bond_break``, ``cache``, ``run_dir`` and the ``coordinate_pairs``'
        state columns.
        """
        data = config.get("data") or {}
        cache = config.get("cache") or {}
        if not data.get("path_pattern"):
            raise ValueError("data.path_pattern is not set in config.")
        if cache.get("coordinate_table_path"):
            raise ValueError(
                "cache.coordinate_table_path is no longer used: each trajectory's table "
                "is cached under cache.trajectory_cache_dir (default {run_dir}/.cache). "
                "Remove the setting."
            )
        run_dir = config.get("run_dir")
        cache_dir = cache.get("trajectory_cache_dir") or (
            Path(run_dir) / ".cache" if run_dir else None
        )
        if not cache_dir:
            raise ValueError(
                "Set run_dir (or cache.trajectory_cache_dir) in config: the coordinate "
                "table is cached there."
            )
        format_ = str(data.get("format") or "xyz").lower()
        frame_range = config.get("frame_range") or {}
        bond_break = config.get("bond_break") or {}
        index_cache_dir = cache.get("index_cache_dir")
        return cls(
            path_pattern=str(data["path_pattern"]),
            dofs=tuple(resolve_dof_definitions(config)),
            cache_dir=Path(cache_dir),
            format=format_,  # type: ignore[arg-type]  # checked in __post_init__
            trajectory_id_pattern=_compile(data, "trajectory_id_pattern"),
            bead_id_pattern=_compile(data, "bead_id_pattern"),
            positions_source=(
                data.get("positions_source") or ("bead" if format_ == "hdf5" else None)
            ),
            shifts=resolve_coordinate_transforms(config),
            frame_range=FrameRange(
                start=int(frame_range.get("start_frame") or 0),
                end=None if frame_range.get("end_frame") is None else int(frame_range["end_frame"]),
            ),
            bond_cutoff=float(bond_break.get("cutoff", 2.0)) if bond_break.get("enabled") else None,
            index_cache_dir=Path(index_cache_dir) if index_cache_dir else None,
            n_jobs=int(cache.get("n_jobs", 1)),
            state_columns=tuple(pair.state_col for _, pair in list_coordinate_pairs(config)),
        )


def _compile(data: dict[str, Any], key: str) -> re.Pattern[str] | None:
    """Compile the ``data.<key>`` regex, or return None when it is unset."""
    raw = data.get(key)
    if not raw:
        return None
    try:
        return re.compile(str(raw))
    except re.error as exc:
        raise ValueError(f"data.{key} {raw!r} is not a valid regular expression: {exc}") from exc


# ---------------------------------------------------------------------------
# Readers: the only part that differs between formats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reader:
    """What one input format supplies: its file suffixes, its default id, and reading."""

    suffixes: tuple[str, ...]
    default_trajectory_id: Callable[[Path], str]
    """``trajectory_id`` of a file when ``data.trajectory_id_pattern`` is unset."""
    describe: Callable[[Path, str | None, InputSettings], TrajectorySource]
    read_blocks: Callable[[TrajectorySource, int, int], Iterator[FrameBlock]]


def _folder_name(path: Path) -> str:
    """The name of the folder holding *path* (its stem at a file-system root)."""
    return path.parent.name or path.stem


def _describe_xyz(path: Path, bead_id: str | None, settings: InputSettings) -> TrajectorySource:
    """Describe an xyz file through its frame index (``cache.index_cache_dir``)."""
    return describe_xyz_source(path, bead_id=bead_id, index_cache_dir=settings.index_cache_dir)


def _describe_hdf5(path: Path, bead_id: str | None, settings: InputSettings) -> TrajectorySource:
    """Describe an HDF5 run; its beads come from the file, not from *bead_id*."""
    assert settings.positions_source is not None  # checked by InputSettings
    return describe_hdf5_source(path, positions_source=settings.positions_source)


_READERS: dict[str, _Reader] = {
    "xyz": _Reader(XYZ_SUFFIXES, _folder_name, _describe_xyz, read_xyz_blocks),
    "hdf5": _Reader(HDF5_SUFFIXES, run_trajectory_id, _describe_hdf5, read_hdf5_blocks),
}


# ---------------------------------------------------------------------------
# Steps 1–2: discovery and the one id rule
# ---------------------------------------------------------------------------


def _discover(settings: InputSettings) -> list[Path]:
    """Return the sorted, absolute trajectory files matched by ``data.path_pattern``.

    Files with the format's suffixes are kept; when none has one, every
    matched file is (so a trajectory with an unusual suffix still loads).
    Frame-index sidecars are never trajectories.
    """
    matched = sorted({Path(p).resolve() for p in glob.glob(settings.path_pattern, recursive=True)})
    matched = [p for p in matched if p.is_file() and not p.name.endswith(".frameindex.npz")]
    suffixes = _READERS[settings.format].suffixes
    files = [p for p in matched if p.suffix.lower() in suffixes] or matched
    if not files:
        raise FileNotFoundError(f"No files matched data.path_pattern={settings.path_pattern!r}")
    return files


def _id_from_name(pattern: re.Pattern[str], path: Path, key: str) -> str:
    """Group 1 of *pattern* searched in the file name without its extension."""
    match = pattern.search(path.stem)
    value = match.group(1) if match else None
    if not value:
        raise ValueError(
            f"data.{key} {pattern.pattern!r} gives no id for the file {path.name!r} "
            f"({path}); every matched file needs one."
        )
    return value


def _trajectory_id(path: Path, settings: InputSettings) -> str:
    """The one ``trajectory_id`` rule.

    Group 1 of ``data.trajectory_id_pattern`` searched in the file name (no
    extension); without a pattern, the folder holding the file, or for HDF5
    the run's simulation folder (the parent of ``hdf5/``).
    """
    if settings.trajectory_id_pattern is not None:
        return _id_from_name(settings.trajectory_id_pattern, path, "trajectory_id_pattern")
    return _READERS[settings.format].default_trajectory_id(path)


def _bead_id(path: Path, settings: InputSettings) -> str | None:
    """The one ``bead_id`` rule for a file: group 1 of ``data.bead_id_pattern``, else no bead.

    HDF5 beads are not named by file: the reader labels them ``bead_00``, ….
    """
    if settings.bead_id_pattern is None:
        return None
    return _id_from_name(settings.bead_id_pattern, path, "bead_id_pattern")


def _group_by_trajectory(
    files: list[Path], settings: InputSettings
) -> dict[str, list[tuple[Path, str | None]]]:
    """Map each ``trajectory_id`` to its ``(file, bead_id)`` list, in file order."""
    trajectories: dict[str, list[tuple[Path, str | None]]] = {}
    for path in files:
        trajectories.setdefault(_trajectory_id(path, settings), []).append(
            (path, _bead_id(path, settings))
        )
    return trajectories


# ---------------------------------------------------------------------------
# Step 3: per-trajectory cache
# ---------------------------------------------------------------------------


def _fingerprint(
    settings: InputSettings, trajectory_id: str, files: list[tuple[Path, str | None]]
) -> dict[str, Any]:
    """Return the cache key of one trajectory, as JSON-plain values.

    It covers every setting that can change the trajectory's rows or values.
    The id patterns and ``data.path_pattern`` enter through what they decide:
    the trajectory's id, its files (path, size, mtime) and each file's bead id.
    So a widened glob does not rebuild the trajectories it already matched.
    """
    stats = fingerprint_files([path for path, _ in files])
    meta = {
        "version": _CACHE_VERSION,
        "format": settings.format,
        "positions_source": settings.positions_source,
        "dofs": [
            {"name": d.name, "type": d.type, "atoms": d.atoms, "domain": d.domain}
            for d in settings.dofs
        ],
        "frame_range": {"start": settings.frame_range.start, "end": settings.frame_range.end},
        "bond_cutoff": settings.bond_cutoff,
        "trajectory_id": trajectory_id,
        "files": [{**stat, "bead_id": bead_id} for stat, (_, bead_id) in zip(stats, files)],
    }
    # Round-trip through JSON so tuples compare equal to the lists read back.
    return json.loads(json.dumps(meta))


def _cache_path(settings: InputSettings, trajectory_id: str) -> Path:
    """``{cache_dir}/{readable id}__{hash of id}__coordinates.npz``: unique per id."""
    readable = re.sub(r"[^A-Za-z0-9._-]", "_", trajectory_id)
    digest = hashlib.sha1(trajectory_id.encode("utf-8")).hexdigest()[:8]
    return settings.cache_dir / f"{readable}__{digest}__coordinates.npz"


# ---------------------------------------------------------------------------
# Steps 4–5: read, cut and compute one trajectory
# ---------------------------------------------------------------------------


_FRAME_METADATA: dict[str, type] = {
    "frame_number": np.int64,
    "byte_offset": np.int64,
    "step_number": np.int64,
    "step_missing": np.bool_,
    "energy": np.float32,
}
"""Per-frame arrays a :class:`~confana.models.FrameBlock` carries besides coordinates."""


@dataclass
class _SourceRows:
    """The frames read from one source, ``frame_range.start`` on, with their DoF values."""

    frames: dict[str, np.ndarray]
    """The per-frame arrays named in :data:`_FRAME_METADATA`, one value per frame."""
    dof_values: dict[str, np.ndarray]
    """float32 values of each DoF, shape ``(n_streams, n_frames)``."""
    first_break: int | None
    """Frame number of the earliest bond break in any stream; None without one."""


def _section(timer: StageTimer | None, label: str) -> ContextManager[None]:
    """Time the enclosed stage as *label* when a timer is given."""
    return timer.section(label) if timer is not None else nullcontext()


def _check_atoms(
    trajectory_id: str, sources: list[TrajectorySource], dofs: tuple[DoFDefinition, ...]
) -> None:
    """Raise if a trajectory's files differ in atom count, or a DoF names a missing atom."""
    sources = [source for source in sources if source.n_frames]
    if not sources:
        return
    if len({source.atom_count for source in sources}) > 1:
        counts = ", ".join(f"{source.path} has {source.atom_count}" for source in sources)
        raise ValueError(
            f"The files of trajectory {trajectory_id!r} have different atom counts "
            f"({counts}); every frame of a trajectory must have the same atoms."
        )
    first = sources[0]
    for dof in dofs:
        bad = [a for a in dof.atoms or () if not 0 <= a < first.atom_count]
        if bad:
            raise ValueError(
                f"DoF {dof.name!r} uses atom index {bad[0]}, but {first.path} has "
                f"{first.atom_count} atoms per frame "
                f"(0-based indices 0–{first.atom_count - 1})."
            )


def _read_source(
    format_: str,
    source: TrajectorySource,
    dofs: tuple[DoFDefinition, ...],
    frame_range: FrameRange,
    bond_cutoff: float | None,
    timer: StageTimer | None = None,
) -> _SourceRows:
    """Read one source within *frame_range* and compute its DoF values, block by block.

    With *bond_cutoff*, each stream's bonds are taken from its first frame in
    range, and reading stops after the block holding the first break: the
    trajectory is cut at or before it. Runs in a joblib worker for
    ``cache.n_jobs != 1``.
    """
    names = [dof.name for dof in dofs]
    n_streams = len(source.bead_ids)
    blocks = _READERS[format_].read_blocks(
        source, frame_range.start, frame_range.stop(source.n_frames)
    )
    metadata: dict[str, list[np.ndarray]] = {key: [] for key in _FRAME_METADATA}
    values: list[dict[str, np.ndarray]] = []
    bonds: list[tuple[np.ndarray, np.ndarray]] | None = None
    first_break: int | None = None
    while first_break is None:
        with _section(timer, "frame_iter"):
            block = next(blocks, None)
        if block is None:
            break
        n = len(block.frame_number)
        with _section(timer, "batch_geometry"):
            block_values = {name: np.empty((n_streams, n), dtype=np.float32) for name in names}
            for stream in range(n_streams):
                try:
                    computed = batch_extract_geometry_dof(block.coords[:, stream], list(dofs))
                except ValueError as exc:
                    raise ValueError(
                        f"{source.path}, frames {block.frame_number[0]}–"
                        f"{block.frame_number[-1]}: {exc}"
                    ) from exc
                for name in names:
                    block_values[name][stream] = computed[name]
        if bond_cutoff is not None:
            with _section(timer, "bond_scan"):
                skip = 0
                if bonds is None:  # the reference frame is not checked against itself
                    bonds = [
                        bonded_pairs(block.coords[0, s], block.elements) for s in range(n_streams)
                    ]
                    skip = 1
                for stream in range(n_streams):
                    position = first_broken_frame(
                        block.coords[skip:, stream], bonds[stream], bond_cutoff
                    )
                    if position is not None:
                        frame = int(block.frame_number[skip + position])
                        first_break = frame if first_break is None else min(first_break, frame)
        for key in _FRAME_METADATA:  # the block's coordinates are dropped here
            metadata[key].append(getattr(block, key))
        values.append(block_values)

    return _SourceRows(
        frames={
            key: np.concatenate(parts) if parts else np.empty(0, dtype=_FRAME_METADATA[key])
            for key, parts in metadata.items()
        },
        dof_values={
            name: np.concatenate([v[name] for v in values], axis=1)
            if values else np.empty((n_streams, 0), dtype=np.float32)
            for name in names
        },
        first_break=first_break,
    )


def _stop_frames(
    settings: InputSettings, sources: list[TrajectorySource], rows: list[_SourceRows]
) -> list[int]:
    """Return, per source, the first frame number past what the table keeps.

    Without ``bond_break`` each file keeps ``frame_range``. With it, every file
    of the trajectory stops at the earliest break among them (a file without a
    break counts as breaking at its last frame + 1), and at ``frame_range``.
    """
    if settings.bond_cutoff is None:
        return [settings.frame_range.stop(source.n_frames) for source in sources]
    raw = [
        source_rows.first_break if source_rows.first_break is not None else source.n_frames
        for source, source_rows in zip(sources, rows)
    ]
    stop = settings.frame_range.stop(min(raw))
    for source in sources:
        if stop < source.n_frames:
            print(
                f"    bond-break: {source.path.name} truncated at frame {stop} "
                f"(of {source.n_frames})"
            )
    return [stop] * len(sources)


def _assemble(
    trajectory_id: str,
    sources: list[TrajectorySource],
    rows: list[_SourceRows],
    stops: list[int],
    dofs: tuple[DoFDefinition, ...],
) -> pd.DataFrame:
    """Lay out one trajectory's rows: files in order, each file stream by stream (bead-major)."""
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in _FRAME_METADATA}
    dof_pieces: dict[str, list[np.ndarray]] = {dof.name: [] for dof in dofs}
    source_codes: list[np.ndarray] = []
    bead_codes: list[np.ndarray] = []
    atom_counts: list[np.ndarray] = []
    bead_labels: list[str] = []
    for index, (source, source_rows, stop) in enumerate(zip(sources, rows, stops)):
        keep = int(np.count_nonzero(source_rows.frames["frame_number"] < stop))
        for stream, bead_id in enumerate(source.bead_ids):
            for key in pieces:
                pieces[key].append(source_rows.frames[key][:keep])
            for name in dof_pieces:
                dof_pieces[name].append(source_rows.dof_values[name][stream, :keep])
            source_codes.append(np.full(keep, index, dtype=np.int32))
            atom_counts.append(np.full(keep, source.atom_count, dtype=np.int64))
            if bead_id is None:
                code = -1
            else:
                if bead_id not in bead_labels:
                    bead_labels.append(bead_id)
                code = bead_labels.index(bead_id)
            bead_codes.append(np.full(keep, code, dtype=np.int32))

    def joined(parts: list[np.ndarray], dtype: type) -> np.ndarray:
        """Concatenate *parts*, or an empty *dtype* array when there are none."""
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    n = sum(len(p) for p in source_codes)
    step_number = pd.array(joined(pieces["step_number"], np.int64), dtype="Int64")
    step_number[joined(pieces["step_missing"], bool)] = pd.NA
    data: dict[str, Any] = {
        "frame_id": np.arange(n, dtype=np.int64),
        "source_file": pd.Categorical.from_codes(
            joined(source_codes, np.int32), categories=pd.Index([str(s.path) for s in sources])
        ),
        "trajectory_id": pd.Categorical.from_codes(
            np.zeros(n, dtype=np.int8), categories=pd.Index([trajectory_id])
        ),
        "bead_id": pd.Categorical.from_codes(
            joined(bead_codes, np.int32), categories=pd.Index(bead_labels, dtype=object)
        ),
        "frame_number": joined(pieces["frame_number"], np.int64),
        "byte_offset": joined(pieces["byte_offset"], np.int64),
        "atom_count": joined(atom_counts, np.int64),
        "comment_line": pd.Categorical.from_codes(
            np.zeros(n, dtype=np.int8), categories=pd.Index([""])
        ),
        "local_frame_index": joined(pieces["frame_number"], np.int64),
        "global_frame_index": np.arange(n, dtype=np.int64),
    }
    for name, parts in dof_pieces.items():
        data[name] = joined(parts, np.float32)
    data["energy"] = joined(pieces["energy"], np.float32)
    data["step_number"] = step_number
    return pd.DataFrame(data)


def _build_trajectories(
    stale: list[tuple[str, list[tuple[Path, str | None]], dict[str, Any], Path]],
    settings: InputSettings,
    timer: StageTimer | None,
) -> dict[str, pd.DataFrame]:
    """Build, cache and read back the table of every stale trajectory.

    Every stale source of every trajectory is read in one pass, in parallel
    when ``cache.n_jobs != 1`` and there is more than one.
    """
    reader = _READERS[settings.format]
    with _section(timer, "index_loading"):
        sources = [
            [reader.describe(path, bead_id, settings) for path, bead_id in files]
            for _, files, _, _ in stale
        ]
    for (trajectory_id, _, _, _), group in zip(stale, sources):
        _check_atoms(trajectory_id, group, settings.dofs)

    work = [source for group in sources for source in group]
    args = (settings.dofs, settings.frame_range, settings.bond_cutoff)
    if settings.n_jobs != 1 and len(work) > 1:
        from joblib import Parallel, delayed  # noqa: PLC0415

        flat_rows = Parallel(n_jobs=settings.n_jobs, prefer="processes")(
            delayed(_read_source)(settings.format, source, *args) for source in work
        )
    else:
        flat_rows = [_read_source(settings.format, source, *args, timer=timer) for source in work]

    tables: dict[str, pd.DataFrame] = {}
    position = 0
    for (trajectory_id, _, meta, cache_path), group in zip(stale, sources):
        rows = flat_rows[position:position + len(group)]
        position += len(group)
        with _section(timer, "df_assembly"):
            stops = _stop_frames(settings, group, rows)
            table = _assemble(trajectory_id, group, rows, stops, settings.dofs)
        with _section(timer, "cache_write"):
            save_coordinate_table(table, cache_path, cache_metadata=meta)
            del table
            # Read back, so a fresh build and a cache hit return the same table.
            tables[trajectory_id] = load_coordinate_table(cache_path)
    return tables


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_or_build_coordinate_table(
    settings: InputSettings,
    *,
    force_rebuild: bool = False,
    timer: StageTimer | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load or build the coordinate table of *settings*, one cached table per trajectory.

    Parameters
    ----------
    settings:
        The run's input settings (see :meth:`InputSettings.from_config`).
    force_rebuild:
        Rebuild every trajectory, ignoring its cache.
    timer:
        Optional :class:`confana.bench.StageTimer`; stages are timed in serial
        builds only.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(table, cache_hit)``; ``cache_hit`` is True when no trajectory was
        rebuilt. ``frame_id`` and ``global_frame_index`` number the rows 0…N-1
        in trajectory order.

    Raises
    ------
    FileNotFoundError
        If ``data.path_pattern`` matches no file, or an HDF5 run has no
        ``input.xyz``.
    ValueError
        On a file that gives no id, malformed input (frames, datasets, atom
        counts) or a DoF atom index past a file's atom count; the message names
        the file.
    """
    trajectories = _group_by_trajectory(_discover(settings), settings)

    tables: dict[str, pd.DataFrame] = {}
    stale = []
    for trajectory_id, files in trajectories.items():
        meta = _fingerprint(settings, trajectory_id, files)
        cache_path = _cache_path(settings, trajectory_id)
        if not force_rebuild and matches(cache_path, meta):
            tables[trajectory_id] = load_coordinate_table(cache_path)
        else:
            stale.append((trajectory_id, files, meta, cache_path))
    if stale:
        tables.update(_build_trajectories(stale, settings, timer))

    ordered = [tables[trajectory_id] for trajectory_id in trajectories]
    # An empty table has untyped columns; leaving it out keeps the dtypes.
    non_empty = [t for t in ordered if len(t)] or ordered[:1]
    df = non_empty[0] if len(non_empty) == 1 else pd.concat(non_empty, ignore_index=True)
    df["frame_id"] = pd.array(np.arange(len(df)), dtype="Int64")
    df["global_frame_index"] = pd.array(np.arange(len(df)), dtype="Int64")

    df = apply_coordinate_shifts(df, dict(settings.shifts))
    # No cached table carries state labels; this keeps it so for any table.
    for col in settings.state_columns:
        if col in df.columns:
            df[col] = pd.NA
    validate_coordinate_table(
        df,
        value_columns=[d.name for d in settings.dofs] + [f"{c}_shifted" for c in settings.shifts],
    )
    return df, not stale


def load_or_build_coordinate_table_from_config(
    config: dict[str, Any],
    *,
    force_rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Load or build the coordinate table of a run config.

    The CLI's entry point: parses :class:`InputSettings` from *config* and
    runs :func:`load_or_build_coordinate_table`. Returns ``(table, cache_hit)``.
    """
    return load_or_build_coordinate_table(
        InputSettings.from_config(config), force_rebuild=force_rebuild_cache
    )
