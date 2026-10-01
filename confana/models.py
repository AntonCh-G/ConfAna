"""Core data model for the ConfAna workflow.

All downstream analysis consumes FrameRecord (for single-frame access) and the
standard coordinate table (pandas DataFrame produced by coordinates.py).  This
module intentionally has no plotting or I/O dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Frame index (lightweight; no coordinates stored)
# ---------------------------------------------------------------------------


@dataclass
class FrameIndexEntry:
    """Metadata for one frame in an xyz file, without coordinate data."""

    frame_number: int
    """0-based position within the source file."""

    byte_offset: int
    """Byte position in the file where this frame begins (atom-count line)."""

    atom_count: int
    """Number of atoms in this frame."""

    comment_line: str
    """Raw comment / metadata line (line 2 of the xyz block)."""


@dataclass
class FrameIndex:
    """Per-file index of all frame positions, used for random-access retrieval.

    The index is built by a one-pass scan and can be persisted to disk so that
    repeated runs do not rescan the source file.
    """

    source_file: str
    """Absolute path of the xyz file at scan time."""

    file_size: int
    """os.stat().st_size at scan time — used for cache invalidation."""

    file_mtime: float
    """os.stat().st_mtime at scan time — used for cache invalidation."""

    entries: list[FrameIndexEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, frame_number: int) -> FrameIndexEntry:
        """Return the entry for the given 0-based frame number.

        Raises IndexError if frame_number is out of range.
        """
        if frame_number < 0 or frame_number >= len(self.entries):
            raise IndexError(
                f"Frame {frame_number} out of range "
                f"(file has {len(self.entries)} frames): {self.source_file}"
            )
        return self.entries[frame_number]


# ---------------------------------------------------------------------------
# Full frame record (includes coordinates)
# ---------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """One parsed frame (from an xyz or HDF5 trajectory), with coordinates and metadata.

    Attributes set by the I/O layer may be None until resolved by the calling
    code (e.g., trajectory_id and bead_id come from the coordinate-table row the
    frame is read for; see ``confana.frame_source``).
    """

    source_file: str
    frame_number: int
    byte_offset: int
    atom_count: int
    comment_line: str

    elements: list[str]
    """Atomic element symbols, length == atom_count."""

    coords: np.ndarray
    """Coordinate array, shape (atom_count, 3), units angstrom: float32 when
    streamed for the coordinate table, float64 when read one frame at a time
    (``confana.frame_source.read_frames``)."""

    energy: Optional[float] = None
    """Potential energy parsed from the comment line; None if not present."""

    step_number: Optional[int] = None
    """MD step number parsed from comment line (keyword 'Step:')."""

    bead_comment: Optional[int] = None
    """Bead index parsed from comment line (keyword 'Bead:'), for cross-validation."""

    trajectory_id: Optional[str] = None
    """Trajectory the frame belongs to; None until a caller sets it."""

    bead_id: Optional[str] = None
    """PIMD bead of the frame; None for non-PIMD frames or until a caller sets it."""

    local_frame_index: int = 0
    """0-based frame position within the source file."""

    global_frame_index: int = 0
    """0-based frame position across all files in the current run."""


# ---------------------------------------------------------------------------
# Trajectory sources (what the trajectory readers hand the coordinate-table build)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectorySource:
    """One trajectory file as a reader describes it, before any frame is read.

    A file holds one or more *streams*: ordered runs of frames read side by
    side. An xyz file is one stream; an HDF5 PIMD file holds one stream per
    bead, or one centroid stream.
    """

    path: Path
    """Absolute path of the file."""

    bead_ids: tuple[Optional[str], ...]
    """``bead_id`` of each stream, in stream order (None: no bead)."""

    n_frames: int
    """Frames per stream."""

    atom_count: int
    """Atoms per frame (the same for every frame)."""


@dataclass
class FrameBlock:
    """Consecutive frames ``frame_number[0] …`` of every stream of one source."""

    coords: np.ndarray
    """float32, shape ``(n_frames, n_streams, atom_count, 3)``, angstrom."""

    frame_number: np.ndarray
    """int64, shape ``(n_frames,)``: 0-based frame numbers in the file."""

    byte_offset: np.ndarray
    """int64, shape ``(n_frames,)``: xyz byte offsets, or ``io_hdf5.HDF5_BYTE_OFFSET``."""

    step_number: np.ndarray
    """int64, shape ``(n_frames,)``: MD step numbers (0 where missing)."""

    step_missing: np.ndarray
    """bool, shape ``(n_frames,)``: True where the step number is unknown."""

    energy: np.ndarray
    """float32, shape ``(n_frames,)``: potential energy, NaN where unknown."""

    elements: list[str]
    """Element symbols of the block's first frame."""


# ---------------------------------------------------------------------------
# Generalised degree-of-freedom model
# ---------------------------------------------------------------------------

_DOF_TYPES = frozenset({"dihedral", "distance", "angle", "collective", "external"})


@dataclass(frozen=True)
class DoFDefinition:
    """Validated config model for one named degree of freedom.

    Supported types
    ---------------
    dihedral  : signed 4-atom dihedral angle, range [-180, 180)
    distance  : Euclidean distance between two atoms (Å)
    angle     : 3-atom bond angle (°)
    collective: linear combination or PCA of other DoF columns (stub)
    external  : value loaded from an external column / file (stub)
    """

    name: str
    """Unique column name produced in the coordinate table."""

    type: Literal["dihedral", "distance", "angle", "collective", "external"]
    """Computation type — determines which geometry function is called."""

    label: str
    """User-facing axis label (include units, e.g. 'Carboxyl dihedral (°)')."""

    domain: tuple[float, float]
    """Expected value range (min, max) used for axis scaling and grid wrapping."""

    enabled: bool = True
    """Whether this DoF is computed in the current run."""

    # --- geometry types (dihedral / distance / angle) -----------------------
    atoms: Optional[tuple[int, ...]] = None
    """0-based atom indices.  Length: 4 for dihedral, 2 for distance, 3 for angle."""

    # --- dihedral-specific --------------------------------------------------
    convention: str = "signed"
    """Angle convention.  Currently only 'signed' (→ [-180, 180)) is implemented."""

    # --- collective type ----------------------------------------------------
    method: Optional[str] = None
    """Collective variable method: 'pca' or 'linear' (stub — not yet implemented)."""

    input_dof: Optional[tuple[str, ...]] = None
    """Names of DoF columns used as input to the collective variable."""

    component: Optional[int] = None
    """Which component to extract (e.g. 0 for PC1)."""

    # --- external type ------------------------------------------------------
    source_column: Optional[str] = None
    """Column name in the external data source to join onto the coordinate table."""

    def __post_init__(self) -> None:
        if self.type not in _DOF_TYPES:
            raise ValueError(f"Unknown DoF type {self.type!r}. Must be one of {sorted(_DOF_TYPES)}.")
        if self.type in {"dihedral", "distance", "angle"} and self.atoms is None:
            raise ValueError(f"DoF '{self.name}' of type '{self.type}' requires 'atoms'.")
        expected_len = {"dihedral": 4, "distance": 2, "angle": 3}
        if self.type in expected_len and self.atoms is not None:
            if len(self.atoms) != expected_len[self.type]:
                raise ValueError(
                    f"DoF '{self.name}' type '{self.type}' needs "
                    f"{expected_len[self.type]} atoms, got {len(self.atoms)}."
                )
        if self.type == "external" and self.source_column is None:
            raise ValueError(f"DoF '{self.name}' of type 'external' requires 'source_column'.")
        if len(self.domain) != 2 or self.domain[0] >= self.domain[1]:
            raise ValueError(
                f"DoF '{self.name}' domain must be (min, max) with min < max, "
                f"got {self.domain}."
            )


# ---------------------------------------------------------------------------
# Coordinate pair model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinatePair:
    """Axis specification for 2D analyses such as density / state / transition plots.

    Instances are built from config by coordinate_config.resolve_coordinate_pair().
    Domains and labels are inherited from the referenced DoFDefinition objects.
    """

    name: str
    """Unique identifier for this pair (used in output filenames and state column names)."""

    x_col: str
    """Base DataFrame column for the x axis (raw DoF column name)."""

    y_col: str
    """Base DataFrame column for the y axis (raw DoF column name)."""

    x_label: str
    """User-facing x-axis label."""

    y_label: str
    """User-facing y-axis label."""

    title: str
    """Default plot title."""

    x_domain: tuple[float, float]
    """Value range (min, max) for the x axis."""

    y_domain: tuple[float, float]
    """Value range (min, max) for the y axis."""

    # Optional overrides for when coordinate transforms produce shifted columns
    feature_x_col: Optional[str] = None
    """Effective x column for analysis (e.g. x_col + '_shifted').  Defaults to x_col."""

    feature_y_col: Optional[str] = None
    """Effective y column for analysis (e.g. y_col + '_shifted').  Defaults to y_col."""

    periodic: bool = False
    """True when both axes are periodic (e.g. dihedral–dihedral pairs).  Used to
    enable grid wrapping across the ±180° boundary in clustering."""

    # Optional per-pair plotting overrides (None = fall back to global density config)
    bins: Optional[int] = None
    x_range: Optional[tuple[float, float]] = None
    y_range: Optional[tuple[float, float]] = None
    colormap: Optional[str] = None
    log_scale: Optional[bool] = None

    # Atoms defining each axis, copied from the referenced DoFDefinitions so
    # plotting code can highlight them without reading config.  0-based file
    # indices; None when the DoF has no atoms (collective / external types).
    x_atoms: Optional[tuple[int, ...]] = None
    y_atoms: Optional[tuple[int, ...]] = None
    x_dof_type: Optional[str] = None
    """DoF type of the x axis (e.g. 'dihedral'); None when not built from a DoF."""
    y_dof_type: Optional[str] = None
    """DoF type of the y axis (e.g. 'dihedral'); None when not built from a DoF."""

    @property
    def feature_columns(self) -> list[str]:
        """Return the effective feature columns used for clustering / transition analysis."""
        return [
            self.feature_x_col if self.feature_x_col is not None else self.x_col,
            self.feature_y_col if self.feature_y_col is not None else self.y_col,
        ]

    @property
    def state_col(self) -> str:
        """Column name written by state assignment for this pair."""
        return f"state_{self.name}"
