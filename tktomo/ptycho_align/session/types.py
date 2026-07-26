"""Payload types that cross the boundary between the GUI and whatever runs the engine.

The rule for everything in this module: it must survive a round trip through msgpack,
and it must not carry a projection stack or a volume. A stack is 104 MiB and a volume
511 MiB for a realistic dataset, so anything that ships one per refresh has already
failed -- pixels move separately, one displayed 2-D slice at a time.

The two shapes that recur are worth naming up front:

``epoch``
    Bumped whenever the engine is rebuilt (a new dataset, new preprocessing, a new bin
    factor). Anything the client cached about the old engine -- timing calibration,
    slice caches, slider positions -- is stale when this changes. Object identity used
    to serve this purpose, which is why ``id(engine)`` appears in the code being
    replaced; identity does not survive a wire, so it becomes an explicit counter.

``pixel_epoch``
    Bumped whenever the *pixels* change without the engine being rebuilt: an iteration
    completing, a centre change, a cache invalidation. Slice caches key on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tktomo.ptycho_align.core.telemetry import ResourceSample

__all__ = [
    "DatasetSummary",
    "IterationSummary",
    "PlaneRef",
    "PreprocessReport",
    "ResourceSample",
    "RunPreflight",
    "SessionSummary",
    "StackSpec",
]


@dataclass(frozen=True)
class PlaneRef:
    """Which 2-D plane of which stack a viewer wants.

    ``axis`` is the axis to index *into*, in the stack's own ``(angle, v, u)`` frame:
    axis 0 with index ``i`` is projection ``i``, axis 1 with index ``r`` is row ``r``'s
    sinogram. The viewers already thought in exactly those two slices; naming them here
    is what lets 200 KiB cross the boundary instead of the 104 MiB stack around them.

    Frozen and hashable because the client caches on it -- see the ``pixel_epoch`` note
    above for what makes a cached plane stale.
    """

    key: str  # one of the STACK_* constants
    axis: int
    index: int

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, not {self.axis!r}")


@dataclass(frozen=True)
class DatasetSummary:
    """What the data panel shows, without the data.

    The panel used to take a whole ``ProjectionData`` and call ``min``/``max``/``mean``
    on it, which is three full passes over 104 MiB -- affordable in-process, absurd over
    a network. The reductions are computed once, where the array lives.
    """

    name: str
    shape: tuple[int, int, int]
    angle_min_deg: float
    angle_max_deg: float
    angle_step_deg: float
    vmin: float
    vmax: float
    vmean: float
    data_path: str | None = None
    crop: tuple[int, int, int, int] | None = None
    full_shape: tuple[int, int, int] | None = None
    component: str | None = None
    pixel_size_nm: float | None = None
    source_path: str | None = None

    @property
    def is_full_frame(self) -> bool:
        if self.crop is None or self.full_shape is None:
            return True
        v0, v1, u0, u1 = self.crop
        return (v0, v1, u0, u1) == (0, self.full_shape[1], 0, self.full_shape[2])

    @classmethod
    def from_projection_data(cls, data, name: str = "dataset") -> "DatasetSummary":
        """Reduce a :class:`~tktomo.io.ProjectionData` where it lives."""
        array = np.asarray(data.data)
        angles = np.rad2deg(np.asarray(data.angles))
        step = float(np.mean(np.diff(angles))) if len(angles) > 1 else 0.0
        meta = dict(data.metadata)

        crop = meta.get("crop")
        full_shape = meta.get("full_shape")
        return cls(
            name=str(meta.get("name", name)),
            shape=tuple(int(n) for n in array.shape),  # type: ignore[arg-type]
            angle_min_deg=float(angles.min()) if len(angles) else 0.0,
            angle_max_deg=float(angles.max()) if len(angles) else 0.0,
            angle_step_deg=step,
            vmin=float(array.min()),
            vmax=float(array.max()),
            vmean=float(array.mean()),
            data_path=meta.get("data_path"),
            crop=None if crop is None else tuple(int(v) for v in crop),  # type: ignore[arg-type]
            full_shape=(
                None if full_shape is None else tuple(int(v) for v in full_shape)  # type: ignore[arg-type]
            ),
            component=meta.get("component"),
            pixel_size_nm=meta.get("pixel_size_nm"),
            source_path=meta.get("source_path"),
        )


@dataclass(frozen=True)
class StackSpec:
    """Shape and freshness of one displayable stack, with no pixels attached.

    ``shape`` is present even when the stack has not been computed yet -- every stack
    has the shape of ``state.original`` -- because the viewers size their sliders from
    it before any frame arrives. ``available`` is what says whether a frame can be had.
    """

    key: str  # one of the MODE_* constants
    shape: tuple[int, int, int]
    dtype: str = "float32"
    available: bool = True
    pixel_epoch: int = 0
    stale: bool = False  # pixels predate the current shifts; see the revert-mid-run case


@dataclass(frozen=True)
class IterationSummary:
    """An :class:`~tktomo.ptycho_align.core.state.IterationResult` minus its volume.

    ~13 KiB against the volume's 511 MiB, which is what makes streaming every iteration
    to a remote GUI free. ``has_volume`` cannot be derived from a null volume here the
    way it is on the core type -- the volume is always absent -- so the server states it.
    """

    iteration: int
    sx: np.ndarray
    sy: np.ndarray
    dsx: np.ndarray
    dsy: np.ndarray
    error: float
    residual: float
    center: float
    wallclock_s: float
    config_changed: bool = False
    diverging: bool = False
    runaway: str | None = None
    has_volume: bool = False

    @classmethod
    def from_result(cls, result) -> "IterationSummary":
        return cls(
            iteration=result.iteration,
            sx=np.asarray(result.sx, dtype=np.float64),
            sy=np.asarray(result.sy, dtype=np.float64),
            dsx=np.asarray(result.dsx, dtype=np.float64),
            dsy=np.asarray(result.dsy, dtype=np.float64),
            error=float(result.error),
            residual=float(result.residual),
            center=float(result.center),
            wallclock_s=float(result.wallclock_s),
            config_changed=bool(result.config_changed),
            diverging=bool(result.diverging),
            runaway=result.runaway,
            has_volume=result.has_volume,
        )


@dataclass(frozen=True)
class PreprocessReport:
    """Outcome of a preprocessing pass, enough to raise the dialogs it can trigger."""

    dataset: DatasetSummary
    mass_is_positive: bool
    mass_total: float
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunPreflight:
    """Everything the three pre-run dialogs need, in one round trip.

    Priced together on purpose: the negative-data check is a full pass over the stack,
    and the memory and wallclock estimates depend on numbers only the engine's machine
    knows. Asking for them one at a time would be three round trips before a run starts.
    """

    footprint_bytes: int
    footprint_text: str
    ram_available: int | None
    negative_reason: str | None = None
    predicted_seconds: float | None = None


@dataclass(frozen=True)
class SessionSummary:
    """Everything needed to draw the window except pixels.

    ``history`` is a **delta**: a full history at 500 iterations is four float64 arrays
    per entry, ~6.5 MB, far too much to resend on every refresh. Ask with
    ``since_iteration`` and accumulate; ``history_from`` tells the client whether its
    accumulated copy is still a prefix of the server's or whether a revert truncated it.
    """

    epoch: int
    pixel_epoch: int
    seq: int  # the event sequence number this summary is consistent with
    has_engine: bool
    running: bool
    iteration: int
    config: dict
    bin_factor: int
    center: float
    angles: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    original_shape: tuple[int, int, int]
    # Shape as loaded, before preprocessing padded it and before binning shrank it. The
    # crop dialog needs it to tell whether an ROI drawn on the displayed stack still
    # indexes the file.
    raw_shape: tuple[int, int, int] | None = None
    stacks: tuple[StackSpec, ...] = ()
    history: tuple[IterationSummary, ...] = ()
    history_from: int = 0
    volume_shape: tuple[int, int, int] | None = None
    volume_iterations: tuple[int, ...] = ()
    volume_policy: tuple[int, int] = (3, 10)
    current_job: str | None = None
    dataset: DatasetSummary | None = None
    com: object | None = None  # ComResult; small (a handful of length-n_angles arrays)
    com_amplitude: float | None = None
    metadata: dict = field(default_factory=dict)
    resources: ResourceSample | None = None
