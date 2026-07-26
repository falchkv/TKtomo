"""The interface the GUI drives, and the event plumbing underneath it.

Everything the window needs from the alignment engine is a verb here. Two things
implement it: one driving an engine in this process, one driving a server on a cluster
node. The window cannot tell them apart, which is the point -- where the compute runs
becomes a connection detail rather than an architecture.

**Cheap verbs return values; heavy verbs return a** :class:`JobHandle`. A verb is heavy
if it touches tomopy, walks a whole stack, or reads a file, and heavy work is serialised
onto a single thread -- ``tomopy.util.mproc`` keeps its shared arrays in module-level
globals, so a ``project()`` running while another thread is inside ``recon()`` corrupts
them and segfaults the process. That constraint is why the window has a ``_busy()``
guard today; here it becomes structural.

Heavy verbs come in two kinds. **Exclusive** ones mutate state and are *rejected* with
:class:`Busy` while a run is in flight rather than queued behind it: queuing a COM
pre-alignment behind a 60-minute run means that an hour later the session silently
clears the history the run just built. **Queued** ones only read, so they can wait their
turn safely.

Results and progress come back as :class:`Event` objects on an :class:`EventLog`, not as
callbacks or Qt signals -- this layer sits below the GUI and must import without it.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from tktomo.ptycho_align.session.types import (
    PlaneRef,
    ResourceSample,
    RunPreflight,
    SessionSummary,
)

__all__ = [
    "AlignmentSession",
    "Busy",
    "Event",
    "EventBatch",
    "EventLog",
    "JobFailed",
    "JobHandle",
    "JobState",
    "NoEngine",
    "SessionError",
    "EVENT_HEARTBEAT",
    "EVENT_ITERATION",
    "EVENT_ITERATION_STARTED",
    "EVENT_JOB_FAILED",
    "EVENT_JOB_FINISHED",
    "EVENT_JOB_STARTED",
    "EVENT_LOG",
    "EVENT_PROGRESS",
    "EVENT_RUN_FINISHED",
    "EVENT_RUN_STARTED",
    "EVENT_SUMMARY",
    "EVENT_TELEMETRY",
]

# Event kinds. `iteration` carries an IterationSummary *and* the summary delta, so the
# client never has to make a synchronous round trip on the GUI thread just to redraw.
EVENT_PROGRESS = "progress"
EVENT_ITERATION_STARTED = "iteration_started"
EVENT_ITERATION = "iteration"
EVENT_RUN_STARTED = "run_started"
EVENT_RUN_FINISHED = "run_finished"
EVENT_JOB_STARTED = "job_started"
EVENT_JOB_FINISHED = "job_finished"
EVENT_JOB_FAILED = "job_failed"
EVENT_LOG = "log"
EVENT_SUMMARY = "summary"
EVENT_TELEMETRY = "telemetry"
EVENT_HEARTBEAT = "heartbeat"


class SessionError(Exception):
    """Base for failures the session reports rather than raising out of a thread."""


class Busy(SessionError):
    """An exclusive verb was asked for while a run was in flight.

    Deliberately not a queue: see the module docstring for why waiting an hour and then
    wiping the user's history is the worse option.
    """


class NoEngine(SessionError):
    """A verb needing a loaded dataset was called before one was opened."""


class JobFailed(SessionError):
    """A job raised. Carries the original exception type name and traceback."""

    def __init__(self, message: str, *, exc_type: str = "Exception", traceback: str = "") -> None:
        super().__init__(message)
        self.exc_type = exc_type
        self.traceback = traceback


@dataclass(frozen=True)
class Event:
    """One thing that happened, with a sequence number a client can resume from."""

    seq: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None


@dataclass(frozen=True)
class EventBatch:
    """A replay window, and whether it is contiguous with what the client already has.

    ``gap`` is the honest answer to a client that was away too long: the ring dropped
    events it never saw, so it must resync from a fresh summary rather than pretend its
    accumulated history is still a prefix of the server's.
    """

    events: tuple[Event, ...]
    last_seq: int
    oldest_seq: int
    gap: bool = False


@dataclass(frozen=True)
class JobHandle:
    """Identifies one unit of heavy work. Terminal outcome arrives as an event."""

    job_id: str
    kind: str

    @classmethod
    def new(cls, kind: str) -> "JobHandle":
        return cls(job_id=uuid.uuid4().hex[:12], kind=kind)


@dataclass
class JobState:
    """Where a job got to, for a client that reconnected after it finished."""

    handle: JobHandle
    done: bool = False
    result: Any = None
    error: JobFailed | None = None
    cancelled: bool = False


class EventLog:
    """A bounded, monotonically numbered event history with live fan-out.

    Both halves matter. Live subscribers get pushed to, so a GUI updates without
    polling; the ring lets a client that dropped its connection ask for what it missed.
    The ring is bounded because a ten-hour run would otherwise accumulate events without
    limit -- and a client further behind than the ring is told so rather than handed a
    silently truncated history.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self._events: deque[Event] = deque(maxlen=capacity)
        self._subscribers: list[Callable[[Event], None]] = []
        self._lock = threading.RLock()
        self._seq = 0

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def emit(self, kind: str, payload: dict[str, Any] | None = None, job_id: str | None = None):
        """Record an event and hand it to every live subscriber."""
        with self._lock:
            self._seq += 1
            event = Event(seq=self._seq, kind=kind, payload=payload or {}, job_id=job_id)
            self._events.append(event)
            subscribers = list(self._subscribers)

        # Outside the lock: a subscriber that emits, or blocks, must not deadlock the
        # compute thread that is reporting progress.
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # a broken listener must not take the run down
                pass
        return event

    def since(self, seq: int, max_n: int = 256) -> EventBatch:
        """Events after ``seq``, oldest first."""
        with self._lock:
            if not self._events:
                return EventBatch((), self._seq, self._seq, gap=False)
            oldest = self._events[0].seq
            # A gap means the ring rolled past what the client last saw. seq == oldest-1
            # is still contiguous: the client has everything up to the first event held.
            gap = seq < oldest - 1
            selected = [e for e in self._events if e.seq > seq][:max_n]
            return EventBatch(tuple(selected), self._seq, oldest, gap=gap)

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Register a live listener. Returns the function that unregisters it."""
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe


@runtime_checkable
class AlignmentSession(Protocol):
    """What the window is allowed to ask for.

    Implementations must agree on more than these signatures: the same call sequence has
    to produce the same shifts, the same errors and the same refusals whether the engine
    is in this process or on a cluster. That is what the conformance suite pins down.
    """

    # -- where the compute is ------------------------------------------------------
    #
    # The window is not supposed to branch on these -- if it needs to know where the
    # engine is in order to work, the abstraction has already failed. They are for
    # telling the *user*, who very much does need to know which machine is about to
    # spend an hour, and whose filesystem a path refers to.

    @property
    def is_remote(self) -> bool: ...

    def describe(self) -> str:
        """Somewhere to show in the title bar: "this machine", or an address."""
        ...

    # -- events and state ----------------------------------------------------------

    def summary(self, since_iteration: int = 0) -> SessionSummary:
        """Everything needed to draw the window except pixels.

        ``history`` is a delta from ``since_iteration``; a full history is megabytes and
        has no business being resent on every refresh.
        """
        ...

    def poll_events(self, since_seq: int, max_n: int = 256) -> EventBatch: ...

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]: ...

    def telemetry(self) -> ResourceSample | None:
        """Load on the machine actually doing the work."""
        ...

    # -- cheap mutations -----------------------------------------------------------

    def set_config(self, config: dict) -> None:
        """Stage a config change; it takes effect between iterations, never inside one."""
        ...

    def set_center(self, value: float) -> None: ...

    def cancel_run(self) -> None:
        """Ask the run to stop at the next row chunk. Must never block."""
        ...

    def cancel_job(self, job_id: str) -> None: ...

    # -- queries -------------------------------------------------------------------

    def list_hdf5(self, path: str) -> list:
        """Datasets in an HDF5 file, resolved where the file lives."""
        ...

    def run_preflight(self, n: int) -> RunPreflight:
        """Price a run of ``n`` iterations: negatives, memory, wallclock. One trip."""
        ...

    def fetch_table(self, kind: str) -> bytes:
        """``"shifts"`` or ``"convergence"`` as CSV bytes. Kilobytes, so no job needed."""
        ...

    # -- pixels --------------------------------------------------------------------
    #
    # One displayed 2-D plane at a time, never a whole stack. A stack is 104 MiB and a
    # volume 511 MiB, and no viewer ever shows more than one plane of either; a session
    # that will hand over the whole array is one a remote client will ask to, and then
    # the slider is unusable. Whole arrays stay on EngineHost, where they are local.
    #
    # ``None`` means "no plane there" -- the stack has not been computed yet, or the
    # index is past the end. Callers already have ``StackSpec.available`` to distinguish
    # the two before asking.

    def read_planes(self, refs: Sequence[PlaneRef]) -> tuple[np.ndarray | None, ...]:
        """The named planes, in order. Batched for the side-by-side modes."""
        ...

    def read_plane(self, key: str, axis: int, index: int) -> np.ndarray | None: ...

    def read_volume_plane(
        self,
        axis: int,
        index: int,
        *,
        iteration: int | None = None,
        against: int | None = None,
    ) -> np.ndarray | None:
        """One plane of the volume, optionally minus that plane of an earlier one."""
        ...

    # -- heavy: exclusive ----------------------------------------------------------

    def open_dataset(self, path: str, load_kwargs: dict | None = None) -> JobHandle: ...

    def apply_preprocessing(self, options, roi: tuple | None = None) -> JobHandle: ...

    def reset_preprocessing(self) -> JobHandle: ...

    def set_bin_factor(self, factor: int) -> JobHandle: ...

    def run_com(self, vertical_reference: str = "mean") -> JobHandle: ...

    def estimate_center(self, method: str) -> JobHandle: ...

    def start_run(self, n: int) -> JobHandle: ...

    def revert(self, iteration: int) -> JobHandle: ...

    def open_session(self, path: str) -> JobHandle: ...

    # -- heavy: queued (read-only, safe during a run) -------------------------------

    def materialize(self, keys: Sequence[str]) -> JobHandle:
        """Compute the aligned stack and/or reprojection for display."""
        ...

    def save_session(self, path: str, *, include_arrays: bool = True) -> JobHandle: ...

    def export(self, kind: str, path: str) -> JobHandle: ...

    # -- lifecycle -----------------------------------------------------------------

    def wait(self, handle: JobHandle, timeout: float | None = None) -> Any:
        """Block until a job finishes and return its result, or raise :class:`JobFailed`.

        For tests and for the places the GUI legitimately blocks behind a modal progress
        dialog. Never call it from a thread that also has to service the job.
        """
        ...

    def job_state(self, job_id: str) -> JobState | None: ...

    def close(self) -> None: ...
