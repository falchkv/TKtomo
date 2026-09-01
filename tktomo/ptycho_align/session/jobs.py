"""The single compute thread and the job queue, independent of what the jobs do.

:class:`EngineHost` (ptycho-align) and :class:`~tktomo.tracking.remote.host.TrackingHost`
both need exactly this: one worker thread, a priority queue of jobs, per-job state and
cancellation, an event log narrating starts/finishes/failures, and a blocking ``wait``.
None of it knows about engines or stacks, so it lives here once. The sharing is what
makes the local/remote conformance suites meaningful: local and remote differ only by
transport, and the two hosts differ only by their verbs.

Subclasses hook in through three methods: :meth:`_refuse` (return a message to refuse
a submission outright instead of queuing it), :meth:`_on_submitted` and
:meth:`_on_finished` (bookkeeping around a job's life, called under the state lock).
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from tktomo.ptycho_align.core.engine import Cancelled
from tktomo.ptycho_align.session.protocol import (
    EVENT_JOB_FAILED,
    EVENT_JOB_FINISHED,
    EVENT_JOB_STARTED,
    EVENT_PROGRESS,
    Busy,
    EventLog,
    JobFailed,
    JobHandle,
    JobState,
)

logger = logging.getLogger("tktomo.ptycho_align")

__all__ = ["JobContext", "JobHost", "job_verbs"]


@dataclass(order=True)
class _Job:
    """One unit of heavy work, ordered so cheap mutations overtake bulk reads."""

    priority: int
    sequence: int
    handle: JobHandle = field(compare=False)
    call: Callable[["JobContext"], Any] = field(compare=False)
    cancel: threading.Event = field(compare=False, default_factory=threading.Event)


class JobContext:
    """What a running job may report through. Also its cancellation flag.

    ``is_set`` is here because that is the entire interface ``engine.step`` requires of
    a cancel object -- it never asks for a ``threading.Event``, only something that can
    say whether it has been asked to stop.
    """

    def __init__(self, host: "JobHost", job: _Job) -> None:
        self._host = host
        self._job = job

    @property
    def job_id(self) -> str:
        return self._job.handle.job_id

    def is_set(self) -> bool:
        return self._job.cancel.is_set()

    def report(self, fraction: float, message: str) -> None:
        self._host._events.emit(
            EVENT_PROGRESS,
            {"fraction": float(fraction), "message": str(message)},
            job_id=self.job_id,
        )

    def emit(self, payload: dict) -> None:
        """A progress event with a caller-defined payload (auto-track counts etc.)."""
        self._host._events.emit(EVENT_PROGRESS, dict(payload), job_id=self.job_id)


class JobHost:
    """Serialises every heavy operation onto one thread and narrates what it did."""

    def __init__(self, *, event_capacity: int = 5000, thread_name: str = "compute") -> None:
        self._events = EventLog(capacity=event_capacity)
        self._state_lock = threading.RLock()
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._jobs: dict[str, JobState] = {}
        self._job_done: dict[str, threading.Event] = {}
        self._sequence = 0
        self._closing = threading.Event()
        self._current: _Job | None = None
        self._thread = threading.Thread(target=self._loop, name=thread_name, daemon=True)
        self._thread.start()

    # -- hooks -------------------------------------------------------------------------

    def _refuse(self, kind: str) -> str | None:
        """A reason to refuse queuing ``kind`` right now, or None to accept it."""
        return None

    def _on_submitted(self, job: _Job) -> None:
        """Called under the state lock once the job exists, before it is queued."""

    def _on_finished(self, job: _Job) -> None:
        """Called under the state lock after the job's terminal event was emitted."""

    # -- event access ------------------------------------------------------------------

    @property
    def events(self) -> EventLog:
        return self._events

    # -- job plumbing ------------------------------------------------------------------

    def _submit(self, kind: str, call, *, priority: int = 1) -> JobHandle:
        """Queue heavy work, or raise ``Busy`` when :meth:`_refuse` says so."""
        reason = self._refuse(kind)
        if reason is not None:
            raise Busy(reason)
        handle = JobHandle.new(kind)
        with self._state_lock:
            self._sequence += 1
            job = _Job(priority=priority, sequence=self._sequence, handle=handle, call=call)
            self._jobs[handle.job_id] = JobState(handle=handle)
            self._job_done[handle.job_id] = threading.Event()
            self._on_submitted(job)
        self._queue.put(job)
        return handle

    def _loop(self) -> None:
        while not self._closing.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            self._execute(job)

    def _execute(self, job: _Job) -> None:
        job_id = job.handle.job_id
        with self._state_lock:
            self._current = job

        self._events.emit(EVENT_JOB_STARTED, {"kind": job.handle.kind}, job_id=job_id)
        state = self._jobs[job_id]
        try:
            if job.cancel.is_set():
                raise Cancelled("cancelled before it started")
            state.result = job.call(JobContext(self, job))
            state.done = True
            self._events.emit(
                EVENT_JOB_FINISHED,
                {"kind": job.handle.kind, "result": state.result},
                job_id=job_id,
            )
        except Cancelled:
            state.done = True
            state.cancelled = True
            self._events.emit(EVENT_JOB_FINISHED, {"kind": job.handle.kind, "cancelled": True}, job_id=job_id)
        except Exception as exc:
            state.done = True
            state.error = JobFailed(
                str(exc), exc_type=type(exc).__name__, traceback=traceback.format_exc()
            )
            logger.warning("%s failed: %s", job.handle.kind, exc)
            self._events.emit(
                EVENT_JOB_FAILED,
                {
                    "kind": job.handle.kind,
                    "message": str(exc),
                    "exc_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
                job_id=job_id,
            )
        finally:
            with self._state_lock:
                self._current = None
                self._on_finished(job)
            self._job_done[job_id].set()

    def wait(self, handle: JobHandle, timeout: float | None = None) -> Any:
        """Block until ``handle`` reaches a terminal state; return or raise its outcome."""
        done = self._job_done.get(handle.job_id)
        if done is None:
            raise KeyError(f"No such job: {handle.job_id}")
        if not done.wait(timeout):
            raise TimeoutError(f"{handle.kind} did not finish within {timeout}s")
        state = self._jobs[handle.job_id]
        if state.error is not None:
            raise state.error
        return state.result

    def job_state(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def job_settled(self, job_id: str) -> bool:
        """Has the job finished *and* had its terminal event emitted?

        Not the same question as ``job_state(...).done``, which ``_execute`` sets one
        statement before the emit. Local ``wait`` blocks on this flag, so by the time it
        returns every subscriber has already seen the job's events; a client polling
        ``done`` instead would return in that window and miss them. Same guarantee,
        askable from the other side of a socket.
        """
        event = self._job_done.get(job_id)
        return bool(event is not None and event.is_set())

    def cancel_job(self, job_id: str) -> None:
        with self._state_lock:
            state = self._jobs.get(job_id)
            current = self._current
        if state is None or state.done:
            return
        # The job object holds the flag; find it by asking every queued job. Cheap:
        # the queue is short by construction (one run, a handful of reads).
        for job in list(self._queue.queue):
            if job.handle.job_id == job_id:
                job.cancel.set()
                return
        if current is not None and current.handle.job_id == job_id:
            current.cancel.set()

    @property
    def current_job_id(self) -> str | None:
        with self._state_lock:
            return None if self._current is None else self._current.handle.job_id

    def close(self) -> None:
        self._closing.set()
        with self._state_lock:
            current = self._current
        if current is not None:
            current.cancel.set()
        self._queue.put(None)
        self._thread.join(timeout=5.0)


def job_verbs(host: JobHost) -> dict[str, Callable[..., Any]]:
    """The generic lifecycle verbs every server exposes, to splice into an allowlist."""
    return {
        "poll_events": lambda since_seq, max_n=256: host.events.since(since_seq, max_n),
        "job_state": host.job_state,
        "cancel_job": host.cancel_job,
        # Settled-ness and the current sequence number together, because the client
        # needs both to reproduce local `wait`'s guarantee and asking twice would leave
        # a window in which more events arrive between the two answers.
        "job_settled": lambda job_id: {
            "settled": host.job_settled(job_id),
            "seq": host.events.last_seq,
        },
    }
