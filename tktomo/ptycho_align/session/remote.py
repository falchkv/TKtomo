"""An :class:`AlignmentSession` whose engine is on another machine.

Interchangeable with ``LocalSession`` -- the window takes one or the other and cannot
tell which, which is what the conformance suite exists to keep true. What differs is
only what happens between the call and the answer.

**Two sockets, because ZeroMQ sockets belong to one thread.** Verbs go over a
lock-guarded socket owned by whoever calls them; the event poller has its own. Sharing
one would corrupt both.

**Requests are numbered.** A verb that times out leaves an answer in flight; without an
id the client would read it as the reply to its *next* question and quietly return the
wrong summary for the rest of the session. Late replies are recognised and dropped.

**Events are pulled, not pushed.** ``poll_events`` takes the sequence number the client
last saw, so a poller that misses a beat -- or a laptop that slept -- resumes exactly
where it left off. When the server's ring has rolled past that point the batch says so,
and the client resyncs from a fresh summary rather than pretending its accumulated
history is still a prefix of the server's.

**``wait`` polls here rather than blocking there.** Serving a blocking wait would tie up
the server loop for the hour a run takes, stalling the plane reads the user is scrubbing
through while it runs.

**Closing disconnects; it does not cancel.** A closed lid should not throw away 40
minutes of iterations. Reconnecting to the same server picks the run up where it is.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from tktomo.ptycho_align.session.codec import decode, encode
from tktomo.ptycho_align.session.protocol import (
    EVENT_SUMMARY,
    Event,
    EventBatch,
    JobHandle,
    JobState,
    SessionError,
)
from tktomo.ptycho_align.session.server import DEFAULT_ADDRESS
from tktomo.ptycho_align.session.types import (
    PlaneRef,
    ResourceSample,
    RunPreflight,
    SessionSummary,
)

logger = logging.getLogger("tktomo.ptycho_align")

__all__ = ["Disconnected", "RemoteSession"]

#: How often the poller asks for new events. Fast enough that a step feels immediate,
#: slow enough to be nothing on a link shared with plane reads.
POLL_INTERVAL_S = 0.05

#: How long a verb waits for its answer before giving up. Generous, because a queued
#: read can sit behind a whole-stack materialise on the host's compute thread.
DEFAULT_TIMEOUT_S = 30.0


class Disconnected(SessionError):
    """The engine did not answer. The run, if any, is still going on over there."""


class RemoteSession:
    """Drives an engine on a server. Satisfies :class:`AlignmentSession`."""

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        poll_interval: float = POLL_INTERVAL_S,
    ) -> None:
        import zmq  # noqa: PLC0415

        self.address = address
        self._zmq = zmq
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._context = zmq.Context()
        self._lock = threading.Lock()
        self._socket = self._connect()
        self._ids = itertools.count(1)

        self._subscribers: list[Callable[[Event], None]] = []
        self._subscriber_lock = threading.RLock()
        # Held across "fetch the batch, then deliver it", so that the poller and a
        # thread inside wait() cannot interleave and deliver the same event twice or
        # out of order. Always taken *before* the socket lock, never the other way.
        self._event_lock = threading.RLock()
        self._seq = 0
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, name="session-events")
        self._poller.daemon = True
        self._poller.start()

    def _connect(self):
        socket = self._context.socket(self._zmq.DEALER)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.connect(self.address)
        return socket

    @property
    def is_remote(self) -> bool:
        return True

    def describe(self) -> str:
        return self.address

    # -- the wire ----------------------------------------------------------------------

    def _call(self, verb: str, *args, **kwargs) -> Any:
        """One request, one reply.

        One socket, serialised by a lock, for the poller as well as for verbs. A second
        socket would buy nothing: the server's ROUTER loop handles one request at a time,
        so the concurrency the client thinks it has does not exist on the other end.
        """
        with self._lock:
            return self._call_on(self._socket, verb, *args, **kwargs)

    def _call_on(self, socket, verb: str, *args, **kwargs) -> Any:
        request_id = next(self._ids)
        socket.send_multipart(
            encode({"id": request_id, "verb": verb, "args": list(args), "kwargs": kwargs})
        )

        deadline = time.monotonic() + self._timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not socket.poll(int(remaining * 1000), self._zmq.POLLIN):
                raise Disconnected(
                    f"{verb} got no answer from {self.address} within {self._timeout:g}s"
                )
            reply = decode(socket.recv_multipart())
            if reply.get("id") != request_id:
                # The answer to something we already gave up on.
                logger.debug("Dropping a late reply to request %s", reply.get("id"))
                continue
            if "error" in reply:
                raise reply["error"]
            return reply["ok"]

    # -- events ------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            try:
                self._pump()
            except Exception as exc:  # a hiccup must not kill the poller
                logger.debug("Event poll failed: %s", exc)

    def _pump(self) -> None:
        """Fetch everything new and deliver it to the subscribers."""
        with self._event_lock:
            self._absorb(self._call("poll_events", self._seq))

    def _absorb(self, batch: EventBatch) -> None:
        if batch.gap:
            # We are further behind than the server's ring. The accumulated history is
            # no longer a prefix of theirs, so replaying from here would interleave a
            # stale view with fresh events; resync from a summary instead.
            logger.info("Missed events %s..%s; resyncing", self._seq, batch.oldest_seq)
            self._seq = batch.last_seq
            try:
                summary = self.summary()
            except Exception:
                return
            self._fan_out(
                Event(seq=batch.last_seq, kind=EVENT_SUMMARY, payload={"summary": summary})
            )
            return

        for event in batch.events:
            self._seq = max(self._seq, event.seq)
            self._fan_out(event)

    def _fan_out(self, event: Event) -> None:
        with self._subscriber_lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:  # a broken listener must not stop the stream
                pass

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        with self._subscriber_lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._subscriber_lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def poll_events(self, since_seq: int, max_n: int = 256) -> EventBatch:
        return self._call("poll_events", since_seq, max_n)

    # -- state -------------------------------------------------------------------------

    def summary(self, since_iteration: int = 0) -> SessionSummary:
        return self._call("summary", since_iteration)

    def telemetry(self) -> ResourceSample | None:
        return self._call("telemetry")

    # -- cheap mutations ---------------------------------------------------------------

    def set_config(self, config: dict) -> None:
        self._call("set_config", config)

    def set_center(self, value: float) -> None:
        self._call("set_center", value)

    def cancel_run(self) -> None:
        self._call("cancel_run")

    def cancel_job(self, job_id: str) -> None:
        self._call("cancel_job", job_id)

    # -- queries -----------------------------------------------------------------------

    def list_hdf5(self, path: str) -> list:
        return self._call("list_hdf5", path)

    def run_preflight(self, n: int) -> RunPreflight:
        return self._call("run_preflight", n)

    def cost_units(self) -> float:
        return self._call("cost_units")

    def fetch_table(self, kind: str) -> bytes:
        return self._call("fetch_table", kind)

    # -- pixels ------------------------------------------------------------------------

    def read_planes(self, refs: Sequence[PlaneRef]) -> tuple[np.ndarray | None, ...]:
        return tuple(self._call("read_planes", list(refs)))

    def read_plane(self, key: str, axis: int, index: int) -> np.ndarray | None:
        return self._call("read_plane", key, axis, index)

    def read_volume_plane(
        self,
        axis: int,
        index: int,
        *,
        iteration: int | None = None,
        against: int | None = None,
    ) -> np.ndarray | None:
        return self._call("read_volume_plane", axis, index, iteration=iteration, against=against)

    # -- heavy: exclusive --------------------------------------------------------------

    def open_dataset(self, path: str, load_kwargs: dict | None = None) -> JobHandle:
        return self._call("open_dataset", path, load_kwargs)

    def apply_preprocessing(self, options, roi: tuple | None = None) -> JobHandle:
        return self._call("apply_preprocessing", options, roi)

    def reset_preprocessing(self) -> JobHandle:
        return self._call("reset_preprocessing")

    def set_bin_factor(self, factor: int) -> JobHandle:
        return self._call("set_bin_factor", factor)

    def run_com(self, vertical_reference: str = "mean") -> JobHandle:
        return self._call("run_com", vertical_reference)

    def estimate_center(self, method: str) -> JobHandle:
        return self._call("estimate_center", method)

    def start_run(self, n: int) -> JobHandle:
        return self._call("start_run", n)

    def revert(self, iteration: int) -> JobHandle:
        return self._call("revert", iteration)

    def open_session(self, path: str) -> JobHandle:
        return self._call("open_session", path)

    # -- heavy: queued -----------------------------------------------------------------

    def materialize(self, keys: Sequence[str]) -> JobHandle:
        return self._call("materialize", list(keys))

    def save_session(self, path: str, *, include_arrays: bool = True) -> JobHandle:
        return self._call("save_session", path, include_arrays=include_arrays)

    def export(self, kind: str, path: str) -> JobHandle:
        return self._call("export", kind, path)

    # -- lifecycle ---------------------------------------------------------------------

    def job_state(self, job_id: str) -> JobState | None:
        return self._call("job_state", job_id)

    def wait(self, handle: JobHandle, timeout: float | None = None) -> Any:
        """Poll until the job is settled, then return or raise its outcome.

        Matches ``EngineHost.wait``: ``KeyError`` for a job that never existed,
        ``TimeoutError`` if it outlasts ``timeout``, the job's own exception if it
        failed -- and, less obviously, the same guarantee about *events*. When the local
        version returns, every subscriber has already been handed the job's events,
        because it blocks on a flag set after they are emitted. So this asks
        ``job_settled``, which reports that same flag along with the sequence number it
        was true at, and then holds until the poller has delivered up to there.

        Delivery is left to the poller rather than done here. Pumping on the calling
        thread would be quicker, but it would mean events reaching subscribers on
        whichever thread happened to call ``wait`` -- an asymmetry with the local session
        that the conformance suite rightly refuses.

        The polling is on this side rather than the server's so that an hour-long run
        does not occupy the server loop for an hour, stalling the plane reads the user is
        scrubbing through while it runs.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        interval = 0.005
        while True:
            reply = self._call("job_settled", handle.job_id)
            if reply["settled"]:
                self._await_delivery(reply["seq"])
                return self._outcome(handle)
            if self.job_state(handle.job_id) is None:
                raise KeyError(f"No such job: {handle.job_id}")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"{handle.kind} did not finish within {timeout}s")
            time.sleep(interval)
            # Back off to the event cadence: a 60-minute run does not need 200 status
            # questions a second, and it is the first few iterations that feel slow.
            interval = min(interval * 1.5, self._poll_interval)

    def _await_delivery(self, seq: int, grace: float = 5.0) -> None:
        """Hold until the poller has handed subscribers everything up to ``seq``.

        Bounded, because a dead or stopped poller would otherwise turn a finished job
        into a hang. Exceeding the grace period means events were lost, not that the job
        failed, so it is logged and the result still returned.
        """
        deadline = time.monotonic() + grace
        while self._seq < seq:
            if self._stop.is_set() or time.monotonic() > deadline:
                logger.warning("Events up to %s were not delivered within %gs", seq, grace)
                return
            time.sleep(0.002)

    def _outcome(self, handle: JobHandle) -> Any:
        state = self.job_state(handle.job_id)
        if state is None:
            raise KeyError(f"No such job: {handle.job_id}")
        if state.error is not None:
            raise state.error
        return state.result

    def close(self) -> None:
        """Disconnect. The engine keeps running; reconnecting picks it back up."""
        self._stop.set()
        if self._poller.is_alive():
            self._poller.join(timeout=2.0)
        self._socket.close(0)
        self._context.term()
