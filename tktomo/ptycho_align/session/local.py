"""An :class:`AlignmentSession` backed by an engine in this process.

Deliberately thin. All the interesting behaviour -- the single compute thread, the state
lock, the job queue, the ``Busy`` rejections, the event log -- lives in
:class:`~tktomo.ptycho_align.session.engine_host.EngineHost`, which the remote server
also drives. If any of that logic lived here instead, local and remote would differ in
exactly the places that are hard to get right, and testing one would say nothing about
the other.

So this class exists to give the window a stable surface and to keep ``EngineHost``'s
internals out of it, not to add behaviour of its own.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from tktomo.ptycho_align.session.engine_host import EngineHost
from tktomo.ptycho_align.session.protocol import (
    Event,
    EventBatch,
    JobHandle,
    JobState,
)
from tktomo.ptycho_align.session.types import (
    PlaneRef,
    ResourceSample,
    RunPreflight,
    SessionSummary,
)

__all__ = ["LocalSession"]


class LocalSession:
    """Drives an engine in-process. Satisfies :class:`AlignmentSession`."""

    def __init__(self, host: EngineHost | None = None) -> None:
        self._host = host if host is not None else EngineHost()

    @property
    def host(self) -> EngineHost:
        """The underlying host. For tests and for the server, not for the window."""
        return self._host

    @property
    def is_remote(self) -> bool:
        return False

    def describe(self) -> str:
        return "this machine"

    # -- events and state ---------------------------------------------------------

    def summary(self, since_iteration: int = 0) -> SessionSummary:
        return self._host.summary(since_iteration)

    def poll_events(self, since_seq: int, max_n: int = 256) -> EventBatch:
        return self._host.events.since(since_seq, max_n)

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        return self._host.events.subscribe(callback)

    def telemetry(self) -> ResourceSample | None:
        return self._host.telemetry()

    # -- cheap mutations ----------------------------------------------------------

    def set_config(self, config: dict) -> None:
        self._host.set_config(config)

    def set_center(self, value: float) -> None:
        self._host.set_center(value)

    def cancel_run(self) -> None:
        self._host.cancel_run()

    def cancel_job(self, job_id: str) -> None:
        self._host.cancel_job(job_id)

    # -- queries ------------------------------------------------------------------

    def list_hdf5(self, path: str) -> list:
        return self._host.list_hdf5(path)

    def run_preflight(self, n: int) -> RunPreflight:
        return self._host.run_preflight(n)

    def cost_units(self) -> float:
        return self._host.cost_units()

    def fetch_table(self, kind: str) -> bytes:
        return self._host.fetch_table(kind)

    # -- pixels -------------------------------------------------------------------
    #
    # Planes only, even though the arrays are right there in this process. Exposing the
    # whole stack here because it happens to be cheap locally is how the two
    # implementations drift: the window would grow a dependency on it, and the remote
    # session could not follow. `host.read_stack` remains for code that legitimately
    # runs where the array lives.

    def read_planes(self, refs: Sequence[PlaneRef]) -> tuple[np.ndarray | None, ...]:
        return self._host.read_planes(refs)

    def read_plane(self, key: str, axis: int, index: int) -> np.ndarray | None:
        return self._host.read_plane(key, axis, index)

    def read_volume_plane(
        self,
        axis: int,
        index: int,
        *,
        iteration: int | None = None,
        against: int | None = None,
    ) -> np.ndarray | None:
        return self._host.read_volume_plane(axis, index, iteration=iteration, against=against)

    # -- heavy: exclusive ---------------------------------------------------------

    def open_dataset(self, path: str, load_kwargs: dict | None = None) -> JobHandle:
        return self._host.open_dataset(path, load_kwargs)

    def apply_preprocessing(self, options, roi: tuple | None = None) -> JobHandle:
        return self._host.apply_preprocessing(options, roi)

    def reset_preprocessing(self) -> JobHandle:
        return self._host.reset_preprocessing()

    def set_bin_factor(self, factor: int) -> JobHandle:
        return self._host.set_bin_factor(factor)

    def run_com(self, vertical_reference: str = "mean") -> JobHandle:
        return self._host.run_com(vertical_reference)

    def estimate_center(self, method: str) -> JobHandle:
        return self._host.estimate_center(method)

    def start_run(self, n: int) -> JobHandle:
        return self._host.start_run(n)

    def revert(self, iteration: int) -> JobHandle:
        return self._host.revert(iteration)

    def open_session(self, path: str) -> JobHandle:
        return self._host.open_session(path)

    # -- heavy: queued ------------------------------------------------------------

    def materialize(self, keys: Sequence[str]) -> JobHandle:
        return self._host.materialize(keys)

    def save_session(self, path: str, *, include_arrays: bool = True) -> JobHandle:
        return self._host.save_session(path, include_arrays=include_arrays)

    def export(self, kind: str, path: str) -> JobHandle:
        return self._host.export(kind, path)

    # -- lifecycle ----------------------------------------------------------------

    def wait(self, handle: JobHandle, timeout: float | None = None) -> Any:
        return self._host.wait(handle, timeout)

    def job_state(self, job_id: str) -> JobState | None:
        return self._host.job_state(job_id)

    def close(self) -> None:
        self._host.close()
