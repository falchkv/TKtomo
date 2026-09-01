"""The stack host: what `tktomo-track-server` runs on the node with the data.

Holds one projection stack in RAM (the node has the memory; the laptop is
what does not) and answers the `StackSource` verbs. Cheap verbs (one frame,
the info, a directory listing) are served straight from the ROUTER thread;
the compute verbs (open, gridrec slice, auto-track, aligned export) are
jobs on the single compute thread inherited from `JobHost`, with progress
events and cancellation. There is deliberately no verb that returns the
whole stack.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np

from tktomo.ptycho_align.core.engine import Cancelled
from tktomo.ptycho_align.session.jobs import JobContext, JobHost
from tktomo.ptycho_align.session.protocol import JobHandle
from tktomo.tracking.remote.types import FramePacket, NoStack, pack_frame
from tktomo.tracking.stacksource import (
    AlignedExportRequest,
    LocalStackSource,
    StackInfo,
)

logger = logging.getLogger("tktomo.tracking")

__all__ = ["TrackingHost"]


class TrackingHost(JobHost):
    """A `LocalStackSource` behind a job queue. See the module docstring."""

    def __init__(self, *, event_capacity: int = 5000) -> None:
        self._local = LocalStackSource()
        self._h5_lock = threading.Lock()  # h5py is not reliably thread-safe
        super().__init__(event_capacity=event_capacity, thread_name="track-compute")

    # -- cheap verbs (ROUTER thread) ---------------------------------------------------

    def info(self) -> StackInfo | None:
        return self._local.info()

    def detect_format(self, path: str) -> str | None:
        with self._h5_lock:
            return self._local.detect_format(path)

    def list_hdf5(self, path: str) -> list:
        with self._h5_lock:
            return self._local.list_hdf5(path)

    def autotrack_available(self) -> tuple[bool, str]:
        return self._local.autotrack_available()

    def read_view(self, index: int) -> np.ndarray:
        # The stack is rebound only by open_stack on the compute thread and the
        # arrays are never written in place, so a reference taken here stays valid.
        self._require()
        return self._local.view(int(index))

    def read_views(self, indices: Sequence[int]) -> tuple[np.ndarray, ...]:
        self._require()
        return self._local.views([int(i) for i in indices])

    def read_frames(self, indices: Sequence[int], quantise: bool = True,
                    compress: bool = True) -> list[FramePacket]:
        """The same frames, packed small. What a remote window actually calls.

        `read_view` stays for a client that wants the bytes verbatim, and
        because a window built before packed frames existed still talks to
        this host through it.
        """
        self._require()
        return [pack_frame(self._local.view(int(i)), quantise=bool(quantise),
                           compress=bool(compress))
                for i in indices]

    def set_angles(self, angles) -> StackInfo:
        self._require()
        return self._local.set_angles(np.asarray(angles, float))

    def _require(self) -> None:
        if self._local.info() is None:
            raise NoStack("Open a stack on the server first.")

    # -- jobs (compute thread) ---------------------------------------------------------

    def open_stack(self, path: str, load_kwargs: dict | None = None) -> JobHandle:
        def call(ctx: JobContext) -> StackInfo:
            def cancelled() -> bool:
                return ctx.is_set()

            with self._h5_lock:
                info = self._local.open_stack(str(path), load_kwargs=load_kwargs,
                                              progress=ctx.report,
                                              cancelled=cancelled)
            if ctx.is_set():
                raise Cancelled("open cancelled")
            logger.info("Opened %s: %s views of %sx%s", path, *info.shape)
            return info

        return self._submit("open_stack", call, priority=0)

    def set_binning(self, rebin: int) -> JobHandle:
        """Rebin on the compute thread: pooling a large stack takes seconds,
        and it rebinds the served array, which only that thread may do."""
        def call(ctx: JobContext) -> StackInfo:
            self._require()
            info = self._local.set_binning(int(rebin), progress=ctx.report,
                                           cancelled=ctx.is_set)
            logger.info("Serving %s at bin %d: %sx%s", info.path, info.rebin,
                        *info.shape[1:])
            return info

        return self._submit("set_binning", call, priority=0)

    def gridrec_slice(self, req) -> JobHandle:
        def call(ctx: JobContext) -> np.ndarray:
            self._require()
            return self._local.gridrec_slice(req)

        # Ahead of auto-track in the queue: a slider tick should not sit behind a
        # batch that takes a minute.
        return self._submit("gridrec_slice", call, priority=0)

    def autotrack(self, jobs, hp_sigma: float) -> JobHandle:
        def call(ctx: JobContext) -> list:
            self._require()

            def progress(done: int, total: int, fid) -> None:
                ctx.emit({"done": int(done), "total": int(total), "fid": fid})

            return self._local.autotrack(list(jobs), hp_sigma=float(hp_sigma),
                                         progress=progress, cancelled=ctx.is_set)

        return self._submit("autotrack", call, priority=1)

    def export_aligned(self, req: AlignedExportRequest, out_path: str) -> JobHandle:
        def call(ctx: JobContext) -> str:
            self._require()
            path = self._local.export_aligned(req, str(out_path),
                                              progress=ctx.report,
                                              cancelled=ctx.is_set)
            if ctx.is_set():
                raise Cancelled("export cancelled")
            return path

        return self._submit("export_aligned", call, priority=1)

    def close(self) -> None:
        super().close()
        self._local.close()
