"""`RemoteStackSource`: the window's `StackSource` when the stack is elsewhere.

One frame per view change, through a byte-bounded cache; the compute verbs
submit a job to the host and block (on the calling worker thread) until it
settles, relaying progress events and forwarding cancellation. Same
blocking-with-callbacks contract as `LocalStackSource`, so the window's
worker threads do not know which one they hold.

Frames arrive packed to display precision (`types.FramePacket`) unless the
caller asks for exactness, because the link this exists for carries well
under a megabyte a second. Everything that computes on pixels runs on the
host, so the packing never reaches a number anyone fits a model to.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from tktomo.ptycho_align.session.protocol import EVENT_PROGRESS, JobHandle
from tktomo.ptycho_align.session.remote import RemoteClient
from tktomo.tracking.remote import types as _types  # noqa: F401 - registers the codec
from tktomo.tracking.remote.server import DEFAULT_ADDRESS
from tktomo.tracking.stacksource import (
    AlignedExportRequest,
    StackInfo,
    ViewCache,
)

logger = logging.getLogger("tktomo.tracking")

__all__ = ["RemoteStackSource"]


class RemoteStackSource(RemoteClient):
    """Satisfies `tktomo.tracking.stacksource.StackSource` over a socket."""

    def __init__(self, address: str = DEFAULT_ADDRESS, *,
                 cache_bytes: int | None = None, quantise: bool = True,
                 compress: bool = True, **kwargs) -> None:
        super().__init__(address, **kwargs)
        self._cache = ViewCache() if cache_bytes is None else ViewCache(cache_bytes)
        self._info: StackInfo | None = None
        self._info_fresh = False
        self._quantise = bool(quantise)
        self._compress = bool(compress)
        # None until the first frame says whether this host knows read_frames.
        self._packed: bool | None = None

    # -- identity ----------------------------------------------------------------------

    def describe(self) -> str:
        return self.address

    @property
    def cache(self) -> ViewCache:
        return self._cache

    @property
    def packs_frames(self) -> bool:
        """Do frames arrive quantised to display precision? See `FramePacket`."""
        return self._quantise

    def info(self) -> StackInfo | None:
        if not self._info_fresh:
            self._info = self._call("info")
            self._info_fresh = True
        return self._info

    def refresh(self) -> StackInfo | None:
        self._info_fresh = False
        return self.info()

    @property
    def n_views(self) -> int:
        info = self.info()
        return 0 if info is None else info.n_views

    @property
    def shape(self) -> tuple[int, int, int]:
        info = self.info()
        return (0, 0, 0) if info is None else info.shape

    @property
    def angles(self) -> np.ndarray:
        info = self.info()
        return np.empty(0) if info is None else info.angles

    # -- pixels ------------------------------------------------------------------------

    def view(self, index: int) -> np.ndarray:
        info = self._need_info()
        key = (info.epoch, int(index))
        frame = self._cache.get(key)
        if frame is None:
            frame = self._fetch([int(index)])[0]
            self._cache.put(key, frame)
        return frame.copy()

    def views(self, indices: Sequence[int]) -> tuple[np.ndarray, ...]:
        info = self._need_info()
        missing = [int(i) for i in indices
                   if self._cache.get((info.epoch, int(i))) is None]
        if missing:
            for i, frame in zip(missing, self._fetch(missing)):
                self._cache.put((info.epoch, i), frame)
        return tuple(self.view(i) for i in indices)

    def cached(self, index: int) -> bool:
        """Would `view(index)` answer without going to the host?

        What the prefetcher asks before spending a transfer, so it must not
        count as a cache hit and must not itself go to the host.
        """
        info = self._info if self._info_fresh else self.info()
        return info is not None and self._cache.peek((info.epoch, int(index)))

    def prefetch(self, index: int) -> None:
        """Put a frame in the cache for later, without counting as a read.

        The prefetcher is not a user of frames: if its fetches moved the hit
        counters, they would stop saying anything about the scrubbing this
        exists to speed up. It also skips the copy `view` owes its caller.
        """
        info = self._need_info()
        key = (info.epoch, int(index))
        if not self._cache.peek(key):
            self._cache.put(key, self._fetch([int(index)])[0])

    def _need_info(self) -> StackInfo:
        info = self.info()
        if info is None:
            raise RuntimeError("no stack is open on the server")
        return info

    def _fetch(self, indices: list[int]) -> list[np.ndarray]:
        """Frames from the host, packed where the host can pack them."""
        if self._packed is not False:
            try:
                packets = self._call("read_frames", indices, self._quantise,
                                     self._compress)
            except KeyError:
                # A host from before packed frames. Say so once and use the
                # plain verb from here on rather than paying a failed round
                # trip per view.
                logger.info("%s has no read_frames verb: sending frames "
                            "verbatim", self.address)
                self._packed = False
            else:
                self._packed = True
                return [p.unpack() for p in packets]
        return list(self._call("read_views", indices))

    # -- files -------------------------------------------------------------------------

    def detect_format(self, path: str) -> str | None:
        return self._call("detect_format", str(path))

    def list_hdf5(self, path: str) -> list:
        return self._call("list_hdf5", str(path))

    def open_stack(self, path: str, *, load_kwargs: dict | None = None,
                   progress=None, cancelled=None) -> StackInfo:
        handle = self._call("open_stack", str(path), load_kwargs)
        info = self._wait_with_progress(handle, progress, cancelled)
        self._info, self._info_fresh = info, True
        self._cache.clear()
        return info

    def set_angles(self, angles) -> StackInfo:
        info = self._call("set_angles", np.asarray(angles, float))
        self._info, self._info_fresh = info, True
        return info

    def set_binning(self, rebin: int, *, progress=None,
                    cancelled=None) -> StackInfo:
        handle = self._call("set_binning", int(rebin))
        info = self._wait_with_progress(handle, progress, cancelled)
        self._info, self._info_fresh = info, True
        self._cache.clear()
        return info

    # -- compute -----------------------------------------------------------------------

    def gridrec_slice(self, req) -> np.ndarray:
        return self._wait_with_progress(self._call("gridrec_slice", req), None, None)

    def autotrack_available(self) -> tuple[bool, str]:
        ok, why = self._call("autotrack_available")
        return bool(ok), str(why)

    def autotrack(self, jobs, *, hp_sigma: float, progress=None,
                  cancelled=None) -> list:
        handle = self._call("autotrack", list(jobs), float(hp_sigma))

        def relay(payload: dict) -> None:
            if progress is not None and "done" in payload:
                progress(payload["done"], payload["total"], payload.get("fid"))

        result = self._wait_with_progress(handle, None, cancelled, raw=relay)
        return [] if result is None else list(result)

    def export_aligned(self, req: AlignedExportRequest, out_path: str, *,
                       progress=None, cancelled=None) -> str:
        handle = self._call("export_aligned", req, str(out_path))
        return self._wait_with_progress(handle, progress, cancelled)

    # -- the blocking wait -------------------------------------------------------------

    def _wait_with_progress(self, handle: JobHandle, progress, cancelled, *,
                            raw=None):
        """Block until the job settles, relaying its progress events meanwhile.

        `progress(fraction, message)` gets the generic reports; `raw(payload)`
        gets every progress payload for callers with their own shape.
        `cancelled()` turning true sends one cancel_job and keeps waiting for
        the host to acknowledge, so the outcome is the host's, not a guess.
        """
        def on_event(event) -> None:
            if event.job_id != handle.job_id or event.kind != EVENT_PROGRESS:
                return
            payload = event.payload or {}
            if raw is not None:
                raw(payload)
            if progress is not None and "fraction" in payload:
                progress(payload["fraction"], payload.get("message", ""))

        unsubscribe = self.subscribe(on_event)
        sent_cancel = False
        try:
            while True:
                try:
                    return self.wait(handle, timeout=self._poll_interval * 4)
                except TimeoutError:
                    pass
                if not sent_cancel and cancelled is not None and cancelled():
                    self.cancel_job(handle.job_id)
                    sent_cancel = True
                time.sleep(0)
        finally:
            unsubscribe()
