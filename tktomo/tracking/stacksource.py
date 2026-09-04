"""Where the track-model window gets its pixels from.

The window used to hold the whole projection stack and index into it. That
is fine when the stack is a pointer away and untenable when it lives on a
cluster node: a large dataset is hundreds of MB, and the window only ever
draws ONE projection at a time. So the window holds a `StackSource` and asks
it for exactly the frame being drawn, and hands the pixel-heavy work (the
live gridrec slice, auto-track, aligned export) to the source as well, so
it runs wherever the pixels are.

Two implementations: `LocalStackSource` here (in-memory stack, the default,
what every test and the phantom use) and `RemoteStackSource` in
`tktomo.tracking.remote.client` (a ZeroMQ client to a `TrackingHost`).
Both are blocking with progress/cancel callbacks; the window's worker
threads supply the concurrency. Nothing in the window branches on which one
it has, except user-facing text and file pickers.

Qt-free, like everything under `tktomo.tracking`.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from tktomo.tracking.coords import CoordinateChain

__all__ = [
    "AlignedExportRequest",
    "LocalStackSource",
    "StackInfo",
    "StackSource",
    "ViewCache",
    "ViewPrefetcher",
    "open_stack_file",
]

logger = logging.getLogger("tktomo.tracking")

# 64 MiB. Enough that a scrub over a working range of views stays resident,
# small enough to be beneath notice next to the stack itself.
DEFAULT_CACHE_BYTES = 64 * 1024 * 1024

# How many views ahead the prefetcher runs, and the widest stride it will
# believe. A stride comes from the jump the user just made, so a one-off leap
# across the stack must not send it fetching views nobody will look at.
DEFAULT_PREFETCH_AHEAD = 3
MAX_PREFETCH_STRIDE = 32


@dataclass(frozen=True)
class StackInfo:
    """Everything about an open stack except its pixels. Crosses the wire.

    `binning`, `crop` and `view_origin` are the file's own provenance (what
    `stackio.load_tracking_stack` read from the attrs); the window may still
    override binning/crop through the provenance dialog. `rebin` is the
    run-time mean-pool the source currently applies on top of the file
    (`set_binning`): `shape` is the shape AFTER it. `epoch` bumps on every
    open and every rebin, so a cached frame from a previous grid can never
    be served.
    """

    shape: tuple[int, int, int]
    angles: np.ndarray
    kind: str | None
    binning: int = 1
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)
    view_origin: np.ndarray | None = None
    path: str | None = None
    metadata: dict = field(default_factory=dict)
    epoch: int = 0
    rebin: int = 1

    @property
    def n_views(self) -> int:
        return int(self.shape[0])


@dataclass(frozen=True)
class AlignedExportRequest:
    """Per-view transforms plus the provenance the aligned stack should carry."""

    dx: np.ndarray
    dy: np.ndarray
    rot_deg: np.ndarray
    metadata: dict = field(default_factory=dict)
    order: int = 1


@runtime_checkable
class StackSource(Protocol):
    """What the window talks to. See the module docstring."""

    @property
    def is_remote(self) -> bool: ...

    def describe(self) -> str: ...

    def info(self) -> StackInfo | None: ...

    def view(self, index: int) -> np.ndarray: ...

    def views(self, indices: Sequence[int]) -> tuple[np.ndarray, ...]: ...

    def cached(self, index: int) -> bool: ...

    def prefetch(self, index: int) -> None: ...

    def detect_format(self, path: str) -> str | None: ...

    def list_hdf5(self, path: str) -> list: ...

    def open_stack(self, path: str, *, load_kwargs: dict | None = None,
                   progress=None, cancelled=None) -> StackInfo: ...

    def set_angles(self, angles: np.ndarray) -> StackInfo: ...

    def set_binning(self, rebin: int, *, progress=None,
                    cancelled=None) -> StackInfo: ...

    def gridrec_slice(self, req) -> np.ndarray: ...

    def autotrack_available(self) -> tuple[bool, str]: ...

    def autotrack(self, jobs, *, hp_sigma: float, progress=None,
                  cancelled=None) -> list: ...

    def export_aligned(self, req: AlignedExportRequest, out_path: str, *,
                       progress=None, cancelled=None) -> str: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

class ViewCache:
    """Recent frames, bounded in bytes, keyed by (epoch, index).

    Scrubbing revisits frames constantly. Bounded in bytes rather than
    entries because a frame is 6 KiB for a phantom and several MiB for real
    data, and an entry count tuned for one is useless or ruinous for the
    other. Mirrors `tktomo.ptycho_align.ui.planes.PlaneSource`, which is
    coupled to the ptycho session's summary types.

    Locked, because `ViewPrefetcher` fills it from its own thread while the
    UI thread reads it.
    """

    def __init__(self, budget_bytes: int = DEFAULT_CACHE_BYTES) -> None:
        self._budget = int(budget_bytes)
        self._store: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            frame = self._store.get(key)
            if frame is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return frame

    def peek(self, key: tuple) -> bool:
        """Is it here? Without counting as a hit or a miss.

        The prefetcher asks constantly and is not a user of the frames, so
        letting it move entries or move the counters would make the hit rate
        say nothing about the scrubbing it exists to speed up.
        """
        return key in self._store

    def put(self, key: tuple, frame: np.ndarray) -> None:
        with self._lock:
            if key in self._store:
                self._bytes -= self._store.pop(key).nbytes
            self._store[key] = frame
            self._bytes += frame.nbytes
            while self._bytes > self._budget and len(self._store) > 1:
                _, old = self._store.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0


class ViewPrefetcher:
    """Pulls the views around the one on screen into the source's cache.

    A frame over a tunnelled link costs the better part of a second; a frame
    already in the client's `ViewCache` costs nothing. Labelling walks the
    stack in one direction with a fixed advance, so the views the user is
    about to ask for are predictable, and fetching them while they work on
    the current one hides the wait almost entirely.

    Three things are deliberate:

    **One frame at a time, replanning after each.** The client's request
    socket is serialised, so a batch in flight would put the frame the user
    just asked for behind it. One frame bounds that to a single transfer.

    **The stride is the jump the user just made**, not 1, because the advance
    box means a session may step 5 views at a time and the intervening four
    are never drawn. Capped, so one leap across the stack does not aim the
    prefetcher at nothing.

    **Failures stop it, quietly.** A dead server makes every fetch wait out
    the client timeout, so one failure ends the round and a run of them ends
    the thread. Retries then come from the user moving, not from a loop, and
    the read they are waiting on is what reports the problem to them.

    Qt-free: it takes a `StackSource` and drives it on a daemon thread. The
    window owns one only when the source is remote (a local source's cache
    is the stack itself).
    """

    def __init__(self, source: "StackSource", *,
                 ahead: int = DEFAULT_PREFETCH_AHEAD,
                 max_failures: int = 3) -> None:
        self._source = source
        self._ahead = int(ahead)
        self._max_failures = int(max_failures)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._target: int | None = None
        self._stride = 1
        self._direction = 1
        self._stopping = False
        self.fetched = 0     # frames pulled in
        self.failures = 0    # consecutive failures; a good fetch clears it
        self._thread = threading.Thread(target=self._run, name="view-prefetch",
                                        daemon=True)
        self._thread.start()

    @property
    def source(self) -> "StackSource":
        return self._source

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def want(self, index: int, step: int | None = None) -> None:
        """The window is on `index`, having just moved by `step` views."""
        with self._lock:
            self._target = int(index)
            if step:
                self._direction = 1 if step > 0 else -1
                self._stride = min(abs(int(step)), MAX_PREFETCH_STRIDE)
        self._wake.set()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop fetching and join. Safe to call twice, and after a failure."""
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout)

    # -- the thread --------------------------------------------------------------------

    def _plan(self) -> list[int]:
        """Views worth having next, nearest first, direction of travel first."""
        with self._lock:
            target, stride, direction = self._target, self._stride, self._direction
        info = self._source.info()
        if target is None or info is None:
            return []
        offsets = [d * stride * direction for d in range(1, self._ahead + 1)]
        offsets.append(-stride * direction)      # one step back, for a reversal
        return [i for i in (target + o for o in offsets) if 0 <= i < info.n_views]

    def _run(self) -> None:
        while not self._stopping:
            self._wake.clear()
            for index in self._plan():
                # A new target means this plan is stale: drop it and rebuild.
                if self._stopping or self._wake.is_set():
                    break
                if self._source.cached(index):
                    continue
                try:
                    self._source.prefetch(index)
                except Exception as exc:  # noqa: BLE001 - never take the app down
                    self.failures += 1
                    logger.debug("prefetch of view %d failed: %s", index, exc)
                    if self.failures >= self._max_failures:
                        logger.info("prefetching stopped after %d failures "
                                    "in a row", self.failures)
                        return
                    break
                else:
                    self.fetched += 1
                    self.failures = 0
            if not self._wake.is_set():
                self._wake.wait()


def _plain(value: Any) -> Any:
    """Restrict metadata to what the wire and the session file both accept."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim <= 1 else value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "as_tuple"):
        return list(value.as_tuple())
    return str(value)


def open_stack_file(path: str | Path, load_kwargs: dict | None = None,
                    *, progress=None, cancelled=None):
    """Read a stack from disk: (ProjectionData, chain, kind).

    The dispatch the "Load stack" flow has always used, minus the dialogs:
    the tracking formats through `stackio`, generic HDF5 with the browser's
    `load_kwargs` through `load_dataset`, TIFF stacks and `.npy` through
    `load_dataset` too (angles then default to an even sweep over pi, and
    the caller is expected to `set_angles` once it knows better).
    """
    from tktomo.io import ProjectionData  # noqa: PLC0415
    from tktomo.ptycho_align.core.dataset import (  # noqa: PLC0415
        coerce_load_kwargs,
        load_dataset,
    )
    from tktomo.tracking import stackio  # noqa: PLC0415

    path = Path(path)
    kind = stackio.detect_format(path)
    if kind is not None:
        data, chain = stackio.load_tracking_stack(path)
        return data, chain, kind
    kwargs = coerce_load_kwargs(load_kwargs or {})
    extra_crop = kwargs.get("crop")
    if extra_crop is not None:
        extra_crop = tuple(int(x) for x in (
            extra_crop.as_tuple() if hasattr(extra_crop, "as_tuple")
            else extra_crop))
    suffix = path.suffix.lower()
    if suffix in (".h5", ".hdf5", ".nx", ".nxs"):
        if "data_path" in kwargs and (progress is not None
                                      or cancelled is not None):
            def report(done, total):
                if progress is not None:
                    progress(done / max(total, 1), f"reading {done}/{total}")
                return not (cancelled is not None and cancelled())
            kwargs.setdefault("progress", report)
        data = load_dataset(path, **kwargs)
        kind = "generic_h5"
    elif suffix in (".npy", ".npz"):
        data = load_dataset(path)
        kind = "npy"
    else:
        import tifffile  # noqa: PLC0415
        stack = np.asarray(tifffile.imread(str(path)), np.float32)
        if stack.ndim != 3:
            raise ValueError(f"{path.name} is not a 3-D stack "
                             f"(shape {stack.shape})")
        data = ProjectionData(
            data=stack, angles=np.linspace(0, np.pi, stack.shape[0],
                                           endpoint=False),
            metadata={"source_path": str(path), "angles": "assumed"})
        kind = "tiff"
    data.data = np.asarray(data.data, np.float32)
    return data, CoordinateChain(extra_crop=extra_crop), kind


# ---------------------------------------------------------------------------
# the in-process source
# ---------------------------------------------------------------------------

class LocalStackSource:
    """A stack held in this process. The default, and what the phantom uses."""

    def __init__(self, data=None, *, chain: CoordinateChain | None = None,
                 kind: str | None = "phantom", path: str | None = None) -> None:
        from tktomo.tracking.autotrack import HighpassCache  # noqa: PLC0415

        self._data = None      # what is served: the base, mean-pooled by _rebin
        self._base = None      # the stack as loaded
        self._rebin = 1
        self._info: StackInfo | None = None
        self._epoch = 0
        self._hp_cache = HighpassCache()
        self._matcher = None
        if data is not None:
            self._adopt(data, chain or CoordinateChain(), kind, path)

    @classmethod
    def from_projection_data(cls, data, chain: CoordinateChain | None = None,
                             kind: str | None = "phantom",
                             path: str | None = None) -> "LocalStackSource":
        return cls(data, chain=chain, kind=kind, path=path)

    # -- identity ----------------------------------------------------------------------

    @property
    def is_remote(self) -> bool:
        return False

    def describe(self) -> str:
        return "this machine"

    def info(self) -> StackInfo | None:
        return self._info

    @property
    def data(self):
        """The in-memory `ProjectionData` as served (rebinned), for in-process
        code that wants it all."""
        return self._data

    @property
    def rebin(self) -> int:
        return self._rebin

    @property
    def n_views(self) -> int:
        return 0 if self._info is None else self._info.n_views

    @property
    def shape(self) -> tuple[int, int, int]:
        return (0, 0, 0) if self._info is None else self._info.shape

    @property
    def angles(self) -> np.ndarray:
        return np.empty(0) if self._info is None else self._info.angles

    # -- pixels ------------------------------------------------------------------------

    def view(self, index: int) -> np.ndarray:
        self._require()
        n = self._data.data.shape[0]
        if not 0 <= index < n:
            raise IndexError(f"view {index} out of range for {n} views")
        return np.array(self._data.data[index], np.float32)

    def views(self, indices: Sequence[int]) -> tuple[np.ndarray, ...]:
        return tuple(self.view(int(i)) for i in indices)

    def cached(self, index: int) -> bool:
        """Always: the whole stack is here. Nothing to prefetch."""
        return True

    def prefetch(self, index: int) -> None:
        """Nothing to do: the frame is a slice away."""

    def slab(self, lo: int, hi: int) -> np.ndarray:
        self._require()
        return self._data.data[:, lo:hi, :]

    # -- files -------------------------------------------------------------------------

    def detect_format(self, path: str) -> str | None:
        from tktomo.tracking import stackio  # noqa: PLC0415

        return stackio.detect_format(path)

    def list_hdf5(self, path: str) -> list:
        from tktomo.ptycho_align.core.dataset import (  # noqa: PLC0415
            list_hdf5_datasets,
        )

        return list_hdf5_datasets(path)

    def open_stack(self, path: str, *, load_kwargs: dict | None = None,
                   progress=None, cancelled=None) -> StackInfo:
        data, chain, kind = open_stack_file(path, load_kwargs,
                                            progress=progress,
                                            cancelled=cancelled)
        return self._adopt(data, chain, kind, str(path))

    def set_angles(self, angles) -> StackInfo:
        self._require()
        angles = np.asarray(angles, float)
        if angles.shape != (self._data.data.shape[0],):
            raise ValueError("angles must have one entry per view")
        self._data.angles = angles
        self._base.angles = angles
        self._info = StackInfo(**{**self._info.__dict__, "angles": angles})
        return self._info

    def set_binning(self, rebin: int, *, progress=None,
                    cancelled=None) -> StackInfo:
        """Serve the stack mean-pooled by `rebin` (1 restores the file's grid).

        The pool is over the loaded frame, so `CoordinateChain.with_rebin`
        is the matching coordinate change: labels in raw px stay put. The
        base stack is kept, so going back to 1 costs nothing and any factor
        pools the original rather than compounding rounding.
        """
        from tktomo.io import ProjectionData  # noqa: PLC0415
        from tktomo.ptycho_align.core.preprocess import bin_stack  # noqa: PLC0415

        self._require()
        rebin = int(rebin)
        if rebin < 1:
            raise ValueError(f"rebin must be >= 1, got {rebin}")
        if rebin == self._rebin:
            return self._info
        if rebin == 1:
            data = self._base
        else:
            if progress is not None:
                progress(0.0, f"binning by {rebin}")
            data = ProjectionData(data=bin_stack(self._base.data, rebin),
                                  angles=self._base.angles,
                                  metadata=self._base.metadata)
            if cancelled is not None and cancelled():
                return self._info
            if progress is not None:
                progress(1.0, "done")
        self._epoch += 1
        # the high-pass cache is keyed on the base stack and the track
        # grid, not on the served grid, so it survives a display rebin
        self._data = data
        self._rebin = rebin
        self._info = StackInfo(**{**self._info.__dict__,
                                  "shape": tuple(int(x) for x in data.data.shape),
                                  "epoch": self._epoch, "rebin": rebin})
        return self._info

    def _adopt(self, data, chain: CoordinateChain, kind, path) -> StackInfo:
        self._epoch += 1
        self._hp_cache.clear()
        self._data = self._base = data
        self._rebin = 1
        self._info = StackInfo(
            shape=tuple(int(x) for x in data.data.shape),
            angles=np.asarray(data.angles, float),
            kind=kind,
            binning=int(chain.binning),
            crop=tuple(int(x) for x in chain.crop),
            view_origin=(None if chain.view_origin is None
                         else np.asarray(chain.view_origin, float)),
            path=path,
            metadata=_plain(dict(getattr(data, "metadata", {}) or {})),
            epoch=self._epoch,
        )
        return self._info

    def _require(self) -> None:
        if self._data is None:
            raise RuntimeError("no stack is open")

    # -- compute -----------------------------------------------------------------------

    def gridrec_slice(self, req) -> np.ndarray:
        from tktomo.tracking.recon import reconstruct_slice  # noqa: PLC0415

        self._require()
        return reconstruct_slice(self.slab(req.lo, req.hi), self._data.angles,
                                 req)

    def autotrack_available(self) -> tuple[bool, str]:
        from tktomo.tracking import learned_match  # noqa: PLC0415

        return learned_match.available()

    def _learned_matcher(self):
        if self._matcher is None:
            from tktomo.tracking.learned_match import (  # noqa: PLC0415
                LearnedMatcher,
            )
            self._matcher = LearnedMatcher()
        return self._matcher

    def autotrack(self, jobs, *, hp_sigma: float, progress=None,
                  cancelled=None) -> list:
        """Complete the jobs on the base stack, each on its `track_bin`.

        The high-passed copy of the track grid is what costs memory (the
        pooled frames are never materialised): n * ny/b * nx/b float32.
        When that does not fit in 60 percent of what is free, the bin is
        stepped up and every result says so, with the numbers.
        """
        from dataclasses import replace  # noqa: PLC0415

        from tktomo.ptycho_align.core.telemetry import (  # noqa: PLC0415
            available_ram_bytes,
        )
        from tktomo.tracking.autotrack import run_autotrack  # noqa: PLC0415

        self._require()
        base = self._base.data
        n, ny, nx = base.shape
        free = available_ram_bytes()
        notes = []
        jobs = list(jobs)
        for i, job in enumerate(jobs):
            b = int(job.track_bin) or self._rebin
            need = _highpass_bytes(base.shape, b)
            if free is not None and need > 0.6 * free:
                chosen = b
                while chosen < 8 and _highpass_bytes(base.shape, chosen) \
                        > 0.6 * free:
                    chosen *= 2
                note = (f"feature {job.fid} tracked at bin {chosen} instead "
                        f"of bin {b}: the high-passed copy at bin {b} needs "
                        f"{need / 1e9:.1f} GB and {free / 1e9:.1f} GB are "
                        f"free. Close other stacks, or run the stack server "
                        f"on a node with more memory, to track at bin {b}.")
                logger.warning(note)
                notes.append((i, note))
                jobs[i] = replace(job, track_bin=chosen)
        logger.info("autotrack: %d job(s) on %dx%dx%d, served bin %d, "
                    "track bins %s, hp sigma %.1f", len(jobs), n, ny, nx,
                    self._rebin,
                    sorted({int(j.track_bin) or self._rebin for j in jobs}),
                    hp_sigma)
        out = run_autotrack(base, self._base.angles, jobs,
                            served_bin=self._rebin, hp_sigma=hp_sigma,
                            matcher=self._learned_matcher(),
                            cache=self._hp_cache, progress=progress,
                            cancelled=cancelled)
        for i, note in notes:
            if i < len(out) and out[i] is not None:
                out[i][1].warnings.insert(0, note)
        for fid, res in out:
            st = res.stats
            logger.info("autotrack feature %s: %d labels of %d views at bin "
                        "%s (patch %s px, radius %.1f px), %d none, %d low p, "
                        "%d fb, %d stopped, %d seed(s) refused", fid,
                        st.get("n_accepted", 0), st.get("n_unlabelled", 0),
                        st.get("track_bin"), st.get("patch_track_px"),
                        st.get("search_radius_track_px", float("nan")),
                        st.get("n_none", 0), st.get("n_low_p", 0),
                        st.get("n_fb_miss", 0) + st.get("n_fb_corr", 0),
                        st.get("n_stopped", 0), st.get("n_seeds_refused", 0))
        return out

    def export_aligned(self, req: AlignedExportRequest, out_path: str, *,
                       progress=None, cancelled=None) -> str:
        from tktomo.align.transform import Transform  # noqa: PLC0415
        from tktomo.io import ProjectionData, save_projections  # noqa: PLC0415
        from tktomo.tracking.export import warp_stack  # noqa: PLC0415

        self._require()
        transforms = [Transform(dx=float(x), dy=float(y), rotation=float(r))
                      for x, y, r in zip(req.dx, req.dy, req.rot_deg)]

        def report(done, total):
            if progress is not None:
                progress(done / max(total, 1), f"warping {done}/{total}")
            return not (cancelled is not None and cancelled())

        out = warp_stack(self._data.data, transforms, order=int(req.order),
                         progress=report)
        aligned = ProjectionData(data=out, angles=self._data.angles.copy(),
                                 metadata=dict(req.metadata))
        save_projections(out_path, aligned)
        return str(out_path)

    def close(self) -> None:
        self._data = self._base = None
        self._rebin = 1
        self._info = None
        self._hp_cache.clear()


def _highpass_bytes(shape, track_bin: int) -> int:
    """Memory of the high-passed float32 copy of a stack on the track grid."""
    n, ny, nx = (int(x) for x in shape[:3])
    b = int(track_bin)
    return n * (ny // b) * (nx // b) * 4
