"""Owns the engine and everything heavy that touches it.

This is the concurrency model, and it lives here rather than in the server on purpose.
If the in-process session were a plain synchronous wrapper and only the remote one were
threaded and event-driven, the two would differ in exactly the dimension that is hard to
get right -- and a conformance suite comparing them would prove nothing. Sharing this
class means local and remote differ only by transport.

Three rules hold the design together.

**One compute thread.** ``tomopy.util.mproc`` keeps its shared arrays in module-level
globals, so ``project()`` running on one thread while ``recon()`` runs on another
corrupts them and segfaults the process -- no exception, no traceback. Every tomopy
call, every whole-stack numpy pass and every bulk h5py read goes through a single
worker thread, and nothing else may touch them.

**The state lock guards reference capture, not work.** ``state.original`` is never
modified in place and the engine copies tomopy's recycled buffers before storing them,
so a reader can take a reference under the lock, release it, and then read the array at
leisure. Holding the lock is nanoseconds; a slice read is never blocked behind a
60-minute reconstruction. Three things break that invariant and are handled explicitly:
``VolumePolicy.apply`` nulls the volume on history entries a reader may be holding (so
resolve iteration to *array* inside the lock), ``invalidate_cache`` rebinds the cached
stacks, and ``shift_images`` mutates its argument in place.

**Config changes land between iterations, never inside one.** ``engine.step`` snapshots
``cfg = self.config`` at the top but re-reads ``self.config`` in the reconstruct,
reproject and conditioning helpers, so swapping the algorithm mid-iteration can
reconstruct with one and reproject with another. Staging the change here closes that.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from tktomo.io import ProjectionData
from tktomo.ptycho_align.core import preprocess as pp
from tktomo.ptycho_align.core.com import ComResult, com_prealign, find_center
from tktomo.ptycho_align.core.dataset import coerce_load_kwargs, inspect_dataset, load_dataset
from tktomo.ptycho_align.core.engine import (
    AlignConfig,
    AlignmentEngine,
    Cancelled,
    algorithm_rejects_negatives,
)
from tktomo.ptycho_align.core.estimates import format_bytes, iteration_cost_units
from tktomo.ptycho_align.core.telemetry import ResourceMonitor, available_ram_bytes
from tktomo.ptycho_align.session.protocol import (
    EVENT_ITERATION,
    EVENT_ITERATION_STARTED,
    EVENT_JOB_FAILED,
    EVENT_JOB_FINISHED,
    EVENT_JOB_STARTED,
    EVENT_PROGRESS,
    EVENT_RUN_FINISHED,
    EVENT_RUN_STARTED,
    EVENT_SUMMARY,
    Busy,
    EventLog,
    JobFailed,
    JobHandle,
    JobState,
    NoEngine,
)
from tktomo.ptycho_align.session.types import (
    DatasetSummary,
    IterationSummary,
    PlaneRef,
    PreprocessReport,
    RunPreflight,
    SessionSummary,
    StackSpec,
)

logger = logging.getLogger("tktomo.ptycho_align")

# Stack keys. They match the MODE_* constants the viewers use, but are defined here so
# the headless side does not have to import anything from ui/.
STACK_RAW = "Raw"
STACK_ALIGNED = "Aligned"
STACK_REPROJECTION = "Reprojection"
STACK_DIFFERENCE = "Difference"
STACK_KEYS = (STACK_RAW, STACK_ALIGNED, STACK_REPROJECTION, STACK_DIFFERENCE)

# Verbs that mutate the alignment. Refused outright while a run is in flight rather than
# queued behind it -- see the protocol module for why waiting an hour and then clearing
# the user's history is the worse failure.
EXCLUSIVE = frozenset(
    {
        "open_dataset",
        "apply_preprocessing",
        "reset_preprocessing",
        "set_bin_factor",
        "run_com",
        "estimate_center",
        "revert",
        "open_session",
    }
)


def _plane(array: np.ndarray | None, axis: int, index: int) -> np.ndarray | None:
    """Slice one 2-D plane out of a 3-D array, as a contiguous copy.

    A copy, always. The slice would otherwise be a view onto ``state.original`` or onto
    a cached stack -- handing that out lets a viewer mutate engine state, and over a wire
    a non-contiguous view has to be copied to be encoded anyway.

    ``None`` for an out-of-range index rather than an ``IndexError``: a slider can
    legitimately still hold last dataset's position when a smaller one is opened, and
    the viewers already treated "no frame there" as a thing that happens.
    """
    if array is None:
        return None
    if not 0 <= index < array.shape[axis]:
        return None
    return np.ascontiguousarray(np.take(array, index, axis=axis))


@dataclass(order=True)
class _Job:
    """One unit of heavy work, ordered so cheap mutations overtake bulk reads."""

    priority: int
    sequence: int
    handle: JobHandle = field(compare=False)
    call: Callable[["_JobContext"], Any] = field(compare=False)
    cancel: threading.Event = field(compare=False, default_factory=threading.Event)


class _JobContext:
    """What a running job may report through. Also its cancellation flag.

    ``is_set`` is here because that is the entire interface ``engine.step`` requires of
    a cancel object -- it never asks for a ``threading.Event``, only something that can
    say whether it has been asked to stop.
    """

    def __init__(self, host: "EngineHost", job: _Job) -> None:
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


class EngineHost:
    """Serialises every heavy operation onto one thread and narrates what it did."""

    def __init__(self, *, event_capacity: int = 5000) -> None:
        self._events = EventLog(capacity=event_capacity)
        self._state_lock = threading.RLock()
        self._h5_lock = threading.Lock()  # h5py is not reliably thread-safe
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._jobs: dict[str, JobState] = {}
        self._job_done: dict[str, threading.Event] = {}
        self._sequence = 0
        self._closing = threading.Event()

        # Engine state. Guarded by _state_lock for *rebinding*; the arrays themselves are
        # treated as immutable once published.
        self.engine: AlignmentEngine | None = None
        self.raw: ProjectionData | None = None
        self.preprocessed: ProjectionData | None = None
        self.com: ComResult | None = None
        self._source: dict | None = None
        self._config = AlignConfig()
        self._pending_config: AlignConfig | None = None
        self._bin_factor = 1
        self._epoch = 0
        self._pixel_epoch = 0
        self._run_active = False
        self._run_cancel = threading.Event()
        self._current_job: str | None = None
        self._monitor = ResourceMonitor()

        self._thread = threading.Thread(target=self._loop, name="ptycho-align-compute", daemon=True)
        self._thread.start()

    # -- event access ------------------------------------------------------------------

    @property
    def events(self) -> EventLog:
        return self._events

    # -- job plumbing ------------------------------------------------------------------

    def _submit(self, kind: str, call, *, priority: int = 1) -> JobHandle:
        """Queue heavy work. Rejects exclusive verbs while a run is in flight."""
        if kind in EXCLUSIVE and self.is_running:
            raise Busy(
                f"A run is in progress; stop it before {kind.replace('_', ' ')}. "
                "(Queuing it would apply it to a different state an hour from now.)"
            )
        handle = JobHandle.new(kind)
        with self._state_lock:
            self._sequence += 1
            job = _Job(priority=priority, sequence=self._sequence, handle=handle, call=call)
            self._jobs[handle.job_id] = JobState(handle=handle)
            self._job_done[handle.job_id] = threading.Event()
            if kind == "start_run":
                self._run_cancel = job.cancel
                # A run counts as in flight the moment it is queued, not when the compute
                # thread happens to pick it up. Otherwise there is a window in which the
                # caller has started a run, sees `running` as False, and an exclusive verb
                # slips in ahead of it -- or Stop, arriving in that window, does nothing.
                self._run_active = True
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
            self._current_job = job_id

        self._events.emit(EVENT_JOB_STARTED, {"kind": job.handle.kind}, job_id=job_id)
        state = self._jobs[job_id]
        try:
            if job.cancel.is_set():
                raise Cancelled("cancelled before it started")
            state.result = job.call(_JobContext(self, job))
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
                self._current_job = None
                if job.handle.kind == "start_run":
                    self._run_active = False
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

    def cancel_job(self, job_id: str) -> None:
        with self._state_lock:
            state = self._jobs.get(job_id)
        if state is not None and not state.done:
            # The job object holds the flag; find it by asking every queued job. Cheap:
            # the queue is short by construction (one run, a handful of reads).
            for job in list(self._queue.queue):
                if job.handle.job_id == job_id:
                    job.cancel.set()
                    return
            if self._current_job == job_id and self._run_cancel is not None:
                self._run_cancel.set()

    def cancel_run(self) -> None:
        """Ask the run to stop at the next row chunk. Must never block."""
        self._run_cancel.set()

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._run_active

    def close(self) -> None:
        self.cancel_run()
        self._closing.set()
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    # -- state helpers -----------------------------------------------------------------

    def _require_engine(self) -> AlignmentEngine:
        engine = self.engine
        if engine is None:
            raise NoEngine("Load a projection stack first.")
        return engine

    def _bump_pixels(self) -> None:
        with self._state_lock:
            self._pixel_epoch += 1

    def _rebuild_engine(self, *, sx=None, sy=None, center=None) -> None:
        """Make a fresh engine over the current preprocessed stack and bin factor.

        Bumps ``epoch``: everything a client cached about the previous engine -- timing
        calibration, slice caches, slider extents -- describes a grid that no longer
        exists.
        """
        if self.preprocessed is None:
            return
        binned = pp.bin_stack(np.asarray(self.preprocessed.data, dtype=np.float32), self._bin_factor)
        dataset = ProjectionData(
            data=binned,
            angles=self.preprocessed.angles,
            metadata=dict(self.preprocessed.metadata),
        )
        # The engine's own pad stays (0,0): padding happens in preprocessing so that what
        # the viewers show is exactly what the engine aligns.
        config = AlignConfig.from_dict({**self._config.to_dict(), "pad": (0, 0)})

        with self._state_lock:
            self.engine = AlignmentEngine(
                dataset=dataset, config=config, sx0=sx, sy0=sy, center=center
            )
            if self.com is not None:
                # A rebuild makes a new engine, but the COM survives it (rescaled to the
                # new grid), so carry the amplitude that bounds a plausible shift.
                self.engine.com_amplitude = self.com.amplitude
            self._epoch += 1
            self._pixel_epoch += 1

    # -- cheap verbs -------------------------------------------------------------------

    def set_config(self, config: dict | AlignConfig) -> None:
        """Stage a config change.

        Applied by the compute thread between iterations. Doing it immediately would let
        an algorithm change land between the reconstruct and the reproject of the same
        iteration -- see the module docstring.
        """
        parsed = config if isinstance(config, AlignConfig) else AlignConfig.from_dict(config)
        with self._state_lock:
            self._config = parsed
            if self.engine is None:
                return
            if self._run_active:
                self._pending_config = parsed
            else:
                self.engine.set_config(AlignConfig.from_dict({**parsed.to_dict(), "pad": (0, 0)}))

    def _apply_pending_config(self) -> None:
        with self._state_lock:
            pending, self._pending_config = self._pending_config, None
            engine = self.engine
        if pending is not None and engine is not None:
            engine.set_config(AlignConfig.from_dict({**pending.to_dict(), "pad": (0, 0)}))

    def set_center(self, value: float) -> None:
        with self._state_lock:
            engine = self._require_engine()
            engine.state.center = float(value)
            engine.invalidate_cache()
            self._pixel_epoch += 1
        logger.info("Centre set to %.3f px", value)
        self._emit_summary()

    def set_bin_factor_value(self, factor: int) -> None:
        with self._state_lock:
            self._bin_factor = int(factor)

    def telemetry(self):
        return self._monitor.sample()

    def list_hdf5(self, path: str) -> list:
        """Walk an HDF5 file. Takes its own lock so it never queues behind a run."""
        from tktomo.ptycho_align.core.dataset import list_hdf5_datasets

        with self._h5_lock:
            return list_hdf5_datasets(path)

    # -- pixels ------------------------------------------------------------------------

    def read_planes(self, refs: Sequence[PlaneRef]) -> tuple[np.ndarray | None, ...]:
        """The 2-D planes named by ``refs``, in the order asked for.

        Batched because the side-by-side display mode wants aligned, reprojection and
        difference for the same index at once, and over a wire three round trips per
        slider tick is what makes scrubbing feel broken.

        A difference plane is subtracted *here*, so one plane crosses rather than two.
        Same reasoning as the whole stack: the client has no use for the operands.
        """
        for ref in refs:
            if ref.key not in STACK_KEYS:
                raise KeyError(f"Unknown stack {ref.key!r}; expected one of {STACK_KEYS}")

        # One trip through the lock for the whole batch: taking a reference is
        # nanoseconds, but the difference mode would otherwise take it three times and
        # could straddle an `invalidate_cache` between operands.
        with self._state_lock:
            engine = self.engine
            if engine is None:
                return tuple(None for _ in refs)
            arrays = {
                STACK_RAW: engine.state.original,
                STACK_ALIGNED: engine.last_aligned,
                STACK_REPROJECTION: engine.last_simulated,
            }

        planes: list[np.ndarray | None] = []
        for ref in refs:
            if ref.key == STACK_DIFFERENCE:
                aligned = _plane(arrays[STACK_ALIGNED], ref.axis, ref.index)
                simulated = _plane(arrays[STACK_REPROJECTION], ref.axis, ref.index)
                planes.append(None if aligned is None or simulated is None else aligned - simulated)
            else:
                planes.append(_plane(arrays[ref.key], ref.axis, ref.index))
        return tuple(planes)

    def read_plane(self, key: str, axis: int, index: int) -> np.ndarray | None:
        """One plane. Sugar over :meth:`read_planes` for the single-image modes."""
        return self.read_planes([PlaneRef(key, axis, index)])[0]

    def read_volume_plane(
        self,
        axis: int,
        index: int,
        *,
        iteration: int | None = None,
        against: int | None = None,
    ) -> np.ndarray | None:
        """One plane of the volume, optionally minus the same plane of an earlier one.

        ``against`` exists so the tomogram view's "compare to iteration N" costs one
        plane rather than two whole volumes. The window used to fetch *every* retained
        volume just to populate that dropdown -- three times 511 MiB to offer three
        menu entries.
        """
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, not {axis!r}")

        # Both resolved inside the lock: VolumePolicy.apply nulls the volume on history
        # entries, so a reference taken across the boundary can go hollow underneath us.
        with self._state_lock:
            engine = self.engine
            if engine is None:
                return None
            volume = self._volume_locked(engine, iteration)
            reference = None if against is None else self._volume_locked(engine, against)

        plane = _plane(volume, axis, index)
        if plane is None or against is None:
            return plane
        other = _plane(reference, axis, index)
        if other is None or other.shape != plane.shape:
            return plane
        return plane - other

    def _volume_locked(self, engine, iteration: int | None) -> np.ndarray | None:
        """Resolve an iteration to its volume. Call with ``_state_lock`` held."""
        if iteration is None:
            return engine.state.volume
        for result in engine.state.history:
            if result.iteration == iteration:
                return result.volume
        return None

    # -- pixels: whole arrays ----------------------------------------------------------
    #
    # In-process only. Deliberately absent from the AlignmentSession protocol: a stack is
    # 104 MiB and a volume 511 MiB, and a client that can ask for one will, at which
    # point the remote session is unusable over anything slower than a backplane.
    #
    # They stay here because the core is meant to be scriptable from a notebook, and
    # `host.read_stack("Aligned")` is the honest way to get the aligned stack out when
    # you are already in the process that holds it. The exporters and the session writer
    # do not use them -- those run inside a job and go to `engine.state` directly.

    def read_stack(self, key: str) -> np.ndarray | None:
        with self._state_lock:
            engine = self.engine
            if engine is None:
                return None
            if key == STACK_RAW:
                return engine.state.original
            aligned = engine.last_aligned
            simulated = engine.last_simulated

        if key == STACK_ALIGNED:
            return aligned
        if key == STACK_REPROJECTION:
            return simulated
        if key == STACK_DIFFERENCE:
            if aligned is None or simulated is None:
                return None
            return aligned - simulated
        raise KeyError(f"Unknown stack {key!r}; expected one of {STACK_KEYS}")

    def read_volume(self, iteration: int | None = None) -> np.ndarray | None:
        with self._state_lock:
            engine = self.engine
            if engine is None:
                return None
            if iteration is None:
                return engine.state.volume
            # Resolve to the array *inside* the lock: VolumePolicy.apply nulls the volume
            # on history entries, so a reference to the IterationResult can go hollow
            # under a reader that held it across the boundary.
            for result in engine.state.history:
                if result.iteration == iteration:
                    return result.volume
            return None

    # -- summary -----------------------------------------------------------------------

    def summary(self, since_iteration: int = 0) -> SessionSummary:
        with self._state_lock:
            engine = self.engine
            epoch, pixel_epoch = self._epoch, self._pixel_epoch
            running, current_job = self._run_active, self._current_job
            bin_factor, com = self._bin_factor, self.com
            config = self._config.to_dict()
            source = dict(self._source) if self._source else None

            if engine is None:
                return SessionSummary(
                    epoch=epoch,
                    pixel_epoch=pixel_epoch,
                    seq=self._events.last_seq,
                    has_engine=False,
                    running=running,
                    iteration=0,
                    config=config,
                    bin_factor=bin_factor,
                    center=0.0,
                    angles=np.zeros(0),
                    sx=np.zeros(0),
                    sy=np.zeros(0),
                    original_shape=(0, 0, 0),
                    current_job=current_job,
                    metadata={"source": source} if source else {},
                    resources=self._monitor.sample(),
                )

            state = engine.state
            shape = tuple(int(n) for n in state.original.shape)
            history = [r for r in state.history if r.iteration > since_iteration]
            volume = state.volume
            has_aligned = engine.last_aligned is not None
            has_simulated = engine.last_simulated is not None
            volume_iterations = tuple(r.iteration for r in state.history if r.has_volume)
            policy = state.policy
            metadata = dict(state.metadata)
            com_amplitude = engine.com_amplitude
            center = float(state.center)
            sx, sy = state.sx.copy(), state.sy.copy()
            angles = state.angles
            raw_shape = (
                None if self.raw is None else tuple(int(n) for n in self.raw.data.shape)
            )

        stacks = (
            StackSpec(STACK_RAW, shape, pixel_epoch=pixel_epoch),
            StackSpec(STACK_ALIGNED, shape, available=has_aligned, pixel_epoch=pixel_epoch),
            StackSpec(
                STACK_REPROJECTION, shape, available=has_simulated, pixel_epoch=pixel_epoch
            ),
            StackSpec(
                STACK_DIFFERENCE,
                shape,
                available=has_aligned and has_simulated,
                pixel_epoch=pixel_epoch,
            ),
        )
        if source:
            metadata["source"] = source

        return SessionSummary(
            epoch=epoch,
            pixel_epoch=pixel_epoch,
            seq=self._events.last_seq,
            has_engine=True,
            running=running,
            iteration=state.iteration,
            config=config,
            bin_factor=bin_factor,
            center=center,
            angles=angles,
            sx=sx,
            sy=sy,
            original_shape=shape,  # type: ignore[arg-type]
            raw_shape=raw_shape,  # type: ignore[arg-type]
            stacks=stacks,
            history=tuple(IterationSummary.from_result(r) for r in history),
            history_from=since_iteration,
            volume_shape=None if volume is None else tuple(int(n) for n in volume.shape),
            volume_iterations=volume_iterations,
            volume_policy=(policy.keep_last, policy.keep_every),
            current_job=current_job,
            dataset=(
                None
                if self.preprocessed is None
                else DatasetSummary.from_projection_data(self.preprocessed)
            ),
            com=com,
            com_amplitude=com_amplitude,
            metadata=metadata,
            resources=self._monitor.sample(),
        )

    def _emit_summary(self) -> None:
        self._events.emit(EVENT_SUMMARY, {"summary": self.summary()})

    # -- preflight ---------------------------------------------------------------------

    def run_preflight(self, n: int) -> RunPreflight:
        """Price a run in one call: negatives, memory, and (if calibrated) wallclock.

        Bundled because the negative-data check is a full pass over the stack and the
        memory numbers depend on the engine's own shapes -- three separate questions
        would be three round trips before a run can start.
        """
        with self._state_lock:
            engine = self._require_engine()
            state = engine.state
            projections = state.original
            algorithm = engine.config.recon_algorithm
            policy = state.policy

        n_angles, rows, width = (int(v) for v in projections.shape)
        volume_bytes = policy.estimate_bytes((rows, width, width), n)
        # Three cached display stacks rather than four: the difference is computed per
        # displayed slice rather than materialised.
        stack_bytes = 3 * n_angles * rows * width * 4
        total = volume_bytes + stack_bytes

        return RunPreflight(
            footprint_bytes=int(total),
            footprint_text=(
                f"{format_bytes(total)} peak "
                f"({format_bytes(volume_bytes)} volumes + {format_bytes(stack_bytes)} stacks)"
            ),
            ram_available=available_ram_bytes(),
            negative_reason=algorithm_rejects_negatives(algorithm, projections),
            predicted_seconds=None,  # calibrated client-side from measured iterations
        )

    def cost_units(self) -> float:
        with self._state_lock:
            engine = self.engine
            if engine is None:
                return 0.0
            n_angles, rows, width = (int(v) for v in engine.state.original.shape)
            return iteration_cost_units(
                n_angles, rows, width, engine.config.recon_algorithm, engine.config.recon_inner_iters
            )

    # -- heavy verbs -------------------------------------------------------------------

    def open_dataset(self, path: str, load_kwargs: dict | None = None) -> JobHandle:
        kwargs = coerce_load_kwargs(load_kwargs or {})

        def call(ctx: _JobContext) -> DatasetSummary:
            def progress(done: int, total: int) -> bool:
                ctx.report(done / max(total, 1), f"reading projection {done} of {total}")
                return not ctx.is_set()

            with self._h5_lock:
                if "data_path" in kwargs:
                    # Only the explicitly-named-dataset path reads projection by
                    # projection, so only it can report progress or be interrupted. The
                    # layout-probing path reads the array in one call.
                    data = load_dataset(path, progress=progress, **kwargs)
                else:
                    ctx.report(0.0, "reading projections")
                    data = load_dataset(path, **kwargs)
                    ctx.report(1.0, "read projections")

            problems = inspect_dataset(data)
            with self._state_lock:
                self.raw = data
                self.preprocessed = data
                self.com = None
                self._source = {"path": path, "kwargs": dict(kwargs)}
            self._rebuild_engine()
            logger.info(
                "Loaded %s %s from %s", path, data.data.shape, data.metadata.get("data_path")
            )
            self._emit_summary()
            summary = DatasetSummary.from_projection_data(data)
            return {"dataset": summary, "problems": tuple(problems)}

        return self._submit("open_dataset", call)

    def apply_preprocessing(self, options, roi: tuple | None = None) -> JobHandle:
        def call(ctx: _JobContext) -> PreprocessReport:
            with self._state_lock:
                raw = self.raw
            if raw is None:
                raise NoEngine("Load a projection stack first.")

            array = np.asarray(raw.data, dtype=np.float32)
            mask = None
            if roi is not None:
                v0, v1, u0, u1 = roi
                mask = np.zeros(array.shape[1:], dtype=bool)
                mask[v0:v1, u0:u1] = True

            ctx.report(0.1, "preprocessing")
            if options.invert:
                array = pp.invert(array)
            if options.unwrap:
                array = pp.unwrap_phase(array)
            if options.remove_ramp:
                array = pp.remove_phase_ramp(array, mask=mask, border=options.border)
            if options.remove_offset:
                array = pp.remove_phase_offset(array, mask=mask, border=options.border)

            ok, total = pp.check_mass_positive(array)

            if options.pad_percent > 0:
                pad_v = int(round(array.shape[1] * options.pad_percent / 100.0))
                pad_u = int(round(array.shape[2] * options.pad_percent / 100.0))
                array = pp.pad(array, pad_u=pad_u, pad_v=pad_v)

            with self._state_lock:
                self.preprocessed = ProjectionData(
                    data=array, angles=raw.angles, metadata=dict(raw.metadata)
                )
                self.com = None
            self._rebuild_engine()
            logger.info("Preprocessed -> %s", array.shape)
            self._emit_summary()
            return PreprocessReport(
                dataset=DatasetSummary.from_projection_data(self.preprocessed),
                mass_is_positive=ok,
                mass_total=total,
            )

        return self._submit("apply_preprocessing", call)

    def reset_preprocessing(self) -> JobHandle:
        def call(ctx: _JobContext) -> DatasetSummary:
            with self._state_lock:
                if self.raw is None:
                    raise NoEngine("Load a projection stack first.")
                self.preprocessed = self.raw
                self.com = None
            self._rebuild_engine()
            self._emit_summary()
            return DatasetSummary.from_projection_data(self.preprocessed)

        return self._submit("reset_preprocessing", call)

    def set_bin_factor(self, factor: int) -> JobHandle:
        def call(ctx: _JobContext) -> SessionSummary:
            with self._state_lock:
                old = self._bin_factor
                if factor == old:
                    return self.summary()
                scale = old / float(factor)
                engine = self.engine
                sx = sy = None
                center = None
                if engine is not None:
                    sx = engine.state.sx * scale
                    sy = engine.state.sy * scale
                    center = engine.state.center * scale
                if self.com is not None:
                    self.com = self.com.scaled(scale)
                self._bin_factor = int(factor)

            self._rebuild_engine(sx=sx, sy=sy, center=center)
            logger.info("Bin factor %d -> %d", old, factor)
            self._emit_summary()
            return self.summary()

        return self._submit("set_bin_factor", call)

    def run_com(self, vertical_reference: str = "mean") -> JobHandle:
        """Centre-of-mass pre-alignment, and the state surgery that follows it.

        The six mutations below used to live in the window, which meant the GUI reached
        into ``engine.state`` and rewrote the shifts, the centre, the history and the
        volume by hand. They belong with the engine: they are what the COM result *means*,
        not a way of displaying it.
        """

        def call(ctx: _JobContext) -> ComResult:
            with self._state_lock:
                engine = self._require_engine()
                projections, angles = engine.state.original, engine.state.angles

            ctx.report(0.2, "computing centroids")
            result = com_prealign(projections, angles, vertical_reference=vertical_reference)

            with self._state_lock:
                self.com = result
                engine.com_amplitude = result.amplitude
                engine.state.sx = result.sx.copy()
                engine.state.sy = result.sy.copy()
                engine.state.center = result.center
                engine.state.history.clear()
                engine.state.volume = None
                engine.invalidate_cache()
                self._pixel_epoch += 1

            logger.info(
                "COM pre-alignment: centre %.3f px, fit residual %.3f px, amplitude %.2f px",
                result.center,
                result.fit_residual,
                result.amplitude,
            )
            self._emit_summary()
            return result

        return self._submit("run_com", call)

    def estimate_center(self, method: str) -> JobHandle:
        def call(ctx: _JobContext) -> float:
            with self._state_lock:
                engine = self._require_engine()
            ctx.report(0.2, f"finding the centre ({method})")
            center = find_center(engine.aligned_projections(), engine.state.angles, method=method)
            return float(center)

        return self._submit("estimate_center", call)

    def start_run(self, n: int) -> JobHandle:
        """Run ``n`` outer iterations, streaming each one back as it lands."""

        def call(ctx: _JobContext) -> int:
            engine = self._require_engine()
            self._events.emit(EVENT_RUN_STARTED, {"n": int(n)}, job_id=ctx.job_id)
            completed = 0
            try:
                for index in range(n):
                    if ctx.is_set():
                        break
                    self._apply_pending_config()

                    total = engine.iteration + n - index
                    label = f"Iteration {engine.iteration + 1} of {total}"
                    self._events.emit(
                        EVENT_ITERATION_STARTED,
                        {"iteration": engine.iteration + 1, "of": total},
                        job_id=ctx.job_id,
                    )

                    def report(fraction: float, message: str, index=index, label=label) -> None:
                        # Map the within-iteration fraction onto the whole run, so a
                        # single 20-minute iteration still shows steady movement.
                        ctx.report((index + fraction) / n, f"{label}: {message}...")

                    try:
                        result = engine.step(cancel=ctx, report=report)
                    except Cancelled:
                        # Stopped between row chunks. The engine records nothing on this
                        # path, so the state is exactly as it was before the iteration.
                        break

                    completed += 1
                    self._bump_pixels()
                    self._events.emit(
                        EVENT_ITERATION,
                        {
                            "result": IterationSummary.from_result(result),
                            "summary": self.summary(since_iteration=result.iteration - 1),
                        },
                        job_id=ctx.job_id,
                    )

                    if result.runaway:
                        # Stop here, on this thread. Waiting for a client to notice and
                        # call cancel is a race we lose: the next iteration would already
                        # be under way, warping the data by a larger bogus offset.
                        break
            finally:
                self._events.emit(
                    EVENT_RUN_FINISHED, {"completed": completed}, job_id=ctx.job_id
                )
            return completed

        return self._submit("start_run", call, priority=2)

    def revert(self, iteration: int) -> JobHandle:
        def call(ctx: _JobContext) -> SessionSummary:
            with self._state_lock:
                engine = self._require_engine()
                engine.state.revert_to(iteration)
                engine.invalidate_cache()
                self._pixel_epoch += 1
            logger.info("Reverted to iteration %d", iteration)
            self._emit_summary()
            return self.summary()

        return self._submit("revert", call)

    def materialize(self, keys: Sequence[str]) -> JobHandle:
        """Recompute the aligned stack (and reprojection) for display.

        Queued rather than exclusive: it only reads. But it is genuinely heavy --
        ``shift_images`` is a whole-stack order-5 spline warp, ~30 s on a 410x300x600
        stack -- so it does not belong on whatever thread is drawing.
        """

        def call(ctx: _JobContext) -> tuple[str, ...]:
            with self._state_lock:
                engine = self._require_engine()
            done = []
            if STACK_ALIGNED in keys and engine.last_aligned is None:
                ctx.report(0.2, "applying shifts")
                engine._last_aligned = engine.aligned_projections()  # noqa: SLF001
                done.append(STACK_ALIGNED)
            if STACK_REPROJECTION in keys and engine.last_simulated is None:
                with self._state_lock:
                    volume = engine.state.volume
                if volume is not None:
                    ctx.report(0.7, "reprojecting")
                    engine._last_simulated = engine.reproject()  # noqa: SLF001
                    done.append(STACK_REPROJECTION)
            self._bump_pixels()
            return tuple(done)

        return self._submit("materialize", call, priority=3)

    def save_session(self, path: str, *, include_arrays: bool = True) -> JobHandle:
        def call(ctx: _JobContext) -> str:
            from tktomo.ptycho_align.core import io as session_io

            with self._state_lock:
                engine = self._require_engine()
            with self._h5_lock:
                session_io.save_session(path, engine)
            logger.info("Session saved to %s", path)
            return path

        return self._submit("save_session", call, priority=3)

    def open_session(self, path: str) -> JobHandle:
        def call(ctx: _JobContext) -> SessionSummary:
            from tktomo.ptycho_align.core import io as session_io

            with self._h5_lock:
                engine = session_io.load_session(path)

            with self._state_lock:
                self.engine = engine
                self.preprocessed = ProjectionData(
                    data=engine.state.original,
                    angles=engine.state.angles,
                    metadata=dict(engine.state.metadata),
                )
                self.raw = self.preprocessed
                self.com = None
                self._source = None
                self._config = engine.config
                self._bin_factor = 1
                self._epoch += 1
                self._pixel_epoch += 1
            logger.info("Session restored from %s (iteration %d)", path, engine.iteration)
            self._emit_summary()
            return self.summary()

        return self._submit("open_session", call)

    def export(self, kind: str, path: str) -> JobHandle:
        def call(ctx: _JobContext) -> str:
            from tktomo.ptycho_align.core import io as session_io

            with self._state_lock:
                engine = self._require_engine()
                angles = engine.state.angles
                metadata = dict(engine.state.metadata)
                volume = engine.state.volume

            if kind == "projections":
                ctx.report(0.3, "applying shifts")
                aligned = engine.aligned_projections()
                session_io.export_projections(
                    path, ProjectionData(data=aligned, angles=angles, metadata=metadata)
                )
            elif kind == "volume":
                session_io.export_volume(path, volume, angles=angles)
            else:
                raise ValueError(f"Unknown export {kind!r}; expected 'projections' or 'volume'")
            logger.info("Exported %s to %s", kind, path)
            return path

        return self._submit("export", call, priority=3)

    def fetch_table(self, kind: str) -> bytes:
        """Shifts or convergence as CSV bytes -- kilobytes, so no job is warranted.

        Returns bytes rather than writing a file because the disk the user wants to save
        to is the one under the window, which is not where the engine is running.
        """
        from tktomo.ptycho_align.core import io as session_io

        with self._state_lock:
            engine = self._require_engine()
            angles = engine.state.angles
            sx, sy = engine.state.sx.copy(), engine.state.sy.copy()
            history = list(engine.state.history)

        if kind == "shifts":
            return session_io.shifts_csv(angles, sx, sy).encode()
        if kind == "convergence":
            return session_io.convergence_csv(history).encode()
        raise ValueError(f"Unknown table {kind!r}; expected 'shifts' or 'convergence'")
