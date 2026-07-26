"""What the viewers read pixels through.

A viewer wants one 2-D plane: the projection at the angle its slider is on, the sinogram
for the row its slider is on, a slice of the volume. It used to be handed the whole stack
and index into it, which is fine when the array is a pointer away and untenable when it
is on a cluster node -- 104 MiB for a stack, 511 MiB for a volume, per refresh.

So the viewers ask this object, and it asks the session for exactly the plane being
drawn. Two things make that affordable:

**A cache.** Scrubbing a slider revisits planes constantly -- back and forth over the
same few angles, and every mode switch re-asks for a plane the user just looked at. The
cache is bounded in *bytes* rather than entries because a plane is 6 KiB for a phantom
and 5.7 MiB for the real graphite dataset, and an entry count tuned for one is either
useless or ruinous for the other.

**Epoch keys.** ``pixel_epoch`` moves when the pixels change under a stack that keeps its
identity (an iteration lands, the centre moves, a cache is invalidated) and ``epoch``
moves when the engine is rebuilt entirely. Both are in the cache key, so a stale plane
cannot be served: the entry for the new epoch simply is not there. Nothing has to be
explicitly invalidated, which matters because the events that dirty pixels arrive on the
compute thread and the cache is read on the GUI thread.

Qt-free, so it can be tested without a display.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from tktomo.ptycho_align.session import PlaneRef, SessionSummary, StackSpec

__all__ = ["PlaneSource", "VOLUME"]

# The volume is not one of the STACK_* keys -- it has its own read verb, its own shape
# and its own compare mode -- but it shares the cache, so it needs a key here.
VOLUME = "__volume__"

# 64 MiB. Eleven planes of the real dataset, thousands of a phantom's; enough that a
# scrub over a working range of angles stays resident, small enough to be beneath notice
# next to the stacks themselves.
DEFAULT_BUDGET_BYTES = 64 * 1024 * 1024


class PlaneSource:
    """Fetches the displayed plane from a session, and remembers recent ones.

    The viewers hold one of these instead of holding stacks. It also answers the
    questions they used to answer by looking at ``stack.shape`` -- how many angles, how
    many rows, is this mode computed yet -- from the summary, so sizing a slider no
    longer requires the pixels to have arrived.
    """

    def __init__(self, session, *, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self._session = session
        self._budget = int(budget_bytes)
        self._cache: OrderedDict[tuple, np.ndarray | None] = OrderedDict()
        self._bytes = 0
        self._summary: SessionSummary | None = None
        self._specs: dict[str, StackSpec] = {}
        # Counted, not for correctness but because "did the slider actually stop
        # fetching" is otherwise unobservable from a test.
        self.hits = 0
        self.misses = 0

    # -- what the summary tells us -----------------------------------------------------

    def update(self, summary: SessionSummary) -> None:
        """Adopt a new summary. Planes cached under an older epoch become unreachable."""
        self._summary = summary
        self._specs = {spec.key: spec for spec in summary.stacks}

    @property
    def summary(self) -> SessionSummary | None:
        return self._summary

    def spec(self, key: str) -> StackSpec | None:
        return self._specs.get(key)

    def shape(self, key: str) -> tuple[int, int, int] | None:
        """The stack's ``(n_angles, n_v, n_u)``, known even before it is computed."""
        spec = self._specs.get(key)
        return None if spec is None else spec.shape

    def available(self, key: str) -> bool:
        spec = self._specs.get(key)
        return bool(spec and spec.available)

    def volume_shape(self) -> tuple[int, int, int] | None:
        return None if self._summary is None else self._summary.volume_shape

    def extent(self, key: str, axis: int) -> int:
        """How far the slider for ``axis`` may go. 0 when there is nothing to show."""
        shape = self.shape(key)
        return 0 if shape is None else int(shape[axis])

    # -- pixels ------------------------------------------------------------------------

    def plane(self, key: str, axis: int, index: int) -> np.ndarray | None:
        return self.planes([PlaneRef(key, axis, index)])[0]

    def planes(self, refs: list[PlaneRef]) -> tuple[np.ndarray | None, ...]:
        """Cached planes for ``refs``; whatever is missing goes to the session in one call.

        The single request for the misses is the point of the batch: side-by-side asks
        for three planes, and after a mode switch two of them are usually already here.
        """
        if self._summary is None:
            return tuple(None for _ in refs)

        found: dict[int, np.ndarray | None] = {}
        wanted: list[tuple[int, PlaneRef, tuple]] = []
        for position, ref in enumerate(refs):
            key = self._key(ref)
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                found[position] = self._cache[key]
            else:
                wanted.append((position, ref, key))

        if wanted:
            self.misses += len(wanted)
            fetched = self._session.read_planes([ref for _, ref, _ in wanted])
            for (position, _ref, key), plane in zip(wanted, fetched):
                # A plane that is not there yet is deliberately not cached: `available`
                # flips the moment materialise finishes, and a cached None under an
                # unchanged pixel_epoch would keep the viewer blank until the next
                # iteration moved it.
                if plane is not None:
                    self._store(key, plane)
                found[position] = plane

        return tuple(found[position] for position in range(len(refs)))

    def volume_plane(
        self, axis: int, index: int, *, against: int | None = None
    ) -> np.ndarray | None:
        """A plane of the current volume, or its difference against an earlier one."""
        if self._summary is None:
            return None

        key = (VOLUME, axis, index, against, self._summary.epoch, self._summary.pixel_epoch)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]

        self.misses += 1
        plane = self._session.read_volume_plane(axis, index, against=against)
        if plane is not None:
            self._store(key, plane)
        return plane

    # -- cache -------------------------------------------------------------------------

    def clear(self) -> None:
        self._cache.clear()
        self._bytes = 0

    @property
    def cached_bytes(self) -> int:
        return self._bytes

    def _key(self, ref: PlaneRef) -> tuple:
        summary = self._summary
        assert summary is not None  # guarded by every caller
        return (ref.key, ref.axis, ref.index, None, summary.epoch, summary.pixel_epoch)

    def _store(self, key: tuple, plane: np.ndarray) -> None:
        size = int(plane.nbytes)
        if size > self._budget:
            return  # one plane larger than the whole budget: serve it, don't hold it

        self._cache[key] = plane
        self._bytes += size
        while self._bytes > self._budget and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._bytes -= int(evicted.nbytes)
