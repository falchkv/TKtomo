"""The client-side plane cache.

Driven against a stub session rather than a real one: what matters here is *how many
times the boundary is crossed*, which a stub can count and a real session cannot. The
tests that check the planes themselves are correct live in the conformance suite.

No Qt, no tomopy -- ``PlaneSource`` is deliberately free of both.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.ptycho_align.session import PlaneRef, SessionSummary, StackSpec
from tktomo.ptycho_align.ui.planes import PlaneSource

RAW = "Raw"
ALIGNED = "Aligned"
SHAPE = (8, 6, 5)


class StubSession:
    """Counts what it is asked for, and hands back a plane stamped with its index."""

    def __init__(self, *, shape=SHAPE, missing=()):
        self.shape = shape
        self.missing = set(missing)
        self.plane_calls: list[list[PlaneRef]] = []
        self.volume_calls: list[tuple] = []

    def read_planes(self, refs):
        refs = list(refs)
        self.plane_calls.append(refs)
        planes = []
        for ref in refs:
            if ref.key in self.missing:
                planes.append(None)
                continue
            rows = [n for axis, n in enumerate(self.shape) if axis != ref.axis]
            planes.append(np.full(rows, float(ref.index), dtype=np.float32))
        return tuple(planes)

    def read_plane(self, key, axis, index):
        return self.read_planes([PlaneRef(key, axis, index)])[0]

    def read_volume_plane(self, axis, index, *, iteration=None, against=None):
        self.volume_calls.append((axis, index, iteration, against))
        return np.full((4, 4), float(index), dtype=np.float32)

    @property
    def calls(self) -> int:
        return sum(len(batch) for batch in self.plane_calls)


def summary(*, epoch=1, pixel_epoch=1, available=True, volume_shape=(4, 4, 4)) -> SessionSummary:
    specs = tuple(
        StackSpec(key, SHAPE, available=available or key == RAW, pixel_epoch=pixel_epoch)
        for key in (RAW, ALIGNED)
    )
    return SessionSummary(
        epoch=epoch,
        pixel_epoch=pixel_epoch,
        seq=0,
        has_engine=True,
        running=False,
        iteration=0,
        config={},
        bin_factor=1,
        center=0.0,
        angles=np.zeros(SHAPE[0]),
        sx=np.zeros(SHAPE[0]),
        sy=np.zeros(SHAPE[0]),
        original_shape=SHAPE,
        stacks=specs,
        volume_shape=volume_shape,
    )


@pytest.fixture
def source():
    session = StubSession()
    made = PlaneSource(session)
    made.update(summary())
    return made, session


# -- shapes without pixels ------------------------------------------------------------


def test_slider_ranges_come_from_the_summary_not_from_an_array(source):
    """Sizing a slider must not require the stack to have been computed."""
    made, session = source
    assert made.extent(RAW, 0) == SHAPE[0]
    assert made.extent(RAW, 1) == SHAPE[1]
    assert made.shape(RAW) == SHAPE
    assert session.calls == 0


def test_availability_is_reported_before_anything_is_fetched(source):
    made, session = source
    made.update(summary(available=False))
    assert made.available(RAW) is True
    assert made.available(ALIGNED) is False
    assert session.calls == 0


def test_a_source_with_no_summary_yet_asks_for_nothing():
    session = StubSession()
    made = PlaneSource(session)
    assert made.plane(RAW, 0, 0) is None
    assert made.volume_plane(0, 0) is None
    assert session.calls == 0


# -- caching --------------------------------------------------------------------------


def test_the_same_plane_is_fetched_once(source):
    """Scrubbing back and forth over a few angles must not re-cross the boundary."""
    made, session = source
    for _ in range(5):
        made.plane(RAW, 0, 3)

    assert session.calls == 1
    assert made.hits == 4


def test_a_batch_only_asks_for_what_it_is_missing(source):
    """Side-by-side after a mode switch: two of the three are usually already here."""
    made, session = source
    made.plane(RAW, 0, 2)
    session.plane_calls.clear()

    made.planes([PlaneRef(RAW, 0, 2), PlaneRef(ALIGNED, 0, 2)])

    assert len(session.plane_calls) == 1, "the misses went in one round trip"
    assert [ref.key for ref in session.plane_calls[0]] == [ALIGNED]


def test_a_batch_keeps_the_order_it_was_asked_in_even_when_partly_cached(source):
    made, _ = source
    made.plane(RAW, 0, 7)

    first, second, third = made.planes(
        [PlaneRef(RAW, 0, 1), PlaneRef(RAW, 0, 7), PlaneRef(RAW, 0, 4)]
    )

    assert (first[0, 0], second[0, 0], third[0, 0]) == (1.0, 7.0, 4.0)


def test_new_pixels_are_not_served_from_the_old_cache(source):
    """An iteration lands: same stack, same index, different pixels."""
    made, session = source
    made.plane(ALIGNED, 0, 1)
    made.update(summary(pixel_epoch=2))
    made.plane(ALIGNED, 0, 1)

    assert session.calls == 2


def test_a_rebuilt_engine_invalidates_everything(source):
    """A new dataset reuses index 0; nothing cached about the old one may survive."""
    made, session = source
    made.plane(RAW, 0, 0)
    made.update(summary(epoch=2))
    made.plane(RAW, 0, 0)

    assert session.calls == 2


def test_a_plane_that_is_not_computed_yet_is_not_cached():
    """Otherwise the view stays blank until the *next* iteration moves the pixel epoch.

    `materialize` makes a stack available without changing what the pixels are, so the
    pixel epoch does not move -- a cached None would outlive the reason for it.
    """
    session = StubSession(missing={ALIGNED})
    made = PlaneSource(session)
    made.update(summary())

    assert made.plane(ALIGNED, 0, 0) is None
    session.missing.clear()
    assert made.plane(ALIGNED, 0, 0) is not None


# -- the budget -----------------------------------------------------------------------


def test_the_cache_is_bounded_in_bytes(source):
    """A plane is 6 KiB for a phantom and 5.7 MiB for the real dataset; count bytes."""
    made, session = source
    plane_bytes = made.plane(RAW, 0, 0).nbytes
    made._budget = plane_bytes * 3

    for index in range(SHAPE[0]):
        made.plane(RAW, 0, index)

    assert made.cached_bytes <= plane_bytes * 3


def test_eviction_is_least_recently_used(source):
    made, session = source
    plane_bytes = made.plane(RAW, 0, 0).nbytes
    made.clear()
    made._budget = plane_bytes * 2

    made.plane(RAW, 0, 0)
    made.plane(RAW, 0, 1)
    made.plane(RAW, 0, 0)  # refreshes 0, so 1 is now the oldest
    made.plane(RAW, 0, 2)  # evicts 1
    session.plane_calls.clear()

    made.plane(RAW, 0, 0)
    assert session.calls == 0, "the recently used plane was evicted"
    made.plane(RAW, 0, 1)
    assert session.calls == 1


def test_a_plane_bigger_than_the_whole_budget_is_still_served(source):
    """A 4k detector against a small budget must degrade to no caching, not to no image."""
    made, session = source
    made._budget = 1

    assert made.plane(RAW, 0, 0) is not None
    assert made.cached_bytes == 0


# -- the volume -----------------------------------------------------------------------


def test_volume_planes_are_cached_per_axis_and_index(source):
    made, session = source
    made.volume_plane(0, 2)
    made.volume_plane(0, 2)
    made.volume_plane(1, 2)

    assert len(session.volume_calls) == 2


def test_comparing_against_an_iteration_is_a_separate_cache_entry(source):
    """The difference plane is not the plane; caching them together would show the wrong one."""
    made, session = source
    made.volume_plane(0, 1)
    made.volume_plane(0, 1, against=3)

    assert len(session.volume_calls) == 2
    assert session.volume_calls[-1] == (0, 1, None, 3)


def test_the_volume_shape_comes_from_the_summary(source):
    made, session = source
    assert made.volume_shape() == (4, 4, 4)
    assert not session.volume_calls
