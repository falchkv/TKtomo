"""The tracking wire format and the guards on the stack server."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tests.helpers_tracking import write_blob_file
from tktomo.tracking.autotrack import (
    AutoLabel,
    AutoTrackJob,
    AutoTrackParams,
    TrackResult,
)
from tktomo.tracking.recon import SliceRequest
from tktomo.tracking.stacksource import (
    AlignedExportRequest,
    StackInfo,
    ViewCache,
    ViewPrefetcher,
)

pytest.importorskip("msgpack")


def _display_tol(frame) -> float:
    """What a packed frame may differ by: half a level plus float32 rounding."""
    span = float(frame.max() - frame.min())
    peak = float(np.abs(frame).max())
    return span / (2 * 65535) + float(np.spacing(np.float32(peak)))


def _roundtrip(obj):
    from tktomo.ptycho_align.session.codec import decode, encode
    import tktomo.tracking.remote.types  # noqa: F401 - registers

    return decode(encode(obj))


def test_payloads_round_trip():
    info = StackInfo(shape=(3, 4, 5), angles=np.arange(3.0), kind="tiff",
                     binning=2, crop=(1, 2, 3, 4), view_origin=np.zeros((3, 2)),
                     path="/x", metadata={"crop": [1, 2], "n": 3}, epoch=7)
    back = _roundtrip(info)
    assert back.shape == (3, 4, 5) and isinstance(back.shape, tuple)
    assert back.crop == (1, 2, 3, 4)
    assert np.allclose(back.view_origin, 0) and back.metadata == info.metadata

    req = SliceRequest(row=5, lo=2, hi=9, sy=np.ones(3), sx=np.zeros(3),
                       rot_deg=np.zeros(3), center=2.5, extra_bin=2)
    back = _roundtrip(req)
    assert back.row_in_slab == 3 and back.extra_bin == 2

    job = AutoTrackJob(fid=1, seeds=((0, 1.0, 2.0), (5, 3.0, 4.0)),
                       params=AutoTrackParams(patch=30, fb_min_corr=0.2),
                       track_bin=2)
    back = _roundtrip(job)
    assert back.seeds == job.seeds and isinstance(back.seeds[0], tuple)
    assert back.params.patch == 30 and back.params.fb_min_corr == 0.2
    assert back.track_bin == 2

    result = TrackResult(labels=[AutoLabel(view=1, u=2.0, v=3.0, quality=0.5)],
                         seed_report=[{"view": 1, "ok": True}],
                         warnings=["w"], cancelled=False,
                         stats={"track_bin": 2, "largest_gap": (3, 9)},
                         outcomes={3: "low_p"})
    back = _roundtrip(result)
    assert back.labels[0].quality == 0.5 and back.seed_report == result.seed_report
    assert back.stats["track_bin"] == 2
    assert tuple(back.stats["largest_gap"]) == (3, 9)
    assert list(back.outcomes.values()) == ["low_p"]

    exp = AlignedExportRequest(dx=np.zeros(2), dy=np.zeros(2),
                               rot_deg=np.zeros(2), metadata={"a": 1})
    assert _roundtrip(exp).metadata == {"a": 1}


def test_frame_packet_round_trips_at_display_precision():
    from tktomo.tracking.remote.types import FramePacket, pack_frame

    rng = np.random.default_rng(0)
    # a narrow band on a large offset: what float16 would have destroyed
    frame = (100.0 + 0.5 * rng.random((64, 96))).astype(np.float32)
    packet = _roundtrip(pack_frame(frame))
    assert isinstance(packet, FramePacket) and packet.quantised
    back = packet.unpack()
    assert back.dtype == np.float32 and back.shape == frame.shape
    assert back.flags.writeable
    # half a level, plus what float32 cannot represent this far from zero
    span = float(frame.max() - frame.min())
    tol = span / (2 * 65535) + float(np.spacing(np.float32(100.5)))
    assert np.abs(back - frame).max() <= tol
    assert packet.data.nbytes < frame.nbytes / 1.9        # worth the trouble


def test_frame_packet_keeps_awkward_frames_verbatim():
    """NaN has no quantised representation, and exactness is on request."""
    from tktomo.tracking.remote.types import pack_frame

    frame = np.arange(12, dtype=np.float32).reshape(3, 4)
    frame[1, 1] = np.nan
    packet = pack_frame(frame)
    assert not packet.quantised
    assert np.array_equal(packet.unpack(), frame, equal_nan=True)

    exact = pack_frame(np.arange(12, dtype=np.float32).reshape(3, 4),
                       quantise=False, compress=False)
    assert exact.dtype == "float32" and exact.compression == "none"
    assert np.array_equal(exact.unpack(), np.arange(12).reshape(3, 4))


def test_frame_packet_survives_a_constant_frame():
    from tktomo.tracking.remote.types import pack_frame

    frame = np.full((8, 8), 3.5, np.float32)
    assert np.array_equal(pack_frame(frame).unpack(), frame)


def test_nostack_survives_the_wire():
    from tktomo.tracking.remote.types import NoStack

    back = _roundtrip(NoStack("open first"))
    assert isinstance(back, NoStack)


@pytest.fixture
def served(tmp_path):
    pytest.importorskip("zmq")
    from tktomo.tracking.remote import RemoteStackSource, make_server

    path, stack, theta, u, v = write_blob_file(tmp_path / "blob.h5")
    server = make_server("tcp://127.0.0.1:*").start()
    client = RemoteStackSource(server.endpoint, timeout=60.0)
    yield server, client, path, stack, u, v
    client.close()
    server.stop()


def test_whole_stack_verbs_do_not_exist(served):
    server, client, path, *_ = served
    client.open_stack(path)
    for verb in ("read_stack", "data", "slab"):
        with pytest.raises(KeyError):
            client._call(verb)


def test_pixel_verbs_refuse_before_open(served):
    from tktomo.tracking.remote.types import NoStack

    _, client, *_ = served
    with pytest.raises(NoStack):
        client._call("read_view", 0)


def test_read_view_served_while_autotrack_runs(served):
    """The ROUTER thread answers frame reads while the compute thread is busy."""
    from tktomo.tracking.autotrack import patch_size

    server, client, path, stack, u, v = served
    ok, why = client.autotrack_available()
    if not ok:
        pytest.skip(why)
    client.open_stack(path)
    seeds = tuple((w, float(u[w]), float(v[w])) for w in (5, 20, 35))
    job = AutoTrackJob(fid=0, seeds=seeds,
                       params=AutoTrackParams(patch=patch_size(10.0)))
    handle = client._call("autotrack", [job], 12.0)
    t0 = time.monotonic()
    frame = client.view(3)                     # must not wait for the job
    assert time.monotonic() - t0 < 2.0
    assert np.allclose(frame, stack[3], rtol=0, atol=_display_tol(stack[3]))
    result = client.wait(handle, timeout=120)
    assert result and len(result[0][1].labels) >= 20


def test_view_cache_hits_and_budget(served):
    _, client, path, stack, *_ = served
    client.open_stack(path)
    client.view(1)
    client.view(1)
    assert client.cache.misses == 1 and client.cache.hits == 1
    client.views([1, 2])
    assert client.cache.misses == 2                # only view 2 was fetched
    info = client.open_stack(path)                 # new epoch: cache cleared
    assert len(client.cache) == 0
    client.view(1)
    assert client.cache.get((info.epoch, 1)) is not None


def test_view_cache_evicts_by_bytes():
    cache = ViewCache(budget_bytes=3 * 16)
    for i in range(5):
        cache.put((0, i), np.zeros(4, np.float32))     # 16 bytes each
    assert len(cache) == 3
    assert cache.get((0, 0)) is None and cache.get((0, 4)) is not None


def test_frames_arrive_packed_and_exactly_when_asked(served, tmp_path):
    """The default is packed; --exact-frames is bit-exact. Both through one server."""
    from tktomo.tracking.remote import RemoteStackSource

    server, client, path, stack, *_ = served
    client.open_stack(path)
    assert client.packs_frames
    packed = client.view(4)
    tol = _display_tol(stack[4])
    assert not np.array_equal(packed, stack[4])          # it really is lossy
    assert np.abs(packed - stack[4]).max() <= tol

    exact = RemoteStackSource(server.endpoint, timeout=60.0, quantise=False)
    try:
        assert np.array_equal(exact.view(4), stack[4])
    finally:
        exact.close()


def test_client_falls_back_when_the_host_has_no_read_frames(tmp_path):
    """An old server still works: one KeyError, then the plain verb forever."""
    pytest.importorskip("zmq")
    from tktomo.ptycho_align.session.server import SessionServer
    from tktomo.tracking.remote import RemoteStackSource
    from tktomo.tracking.remote.host import TrackingHost
    from tktomo.tracking.remote.server import tracking_verbs

    path, stack, *_ = write_blob_file(tmp_path / "old.h5")
    host = TrackingHost()
    verbs = tracking_verbs(host)
    verbs.pop("read_frames")
    server = SessionServer(host, "tcp://127.0.0.1:*", verbs=verbs,
                           name="old-track-server").start()
    client = RemoteStackSource(server.endpoint, timeout=60.0)
    try:
        client.open_stack(path)
        assert np.array_equal(client.view(2), stack[2])
        assert client._packed is False
        assert np.array_equal(client.views([5, 6])[1], stack[6])
    finally:
        client.close()
        server.stop()


# --- reading ahead -------------------------------------------------------------------

class _SlowSource:
    """A source that counts fetches and can be made to fail, for the prefetcher."""

    def __init__(self, n_views=40, fail=False):
        self._info = StackInfo(shape=(n_views, 4, 4), angles=np.zeros(n_views),
                               kind="fake")
        self.asked: list[int] = []
        self.held: set[int] = set()
        self.fail = fail
        self.gate = threading.Event()
        self.gate.set()

    def info(self):
        return self._info

    def cached(self, index):
        return index in self.held

    def prefetch(self, index):
        self.gate.wait(5.0)
        self.asked.append(index)
        if self.fail:
            raise RuntimeError("no server")
        self.held.add(index)


def _settle(pf, predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_prefetch_follows_the_stride_and_the_direction():
    source = _SlowSource()
    pf = ViewPrefetcher(source, ahead=3)
    try:
        pf.want(10, step=5)                       # advancing five views at a time
        assert _settle(pf, lambda: len(source.asked) >= 4)
        assert sorted(source.asked[:4]) == [5, 15, 20, 25]
        assert source.asked[0] == 15              # nearest, ahead, first
        before = len(source.asked)
        pf.want(15, step=5)                       # 20 and 25 are already here
        assert _settle(pf, lambda: 30 in source.asked)
        # only the new view ahead, and the one behind, for a change of mind
        assert set(source.asked[before:]) == {30, 10}
    finally:
        pf.stop()
    assert not pf.running


def test_prefetch_drops_a_stale_plan_when_the_view_moves():
    source = _SlowSource()
    source.gate.clear()                            # hold the first fetch
    pf = ViewPrefetcher(source, ahead=3)
    try:
        pf.want(10, step=1)
        pf.want(30, step=1)                        # the user jumped meanwhile
        source.gate.set()
        assert _settle(pf, lambda: 31 in source.asked)
        assert not ({12, 13} & set(source.asked))  # never finished the old plan
    finally:
        pf.stop()


def test_prefetch_gives_up_after_failures_and_retries_only_when_moved():
    """A dead server costs one fetch per view change, then nothing at all."""
    source = _SlowSource(fail=True)
    pf = ViewPrefetcher(source, ahead=3, max_failures=3)
    for round_no in range(1, 4):
        pf.want(10 + round_no, step=1)
        assert _settle(pf, lambda n=round_no: pf.failures >= n)
        assert len(source.asked) == round_no        # one attempt, not a loop
    assert _settle(pf, lambda: not pf.running)
    assert pf.fetched == 0
    pf.stop()                                       # idempotent after it gave up


def test_prefetch_fills_the_client_cache(served):
    _, client, path, stack, *_ = served
    client.open_stack(path)
    pf = ViewPrefetcher(client, ahead=2)
    try:
        misses = client.cache.misses
        pf.want(4, step=1)
        assert _settle(pf, lambda: client.cached(5) and client.cached(6))
        assert np.allclose(client.view(5), stack[5], rtol=0,
                           atol=_display_tol(stack[5]))
        assert client.cache.misses == misses       # the wait was already paid
    finally:
        pf.stop()


def test_peek_does_not_disturb_the_hit_rate():
    cache = ViewCache()
    cache.put((0, 1), np.zeros(4, np.float32))
    assert cache.peek((0, 1)) and not cache.peek((0, 2))
    assert cache.hits == 0 and cache.misses == 0


def test_set_binning_is_a_job_that_shrinks_frames(served):
    server, client, path, stack, *_ = served
    info = client.open_stack(path)
    n, ny, nx = info.shape
    client.view(0)
    assert client.cached(0)
    info2 = client.set_binning(2)
    assert info2.rebin == 2 and info2.shape == (n, ny // 2, nx // 2)
    assert info2.epoch != info.epoch
    assert not client.cached(0)                   # the old grid's frame is gone
    frame = client.view(0)
    assert frame.shape == (ny // 2, nx // 2)
    expected = stack[0, :ny - ny % 2, :nx - nx % 2].reshape(
        ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    assert np.abs(frame - expected).max() <= _display_tol(expected)
    assert client.set_binning(1).shape == info.shape
