"""What an ``AlignmentSession`` must do, whoever implements it.

Written parameterised over a ``session`` fixture from the start. Today the only
parameter is the in-process implementation; when the remote one lands it joins the list
and every test here runs against both. That is the whole point: local and remote must
not merely have matching signatures, they must produce the same shifts, the same errors
and the same refusals.

These run without a GUI. Several also run without tomopy -- the ones that do need it say
so, because the suite should still be useful in the light environment the package's
layering exists to protect.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from make_phantom import make_misaligned_dataset  # noqa: E402

from tktomo.io import save_projections  # noqa: E402
from tktomo.ptycho_align.core.preprocess import PreprocessOptions  # noqa: E402
from tktomo.ptycho_align.session import (  # noqa: E402
    STACK_ALIGNED,
    STACK_DIFFERENCE,
    STACK_RAW,
    STACK_REPROJECTION,
    AlignmentSession,
    Busy,
    JobFailed,
    LocalSession,
    NoEngine,
    PlaneRef,
    RemoteSession,
    SessionServer,
)
from tktomo.ptycho_align.session.protocol import (  # noqa: E402
    EVENT_ITERATION,
    EVENT_PROGRESS,
    EVENT_RUN_FINISHED,
)

TIMEOUT = 300.0


@pytest.fixture(params=["local", "remote"])
def session(request):
    """One session per implementation. Every test below runs against both.

    The remote one goes over a real TCP socket on an ephemeral port, not a stub: the
    encoding, the threading and the polling are exactly what a cluster connection would
    use, because those are where local and remote can actually differ.
    """
    if request.param == "local":
        made = LocalSession()
        yield made
        made.close()
        return

    server = SessionServer(address="tcp://127.0.0.1:*").start()
    made = RemoteSession(server.endpoint)
    try:
        yield made
    finally:
        made.close()
        server.stop()


@pytest.fixture
def phantom_file(tmp_path):
    """A small misaligned stack on disk, plus the shifts that produced it."""
    data, sx_true, sy_true, _volume = make_misaligned_dataset(
        size=32, n_angles=24, max_shift=1.5, seed=3
    )
    path = tmp_path / "phantom.h5"
    save_projections(path, data)
    # The projections are wider than the phantom volume, so the shape is taken from the
    # data rather than assumed from `size`.
    return path, sx_true, sy_true, tuple(int(n) for n in data.data.shape)


def _open(session, path) -> None:
    session.wait(session.open_dataset(str(path)), timeout=TIMEOUT)


# -- shape of the interface ---------------------------------------------------------


def test_the_local_session_satisfies_the_protocol(session):
    assert isinstance(session, AlignmentSession)


def test_a_fresh_session_reports_no_engine(session):
    summary = session.summary()
    assert summary.has_engine is False
    assert summary.running is False
    assert summary.iteration == 0


def test_verbs_needing_data_fail_cleanly_before_a_dataset_is_open(session):
    """A clear refusal, not an AttributeError from somewhere inside the engine."""
    with pytest.raises(JobFailed) as excinfo:
        session.wait(session.run_com(), timeout=TIMEOUT)
    assert excinfo.value.exc_type == NoEngine.__name__


# -- loading ------------------------------------------------------------------------


def test_open_dataset_populates_the_summary(session, phantom_file):
    path, _, _, shape = phantom_file
    result = session.wait(session.open_dataset(str(path)), timeout=TIMEOUT)

    summary = session.summary()
    assert summary.has_engine
    assert summary.original_shape == shape
    assert summary.dataset is not None
    assert summary.dataset.shape == shape
    assert result["dataset"].shape == shape


def test_open_dataset_never_hands_back_the_pixels(session, phantom_file):
    """The summary is what the window draws from; a stack is 104 MiB on real data."""
    path, _, _, shape = phantom_file
    result = session.wait(session.open_dataset(str(path)), timeout=TIMEOUT)

    assert not isinstance(result["dataset"], np.ndarray)
    assert not hasattr(result["dataset"], "data")
    summary = session.summary()
    for value in (summary.dataset, *summary.stacks):
        assert not isinstance(value, np.ndarray)


def test_opening_a_dataset_bumps_the_epoch(session, phantom_file):
    """Anything the client cached about the previous engine is stale after this."""
    path, _, _, shape = phantom_file
    before = session.summary().epoch
    _open(session, path)
    assert session.summary().epoch > before


def test_list_hdf5_finds_the_datasets(session, phantom_file):
    path, _, _, shape = phantom_file
    entries = session.list_hdf5(str(path))
    assert any(e.is_stack for e in entries)


# -- preprocessing ------------------------------------------------------------------


def test_preprocessing_replaces_the_stack_and_reports_mass(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)
    before = session.summary().original_shape

    report = session.wait(
        session.apply_preprocessing(PreprocessOptions(pad_percent=10.0)), timeout=TIMEOUT
    )

    after = session.summary().original_shape
    assert after[1] > before[1] and after[2] > before[2]  # padded
    assert isinstance(report.mass_is_positive, bool)
    assert report.dataset.shape == after


def test_reset_preprocessing_restores_the_raw_shape(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)
    raw_shape = session.summary().original_shape

    session.wait(session.apply_preprocessing(PreprocessOptions(pad_percent=10.0)), timeout=TIMEOUT)
    assert session.summary().original_shape != raw_shape

    session.wait(session.reset_preprocessing(), timeout=TIMEOUT)
    assert session.summary().original_shape == raw_shape


# -- COM ----------------------------------------------------------------------------


def test_run_com_sets_the_shifts_and_centre(session, phantom_file):
    """The state surgery that used to live in the window, done where it belongs."""
    path, _, _, shape = phantom_file
    _open(session, path)

    result = session.wait(session.run_com(), timeout=TIMEOUT)
    summary = session.summary()

    assert summary.center == pytest.approx(result.center)
    assert np.allclose(summary.sx, result.sx)
    assert np.allclose(summary.sy, result.sy)
    assert summary.com_amplitude == pytest.approx(result.amplitude)
    assert summary.history == ()  # COM invalidates any iterations that preceded it


def test_run_com_clears_earlier_iterations(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(1), timeout=TIMEOUT)
    assert session.summary().iteration == 1

    session.wait(session.run_com(), timeout=TIMEOUT)
    assert session.summary().iteration == 0


# -- running ------------------------------------------------------------------------


def test_a_run_advances_the_iteration_and_emits_events(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    seen: list = []
    unsubscribe = session.subscribe(seen.append)
    try:
        completed = session.wait(session.start_run(2), timeout=TIMEOUT)
    finally:
        unsubscribe()

    assert completed == 2
    assert session.summary().iteration == 2

    kinds = [e.kind for e in seen]
    assert kinds.count(EVENT_ITERATION) == 2
    assert EVENT_PROGRESS in kinds
    assert EVENT_RUN_FINISHED in kinds


def test_iteration_events_carry_the_result_without_a_volume(session, phantom_file):
    """13 KiB per iteration instead of 511 MiB is what makes streaming these free."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    seen: list = []
    unsubscribe = session.subscribe(lambda e: seen.append(e) if e.kind == EVENT_ITERATION else None)
    try:
        session.wait(session.start_run(1), timeout=TIMEOUT)
    finally:
        unsubscribe()

    result = seen[0].payload["result"]
    assert result.iteration == 1
    assert not hasattr(result, "volume")
    assert result.sx.dtype == np.float64  # float32 here would corrupt the alignment


def test_progress_fractions_are_monotonic_within_a_run(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    fractions: list[float] = []
    unsubscribe = session.subscribe(
        lambda e: fractions.append(e.payload["fraction"]) if e.kind == EVENT_PROGRESS else None
    )
    try:
        session.wait(session.start_run(2), timeout=TIMEOUT)
    finally:
        unsubscribe()

    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)


def test_cancelling_a_run_records_nothing_for_the_abandoned_iteration(session, phantom_file):
    """The engine mutates only at the end of a step, and that must survive the session."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    handle = session.start_run(3)
    # Cancel as soon as the run is actually under way, so we interrupt mid-iteration
    # rather than before it starts.
    deadline = time.monotonic() + 30
    while not session.summary().running and time.monotonic() < deadline:
        time.sleep(0.01)
    session.cancel_run()
    completed = session.wait(handle, timeout=TIMEOUT)

    assert completed < 3
    assert session.summary().iteration == completed


def test_revert_rolls_the_history_back(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(2), timeout=TIMEOUT)

    session.wait(session.revert(1), timeout=TIMEOUT)
    assert session.summary().iteration == 1


def test_reverting_to_an_iteration_that_never_happened_is_a_clean_error(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    with pytest.raises(JobFailed) as excinfo:
        session.wait(session.revert(9), timeout=TIMEOUT)
    assert excinfo.value.exc_type == "ValueError"


# -- the Busy rule ------------------------------------------------------------------


def test_exclusive_verbs_are_refused_during_a_run(session, phantom_file):
    """Refused, not queued.

    Queuing a COM behind a 60-minute run means that an hour later the session wipes the
    history that run just built -- the user sees their results vanish for no reason they
    can connect to anything they did.
    """
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    handle = session.start_run(3)
    deadline = time.monotonic() + 30
    while not session.summary().running and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        with pytest.raises(Busy):
            session.run_com()
        with pytest.raises(Busy):
            session.set_bin_factor(2)
        with pytest.raises(Busy):
            session.revert(0)
    finally:
        session.cancel_run()
        session.wait(handle, timeout=TIMEOUT)


def test_read_only_verbs_are_allowed_during_a_run(session, phantom_file):
    """Reads must not be blocked by an hour of reconstruction."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)

    handle = session.start_run(3)
    deadline = time.monotonic() + 30
    while not session.summary().running and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        started = time.monotonic()
        summary = session.summary()
        raw = session.read_plane(STACK_RAW, 0, 0)
        elapsed = time.monotonic() - started

        assert summary.running is True
        assert raw is not None
        assert elapsed < 2.0, "a read waited on the compute thread"
    finally:
        session.cancel_run()
        session.wait(handle, timeout=TIMEOUT)


# -- stacks and volume --------------------------------------------------------------


def test_stacks_report_availability_before_they_are_computed(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)

    specs = {s.key: s for s in session.summary().stacks}
    assert specs[STACK_RAW].available is True
    assert specs[STACK_ALIGNED].available is False
    assert specs[STACK_REPROJECTION].available is False
    # Shape is known even when the pixels are not: the viewers size their sliders from it.
    assert specs[STACK_ALIGNED].shape == specs[STACK_RAW].shape


def test_the_session_never_offers_a_whole_stack(session):
    """The refusal is the feature.

    A stack is 104 MiB and a volume 511 MiB. A session that will hand one over is one a
    client will ask, and the remote implementation then has to either ship it or differ
    from the local one. Neither is acceptable, so the verb does not exist -- whole arrays
    live on EngineHost, reachable only where they already are.
    """
    assert not hasattr(session, "read_stack")
    assert not hasattr(session, "read_volume")


def test_the_axes_agree_with_each_other(session, phantom_file):
    """Axis 0 is a projection, axis 1 is a row's sinogram, axis 2 a column's.

    Checked by consistency rather than against the whole array, so this says the same
    thing about a session that has no local array to compare with. ``prj[a, v, u]`` has
    to be the same number whichever of the three planes it is read out of.
    """
    path, _, _, shape = phantom_file
    _open(session, path)
    angle, row, column = 2, 3, 4

    projection = session.read_plane(STACK_RAW, 0, angle)  # (v, u)
    sinogram = session.read_plane(STACK_RAW, 1, row)  # (angle, u)
    slab = session.read_plane(STACK_RAW, 2, column)  # (angle, v)

    assert projection.shape == (shape[1], shape[2])
    assert sinogram.shape == (shape[0], shape[2])
    assert slab.shape == (shape[0], shape[1])

    np.testing.assert_array_equal(projection[row], sinogram[angle])
    np.testing.assert_array_equal(projection[:, column], slab[angle])


def test_a_plane_is_the_slice_the_viewer_would_have_taken(phantom_file):
    """The local anchor: planes really are slices of the engine's own array.

    Local-only, because it needs the whole stack to compare against. The cross-axis test
    above is what carries the same convention to an implementation that has no such
    array within reach.
    """
    path, _, _, _shape = phantom_file
    session = LocalSession()
    try:
        _open(session, path)
        whole = session.host.read_stack(STACK_RAW)

        np.testing.assert_array_equal(session.read_plane(STACK_RAW, 0, 2), whole[2])
        np.testing.assert_array_equal(session.read_plane(STACK_RAW, 1, 3), whole[:, 3, :])
        np.testing.assert_array_equal(session.read_plane(STACK_RAW, 2, 4), whole[:, :, 4])
    finally:
        session.close()


def test_the_difference_plane_is_the_two_others_subtracted(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(1), timeout=TIMEOUT)

    aligned = session.read_plane(STACK_ALIGNED, 0, 1)
    simulated = session.read_plane(STACK_REPROJECTION, 0, 1)
    difference = session.read_plane(STACK_DIFFERENCE, 0, 1)

    assert aligned is not None and simulated is not None
    assert np.allclose(difference, aligned - simulated)


def test_planes_come_back_in_the_order_they_were_asked_for(session, phantom_file):
    """Side-by-side asks for three at once and lays them out positionally."""
    path, _, _, shape = phantom_file
    _open(session, path)

    refs = [PlaneRef(STACK_RAW, 0, 5), PlaneRef(STACK_RAW, 1, 0), PlaneRef(STACK_RAW, 0, 1)]
    batched = session.read_planes(refs)

    for ref, plane in zip(refs, batched):
        np.testing.assert_array_equal(plane, session.read_plane(ref.key, ref.axis, ref.index))


def test_a_plane_is_a_copy_not_a_window_onto_the_engine(session, phantom_file):
    """A viewer that scribbles on what it was handed must not corrupt the alignment."""
    path, _, _, shape = phantom_file
    _open(session, path)

    plane = session.read_plane(STACK_RAW, 0, 0)
    plane[:] = 12345.0

    assert not np.allclose(session.read_plane(STACK_RAW, 0, 0), 12345.0)


def test_a_plane_of_an_uncomputed_stack_is_none_rather_than_an_error(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)
    assert session.read_plane(STACK_ALIGNED, 0, 0) is None


def test_an_index_past_the_end_is_none(session, phantom_file):
    """A slider can still hold the last dataset's position when a smaller one opens."""
    path, _, _, shape = phantom_file
    _open(session, path)
    assert session.read_plane(STACK_RAW, 0, shape[0] + 50) is None
    assert session.read_plane(STACK_RAW, 0, -1) is None


def test_read_plane_rejects_an_unknown_key_and_a_bad_axis(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)
    with pytest.raises(KeyError):
        session.read_plane("Nonsense", 0, 0)
    with pytest.raises(ValueError):
        session.read_plane(STACK_RAW, 3, 0)


def test_volume_iterations_track_what_the_policy_kept(session, phantom_file):
    """`has_volume` cannot be derived from a stripped volume, so the summary states it."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(2), timeout=TIMEOUT)

    summary = session.summary()
    assert summary.volume_iterations
    for iteration in summary.volume_iterations:
        assert session.read_volume_plane(0, 0, iteration=iteration) is not None


def test_the_three_volume_planes_agree_with_each_other(session, phantom_file):
    """Axial / coronal / sagittal are axis 0 / 1 / 2, and they intersect consistently."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(1), timeout=TIMEOUT)

    z, y, x = (n // 2 for n in session.summary().volume_shape)
    axial = session.read_volume_plane(0, z)  # (y, x)
    coronal = session.read_volume_plane(1, y)  # (z, x)
    sagittal = session.read_volume_plane(2, x)  # (z, y)

    np.testing.assert_array_equal(axial[y], coronal[z])
    np.testing.assert_array_equal(axial[:, x], sagittal[z])
    np.testing.assert_array_equal(coronal[:, x], sagittal[:, y])


def test_comparing_against_an_earlier_iteration_subtracts_where_the_volumes_live(
    session, phantom_file
):
    """The tomogram view's compare mode costs one plane, not two volumes."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(2), timeout=TIMEOUT)

    earlier = min(session.summary().volume_iterations)
    current = session.read_volume_plane(0, 0)
    reference = session.read_volume_plane(0, 0, iteration=earlier)
    difference = session.read_volume_plane(0, 0, against=earlier)

    assert np.allclose(difference, current - reference)


def test_comparing_against_a_discarded_iteration_falls_back_to_the_plane(session, phantom_file):
    """The policy drops old volumes; the view must show the current slice, not nothing."""
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(1), timeout=TIMEOUT)

    plane = session.read_volume_plane(0, 0, against=9999)
    np.testing.assert_array_equal(plane, session.read_volume_plane(0, 0))


# -- config -------------------------------------------------------------------------


def test_set_config_round_trips_through_the_summary(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)

    config = dict(session.summary().config)
    config["recon_algorithm"] = "gridrec"
    config["upsample_factor"] = 33
    session.set_config(config)

    updated = session.summary().config
    assert updated["recon_algorithm"] == "gridrec"
    assert updated["upsample_factor"] == 33


def test_set_center_is_reflected_immediately(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)
    before = session.summary().pixel_epoch

    session.set_center(17.5)

    summary = session.summary()
    assert summary.center == pytest.approx(17.5)
    assert summary.pixel_epoch > before  # cached frames describe the old centre


# -- preflight, tables, history delta ------------------------------------------------


def test_run_preflight_prices_the_run(session, phantom_file):
    path, _, _, shape = phantom_file
    _open(session, path)

    preflight = session.run_preflight(5)
    assert preflight.footprint_bytes > 0
    assert "GB" in preflight.footprint_text


def test_history_is_delivered_as_a_delta(session, phantom_file):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(2), timeout=TIMEOUT)

    assert len(session.summary(since_iteration=0).history) == 2
    assert len(session.summary(since_iteration=1).history) == 1
    assert session.summary(since_iteration=1).history_from == 1


def test_tables_come_back_as_bytes(session, phantom_file):
    """The disk the user saves to is under the window, not under the engine."""
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.run_com(), timeout=TIMEOUT)

    shifts = session.fetch_table("shifts")
    assert shifts.startswith(b"angle_rad,angle_deg,sx,sy")
    assert len(shifts.splitlines()) == shape[0] + 1  # header + one row per angle

    assert session.fetch_table("convergence").startswith(b"iteration,shift_rms")
    with pytest.raises(ValueError):
        session.fetch_table("nonsense")


# -- events -------------------------------------------------------------------------


def test_events_can_be_replayed_from_a_sequence_number(session, phantom_file):
    """A client that dropped its connection asks for what it missed."""
    path, _, _, shape = phantom_file
    before = session.summary().seq
    _open(session, path)

    batch = session.poll_events(before)
    assert batch.events
    assert all(e.seq > before for e in batch.events)
    assert batch.last_seq >= batch.events[-1].seq
    assert batch.gap is False


def test_a_client_further_behind_than_the_ring_is_told_so(phantom_file):
    """Silently handing back a truncated history would desync the client for good."""
    from tktomo.ptycho_align.session.engine_host import EngineHost

    host = EngineHost(event_capacity=4)
    try:
        for index in range(10):
            host.events.emit("log", {"message": str(index)})
        assert host.events.since(0).gap is True
        assert host.events.since(host.events.last_seq - 1).gap is False
    finally:
        host.close()


def test_subscribers_are_not_called_under_the_state_lock(session, phantom_file):
    """A listener that calls back into the session must not deadlock the compute thread."""
    path, _, _, shape = phantom_file
    reentered: list = []

    def listener(event):
        if event.kind == EVENT_PROGRESS and not reentered:
            reentered.append(session.summary().running)

    unsubscribe = session.subscribe(listener)
    try:
        _open(session, path)
    finally:
        unsubscribe()

    assert reentered, "no progress event reached the listener"


# -- session files ------------------------------------------------------------------


def test_a_session_round_trips_through_a_file(session, phantom_file, tmp_path):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(1), timeout=TIMEOUT)
    before = session.summary()

    saved = tmp_path / "session.h5"
    session.wait(session.save_session(str(saved)), timeout=TIMEOUT)
    session.wait(session.open_session(str(saved)), timeout=TIMEOUT)

    after = session.summary()
    assert after.iteration == before.iteration
    assert np.allclose(after.sx, before.sx)
    assert np.allclose(after.sy, before.sy)
    assert after.center == pytest.approx(before.center)


def test_exports_are_written_where_the_engine_runs(session, phantom_file, tmp_path):
    pytest.importorskip("tomopy")
    path, _, _, shape = phantom_file
    _open(session, path)
    session.wait(session.start_run(1), timeout=TIMEOUT)

    target = tmp_path / "aligned.h5"
    session.wait(session.export("projections", str(target)), timeout=TIMEOUT)
    assert target.exists()

    with pytest.raises(JobFailed):
        session.wait(session.export("nonsense", str(tmp_path / "x.h5")), timeout=TIMEOUT)


# -- the compute thread --------------------------------------------------------------


def test_heavy_work_never_runs_on_the_calling_thread(session, phantom_file):
    """The whole design rests on this: tomopy segfaults if two threads touch it."""
    path, _, _, shape = phantom_file
    caller = threading.get_ident()
    threads: list[int] = []

    unsubscribe = session.subscribe(
        lambda e: threads.append(threading.get_ident()) if e.kind == EVENT_PROGRESS else None
    )
    try:
        _open(session, path)
    finally:
        unsubscribe()

    assert threads, "no progress event was emitted"
    assert all(ident != caller for ident in threads)
    assert len(set(threads)) == 1, "heavy work was spread across threads"
