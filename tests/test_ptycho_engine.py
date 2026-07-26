"""The definition of done for the core: the loop must recover known shifts.

Comparing recovered shifts against the truth needs care, because some shifts are
physically unobservable -- see :func:`observable_error`. If the registration's sign
convention is ever flipped, these tests fail immediately and loudly, which is the
whole point of them.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tomopy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from make_phantom import make_misaligned_dataset  # noqa: E402

from tktomo.ptycho_align.core import (  # noqa: E402
    AlignConfig,
    AlignmentEngine,
    apply_shifts,
    com_prealign,
)


def observable_error(
    sx: np.ndarray, sy: np.ndarray, sx_true: np.ndarray, sy_true: np.ndarray, angles: np.ndarray
) -> float:
    """RMS error between recovered and true shifts, ignoring unobservable modes.

    You cannot compare absolute shifts, only shifts modulo the transformations that
    leave the projections self-consistent:

    * **Vertical:** a constant ``sy`` translates the whole volume along the rotation
      axis. One degenerate mode: ``{1}``.
    * **Horizontal:** translating the object in-plane by ``(dx, dy)`` shifts
      projection ``i`` by ``dx*cos(theta_i) + dy*sin(theta_i)``, and a constant
      offset is absorbed by the rotation centre. Three degenerate modes:
      ``{sin, cos, 1}``.

    So we project both the recovered and the true shifts onto the *observable*
    complement and compare there. The build spec only removes the constant, which is
    right for the vertical axis but misses two of the three horizontal modes -- with
    only the mean removed, a perfectly correct alignment still scores ~0.2 px here.
    """
    horizontal_modes = np.column_stack([np.sin(angles), np.cos(angles), np.ones_like(angles)])
    vertical_modes = np.ones((len(angles), 1))

    def observable(vector: np.ndarray, modes: np.ndarray) -> np.ndarray:
        coefficients, *_ = np.linalg.lstsq(modes, vector, rcond=None)
        return vector - modes @ coefficients

    error_x = observable(sx, horizontal_modes) - observable(sx_true, horizontal_modes)
    error_y = observable(sy, vertical_modes) - observable(sy_true, vertical_modes)
    return float(np.sqrt(np.mean(error_x**2 + error_y**2)))


@pytest.fixture(scope="module")
def phantom():
    data, sx_true, sy_true, _volume = make_misaligned_dataset(
        size=48, n_angles=90, max_shift=2.0, margin=8, seed=1
    )
    return data, sx_true, sy_true


def test_apply_shifts_moves_the_image_by_the_requested_amount():
    """Pins TomoPy's argument order, whose own parameter names are misleading.

    ``shift_images(prj, sx, sy)`` really takes (rows, columns); we wrap it as
    ``apply_shifts(prj, sy, sx)``. Getting this backwards transposes every shift.
    """
    frame = np.zeros((32, 32), dtype=np.float32)
    frame[10, 20] = 1.0  # row 10, column 20

    shifted = apply_shifts(frame[np.newaxis].copy(), sy=np.array([3.0]), sx=np.array([0.0]))
    row, column = np.unravel_index(np.argmax(shifted[0]), shifted[0].shape)
    assert (row, column) == (7, 20), "sy must move rows (vertically)"

    shifted = apply_shifts(frame[np.newaxis].copy(), sy=np.array([0.0]), sx=np.array([-4.0]))
    row, column = np.unravel_index(np.argmax(shifted[0]), shifted[0].shape)
    assert (row, column) == (10, 24), "sx must move columns (horizontally)"


def test_apply_shifts_does_not_mutate_the_caller_array():
    """TomoPy's shift_images rescales its input in place; the engine must copy."""
    original = np.random.default_rng(0).random((3, 16, 16)).astype(np.float32)
    reference = original.copy()
    apply_shifts(original.copy(), np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(original, reference)


def test_com_prealign_beats_no_prealignment(phantom):
    data, sx_true, sy_true = phantom
    result = com_prealign(data.data, data.angles)

    nothing = observable_error(
        np.zeros_like(sx_true), np.zeros_like(sy_true), sx_true, sy_true, data.angles
    )
    prealigned = observable_error(result.sx, result.sy, sx_true, sy_true, data.angles)
    assert prealigned < nothing / 2, (
        f"COM pre-alignment must improve on doing nothing: {prealigned:.3f} vs {nothing:.3f} px"
    )


def test_engine_recovers_known_shifts(phantom):
    """THE GATE. 20 outer iterations must recover the shifts to better than 0.1 px."""
    data, sx_true, sy_true = phantom

    initial = com_prealign(data.data, data.angles)
    engine = AlignmentEngine(
        dataset=data,
        config=AlignConfig(
            recon_algorithm="sirt", recon_inner_iters=2, mode="joint", upsample_factor=50
        ),
        sx0=initial.sx,
        sy0=initial.sy,
        center=initial.center,
    )
    results = engine.run(20)

    assert len(results) == 20
    final = results[-1]
    error = observable_error(final.sx, final.sy, sx_true, sy_true, data.angles)

    assert error < 0.1, f"recovered shifts are {error:.4f} px RMS from truth (want < 0.1)"
    # The update size must decay -- a loop that keeps making large corrections is not
    # converging, it is hunting.
    assert final.error < results[0].error / 5
    assert final.residual < results[0].residual


def test_sequential_mode_also_converges(phantom):
    """align_seq-style: reconstruct from scratch each outer iteration."""
    data, sx_true, sy_true = phantom
    initial = com_prealign(data.data, data.angles)

    engine = AlignmentEngine(
        dataset=data,
        config=AlignConfig(
            recon_algorithm="sirt",
            recon_inner_iters=4,
            mode="sequential",
            upsample_factor=50,
        ),
        sx0=initial.sx,
        sy0=initial.sy,
        center=initial.center,
    )
    engine.run(10)
    error = observable_error(engine.state.sx, engine.state.sy, sx_true, sy_true, data.angles)
    assert error < 0.15


def test_engine_leaves_the_original_projections_untouched(phantom):
    data, _sx_true, _sy_true = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))
    reference = engine.state.original.copy()
    engine.step()
    np.testing.assert_array_equal(
        engine.state.original, reference, "the immutable original was modified in place"
    )


def test_step_is_resumable_and_matches_run(phantom):
    """Stepping N times must equal run(N) -- the GUI relies on this."""
    data, _sx, _sy = phantom
    config = AlignConfig(recon_inner_iters=1, upsample_factor=20)

    stepped = AlignmentEngine(dataset=data, config=config)
    for _ in range(3):
        stepped.step()

    ran = AlignmentEngine(dataset=data, config=config)
    ran.run(3)

    np.testing.assert_allclose(stepped.state.sx, ran.state.sx)
    np.testing.assert_allclose(stepped.state.sy, ran.state.sy)
    assert stepped.iteration == ran.iteration == 3


def test_run_honours_the_cancel_event(phantom):
    data, _sx, _sy = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))

    cancel = threading.Event()

    def stop_after_two(result):
        if result.iteration == 2:
            cancel.set()

    results = engine.run(10, cancel_event=cancel, callback=stop_after_two)

    # Cancelling stops before the next iteration, leaving a complete, valid state.
    assert len(results) == 2
    assert engine.iteration == 2
    assert engine.state.volume is not None


def test_revert_rolls_the_state_back(phantom):
    data, _sx, _sy = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))
    engine.run(3)

    sx_at_2 = engine.history[1].sx.copy()
    engine.state.revert_to(2)

    assert engine.iteration == 2
    np.testing.assert_allclose(engine.state.sx, sx_at_2)
    # And stepping on from there works.
    engine.step()
    assert engine.iteration == 3


def test_volume_memory_policy_drops_old_volumes(phantom):
    data, _sx, _sy = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))
    engine.policy.keep_last = 2
    engine.policy.keep_every = 0
    engine.state.policy = engine.policy

    engine.run(5)

    kept = [r.iteration for r in engine.history if r.has_volume]
    assert kept == [4, 5]
    # Shifts survive for every iteration, so revert always works.
    assert all(r.sx is not None for r in engine.history)


def test_emission_algorithms_are_rejected_on_negative_data(phantom):
    """mlem/osem diverge explosively on phase data instead of failing.

    Observed in a real session: switching to mlem drove the residual from 0.045 to 62
    and the shifts to 21 px within five iterations. Phase projections are ~20% negative
    after ramp/offset removal, and an emission algorithm's multiplicative update cannot
    cope with that.
    """
    from tktomo.ptycho_align.core import algorithm_rejects_negatives

    data, _sx, _sy = phantom
    negative = np.array([[[-1.0, 2.0]]], dtype=np.float32)
    positive = np.abs(negative)

    reason = algorithm_rejects_negatives("mlem", negative)
    assert reason is not None and "diverge" in reason

    assert algorithm_rejects_negatives("mlem", positive) is None
    assert algorithm_rejects_negatives("sirt", negative) is None
    # The real phantom is negative-valued, so the guard must fire on it.
    assert algorithm_rejects_negatives("mlem", data.data) is not None


def test_divergence_is_flagged(phantom):
    """A run whose residual climbs far above its own best must say so."""
    data, _sx, _sy = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))
    engine.run(2)

    assert not any(r.diverging for r in engine.history)

    # Force the detector's hand: a residual far above the best seen so far.
    best = min(r.residual for r in engine.history)
    assert engine._is_diverging(best * 10)
    assert engine._is_diverging(float("nan"))


def test_alignment_config_can_change_mid_run(phantom):
    data, _sx, _sy = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))
    engine.run(1)

    engine.set_config(AlignConfig(recon_inner_iters=1, align_horizontal=False))
    result = engine.step()

    assert result.config_changed  # the history plot marks this iteration
    np.testing.assert_array_equal(result.dsx, np.zeros_like(result.dsx))
    assert np.any(result.dsy != 0.0)


def test_shift_update_is_runaway_uses_the_com_amplitude_as_its_yardstick():
    from tktomo.ptycho_align.core import shift_update_is_runaway

    # The failure this exists for: registration matching noise because the volume is too
    # poor to reproject. Observed at 101 px RMS on a 1137 px detector whose object only
    # swings 16 px -- while the residual *fell*, so the divergence check stayed silent.
    assert shift_update_is_runaway(101.0, 1137, 16.25)
    assert "101.0 px" in shift_update_is_runaway(101.0, 1137, 16.25)

    # A normal correction is not flagged.
    assert shift_update_is_runaway(3.0, 1137, 16.25) is None

    # Without a COM the fallback is a fraction of the detector width (5% of 1137 = 57 px).
    assert shift_update_is_runaway(101.0, 1137) is not None
    assert shift_update_is_runaway(20.0, 1137) is None

    # A genuinely large first correction on a badly misaligned scan is NOT flagged: the
    # bound is the larger of the two yardsticks, so a big COM amplitude widens it.
    assert shift_update_is_runaway(101.0, 1137, 40.0) is None

    assert shift_update_is_runaway(float("nan"), 1137) is not None


def test_row_chunking_does_not_change_the_reconstruction():
    """Rows are independent in parallel-beam geometry -- chunking must be exact.

    This is the assumption the whole interruptible-Stop design rests on: if a chunked
    reconstruction differed from a whole one, we would have traded correctness for
    responsiveness.
    """
    from tktomo.ptycho_align.core import AlignConfig, AlignmentEngine

    data, _sx, _sy = make_misaligned_dataset(size=32, n_angles=24, max_shift=1.0, seed=5)[:3]

    def volume_with(ncore_hint: int) -> np.ndarray:
        engine = AlignmentEngine(
            dataset=data,
            config=AlignConfig(recon_algorithm="sirt", recon_inner_iters=2, ncore=ncore_hint),
        )
        return engine.step().volume

    # ncore=1 makes one row per chunk; a large ncore puts every row in a single chunk.
    fine = volume_with(1)
    whole = volume_with(1024)
    np.testing.assert_allclose(fine, whole, rtol=1e-5, atol=1e-6)


def test_row_chunk_size_never_starves_the_cores():
    from tktomo.ptycho_align.core import row_chunk_size

    # A chunk narrower than the core count would leave cores idle: tomopy parallelises
    # across slices, so over-fine chunking buys cancellation granularity with wallclock.
    # Measured: 8 rows/chunk costs +4%, 4 costs +47%, 1 costs +314%.
    assert row_chunk_size(100, 8) == 8
    assert row_chunk_size(100, 1) == 1
    # Never more rows than exist, and never zero.
    assert row_chunk_size(3, 8) == 3
    assert row_chunk_size(1, 8) == 1
    # An explicit request overrides, for a deliberate speed/granularity trade.
    assert row_chunk_size(100, 8, 16) == 16
    assert row_chunk_size(100, 8, 0) == 8  # 0 means auto


def test_the_reprojection_is_one_call_not_chunked(monkeypatch):
    """tomopy.project costs ~3.4 s per call regardless of size, so chunking it is
    almost pure overhead (32 slices: 5.8 s in one call, 16.0 s in four). It must stay a
    single call even though the reconstruction beside it is chunked."""
    from tktomo.ptycho_align.core import AlignConfig, AlignmentEngine

    data = make_misaligned_dataset(size=32, n_angles=24, max_shift=1.0, seed=8)[0]
    # ncore=1 forces the recon into one chunk per row -- the reprojection must not follow.
    engine = AlignmentEngine(dataset=data, config=AlignConfig(ncore=1))

    calls = []
    backend = engine._backend()
    original = backend.reproject

    def counting_reproject(volume, angles, **kwargs):
        calls.append(volume.shape[0])
        return original(volume, angles, **kwargs)

    monkeypatch.setattr(backend, "reproject", counting_reproject)
    engine.step()

    assert len(calls) == 1, f"the reprojection was split into {len(calls)} calls"
    assert calls[0] == engine.state.original.shape[1]  # the whole volume at once


def test_cancelling_mid_iteration_records_nothing():
    from tktomo.ptycho_align.core import AlignConfig, AlignmentEngine, Cancelled

    data = make_misaligned_dataset(size=32, n_angles=24, max_shift=1.0, seed=6)[0]
    engine = AlignmentEngine(dataset=data, config=AlignConfig(ncore=1))

    sx_before = engine.state.sx.copy()
    cancel = threading.Event()
    cancel.set()  # already cancelled: the very first chunk check must trip

    with pytest.raises(Cancelled):
        engine.step(cancel=cancel)

    # An abandoned iteration must leave the state pristine -- stopping is always safe.
    assert engine.iteration == 0
    assert engine.state.volume is None
    assert engine.history == []
    np.testing.assert_array_equal(engine.state.sx, sx_before)


def test_step_reports_progress_within_the_iteration():
    from tktomo.ptycho_align.core import AlignConfig, AlignmentEngine

    data = make_misaligned_dataset(size=32, n_angles=24, max_shift=1.0, seed=7)[0]
    engine = AlignmentEngine(dataset=data, config=AlignConfig(ncore=4))

    seen: list[tuple[float, str]] = []
    engine.step(report=lambda fraction, message: seen.append((fraction, message)))

    # A 20-minute iteration used to leave the progress bar frozen; it must now move, and
    # reach the reprojection phase. (The reprojection itself is one call -- chunking it is
    # nearly all overhead -- so it reports once rather than per chunk.)
    assert len(seen) > 1
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert any("reconstructing" in m for _, m in seen)
    assert any("reprojecting" in m for _, m in seen)


def test_align_config_survives_a_json_round_trip():
    """Every field, not just the ones a default config exercises.

    JSON has no tuple, so `pad` comes back as a list; rebuilding with `AlignConfig(**raw)`
    would silently leave it one. The engine then unpacks `pad_u, pad_v = self.config.pad`
    and pads the stack from a list, which happens to work -- until something compares two
    configs and finds them unequal, or the value goes back out over a wire.
    """
    import json

    from tktomo.ptycho_align.core import AlignConfig

    config = AlignConfig(
        recon_algorithm="mlem",
        recon_inner_iters=5,
        mode="sequential",
        upsample_factor=50,
        blur_edges=False,
        pad=(12, 34),
        refine_center=True,
        align_vertical=False,
        shift_damping=0.5,
        max_shift_per_iter=2.5,
        median_filter_shifts=True,
        ncore=8,
        row_chunk=4,
    )

    restored = AlignConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert isinstance(restored.pad, tuple)


def test_align_config_from_dict_keeps_none_fields_none():
    from tktomo.ptycho_align.core import AlignConfig

    config = AlignConfig(max_shift_per_iter=None, ncore=None)
    restored = AlignConfig.from_dict(config.to_dict())
    assert restored.max_shift_per_iter is None
    assert restored.ncore is None


def test_align_config_from_dict_rejects_unknown_fields():
    """A config written by a newer tktomo must fail loudly, not silently drop settings."""
    from tktomo.ptycho_align.core import AlignConfig

    with pytest.raises(ValueError, match="regularisation_weight"):
        AlignConfig.from_dict({**AlignConfig().to_dict(), "regularisation_weight": 0.1})


def test_align_config_from_dict_rejects_a_malformed_pad():
    from tktomo.ptycho_align.core import AlignConfig

    with pytest.raises(ValueError, match="two entries"):
        AlignConfig.from_dict({**AlignConfig().to_dict(), "pad": [1, 2, 3]})
