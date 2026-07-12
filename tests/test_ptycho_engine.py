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


def test_alignment_config_can_change_mid_run(phantom):
    data, _sx, _sy = phantom
    engine = AlignmentEngine(dataset=data, config=AlignConfig(recon_inner_iters=1))
    engine.run(1)

    engine.set_config(AlignConfig(recon_inner_iters=1, align_horizontal=False))
    result = engine.step()

    assert result.config_changed  # the history plot marks this iteration
    np.testing.assert_array_equal(result.dsx, np.zeros_like(result.dsx))
    assert np.any(result.dsy != 0.0)
