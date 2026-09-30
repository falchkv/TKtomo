"""The definition of done for joint gradient-descent alignment: recover known shifts.

Everything here runs on a synthetic phantom with numpy + scipy only -- no GPU, no
ASTRA, no TomoPy, no scikit-image, no beamtime data. The :class:`NumpyProjector3D`
fallback exists precisely so this file can exercise the real optimiser rather than a
mock of it.

Two deliberate choices, both worth keeping:

* The misalignment is injected with :func:`scipy.ndimage.fourier_shift`, a *different*
  implementation from the loop's :func:`scipy.ndimage.shift`. Recovering the truth then
  proves the loop works rather than merely that it agrees with itself. (Same reasoning
  as ``examples/make_phantom.py``.)
* Recovered shifts are compared through :func:`observable_error`, which projects out the
  four physically unobservable modes. Comparing raw shifts would fail a *correct*
  alignment, because the volume is free to translate.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.ptycho_align.core.joint_gd import (
    STAGES_REFINE,
    STAGES_STANDARD,
    FinalizedShifts,
    GDStage,
    JointGDAligner,
    JointGDConfig,
    JointGDDivergence,
    NumpyProjector3D,
    available_projectors,
    clean_shifts,
    make_projector,
    quality_weights,
    register_projector,
)

N_SLICES, N_COLS, N_ANGLES = 20, 48, 32
#: Fast stand-in for STAGES_STANDARD: same coarse-to-fine shape, sized for a 48 px
#: phantom (binning 16 would leave 3 px) and short enough for a unit test.
TEST_STAGES = (GDStage(4, 30, 1.0), GDStage(2, 45, 0.0), GDStage(1, 70, 0.0))
MAX_INJECTED_SHIFT = 2.5


# ------------------------------------------------------------------------------
# Phantom
# ------------------------------------------------------------------------------


def blob_volume(n_slices: int = N_SLICES, n_cols: int = N_COLS) -> np.ndarray:
    """An off-centre-blobbed ellipsoid, sized to leave a margin all round.

    The margin matters twice over: ``fourier_shift`` wraps, so the injected
    displacement must not carry the object across the frame edge, and the loop's
    ``mode="nearest"`` re-shift assumes the border is vacuum. The blobs matter because a
    centred, symmetric object has no horizontal information at all -- its projections
    are identical at every angle, so the horizontal shift is unidentifiable and the test
    would pass on an algorithm that does nothing.
    """
    z, y, x = np.mgrid[0:n_slices, 0:n_cols, 0:n_cols].astype(np.float32)
    zz, yy, xx = z / n_slices - 0.5, y / n_cols - 0.5, x / n_cols - 0.5
    volume = np.zeros((n_slices, n_cols, n_cols), dtype=np.float32)
    volume[(zz / 0.25) ** 2 + (yy / 0.28) ** 2 + (xx / 0.28) ** 2 < 1] = 1.0
    for (cz, cy, cx), radius, amplitude in (
        ((0.06, -0.12, -0.09), 0.09, 0.9),
        ((-0.07, 0.14, 0.11), 0.07, 1.3),
        ((0.10, 0.05, -0.15), 0.05, 0.7),
    ):
        volume[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2] += amplitude
    return volume


def fourier_translate(frame: np.ndarray, dy: float, dx: float) -> np.ndarray:
    from scipy.ndimage import fourier_shift

    return np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(frame), (dy, dx)))).astype(
        np.float32
    )


def observable_error(shifts: np.ndarray, truth: np.ndarray, angles: np.ndarray) -> float:
    """RMS error between recovered and true ``(dy, dx)``, ignoring unobservable modes.

    The objective ``||P v - T_s d||`` is blind to any shift pattern that a rigid motion
    of the volume can reproduce, because the optimiser is free to move ``v`` to match:

    * **vertical** -- a constant ``dy`` slides the volume along the rotation axis. One
      degenerate mode, ``{1}``.
    * **horizontal** -- translating the object in-plane by ``(a, b)`` shifts projection
      ``i`` by ``a*sin(theta_i) + b*cos(theta_i)``, and a constant is the rotation-axis
      position. Three degenerate modes, ``{sin, cos, 1}``.

    So both vectors are projected onto the observable complement before comparison.
    This is the same yardstick ``tests/test_ptycho_engine.py`` uses for the reprojection
    engine, which is what makes the two methods comparable at all.
    """
    horizontal = np.column_stack([np.sin(angles), np.cos(angles), np.ones_like(angles)])
    vertical = np.ones((len(angles), 1))

    def observable(vector: np.ndarray, modes: np.ndarray) -> np.ndarray:
        coefficients, *_ = np.linalg.lstsq(modes, vector, rcond=None)
        return vector - modes @ coefficients

    error_y = observable(shifts[:, 0], vertical) - observable(truth[:, 0], vertical)
    error_x = observable(shifts[:, 1], horizontal) - observable(truth[:, 1], horizontal)
    return float(np.sqrt(np.mean(error_y**2 + error_x**2)))


@pytest.fixture(scope="module")
def geometry():
    angles = np.linspace(0.0, np.pi, N_ANGLES, endpoint=False)
    projector = NumpyProjector3D(N_SLICES, N_COLS, angles)
    volume = blob_volume()
    # (n_slices, n_angles, n_cols) -> the measured-stack layout (n_angles, rows, cols).
    clean = np.ascontiguousarray(np.transpose(projector.forward(volume), (1, 0, 2)))
    return angles, projector, volume, clean


def misaligned(clean: np.ndarray, seed: int, amplitude: float = MAX_INJECTED_SHIFT):
    """Displace every projection by a known random ``(dy, dx)``.

    ``fourier_shift`` displaces in ``scipy.ndimage``'s sense, while the aligner reports
    in ``apply_shifts``' sense -- which is the *negative* (a feature at row 10 given
    ``sy=+3`` lands on row 7; ``tests/test_ptycho_engine.py`` pins this). So ``truth``
    is ``+displacement``, and getting that backwards is the single most likely way to
    write a test that passes on a sign-inverted aligner. It very nearly happened: the
    first draft of this port returned the negated answer and every accuracy test here
    passed, because the injection and the recovery were flipped together. The benchmark
    harness, which applies the answer with ``apply_shifts``, is what caught it.
    """
    rng = np.random.default_rng(seed)
    displacement = rng.uniform(-amplitude, amplitude, size=(clean.shape[0], 2))
    displacement -= np.median(displacement, axis=0)
    data = np.stack(
        [fourier_translate(clean[i], *displacement[i]) for i in range(clean.shape[0])]
    )
    return data, displacement


@pytest.fixture(scope="module")
def aligned(geometry):
    """One full three-stage run, shared by the tests that inspect its history."""
    angles, _projector, _volume, clean = geometry
    data, truth = misaligned(clean, seed=3)
    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(stages=TEST_STAGES, warmup_iters=8, projector="numpy"),
    )
    pristine = data.copy()
    aligner.run()
    return aligner, aligner.finalize(), truth, pristine


# ------------------------------------------------------------------------------
# The gate
# ------------------------------------------------------------------------------


def test_recovers_injected_shifts(aligned, geometry):
    """THE GATE. Known shifts must come back to better than 0.15 px RMS.

    Measured across seeds 1, 2, 3, 7 and 11 on this phantom: 0.012-0.048 px, from a
    1.8-2.1 px baseline. The tolerance is set several times above the worst observed so
    that a real regression trips it and interpolation noise does not.
    """
    angles = geometry[0]
    aligner, result, truth, _data = aligned

    error = observable_error(result.shifts, truth, angles)
    baseline = observable_error(np.zeros_like(truth), truth, angles)

    assert error < 0.15, f"recovered shifts are {error:.3f} px RMS from truth (want < 0.15)"
    assert error < baseline / 10, (
        f"alignment must beat doing nothing by an order of magnitude: "
        f"{error:.3f} vs {baseline:.3f} px"
    )
    assert aligner.done
    assert aligner.iteration == sum(s.iterations for s in TEST_STAGES)


def test_each_stage_reduces_its_loss(aligned):
    """Every stage must end below where it started, and the finest must be the best.

    A stage that ends higher than it began is the divergence the damping and the cap
    exist to prevent; a finest stage no better than the coarse one means the
    multi-resolution ladder is not buying anything.
    """
    aligner = aligned[0]
    finals = []
    for stage in range(len(TEST_STAGES)):
        history = [r for r in aligner.history if r.stage == stage]
        assert history[-1].loss < history[0].loss / 10, (
            f"stage {stage} went {history[0].loss:.3e} -> {history[-1].loss:.3e}"
        )
        finals.append(history[-1].loss)
    assert finals == sorted(finals, reverse=True), (
        f"each finer stage should end below the previous one, got {finals}"
    )


def test_no_shift_update_exceeds_the_cap(aligned):
    """``shift_cap_px`` bounds every per-iteration step, in that stage's binned px.

    Load-bearing: the Gauss-Newton denominator ``<grad, grad>`` collapses on a flat
    projection, and without the clip the step there is unbounded. Checked from the
    recorded history rather than from the internals, so it also pins that the reported
    shifts really are in full-resolution px (convention 2).
    """
    aligner = aligned[0]
    cap = aligner.config.shift_cap_px
    previous: dict[int, np.ndarray] = {}
    for result in aligner.history:
        last = previous.get(result.stage)
        previous[result.stage] = result.shifts
        if last is None:
            continue
        step = np.abs(result.shifts - last) / result.binning
        assert step.max() <= cap + 1e-6, (
            f"iteration {result.iteration} moved a projection {step.max():.4f} binned px, "
            f"past the {cap} px cap"
        )


def test_shifts_are_frozen_during_the_warmup(aligned):
    """No shift may move before the volume has formed -- at the start of *every* stage."""
    aligner = aligned[0]
    warmup = aligner.config.warmup_iters
    for stage in range(len(TEST_STAGES)):
        history = [r for r in aligner.history if r.stage == stage]
        for result in history[:warmup]:
            assert not result.shifts_engaged
            assert result.update_rms == 0.0
        assert history[warmup].shifts_engaged


def test_the_input_stack_is_never_modified(aligned):
    """Convention 3: the loop re-shifts the pristine stack, it does not warp in place.

    ``pristine`` is a copy taken before the run, so this compares against bytes the
    aligner has never seen. Re-warping already-warped data compounds the interpolation
    and blurs the stack away over a few hundred iterations -- and it would be invisible
    to every other test here, because the shifts would still come out right.
    """
    aligner, result, _truth, pristine = aligned
    np.testing.assert_array_equal(aligner.projections, pristine)
    # And finalize() must be idempotent -- nothing is consumed by reading the answer.
    np.testing.assert_allclose(aligner.finalize().shifts, result.shifts)


def test_aligned_projections_undo_the_injected_displacement(aligned, geometry):
    """End to end: applying the answer must bring the data back onto the clean stack."""
    _angles, _projector, _volume, clean = geometry
    aligner, result, _truth, data = aligned

    def mismatch(stack: np.ndarray) -> float:
        # Compare interiors only: mode="nearest" smears the outermost rows/columns.
        a, b = stack[:, 4:-4, 6:-6], clean[:, 4:-4, 6:-6]
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    assert mismatch(aligner.aligned_projections(result.shifts)) < mismatch(data) / 5


# ------------------------------------------------------------------------------
# THE sign anchor
# ------------------------------------------------------------------------------


def test_a_positive_shift_moves_rows_up_like_apply_shifts():
    """The one test that cannot pass on a sign-inverted aligner.

    Every accuracy test in this file compares a recovered shift against an injected one,
    so all of them pass if the injection and the recovery are flipped *together* -- and
    in the first draft of this port they were. What pins the sign is an external
    reference, and TKtomo's is ``engine.apply_shifts``, whose behaviour
    ``tests/test_ptycho_engine.py`` fixes: a feature at row 10 given ``sy=+3`` ends up
    at row **7**, and one at column 20 given ``sx=-4`` ends up at column 24.

    That is the *opposite* of ``scipy.ndimage.shift``'s sign, and the opposite of the
    P06 script this module was ported from. Asserting it here, with the same numbers as
    the engine's test, is what stops the two halves of the alignment stack from drifting
    apart: a benchmark that runs both methods and applies their answers the same way
    would otherwise score one of them as catastrophically wrong while it was in fact
    converging perfectly.
    """
    frame = np.zeros((1, 32, 32), dtype=np.float32)
    frame[0, 10, 20] = 1.0
    aligner = JointGDAligner(
        np.repeat(frame, 8, axis=0),
        np.linspace(0.0, np.pi, 8, endpoint=False),
        JointGDConfig(stages=(GDStage(4, 1, 0.0),), projector="numpy"),
    )

    moved = aligner.aligned_projections(np.tile([3.0, 0.0], (8, 1)))
    assert np.unravel_index(np.argmax(moved[0]), moved[0].shape) == (7, 20), (
        "sy=+3 must move rows UP, matching apply_shifts -- not down, which is "
        "scipy.ndimage.shift's convention and the original script's"
    )

    moved = aligner.aligned_projections(np.tile([0.0, -4.0], (8, 1)))
    assert np.unravel_index(np.argmax(moved[0]), moved[0].shape) == (10, 24)


def test_the_sign_matches_apply_shifts_exactly_when_tomopy_is_present():
    """Cross-check against the real thing, when the optional dependency is installed."""
    pytest.importorskip("tomopy")
    from tktomo.ptycho_align.core.engine import apply_shifts

    rng = np.random.default_rng(0)
    stack = rng.random((6, 24, 24)).astype(np.float32)
    shifts = rng.uniform(-2.0, 2.0, size=(6, 2))

    aligner = JointGDAligner(
        stack,
        np.linspace(0.0, np.pi, 6, endpoint=False),
        JointGDConfig(stages=(GDStage(4, 1, 0.0),), projector="numpy"),
    )
    ours = aligner.aligned_projections(shifts)
    theirs = apply_shifts(stack.copy(), shifts[:, 0], shifts[:, 1])

    # Different interpolators (linear vs TomoPy's spline warp), so compare loosely --
    # a sign error would show up as a mismatch many times this size.
    interior = (slice(None), slice(4, -4), slice(4, -4))
    assert np.abs(ours[interior] - theirs[interior]).mean() < 0.05


# ------------------------------------------------------------------------------
# Robustness: the MAD outlier rule and the failure modes it exists for
# ------------------------------------------------------------------------------


def test_a_degraded_projection_slides_out_of_frame_and_is_caught(geometry):
    """The real failure mode, reproduced: a ramp-dominated projection escapes.

    A projection whose ptychographic phase retrieval failed carries a strong residual
    ramp and little structure. Its image gradient is then almost constant, so the
    Gauss-Newton step points the same way every iteration and the projection walks out
    of the frame -- it does not stall, and nothing in the loss says anything is wrong.
    20 of 918 did this on the first real run. Here projection 7 reaches ~47 px on a
    48 px detector while every healthy one stays under 4 px, and the MAD rule catches it.
    """
    angles, _projector, _volume, clean = geometry
    data, _truth = misaligned(clean, seed=3)
    rows = np.arange(N_SLICES, dtype=np.float32)[:, None]
    cols = np.arange(N_COLS, dtype=np.float32)[None, :]
    data = data.copy()
    data[7] = 0.15 * data[7] + 0.9 * cols + 0.4 * rows

    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(
            stages=(GDStage(4, 30, 1.0), GDStage(2, 45, 0.0)),
            warmup_iters=8,
            projector="numpy",
        ),
    )
    aligner.run()
    result = aligner.finalize()

    assert result.outliers[7], (
        f"the degraded projection reached {result.raw[7].round(1)} px and must be "
        f"flagged; MAD was {result.mad.round(2)}"
    )
    assert result.n_outliers == 1, f"healthy projections must survive: {result.n_outliers}"
    np.testing.assert_allclose(result.shifts[7], [0.0, 0.0])


def test_clean_shifts_removes_the_degenerate_median_offset():
    """A constant added to every shift is unobservable and must not survive."""
    rng = np.random.default_rng(0)
    shifts = rng.normal(0.0, 1.0, size=(40, 2))
    offset = np.array([7.5, -3.25])

    plain = clean_shifts(shifts)
    offsetted = clean_shifts(shifts + offset)

    np.testing.assert_allclose(offsetted.shifts, plain.shifts, atol=1e-12)
    np.testing.assert_allclose(offsetted.median_offset - plain.median_offset, offset)


def test_clean_shifts_keeps_the_offset_when_centring_is_off():
    shifts = np.tile([4.0, -2.0], (12, 1))
    result = clean_shifts(shifts, median_center=False)
    np.testing.assert_allclose(result.shifts, shifts)


def test_clean_shifts_rejects_a_runaway_and_leaves_the_rest_alone():
    rng = np.random.default_rng(1)
    shifts = rng.normal(0.0, 1.0, size=(60, 2))
    shifts[13] = [0.0, 240.0]

    result = clean_shifts(shifts)

    assert result.outliers[13]
    assert result.n_outliers == 1
    np.testing.assert_allclose(result.shifts[13], [0.0, 0.0])
    # The mean would have been dragged by 4 px; the median must not be.
    assert abs(result.median_offset[1]) < 0.5


def test_clean_shifts_honours_a_per_projection_fallback():
    shifts = np.zeros((10, 2))
    shifts[4] = [0.0, 500.0]
    fallback = np.full((10, 2), 1.5)

    result = clean_shifts(shifts, fallback=fallback)

    assert result.outliers[4]
    np.testing.assert_allclose(result.shifts[4], [1.5, 1.5])


def test_clean_shifts_refuses_a_diverged_solution():
    shifts = np.zeros((5, 2))
    shifts[2, 1] = np.nan
    with pytest.raises(JointGDDivergence, match="non-finite"):
        clean_shifts(shifts)


def test_finalize_can_revert_outliers_to_the_initial_estimate(geometry):
    """``outlier_fallback='initial'`` keeps a supplied pre-alignment for bad frames."""
    angles, _projector, _volume, clean = geometry
    data, _truth = misaligned(clean, seed=5)
    initial = np.zeros((N_ANGLES, 2))
    initial[:, 1] = 1.25

    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(
            stages=(GDStage(4, 12, 1.0),),
            warmup_iters=4,
            projector="numpy",
            outlier_fallback="initial",
        ),
        initial_shifts=initial,
    )
    aligner.run()
    # Force one obvious outlier rather than waiting for the physics to produce one.
    aligner._shifts[6] = [0.0, 400.0 / aligner.binning]
    result = aligner.finalize()

    assert result.outliers[6]
    np.testing.assert_allclose(result.shifts[6], initial[6] - result.median_offset)


# ------------------------------------------------------------------------------
# Direction toggles -- what the roadmap wants this loop used for
# ------------------------------------------------------------------------------


def test_horizontal_only_leaves_the_vertical_shifts_untouched(geometry):
    """Vertical is decoupled and cheaper to solve elsewhere; this loop can skip it."""
    angles, _projector, _volume, clean = geometry
    data, _truth = misaligned(clean, seed=4)
    initial = np.zeros((N_ANGLES, 2))
    initial[:, 0] = np.linspace(-1.0, 1.0, N_ANGLES)  # a pre-solved vertical

    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(
            stages=(GDStage(2, 25, 0.0),),
            warmup_iters=5,
            projector="numpy",
            align_vertical=False,
        ),
        initial_shifts=initial,
    )
    aligner.run()

    np.testing.assert_allclose(aligner.shifts[:, 0], initial[:, 0], atol=1e-9)
    assert np.abs(aligner.shifts[:, 1]).max() > 0.1, "horizontal must still move"


def test_solving_neither_direction_is_rejected():
    with pytest.raises(ValueError, match="nothing"):
        JointGDConfig(align_vertical=False, align_horizontal=False)


# ------------------------------------------------------------------------------
# Multi-resolution bookkeeping
# ------------------------------------------------------------------------------


def test_shifts_are_continuous_in_full_resolution_px_across_a_stage_boundary(geometry):
    """Convention 2: rescaling at a stage boundary must be invisible from outside.

    The internal shift array is divided by the new binning factor; if the public
    accessor did not multiply it back, the answer would silently jump by 2x or 4x at
    every boundary -- big enough to look like a real misalignment, small enough to
    survive review.
    """
    angles, _projector, _volume, clean = geometry
    data, _truth = misaligned(clean, seed=6)
    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(
            stages=(GDStage(4, 20, 1.0), GDStage(2, 20, 0.0)),
            warmup_iters=5,
            projector="numpy",
        ),
    )
    aligner.run(20)
    before = aligner.shifts.copy()
    assert aligner.binning == 4
    aligner.step()  # first iteration of stage 1: rebinning happens here
    assert aligner.binning == 2
    # Stage 1 opens with a warm-up, so the only change can be the rescaling itself.
    np.testing.assert_allclose(aligner.shifts, before, atol=1e-9)


def test_initial_shifts_are_carried_into_the_first_stage(geometry):
    angles, _projector, _volume, clean = geometry
    data, _truth = misaligned(clean, seed=8)
    initial = np.column_stack(
        [np.linspace(-2.0, 2.0, N_ANGLES), np.linspace(1.0, -1.0, N_ANGLES)]
    )
    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(stages=(GDStage(4, 6, 1.0),), warmup_iters=6, projector="numpy"),
        initial_shifts=initial,
    )
    aligner.run()
    np.testing.assert_allclose(aligner.shifts, initial, atol=1e-9)


def test_a_stage_that_bins_the_data_away_is_refused(geometry):
    """A 48 px phantom cannot use the standard 16x schedule, and must say so."""
    angles, _projector, _volume, clean = geometry
    aligner = JointGDAligner(
        clean, angles, JointGDConfig(stages=STAGES_STANDARD, projector="numpy")
    )
    with pytest.raises(ValueError, match="too small to align"):
        aligner.step()


# ------------------------------------------------------------------------------
# Honest failure
# ------------------------------------------------------------------------------


def test_a_runaway_shift_raises_rather_than_returning_nonsense(geometry):
    angles, _projector, _volume, clean = geometry
    data, _truth = misaligned(clean, seed=9)
    aligner = JointGDAligner(
        data,
        angles,
        JointGDConfig(
            stages=(GDStage(2, 60, 0.0),),
            warmup_iters=3,
            projector="numpy",
            runaway_shift_px=0.5,  # absurdly tight, so the guard must fire
        ),
    )
    with pytest.raises(JointGDDivergence, match="runaway_shift_px"):
        aligner.run()


def test_stepping_past_the_schedule_is_an_error(geometry):
    angles, _projector, _volume, clean = geometry
    aligner = JointGDAligner(
        clean,
        angles,
        JointGDConfig(stages=(GDStage(4, 3, 0.0),), warmup_iters=1, projector="numpy"),
    )
    aligner.run()
    assert aligner.done
    with pytest.raises(RuntimeError, match="exhausted"):
        aligner.step()


def test_degrees_are_rejected(geometry):
    _angles, _projector, _volume, clean = geometry
    with pytest.raises(ValueError, match="radians|not radians"):
        JointGDAligner(clean, np.linspace(0, 180, N_ANGLES), JointGDConfig())


def test_angle_count_must_match(geometry):
    _angles, _projector, _volume, clean = geometry
    with pytest.raises(ValueError, match="angles for"):
        JointGDAligner(clean, np.linspace(0, np.pi, N_ANGLES + 1), JointGDConfig())


def test_astra_is_optional_and_says_so():
    """Missing ASTRA must raise a message that names the alternative, not fall back."""
    try:
        import astra  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        pytest.skip("astra is installed; the missing-dependency path cannot be exercised")

    with pytest.raises(ImportError, match="astra-toolbox"):
        make_projector("astra", 8, 16, np.linspace(0, np.pi, 8))


def test_unknown_projector_names_are_listed():
    with pytest.raises(KeyError, match="numpy"):
        make_projector("gridrec", 8, 16, np.linspace(0, np.pi, 8))


# ------------------------------------------------------------------------------
# Projector layer
# ------------------------------------------------------------------------------


def test_numpy_projector_is_approximately_adjoint(geometry):
    """``<P v, p> == <v, P^T p>`` to interpolation accuracy.

    SIRT tolerates a mismatched adjoint pair -- the preconditioners absorb it -- but a
    *badly* mismatched one turns the descent direction into an ascent direction on some
    modes, and the symptom is a loss that plateaus for no visible reason. 5% is what
    rotate-and-sum with linear interpolation delivers; a sign error scores -100%.
    """
    angles, projector, _volume, _clean = geometry
    rng = np.random.default_rng(0)
    volume = rng.random((N_SLICES, N_COLS, N_COLS)).astype(np.float32)
    sino = rng.random((N_SLICES, len(angles), N_COLS)).astype(np.float32)

    forward = float(np.sum(projector.forward(volume) * sino, dtype=np.float64))
    backward = float(np.sum(volume * projector.backward(sino), dtype=np.float64))

    assert abs(forward - backward) / abs(forward) < 0.05


def test_numpy_projector_shapes(geometry):
    angles, projector, volume, _clean = geometry
    assert projector.forward(volume).shape == (N_SLICES, len(angles), N_COLS)
    sino = np.zeros((N_SLICES, len(angles), N_COLS), dtype=np.float32)
    assert projector.backward(sino).shape == (N_SLICES, N_COLS, N_COLS)


def test_projectors_can_be_registered():
    calls = []

    def factory(n_slices, n_cols, angles):
        calls.append((n_slices, n_cols, len(angles)))
        return NumpyProjector3D(n_slices, n_cols, angles)

    register_projector("fake-for-test", factory)
    try:
        assert "fake-for-test" in available_projectors()
        make_projector("fake-for-test", 8, 16, np.linspace(0, np.pi, 4))
        assert calls == [(8, 16, 4)]
    finally:
        from tktomo.ptycho_align.core import joint_gd

        joint_gd._PROJECTORS.pop("fake-for-test", None)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------


def test_config_round_trips_through_json():
    """The stage list must survive JSON, which has neither tuples nor dataclasses."""
    import json

    config = JointGDConfig(stages=STAGES_REFINE, shift_cap_px=0.25, projector="numpy")
    restored = JointGDConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert isinstance(restored.stages, tuple)
    assert all(isinstance(stage, GDStage) for stage in restored.stages)


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown JointGDConfig field"):
        JointGDConfig.from_dict({"stages": [], "lr_shift": 0.5, "typo": 3})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr_volume": 2.5},
        {"lr_shift": 0.0},
        {"shift_cap_px": -1.0},
        {"momentum": 1.0},
        {"warmup_iters": -1},
        {"outlier_fallback": "median"},
        {"stages": ()},
    ],
)
def test_config_validates_its_numbers(kwargs):
    with pytest.raises(ValueError):
        JointGDConfig(**kwargs)


def test_stage_validates_its_numbers():
    with pytest.raises(ValueError):
        GDStage(binning=0, iterations=10)
    with pytest.raises(ValueError):
        GDStage(binning=4, iterations=0)
    with pytest.raises(ValueError):
        GDStage(binning=4, iterations=10, smooth_sigma=-1.0)


def test_named_schedules_match_the_original_environment_presets():
    """These four are ``JOINT_*``; changing one silently changes what a run means."""
    assert [(s.binning, s.iterations, s.smooth_sigma) for s in STAGES_STANDARD] == [
        (16, 150, 2.0),
        (8, 150, 1.0),
        (4, 100, 0.0),
    ]
    assert [(s.binning, s.iterations, s.smooth_sigma) for s in STAGES_REFINE] == [
        (4, 100, 0.5),
        (2, 80, 0.0),
    ]


def test_quality_weights_clip_into_range():
    steepness = np.array([0.0, 0.005, 0.01, 0.05, 1.0])
    weights = quality_weights(steepness)
    assert weights.min() >= 0.2 and weights.max() <= 1.0
    assert weights[0] == pytest.approx(1.0)  # a perfect projection is not de-weighted
    assert weights[-1] == pytest.approx(0.2)  # the worst is floored, not zeroed
    assert np.all(np.diff(weights) <= 0)


def test_weights_must_match_the_stack(geometry):
    angles, _projector, _volume, clean = geometry
    with pytest.raises(ValueError, match="weights for"):
        JointGDAligner(clean, angles, JointGDConfig(), weights=np.ones(3))


def test_finalized_shifts_summary_is_readable():
    result: FinalizedShifts = clean_shifts(np.zeros((8, 2)))
    text = result.summary()
    assert "rms dy" in text and "MAD outlier" in text
