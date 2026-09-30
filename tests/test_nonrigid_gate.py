"""The definition of done for the non-rigid decision gate.

The gate's job is to say **no** correctly, so most of these tests are about the ways a
residual can look like deformation without being it. Three carry the weight:

* :func:`test_sparse_spikes_are_concentrated_but_not_localised` -- a heavy-tailed
  residual whose raw concentration (0.77) sails past any fixed threshold while its
  permutation null sits at exactly the same value. This is the case a threshold-based
  localisation test gets wrong, and the reason the null exists.
* :func:`test_vacuum_makes_everything_look_localised_without_a_support_mask` -- the same
  residual scores z = 158 without an object support and z = 0.5 with one. A frame that is
  mostly air makes every residual "localised".
* :func:`test_a_real_local_deformation_passes_the_whole_gate` -- a phantom, a genuine
  local warp, a real parallel-beam projector, and the gate saying yes with every
  alternative excluded; paired with the control on the same phantom with no deformation,
  where it says no.

numpy + scipy + pytest only: no GPU, no tomopy, no scikit-image, no beamtime data.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
from scipy.ndimage import shift as ndshift

from tktomo.ptycho_align.core.nonrigid_gate import (
    DIAGNOSTICS_AVAILABLE,
    Alternative,
    GateConfig,
    GateVerdict,
    Recommendation,
    evaluate_gate,
    format_gate,
    gate_from_engine,
    measure_localisation,
    measure_plateau,
    measure_temporal_change,
    measure_upstream,
    rigid_reduced_residual,
)

# A residual history that has genuinely flattened, and the matching shift updates. Used
# wherever a test is about something other than the plateau, so the plateau never
# accidentally becomes the reason for the verdict.
FLAT_HISTORY = [0.50, 0.30, 0.20, 0.160, 0.1552, 0.1549, 0.1551, 0.1550]
CONVERGED_SHIFTS = [2.0, 1.0, 0.5, 0.2, 0.08, 0.05, 0.03, 0.02]

SMALL = GateConfig(block=8, n_null=24)


# ---------------------------------------------------------------------------------
# Fixtures: a compact object and its projections
# ---------------------------------------------------------------------------------

N_ANGLES, N_V, N_U = 32, 64, 64


def frame_object() -> np.ndarray:
    """A disc filling roughly a third of the frame, so vacuum is a real fraction of it."""
    y, x = np.mgrid[0:N_V, 0:N_U]
    return np.exp(-(((y - 32) / 14.0) ** 2 + ((x - 32) / 15.0) ** 2))


def simulated_stack(n: int = N_ANGLES) -> tuple[np.ndarray, np.ndarray]:
    """A stack of ``n`` "reprojections" and their angles. Smooth, compact, band-limited."""
    angles = np.linspace(0.0, np.pi, n, endpoint=False)
    obj = frame_object()
    return np.stack([obj * (1.0 + 0.05 * math.sin(a)) for a in angles]), angles


# ---------------------------------------------------------------------------------
# (a) the plateau, measured rather than asserted
# ---------------------------------------------------------------------------------


def test_a_still_falling_residual_is_not_a_plateau():
    report = measure_plateau([1.0, 0.8, 0.64, 0.51, 0.41, 0.33], shift_rms=[1.0] * 6)
    assert not report.plateaued
    assert report.tail_improvement > 0.3
    # And it says what continuing would buy, which is the number the user needs.
    assert report.projected_gain > 0.3


def test_a_flat_residual_is_a_plateau():
    report = measure_plateau(FLAT_HISTORY, shift_rms=CONVERGED_SHIFTS)
    assert report.plateaued
    assert report.shift_converged
    assert abs(report.tail_improvement) < 0.02


def test_a_slow_but_clean_decay_is_not_called_a_plateau():
    """0.5%/iteration is under the tolerance -- and is still convergence, not a plateau.

    This is the case the naive "improvement < tolerance" test gets wrong, and it is not
    academic: it is exactly what a well-conditioned rigid loop looks like in its last
    iterations, where five more iterations are cheap and would each buy another 0.5%.
    """
    history = [0.2 * 0.995**k for k in range(10)]
    report = measure_plateau(history, shift_rms=[0.01] * 10)
    assert report.tail_improvement < GateConfig().plateau_tolerance
    assert report.snr > GateConfig().plateau_snr
    assert not report.plateaued


def test_a_noisy_flat_history_is_a_plateau():
    """The same tolerance, but the wobble is bigger than the trend, so it IS flat."""
    rng = np.random.default_rng(3)
    history = list(0.155 + 0.004 * rng.standard_normal(12))
    report = measure_plateau(history, shift_rms=[0.02] * 12)
    assert report.plateaued
    assert report.snr < GateConfig().plateau_snr


def test_a_rising_residual_is_divergence_and_not_a_plateau():
    """A residual climbing by 1%/iteration has a tail "improvement" below tolerance too.

    Testing only ``improvement < tolerance`` calls that a plateau and waves a diverging
    alignment through to the non-rigid stage, which is the worst possible moment for it.
    """
    history = [0.15 * 1.01**k for k in range(8)]
    report = measure_plateau(history, shift_rms=[0.02] * 8)
    assert report.tail_improvement < GateConfig().plateau_tolerance
    assert not report.plateaued
    assert "RISING" in report.reason


def test_too_short_a_history_cannot_be_judged():
    report = measure_plateau([0.5, 0.4], shift_rms=[1.0, 0.5])
    assert not report.plateaued
    assert "at least" in report.reason
    assert report.n_iterations == 2


def test_a_plateau_at_the_noise_floor_is_flagged():
    report = measure_plateau(FLAT_HISTORY, shift_rms=CONVERGED_SHIFTS, noise_floor=0.150)
    assert report.plateaued
    assert report.at_noise_floor
    assert report.floor_ratio == pytest.approx(0.155 / 0.150, rel=0.02)


def test_a_plateau_far_above_the_noise_floor_is_not():
    report = measure_plateau(FLAT_HISTORY, shift_rms=CONVERGED_SHIFTS, noise_floor=0.02)
    assert report.plateaued
    assert not report.at_noise_floor


# ---------------------------------------------------------------------------------
# (b) localisation against a null of uniform spread
# ---------------------------------------------------------------------------------


def test_gaussian_noise_is_not_localised():
    rng = np.random.default_rng(1)
    measured = np.stack([frame_object()] * N_ANGLES)
    stat = measure_localisation(
        0.01 * rng.standard_normal((N_ANGLES, N_V, N_U)), measured=measured, config=SMALL
    )
    assert stat.z < 3.0
    assert not stat.is_localised


def test_sparse_spikes_are_concentrated_but_not_localised():
    """The case a fixed threshold gets wrong, in both directions.

    Six hot pixels per frame at random places -- cosmic rays, hot detector pixels, a
    residual with heavy tails. The raw concentration is 0.77, far above any threshold
    anyone would set, but the permutation null of the SAME pixel values is 0.77 too: the
    concentration is forced by the amplitude distribution and says nothing about spatial
    structure. Compare with :func:`test_a_localised_blob_beats_the_null`, where a genuine
    localised feature scores a raw concentration of only 0.29 -- below the same threshold
    it would have to pass. Fixed thresholds fire on the wrong one and miss the right one.
    """
    rng = np.random.default_rng(1)
    measured = np.stack([frame_object()] * N_ANGLES)
    residual = 0.01 * rng.standard_normal((N_ANGLES, N_V, N_U))
    for i in range(N_ANGLES):
        rows = rng.integers(8, N_V - 8, size=6)
        columns = rng.integers(8, N_U - 8, size=6)
        residual[i, rows, columns] += 0.5 * rng.standard_normal(6)

    stat = measure_localisation(residual, measured=measured, config=SMALL)
    assert stat.concentration > 0.5  # a fixed threshold would fire here
    assert stat.null_mean == pytest.approx(stat.concentration, abs=0.05)
    assert abs(stat.z) < 3.0
    assert not stat.is_localised


def test_a_localised_blob_beats_the_null():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:N_V, 0:N_U]
    blob = np.exp(-(((y - 22) / 4.0) ** 2 + ((x - 40) / 4.0) ** 2))
    measured = np.stack([frame_object()] * N_ANGLES)
    residual = 0.01 * rng.standard_normal((N_ANGLES, N_V, N_U)) + 0.05 * blob[None]

    stat = measure_localisation(residual, measured=measured, config=SMALL)
    assert stat.concentration < 0.35  # smaller than the spikes above, and still real
    assert stat.z > 10.0
    assert stat.is_localised
    assert stat.is_angle_consistent
    assert stat.p_value <= 1.0 / (SMALL.n_null + 1) + 1e-12


def test_hot_spots_that_move_between_angles_are_not_angle_consistent():
    """Localised at every angle, but never in the same place: not sample deformation."""
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:N_V, 0:N_U]
    measured = np.stack([frame_object()] * N_ANGLES)
    frames = []
    for _ in range(N_ANGLES):
        cy, cx = rng.integers(16, 48, size=2)
        frames.append(0.05 * np.exp(-(((y - cy) / 4.0) ** 2 + ((x - cx) / 4.0) ** 2)))
    stat = measure_localisation(np.stack(frames), measured=measured, config=SMALL)
    assert stat.is_localised
    assert not stat.is_angle_consistent


def test_vacuum_makes_everything_look_localised_without_a_support_mask():
    """The residual is uniform *inside the object* and zero outside; it is not localised.

    Without a support mask the vacuum blocks count as sample, all the energy lands in the
    object blocks, and the statistic reports z = 158. With the object's own footprint the
    same residual scores z ~ 0. The square is aligned to the block grid so the effect is
    the mask and not a partial-coverage artefact.
    """
    rng = np.random.default_rng(2)
    square = np.zeros((N_V, N_U))
    square[16:48, 16:48] = 1.0
    measured = np.stack([square] * N_ANGLES)
    residual = np.zeros((N_ANGLES, N_V, N_U))
    residual[:, 16:48, 16:48] = 0.05 * rng.standard_normal((N_ANGLES, 32, 32))

    blind = measure_localisation(residual, config=SMALL)
    masked = measure_localisation(residual, measured=measured, config=SMALL)

    assert blind.is_localised and blind.z > 20.0
    assert not masked.is_localised
    assert masked.n_support_blocks < blind.n_support_blocks
    assert blind.support_from.startswith("ALL blocks")


def test_an_explicit_support_mask_is_honoured():
    rng = np.random.default_rng(2)
    square = np.zeros((N_V, N_U), dtype=bool)
    square[16:48, 16:48] = True
    residual = np.zeros((N_ANGLES, N_V, N_U))
    residual[:, 16:48, 16:48] = 0.05 * rng.standard_normal((N_ANGLES, 32, 32))
    stat = measure_localisation(residual, support=square, config=SMALL)
    assert stat.n_support_blocks == 16
    assert not stat.is_localised
    assert "caller-supplied" in stat.support_from


def test_too_few_blocks_is_refused_rather_than_guessed():
    residual = np.random.default_rng(0).standard_normal((4, 20, 20))
    with pytest.raises(ValueError, match="2x2"):
        measure_localisation(residual, config=GateConfig(block=16, n_null=4))


# ---------------------------------------------------------------------------------
# (c) the upstream decomposition
# ---------------------------------------------------------------------------------


def shifted(stack: np.ndarray, dv: np.ndarray, du: np.ndarray) -> np.ndarray:
    """Shift each frame of a stack by ``(dv, du)`` px. Content moves in +v / +u."""
    return np.stack(
        [
            ndshift(stack[i], (float(dv[i]), float(du[i])), order=3, mode="nearest")
            for i in range(stack.shape[0])
        ]
    )


def test_a_pure_shift_is_attributed_to_shifts_and_not_to_a_ramp():
    """Sequential attribution credits a pure shift with 16% 'ramp'; unique credits ~0.

    That 16% is enough to fire the phase-ramp test and send someone off to remove a ramp
    that was never in the data. It is the reason this decomposition reports each group's
    unique contribution and puts the ambiguous part in ``shared_fraction`` instead.
    """
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(5)
    du = rng.normal(0.0, 0.8, size=N_ANGLES)
    dv = rng.normal(0.0, 0.8, size=N_ANGLES)
    measured = shifted(simulated, dv, du)

    report = measure_upstream(measured, simulated, angles=angles)
    assert report.shift_fraction > 0.5
    assert report.ramp_fraction < 0.05
    assert report.dominant == "shift"
    # ...and the shift it recovers is the shift that was injected.
    assert report.du_residual_rms_px == pytest.approx(
        float(np.sqrt(np.mean((du - du.mean()) ** 2))), rel=0.3
    )


def test_a_constant_shift_is_read_as_a_rotation_centre_error():
    simulated, angles = simulated_stack()
    measured = shifted(simulated, np.zeros(N_ANGLES), np.full(N_ANGLES, 0.6))
    report = measure_upstream(measured, simulated, angles=angles)
    assert report.center_offset_px == pytest.approx(0.6, abs=0.15)
    assert report.center_offset_se_px < 0.05
    assert report.du_residual_rms_px < 0.1


def test_the_in_plane_translation_gauge_is_not_counted_as_an_error():
    """``X cos t + Y sin t`` moves the object, not the geometry. It must not be an error.

    Conflating this with a misalignment is how a correctly aligned dataset gets sent back
    for more alignment; this repo has measured a perfect alignment scoring 0.2 px under
    mean-only gauge removal.
    """
    simulated, angles = simulated_stack()
    du = 1.5 * np.cos(angles) + 0.9 * np.sin(angles)
    measured = shifted(simulated, np.zeros(N_ANGLES), du)

    report = measure_upstream(measured, simulated, angles=angles)
    assert report.gauge_amplitude_px == pytest.approx(math.hypot(1.5, 0.9), rel=0.25)
    assert abs(report.center_offset_px) < 0.2
    assert report.du_residual_rms_px < 0.2


def test_a_phase_ramp_is_attributed_to_the_ramp():
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(6)
    ramp = np.linspace(-1.0, 1.0, N_U)[None, :] * np.ones((N_V, 1))
    measured = simulated + 0.05 * rng.standard_normal((N_ANGLES, 1, 1)) * ramp[None]

    report = measure_upstream(measured, simulated, angles=angles)
    assert report.ramp_fraction > 0.5
    assert report.shift_fraction < 0.05
    assert report.dominant == "ramp"


def test_a_gain_mismatch_is_absorbed_as_a_nuisance_and_not_diagnosed():
    """A per-angle gain error must NOT fire an upstream verdict, and here is why.

    Reconstructing an inconsistent (deforming) volume and reprojecting it mismatches the
    measurement in overall scale at every angle -- 15% of the residual energy on the
    scenario phantom, with no normalisation error anywhere in the data. Diagnosing that
    would send every real deformation off to fix an imaginary flat-field. A constant and
    a global scale also move nothing, so neither can be confused with a misalignment.
    """
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(7)
    measured = simulated * (1.0 + 0.1 * rng.standard_normal((N_ANGLES, 1, 1)))
    report = measure_upstream(measured, simulated, angles=angles)
    assert report.nuisance_fraction > 0.4
    assert report.ramp_fraction < 0.05
    assert report.shift_fraction < 0.05
    assert report.dominant == "nuisance"

    verdict = gate(measured, simulated, angles)
    assert verdict.recommendation is not Recommendation.FIX_UPSTREAM


def test_drift_and_jitter_are_told_apart_by_their_time_correlation():
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(8)
    acquisition = np.arange(N_ANGLES)

    smooth = 1.2 * np.sin(np.linspace(0.0, 2.0 * np.pi, N_ANGLES))
    drift = measure_upstream(
        shifted(simulated, smooth, np.zeros(N_ANGLES)), simulated,
        angles=angles, acquisition_index=acquisition,
    )
    jitter = measure_upstream(
        shifted(simulated, rng.normal(0.0, 0.8, N_ANGLES), np.zeros(N_ANGLES)), simulated,
        angles=angles, acquisition_index=acquisition,
    )
    assert drift.dv_lag1 > 0.5
    assert jitter.dv_lag1 < 0.5
    assert drift.time_order_known and jitter.time_order_known


def test_acquisition_order_is_used_when_it_differs_from_angle_order():
    """A drift smooth in TIME is angle-random when the scan interlaces sub-tomograms.

    This is the ordering bug the non-rigid module refuses to guess at, and it is silent:
    read in angle order the same drift looks like jitter and gets the opposite verdict.
    """
    simulated, angles = simulated_stack()
    # Four interlaced sub-tomograms: angle order and acquisition order are not the same.
    acquisition = np.concatenate([np.arange(8) * 4 + k for k in range(4)])
    smooth_in_time = np.zeros(N_ANGLES)
    smooth_in_time[np.argsort(acquisition)] = np.linspace(-1.2, 1.2, N_ANGLES)
    measured = shifted(simulated, smooth_in_time, np.zeros(N_ANGLES))

    with_time = measure_upstream(
        measured, simulated, angles=angles, acquisition_index=acquisition
    )
    without = measure_upstream(measured, simulated, angles=angles)
    assert with_time.dv_lag1 > 0.5
    assert without.dv_lag1 < with_time.dv_lag1


# ---------------------------------------------------------------------------------
# The gate: the five outcomes
# ---------------------------------------------------------------------------------


def gate(measured, simulated, angles, *, history=None, shifts=None, **kwargs) -> GateVerdict:
    return evaluate_gate(
        residual_history=history if history is not None else FLAT_HISTORY,
        shift_history=shifts if shifts is not None else CONVERGED_SHIFTS,
        measured=measured, simulated=simulated, angles=angles,
        acquisition_index=np.arange(measured.shape[0]), config=SMALL, **kwargs,
    )


def test_gate_says_more_rigid_iterations_while_the_residual_is_still_falling():
    simulated, angles = simulated_stack()
    y, x = np.mgrid[0:N_V, 0:N_U]
    blob = 0.05 * np.exp(-(((y - 22) / 4.0) ** 2 + ((x - 40) / 4.0) ** 2))
    verdict = gate(
        simulated + blob[None], simulated, angles,
        history=[1.0, 0.8, 0.64, 0.51, 0.41], shifts=[1.0, 0.8, 0.6, 0.4, 0.3],
    )
    assert verdict.recommendation is Recommendation.MORE_RIGID_ITERATIONS
    assert not verdict
    # The localisation test itself passed -- the verdict must say so, or the user
    # concludes there is no deformation when the answer is "not yet".
    assert any("localisation test itself PASSED" in note for note in verdict.notes)


def test_gate_says_more_rigid_iterations_when_the_shifts_have_not_settled():
    simulated, angles = simulated_stack()
    y, x = np.mgrid[0:N_V, 0:N_U]
    blob = 0.05 * np.exp(-(((y - 22) / 4.0) ** 2 + ((x - 40) / 4.0) ** 2))
    verdict = gate(
        simulated + blob[None], simulated, angles,
        shifts=[2.0, 1.0, 0.5, 0.4, 0.45, 0.42, 0.44, 0.43],
    )
    assert verdict.recommendation is Recommendation.MORE_RIGID_ITERATIONS
    assert "wandering" in verdict.headline


def test_gate_says_fix_upstream_on_an_unremoved_phase_ramp():
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(9)
    ramp = np.linspace(-1.0, 1.0, N_U)[None, :] * np.ones((N_V, 1))
    verdict = gate(
        simulated + 0.05 * rng.standard_normal((N_ANGLES, 1, 1)) * ramp[None], simulated, angles
    )
    assert verdict.recommendation is Recommendation.FIX_UPSTREAM
    assert "phase ramp" in verdict.headline
    assert {a.name for a in verdict.unexcluded} >= {"phase_ramp_or_offset"}


def test_gate_says_fix_upstream_on_a_rotation_centre_error():
    simulated, angles = simulated_stack()
    measured = shifted(simulated, np.zeros(N_ANGLES), np.full(N_ANGLES, 0.8))
    verdict = gate(measured, simulated, angles)
    assert verdict.recommendation is Recommendation.FIX_UPSTREAM
    assert "rotation centre" in verdict.headline


def test_gate_says_more_rigid_iterations_on_an_unconverged_drift():
    """A smooth leftover shift is drift the rigid stage has not finished, not deformation."""
    simulated, angles = simulated_stack()
    acquisition = np.arange(N_ANGLES)
    smooth = 1.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, N_ANGLES))
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=shifted(simulated, smooth, np.zeros(N_ANGLES)), simulated=simulated,
        angles=angles, acquisition_index=acquisition, config=SMALL,
    )
    assert verdict.recommendation is Recommendation.FIX_UPSTREAM
    assert "unconverged rigid drift" in verdict.headline


def test_gate_says_accept_rigid_on_spread_angle_random_residual():
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(10)
    verdict = gate(simulated + 0.01 * rng.standard_normal(simulated.shape), simulated, angles)
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    assert "SPREAD" in verdict.headline
    # And it warns that this criterion vetoes a global deformation.
    assert any("GLOBAL deformation" in note for note in verdict.notes)


def test_gate_says_accept_rigid_on_jitter():
    """Angle-random per-projection shifts: irreducible, and the classic overfitting trap.

    Note what the gate actually says here, which is *stronger* than "this looks like
    jitter": the whole residual is inside the rigid model's reach, so there is nothing for
    a deformation field to describe at all. A leftover 0.7 px jitter is not a deformation,
    it is a shift the rigid stage cannot remove because it is not smooth in anything.

    The residual is nonetheless intensely "localised" by the raw statistic (z = 350, on the
    object edges where a shift shows up) and angle-consistent (z = 64), which is exactly
    why the localisation test alone must never be the gate.
    """
    simulated, angles = simulated_stack(64)
    rng = np.random.default_rng(11)
    n = simulated.shape[0]
    measured = shifted(simulated, rng.normal(0, 0.7, n), rng.normal(0, 0.7, n))
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=measured, simulated=simulated, angles=angles,
        acquisition_index=np.arange(n), config=SMALL,
    )
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    assert not verdict
    assert "nothing_beyond_rigid" in {a.name for a in verdict.unexcluded}
    assert verdict.upstream.unexplained_fraction < 0.05
    # It is jitter and not drift, and the gate has the numbers to say so.
    assert verdict.upstream.du_lag1 < 0.5
    assert "unconverged_rigid_drift" not in {a.name for a in verdict.unexcluded}
    # The raw residual would have looked beautifully localised. That is the trap.
    raw = measure_localisation(measured - simulated, measured=measured, config=SMALL)
    assert raw.is_localised and raw.is_angle_consistent


def test_the_localisation_is_measured_on_the_rigid_reduced_residual():
    """The single most consequential choice in the gate, pinned.

    A per-angle gain error alone accounts for essentially the whole raw residual. Measured
    raw, what is left over is the object's own shape and the concentration is enormous;
    measured on the rigid-reduced residual there is nothing left to concentrate.
    """
    simulated, angles = simulated_stack()
    rng = np.random.default_rng(21)
    measured = simulated * (1.0 + 0.1 * rng.standard_normal((N_ANGLES, 1, 1)))

    raw_energy = float(np.sum((measured - simulated) ** 2))
    reduced = rigid_reduced_residual(measured, simulated)
    assert float(np.sum(reduced**2)) < 0.01 * raw_energy

    verdict = gate(measured, simulated, angles)
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    assert "nothing for a deformation field" in verdict.headline


def test_the_rigid_reduction_keeps_what_no_rigid_model_can_remove():
    """It removes offset, gain and shift exactly; a localised bump survives it."""
    simulated, _angles = simulated_stack()
    y, x = np.mgrid[0:N_V, 0:N_U]
    bump = 0.05 * np.exp(-(((y - 22) / 4.0) ** 2 + ((x - 40) / 4.0) ** 2))

    offset_gain_shift = shifted(
        1.2 * simulated + 0.3, np.full(N_ANGLES, 0.4), np.full(N_ANGLES, -0.3)
    )
    reduced = rigid_reduced_residual(offset_gain_shift, simulated)
    scale = float(np.sqrt(np.mean(simulated**2)))
    assert float(np.sqrt(np.mean(reduced**2))) < 0.05 * scale

    with_bump = rigid_reduced_residual(simulated + bump[None], simulated)
    assert float(np.sqrt(np.mean(with_bump**2))) > 0.3 * float(np.sqrt(np.mean(bump**2)))


def test_gate_says_accept_rigid_when_the_plateau_is_the_noise_floor():
    simulated, angles = simulated_stack()
    y, x = np.mgrid[0:N_V, 0:N_U]
    blob = 0.05 * np.exp(-(((y - 22) / 4.0) ** 2 + ((x - 40) / 4.0) ** 2))
    verdict = gate(simulated + blob[None], simulated, angles, noise_floor=0.150)
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    assert "noise floor" in verdict.headline


def test_gate_refuses_to_decide_without_a_rigid_history():
    simulated, angles = simulated_stack()
    verdict = evaluate_gate(measured=simulated, simulated=simulated * 1.01, angles=angles,
                            config=SMALL)
    assert verdict.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert "residual_history" in verdict.headline
    assert not verdict


def test_gate_refuses_to_decide_without_residual_maps():
    verdict = evaluate_gate(residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS)
    assert verdict.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert "measured=" in verdict.headline


def test_residual_maps_alone_lose_the_upstream_test_and_say_so():
    """Handing over the residual without the simulated stack is allowed -- and weaker."""
    rng = np.random.default_rng(12)
    residual = 0.01 * rng.standard_normal((N_ANGLES, N_V, N_U))
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        residual=residual, config=SMALL,
    )
    assert verdict.upstream is None
    assert any("upstream" in note for note in verdict.notes)


# ---------------------------------------------------------------------------------
# The whole gate on a real local deformation, and its control
# ---------------------------------------------------------------------------------

NZ = NX = 32


def deformable_phantom() -> np.ndarray:
    z, y, x = np.mgrid[0:NZ, 0:NX, 0:NX].astype(np.float32)
    volume = np.zeros((NZ, NX, NX), dtype=np.float32)
    volume[((z - 15.5) / 12.0) ** 2 + ((y - 15.5) / 11.0) ** 2 + ((x - 15.5) / 10.0) ** 2 < 1] = 1.0
    for (fz, fy, fx), radius, amplitude in (
        ((0.35, 0.40, 0.42), 0.10, 0.9),
        ((0.62, 0.58, 0.60), 0.08, -0.6),
    ):
        inside = ((z - fz * NZ) ** 2 + (y - fy * NX) ** 2 + (x - fx * NX) ** 2) < (radius * NX) ** 2
        volume[inside] += amplitude
    return gaussian_filter(volume, 0.8)


def locally_deformed(volume: np.ndarray, amplitude: float = 2.0) -> np.ndarray:
    """Warp one corner of the phantom and leave the rest alone. This is the target case."""
    z, y, x = np.mgrid[0:NZ, 0:NX, 0:NX].astype(np.float32)
    bump = np.exp(-(((z - 10) / 4.0) ** 2 + ((y - 12) / 4.0) ** 2 + ((x - 20) / 4.0) ** 2))
    coordinates = np.stack(
        [z + amplitude * bump, y + 0.5 * amplitude * bump, x - amplitude * bump]
    )
    return map_coordinates(volume, coordinates, order=3, mode="nearest")


def project(volume: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Parallel-beam projection: rotate about z, integrate along y. A real projector."""
    out = np.empty((angles.size, NZ, NX), dtype=np.float32)
    for i, theta in enumerate(angles):
        out[i] = rotate(
            volume, np.degrees(theta), axes=(1, 2), reshape=False, order=1, mode="constant"
        ).sum(axis=1)
    return out


N_TIME_BLOCKS, ANGLES_PER_BLOCK = 4, 8


@pytest.fixture(scope="module")
def deformation_case():
    """Four interlaced sub-tomograms, the deformation growing between them.

    The deformation must EVOLVE, not merely be present: a warp applied identically to every
    projection is a different sample, not a deforming one, and the gate is required to tell
    those apart (a static localised residual is a reconstruction artefact). Angle order and
    acquisition order differ here, as they do in a real series.
    """
    volume = deformable_phantom()
    angles, acquisition, deformed, clean = [], [], [], []
    for k in range(N_TIME_BLOCKS):
        theta = np.linspace(0.0, np.pi, ANGLES_PER_BLOCK, endpoint=False) + k * np.pi / (
            ANGLES_PER_BLOCK * N_TIME_BLOCKS
        )
        amplitude = 2.5 * (k / (N_TIME_BLOCKS - 1)) ** 1.2
        deformed.append(project(locally_deformed(volume, amplitude), theta))
        clean.append(project(volume, theta))
        angles.append(theta)
        acquisition.append(np.arange(ANGLES_PER_BLOCK) + k * ANGLES_PER_BLOCK)
    # The same total deformation, applied identically at every time: a different sample,
    # not a deforming one. Localised and angle-consistent, and NOT a case for non-rigid.
    static = np.concatenate(
        [project(locally_deformed(volume, 2.5), theta) for theta in angles]
    )
    return {
        "angles": np.concatenate(angles),
        "acquisition_index": np.concatenate(acquisition),
        "simulated": np.concatenate(clean),
        "deformed": np.concatenate(deformed),
        "static": static,
    }


def test_a_real_local_deformation_passes_the_whole_gate(deformation_case):
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["deformed"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"],
        acquisition_index=deformation_case["acquisition_index"], config=SMALL,
    )
    assert verdict.recommendation is Recommendation.RUN_NONRIGID
    assert bool(verdict)
    assert verdict.unexcluded == ()
    assert verdict.localisation.is_localised and verdict.localisation.is_angle_consistent
    # A local warp is not a rigid shift: most of the residual is beyond the rigid basis.
    assert verdict.upstream.unexplained_fraction > 0.3
    # And a "yes" is a licence to run the method, not to believe its output.
    assert any("held-out" in note for note in verdict.notes)


def test_the_control_with_no_deformation_says_no(deformation_case):
    """Same phantom, same projector, noise instead of a warp. The gate must decline."""
    rng = np.random.default_rng(13)
    simulated = deformation_case["simulated"]
    measured = simulated + 0.002 * float(np.abs(simulated).max()) * rng.standard_normal(
        simulated.shape
    )
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=measured, simulated=simulated, angles=deformation_case["angles"],
        acquisition_index=np.arange(deformation_case["angles"].size), config=SMALL,
    )
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    assert not verdict


# ---------------------------------------------------------------------------------
# The verdict object itself
# ---------------------------------------------------------------------------------


def test_a_static_localised_residual_is_a_reconstruction_artefact_not_deformation(
    deformation_case,
):
    """The false positive the benchmark caught, pinned.

    A residual that is localised AND angle-consistent passes both of the roadmap's tests
    and can still have nothing to do with the sample moving: an FBP streak or an edge the
    band-limited projector cannot reproduce sits in the same place at every angle, for the
    whole scan. Measured on the undeformed benchmark phantom at 32 x 64 x 64: localisation
    z = 108, angle-consistency z = 305, and the gate said "go non-rigid" until the temporal
    test existed.

    Here the whole deformation is applied identically to every projection -- a *different*
    sample, not a deforming one -- so the residual is localised, consistent, and static.
    """
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["static"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"],
        acquisition_index=deformation_case["acquisition_index"], config=SMALL,
    )
    assert verdict.localisation.is_localised
    assert verdict.localisation.is_angle_consistent
    assert not verdict.temporal.evolves
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    assert "does not CHANGE with acquisition time" in verdict.headline
    assert "static_reconstruction_artefact" in {a.name for a in verdict.unexcluded}


def test_without_acquisition_order_a_localised_residual_cannot_be_judged(deformation_case):
    """No acquisition order, no temporal test, no yes. The aligner needs it too."""
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["deformed"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"], config=SMALL,
    )
    assert verdict.temporal is None
    assert verdict.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert "acquisition_index" in verdict.headline
    assert not verdict


def test_measure_temporal_change_separates_an_evolving_pattern_from_a_static_one():
    """Directly, on constructed residual stacks with the same energy in both."""
    rng = np.random.default_rng(31)
    n, blocks = 32, 4
    acquisition = np.arange(n)
    y, x = np.mgrid[0:N_V, 0:N_U]
    measured = np.stack([frame_object()] * n)

    def blob(cy, cx):
        return np.exp(-(((y - cy) / 5.0) ** 2 + ((x - cx) / 5.0) ** 2))

    static = np.stack([0.05 * blob(24, 40) for _ in range(n)])
    static += 0.005 * rng.standard_normal(static.shape)

    moving = np.stack(
        [0.05 * blob(24 + 8 * (i // (n // blocks)), 40) for i in range(n)]
    )
    moving += 0.005 * rng.standard_normal(moving.shape)

    still = measure_temporal_change(static, acquisition, measured=measured, config=SMALL)
    walks = measure_temporal_change(moving, acquisition, measured=measured, config=SMALL)
    assert not still.evolves
    assert walks.evolves
    assert walks.z > still.z


def test_temporal_change_needs_enough_projections_to_block():
    residual = np.random.default_rng(0).standard_normal((5, N_V, N_U))
    with pytest.raises(ValueError, match="acquisition-time blocks"):
        measure_temporal_change(residual, np.arange(5), config=SMALL)


def test_a_verdict_serialises_to_json_with_no_nan(deformation_case):
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["deformed"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"],
        acquisition_index=deformation_case["acquisition_index"], config=SMALL,
    )
    text = verdict.to_json()
    # json.dumps happily writes NaN and Infinity, which no other parser accepts. Reading
    # it back with the strict parser is the only way to catch that.
    restored = json.loads(text, parse_constant=_reject)
    assert restored["recommendation"] == verdict.recommendation.value
    assert restored["localisation"]["z"] == pytest.approx(verdict.localisation.z)
    assert isinstance(restored["alternatives"], list)


def _reject(name: str):  # pragma: no cover - only runs when the test is failing
    raise AssertionError(f"the verdict JSON contains a non-standard constant: {name}")


def test_bool_is_true_only_for_run_nonrigid():
    for recommendation in Recommendation:
        verdict = GateVerdict(
            recommendation=recommendation, confidence=0.0, headline="",
            plateau=None, localisation=None, upstream=None,
        )
        assert bool(verdict) is (recommendation is Recommendation.RUN_NONRIGID)


def test_unexcluded_lists_only_the_alternatives_that_survived():
    verdict = GateVerdict(
        recommendation=Recommendation.ACCEPT_RIGID, confidence=0.5, headline="",
        plateau=None, localisation=None, upstream=None,
        alternatives=(
            Alternative("a", True, 1.0, 2.0, ""),
            Alternative("b", False, 3.0, 2.0, ""),
        ),
    )
    assert [a.name for a in verdict.unexcluded] == ["b"]


def test_config_round_trips():
    config = GateConfig(block=16, localisation_z=3.0, n_null=8)
    assert GateConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="Unknown GateConfig"):
        GateConfig.from_dict({"block": 16, "nonsense": 1})


def test_format_gate_mentions_the_recommendation_and_the_alternatives(deformation_case):
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["deformed"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"],
        acquisition_index=deformation_case["acquisition_index"], config=SMALL,
    )
    text = format_gate(verdict)
    assert "RUN_NONRIGID" in text
    assert "ALTERNATIVES CONSIDERED" in text
    assert "rotation_centre" in text


# ---------------------------------------------------------------------------------
# Integration: the engine, the aligner's own precondition, and the diagnostics package
# ---------------------------------------------------------------------------------


class _FakeIteration:
    def __init__(self, residual: float, error: float) -> None:
        self.residual = residual
        self.error = error


class _FakeState:
    def __init__(self, angles: np.ndarray) -> None:
        self.angles = angles


class _FakeEngine:
    """The subset of ``AlignmentEngine`` the gate touches. Keeps this test tomopy-free."""

    def __init__(self, measured, simulated, angles):
        self.history = [
            _FakeIteration(r, e) for r, e in zip(FLAT_HISTORY, CONVERGED_SHIFTS, strict=True)
        ]
        self.last_aligned = measured
        self.last_simulated = simulated
        self.state = _FakeState(angles)


def test_gate_from_engine_reads_history_and_the_cached_stacks(deformation_case):
    engine = _FakeEngine(
        deformation_case["deformed"], deformation_case["simulated"], deformation_case["angles"]
    )
    verdict = gate_from_engine(
        engine, acquisition_index=deformation_case["acquisition_index"], config=SMALL
    )
    assert verdict.recommendation is Recommendation.RUN_NONRIGID
    assert verdict.plateau.n_iterations == len(FLAT_HISTORY)


def test_gate_from_engine_survives_an_empty_cache(deformation_case):
    engine = _FakeEngine(None, None, deformation_case["angles"])
    verdict = gate_from_engine(
        engine, acquisition_index=deformation_case["acquisition_index"], config=SMALL
    )
    assert verdict.recommendation is Recommendation.INSUFFICIENT_EVIDENCE


def test_the_verdict_converts_to_the_aligners_own_evidence_type(deformation_case):
    """The gate and NonRigidAligner's internal precondition must not disagree.

    ``as_rigid_evidence`` is the bridge, and this checks that a gate saying yes produces
    evidence the aligner also accepts -- otherwise a user is told to go ahead and then
    refused by the thing they were told to run.
    """
    nonrigid = pytest.importorskip("tktomo.ptycho_align.core.nonrigid")
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["deformed"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"],
        acquisition_index=deformation_case["acquisition_index"], config=SMALL,
    )
    assert verdict.recommendation is Recommendation.RUN_NONRIGID
    evidence = verdict.as_rigid_evidence(FLAT_HISTORY, CONVERGED_SHIFTS)
    assert evidence.localisation is not None
    assert evidence.localisation.is_localised
    assert bool(nonrigid.nonrigid_is_warranted(evidence))


def test_a_no_verdict_also_refuses_through_the_aligners_evidence_type(deformation_case):
    nonrigid = pytest.importorskip("tktomo.ptycho_align.core.nonrigid")
    rng = np.random.default_rng(14)
    simulated = deformation_case["simulated"]
    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=simulated + 0.01 * float(np.abs(simulated).max()) * rng.standard_normal(
            simulated.shape
        ),
        simulated=simulated, angles=deformation_case["angles"], config=SMALL,
    )
    assert verdict.recommendation is Recommendation.ACCEPT_RIGID
    evidence = verdict.as_rigid_evidence(FLAT_HISTORY, CONVERGED_SHIFTS)
    assert not bool(nonrigid.nonrigid_is_warranted(evidence))


@pytest.mark.skipif(not DIAGNOSTICS_AVAILABLE, reason="tktomo.diagnostics is not importable")
def test_a_yes_becomes_a_diagnostics_finding(deformation_case):
    from tktomo.diagnostics import FailureMode, ProbeStatus, TriageStage

    verdict = evaluate_gate(
        residual_history=FLAT_HISTORY, shift_history=CONVERGED_SHIFTS,
        measured=deformation_case["deformed"], simulated=deformation_case["simulated"],
        angles=deformation_case["angles"],
        acquisition_index=deformation_case["acquisition_index"], config=SMALL,
    )
    finding = verdict.to_finding()
    assert finding is not None
    assert finding.mode is FailureMode.DEFORMATION
    assert finding.spec.stage is TriageStage.NON_RIGID

    probe = verdict.to_probe_result()
    assert probe.status is ProbeStatus.FIRED
    assert probe.stage is TriageStage.NON_RIGID
    assert probe.findings == (finding,)
    # The whole thing has to survive the diagnostics serialiser too.
    assert json.loads(json.dumps(probe.to_dict()))["probe"] == "nonrigid_gate"


@pytest.mark.skipif(not DIAGNOSTICS_AVAILABLE, reason="tktomo.diagnostics is not importable")
def test_a_no_is_not_reported_as_a_deformation_finding():
    """An outcome that sends you upstream must not claim the non-rigid mode is present."""
    from tktomo.diagnostics import ProbeStatus

    verdict = GateVerdict(
        recommendation=Recommendation.FIX_UPSTREAM, confidence=0.7, headline="ramp",
        plateau=None, localisation=None, upstream=None,
    )
    assert verdict.to_finding() is None
    assert verdict.to_probe_result().status is ProbeStatus.NOT_APPLICABLE
    assert verdict.to_probe_result().findings == ()


def test_the_module_works_without_the_diagnostics_package(monkeypatch):
    """The fallback path, forced, so it is exercised even where diagnostics is installed."""
    import tktomo.ptycho_align.core.nonrigid_gate as gate_module

    monkeypatch.setattr(gate_module, "DIAGNOSTICS_AVAILABLE", False)
    verdict = GateVerdict(
        recommendation=Recommendation.RUN_NONRIGID, confidence=1.0, headline="",
        plateau=None, localisation=None, upstream=None,
    )
    assert verdict.to_finding() is None
    assert verdict.to_probe_result() is None
    assert json.loads(verdict.to_json())["recommendation"] == "run_nonrigid"

