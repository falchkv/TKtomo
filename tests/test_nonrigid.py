"""The definition of done for the non-rigid stage.

Two tests carry the weight, and they are the two halves of the same question:

* :func:`test_recovers_a_known_time_varying_deformation` -- apply a *known* smooth
  deformation that evolves over acquisition time, forward-project it, and check the
  recovered deformation field against the truth in pixels.
* :func:`test_negative_control_invents_no_deformation` -- run the identical pipeline
  on data with **no** deformation and check it does not manufacture one. A non-rigid
  method that has not been shown to stay quiet on rigid data is not safe to ship, and
  this is the test that shows it.

Everything runs on numpy + scipy: the parallel-beam projector below stands in for the
TomoPy backend so the suite needs neither a reconstruction stack nor beamtime data.
It is a real projector and a real FBP, not a mock -- a mock would let a sign error
through, which is the whole class of bug these tests exist to catch.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, rotate
from scipy.ndimage import shift as ndshift

from tktomo.ptycho_align.core.deformation import (
    DeformationField,
    DeformationSequence,
    coarse_support_mask,
    sequence_rms_difference,
    warp_volume,
)
from tktomo.ptycho_align.core.engine import Cancelled
from tktomo.ptycho_align.core.nonrigid import (
    LocalisationReport,
    NonRigidAligner,
    NonRigidConfig,
    NonRigidResult,
    RigidAlignmentRequired,
    RigidEvidence,
    angular_gap_deg,
    nonrigid_is_warranted,
    residual_localisation,
    time_subsets,
)


# ---------------------------------------------------------------------------------
# A scipy-only parallel-beam backend, conforming to ReconBackend
# ---------------------------------------------------------------------------------


class NumpyParallelBackend:
    """Parallel-beam projector + FBP in scipy, matching the ReconBackend protocol.

    Volume ``(nz, ny, nx)``, projections ``(n_angles, nz, nx)`` -- TomoPy's layout, so
    the aligner cannot tell it apart from the real backend. Projection at angle theta
    rotates the volume about the z axis and integrates along y; backprojection rotates
    the other way, which makes the pair adjoint up to interpolation.

    Its data-consistency floor is about 0.13 (reprojecting its own reconstruction),
    coming from the ramp filter's band limit and the linear interpolation. That floor
    is common to every residual the aligner reports, so it cancels in the comparisons
    that matter, but it is why the absolute residuals here are not near zero.
    """

    name = "numpy_parallel_test"

    def __init__(self, order: int = 1) -> None:
        self.order = order

    @staticmethod
    def _shift_for(center: float | None, nx: int) -> float:
        return 0.0 if center is None else float(center) - nx / 2.0

    def reproject(self, volume, angles, *, center=None, **_kwargs) -> np.ndarray:
        volume = np.asarray(volume, dtype=np.float32)
        nz, _ny, nx = volume.shape
        angles = np.asarray(angles, dtype=np.float64)
        out = np.empty((angles.size, nz, nx), dtype=np.float32)
        for i, theta in enumerate(angles):
            rotated = rotate(
                volume,
                np.degrees(theta),
                axes=(1, 2),
                reshape=False,
                order=self.order,
                mode="constant",
                cval=0.0,
            )
            out[i] = rotated.sum(axis=1)
        delta = self._shift_for(center, nx)
        if abs(delta) > 1e-9:
            out = ndshift(out, (0, 0, delta), order=1, mode="constant", cval=0.0)
        return out

    def reconstruct(self, projections, angles, *, center=None, algorithm="fbp", **_kwargs):
        projections = np.asarray(projections, dtype=np.float32)
        n_angles, nz, nx = projections.shape
        delta = self._shift_for(center, nx)
        if abs(delta) > 1e-9:
            projections = ndshift(projections, (0, 0, -delta), order=1, mode="constant")

        pad = int(2 ** np.ceil(np.log2(max(64, 2 * nx))))
        frequency = np.fft.rfftfreq(pad)
        ramp = 2.0 * frequency * np.sinc(frequency)  # Shepp-Logan window
        padded = np.zeros((n_angles, nz, pad), dtype=np.float32)
        padded[:, :, :nx] = projections
        filtered = np.fft.irfft(np.fft.rfft(padded, axis=2) * ramp, n=pad, axis=2)[:, :, :nx]

        volume = np.zeros((nz, nx, nx), dtype=np.float32)
        for i, theta in enumerate(np.asarray(angles, dtype=np.float64)):
            smeared = np.repeat(filtered[i][:, None, :], nx, axis=1)
            volume += rotate(
                smeared,
                -np.degrees(theta),
                axes=(1, 2),
                reshape=False,
                order=self.order,
                mode="constant",
                cval=0.0,
            )
        volume *= np.pi / (2.0 * n_angles)
        return volume.astype(np.float32)


# ---------------------------------------------------------------------------------
# A phantom that deforms over acquisition time
# ---------------------------------------------------------------------------------

NZ, NX, N_SUBTOMOS, ANGLES_PER = 18, 32, 4, 30
SHAPE = (NZ, NX, NX)
SPACING = 10.0
GRID = DeformationField.grid_for(SHAPE, SPACING)


def phantom_volume(nz: int = NZ, nx: int = NX) -> np.ndarray:
    """An ellipsoid with off-centre inclusions, smoothed to be band-limited."""
    z, y, x = np.mgrid[0:nz, 0:nx, 0:nx].astype(np.float32)
    cz, cy, cx = (nz - 1) / 2, (nx - 1) / 2, (nx - 1) / 2
    volume = np.zeros((nz, nx, nx), dtype=np.float32)
    body = ((z - cz) / (0.42 * nz)) ** 2 + ((y - cy) / (0.34 * nx)) ** 2 + (
        (x - cx) / (0.30 * nx)
    ) ** 2
    volume[body < 1.0] = 1.0
    for (fz, fy, fx), radius, amplitude in (
        ((0.35, 0.40, 0.42), 0.09, 0.9),
        ((0.62, 0.58, 0.60), 0.07, -0.6),
        ((0.50, 0.62, 0.38), 0.06, 0.8),
        ((0.45, 0.45, 0.60), 0.05, -0.5),
    ):
        inside = ((z - fz * nz) ** 2 + (y - fy * nx) ** 2 + (x - fx * nx) ** 2) < (
            radius * nx
        ) ** 2
        volume[inside] += amplitude
    return gaussian_filter(volume, 0.8)


def truth_field(t: float, amplitude: float = 2.0) -> DeformationField:
    """The known deformation at normalised acquisition time ``t`` in [0, 1].

    Two spatial modes with two different, smooth time profiles: a z-dependent axial
    displacement that drifts linearly (creep), and an in-plane shear that reverses
    (thermal). Deliberately not separable in space-time, so a method that can only
    represent "one shape, scaled by time" would be caught.

    The in-plane mode is a **pure shear strain**, not a rotation. That is not
    cosmetic: an in-plane rotation of the sample is nearly degenerate with a change in
    the assigned projection angles, so it is the least identifiable deformation mode
    there is -- measured at correlation 0.47 against 0.56 for the same amplitude of
    shear on this phantom. A test article built on the degenerate mode would understate
    what the method can do, and hide which modes it cannot recover.
    """
    gz, gy, gx = GRID
    ones = np.ones(GRID, dtype=np.float32)
    z = np.linspace(0.0, 1.0, gz, dtype=np.float32)[:, None, None] * ones
    y = np.linspace(-1.0, 1.0, gy, dtype=np.float32)[None, :, None] * ones
    x = np.linspace(-1.0, 1.0, gx, dtype=np.float32)[None, None, :] * ones
    drift = (t - 0.5) * 2.0
    shear = float(np.cos(np.pi * t))
    vectors = np.stack(
        [drift * np.sin(np.pi * z), shear * 0.7 * x, shear * 0.7 * y]
    ).astype(np.float32)
    return DeformationField(vectors * amplitude, SHAPE)


def deforming_scan(*, deform: bool = True, amplitude: float = 2.0):
    """A series of interlaced sub-tomograms acquired one after another.

    This is the acquisition geometry the method needs and the one a P06 series
    actually has: the sample is scanned through 0-180 several times over many hours,
    so a *contiguous block of acquisition time* is one whole sub-tomogram -- angularly
    complete, and a single deformation state. Time order is emphatically not angle
    order here: angle sweeps 0-180 four times while time runs monotonically.
    """
    volume = phantom_volume()
    backend = NumpyParallelBackend()
    projections, angles, acquisition = [], [], []
    for k in range(N_SUBTOMOS):
        t = k / (N_SUBTOMOS - 1)
        state = warp_volume(volume, truth_field(t, amplitude), order=3) if deform else volume
        theta = np.linspace(0.0, np.pi, ANGLES_PER, endpoint=False) + k * np.pi / (
            ANGLES_PER * N_SUBTOMOS
        )
        projections.append(backend.reproject(state, theta))
        angles.append(theta)
        acquisition.append(np.arange(ANGLES_PER) + k * ANGLES_PER)
    return {
        "projections": np.concatenate(projections).astype(np.float32),
        "angles": np.concatenate(angles),
        "acquisition_index": np.concatenate(acquisition),
        "volume": volume,
        "backend": backend,
        "center": NX / 2.0,
        "amplitude": amplitude,
    }


def default_config(**overrides) -> NonRigidConfig:
    """The library defaults, with only the scan-shape parameters filled in.

    Deliberately does not tune the flow: these tests judge what a user gets out of
    the box, so a regression in a default is a test failure rather than something
    hidden behind a bespoke configuration.
    """
    base = dict(
        n_subsets=N_SUBTOMOS,
        grid_spacing=SPACING,
        recon_algorithm="fbp",
        # 30 deg rather than the 20 deg default: holding out 15% of a 30-angle
        # sub-tomogram punches random holes in it, and a small test scan feels that
        # much more than a real one with hundreds of projections per sub-tomogram.
        max_angular_gap_deg=30.0,
    )
    base.update(overrides)
    return NonRigidConfig(**base)


def truth_sequence_for(times: np.ndarray, amplitude: float) -> DeformationSequence:
    """The ground truth sampled at the subsets' own mean acquisition times."""
    span = N_SUBTOMOS * ANGLES_PER - 1
    return DeformationSequence(
        tuple(truth_field(float(t) / span, amplitude) for t in times), np.asarray(times)
    )


def run_case(deform: bool, iterations: int = 3, **overrides):
    """Drive the aligner exactly as an unattended caller would, through ``run``.

    Through ``run`` and not a bare loop of ``step`` because ``run`` is where the
    stopping rule lives, and on rigid data the stopping rule is the whole point: the
    refinement accumulates whatever the flow finds, so a caller who ignores the monitor
    and keeps stepping will watch an invented field grow.
    """
    scan = deforming_scan(deform=deform)
    aligner = NonRigidAligner(
        projections=scan["projections"],
        angles=scan["angles"],
        acquisition_index=scan["acquisition_index"],
        center=scan["center"],
        config=default_config(**overrides),
        backend=scan["backend"],
    )
    results = aligner.run(iterations)
    return scan, aligner, results


@pytest.fixture(scope="module")
def deformed_case():
    return run_case(deform=True)


@pytest.fixture(scope="module")
def control_case():
    return run_case(deform=False, iterations=6)


# ---------------------------------------------------------------------------------
# Acquisition-time subsets and their angular coverage
# ---------------------------------------------------------------------------------


def test_time_subsets_follow_acquisition_time_not_angle_order():
    """Angle order and time order differ here, and using the wrong one is silent."""
    scan = deforming_scan(deform=False)
    subsets = time_subsets(scan["acquisition_index"], N_SUBTOMOS)
    for k, subset in enumerate(subsets):
        expected = set(range(k * ANGLES_PER, (k + 1) * ANGLES_PER))
        assert set(subset.tolist()) == expected, "a time block must be one sub-tomogram"
    by_angle = np.argsort(scan["angles"])[:ANGLES_PER]
    assert set(by_angle.tolist()) != set(subsets[0].tolist()), (
        "the test scan must actually distinguish time order from angle order"
    )


def test_each_time_block_is_angularly_complete():
    scan = deforming_scan(deform=False)
    for subset in time_subsets(scan["acquisition_index"], N_SUBTOMOS):
        assert angular_gap_deg(scan["angles"][subset]) < 15.0


def test_interleaved_mode_also_spans_the_angles_but_not_the_time():
    scan = deforming_scan(deform=False)
    subsets = time_subsets(scan["acquisition_index"], N_SUBTOMOS, mode="interleaved")
    for subset in subsets:
        covered = np.mod(np.degrees(scan["angles"][subset]), 180.0)
        assert np.ptp(covered) > 170.0, "an interleaved subset still spans the full range"
        spread = np.ptp(scan["acquisition_index"][subset])
        assert spread > 0.9 * (N_SUBTOMOS * ANGLES_PER - 1), (
            "an interleaved subset spans the whole scan, which is exactly why it "
            "cannot resolve a deformation that evolves within it"
        )


def test_angular_gap_finds_a_missing_wedge():
    full = np.linspace(0.0, np.pi, 60, endpoint=False)
    assert angular_gap_deg(full) < 5.0
    wedge = np.linspace(0.0, np.pi * 0.6, 40)
    assert angular_gap_deg(wedge) > 60.0


def test_aligner_refuses_subsets_with_a_missing_wedge():
    """A sequential single sweep split by time is angular wedges; refuse, do not fudge."""
    volume = phantom_volume()
    backend = NumpyParallelBackend()
    angles = np.linspace(0.0, np.pi, 80, endpoint=False)
    projections = backend.reproject(volume, angles)
    with pytest.raises(RigidAlignmentRequired, match="missing wedge|angular gap"):
        NonRigidAligner(
            projections=projections,
            angles=angles,
            acquisition_index=np.arange(80),  # a single monotonic sweep
            center=NX / 2.0,
            config=default_config(n_subsets=4),
            backend=backend,
        )


def test_acquisition_index_is_required_and_checked():
    scan = deforming_scan(deform=False)
    with pytest.raises(ValueError, match="acquisition indices"):
        NonRigidAligner(
            projections=scan["projections"],
            angles=scan["angles"],
            acquisition_index=np.arange(3),
            center=scan["center"],
            config=default_config(),
            backend=scan["backend"],
        )


# ---------------------------------------------------------------------------------
# Preconditions: rigid first, always
# ---------------------------------------------------------------------------------


def test_aligner_refuses_visibly_unaligned_data():
    """Non-rigid on top of an unfixed rigid error is the mistake the roadmap forbids."""
    scan = deforming_scan(deform=False)
    rng = np.random.default_rng(0)
    n = scan["projections"].shape[0]
    jittered = np.stack(
        [
            ndshift(frame, rng.uniform(-5, 5, 2), order=1, mode="constant")
            for frame in scan["projections"]
        ]
    )
    with pytest.raises(RigidAlignmentRequired, match="do not look rigidly aligned"):
        NonRigidAligner(
            projections=jittered,
            angles=scan["angles"],
            acquisition_index=scan["acquisition_index"],
            center=scan["center"],
            config=default_config(),
            backend=scan["backend"],
        )


def test_the_refusal_can_be_overridden_only_deliberately():
    scan = deforming_scan(deform=False)
    rng = np.random.default_rng(0)
    jittered = np.stack(
        [
            ndshift(frame, rng.uniform(-5, 5, 2), order=1, mode="constant")
            for frame in scan["projections"]
        ]
    )
    aligner = NonRigidAligner(
        projections=jittered,
        angles=scan["angles"],
        acquisition_index=scan["acquisition_index"],
        center=scan["center"],
        config=default_config(require_rigid=False),
        backend=scan["backend"],
    )
    assert aligner.iteration == 0


def test_rigid_evidence_needs_an_engine_that_has_actually_run():
    class EmptyEngine:
        history: list = []

    with pytest.raises(RigidAlignmentRequired, match="no iterations"):
        RigidEvidence.from_engine(EmptyEngine())


# ---------------------------------------------------------------------------------
# The evidential gate
# ---------------------------------------------------------------------------------


def test_localisation_separates_a_localised_residual_from_a_spread_one():
    rng = np.random.default_rng(1)
    shape = (12, 96, 96)
    spread = rng.normal(0, 1, shape)
    localised = rng.normal(0, 0.05, shape)
    localised[:, 30:50, 60:80] += rng.normal(0, 3, (shape[0], 20, 20))

    spread_report = residual_localisation(spread, block=16)
    localised_report = residual_localisation(localised, block=16)
    assert not spread_report.is_localised
    assert localised_report.is_localised
    assert localised_report.concentration > 3 * spread_report.concentration
    assert localised_report.angle_consistency > spread_report.angle_consistency + 0.3


def test_localisation_needs_enough_blocks_to_mean_anything():
    with pytest.raises(ValueError, match="fewer than 2x2"):
        residual_localisation(np.zeros((4, 20, 20)), block=16)


def test_nonrigid_is_not_warranted_while_the_rigid_residual_is_still_falling():
    evidence = RigidEvidence(
        residuals=np.array([1.0, 0.8, 0.6, 0.4, 0.25]),
        shift_rms=np.array([2.0, 1.0, 0.5, 0.3, 0.2]),
        localisation=LocalisationReport(0.6, 0.8, 16, True),
    )
    verdict = nonrigid_is_warranted(evidence)
    assert not verdict
    assert any("still falling" in r for r in verdict.reasons)


def test_nonrigid_is_not_warranted_without_localisation_evidence():
    evidence = RigidEvidence(
        residuals=np.array([0.30, 0.299, 0.2985, 0.2984, 0.2984]),
        shift_rms=np.array([0.5, 0.2, 0.05, 0.02, 0.01]),
    )
    verdict = nonrigid_is_warranted(evidence)
    assert not verdict
    assert any("localisation report" in r for r in verdict.reasons)


def test_nonrigid_is_not_warranted_when_the_residual_is_spread_and_angle_random():
    evidence = RigidEvidence(
        residuals=np.array([0.30, 0.299, 0.2985, 0.2984, 0.2984]),
        shift_rms=np.array([0.5, 0.2, 0.05, 0.02, 0.01]),
        localisation=LocalisationReport(0.12, 0.02, 16, False),
    )
    verdict = nonrigid_is_warranted(evidence)
    assert not verdict
    assert any("jitter or noise" in r for r in verdict.reasons)


def test_nonrigid_is_warranted_on_a_plateaued_localised_residual():
    evidence = RigidEvidence(
        residuals=np.array([0.30, 0.299, 0.2985, 0.2984, 0.2984]),
        shift_rms=np.array([0.5, 0.2, 0.05, 0.02, 0.01]),
        localisation=LocalisationReport(0.55, 0.7, 16, True),
    )
    assert nonrigid_is_warranted(evidence)


# ---------------------------------------------------------------------------------
# Held-out projections
# ---------------------------------------------------------------------------------


def test_held_out_projections_never_enter_a_subset(deformed_case):
    _scan, aligner, _results = deformed_case
    fitted = np.concatenate(aligner.subsets)
    assert aligner.holdout.size > 0
    assert not set(aligner.holdout.tolist()) & set(fitted.tolist()), (
        "a held-out projection that was also fitted measures nothing"
    )
    assert sorted(fitted.tolist() + aligner.holdout.tolist()) == list(
        range(aligner.projections.shape[0])
    )


def test_a_tiny_holdout_is_refused_rather_than_being_useless():
    scan = deforming_scan(deform=False)
    with pytest.raises(ValueError, match="cannot measure anything"):
        NonRigidAligner(
            projections=scan["projections"],
            angles=scan["angles"],
            acquisition_index=scan["acquisition_index"],
            center=scan["center"],
            config=default_config(holdout_fraction=0.01),
            backend=scan["backend"],
        )


def test_overfitting_is_flagged_when_the_held_out_residual_worsens():
    """Unit test of the verdict itself, so the rule is pinned independently of a run."""
    aligner = object.__new__(NonRigidAligner)
    aligner.config = default_config()
    aligner._holdout = np.arange(10)
    aligner._history = []  # no previous iteration, so only the baseline rules apply
    worse = NonRigidResult(
        iteration=1,
        sequence=DeformationSequence((DeformationField.zeros(SHAPE, GRID),), np.zeros(1)),
        residual=0.10,
        holdout_residual=0.30,
        baseline_residual=0.20,
        baseline_holdout_residual=0.20,
        dvf_rms_px=1.0,
        dvf_max_px=2.0,
        wallclock_s=0.0,
    )
    assert "got WORSE" in (NonRigidAligner._overfitting_reason(aligner, worse) or "")

    lopsided = NonRigidResult(
        iteration=1,
        sequence=worse.sequence,
        residual=0.10,
        holdout_residual=0.199,
        baseline_residual=0.20,
        baseline_holdout_residual=0.20,
        dvf_rms_px=1.0,
        dvf_max_px=2.0,
        wallclock_s=0.0,
    )
    reason = NonRigidAligner._overfitting_reason(aligner, lopsided) or ""
    assert "absorbing what it was fitted to" in reason


# ---------------------------------------------------------------------------------
# Stepping contract
# ---------------------------------------------------------------------------------


def test_step_runs_exactly_one_iteration_and_records_it(deformed_case):
    _scan, aligner, results = deformed_case
    assert aligner.iteration == len(aligner.history) == 3
    assert [r.iteration for r in results] == [1, 2, 3]
    assert aligner.sequence is not None
    assert aligner.reference_volume is not None
    assert aligner.reference_volume.shape == SHAPE


def test_step_honours_a_cancel_and_records_nothing():
    class AlwaysCancelled:
        @staticmethod
        def is_set() -> bool:
            return True

    scan = deforming_scan(deform=False)
    aligner = NonRigidAligner(
        projections=scan["projections"],
        angles=scan["angles"],
        acquisition_index=scan["acquisition_index"],
        center=scan["center"],
        config=default_config(),
        backend=scan["backend"],
    )
    with pytest.raises(Cancelled):
        aligner.step(cancel=AlwaysCancelled())
    assert aligner.iteration == 0 and not aligner.history


def test_config_survives_a_round_trip_and_inherits_the_rigid_settings():
    config = default_config(ncore=3)
    assert NonRigidConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="Unknown NonRigidConfig field"):
        NonRigidConfig.from_dict({"nonsense": 1})

    from tktomo.ptycho_align.core.engine import AlignConfig  # noqa: PLC0415

    inherited = NonRigidConfig.from_align_config(
        AlignConfig(recon_algorithm="sirt", backend="tomopy", ncore=4)
    )
    assert (inherited.recon_algorithm, inherited.backend, inherited.ncore) == (
        "sirt",
        "tomopy",
        4,
    )


# ---------------------------------------------------------------------------------
# THE TWO GATES
# ---------------------------------------------------------------------------------


def test_recovers_a_known_time_varying_deformation(deformed_case):
    """Recover a known DVF sequence to better than a pixel inside the object.

    Compared after gauge fixing (:meth:`DeformationSequence.zero_mean`) because the
    time-averaged deformation is unobservable -- it is absorbed into the reference
    volume -- and restricted to the object support because optical flow in empty air
    is the smoothness prior talking, not the data. Odstrcil et al. report ~0.8 px rms
    against simulated ground truth; this is that measurement, on a 32-voxel phantom
    whose deformation is ~1.2 px rms after gauge fixing.
    """
    scan, aligner, results = deformed_case
    result = results[-1]
    truth = truth_sequence_for(aligner.sequence.times, scan["amplitude"])
    support = coarse_support_mask(scan["volume"], GRID, threshold=0.05, sigma=2.0)

    inside = sequence_rms_difference(aligner.sequence, truth, mask=support)
    everywhere = sequence_rms_difference(aligner.sequence, truth)
    assert inside < 0.9, f"recovered DVF is {inside:.3f} px rms from the truth in the object"
    assert everywhere < 1.5, f"{everywhere:.3f} px rms over the whole grid"

    # The recovered field must point the same way as the truth, not merely be small.
    # The correlation bar is 0.4, not 0.9, and that is an honest statement about the
    # method rather than a slack test: the smoothness prior that keeps the negative
    # control quiet also shrinks the recovered amplitude to roughly a third of the
    # truth, so the field is the right shape at the wrong scale. Read the DVF as
    # "where and in what direction the sample moved", not as a calibrated strain.
    a = aligner.sequence.zero_mean().node_array[:, :, support]
    b = truth.zero_mean().node_array[:, :, support]
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.4

    # And it must buy real predictive power on projections it never fitted.
    assert result.holdout_gain > 0.05, (
        f"held-out residual improved by only {result.holdout_gain:.1%}; a deformation "
        "this method cannot predict out of sample is not a deformation it measured"
    )
    assert result.overfitting is None


def test_negative_control_invents_no_deformation(control_case, deformed_case):
    """THE SAFETY GATE. On rigid data the method must stay quiet.

    Two things are asserted, and the second matters more than the first: the spurious
    field is small in absolute terms *and* small compared with the one recovered from
    genuinely deformed data, and -- decisively -- it buys no improvement on held-out
    projections, so the overfitting monitor rejects it. A method that quietly returns
    half a pixel of invented deformation and no way to tell is the failure this test
    exists to prevent.
    """
    _scan, control, control_results = control_case
    _dscan, deformed, deformed_results = deformed_case
    control_result, deformed_result = control_results[-1], deformed_results[-1]

    assert len(control_results) < 6, (
        "run() must stop once the refinement stops earning its parameters; letting it "
        "continue on rigid data is how an invented field grows"
    )
    assert control_result.overfitting is not None
    # The flagged iteration is recorded, not discarded, so the caller can revert.
    last_good = control.revert_to(control_results[-2].iteration)
    assert last_good.overfitting is None

    spurious_rms = last_good.sequence.zero_mean().rms_magnitude
    real_rms = deformed.sequence.zero_mean().rms_magnitude
    assert spurious_rms < 0.35, f"invented {spurious_rms:.3f} px rms of deformation from nothing"
    assert spurious_rms < 0.5 * real_rms, (
        f"the spurious field ({spurious_rms:.3f} px) is not clearly smaller than the real "
        f"one ({real_rms:.3f} px)"
    )
    assert last_good.holdout_gain < 0.02, (
        f"the invented deformation appears to help held-out projections by "
        f"{last_good.holdout_gain:.1%}, which it must not"
    )
    assert deformed_result.holdout_gain > 5 * max(last_good.holdout_gain, 1e-3), (
        "real deformation must buy far more out-of-sample predictive power than the "
        "residue the method finds in rigid data"
    )


def test_an_under_regularised_configuration_is_caught_by_the_guard():
    """The defaults stay quiet; this shows what happens when they are loosened, and
    that the held-out monitor is what stands between the user and a fiction.

    Weak smoothness, no prefilter, no temporal smoothing, a fine grid and a loose
    magnitude cap: on data with **no** deformation at all this manufactures a couple of
    pixels of it, and improves nothing -- the held-out residual gets *worse*, which is
    exactly the signature the monitor exists to detect. Without the held-out set the
    fitted residual alone would not have given this away.
    """
    scan = deforming_scan(deform=False)
    aligner = NonRigidAligner(
        projections=scan["projections"],
        angles=scan["angles"],
        acquisition_index=scan["acquisition_index"],
        center=scan["center"],
        config=default_config(
            grid_spacing=6.0,
            flow_alpha=0.2,
            flow_iterations=100,
            flow_warps=6,
            flow_prefilter_sigma=0.0,
            time_sigma=0.0,
            max_dvf_px=20.0,
        ),
        backend=scan["backend"],
    )
    results = aligner.run(3)
    invented = aligner.sequence.zero_mean().rms_magnitude
    assert invented > 0.5, (
        f"the loosened configuration was expected to overfit, got {invented:.3f} px"
    )
    assert results[-1].overfitting is not None, "the held-out monitor failed to notice"
    assert results[-1].holdout_gain < 0, "the invented field must not help held-out data"
    assert len(results) == 1, "run() must stop at the first overfitting iteration"
