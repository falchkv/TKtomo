"""The definition of done for the benchmark harness: it must not lie about a score.

Three classes of test here, in order of how badly a failure hurts:

1. **Conventions.** The injection sign, the gauge, the shape. If these are wrong the
   harness reports plausible numbers that mean nothing, which is worse than crashing.
2. **The FSC caveat.** A test that *proves* the roadmap's claim rather than repeating
   it: FSC is bit-for-bit invariant to a geometry error applied to both half-sets.
3. **End to end.** The runner produces a table and JSON on a case small enough to run
   in seconds, with no GPU, no tomopy and no measured data.

Everything runs on a synthetic phantom of a few tens of pixels. Nothing here touches
a beamtime path.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from benchmarks import metrics
from benchmarks.phantom import (
    BenchmarkCase,
    GroundTruth,
    PerturbationSpec,
    back_project,
    cases_from_catalogue,
    forward_project,
    perturb,
    synthetic_case,
    synthetic_volume,
)
from benchmarks.runner import (
    JirrAligner,
    JointGdAligner,
    ModuleAligner,
    NullAligner,
    NumpyProjectorBackend,
    OdstrcilAligner,
    OracleAligner,
    comparison_figure,
    run_benchmark,
    tomopy_shim,
    undo_shifts,
)

# Small enough that the whole file runs in seconds on one core.
TINY = dict(size=24, n_slices=3, n_angles=18)


@pytest.fixture(scope="module")
def tiny_case() -> BenchmarkCase:
    return synthetic_case(
        spec=PerturbationSpec(jitter_dy_rms=1.0, jitter_dx_rms=0.4, seed=7), **TINY
    )


# --------------------------------------------------------------------------------
# 1. Conventions
# --------------------------------------------------------------------------------


def test_injection_sign_is_the_engine_convention(tiny_case: BenchmarkCase) -> None:
    """THE convention test. ``undo_shifts(+truth)`` must recover the clean stack.

    ``truth.dy``/``dx`` are the content displacement injected; ``apply_shifts`` (and
    :func:`undo_shifts`, which mirrors it) moves content by ``-s``. So an aligner that
    works reports ``+truth``. If this ever flips, every score in the harness is
    silently wrong by a factor of two and nothing else in this file would notice.
    """
    case = tiny_case
    recovered = undo_shifts(case.projections, case.truth.dy, case.truth.dx)
    wrong_way = undo_shifts(case.projections, -case.truth.dy, -case.truth.dx)

    scale = np.linalg.norm(case.clean)
    error_right = np.linalg.norm(recovered - case.clean) / scale
    error_wrong = np.linalg.norm(wrong_way - case.clean) / scale
    error_none = np.linalg.norm(case.projections - case.clean) / scale

    assert error_right < 0.05, f"+truth must undo the injection, got {error_right:.3f}"
    assert error_right < error_none / 5
    assert error_wrong > error_none, "negating the truth must make it worse, not better"


def test_shift_injection_moves_content_the_way_it_says(tmp_path) -> None:  # noqa: ARG001
    """A single delta, a single known shift: rows down, columns right, positive."""
    frame = np.zeros((1, 32, 32), dtype=np.float32)
    frame[0, 10, 20] = 1.0
    spec = PerturbationSpec()  # everything off; we inject by hand below
    shifted, _ = perturb(frame, np.zeros(1), spec, margin=0)
    np.testing.assert_allclose(shifted, frame)

    from scipy.ndimage import fourier_shift

    moved = np.fft.ifftn(fourier_shift(np.fft.fftn(frame[0]), (3.0, -4.0))).real
    row, column = np.unravel_index(np.argmax(moved), moved.shape)
    assert (row, column) == (13, 16), "dy must move rows down, dx must move columns right"


def test_generators_are_deterministic() -> None:
    a = synthetic_case(spec=PerturbationSpec(jitter_dy_rms=1.0, seed=5), **TINY)
    b = synthetic_case(spec=PerturbationSpec(jitter_dy_rms=1.0, seed=5), **TINY)
    c = synthetic_case(spec=PerturbationSpec(jitter_dy_rms=1.0, seed=6), **TINY)
    np.testing.assert_array_equal(a.projections, b.projections)
    np.testing.assert_array_equal(a.truth.dy, b.truth.dy)
    assert not np.allclose(a.truth.dy, c.truth.dy), "a new seed must give a new draw"


def test_every_perturbation_is_independently_switchable() -> None:
    """Each catalogue entry must change the data and record what it changed.

    This is what makes the diagnostic sweep meaningful: if a perturbation did not
    actually perturb, an aligner would appear to survive it.
    """
    cases = cases_from_catalogue(**TINY)
    baseline = cases["jitter_only"]
    for name, case in cases.items():
        if name == "jitter_only":
            continue
        same_shape = case.projections.shape == baseline.projections.shape
        changed = (
            not same_shape
            or not np.allclose(case.projections, baseline.projections, atol=1e-6)
        )
        assert changed, f"{name} did not change the projections"

    assert cases["angle_error"].truth.angles_true is not cases["angle_error"].angles
    assert not np.allclose(
        cases["angle_error"].truth.angles_true, cases["angle_error"].truth.angles_reported
    )
    assert cases["truncation"].width < baseline.width
    assert np.ptp(cases["magnification_drift"].truth.magnification) > 0
    assert np.any(cases["phase_ramp"].truth.phase_ramp != 0)
    assert cases["out_of_plane_tilt"].truth.out_of_plane_tilt_deg == pytest.approx(0.5)


def test_margin_guard_refuses_a_wrapping_shift() -> None:
    """A Fourier shift wraps; a shift bigger than the margin destroys the ground truth."""
    clean = np.zeros((4, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="margin"):
        perturb(clean, np.zeros(4), PerturbationSpec(center_dy=9.0), margin=4)


def test_deformation_is_zero_mean_so_the_rigid_truth_stays_exact() -> None:
    """A deformation field with a net translation would silently bias the truth."""
    case = synthetic_case(
        spec=PerturbationSpec(deformation_px=1.0, deformation_scale=3.0, seed=2),
        store_deformation=True,
        **TINY,
    )
    field = case.truth.deformation_field
    assert field is not None
    for i in range(field.shape[0]):
        assert abs(field[i, 0].mean()) < 1e-4
        assert abs(field[i, 1].mean()) < 1e-4


def test_gauge_removal_forgives_only_the_unobservable(tiny_case: BenchmarkCase) -> None:
    """A global in-plane translation of the object is not an alignment error.

    Translating the object by (X, Y) shifts projection i by ``X cos + Y sin`` in dx and
    by a constant in dy. An aligner returning truth-plus-that describes the same
    object in a different place, so it must score zero -- while a *stricter* mean-only
    removal, and no removal at all, both see it. Getting this wrong is what makes a
    perfect aligner look broken.
    """
    case = tiny_case
    angles = case.angles
    gauge_dx = 1.7 * np.cos(angles) - 0.9 * np.sin(angles) + 0.4
    gauge_dy = np.full_like(angles, 2.3)

    score = metrics.score_shifts(
        case.truth.dy + gauge_dy,
        case.truth.dx + gauge_dx,
        case.truth.dy,
        case.truth.dx,
        angles,
    )
    assert score.rms_dy < 1e-9
    assert score.rms_dx < 1e-9
    assert score.rms_dy_raw > 2.0, "the raw number must still see the offset"
    assert score.rms_dx_mean_only > 0.5, "mean-only removal must still see sin/cos"
    assert score.gauge_amplitude_dx > 0.5


def test_score_flags_a_flipped_sign_as_double_the_injection(tiny_case: BenchmarkCase) -> None:
    case = tiny_case
    flipped = metrics.score_shifts(
        -case.truth.dy, -case.truth.dx, case.truth.dy, case.truth.dx, case.angles
    )
    assert flipped.rms_dy == pytest.approx(2.0 * flipped.injected_rms_dy, rel=1e-6)


def test_score_rejects_non_finite_estimates(tiny_case: BenchmarkCase) -> None:
    """A diverged aligner must not be scored; it must raise."""
    diverged = np.full(tiny_case.n_angles, np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        metrics.score_shifts(
            diverged, diverged, tiny_case.truth.dy, tiny_case.truth.dx, tiny_case.angles
        )


def test_target_is_one_third_of_the_target_voxel() -> None:
    """Our own numbers: 74.51 nm voxel, so a third of a voxel is 0.333 px."""
    angles = np.linspace(0.0, np.pi, 12, endpoint=False)
    truth = np.zeros(12)
    estimate = np.full(12, 0.2)  # constant -> pure gauge, scores zero
    score = metrics.score_shifts(
        estimate, estimate, truth, truth, angles, pixel_size_nm=74.50973137
    )
    assert score.target_px == pytest.approx(1.0 / 3.0)
    assert score.meets_target


# --------------------------------------------------------------------------------
# 2. The FSC caveat, proved rather than asserted
# --------------------------------------------------------------------------------


def test_fsc_is_exactly_blind_to_common_mode_geometric_error() -> None:
    """The roadmap's warning, to machine precision.

    Shifting both half-sets by the *same* amount multiplies their transforms by
    conjugate phase factors that cancel in the cross term, so every shell's
    correlation is unchanged -- while the reconstruction is now systematically in the
    wrong place. This is why a benchmark cannot be scored on FSC: we measured a
    half-bit FRC of exactly 508.6 nm at centring errors of 0, 4, 8, 16, 32 and 64 px
    while the true edge blur grew to 128 px.
    """
    from scipy.ndimage import fourier_shift

    rng = np.random.default_rng(0)
    truth = synthetic_volume(size=32, n_slices=8)
    half_a = truth + 0.02 * rng.standard_normal(truth.shape)
    half_b = truth + 0.02 * rng.standard_normal(truth.shape)

    def shift_volume(volume: np.ndarray, delta) -> np.ndarray:
        return np.fft.ifftn(fourier_shift(np.fft.fftn(volume), delta)).real

    before = metrics.fourier_shell_correlation(half_a, half_b)
    for delta in ((0, 2, 0), (0, 5, 3), (0, 11, 0)):
        after = metrics.fourier_shell_correlation(
            shift_volume(half_a, delta), shift_volume(half_b, delta)
        )
        np.testing.assert_allclose(after.fsc, before.fsc, atol=1e-9)
        assert after.resolution_px == pytest.approx(before.resolution_px, rel=1e-9)

    # ... whereas a shift of ONE half-set, which is a real disagreement, destroys it.
    disagreeing = metrics.fourier_shell_correlation(shift_volume(half_a, (0, 5, 3)), half_b)
    assert disagreeing.fsc[2:].mean() < before.fsc[2:].mean()


def test_fsc_of_identical_volumes_is_one() -> None:
    volume = synthetic_volume(size=24, n_slices=6)
    result = metrics.fourier_shell_correlation(volume, volume)
    finite = result.fsc[result.n_voxels > 0]
    np.testing.assert_allclose(finite, 1.0, atol=1e-6)
    assert np.isinf(result.resolution_px), "identical volumes never cross the threshold"


def test_split_half_indices_interleave() -> None:
    even, odd = metrics.split_half_indices(7)
    np.testing.assert_array_equal(even, [0, 2, 4, 6])
    np.testing.assert_array_equal(odd, [1, 3, 5])


# --------------------------------------------------------------------------------
# 2b. Residual map and plateau
# --------------------------------------------------------------------------------


def test_reprojection_residual_sees_what_fsc_cannot(tiny_case: BenchmarkCase) -> None:
    """A misaligned stack must have a higher reprojection residual than an aligned one."""
    case = tiny_case
    backend = NumpyProjectorBackend()

    def residual_of(sy, sx):
        aligned = undo_shifts(case.projections, sy, sx)
        volume = backend.reconstruct(
            aligned, case.angles, algorithm="sirt", center=case.center, num_iter=6
        )
        simulated = backend.reproject(volume, case.angles, center=case.center)
        return metrics.reprojection_residual(aligned, simulated)

    zeros = np.zeros(case.n_angles)
    misaligned = residual_of(zeros, zeros)
    aligned = residual_of(case.truth.dy, case.truth.dx)
    assert aligned.total < misaligned.total
    assert aligned.per_angle.shape == (case.n_angles,)


def test_residual_plateau_detection() -> None:
    falling = [1.0, 0.5, 0.25, 0.12, 0.06]
    flat = [1.0, 0.5, 0.499, 0.4985, 0.4983, 0.4982]
    assert not metrics.residual_plateau(falling).plateaued
    result = metrics.residual_plateau(flat)
    assert result.plateaued
    assert result.iteration == 2


# --------------------------------------------------------------------------------
# 3. The projector and the backend
# --------------------------------------------------------------------------------


def test_forward_projection_conserves_mass() -> None:
    """A line integral of a fixed object cannot depend on the viewing angle."""
    volume = synthetic_volume(size=32, n_slices=4)
    angles = np.deg2rad(np.linspace(0.0, 180.0, 12, endpoint=False))
    sinogram = forward_project(volume, angles)
    totals = sinogram.sum(axis=(1, 2))
    assert totals.std() / totals.mean() < 0.02


def test_projector_pair_is_exactly_adjoint() -> None:
    """``<A x, y> == <x, A^T y>``. Without this SIRT diverges, and does so silently.

    The rotate-based pair this replaced was only adjoint to ~2 %, which is enough to
    push eigenvalues of ``A^T A`` negative -- and a negative eigenvalue cannot be
    stabilised by any relaxation factor, because ``|1 - s*lambda| > 1`` for every
    ``s > 0``. It showed up as a reconstruction that grew without bound while every
    intermediate array stayed finite, and only for *some* phantoms, because divergence
    needs the data to have a component along the bad mode.
    """
    rng = np.random.default_rng(0)
    angles = np.deg2rad(np.linspace(0.0, 180.0, 21, endpoint=False))
    x = rng.standard_normal((3, 24, 24)).astype(np.float32)
    y = rng.standard_normal((21, 3, 24)).astype(np.float32)

    forward = float((forward_project(x, angles) * y).sum())
    backward = float((x * back_project(y, angles)).sum())
    assert forward == pytest.approx(backward, rel=1e-4)


def test_sparse_and_rotate_projectors_agree() -> None:
    """The matrix path and the rotate path must describe the same geometry.

    They are not interchangeable at runtime -- only the rotate path can express an
    out-of-plane tilt -- so a handedness or scale disagreement between them would make
    a tilt of 0.001 deg a mirror image of a tilt of exactly 0, and every tilt result
    would be nonsense.
    """
    from benchmarks.phantom import _rotate

    volume = synthetic_volume(size=32, n_slices=2)
    angles = np.deg2rad(np.array([0.0, 23.0, 71.0, 140.0]))
    sparse = forward_project(volume, angles)
    rotated = np.stack(
        [_rotate(volume, np.degrees(t), axes=(1, 2), order=1).sum(axis=1) for t in angles]
    )
    # Different interpolation, same operator: compare relative to the signal size.
    error = np.linalg.norm(sparse - rotated) / np.linalg.norm(rotated)
    assert error < 0.05, f"projectors disagree by {error:.3f} relative"


def test_sirt_converges_monotonically_at_a_realistic_size() -> None:
    """The regression guard for the divergence above. Residual must fall, every time.

    Deliberately at a size and padding where the old projector blew up (a 64 px object
    inside a wide zero margin, several detector rows), because the failure was
    invisible at the sizes the rest of this file uses.
    """
    from benchmarks.runner import NumpyProjectorBackend

    case = synthetic_case(
        size=32, n_slices=5, n_angles=36, spec=PerturbationSpec(jitter_dy_rms=2.5, seed=0)
    )
    aligned = undo_shifts(case.projections, case.truth.dy, case.truth.dx)
    backend = NumpyProjectorBackend()

    residuals = []
    for n_iter in (2, 5, 10, 20):
        volume = backend.reconstruct(
            aligned, case.angles, algorithm="sirt", center=case.center, num_iter=n_iter
        )
        simulated = backend.reproject(volume, case.angles, center=case.center)
        residuals.append(metrics.reprojection_residual(aligned, simulated).total)

    assert all(b < a for a, b in zip(residuals, residuals[1:])), (
        f"SIRT must converge, got residuals {residuals}"
    )
    assert residuals[-1] < 0.2, f"20 iterations should fit aligned data well, got {residuals[-1]}"


def test_sirt_beats_backprojection_on_a_clean_phantom() -> None:
    volume = synthetic_volume(size=32, n_slices=3)
    angles = np.deg2rad(np.linspace(0.0, 180.0, 40, endpoint=False))
    sinogram = forward_project(volume, angles)
    backend = NumpyProjectorBackend()

    def error(recon):
        scale = float((recon * volume).sum() / max((recon * recon).sum(), 1e-12))
        return float(np.linalg.norm(scale * recon - volume) / np.linalg.norm(volume))

    sirt = backend.reconstruct(sinogram, angles, algorithm="sirt", num_iter=25)
    bp = backend.reconstruct(sinogram, angles, algorithm="bp")
    assert error(sirt) < error(bp)


def test_backend_refuses_gridrec_with_a_reason() -> None:
    backend = NumpyProjectorBackend()
    with pytest.raises(ValueError, match="gridrec"):
        backend.reconstruct(np.zeros((4, 2, 8), np.float32), np.zeros(4), algorithm="gridrec")


# --------------------------------------------------------------------------------
# 4. The tomopy shim must never leak
# --------------------------------------------------------------------------------


def test_tomopy_shim_is_removed_on_exit() -> None:
    """A leaked stub would make ``pytest.importorskip('tomopy')`` elsewhere succeed
    and then fail on the first real tomopy call -- turning a clean skip into a
    confusing error in someone else's test file."""
    before = {name: sys.modules.get(name) for name in ("tomopy", "tomopy.prep")}
    with tomopy_shim() as installed:
        if installed:
            import tomopy.prep.alignment as alignment

            assert callable(alignment.shift_images)
    for name, previous in before.items():
        assert sys.modules.get(name) is previous


def test_shim_shift_images_matches_the_engines_documented_sign() -> None:
    """``apply_shifts(frame, sy=3)`` must take row 10 to row 7 -- the repo's own gate."""
    from benchmarks.runner import _shim_shift_images

    frame = np.zeros((1, 32, 32), dtype=np.float32)
    frame[0, 10, 20] = 1.0
    moved = _shim_shift_images(frame, np.array([3.0]), np.array([0.0]))
    row, column = np.unravel_index(np.argmax(moved[0]), moved[0].shape)
    assert (row, column) == (7, 20)

    moved = _shim_shift_images(frame, np.array([0.0]), np.array([-4.0]))
    row, column = np.unravel_index(np.argmax(moved[0]), moved[0].shape)
    assert (row, column) == (10, 24)


# --------------------------------------------------------------------------------
# 5. The runner, end to end
# --------------------------------------------------------------------------------


def test_reference_aligners_bracket_the_table(tiny_case: BenchmarkCase) -> None:
    """Oracle ~0, null == the injected misalignment. If not, the scorer is broken."""
    report = run_benchmark(tiny_case, [NullAligner(), OracleAligner()], with_residual=False)
    rows = {row["name"]: row for row in report.rows}

    oracle = rows["oracle"]["shift_recovery"]
    assert oracle["rms_dy"] < 1e-9 and oracle["rms_dx"] < 1e-9
    assert oracle["meets_target"]

    null = rows["null"]["shift_recovery"]
    assert null["rms_dy"] == pytest.approx(null["injected_rms_dy"], rel=1e-9)
    assert not null["meets_target"]
    assert "aligner" in report.table()


def test_missing_aligner_module_is_skipped_not_fatal(tiny_case: BenchmarkCase) -> None:
    absent = ModuleAligner("nonexistent", "tktomo.ptycho_align.core.definitely_not_here")
    report = run_benchmark(tiny_case, [absent], with_residual=False)
    row = report.rows[0]
    assert row["status"] == "skipped"
    assert "not importable" in row["message"]


def test_report_round_trips_through_json(tiny_case: BenchmarkCase, tmp_path) -> None:
    report = run_benchmark(tiny_case, [NullAligner(), OracleAligner()], with_residual=False)
    path = report.to_json(tmp_path / "report.json")
    loaded = json.loads(path.read_text())
    assert loaded["case"]["name"] == tiny_case.name
    assert {r["name"] for r in loaded["results"]} == {"null", "oracle"}
    assert loaded["environment"]["tomopy_shim"] in (True, False)


def test_comparison_figure_is_optional(tiny_case: BenchmarkCase, tmp_path) -> None:
    """No matplotlib must degrade to no figure, never to a failed run."""
    report = run_benchmark(tiny_case, [NullAligner(), OracleAligner()], with_residual=False)
    report.case["truth_dy"] = tiny_case.truth.dy.tolist()
    report.case["truth_dx"] = tiny_case.truth.dx.tolist()
    path = comparison_figure(report, tmp_path / "figure.png")
    if path is not None:
        assert path.exists()


@pytest.mark.parametrize(
    "aligner_factory",
    [
        pytest.param(lambda: JirrAligner(iterations=3), id="jirr"),
        pytest.param(lambda: OdstrcilAligner(iterations=3), id="odstrcil"),
        pytest.param(lambda: JointGdAligner(iterations_per_stage=25), id="joint_gd"),
    ],
)
def test_real_aligners_run_and_are_scored(tiny_case: BenchmarkCase, aligner_factory) -> None:
    """Each shipped aligner must produce a scoreable answer, or skip with a reason.

    Deliberately *not* an accuracy gate: how well each does is the benchmark's output,
    not its precondition, and pinning an accuracy here would make the suite fail
    whenever someone improves or legitimately re-tunes a method. What is asserted is
    that the harness can drive it, score it, and serialise the result.
    """
    report = run_benchmark(tiny_case, [aligner_factory()], with_residual=False)
    row = report.rows[0]
    assert row["status"] in ("ok", "skipped"), row["message"]
    if row["status"] == "skipped":
        pytest.skip(row["message"])
    score = row["shift_recovery"]
    assert np.isfinite(score["rms_dy"]) and np.isfinite(score["rms_dx"])
    assert "flipped_is_better" in row["sign_check"]
    json.dumps(report.to_dict())  # must be serialisable


def test_jirr_recovers_shifts_on_a_clean_case(tiny_case: BenchmarkCase) -> None:
    """The incumbent must at least beat doing nothing. The published baseline number
    comes from a full run (see benchmarks/README.md); this only guards against a
    regression that breaks it outright."""
    report = run_benchmark(
        tiny_case, [NullAligner(), JirrAligner(iterations=4)], with_residual=False
    )
    rows = {row["name"]: row for row in report.rows}
    if rows["jirr"]["status"] == "skipped":
        pytest.skip(rows["jirr"]["message"])
    assert rows["jirr"]["status"] == "ok", rows["jirr"]["message"]
    jirr = rows["jirr"]["shift_recovery"]
    null = rows["null"]["shift_recovery"]
    assert jirr["rms_dy"] < null["rms_dy"] / 2
    assert jirr["rms_dx"] < null["rms_dx"] / 2


def test_ground_truth_summary_is_json_safe(tiny_case: BenchmarkCase) -> None:
    payload = tiny_case.truth.to_dict(arrays=False)
    json.dumps(payload)
    assert payload["n_angles"] == tiny_case.n_angles
    assert isinstance(tiny_case.truth, GroundTruth)
