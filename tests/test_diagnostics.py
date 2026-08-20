"""Tests for the executable artifact-to-cause table.

The tests that matter here are not the unit tests -- they are the **injection** tests.
A detector that has never been shown to fire on a known cause is worthless, so every
failure mode this repo can synthesise is injected into a phantom and the verdict is
checked against the truth, including the null controls (a clean stack, and a clean
stack with 2% noise, must fire nothing).

The injectors are deliberately physical rather than cosmetic:

* modes 4, 5 and 6 are injected by **rotating the phantom volume about a genuinely
  tilted 3-D axis** and projecting, not by shearing the finished projections. The
  difference is load-bearing: a shear reproduces the horizontal signature of an
  out-of-plane tilt but not its vertical one, and the vertical one is exactly what
  distinguishes an out-of-plane tilt from a merely tilted sample.
* the phantom's body is a CYLINDER, so all three row bands of the arc test carry mass
  and the vertical mass profile has sharp edges to register against; and its centre of
  mass sits off the rotation axis, because a centred object gives the centroid sinusoid
  no amplitude and the horizontal probes nothing to work with (they say so, correctly,
  by returning NOT_APPLICABLE -- which makes for a vacuous test).
* every injected shift is applied with ``cval=0`` and the object is enclosed with a
  margin wider than any injected shift, so that the vacuum border stays vacuum. With
  edge replication instead, half the probes report a phase ramp that is really the
  injector's own smearing.

Runs on numpy + scipy alone: no GPU, no beamtime data, no scikit-image, no tomopy.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tktomo.diagnostics import (
    CATALOGUE,
    FailureMode,
    ProbeStatus,
    STAGE_ORDER,
    TriageStage,
    confidence_from_ratio,
    diagnose,
    fbp_slice,
    format_catalogue,
    format_verdict,
    forward_project_slice,
    probe_angle_readback,
    probe_angular_coverage,
    probe_axis_tilt,
    probe_center_consistency,
    probe_center_sweep,
    probe_deformation,
    probe_scale_drift,
    probe_shift_jitter,
    probe_truncation,
    probe_vacuum_phase,
    probe_vertical_drift,
    save_verdict,
    spec_for,
    stack_moments,
    triage,
)

SIZE = 64
N_ROWS = 64
N_ANGLES = 48

# (value, cx, cy, cz, ax, ay, az) in normalised coordinates: x, y over the frame and z
# over the detector rows, both in [-1, 1].
_ELLIPSOIDS = [
    (1.4, -0.24, 0.22, -0.35, 0.13, 0.13, 0.20),
    (-0.6, 0.20, -0.10, 0.10, 0.12, 0.12, 0.18),
    (0.9, 0.10, 0.24, 0.30, 0.13, 0.09, 0.14),
    (0.7, -0.05, -0.24, -0.05, 0.10, 0.10, 0.12),
]
# (value, cx, cy, ax, ay, z0, z1): an elliptical cylinder -- see the module docstring.
_CYLINDERS = [(1.0, 0.18, 0.06, 0.36, 0.32, -0.55, 0.55)]

_PERM = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # xyz -> zyx


# ----------------------------------------------------------------------------------
# phantom and injectors
# ----------------------------------------------------------------------------------


def phantom_volume(size: int = SIZE, n_slices: int = N_ROWS, extra=None) -> np.ndarray:
    """A 3-D phantom: an off-axis cylinder plus blobs at different heights."""
    ys, xs = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    zs = np.linspace(-1.0, 1.0, n_slices)
    vol = np.zeros((n_slices, size, size))
    for value, cx, cy, cz, ax, ay, az in _ELLIPSOIDS + list(extra or []):
        d = (
            ((xs[None] - cx) / ax) ** 2
            + ((ys[None] - cy) / ay) ** 2
            + ((zs[:, None, None] - cz) / az) ** 2
        )
        vol[d <= 1.0] += value
    for value, cx, cy, ax, ay, z0, z1 in _CYLINDERS:
        disc = (((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2) <= 1.0
        inside = (zs >= z0) & (zs <= z1)
        vol[inside[:, None, None] & disc[None]] += value
    return np.clip(vol, 0.0, None)


def project_ideal(vol: np.ndarray, theta_rad: np.ndarray, center: float | None = None) -> np.ndarray:
    """Slice-by-slice projection about a perfect vertical axis."""
    out = np.zeros((theta_rad.size, vol.shape[0], vol.shape[1]))
    for s in range(vol.shape[0]):
        out[:, s, :] = forward_project_slice(vol[s], theta_rad, center=center)
    return out


def _rotation(axis, angle: float) -> np.ndarray:
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def project_about_axis(
    vol: np.ndarray,
    theta_rad: np.ndarray,
    *,
    alpha_deg: float = 0.0,
    beta_deg: float = 0.0,
    center_shift: float = 0.0,
) -> np.ndarray:
    """Project a volume rotated about a genuinely tilted 3-D axis.

    ``alpha_deg`` tilts the axis *within* the detector plane (mode 4), ``beta_deg``
    tilts it toward the beam (mode 6), ``center_shift`` displaces it laterally (mode 5).
    """
    from scipy.ndimage import affine_transform, shift as ndi_shift

    a, b = math.radians(alpha_deg), math.radians(beta_deg)
    axis = (math.sin(a), math.sin(b), math.cos(a) * math.cos(b))
    centre = (np.array(vol.shape, dtype=float) - 1) / 2.0
    out = np.zeros((theta_rad.size, vol.shape[0], vol.shape[2]))
    for i, th in enumerate(theta_rad):
        rot = _PERM @ _rotation(axis, float(th)) @ _PERM.T
        turned = affine_transform(
            vol, rot.T, offset=centre - rot.T @ centre, order=1, mode="constant", cval=0.0
        )
        out[i] = turned.sum(axis=1)
    if center_shift:
        out = ndi_shift(out, (0.0, 0.0, center_shift), order=1, mode="constant", cval=0.0)
    return out


def shift_projections(stack, du, dv=None) -> np.ndarray:
    from scipy.ndimage import shift as ndi_shift

    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        out[i] = ndi_shift(
            stack[i],
            (0.0 if dv is None else float(dv[i]), float(du[i])),
            order=1,
            mode="constant",
            cval=0.0,
        )
    return out


def add_phase_ramp(stack, rng, *, slope_frac=0.05, offset_frac=0.05) -> np.ndarray:
    n_v, n_u = stack.shape[1:]
    v, u = np.mgrid[0:n_v, 0:n_u]
    peak = np.percentile(stack, 99)
    out = stack.copy()
    for i in range(stack.shape[0]):
        a, b = rng.normal(0.0, slope_frac * peak / n_u, 2)
        out[i] += a * u + b * v + rng.normal(0.0, offset_frac * peak)
    return out


def rescale_projections(stack, scales) -> np.ndarray:
    from scipy.ndimage import affine_transform

    out = np.empty_like(stack)
    cv, cu = (np.array(stack.shape[1:], dtype=float) - 1) / 2.0
    for i in range(stack.shape[0]):
        s = 1.0 / float(scales[i])
        out[i] = affine_transform(
            stack[i], np.diag([s, s]), offset=[cv - s * cv, cu - s * cu], order=1,
            mode="constant", cval=0.0,
        )
    return out


_CACHE: dict[str, object] = {}


def theta_deg() -> np.ndarray:
    return np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)


def clean_case() -> tuple[np.ndarray, np.ndarray]:
    """The reference stack: perfectly aligned, noiseless, enclosed."""
    if "clean" not in _CACHE:
        vol = phantom_volume()
        _CACHE["volume"] = vol
        _CACHE["clean"] = project_ideal(vol, np.deg2rad(theta_deg()))
    return _CACHE["clean"], theta_deg()  # type: ignore[return-value]


def clean_volume() -> np.ndarray:
    clean_case()
    return _CACHE["volume"]  # type: ignore[return-value]


# ----------------------------------------------------------------------------------
# the catalogue itself
# ----------------------------------------------------------------------------------


def test_catalogue_covers_the_twelve_modes_with_every_field_filled():
    assert len(CATALOGUE) == 12
    assert sorted(spec.number for spec in CATALOGUE.values()) == list(range(1, 13))
    for mode, spec in CATALOGUE.items():
        assert spec.mode is mode
        assert spec.stage in STAGE_ORDER
        assert len(spec.title) > 5
        for field in (spec.slice_signature, spec.sinogram_signature, spec.confirm, spec.fix):
            assert len(field) > 30, f"{mode} has a stub field"


def test_catalogue_renders_as_text():
    text = format_catalogue()
    assert "ARTIFACT -> CAUSE" in text
    for spec in CATALOGUE.values():
        assert spec.title in text


def test_confidence_is_monotone_and_zero_at_the_threshold():
    assert confidence_from_ratio(1.0, 1.0) == 0.0
    assert confidence_from_ratio(0.5, 1.0) == 0.0
    assert confidence_from_ratio(2.0, 1.0) == pytest.approx(0.5)
    assert confidence_from_ratio(10.0, 1.0) == pytest.approx(0.9)
    assert confidence_from_ratio(float("nan"), 1.0) == 0.0
    with pytest.raises(ValueError):
        confidence_from_ratio(1.0, 0.0)


# ----------------------------------------------------------------------------------
# numerical helpers the probes stand on
# ----------------------------------------------------------------------------------


def test_register_1d_recovers_a_known_subpixel_shift():
    from tktomo.diagnostics.tests_geometry import _register_1d

    u = np.arange(96.0)
    ref = np.exp(-((u - 30.0) ** 2) / (2 * 4.0**2))
    for truth in (0.0, 1.0, -2.5, 5.25, -7.75):
        moving = np.exp(-((u - (30.0 + truth)) ** 2) / (2 * 4.0**2))
        assert _register_1d(ref, moving) == pytest.approx(truth, abs=0.06)

    # A flat-topped profile is where the gradient registration earns its keep: the plain
    # correlation of a plateau has a broad peak and lands 0.15-0.35 px short.
    plateau = ((u > 24) & (u < 56)).astype(float)
    from scipy.ndimage import shift as ndi_shift

    for truth in (0.5, 2.0, -3.0):
        moved = ndi_shift(plateau, truth, order=1, mode="constant", cval=0.0)
        assert _register_1d(plateau, moved) == pytest.approx(truth, abs=0.06)


def test_fbp_and_forward_projector_are_a_consistent_pair():
    """A disc in, a disc out, in the right place and round."""
    size = 96
    y, x = np.mgrid[0:size, 0:size]
    image = (((x - 55.0) ** 2 + (y - 40.0) ** 2) <= 12.0**2).astype(float)
    theta = np.deg2rad(np.linspace(0.0, 180.0, 120, endpoint=False))

    sino = forward_project_slice(image, theta)
    # The projection of a disc has the analytic width 2R at every angle.
    # The projection of a disc is 2*sqrt(R^2 - u^2), so its full width at half maximum
    # is sqrt(3)*R, not 2R.
    widths = (sino > 0.5 * sino.max(axis=1, keepdims=True)).sum(axis=1)
    assert widths.std() < 1.5
    assert abs(widths.mean() - math.sqrt(3) * 12.0) < 2.0

    recon = fbp_slice(sino, theta)
    # A disc reconstructs to a plateau, so its argmax is anywhere inside it: use the
    # centroid of the reconstructed mass instead.
    mass = np.clip(recon, 0.0, None)
    rows, cols = np.mgrid[0 : recon.shape[0], 0 : recon.shape[1]]
    assert (mass * rows).sum() / mass.sum() == pytest.approx(40.0, abs=2.0)
    assert (mass * cols).sum() / mass.sum() == pytest.approx(55.0, abs=2.0)
    reprojected = forward_project_slice(recon, theta)
    gain = np.sum(sino * reprojected) / np.sum(reprojected**2)
    residual = np.sqrt(np.mean((sino - gain * reprojected) ** 2)) / sino.std()
    assert residual < 0.15


def test_theta_units_are_resolved_and_recorded():
    stack, theta = clean_case()
    in_deg = diagnose(stack, theta, only=["angular_coverage"])
    in_rad = diagnose(stack, np.deg2rad(theta), only=["angular_coverage"])
    assert in_deg.context["theta_units"] == "deg"
    assert in_rad.context["theta_units"] == "rad"
    assert in_deg.context["span_deg"] == pytest.approx(in_rad.context["span_deg"])
    with pytest.raises(ValueError):
        diagnose(stack, theta, theta_units="gradians")


def test_moments_flip_negative_phase_and_flag_it():
    """Ptycho phase is negative inside material; every moment here needs it positive."""
    stack, _ = clean_case()
    positive = stack_moments(stack)
    negative = stack_moments(-stack)
    assert not positive.inverted and negative.inverted
    np.testing.assert_allclose(negative.com_u, positive.com_u, atol=1e-9)
    np.testing.assert_allclose(negative.mass, positive.mass, rtol=1e-9)


def test_moments_marginals_match_a_direct_computation():
    stack, _ = clean_case()
    mom = stack_moments(stack, chunk=7)
    direct_u = np.clip(stack[3], 0.0, None).mean(axis=0)
    np.testing.assert_allclose(mom.uprofile_mass[3], direct_u, rtol=1e-10)
    u = np.arange(stack.shape[2], dtype=float)
    np.testing.assert_allclose(mom.com_u[3], (direct_u @ u) / direct_u.sum(), rtol=1e-10)
    # Band heights are mass-weighted, so they sit inside the object, not at the band centre.
    assert mom.band_z[0] > mom.band_center_v[0]
    assert mom.band_z[-1] < mom.band_center_v[-1]


# ----------------------------------------------------------------------------------
# null controls -- the tests that stop this being a random-verdict generator
# ----------------------------------------------------------------------------------


def test_clean_phantom_fires_nothing_and_every_probe_runs():
    stack, theta = clean_case()
    verdict = diagnose(stack, theta)
    assert verdict.findings == (), [f.to_dict() for f in verdict.findings]
    assert verdict.coverage == 1.0
    assert verdict.top is None


def test_noise_alone_does_not_fire_a_detector():
    stack, theta = clean_case()
    rng = np.random.default_rng(4)
    noisy = stack + rng.normal(0.0, 0.02 * stack.max(), stack.shape)
    verdict = diagnose(noisy, theta)
    assert all(f.confidence < 0.25 for f in verdict.findings), [
        (f.mode.value, f.confidence) for f in verdict.findings
    ]


# ----------------------------------------------------------------------------------
# one injection per probe
# ----------------------------------------------------------------------------------


def test_vacuum_phase_detects_an_injected_ramp():
    stack, _ = clean_case()
    assert probe_vacuum_phase(stack).status is ProbeStatus.CLEAR

    ramped = add_phase_ramp(stack, np.random.default_rng(11))
    result = probe_vacuum_phase(ramped)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.PHASE_RAMP
    assert result.metrics["ramp_frac"] > result.metrics["border_signal_frac"]


def test_truncation_detects_a_cropped_object_and_not_an_enclosed_one():
    stack, _ = clean_case()
    assert probe_truncation(stack).status is ProbeStatus.CLEAR
    assert probe_truncation(stack).metrics["u_edge_worst_median"] < 0.05

    cropped = np.ascontiguousarray(stack[:, :, 20:-20])
    result = probe_truncation(cropped)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.LOCAL_TOMOGRAPHY


def test_angular_coverage_measures_the_wedge():
    theta = theta_deg()
    assert probe_angular_coverage(theta).status is ProbeStatus.CLEAR

    kept = theta[theta < 130.0]
    result = probe_angular_coverage(kept)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.MISSING_WEDGE
    assert result.metrics["max_gap_deg"] == pytest.approx(180.0 - kept.max(), abs=0.1)
    assert probe_angular_coverage(np.array([0.0])).status is ProbeStatus.NOT_APPLICABLE


def test_center_consistency_finds_a_constant_lateral_shift():
    stack, theta = clean_case()
    assert probe_center_consistency(stack, theta).status is ProbeStatus.CLEAR

    shifted = shift_projections(stack, np.full(N_ANGLES, 3.0))
    result = probe_center_consistency(shifted, theta)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.WRONG_CENTER
    # Shifting the data to larger u puts the apparent axis at larger u too.
    assert result.metrics["center_offset_px"] == pytest.approx(3.0, abs=0.2)


def test_center_consistency_uses_mirror_pairs_over_a_full_turn():
    """0-vs-180 registration is model-free, so it should be preferred when available."""
    vol = clean_volume()
    theta = np.linspace(0.0, 360.0, 40, endpoint=False)
    stack = project_ideal(vol, np.deg2rad(theta), center=(SIZE - 1) / 2.0 + 2.0)
    result = probe_center_consistency(stack, theta)
    assert result.metrics["n_mirror_pairs"] >= 5
    assert result.metrics["estimator"] == 1.0  # the mirror estimator was used
    assert result.metrics["center_mirror_px"] == pytest.approx((SIZE - 1) / 2.0 + 2.0, abs=0.3)


def test_center_sweep_finds_the_entropy_minimum():
    stack, theta = clean_case()
    clean = probe_center_sweep(stack, theta)
    assert clean.status is ProbeStatus.CLEAR
    assert clean.metrics["center_offset_px"] == pytest.approx(0.0, abs=0.6)

    shifted = shift_projections(stack, np.full(N_ANGLES, 3.0))
    result = probe_center_sweep(shifted, theta)
    assert result.status is ProbeStatus.FIRED
    assert result.metrics["center_offset_px"] == pytest.approx(3.0, abs=1.0)


def test_center_sweep_declines_a_limited_angle_scan():
    stack, theta = clean_case()
    keep = theta < 100.0
    result = probe_center_sweep(stack[keep], theta[keep])
    assert result.status is ProbeStatus.NOT_APPLICABLE
    assert "150 deg" in result.reason


@pytest.mark.parametrize(
    "kwargs, expected, other",
    [
        ({}, None, ()),
        ({"alpha_deg": 8.0}, FailureMode.TILT_AXIS_ANGLE,
         (FailureMode.OUT_OF_PLANE_TILT, FailureMode.TILT_AXIS_LATERAL)),
        ({"center_shift": 3.0}, FailureMode.TILT_AXIS_LATERAL,
         (FailureMode.OUT_OF_PLANE_TILT, FailureMode.TILT_AXIS_ANGLE)),
        ({"beta_deg": 6.0}, FailureMode.OUT_OF_PLANE_TILT,
         (FailureMode.TILT_AXIS_ANGLE, FailureMode.TILT_AXIS_LATERAL)),
    ],
    ids=["ideal_axis", "in_plane_tilt", "lateral_shift", "out_of_plane_tilt"],
)
def test_three_slice_arc_test_discriminates_modes_4_5_and_6(kwargs, expected, other):
    """The marquee test: three geometrically distinct axis errors, told apart.

    Each is injected by rotating the volume about a real 3-D axis, so the vertical
    signature is present too -- which is what stops a merely tilted sample being read as
    an out-of-plane tilt.
    """
    theta = np.deg2rad(theta_deg())
    stack = project_about_axis(clean_volume(), theta, **kwargs)
    result = probe_axis_tilt(stack, theta, theta_units="rad")
    modes = {f.mode for f in result.findings}
    if expected is None:
        assert result.status is ProbeStatus.CLEAR, result.detail
        # The phantom's own slice centroids walk with height, which mimics mode 6; the
        # vertical channel is what stops it being reported as one.
        assert result.metrics["amplitude_travel_px"] > 0.3
        assert abs(result.metrics["vertical_modulation_px"]) < 0.05
    else:
        assert expected in modes, result.detail
        assert not (modes & set(other)), f"cross-talk into {modes & set(other)}"


def test_arc_test_recovers_the_out_of_plane_angle():
    theta = np.deg2rad(theta_deg())
    stack = project_about_axis(clean_volume(), theta, beta_deg=6.0)
    result = probe_axis_tilt(stack, theta, theta_units="rad")
    assert abs(result.metrics["out_of_plane_deg"]) == pytest.approx(6.0, abs=1.5)


def test_arc_test_bands_follow_the_object_not_the_detector():
    """An object filling half the frame must still get three usable bands.

    Splitting the detector into three equal bands leaves the empty half as a band that
    carries no mass, and the probe then (correctly, uselessly) refuses to run. Our own
    tall-detector scans look exactly like this. The bands are laid over the object's row
    support instead, which is the same test with a shorter lever arm.
    """
    stack, theta = clean_case()
    tall = np.zeros((stack.shape[0], stack.shape[1] * 2, stack.shape[2]))
    tall[:, : stack.shape[1]] = stack

    moments = stack_moments(tall)
    assert moments.band_rows[-1][1] <= stack.shape[1], "bands ran off into the empty half"
    result = probe_axis_tilt(tall, theta, moments=moments)
    assert result.status is ProbeStatus.CLEAR, result.reason


def test_arc_test_downgrades_itself_when_its_own_model_is_strained():
    """A large theta-correlated wander is not rigid-object motion.

    The arc test's whole model is the rigid-object centroid sinusoid, and over a 180 deg
    span its basis is not orthogonal, so a residual correlated with theta leaks into the
    constant term the test regresses. This is not hypothetical: on real data the probe
    produced a confident 3.9 deg tilt that an independent per-band entropy centre sweep
    contradicted in both magnitude AND sign, with a band residual twice the axis walk
    being claimed. So the probe compares the two and halves itself when the residual wins.
    """
    theta = np.deg2rad(theta_deg())
    stack = project_about_axis(clean_volume(), theta, alpha_deg=8.0)
    clean = probe_axis_tilt(stack, theta, theta_units="rad")
    assert clean.metrics["band_residual_over_effect"] < 0.1

    wander = 6.0 * np.sin(3 * theta)  # smooth in theta, and no rigid object does it
    strained = probe_axis_tilt(shift_projections(stack, wander), theta, theta_units="rad")
    assert strained.metrics["band_residual_over_effect"] > 1.0

    before = next(f for f in clean.findings if f.mode is FailureMode.TILT_AXIS_ANGLE)
    after = next(f for f in strained.findings if f.mode is FailureMode.TILT_AXIS_ANGLE)
    assert after.confidence == pytest.approx(0.5 * before.confidence, rel=0.25)
    assert "CAVEAT" in after.detail and "probe_center_sweep(rows=" in after.detail


def test_center_sweep_can_be_pointed_at_a_row_band():
    """The independent cross-check for the arc test's verdict, as a one-liner."""
    stack, theta = clean_case()
    top = probe_center_sweep(stack, theta, rows=(20, 32))
    bottom = probe_center_sweep(stack, theta, rows=(32, 44))
    assert top.status is ProbeStatus.CLEAR and bottom.status is ProbeStatus.CLEAR
    walk = top.metrics["center_entropy_px"] - bottom.metrics["center_entropy_px"]
    assert abs(walk) < 1.5, "an untilted phantom's centre must not walk with height"


def test_arc_test_declines_without_three_bands_or_enough_angles():
    stack, theta = clean_case()
    two = stack_moments(stack, n_bands=2)
    assert probe_axis_tilt(stack, theta, moments=two).status is ProbeStatus.NOT_APPLICABLE
    assert "3 row bands" in probe_axis_tilt(stack, theta, moments=two).reason

    keep = theta < 40.0
    narrow = probe_axis_tilt(stack[keep], theta[keep])
    assert narrow.status is ProbeStatus.NOT_APPLICABLE
    assert "60 deg" in narrow.reason


def test_vertical_drift_separates_a_smooth_trend_from_jitter():
    stack, theta = clean_case()
    assert probe_vertical_drift(stack).status is ProbeStatus.CLEAR

    drift = 3.0 * np.arange(N_ANGLES) / (N_ANGLES - 1)
    drifted = shift_projections(stack, np.zeros(N_ANGLES), drift)
    result = probe_vertical_drift(drifted)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.VERTICAL_DRIFT
    assert result.metrics["drift_ptp_px"] == pytest.approx(3.0, abs=0.4)
    assert result.metrics["vertical_jitter_rms_px"] < 0.2
    assert result.metrics["lag1_autocorr"] > 0.8  # smooth, not white


def test_jitter_fires_on_white_residuals_and_declines_smooth_ones():
    stack, theta = clean_case()
    rng = np.random.default_rng(2)
    jittered = shift_projections(stack, rng.normal(0, 1.2, N_ANGLES), rng.normal(0, 1.2, N_ANGLES))
    result = probe_shift_jitter(jittered, theta)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.JITTER
    assert abs(result.metrics["horizontal_lag1"]) < 0.4

    # A large but SMOOTH residual is drift or an angle error, and must not be called
    # jitter. It has to be a shape the centroid sinusoid cannot absorb -- a LINEAR
    # horizontal drift is nearly collinear with cos(theta) over a monotone 0-180 sweep
    # and the fit simply eats it, which is a real degeneracy worth knowing about.
    drift = 2.0 * np.sin(2 * np.deg2rad(theta))
    drifted = shift_projections(stack, drift)
    smooth = probe_shift_jitter(drifted, theta)
    assert smooth.status is ProbeStatus.CLEAR
    assert smooth.metrics["horizontal_rms_px"] > 0.3
    assert smooth.metrics["horizontal_lag1"] > 0.4
    assert "SMOOTH" in smooth.detail


def test_angle_readback_recovers_an_injected_angular_gain():
    vol = clean_volume()
    theta = theta_deg()
    stack = project_ideal(vol, np.deg2rad(theta * 1.06))  # recorded angles are 6% short
    result = probe_angle_readback(stack, theta)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.ANGLE_READBACK
    assert result.metrics["angle_gain"] == pytest.approx(1.06, abs=0.01)

    clean, _ = clean_case()
    assert probe_angle_readback(clean, theta).status is ProbeStatus.CLEAR


def test_angle_readback_declines_a_centred_object():
    """No centroid orbit, no angle information -- and the probe must say so."""
    ys, xs = np.mgrid[-1 : 1 : SIZE * 1j, -1 : 1 : SIZE * 1j]
    disc = ((xs**2 + ys**2) <= 0.3**2).astype(float)
    vol = np.repeat(disc[None], N_ROWS, axis=0)
    stack = project_ideal(vol, np.deg2rad(theta_deg()))
    result = probe_angle_readback(stack, theta_deg())
    assert result.status is ProbeStatus.NOT_APPLICABLE
    assert "rotation axis" in result.reason


def test_scale_drift_recovers_the_injected_magnification():
    stack, theta = clean_case()
    assert probe_scale_drift(stack, theta).status is ProbeStatus.CLEAR

    scales = 1.0 + 0.02 * np.arange(N_ANGLES) / (N_ANGLES - 1)
    result = probe_scale_drift(rescale_projections(stack, scales), theta)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.SCALE_DRIFT
    # Recovered magnitude is right to about 40% -- good enough to detect and to size the
    # problem, not a calibration. See the probe docstring.
    assert result.metrics["scale_change_frac"] == pytest.approx(0.02, rel=0.5)
    assert result.metrics["implied_rim_shift_px"] > result.metrics["implied_rim_shift_se_px"] * 3
    assert result.metrics["non_smooth_frac"] < 0.25


def test_deformation_is_caught_by_projected_mass_conservation():
    """The sharp statistic: parallel-beam projections of a rigid object conserve mass."""
    stack, theta = clean_case()
    clean_result = probe_deformation(stack, theta)
    assert clean_result.status is ProbeStatus.CLEAR

    grown = phantom_volume(extra=[(1.2, 0.30, 0.30, 0.0, 0.13, 0.13, 0.30)])
    half = N_ANGLES // 2
    rad = np.deg2rad(theta)
    deformed = np.concatenate(
        [project_ideal(clean_volume(), rad[:half]), project_ideal(grown, rad[half:])]
    )
    result = probe_deformation(deformed, theta)
    assert result.status is ProbeStatus.FIRED
    assert result.findings[0].mode is FailureMode.DEFORMATION
    assert result.metrics["mass_white_frac"] > 20 * clean_result.metrics["mass_white_frac"]


def test_deformation_uses_a_supplied_volume_and_rejects_a_mismatched_one():
    stack, theta = clean_case()
    volume = np.zeros((N_ROWS, SIZE, SIZE))
    volume[:] = clean_volume()
    ok = probe_deformation(stack, theta, volume=volume)
    assert ok.status in (ProbeStatus.CLEAR, ProbeStatus.FIRED)
    assert "supplied volume" in ok.detail

    bad = probe_deformation(stack, theta, volume=volume[:, :10, :10])
    assert "does not match projections" in bad.detail


# ----------------------------------------------------------------------------------
# the runner, the triage order, and the verdict object
# ----------------------------------------------------------------------------------


def test_triage_stops_at_the_first_firing_probe():
    stack, theta = clean_case()
    ramped = add_phase_ramp(stack, np.random.default_rng(5))
    verdict = triage(ramped, theta)
    assert verdict.stopped_at == "vacuum_phase"
    assert verdict.top is not None and verdict.top.mode is FailureMode.PHASE_RAMP
    later = [p for p in verdict.probes if p.status is ProbeStatus.SKIPPED]
    assert [p.probe for p in later][0] == "angular_coverage"
    assert all("triage stopped at vacuum_phase" in p.reason for p in later)


def test_truncation_is_tested_before_the_ramp_because_it_invalidates_it():
    from tktomo.diagnostics.tests_geometry import PROBES

    names = [name for name, _, _ in PROBES]
    assert names.index("truncation") < names.index("vacuum_phase")

    stack, theta = clean_case()
    cropped = np.ascontiguousarray(stack[:, :, 20:-20])
    verdict = diagnose(cropped, theta)
    assert verdict.top is not None and verdict.top.mode is FailureMode.LOCAL_TOMOGRAPHY
    ramp = verdict.by_mode(FailureMode.PHASE_RAMP)
    assert ramp is not None
    # ... and the ramp finding is downgraded, because its vacuum border is object.
    assert ramp.evidence["confidence_before_stage_discount"] > ramp.confidence


def test_later_stages_are_discounted_when_an_earlier_one_fires():
    stack, theta = clean_case()
    ramped = add_phase_ramp(stack, np.random.default_rng(5))
    verdict = diagnose(ramped, theta)
    assert verdict.top is not None and verdict.top.mode is FailureMode.PHASE_RAMP
    for finding in verdict.findings:
        if finding.spec.stage is not TriageStage.DATA_INTEGRITY:
            assert "confidence_before_stage_discount" in finding.evidence


def test_a_probe_that_raises_is_reported_not_swallowed(monkeypatch):
    import tktomo.diagnostics.tests_geometry as tg

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(tg, "probe_scale_drift", boom)
    stack, theta = clean_case()
    verdict = diagnose(stack, theta)
    broken = verdict.probe("scale_drift")
    assert broken.status is ProbeStatus.ERROR
    assert "synthetic failure" in broken.reason
    assert verdict.probe("deformation").ran  # the rest of the run continued


def test_allow_reconstruction_false_marks_the_recon_probes_not_applicable():
    stack, theta = clean_case()
    verdict = diagnose(stack, theta, allow_reconstruction=False)
    assert verdict.probe("center_sweep").status is ProbeStatus.NOT_APPLICABLE
    # deformation still runs: its mass statistic needs no reconstruction at all
    assert verdict.probe("deformation").ran
    assert "Reprojection skipped" in verdict.probe("deformation").detail


def test_verdict_is_json_serialisable(tmp_path):
    stack, theta = clean_case()
    shifted = shift_projections(stack, np.full(N_ANGLES, 3.0))
    verdict = diagnose(shifted, theta)
    payload = json.loads(verdict.to_json())
    assert payload["findings"][0]["mode"] == FailureMode.WRONG_CENTER.value
    assert payload["findings"][0]["fix"] == spec_for(FailureMode.WRONG_CENTER).fix
    assert 0.0 <= payload["coverage"] <= 1.0
    assert "curves" not in payload["probes"][0]

    path = save_verdict(verdict, tmp_path / "verdict.json", include_curves=True)
    reloaded = json.loads(path.read_text())
    assert isinstance(reloaded["probes"][0]["curves"], dict)


def test_text_report_shows_the_fix_and_every_probe():
    stack, theta = clean_case()
    shifted = shift_projections(stack, np.full(N_ANGLES, 3.0))
    text = format_verdict(diagnose(shifted, theta))
    assert "RANKED FINDINGS" in text and "PROBE LOG" in text
    assert "FIX:" in text
    for name in ("truncation", "vacuum_phase", "axis_tilt", "deformation"):
        assert name in text


def test_plot_verdict_builds_a_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from tktomo.diagnostics import plot_verdict

    stack, theta = clean_case()
    verdict = diagnose(add_phase_ramp(stack, np.random.default_rng(6)), theta)
    figure = plot_verdict(verdict)
    assert figure.axes


def test_unknown_probe_name_is_rejected():
    stack, theta = clean_case()
    with pytest.raises(ValueError, match="unknown probe"):
        diagnose(stack, theta, only=["no_such_probe"])


# ----------------------------------------------------------------------------------
# the confusion matrix: every mode we can synthesise, injected and detected
# ----------------------------------------------------------------------------------


def _injected_cases() -> dict[str, tuple[np.ndarray, np.ndarray, FailureMode | None]]:
    if "cases" in _CACHE:
        return _CACHE["cases"]  # type: ignore[return-value]
    stack, theta = clean_case()
    rad = np.deg2rad(theta)
    vol = clean_volume()
    rng = np.random.default_rng(7)
    grown = phantom_volume(extra=[(1.2, 0.30, 0.30, 0.0, 0.13, 0.13, 0.30)])
    half = N_ANGLES // 2
    cases = {
        "clean": (stack, theta, None),
        "wrong_center": (shift_projections(stack, np.full(N_ANGLES, 3.0)), theta,
                         FailureMode.WRONG_CENTER),
        "jitter": (shift_projections(stack, rng.normal(0, 1.2, N_ANGLES),
                                     rng.normal(0, 1.2, N_ANGLES)), theta, FailureMode.JITTER),
        "vertical_drift": (shift_projections(stack, np.zeros(N_ANGLES),
                                             3.0 * np.arange(N_ANGLES) / (N_ANGLES - 1)),
                           theta, FailureMode.VERTICAL_DRIFT),
        "tilt_axis_angle": (project_about_axis(vol, rad, alpha_deg=8.0), theta,
                            FailureMode.TILT_AXIS_ANGLE),
        "tilt_axis_lateral": (project_about_axis(vol, rad, center_shift=3.0), theta,
                              FailureMode.TILT_AXIS_LATERAL),
        "out_of_plane_tilt": (project_about_axis(vol, rad, beta_deg=6.0), theta,
                              FailureMode.OUT_OF_PLANE_TILT),
        "angle_readback": (project_ideal(vol, np.deg2rad(theta * 1.06)), theta,
                           FailureMode.ANGLE_READBACK),
        "scale_drift": (rescale_projections(stack, 1.0 + 0.02 * np.arange(N_ANGLES) / (N_ANGLES - 1)),
                        theta, FailureMode.SCALE_DRIFT),
        "deformation": (np.concatenate([project_ideal(vol, rad[:half]),
                                        project_ideal(grown, rad[half:])]), theta,
                        FailureMode.DEFORMATION),
        "missing_wedge": (stack[theta < 130.0], theta[theta < 130.0], FailureMode.MISSING_WEDGE),
        "phase_ramp": (add_phase_ramp(stack, rng), theta, FailureMode.PHASE_RAMP),
        "local_tomography": (np.ascontiguousarray(stack[:, :, 20:-20]), theta,
                             FailureMode.LOCAL_TOMOGRAPHY),
    }
    _CACHE["cases"] = cases
    return cases


@pytest.mark.parametrize("case", list(_injected_cases() if False else [
    "clean", "wrong_center", "jitter", "vertical_drift", "tilt_axis_angle",
    "tilt_axis_lateral", "out_of_plane_tilt", "angle_readback", "scale_drift",
    "deformation", "missing_wedge", "phase_ramp", "local_tomography",
]))
def test_confusion_matrix_diagonal(case):
    """Inject one failure mode, and require the verdict to name it.

    Two entries are deliberately weaker than "rank 1", and the reasons are physics, not
    tuning:

    * ``tilt_axis_lateral`` -- a laterally displaced axis and a constant lateral shift of
      the data are the SAME stack of numbers. Modes 1 and 5 are one equivalence class and
      both are reported at equal confidence; which is "the" cause is a question about the
      instrument, not about the data.
    * ``deformation`` -- inconsistent data also moves the entropy-minimising rotation
      centre, so the centre sweep fires too. The model-free centre estimator
      (``center_consistency``) does not corroborate it, and that disagreement is itself
      the tell.
    """
    stack, theta, expected = _injected_cases()[case]
    verdict = diagnose(stack, theta)
    modes = [f.mode for f in verdict.findings]

    if expected is None:
        assert modes == [], [(m.value, f.confidence) for m, f in zip(modes, verdict.findings)]
        return

    assert expected in modes, f"{case}: detected {[m.value for m in modes]}"
    rank = modes.index(expected)
    if case == "tilt_axis_lateral":
        assert {FailureMode.WRONG_CENTER, FailureMode.TILT_AXIS_LATERAL} <= set(modes)
        assert verdict.by_mode(expected).confidence >= 0.5
    elif case == "deformation":
        assert rank <= 1
    else:
        assert rank == 0, f"{case}: ranked {[m.value for m in modes]}"


def test_triage_stops_at_the_right_probe_for_every_injected_mode():
    """The actionable answer: which probe does the prescribed order stop at?"""
    expected_probe = {
        "clean": None,
        "wrong_center": "center_consistency",
        "jitter": "shift_jitter",
        "vertical_drift": "vertical_drift",
        "tilt_axis_angle": "axis_tilt",
        "tilt_axis_lateral": "center_consistency",
        "out_of_plane_tilt": "axis_tilt",
        "angle_readback": "angle_readback",
        "scale_drift": "scale_drift",
        "missing_wedge": "angular_coverage",
        "phase_ramp": "vacuum_phase",
        "local_tomography": "truncation",
    }
    cases = _injected_cases()
    for case, probe in expected_probe.items():
        stack, theta, _ = cases[case]
        assert triage(stack, theta).stopped_at == probe, case
