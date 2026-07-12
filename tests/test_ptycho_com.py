"""Centre-of-mass pre-alignment tests.

Built on a synthetic stack whose centroid sinusoid is known analytically, so the
recovered ``center`` and ``sx`` can be checked against the truth rather than against
another implementation of the same idea.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.ptycho_align.core.com import com_prealign, projection_centroids


def _gaussian_stack(
    com_u: np.ndarray, com_v: np.ndarray, n_v: int = 40, n_u: int = 64
) -> np.ndarray:
    """A Gaussian blob per angle, placed at exactly the requested centroid.

    A symmetric blob's centre of mass *is* its centre, so the centroids the code
    recovers are the ones we asked for -- no discretisation slop to argue about.
    """
    v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float64)
    stack = np.empty((len(com_u), n_v, n_u), dtype=np.float32)
    for i, (cu, cv) in enumerate(zip(com_u, com_v)):
        stack[i] = np.exp(-(((u - cu) ** 2 + (v - cv) ** 2) / (2 * 3.5**2)))
    return stack


def test_projection_centroids_recover_known_positions():
    com_u = np.array([20.0, 32.0, 41.5])
    com_v = np.array([18.0, 20.0, 22.5])
    stack = _gaussian_stack(com_u, com_v)

    got_v, got_u = projection_centroids(stack)
    np.testing.assert_allclose(got_u, com_u, atol=1e-3)
    np.testing.assert_allclose(got_v, com_v, atol=1e-3)


def test_com_prealign_recovers_centre_and_jitter():
    rng = np.random.default_rng(0)
    n = 90
    angles = np.linspace(0.0, np.pi, n, endpoint=False)

    center = 32.0
    amplitude_a, amplitude_b = 6.0, -3.0
    vertical_reference = 20.0

    # The horizontal jitter is made orthogonal to {sin, cos, 1} first. Any component
    # of the misalignment lying *in* that span is indistinguishable from the object
    # simply sitting elsewhere (and from the rotation axis sitting elsewhere), so the
    # fit is entitled to absorb it into a/b/c -- and it does. Note the basis is not
    # orthogonal over a 0..pi scan, since sin has a non-zero mean there, so "the
    # degenerate part" is not just the jitter's mean.
    basis = np.column_stack([np.sin(angles), np.cos(angles), np.ones_like(angles)])
    raw = rng.uniform(-1.5, 1.5, n)
    jitter_u = raw - basis @ np.linalg.lstsq(basis, raw, rcond=None)[0]
    jitter_v = rng.uniform(-1.5, 1.5, n)

    # The ideal (aligned) centroid traces the sinusoid; the jitter is the misalignment.
    ideal_u = amplitude_a * np.sin(angles) + amplitude_b * np.cos(angles) + center
    stack = _gaussian_stack(ideal_u + jitter_u, vertical_reference + jitter_v)

    result = com_prealign(stack, angles)

    assert result.center == pytest.approx(center, abs=0.1)
    assert result.amplitude == pytest.approx(np.hypot(amplitude_a, amplitude_b), abs=0.1)

    # sx/sy are the CORRECTION to apply, so they are +jitter, not -jitter. The vertical
    # is still only defined up to a constant (which just translates the volume).
    np.testing.assert_allclose(result.sx, jitter_u, atol=0.05)
    np.testing.assert_allclose(
        result.sy - result.sy.mean(), jitter_v - jitter_v.mean(), atol=0.05
    )
    assert result.fit_residual == pytest.approx(np.std(jitter_u), abs=0.1)


def test_com_prealign_median_variant_ignores_an_outlier():
    n = 40
    angles = np.linspace(0.0, np.pi, n, endpoint=False)
    com_v = np.full(n, 20.0)
    com_v[7] = 34.0  # one badly-placed projection

    stack = _gaussian_stack(np.full(n, 32.0), com_v)

    mean_based = com_prealign(stack, angles, vertical_reference="mean")
    median_based = com_prealign(stack, angles, vertical_reference="median")

    # The median reference is unmoved by the outlier, so the good projections get a
    # ~zero correction; the mean reference smears the outlier across all of them.
    good = np.ones(n, dtype=bool)
    good[7] = False
    assert np.abs(median_based.sy[good]).max() < 0.01
    assert np.abs(mean_based.sy[good]).max() > 0.2


def test_center_is_plausible_catches_a_failed_estimator():
    """TomoPy's find_center_vo returns a number even when it has failed.

    Observed on a 104 px wide padded phantom whose true axis is at 52.0: Vo returned
    96.5 (and 0.0 with a different `ind`). A bad centre silently ruins a long run, so
    an implausible estimate must be caught before it reaches the engine.
    """
    from tktomo.ptycho_align.core.com import center_is_plausible

    ok, _ = center_is_plausible(52.5, width=104, reference=51.97)
    assert ok

    ok, reason = center_is_plausible(96.5, width=104, reference=51.97)
    assert not ok and "differs from the expected" in reason

    ok, reason = center_is_plausible(0.0, width=104, reference=51.97)
    assert not ok and "outside the detector" in reason

    ok, reason = center_is_plausible(float("nan"), width=104)
    assert not ok and "finite" in reason

    # With no independent reference, the detector midpoint is the fallback.
    assert center_is_plausible(53.0, width=104)[0]
    assert not center_is_plausible(90.0, width=104)[0]


def test_com_prealign_rejects_zero_mass_data():
    angles = np.linspace(0.0, np.pi, 4, endpoint=False)
    with pytest.raises(ValueError, match="zero positive mass"):
        com_prealign(np.zeros((4, 16, 16), dtype=np.float32), angles)
