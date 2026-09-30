"""Evidence that registering on the gradient beats registering on the phase.

The claim under test is narrow and checkable: the estimate must be invariant to the
two gauge freedoms ptychographic phase retrieval leaves behind -- a constant phase
offset and a linear phase ramp -- and the value-domain registration the generic engine
uses must NOT be. The contrast is the scientific point, so both sides are measured here
with the *same* machinery and one variable changed
(:attr:`~tktomo.ptycho_align.core.gradient.GradientConfig.domain`), and the
value-domain path is pinned against ``skimage.registration.phase_cross_correlation``
so nobody can claim the comparison was rigged with a weak baseline.

numpy + scipy only; no tomopy, no GPU, no beamtime data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter

from tktomo.ptycho_align.core.gradient import (
    DOMAINS,
    RAMP_INVARIANT_DOMAINS,
    GradientConfig,
    image_gradient,
    prepare_field,
    register_stack,
    register_translation,
    taper_window,
)

TRUE_SHIFT = (1.3, -2.7)  # (dy, dx), in pixels


def synthetic_projection(shape=(96, 96), seed=3, peak_phase=5.0) -> np.ndarray:
    """A compactly supported "phase projection": structure inside, flat vacuum outside.

    The vacuum border is not decoration. A field that runs to the frame edge makes the
    FFT's periodic wrap and the taper's roll-off both matter, and then *every* method
    picks up an edge bias that has nothing to do with the effect under test. Real phase
    projections are padded into vacuum for exactly the same reason.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros(shape)
    v0, v1 = shape[0] // 5, 4 * shape[0] // 5
    u0, u1 = shape[1] // 4, 3 * shape[1] // 4
    image[v0:v1, u0:u1] = rng.random((v1 - v0, u1 - u0))
    image = gaussian_filter(image, 3.0)
    image -= image.min()
    image /= image.max()
    return image * peak_phase


def fourier_translate(image: np.ndarray, shift) -> np.ndarray:
    """Exact translation of a band-limited image (no interpolation error to confound)."""
    return np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real


def add_ramp(image: np.ndarray, amplitude: float) -> np.ndarray:
    """Add ``a*x + b*y + c`` with ``a = amplitude`` radians across the frame."""
    n_v, n_u = image.shape
    v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float64)
    return image + amplitude * (u / n_u) + 0.37 * amplitude * (v / n_v) + 3.0 * amplitude


@pytest.fixture(scope="module")
def pair():
    """A projection and an exactly translated copy of it."""
    image = synthetic_projection()
    return image, fourier_translate(image, TRUE_SHIFT)


# -- conventions --------------------------------------------------------------------


def test_sign_convention_matches_skimage_and_the_engine(pair):
    """THE convention pin. Getting this backwards negates every update.

    ``register_translation(reference, moving)`` must return what
    ``phase_cross_correlation(reference, moving)`` returns, because the engine calls the
    latter as ``(measured, simulated)`` and *adds* the result to the cumulative shifts.
    """
    skimage_registration = pytest.importorskip("skimage.registration")
    image, moved = pair

    theirs, _error, _phase = skimage_registration.phase_cross_correlation(
        image, moved, upsample_factor=100, normalization=None
    )
    ours = register_translation(
        image, moved, GradientConfig(domain="value", upsample=100, taper=0.0)
    )

    np.testing.assert_allclose([ours.dy, ours.dx], theirs, atol=1e-9)
    # And both must equal minus the applied translation: moved(v) = image(v - s), so the
    # shift that registers `moved` onto `image` is -s.
    np.testing.assert_allclose([ours.dy, ours.dx], [-TRUE_SHIFT[0], -TRUE_SHIFT[1]], atol=1e-9)


@pytest.mark.parametrize("domain", ["value", "gradient-x", "gradient-y", "gradient-both"])
def test_recovers_a_known_subpixel_shift(pair, domain):
    """Every domain must find the truth on clean data. Untapered, so the field is fair."""
    image, moved = pair
    result = register_translation(
        image, moved, GradientConfig(domain=domain, upsample=100, taper=0.0)
    )
    np.testing.assert_allclose(
        [result.dy, result.dx], [-TRUE_SHIFT[0], -TRUE_SHIFT[1]], atol=1.0 / 100
    )
    assert result.quality > 2.0, "the correlation peak did not stand out from the surface"


def test_a_taper_is_safe_on_a_gradient_and_not_on_a_value(pair):
    """Windowing is only translation-equivariant for a field that vanishes in the roll-off.

    A phase *gradient* does vanish there -- it is zero wherever the phase is flat, i.e.
    in the vacuum -- so the taper multiplies both images by the same thing where they
    have content, and the peak does not move. A phase *value* does not vanish: the
    object's own pedestal reaches into the roll-off and gets scaled differently in the
    two images being compared.

    This is a second, independent reason to correlate the gradient, and it is why the
    default ``taper=0.1`` is safe for the default ``domain="gradient-x"`` but would
    quietly cost ~0.7 px if the same config were pointed at the phase itself. Measured
    below on a clean, ramp-free pair -- no gauge ambiguity involved at all.
    """
    image, moved = pair
    truth = np.array([-TRUE_SHIFT[0], -TRUE_SHIFT[1]])

    tapered_gradient = register_translation(
        image, moved, GradientConfig(domain="gradient-x", upsample=100, taper=0.1)
    )
    tapered_value = register_translation(
        image, moved, GradientConfig(domain="value", upsample=100, taper=0.1)
    )

    np.testing.assert_allclose([tapered_gradient.dy, tapered_gradient.dx], truth, atol=0.01)
    value_error = float(np.max(np.abs(np.array([tapered_value.dy, tapered_value.dx]) - truth)))
    assert value_error > 0.1, (
        "expected the taper to bias the value domain; if this ever stops being true the "
        f"docstring's claim needs revisiting (error was {value_error:.3f} px)"
    )


# -- invariance: the point of the module --------------------------------------------


def test_a_constant_offset_changes_the_estimate_by_exactly_zero(pair):
    """d/dx of a constant is zero, so the estimate must not move AT ALL.

    Note this is the easy half: a constant lands entirely in the DC bin, so the value
    domain survives it too (both this module and skimage remove it). It is asserted
    anyway because a regression here -- e.g. dropping the mean subtraction, or the DC
    zeroing -- would be silent otherwise.
    """
    image, moved = pair
    config = GradientConfig(domain="gradient-x", upsample=100)
    base = register_translation(image, moved, config)
    for offset in (1.0, 1e3, 1e6):
        shifted = register_translation(image, moved + offset, config)
        assert abs(shifted.dx - base.dx) == 0.0, f"a constant offset of {offset} moved dx"
        assert abs(shifted.dy - base.dy) == 0.0, f"a constant offset of {offset} moved dy"


RAMP_AMPLITUDES = (0.0, 0.1, 1.0, 5.0, 10.0, 100.0)


def _ramp_sensitivity(image, moved, config) -> dict[float, float]:
    """|estimated dx - truth| as a function of ramp amplitude added to ``moved``."""
    truth = -TRUE_SHIFT[1]
    return {
        amplitude: abs(register_translation(image, add_ramp(moved, amplitude), config).dx - truth)
        for amplitude in RAMP_AMPLITUDES
    }


@pytest.mark.parametrize("domain", ["gradient-x", "gradient-y", "gradient-both"])
def test_a_linear_ramp_leaves_the_gradient_estimate_unchanged(pair, domain):
    """The headline claim, quantified against ramp amplitude.

    d/dx of ``a*x + b*y + c`` is the constant ``a``, and the registration subtracts the
    mean of the gradient image before correlating, so the ramp is annihilated exactly
    rather than approximately. The tolerance is set at 1e-6 px -- six orders of
    magnitude below the 1/3-voxel accuracy target -- and the measured value is ~4e-16 px
    at every amplitude, i.e. floating-point noise. Covers every domain listed in
    ``RAMP_INVARIANT_DOMAINS``, including the default.
    """
    image, moved = pair
    assert domain in RAMP_INVARIANT_DOMAINS
    config = GradientConfig(domain=domain, upsample=100, taper=0.0)
    errors = _ramp_sensitivity(image, moved, config)

    for amplitude, error in errors.items():
        assert error < 1e-6, (
            f"a ramp of {amplitude} rad across the frame moved the gradient-domain "
            f"estimate by {error:.3e} px"
        )
    # No trend with amplitude either: the sensitivity is not merely small, it is absent.
    assert max(errors.values()) - min(errors.values()) < 1e-6


def test_the_same_ramp_biases_the_value_domain(pair):
    """The contrast. Without this, the previous test proves nothing interesting.

    Same images, same correlation code, same sub-pixel refinement -- only ``domain``
    differs. The value-domain estimate degrades steadily and then breaks down entirely:
    by 10 rad it has stopped tracking the sample and locked onto the ramp.
    """
    image, moved = pair
    value = _ramp_sensitivity(
        image, moved, GradientConfig(domain="value", upsample=100, taper=0.0)
    )
    gradient = _ramp_sensitivity(
        image, moved, GradientConfig(domain="gradient-x", upsample=100, taper=0.0)
    )

    assert value[0.0] < 1e-6, "the baseline must be exact, or the comparison is unfair"
    assert value[1.0] > 0.3, f"expected a visible bias at 1 rad, got {value[1.0]:.3f} px"
    assert value[10.0] > 5.0, f"expected breakdown at 10 rad, got {value[10.0]:.3f} px"
    # Monotone degradation, not a fluke of one amplitude.
    assert value[0.1] < value[1.0] < value[5.0] < value[10.0]

    for amplitude in (1.0, 5.0, 10.0, 100.0):
        assert gradient[amplitude] < 1e-6 < value[amplitude]


def test_the_value_domain_baseline_is_skimage(pair):
    """Pin the bias numbers to skimage, so the failing baseline is the real one.

    The engine registers with ``phase_cross_correlation(..., normalization=None)``. If
    this module's ``domain="value"`` path did not reproduce it, the contrast above would
    only prove that a home-made baseline is bad.
    """
    skimage_registration = pytest.importorskip("skimage.registration")
    image, moved = pair
    config = GradientConfig(domain="value", upsample=100, taper=0.0)

    for amplitude in (1.0, 10.0):
        ramped = add_ramp(moved, amplitude)
        theirs, _error, _phase = skimage_registration.phase_cross_correlation(
            image, ramped, upsample_factor=100, normalization=None
        )
        ours = register_translation(image, ramped, config)
        np.testing.assert_allclose([ours.dy, ours.dx], theirs, atol=1e-9)


def test_gradient_magnitude_forfeits_the_ramp_invariance(pair):
    """|grad(phi + ramp)| != |grad phi| + const -- the magnitude is nonlinear.

    Encoded as a test because "use the gradient" is not by itself the right advice: the
    invariance comes from linearity, and taking the magnitude throws it away. The
    module's ``RAMP_INVARIANT_DOMAINS`` says which domains actually deliver it, and this
    checks the claim is true rather than merely written down.
    """
    image, moved = pair
    assert "gradient-magnitude" not in RAMP_INVARIANT_DOMAINS
    assert "gradient-magnitude" in DOMAINS

    magnitude = _ramp_sensitivity(
        image, moved, GradientConfig(domain="gradient-magnitude", upsample=100, taper=0.0)
    )
    assert magnitude[0.0] < 1e-6  # fine with no ramp
    assert magnitude[100.0] > 1.0, "expected the magnitude domain to break under a big ramp"

    assert GradientConfig(domain="gradient-x").is_ramp_invariant
    assert not GradientConfig(domain="gradient-magnitude").is_ramp_invariant
    assert not GradientConfig(domain="value").is_ramp_invariant


# -- the traps the module docstring calls out ---------------------------------------


def test_smoothing_trades_exactness_for_noise_suppression(pair):
    """Quantifies the cost of ``sigma > 0``, which is why it is off by default.

    A Gaussian filter with any finite-support boundary rule does not return a constant
    for a linear ramp near the frame edge, so the annihilation stops being exact there.
    The taper hides this while ``sigma`` is small compared with the roll-off width; once
    it is not, the ramp leaks back in.
    """
    image, moved = pair
    ramped = add_ramp(moved, 10.0)

    sensitivity = {}
    for sigma in (0.0, 1.0, 8.0):
        config = GradientConfig(domain="gradient-x", upsample=100, taper=0.1, sigma=sigma)
        clean = register_translation(image, moved, config).dx
        with_ramp = register_translation(image, ramped, config).dx
        sensitivity[sigma] = abs(with_ramp - clean)
        assert not config.is_ramp_invariant if sigma > 0 else config.is_ramp_invariant

    assert sensitivity[0.0] == 0.0
    assert sensitivity[1.0] < 1e-6, "sigma well inside the taper must still be exact"
    assert sensitivity[8.0] > 1e-3, (
        "a sigma comparable with the taper width must show the leak this test documents; "
        f"got {sensitivity[8.0]:.3e} px"
    )


def test_the_mean_is_subtracted_before_the_taper(pair):
    """Order trap: mean first, window second.

    If the window were applied first, the ramp's constant ``a`` would become ``a*w(x,y)``
    -- not a constant, not removable by subtracting a mean -- and the invariance would
    quietly disappear only when tapering is on. Tapering is on by default.
    """
    image, moved = pair
    config = GradientConfig(domain="gradient-x", upsample=100, taper=0.25)
    base = register_translation(image, moved, config)
    ramped = register_translation(image, add_ramp(moved, 10.0), config)
    assert abs(ramped.dx - base.dx) < 1e-9

    # And the mechanism itself: the prepared field of a ramped image is identical to
    # that of the clean one.
    clean_field = prepare_field(moved, config)[0]
    ramped_field = prepare_field(add_ramp(moved, 10.0), config)[0]
    np.testing.assert_allclose(clean_field, ramped_field, atol=1e-10)


def test_gradient_is_exact_for_a_linear_function_including_at_the_border():
    """``np.gradient`` uses one-sided differences at the border, exact for a plane.

    This is why the derivative is taken with plain finite differences and not a Sobel or
    a Gaussian-derivative kernel: those are not exact at the frame edge, and the ramp
    would leak back in through the border rows.
    """
    n_v, n_u = 40, 50
    v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float64)
    plane = 2.5 * u - 1.25 * v + 7.0

    np.testing.assert_allclose(image_gradient(plane, axis=1), 2.5, atol=1e-12)
    np.testing.assert_allclose(image_gradient(plane, axis=0), -1.25, atol=1e-12)


def test_taper_window_is_unity_inside_and_zero_at_the_edge():
    window = taper_window((64, 80), 0.2)
    assert window.shape == (64, 80)
    assert window[0, 0] == pytest.approx(0.0)
    assert window[-1, -1] == pytest.approx(0.0)
    assert window[32, 40] == pytest.approx(1.0)
    np.testing.assert_allclose(taper_window((16, 16), 0.0), 1.0)


# -- stack level ---------------------------------------------------------------------


def test_register_stack_returns_per_angle_updates_in_the_engine_convention():
    image = synthetic_projection(shape=(48, 48), seed=11)
    truth = np.array([[0.0, 0.0], [1.5, -2.25], [-0.75, 3.0], [0.4, 0.6]])
    measured = np.stack([fourier_translate(image, tuple(s)) for s in truth])
    simulated = np.stack([image] * len(truth))

    dsy, dsx, results = register_stack(
        measured, simulated, GradientConfig(domain="gradient-both", upsample=50, taper=0.1)
    )

    # measured(v) = simulated(v - s), so the correction to add is +s (engine convention).
    np.testing.assert_allclose(dsy, truth[:, 0], atol=0.05)
    np.testing.assert_allclose(dsx, truth[:, 1], atol=0.05)
    assert len(results) == len(truth)
    assert all(np.isfinite(r.quality) for r in results)


def test_register_stack_rejects_mismatched_shapes():
    a = np.zeros((3, 8, 8))
    with pytest.raises(ValueError, match="shape mismatch"):
        register_stack(a, np.zeros((3, 8, 9)))
    with pytest.raises(ValueError, match="n_angles"):
        register_stack(np.zeros((8, 8)), np.zeros((8, 8)))


def test_bad_configuration_fails_loudly_rather_than_degrading():
    with pytest.raises(ValueError, match="domain"):
        GradientConfig(domain="gradiant-x")  # typo
    with pytest.raises(ValueError, match="normalization"):
        GradientConfig(normalization="whiten")
    with pytest.raises(ValueError, match="taper"):
        GradientConfig(taper=1.5)
    with pytest.raises(ValueError, match="upsample"):
        GradientConfig(upsample=0)


def test_max_shift_guards_against_a_wild_lock_on(pair):
    """A half-converged reprojection can produce a huge, meaningless correlation peak."""
    image, moved = pair
    far = fourier_translate(image, (0.0, 20.0))
    config = GradientConfig(domain="gradient-x", upsample=20, max_shift=5.0)
    result = register_translation(image, far, config)
    assert abs(result.dx) <= 6.0, "the search was not confined to +/- max_shift"


def test_phase_normalization_also_recovers_the_shift(pair):
    image, moved = pair
    result = register_translation(
        image, moved, GradientConfig(domain="gradient-x", upsample=100, normalization="phase")
    )
    np.testing.assert_allclose([result.dy, result.dx], [-TRUE_SHIFT[0], -TRUE_SHIFT[1]], atol=0.05)
