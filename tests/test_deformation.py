"""The definition of done for deformation fields: conventions, algebra, optical flow.

Most of these tests exist to pin down a *convention*, not a number. A deformation
field that is applied with the wrong sign, or estimated in the wrong direction, still
produces a smooth, plausible volume -- it is simply wrong by twice the deformation.
Nothing downstream can detect that, so it has to be caught here.

Numpy + scipy only: no tomopy, no scikit-image, no beamtime data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from tktomo.ptycho_align.core.deformation import (
    DeformationField,
    DeformationSequence,
    coarse_support_mask,
    compose,
    estimate_flow,
    invert,
    sequence_rms_difference,
    warp_volume,
)

SHAPE = (32, 40, 48)


def blobby_volume(shape=SHAPE) -> np.ndarray:
    """A volume with enough 3D structure that optical flow has something to lock onto.

    A single sphere would be nearly rotationally symmetric, so a flow estimate could
    slide along its surface unpunished; the off-centre blobs break that.
    """
    z, y, x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]].astype(np.float32)
    volume = np.zeros(shape, dtype=np.float32)
    for (cz, cy, cx), radius, amplitude in (
        ((0.50, 0.50, 0.50), 0.28, 1.0),
        ((0.32, 0.36, 0.34), 0.11, 0.8),
        ((0.66, 0.62, 0.66), 0.09, -0.6),
        ((0.44, 0.68, 0.38), 0.08, 0.9),
        ((0.60, 0.36, 0.62), 0.07, 0.7),
    ):
        inside = (
            (z - cz * shape[0]) ** 2 + (y - cy * shape[1]) ** 2 + (x - cx * shape[2]) ** 2
        ) < (radius * min(shape)) ** 2
        volume[inside] += amplitude
    return gaussian_filter(volume, 1.0)


def smooth_field(shape=SHAPE, spacing=10.0, amplitude=1.2, seed=0) -> DeformationField:
    grid = DeformationField.grid_for(shape, spacing)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(0.0, amplitude, (3, *grid)).astype(np.float32)
    return DeformationField(vectors, shape).smoothed(1.0)


# -- convention 1: warping is a pull-back ------------------------------------------


def test_warp_is_a_pull_back_not_a_push_forward():
    """THE sign test. ``out[p] = volume[p + u(p)]``, so +u moves content in -u.

    If this ever reads (18, 20, 24) instead of (14, 20, 24), every field in the
    package is the inverse of what its docstring claims, and the aligner will warp its
    volume away from the data while reporting that it converged.
    """
    field = DeformationField.zeros(SHAPE, spacing=8.0)
    vectors = field.vectors.copy()
    vectors[0] = 2.0  # sample two voxels further along z
    field = DeformationField(vectors, SHAPE)

    volume = np.zeros(SHAPE, dtype=np.float32)
    volume[16, 20, 24] = 1.0

    warped = warp_volume(volume, field)
    assert np.unravel_index(int(np.argmax(warped)), SHAPE) == (14, 20, 24)


def test_identity_field_leaves_the_volume_alone():
    volume = blobby_volume()
    warped = warp_volume(volume, DeformationField.zeros(SHAPE, spacing=8.0))
    np.testing.assert_allclose(warped, volume, atol=1e-5)


def test_components_map_to_volume_axes_in_z_y_x_order():
    volume = np.zeros(SHAPE, dtype=np.float32)
    volume[16, 20, 24] = 1.0
    for component, expected in enumerate([(13, 20, 24), (16, 17, 24), (16, 20, 21)]):
        vectors = np.zeros((3, *DeformationField.grid_for(SHAPE, 8.0)), dtype=np.float32)
        vectors[component] = 3.0
        warped = warp_volume(volume, DeformationField(vectors, SHAPE))
        assert np.unravel_index(int(np.argmax(warped)), SHAPE) == expected, (
            f"component {component} must move axis {component}"
        )


def test_warp_rejects_a_volume_the_field_was_not_built_for():
    field = DeformationField.zeros(SHAPE, spacing=8.0)
    with pytest.raises(ValueError, match="describes a"):
        warp_volume(np.zeros((8, 8, 8), dtype=np.float32), field)


# -- the coarse grid ----------------------------------------------------------------


def test_grid_spacing_sets_the_parameter_count():
    """The bias-variance knob. Halving the spacing multiplies the parameters by ~8."""
    coarse = DeformationField.zeros(SHAPE, spacing=16.0)
    fine = DeformationField.zeros(SHAPE, spacing=8.0)
    assert fine.vectors.size > 5 * coarse.vectors.size
    assert min(coarse.grid_shape) >= 2, "a 1-node axis would be a rigid translation"
    np.testing.assert_allclose(coarse.spacing, 16.0, rtol=0.35)
    np.testing.assert_allclose(fine.spacing, 8.0, rtol=0.35)


def test_dense_agrees_with_sample_at_the_same_points():
    field = smooth_field()
    dense = field.dense()
    points = np.stack(
        np.meshgrid(*[np.arange(s, dtype=np.float64) for s in SHAPE], indexing="ij"), axis=-1
    )
    np.testing.assert_allclose(field.sample(points), dense, atol=1e-5)


def test_sample_reproduces_the_node_values():
    field = smooth_field()
    axes = [
        np.linspace(0.0, extent - 1, nodes)
        for extent, nodes in zip(field.shape, field.grid_shape)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    np.testing.assert_allclose(field.sample(points), field.vectors, atol=1e-4)


# -- convention 3: compose fields, never re-warp volumes ----------------------------


def test_compose_reproduces_two_successive_warps():
    """One interpolation of the volume must match two, to a few percent.

    This is what makes "never re-warp warped data" affordable: chaining is done on the
    coarse vectors. The residual difference here IS the interpolation blur that
    composing avoids.
    """
    volume = blobby_volume()
    outer = smooth_field(seed=1, amplitude=0.9)
    inner = smooth_field(seed=2, amplitude=0.9)

    twice = warp_volume(warp_volume(volume, inner), outer)
    once = warp_volume(volume, compose(outer, inner))
    error = np.linalg.norm(twice - once) / np.linalg.norm(once)
    assert error < 0.05, f"composition disagrees with two warps by {error:.3f}"


def test_compose_with_the_inverse_is_the_identity_field():
    field = smooth_field(seed=3, amplitude=1.0)
    residual = compose(field, invert(field)).max_magnitude
    assert residual < 0.1, f"u o u^-1 left {residual:.3f} px of deformation"


def test_invert_round_trips_a_volume():
    volume = blobby_volume()
    field = smooth_field(seed=4, amplitude=0.8)
    there_and_back = warp_volume(warp_volume(volume, field), invert(field))
    error = np.linalg.norm(there_and_back - volume) / np.linalg.norm(volume)
    assert error < 0.1, f"warp then unwarp changed the volume by {error:.3f}"


def test_invert_refuses_a_folding_field():
    """A field that folds space has no inverse; returning the last iterate would lie."""
    grid = DeformationField.grid_for(SHAPE, 8.0)
    vectors = np.zeros((3, *grid), dtype=np.float32)
    vectors[1] = np.linspace(-30.0, 30.0, grid[1])[None, :, None]
    with pytest.raises(ValueError, match="did not converge"):
        invert(DeformationField(vectors, SHAPE), iterations=6)


def test_clipping_scales_vectors_without_rotating_them():
    field = smooth_field(seed=5, amplitude=6.0)
    clipped = field.clipped(1.5)
    assert clipped.max_magnitude <= 1.5 + 1e-5
    hot = field.magnitude > 1.5
    if hot.any():
        original = field.vectors[:, hot]
        capped = clipped.vectors[:, hot]
        cosine = (original * capped).sum(axis=0) / (
            np.linalg.norm(original, axis=0) * np.linalg.norm(capped, axis=0)
        )
        np.testing.assert_allclose(cosine, 1.0, atol=1e-5)


# -- convention 2: optical flow direction -------------------------------------------


@pytest.fixture(scope="module")
def flow_case():
    """A known smooth field, the volume it deforms, and the object support mask."""
    volume = blobby_volume()
    grid = DeformationField.grid_for(SHAPE, 10.0)
    ramp_z = np.sin(np.pi * np.linspace(0, 1, grid[0]))[:, None, None]
    ramp_x = np.cos(np.pi * np.linspace(0, 1, grid[2]))[None, None, :]
    truth = DeformationField(
        np.stack(
            [
                (1.5 * ramp_z * np.ones(grid)).astype(np.float32),
                (-1.0 * np.ones(grid)).astype(np.float32),
                (0.8 * ramp_x * np.ones(grid)).astype(np.float32),
            ]
        ),
        SHAPE,
    )
    # order=3 deliberately: the truth must not be made with the same interpolation the
    # estimator uses, or the test would only prove the code agrees with itself.
    deformed = warp_volume(volume, truth, order=3)
    support = coarse_support_mask(volume, grid, threshold=0.05)
    return volume, deformed, truth, support


def test_estimate_flow_recovers_a_known_field(flow_case):
    """THE GATE for the flow solver: sub-pixel inside the object.

    Judged inside the object support only, and the whole-grid number is asserted much
    more loosely: in empty air the data constrains nothing and the answer is whatever
    the smoothness term extrapolates. Odstrcil et al. quote ~0.8 px rms against
    simulated ground truth; this is the same measurement on a smaller phantom.
    """
    volume, deformed, truth, support = flow_case
    estimate = estimate_flow(deformed, volume, spacing=10.0, alpha=0.5)

    error = np.sqrt(((estimate.vectors - truth.vectors) ** 2).sum(axis=0))
    inside = float(np.sqrt((error[support] ** 2).mean()))
    everywhere = float(np.sqrt((error**2).mean()))
    assert inside < 0.8, f"{inside:.3f} px rms inside the object (truth {truth.rms_magnitude:.2f})"
    assert everywhere < 1.5, f"{everywhere:.3f} px rms over the whole grid"


def test_estimate_flow_direction_is_not_inverted(flow_case):
    """Swapping reference and moving must flip the field, not repeat it."""
    volume, deformed, _truth, support = flow_case
    forward = estimate_flow(deformed, volume, spacing=10.0, alpha=0.5)
    backward = estimate_flow(volume, deformed, spacing=10.0, alpha=0.5)
    a = forward.vectors[:, support].ravel()
    b = backward.vectors[:, support].ravel()
    assert np.corrcoef(a, b)[0, 1] < -0.8, "the two directions must be near-opposite"


def test_estimate_flow_warps_the_moving_volume_onto_the_reference(flow_case):
    volume, deformed, _truth, _support = flow_case
    estimate = estimate_flow(deformed, volume, spacing=10.0, alpha=0.5)
    before = np.linalg.norm(deformed - volume)
    after = np.linalg.norm(deformed - warp_volume(volume, estimate))
    assert after < 0.5 * before, f"flow did not reduce the mismatch ({after:.3g} vs {before:.3g})"


def test_regularisation_strength_is_a_parameter_that_does_something(flow_case):
    volume, deformed, _truth, support = flow_case
    loose = estimate_flow(deformed, volume, spacing=10.0, alpha=0.1)
    tight = estimate_flow(deformed, volume, spacing=10.0, alpha=20.0)
    assert tight.rms_magnitude < loose.rms_magnitude, (
        "a large alpha must produce a more conservative field"
    )


def test_estimate_flow_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        estimate_flow(np.zeros((8, 8, 8)), np.zeros((8, 8, 9)))


def test_estimate_flow_rejects_an_unknown_method():
    """Fails on the name before doing any work, so a typo is not a 20-minute wait."""
    with pytest.raises(ValueError, match="unknown flow_method"):
        estimate_flow(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), method="nope")


def test_estimate_flow_rejects_two_constant_volumes():
    with pytest.raises(ValueError, match="nothing to register"):
        estimate_flow(np.ones((8, 8, 8)), np.ones((8, 8, 8)))


def test_tvl1_is_optional_and_says_so():
    """skimage is an optional dependency; its absence must name the scipy fallback."""
    try:
        import skimage.registration  # noqa: F401, PLC0415
    except ImportError:
        with pytest.raises(ImportError, match="scikit-image"):
            estimate_flow(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), method="tvl1")
        return
    small = (16, 16, 16)
    volume = blobby_volume(small)
    grid = DeformationField.grid_for(small, 8.0)
    truth = DeformationField(
        np.stack([np.full(grid, v, dtype=np.float32) for v in (1.0, -0.8, 0.6)]), small
    )
    estimate = estimate_flow(
        warp_volume(volume, truth, order=3), volume, spacing=8.0, method="tvl1"
    )
    support = coarse_support_mask(volume, grid, threshold=0.05, sigma=2.0)
    error = np.sqrt(((estimate.vectors - truth.vectors) ** 2).sum(axis=0))
    assert float(np.sqrt((error[support] ** 2).mean())) < 1.0


# -- sequences in acquisition time --------------------------------------------------


def make_sequence(k=5, spacing=12.0, seed=0) -> DeformationSequence:
    grid = DeformationField.grid_for(SHAPE, spacing)
    rng = np.random.default_rng(seed)
    fields = tuple(
        DeformationField(rng.normal(0, 1.0, (3, *grid)).astype(np.float32), SHAPE)
        for _ in range(k)
    )
    return DeformationSequence(fields, np.arange(k, dtype=float))


def test_temporal_smoothing_damps_jumps_between_subsets():
    sequence = make_sequence()
    jumps = np.abs(np.diff(sequence.node_array, axis=0)).mean()
    smoothed = np.abs(np.diff(sequence.smoothed_in_time(1.5).node_array, axis=0)).mean()
    assert smoothed < 0.5 * jumps


def test_zero_mean_removes_the_unobservable_common_mode():
    sequence = make_sequence()
    shifted = DeformationSequence(
        tuple(
            DeformationField(f.vectors + 3.0, f.shape) for f in sequence.fields
        ),
        sequence.times,
    )
    np.testing.assert_allclose(
        sequence.zero_mean().node_array, shifted.zero_mean().node_array, atol=1e-5
    )
    assert abs(sequence.zero_mean().node_array.mean(axis=0)).max() < 1e-5


def test_interpolation_in_time_is_linear_and_clamped():
    sequence = make_sequence(k=3)
    middle = sequence.at(0.5).vectors
    expected = 0.5 * (sequence.fields[0].vectors + sequence.fields[1].vectors)
    np.testing.assert_allclose(middle, expected, atol=1e-5)
    # Outside the measured range the end field is held, never extrapolated.
    np.testing.assert_allclose(sequence.at(-99.0).vectors, sequence.fields[0].vectors)
    np.testing.assert_allclose(sequence.at(99.0).vectors, sequence.fields[-1].vectors)


def test_sequence_rejects_unsorted_times():
    sequence = make_sequence(k=3)
    with pytest.raises(ValueError, match="non-decreasing"):
        DeformationSequence(sequence.fields, np.array([2.0, 0.0, 1.0]))


def test_sequence_rejects_mismatched_grids():
    a = DeformationField.zeros(SHAPE, spacing=8.0)
    b = DeformationField.zeros(SHAPE, spacing=16.0)
    with pytest.raises(ValueError, match="share the volume shape"):
        DeformationSequence((a, b), np.array([0.0, 1.0]))


def test_rms_difference_ignores_the_gauge():
    sequence = make_sequence(seed=7)
    offset = DeformationSequence(
        tuple(DeformationField(f.vectors + 2.5, f.shape) for f in sequence.fields),
        sequence.times,
    )
    assert sequence_rms_difference(sequence, offset) < 1e-4


def test_support_mask_selects_the_object_and_not_the_air():
    volume = blobby_volume()
    grid = DeformationField.grid_for(SHAPE, 10.0)
    mask = coarse_support_mask(volume, grid)
    assert mask.any() and not mask.all()
    centre = tuple(s // 2 for s in grid)
    assert mask[centre], "the centre of the object must be inside the support"
    assert not mask[0, 0, 0], "the corner is air and must be outside"
