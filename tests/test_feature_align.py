import numpy as np
import pytest

from tktomo.align import (
    Transform,
    estimate_feature_transform,
    estimate_rigid,
    forward_points,
)

SHAPE = (128, 160)


def _apply(points, transform):
    return forward_points(points, transform, SHAPE)


def test_estimate_rigid_recovers_rotation_and_translation():
    rng = np.random.default_rng(0)
    moving = rng.uniform(10, 100, size=(6, 2))
    truth = Transform(dx=7.0, dy=-4.0, rotation=12.0)
    fixed = _apply(moving, truth)

    est = estimate_rigid(moving, fixed, SHAPE)
    # Residual after the estimated transform should be ~0.
    residual = np.linalg.norm(_apply(moving, est) - fixed, axis=1)
    assert residual.max() < 1e-6
    assert est.rotation == pytest.approx(12.0, abs=1e-4)
    assert est.dx == pytest.approx(7.0, abs=1e-4)
    assert est.dy == pytest.approx(-4.0, abs=1e-4)


def test_single_labelled_mark_is_translation_only():
    truth = Transform(dx=5.0, dy=3.0)
    moving = np.array([[40.0, 50.0]])
    fixed = _apply(moving, truth)
    est = estimate_rigid(moving, fixed, SHAPE)
    assert est.rotation == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(_apply(moving, est) - fixed) < 1e-6


def test_no_marks_is_identity():
    result = estimate_feature_transform(
        [], [], [], [], image_shape=SHAPE, use_ransac=True
    )
    assert result.transform.is_identity()
    assert result.n_labelled == 0
    assert result.n_inliers == 0


def test_labelled_only_fit():
    truth = Transform(dx=-6.0, dy=8.0, rotation=-5.0)
    moving = np.array([[30.0, 40.0], [90.0, 70.0], [50.0, 100.0]])
    fixed = _apply(moving, truth)
    result = estimate_feature_transform(
        moving, fixed, [], [], image_shape=SHAPE, use_ransac=False
    )
    assert result.n_labelled == 3
    assert result.used_ransac is False
    assert result.rms_error < 1e-6
    assert np.linalg.norm(_apply(moving, result.transform) - fixed, axis=1).max() < 1e-6


def test_ransac_rejects_unlabelled_outliers():
    truth = Transform(dx=10.0, dy=-6.0, rotation=8.0)
    # Two labelled marks pin the model.
    lab_moving = np.array([[20.0, 30.0], [110.0, 40.0]])
    lab_fixed = _apply(lab_moving, truth)

    # Unlabelled inliers follow the same transform...
    inlier_moving = np.array([[60.0, 90.0], [80.0, 50.0], [100.0, 100.0]])
    inlier_fixed = _apply(inlier_moving, truth)
    # ...plus outlier marks on each side that do NOT correspond.
    outlier_moving = np.array([[15.0, 120.0]])
    outlier_fixed = np.array([[140.0, 15.0]])

    unlab_moving = np.vstack([inlier_moving, outlier_moving])
    unlab_fixed = np.vstack([inlier_fixed, outlier_fixed])

    result = estimate_feature_transform(
        lab_moving,
        lab_fixed,
        unlab_moving,
        unlab_fixed,
        image_shape=SHAPE,
        use_ransac=True,
        n_samples=200,
        threshold=2.0,
        seed=0,
    )
    assert result.used_ransac is True
    assert result.n_inliers == 3  # the three true inliers, outlier rejected
    # Recovered transform matches ground truth.
    assert result.rms_error < 1e-6
    check = np.linalg.norm(_apply(inlier_moving, result.transform) - inlier_fixed, axis=1)
    assert check.max() < 1e-4
