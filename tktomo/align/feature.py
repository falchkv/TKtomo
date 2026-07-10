"""Feature-based alignment from manually placed marks.

Two kinds of marks drive the fit:

* **Labelled** marks carry a label (a digit) and are matched across the two
  images by that label -- a known correspondence.
* **Unlabelled** marks have no identity; their correspondence is unknown and is
  recovered by RANSAC (mutual-nearest-neighbour putative matches, robustly
  filtered).

The estimated model is a rigid 2D :class:`~tktomo.align.transform.Transform`
(rotation about the image centre + translation) mapping the *moving* image onto
the *fixed* image, so ``apply_transform(moving_image, transform)`` overlays them.

Everything here is Qt-free and array-based so it is unit-testable and reusable.

Point convention: all point arrays are ``(N, 2)`` in ``(x, y) = (column, row)``
pixel coordinates -- the convention the UI reads from the mouse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tktomo.align.transform import Transform


@dataclass
class FeatureAlignmentResult:
    """Outcome of :func:`estimate_feature_transform`."""

    transform: Transform
    n_labelled: int  #: labelled correspondences used
    n_inliers: int  #: unlabelled correspondences kept as inliers
    n_putative: int  #: unlabelled putative correspondences considered
    rms_error: float  #: RMS residual over the correspondences fitted, pixels
    used_ransac: bool


def _as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2))
    return arr.reshape(-1, 2)


def _centre_rc(image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    return np.array([(height - 1) / 2.0, (width - 1) / 2.0])  # (row, col)


def forward_points(
    points, transform: Transform, image_shape: tuple[int, int]
) -> np.ndarray:
    """Map feature points the way :func:`apply_transform` moves image content.

    ``apply_transform`` is an inverse (pull) mapping, so a feature at input
    location ``p`` appears in the output at ``R(-phi)(p - c) + c + R(-phi) d``.
    Returns points as ``(N, 2)`` ``(x, y)``.
    """
    pts = _as_points(points)
    if len(pts) == 0:
        return pts
    rc = pts[:, ::-1]  # (x, y) -> (row, col)
    centre = _centre_rc(image_shape)
    phi = np.deg2rad(transform.rotation)
    cos, sin = np.cos(phi), np.sin(phi)
    rot = np.array([[cos, -sin], [sin, cos]])  # apply_transform's `rot` = R(phi)
    shift = np.array([transform.dy, transform.dx])  # (row, col)
    # R(-phi)(rc - c) for row-vectors is (rc - c) @ rot ; plus R(-phi) shift.
    out_rc = (rc - centre) @ rot + centre + rot.T @ shift
    return out_rc[:, ::-1]  # (row, col) -> (x, y)


def estimate_rigid(
    moving, fixed, image_shape: tuple[int, int]
) -> Transform:
    """Least-squares rigid ``Transform`` aligning ``moving`` pts onto ``fixed``.

    Uses the Kabsch/Procrustes solution (rotation + translation, no scale).
    With a single correspondence only a translation is recovered; with none the
    identity is returned.
    """
    M = _as_points(moving)
    F = _as_points(fixed)
    if len(M) != len(F):
        raise ValueError("moving and fixed must have the same number of points")
    if len(M) == 0:
        return Transform()

    M_rc = M[:, ::-1]
    F_rc = F[:, ::-1]
    if len(M) == 1:
        rot = np.eye(2)
        trans = F_rc[0] - M_rc[0]
    else:
        mu_m = M_rc.mean(axis=0)
        mu_f = F_rc.mean(axis=0)
        cov = (M_rc - mu_m).T @ (F_rc - mu_f)
        u, _s, vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, d]) @ u.T  # maps moving -> fixed
        trans = mu_f - rot @ mu_m

    # Re-express (rot, trans) as a Transform about the image centre.
    centre = _centre_rc(image_shape)
    theta = np.arctan2(rot[1, 0], rot[0, 0])
    trans_centre = rot @ centre + trans - centre
    shift = rot.T @ trans_centre  # (dy, dx)
    return Transform(dx=float(shift[1]), dy=float(shift[0]), rotation=float(np.degrees(-theta)))


def _residuals(moving, fixed, transform, image_shape) -> np.ndarray:
    proj = forward_points(moving, transform, image_shape)
    return np.linalg.norm(proj - _as_points(fixed), axis=1)


def _rms(moving, fixed, transform, image_shape) -> float:
    if len(_as_points(moving)) == 0:
        return float("nan")
    res = _residuals(moving, fixed, transform, image_shape)
    return float(np.sqrt(np.mean(res**2)))


def _putative_matches(moving, fixed, transform, image_shape):
    """Mutual-nearest-neighbour correspondences after applying ``transform``."""
    M = _as_points(moving)
    F = _as_points(fixed)
    if len(M) == 0 or len(F) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    proj = forward_points(M, transform, image_shape)
    dist = np.linalg.norm(proj[:, None, :] - F[None, :, :], axis=2)  # (P, Q)
    nn_for_m = dist.argmin(axis=1)  # nearest fixed for each moving
    nn_for_f = dist.argmin(axis=0)  # nearest moving for each fixed
    keep_m, keep_f = [], []
    for i, j in enumerate(nn_for_m):
        if nn_for_f[j] == i:  # mutual
            keep_m.append(M[i])
            keep_f.append(F[j])
    if not keep_m:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.array(keep_m), np.array(keep_f)


def _ransac_unlabelled(
    labelled_moving,
    labelled_fixed,
    put_moving,
    put_fixed,
    image_shape,
    n_samples,
    threshold,
    seed,
):
    """RANSAC over putative unlabelled correspondences; returns inlier subset."""
    P = len(put_moving)
    if P == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    rng = np.random.default_rng(seed)
    n_labelled = len(labelled_moving)
    # Minimal extra correspondences needed to pin down a rigid model (2 points).
    sample_k = min(P, max(1, 2 - n_labelled))

    best_mask = np.zeros(P, dtype=bool)
    best_count = -1
    best_rms = np.inf
    for _ in range(int(n_samples)):
        idx = rng.choice(P, size=sample_k, replace=False)
        samp_m = np.vstack([labelled_moving, put_moving[idx]])
        samp_f = np.vstack([labelled_fixed, put_fixed[idx]])
        model = estimate_rigid(samp_m, samp_f, image_shape)
        res = _residuals(put_moving, put_fixed, model, image_shape)
        mask = res <= threshold
        count = int(mask.sum())
        rms = float(np.sqrt(np.mean(res[mask] ** 2))) if count else np.inf
        if count > best_count or (count == best_count and rms < best_rms):
            best_count, best_rms, best_mask = count, rms, mask
    return put_moving[best_mask], put_fixed[best_mask]


def estimate_feature_transform(
    labelled_moving,
    labelled_fixed,
    unlabelled_moving,
    unlabelled_fixed,
    *,
    image_shape: tuple[int, int],
    use_ransac: bool = False,
    n_samples: int = 500,
    threshold: float = 5.0,
    seed: int | None = 0,
) -> FeatureAlignmentResult:
    """Estimate a rigid transform from labelled + unlabelled marks.

    Parameters
    ----------
    labelled_moving, labelled_fixed:
        ``(L, 2)`` arrays of labelled marks, matched row-for-row by label.
    unlabelled_moving, unlabelled_fixed:
        ``(P, 2)`` / ``(Q, 2)`` arrays of unlabelled marks (unknown pairing).
    image_shape:
        ``(height, width)`` of the images (rotation is about their centre).
    use_ransac:
        If true, recover unlabelled correspondences by RANSAC and fold the
        inliers into the fit.
    n_samples:
        Number of RANSAC iterations (random correspondence hypotheses to try).
    threshold:
        Inlier residual threshold in pixels.
    """
    LM = _as_points(labelled_moving)
    LF = _as_points(labelled_fixed)
    if len(LM) != len(LF):
        raise ValueError("labelled moving/fixed must be matched one-to-one")
    UM = _as_points(unlabelled_moving)
    UF = _as_points(unlabelled_fixed)

    base = estimate_rigid(LM, LF, image_shape) if len(LM) else Transform()

    inlier_m = np.empty((0, 2))
    inlier_f = np.empty((0, 2))
    n_putative = 0
    used = bool(use_ransac and len(UM) and len(UF))
    if used:
        put_m, put_f = _putative_matches(UM, UF, base, image_shape)
        n_putative = len(put_m)
        if n_putative:
            inlier_m, inlier_f = _ransac_unlabelled(
                LM, LF, put_m, put_f, image_shape, n_samples, threshold, seed
            )

    fit_m = np.vstack([LM, inlier_m]) if len(LM) or len(inlier_m) else np.empty((0, 2))
    fit_f = np.vstack([LF, inlier_f]) if len(LF) or len(inlier_f) else np.empty((0, 2))
    transform = estimate_rigid(fit_m, fit_f, image_shape) if len(fit_m) else Transform()

    return FeatureAlignmentResult(
        transform=transform,
        n_labelled=len(LM),
        n_inliers=len(inlier_m),
        n_putative=n_putative,
        rms_error=_rms(fit_m, fit_f, transform, image_shape),
        used_ransac=used,
    )
