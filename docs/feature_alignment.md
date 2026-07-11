# Feature-based alignment — method summary

The **feature alignment app** (`tktomo.ui.feature_alignment_app`) aligns a pair
of 2D images by fitting a rigid transform to correspondences the user marks by
hand. The geometry is implemented Qt-free in `tktomo/align/feature.py` and is
unit-tested independently of the UI.

## Marks

Two kinds of marks are placed with the keyboard while hovering the mouse over an
image:

| Mark | Placed with | Meaning |
| --- | --- | --- |
| **Labelled** | number key `0`–`9` | Carries a label. The mark with a given label on the *fixed* image corresponds to the same-labelled mark on the *moving* image — a **known** correspondence. A label is unique per image (re-pressing a digit moves that label). |
| **Unlabelled** | `a` | No identity. Which unlabelled mark on one image matches which on the other is **unknown** and is recovered automatically. |

`Delete` / `Backspace` removes the mark nearest the cursor.

## Transform model

The model is a **rigid 2D transform** — rotation about the image centre plus
translation — represented by `Transform(dx, dy, rotation)` and applied with the
existing `apply_transform`. It maps the *moving* image onto the *fixed* image, so
`apply_transform(moving, transform)` produces the overlay. (No scale/shear: this
matches the tomographic use case and the shared `Transform` type.)

Points are handled in `(x, y) = (column, row)` pixel coordinates. The point-space
forward map is derived to be exactly consistent with `apply_transform` (which is
an inverse/pull warp): a feature at `p` moves to
`R(-φ)(p − c) + c + R(-φ)·(dy, dx)`, where `c` is the image centre and `φ` the
rotation. This is validated against the real image warp to sub-pixel accuracy.

## Fitting

**1. Labelled marks → least squares.**
Labels present on *both* images give matched point pairs. A rigid transform is
fitted to them in closed form by the **Kabsch / orthogonal-Procrustes** solution
(SVD of the cross-covariance, with a reflection guard). Two or more pairs
determine rotation + translation; a single pair yields translation only; none
yields the identity.

**2. Unlabelled marks → RANSAC (optional).**
Unlabelled marks have unknown pairing, so correspondences are recovered
robustly:

1. **Putative matches.** The moving unlabelled marks are projected through the
   current (labelled-seed) transform and matched to fixed unlabelled marks by
   **mutual nearest neighbour** — a pair is kept only if each is the other's
   closest point. This suppresses obviously wrong matches before RANSAC.
2. **RANSAC.** For `N` iterations (the **samples slider**), a minimal set of
   correspondences is sampled (the labelled pairs, always included, plus enough
   random putative pairs to pin down the rigid model), a candidate transform is
   fitted, and its **inliers** — putative pairs whose residual is within the
   *inlier tolerance* (px) — are counted. The hypothesis with the most inliers
   (ties broken by lower RMS) wins.
3. **Refit.** The final transform is re-fitted by least squares on the labelled
   pairs plus all RANSAC inliers.

The **samples slider** trades robustness for speed: more iterations raise the
chance of finding the correct correspondence set among the unlabelled marks; the
Procrustes fit per iteration is cheap, so hundreds–thousands of samples are
interactive for the handful of marks typically placed.

If RANSAC is disabled, only the labelled marks are used.

## Overlay panel

The right-hand panel overlays the **fixed** image in green and the **aligned
(warped) moving** image in magenta. Where the two agree, green + magenta sum to
grey/white; residual colour fringes reveal misalignment, so the fit quality can
be judged by eye. The status line reports the labelled count, RANSAC
inliers/putative, the RMS residual, and the recovered `dx, dy, rotation`.

## API

```python
from tktomo.align import estimate_feature_transform

result = estimate_feature_transform(
    labelled_moving, labelled_fixed,        # (L, 2) each, matched by row
    unlabelled_moving, unlabelled_fixed,    # (P, 2) / (Q, 2), unmatched
    image_shape=(height, width),
    use_ransac=True, n_samples=500, threshold=5.0,
)
result.transform      # Transform mapping moving -> fixed
result.n_inliers      # unlabelled correspondences accepted
result.rms_error      # RMS residual (px) over fitted correspondences
```
