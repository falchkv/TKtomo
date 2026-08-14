# Track model (`tktomo-track-model`)

Fit a tomography acquisition model from hand-labeled feature tracks, watch
the residuals respond to every parameter, and export the result as a model
file, a slogger-compatible shift table, ASTRA vector geometry, or an
aligned projection stack. Built for datasets where automatic alignment
fails and the missing information is long-baseline correspondence only a
human can supply.

## The model

For feature i at view j (angles theta in radians, coordinates in RAW
detector pixels):

    s  = a_i cos(theta_j) + b_i sin(theta_j)     across the beam
    t  = -a_i sin(theta_j) + b_i cos(theta_j)    along the beam
    u  = s + c(theta_j) + dx_j
    v  = y_i + alpha(theta_j) s + beta(theta_j) t + dy_j

`c` is the rotation-axis position, `alpha` the in-plane tilt, `beta` the
out-of-plane tilt, each a polynomial in angle of user-chosen degree
(constant by default). `dx`/`dy` are per-view shifts. The fit is two
linear IRLS solves (Huber weights), no nonlinear optimizer: every view is
tied to every other through shared features, so there is nothing to chain
and nothing to diverge.

## Labeling

Digits 0 to 9 pick the active feature (the table picks any id), left click
or Space places it in the current view and advances N views (configurable),
Delete removes the nearest label in the view, arrows step views. Labels are
stored in raw coordinates through the stack's provenance chain, so they
survive reloading the same data at another binning or crop, including
feature-crop stacks from `tktomo-feature-isolation`.

## Fixed and free parameters

Every polynomial coefficient has a "fix" checkbox and an editable value;
`dx` and `dy` are fixed or freed as whole groups; a feature row can be
pinned, which fixes its (a, b, y). Editing any value re-evaluates the
residuals WITHOUT fitting, so the response to "what if the center were
here" is immediate; the next fit overwrites free values and respects fixed
ones.

Two consequences of the geometry to keep in mind (the app warns about
both):

- With `dx` free, the constant/cos/sin part of the shifts is
  indistinguishable from the object sitting elsewhere. The app regauges
  it into (c, a, b) so the reported center is canonical, but on a short
  angular arc the center is still barely separable from the feature
  amplitudes (warning W1). Pinning one feature at known coordinates, or
  fixing `dx`, breaks the gauge and makes the center a measurement.
- A coefficient that is FIXED while its shift group is FREE cannot affect
  the fit; the shifts absorb it (warning W3). Fix the shifts too if you
  want the fixed value to mean something.

`Run diagnostics` fits disjoint halves of the features separately and
reports the disagreement (center, tilts, shift curves) plus held-out
residuals. The half-split is the number to trust; the parametric sigma is
shown but loses every argument with it.

## Plots and views without labels

The middle panel holds two plot panes, each with a dropdown selecting
what it shows (defaults: dx and dy shifts). Available: dx/dy shifts,
labels per view, residual u/v colored by feature, per-view MAD spread,
axis center c(theta), tilts alpha/beta, residual histogram. Clicking in
any angle-axis plot jumps to the nearest frame, so a gap or a bad point
is one click from being looked at and labeled.

`dx`/`dy` are measured only where labels exist; unlabeled views are
filled by interpolation over angle. In the shift plots, dots are labeled
views (orange when only ONE label carries the view, so its shift is that
label verbatim), the dashed curve is the interpolation, and red base
ticks mark frames with NO labels at all. The "labels per view" plot
shows the coverage directly, with the same red ticks and an orange
guide line at two labels.

## Exports

- **Model + astra vectors**: the full fit in one HDF5 (coefficients,
  shifts, feature positions, labels, masks, provenance), plus a
  `(n_views, 12)` `parallel3d_vec` dataset whose ASTRA forward projection
  reproduces the fitted model exactly, tilts included, for a
  geometry-aware GPU reconstruction.
- **slogger shifts.h5**: `sy`/`sx` and center/tilt attrs in the pipeline's
  convention (`sx = -dx/b`, `center = (c_raw - u0 - (b-1)/2)/b`), directly
  consumable by the graphite-ball pipeline's recon stage.
- **Aligned projection stack**: undoes `dx`/`dy`, folds the `c(theta)`
  drift into per-view shifts and derotates the constant in-plane tilt, one
  affine resample per view. `beta` and any angle-dependence of `alpha`
  cannot be expressed as 2D image transforms; they stay in the metadata.
- **Session**: labels, model, masks and UI state in a small HDF5 next to
  the data; the stack itself is re-read from its source on load.

## Recon slice

The Recon tab reconstructs one detector row with tomopy gridrec at the
fitted center, after applying each view's full 2D correction: the dx/dy
shifts, the c(theta) drift, and derotation by the in-plane tilt
alpha(theta), so the slice responds to the tilt parameters. Only beta
stays out (it needs 3D geometry, in 2D nothing can honor it). A bin
selector mean-pools the slab before reconstruction (shifts, center and
row rescaled accordingly); cost falls roughly as bin cubed, so bin 2 or 4
makes live evaluation fluid and bin 1 is for the final look. Off by default; the
"Live" checkbox recomputes it (debounced) after every change. All
reconstruction runs on a single worker thread with single-flight
scheduling: tomopy segfaults when called from two threads at once, so a
request arriving while one runs replaces the pending one instead of
queueing.

## 3D view

Fitted feature positions (a, b, y) in the rotating frame, the circle each
feature traces in the lab frame, the rotation axis, and every label
back-projected through the model as a residual cloud around its feature.
Needs PyOpenGL (in the `ui` extra); without it the tab shows a hint and
everything else works.
