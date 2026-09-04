# Build spec: manual feature tracking apps (feature isolation + track model)

This is the spec the two apps were built from, kept for the reasoning.
User docs: `feature_isolation.md`, `track_model.md`.

## 1. What and why

Automatic alignment of a ptycho-tomography dataset (the graphite-ball
scan) failed in instructive ways: reprojection alignment locked onto
static non-sample structure, patch correlation decorrelates after ~16
views so fitted centers were unidentifiable from short tracks, and manual
inspection outperformed every automatic metric. The missing information is
long-baseline feature correspondence. These apps let a human supply it and
fit an explicit acquisition model with it.

Two apps, deliberately separate, connected only by an exported file:

- `tktomo-feature-isolation`: rough-track ONE feature via manual keyframes
  plus interpolation, export a moving crop window with metadata.
- `tktomo-track-model`: label many features (on the full stack or on a
  crop), fit the model, explore residuals, export.

## 2. Architecture constraints

- All math and file formats live in `tktomo/tracking/` and are Qt-free,
  unit-tested in a light environment (scipy/h5py imported lazily inside
  functions per the repo's layering rule). The app files are thin shells.
- The fit runs entirely in RAW detector coordinates. `CoordinateChain`
  (`tracking/coords.py`) is the single place converting loaded-frame
  clicks through crop/binning/per-view-origin provenance. Pixel-center
  rule: `raw = crop0 + index * b + (b - 1) / 2`.
- TKtomo must not import slogger code; `tracking/model.py` reimplements
  and generalizes the slogger `track_bundle.py` two-stage linear solver
  (same IRLS/Huber structure, polynomial axis drift and out-of-plane tilt
  added, fixed/free masks added).
- Reuse, not reinvention: `StackDisplay` (axis-mapping fix) as the image
  panel base, `Hdf5BrowserDialog` for foreign HDF5, `Crop`/`load_dataset`
  for loading, `Transform`/`apply_transform` for the aligned export,
  `tktomo.recon.get_backend("tomopy")` for the slice.

## 3. Model and gauge (the load-bearing part)

    u_ij = s_ij + c(th_j) + dx_j
    v_ij = y_i + alpha(th_j) s_ij + beta(th_j) t_ij + dy_j

with s/t the rotating-frame coordinates projected across/along the beam,
c/alpha/beta polynomials in normalized angle tau (degree 0 default). Stage
1 is linear in (a, b, c_k, dx); s and t are then known and stage 2 is
linear in (y, alpha_k, beta_k, dy). Fixed parameters move to the
right-hand side (a value plus a mask), so "fixed at zero" and "fixed at
the last fit" are the same mechanism.

Gauge, worked out: with dx free, span{P_0..P_Kc, cos, sin} of dx is
degenerate with the free c_k and uniform shifts of a/b; the solver
regauges it into (c, a, b) after each solve. A fixed c_k under free dx is
absorbed by dx and cannot affect the fit (warned, not silently accepted).
Pinning a feature's (a, b) breaks the cos/sin gauge and is the intended
way to inject a known center. Vertically only the mean of dy is gauge
(into y); alpha and beta are identifiable at every polynomial order
because their columns are scaled by each feature's own s or t.

Views without labels get no dx/dy columns; after the solve they are filled
by PCHIP interpolation over angle and flagged, so plots and exports can
distinguish measured from interpolated.

Per-view rotations (rot_horiz, rot_beam, rot_axis, rad, `AxisModel`):
the beam frame rotated relative to the object by w in its own axes,
p' = R(-w) p with p = (s, t, y) and R the Rodrigues rotation
(`rotation_matrices`, exact identity at w = 0 so the plain fit stays bit
for bit). rot_axis alone is theta + rot_axis exactly. `AxisModel.project`
is the one forward model (predict, the window's probe, the diagnostics).
With a rotation free `solve_model` runs a joint Gauss-Newton on all free
parameters (`_joint_pass`, u and v residuals stacked, Huber scales per
stage, three passes): columns are increments, the derivative of p' with
respect to a rotation increment dw is p' x dw, and each rotation carries
prior rows which `_masked_irls` keeps out of the Huber scale and out of
the returned residual (`n_data`). The prior is whitened: the block's
unknown is the increment in units of sigma/noise_px, the prior row is an
identity row with target minus the current angle in those units, and the
data columns carry the factor. With the raw angle as unknown a tight
sigma made the prior rows enormous, LSQR stopped short, residuals that
should be exactly zero were not, and the Huber scale went wrong. That
scale (`_huber_weights`) is estimated from the residuals a free
parameter did not absorb exactly: with staggered labels the absorbed
zeros can be the majority, the p90 floor then sits at zero and the fit
turns into an L1 problem driven by numerical dust. Gauge with
rotations: a constant rot_axis
is degenerate with rotating every (a, b), a constant rot_beam with
alpha_0, a constant rot_horiz with beta_0, and a per-view rot_beam or
rot_horiz with dy_j when a view's labels share s or t. The ridge prior
resolves all of them by minimum norm, so there is no new regauge, and
the horizontal regauge uses the rotated cos/sin profiles. Solving the
rotations inside the two stages instead (rot_beam in v only) was
measured to let the stages trade error back and forth as the prior
loosened, hence the joint solve. Beyond a prior rms of about ten degrees
the angles drift into near-degenerate directions with the feature
heights and tilts, which is why the window caps sigma there. The mask
defaults the rotations to fixed, so `FreeMask.all_free` and a missing
mask still give the plain fit.

## 4. Conventions pinned by tests

- slogger shifts: `sx = -dx_raw/b`, `sy = -dy_raw/b`,
  `center = (c_raw - u0 - (b-1)/2)/b`, attrs as in the pipeline's
  `save_shifts` (`tests/test_tracking_export.py`).
- ASTRA `parallel3d_vec`: detector axes are the DUAL basis of the model
  read-out functionals (`m1 = e_s`, `m2 = alpha e_s + beta e_t + e_z`),
  which reproduces the model exactly rather than to first order in the
  tilts. With per-view rotations the beam basis is rotated first,
  e_k' = sum_l Rm[k, l] e_l with Rm = R(-w), and the same closed forms
  hold in the primed basis. Verified to machine precision without astra
  installed, rotations included.
- Aligned stack: `Transform(dx=-(dx + c(th) - c_ref), dy=-dy,
  rotation=-deg(alpha_0 + rot_beam_j))`. `apply_transform` rotates the
  translation with the content, which is exactly shift-then-derotate
  (pre-rotating the shift applies the rotation twice, a test guards
  this). The model rotates about (c, 0) in raw px and `Transform` about
  the image centre. For the per-view part of the angle that difference is
  compensated by `rotation_centre_shift` (a test guards this too), the
  constant part for alpha_0 stays the harmless constant offset it was.
  `plan_slice` does the same about the slab centre and passes rot_axis to
  gridrec as `SliceRequest.dtheta`.
- Feature-crop coordinate round trip: a label on the crop equals the same
  pixel labeled on the source stack, in raw coordinates.

## 5. Threading rules

TomoPy segfaults when reconstructing from two threads and stalls the GUI
for seconds when called on it. All reconstruction goes through one
persistent `ReconWorker` thread with single-flight scheduling (a pending
request is replaced, never queued). The solver itself runs on the GUI
thread: at manual-label scale (tens of features, hundreds of labels) a
solve is tens of milliseconds, measured 35 ms for 33 labels on the real
410-view stack.

## 6. Testing

- `tests/test_tracking_model.py`: synthetic ground-truth round trips,
  gauge-injection invariance, mask semantics, robustness to outliers.
- `tests/test_tracking_io.py`: coordinate chain, label store, crop format.
- `tests/test_tracking_export.py`: the conventions of section 4.
- `tests/test_tracking_apps.py`: pytest-qt behavior tests through the same
  slots the pointer uses, an App A to App B round trip, recon
  single-flight, and an end-to-end phantom with known injected shifts
  recovered to machine precision modulo gauge.
