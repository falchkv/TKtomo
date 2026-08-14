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

## 4. Conventions pinned by tests

- slogger shifts: `sx = -dx_raw/b`, `sy = -dy_raw/b`,
  `center = (c_raw - u0 - (b-1)/2)/b`, attrs as in the pipeline's
  `save_shifts` (`tests/test_tracking_export.py`).
- ASTRA `parallel3d_vec`: detector axes are the DUAL basis of the model
  read-out functionals (`m1 = e_s`, `m2 = alpha e_s + beta e_t + e_z`),
  which reproduces the model exactly rather than to first order in the
  tilts. Verified to machine precision without astra installed.
- Aligned stack: `Transform(dx=-(dx + c(th) - c_ref), dy=-dy,
  rotation=-deg(alpha_0))`; `apply_transform` rotates the translation with
  the content, which is exactly shift-then-derotate (pre-rotating the
  shift applies the rotation twice; a test guards this).
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
