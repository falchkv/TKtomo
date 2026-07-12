# Build spec: `ptycho-align` — an interactive GUI for reprojection alignment of ptychographic tomography data

## 0. What we are building and why

We need a desktop GUI (or small collection of coupled GUI panels) that drives the **iterative reprojection alignment** workflow for ptychographic X-ray computed tomography (PXCT), built on TomoPy.

The scientific workflow being automated:

1. Load a stack of ptychographic **phase** projections + their rotation angles.
2. Preprocess them (phase ramp/offset removal, optional unwrap, optional crop/pad).
3. **Centre-of-mass (COM) pre-alignment** to get a good initial guess of the per-projection shifts and the rotation-axis position.
4. Run the **iterative reprojection alignment loop**:
   - reconstruct a 3D volume from the currently-aligned sinogram,
   - forward-project (reproject) that volume at every measured angle,
   - cross-correlate each measured projection against its own reprojection to get a subpixel shift,
   - accumulate the shift, re-shift the data, repeat.
5. Inspect, then continue or stop.

The **entire point of this tool** is that step 4 must be **inspectable between iterations**. The user must be able to run 1 iteration, look at the tomogram / projections / sinograms / shift estimates, then run 5 more, look again, and so on. That is the core requirement; everything else is supporting scaffolding.

---

## 1. CRITICAL architectural constraint — read this before writing any code

`tomopy.prep.alignment.align_joint()` and `align_seq()` run their full iteration loop **internally**, with no callback, no generator, and no way to pause. They cannot be used to satisfy the step-wise requirement.

**Therefore: do not call `align_joint` / `align_seq`. Reimplement the loop yourself** using TomoPy's building blocks, so that a single outer iteration is a callable function that returns control to the GUI.

Use exactly these primitives (this mirrors what TomoPy's own implementation does internally, so results will be comparable):

```python
from tomopy.recon.algorithm import recon          # reconstruction operator
from tomopy.sim.project import project            # forward-projection / reprojection operator
from tomopy.prep.alignment import blur_edges, shift_images
from skimage.registration import phase_cross_correlation   # subpixel registration
```

- `recon(prj, ang, center=..., algorithm='sirt', num_iter=...)` → volume
- `project(vol, ang, center=..., pad=False)` → simulated projections at the measured angles
- `phase_cross_correlation(reference_image, moving_image, upsample_factor=N)` → subpixel `(dy, dx)`
- `blur_edges(prj, rin, rout)` → tapers projection borders before registration (suppresses edge artifacts in the cross-correlation)
- `shift_images(prj, sy, sx)` → applies the shifts

Keep a clear TODO/comment noting that our loop is a re-implementation of the Gürsoy et al. (2017) joint re-projection algorithm as found in `tomopy.prep.alignment`, exposed one iteration at a time.

Also implement `align_seq`-style behaviour as a mode toggle: in the **joint** mode the reconstruction is *warm-started* (the volume carries over between outer iterations and gets a small number of inner reconstruction iterations each time); in the **sequential** mode the volume is reconstructed from scratch each outer iteration. Expose this as a radio button.

---

## 2. Technology stack

- **Python** ≥ 3.10.
- **GUI toolkit:** PySide6 (Qt6). Do not use tkinter.
- **Image/plot display:** `pyqtgraph` — it is fast enough for scrubbing through large 3D stacks with a slider, and gives free pan/zoom/level controls. Use `pyqtgraph.ImageView` for image panels and `pyqtgraph.PlotWidget` for the 1D plots.
- **Numerics:** numpy, scipy, scikit-image, tomopy.
- **Optional acceleration:** if `astra` is importable, allow it as an alternative reconstruction backend, but tomopy's own `recon` must be the default and the code must run without astra installed.
- **I/O:** h5py, tifffile.
- Package with `pyproject.toml`. Provide a console entry point `ptycho-align`.

**Threading is mandatory.** All heavy computation (reconstruction, reprojection, registration) runs in a `QThread` worker. The GUI must never block; the user must be able to see a progress bar and cancel a long run. Emit Qt signals for `iteration_started(i)`, `iteration_finished(IterationResult)`, `progress(float, str)`, and `run_finished()`.

---

## 3. Repository layout

```
ptycho_align/
    __init__.py
    core/
        __init__.py
        dataset.py        # ProjectionDataset: data container + I/O
        preprocess.py     # phase ramp removal, unwrap, crop, pad
        com.py            # centre-of-mass pre-alignment
        engine.py         # AlignmentEngine: THE step-wise reprojection loop
        state.py          # IterationResult, AlignmentState, history stack
        io.py             # session save/load
    gui/
        __init__.py
        main_window.py
        worker.py         # QThread wrapper around AlignmentEngine
        panels/
            data_panel.py         # load / preprocess controls
            control_panel.py      # algorithm params, step/run/stop buttons
            projection_view.py    # projection inspector
            sinogram_view.py      # sinogram inspector
            tomogram_view.py      # 3D volume slice inspector
            shift_view.py         # shift-vs-angle + convergence plots
    tests/
        test_com.py
        test_engine.py
        test_preprocess.py
examples/
    make_phantom.py       # generate a synthetic misaligned dataset for testing
pyproject.toml
README.md
```

The `core/` package **must be fully usable without the GUI** — i.e. someone can script the whole alignment from a Jupyter notebook. The GUI is a thin shell over it. Write `core/` first, test it, then build the GUI.

---

## 4. Core data model

### 4.1 `ProjectionDataset` (core/dataset.py)

```python
@dataclass
class ProjectionDataset:
    projections: np.ndarray   # float32, shape (n_theta, n_v, n_u); PHASE, not amplitude
    angles: np.ndarray        # float64, radians, shape (n_theta,)
    pixel_size_nm: float | None = None
    name: str = ""
```

Loaders required:
- `.npy` / `.npz` (projections + angles arrays)
- HDF5 (user supplies the dataset paths for projections and angles via the load dialog; provide a small tree browser or just two text fields with sensible defaults `/exchange/data` and `/exchange/theta`)
- directory of TIFFs + a separate angles file (`.txt`/`.csv`, degrees or radians — ask which, or autodetect: if max|angle| > 6.3 assume degrees)

Always convert angles to **radians** internally. Always store projections as **float32**.

### 4.2 `AlignmentState` and `IterationResult` (core/state.py)

```python
@dataclass
class IterationResult:
    iteration: int
    sx: np.ndarray            # cumulative horizontal shift per angle (pixels)
    sy: np.ndarray            # cumulative vertical shift per angle (pixels)
    dsx: np.ndarray           # shift *update* applied this iteration
    dsy: np.ndarray
    error: float              # e.g. RMS of the shift updates -> convergence metric
    residual: float           # ||measured - reprojected||_2 / ||measured||_2
    volume: np.ndarray        # the reconstruction produced this iteration
    center: float
    wallclock_s: float
```

`AlignmentState` holds: the original (preprocessed, never-modified) projections, the current cumulative `sx`/`sy`, the current volume, the current rotation-axis `center`, and a `history: list[IterationResult]`.

**Important:** always apply the *cumulative* shift to the *original* projections rather than repeatedly re-shifting an already-shifted array. Repeated interpolation blurs the data. The aligned sinogram at any time is `shift_images(original.copy(), sy_cumulative, sx_cumulative)`.

Provide `AlignmentState.revert_to(iteration: int)` so the user can roll back to an earlier iteration if the alignment diverges. The GUI exposes this as "revert to iteration N".

---

## 5. Preprocessing (core/preprocess.py)

Ptychographic phase projections need this before anything else works. Each function takes and returns a `(n_theta, n_v, n_u)` array and must be individually toggleable in the GUI.

1. `remove_phase_ramp(prj, mask=None)` — for each projection, fit a plane `a*u + b*v + c` to the phase over a background/vacuum region and subtract it. The background region defaults to a border frame of configurable width (in pixels), but the user must be able to draw a rectangular ROI on the projection view and use that instead. **This matters more than anything else** — a residual linear phase ramp is mathematically indistinguishable from a lateral shift, so leaving it in poisons the alignment.
2. `remove_phase_offset(prj, mask=None)` — subtract the mean over the background region (do this even if the ramp fit is off).
3. `unwrap_phase(prj)` — wrap `skimage.restoration.unwrap_phase` over each projection. Toggleable; off by default.
4. `crop(prj, roi)` and `pad(prj, pad_u, pad_v)` — padding is important because shifting can move the object toward the frame edge; recommend padding by ~10–20% before the loop and give a sensible default.
5. `invert(prj)` / `scale(prj)` — sign and scaling convenience, because phase is negative-valued for most samples and the reconstruction/COM code assumes positive mass. Provide an automatic "make mass positive" check that warns the user if the projection integral is negative.

Preprocessing produces the `original` array that the alignment loop treats as immutable input. Show a before/after in the projection view.

---

## 6. COM pre-alignment (core/com.py)

This is a required feature and must be runnable independently of the main loop (a "Run COM pre-alignment" button that populates the initial `sx`/`sy`).

Implement:

```python
def com_prealign(prj, angles) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (sx, sy, center)."""
```

**Vertical (`sy`, along the rotation axis):** for each projection compute the mass-weighted vertical centroid, `com_v[i] = sum(v * prj_i) / sum(prj_i)` where `v` is the row index. In an ideally-aligned dataset this is constant with angle (the object does not travel up and down as it rotates). So set `sy[i] = com_v.mean() - com_v[i]`. Offer a "median" variant as a robustness option.

**Horizontal (`sx`, perpendicular to the axis):** for a mass-conserving projection the horizontal centroid traces a sinusoid in angle:

```
com_u(theta) = a*sin(theta) + b*cos(theta) + c
```

Fit `a, b, c` by linear least squares (build the design matrix `[sin, cos, 1]` and use `np.linalg.lstsq`). Then:
- the rotation-axis position is `center = c` (in the projection's `u` coordinate),
- the residual of the fit is the misalignment: `sx[i] = fitted_com_u[i] - measured_com_u[i]`.

Phase projections **are** mass-conserving line integrals of the density, so this works well here and is the right initializer.

Guard rails:
- Compute the centroid on a background-subtracted, positive-mass version of the data. If `sum(prj_i)` is near zero, raise a clear error telling the user their offset removal is wrong.
- Plot the measured `com_u` points and the fitted sinusoid in the shift panel so the user can see the fit quality. A bad sinusoid fit is the earliest possible warning that something upstream is broken.
- Return `center` and push it into the engine as the initial rotation-axis estimate. Also expose TomoPy's `find_center_vo` / `find_center_pc` as alternative "estimate centre" buttons.

---

## 7. The alignment engine (core/engine.py) — THE CORE OF THE APP

```python
@dataclass
class AlignConfig:
    recon_algorithm: str = "sirt"     # sirt | mlem | gridrec | art | fbp (tomopy names)
    recon_inner_iters: int = 2        # tomopy `num_iter` per OUTER iteration
    mode: str = "joint"               # "joint" (warm-start volume) | "sequential" (fresh)
    upsample_factor: int = 20         # subpixel registration precision = 1/upsample px
    blur_edges: bool = True
    rin: float = 0.5
    rout: float = 0.8
    pad: tuple[int, int] = (0, 0)
    refine_center: bool = False       # re-run centre finding each iteration
    align_vertical: bool = True
    align_horizontal: bool = True
    shift_damping: float = 1.0        # multiply shift updates by this (< 1 stabilises)
    max_shift_per_iter: float | None = None   # clip outlier updates, in px
    median_filter_shifts: bool = False        # smooth shifts across angle
    ncore: int | None = None


class AlignmentEngine:
    def __init__(self, dataset, config, sx0=None, sy0=None, center=None): ...

    def step(self) -> IterationResult:
        """Run exactly ONE outer iteration. Must be cheap to call repeatedly."""

    def run(self, n: int, cancel_event=None, callback=None) -> list[IterationResult]:
        """Call step() n times, invoking callback(result) after each; abort if
        cancel_event is set. This is what the 'Run N iterations' button uses."""
```

### `step()` must do exactly this, in this order:

1. Build the currently-aligned sinogram: `prj_aligned = shift_images(self.original.copy(), self.sy, self.sx)`.
2. Reconstruct: `vol = recon(prj_aligned, angles, center=self.center, algorithm=cfg.recon_algorithm, num_iter=cfg.recon_inner_iters, init_recon=self.vol if cfg.mode=="joint" else None, ncore=cfg.ncore)`.
   (Check the current TomoPy signature for warm-starting — the parameter for supplying an initial volume is `init_recon`. If a chosen algorithm doesn't support it, fall back to a fresh reconstruction and log a warning rather than crashing.)
3. Reproject: `sim = project(vol, angles, center=self.center, pad=False)`. Make sure `sim` has the same shape as `prj_aligned`; crop/pad if TomoPy's `project` pads.
4. Optionally taper both: if `cfg.blur_edges`, apply `blur_edges(..., cfg.rin, cfg.rout)` to *copies* of `prj_aligned` and `sim` used only for registration — never to the data itself.
5. For each angle `i`, register: `(dy, dx), _, _ = phase_cross_correlation(sim_blur[i], prj_blur[i], upsample_factor=cfg.upsample_factor)`. **Sanity-check the sign convention against a synthetic test** (see §11) — getting it backwards is the single most likely bug and the alignment will visibly diverge.
6. Post-process the update: zero out `dx` if `align_horizontal` is False and `dy` if `align_vertical` is False; multiply by `shift_damping`; clip to `max_shift_per_iter`; optionally median-filter across angle.
7. Accumulate: `self.sx += dx; self.sy += dy`.
8. Remove the degenerate global modes so the solution doesn't drift: subtract the mean from `sy` (a constant vertical shift is a pure translation of the volume), and subtract the mean from `sx` OR fold it into `center` — pick one and document it. Do not let a slow global drift accumulate over 50 iterations.
9. Optionally refine `center` (if `cfg.refine_center`).
10. Compute `error = sqrt(mean(dx**2 + dy**2))` and `residual = ||prj_aligned - sim|| / ||prj_aligned||`; package everything into an `IterationResult`, append to history, return it.

Store the volume from each iteration in history, but **cap memory**: keep the volumes only for the last K iterations (default K=3, configurable) plus every Nth, and always keep the shift arrays for all iterations (they're tiny). Otherwise a 50-iteration run on a 1024³ volume will exhaust RAM. Make this policy explicit in the code and surface a memory estimate in the GUI before a run starts.

---

## 8. GUI layout

A single main window. Use a `QMainWindow` with dock widgets so the user can rearrange/undock panels; the central widget is a tab bar or a 2×2 splitter grid holding the four viewers.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ menu:  File   Data   Align   View   Help                                 │
├───────────────┬──────────────────────────────────────────────────────────┤
│               │  ┌────────────────────┬─────────────────────┐            │
│  LEFT DOCK    │  │  Projection view   │  Sinogram view      │            │
│               │  ├────────────────────┼─────────────────────┤            │
│  Data panel   │  │  Tomogram view     │  Shifts / converg.  │            │
│  Preprocess   │  └────────────────────┴─────────────────────┘            │
│  COM panel    │                                                          │
│  Align panel  │                                                          │
│  (params)     │                                                          │
│               │                                                          │
├───────────────┴──────────────────────────────────────────────────────────┤
│ [Step 1]  [Run  (5) iterations]  [Stop]  [Revert to iter (n)]            │
│ Iteration: 7   |  shift RMS: 0.043 px  |  residual: 0.081  |  ▓▓▓▓░░ 62% │
└──────────────────────────────────────────────────────────────────────────┘
```

### 8.1 Bottom action bar (the step-wise control — the heart of the UI)

- **Step** button: runs exactly one iteration, then refreshes every view.
- **Run N iterations**: a spinbox (default 5) + a Run button. Runs N iterations in the worker thread, updating the progress bar and the status readout as it goes. Views refresh at the end by default; a checkbox **"live update views each iteration"** makes them refresh after every iteration (slower, but useful for watching convergence).
- **Stop**: sets the cancel event; the run finishes the current iteration and stops cleanly, leaving a valid state.
- **Revert to iteration N**: rolls the state back.
- Status readout: current iteration number, shift-update RMS, residual, elapsed time.

### 8.2 Projection view (`projection_view.py`)

- An angle slider + spinbox + a "play" button to scrub through angles.
- A **display-mode combo box**, this is essential:
  - `Raw` (original, preprocessed, unshifted)
  - `Aligned` (current cumulative shift applied)
  - `Reprojection` (the simulated projection from the current volume)
  - `Difference` (aligned − reprojection) — a symmetric diverging colormap centred at zero. This is the panel the user will stare at most: when the alignment is converged, the difference is structureless noise; when it isn't, you see a characteristic edge-outlined "embossed" residual.
  - `Side by side` (aligned | reprojection | difference in one row)
- Show the current `sx[i]`, `sy[i]` for the displayed angle as text.
- Allow drawing a rectangular ROI, and let that ROI be used as the background region for phase-ramp removal (§5) and as an alignment mask.

### 8.3 Sinogram view (`sinogram_view.py`)

- A row (detector `v`, i.e. slice) slider: shows the sinogram `prj[:, v, :]` — angle on one axis, `u` on the other.
- Same display-mode combo as above (`Raw` / `Aligned` / `Reprojection` / `Difference`).
- Overlay (toggleable) the fitted COM sinusoid and the measured COM points on top of the sinogram — this makes misalignment immediately visible as a wobble in what should be a smooth sinusoidal band.
- Overlay (toggleable) a vertical line at the current rotation-axis `center`.

### 8.4 Tomogram view (`tomogram_view.py`)

- The reconstructed volume from the current iteration.
- Slice-axis selector: **axial (xy)** / **coronal (xz)** / **sagittal (yz)**, with a slice slider for the chosen axis.
- Optional orthoslice mode: three linked panels showing all three planes at once, with crosshairs.
- Contrast/levels controls (pyqtgraph gives you a histogram LUT for free — expose it).
- A **"compare to iteration N"** toggle that shows the current volume alongside, or as a difference against, an earlier stored volume. This is how the user judges whether the last 5 iterations actually helped.
- Display the voxel size if `pixel_size_nm` is known.

### 8.5 Shifts / convergence view (`shift_view.py`)

Four stacked plots:
1. `sx` vs angle (cumulative), with the current iteration in bold and previous iterations as faded traces so you can see the shifts settling.
2. `sy` vs angle (cumulative), same treatment.
3. Shift-update RMS vs iteration (the convergence curve) — log y-axis.
4. Residual vs iteration.

Plus the COM sinusoid fit plot (measured `com_u` scatter + fitted curve) from the pre-alignment step.

### 8.6 Left dock panels

- **Data panel:** load button, file/dataset paths, shape/angle-range summary, angular step, min/max/mean of the data.
- **Preprocess panel:** checkboxes and parameters for each function in §5, an "Apply preprocessing" button, and a "reset to raw" button.
- **COM panel:** background-region selector, "Run COM pre-alignment" button, resulting `center`, fit residual, and a "use this centre" / "override centre manually" control.
- **Align panel:** every field of `AlignConfig` as a labelled widget, with tooltips explaining what each does. Changing a parameter mid-run is allowed and takes effect from the next iteration (do not silently reset the state when config changes — but *do* show a small warning marker in the history plot at the iteration where config changed).

---

## 9. Session save / load (core/io.py)

Save an HDF5 file containing: the preprocessed projections, the angles, the full shift history (`sx`, `sy` per iteration), the config, the current volume, and the metadata. Load restores the exact state so the user can close the app and resume stepping. Also provide **export** buttons for:
- aligned projection stack (TIFF stack or HDF5),
- final volume (TIFF stack or HDF5),
- shifts as a CSV (`angle_rad, angle_deg, sx, sy`),
- the convergence curve as CSV.

---

## 10. Performance and UX requirements

- Downsample for display only (never for compute) if the array is larger than the widget; pyqtgraph handles this but be explicit about not making full-resolution copies on every slider tick.
- Every long operation shows a determinate progress bar and is cancellable.
- Guard against obviously-wrong inputs with clear dialogs, not tracebacks: NaNs in the data, angles not monotonic, angle count not matching projection count, a projection stack that is complex-valued (offer to take `np.angle` of it), all-zero mass.
- Log everything to a scrollable log dock: every iteration's parameters and metrics, so the user can reconstruct what they did.
- A **"binned preview" mode**: run the whole alignment on 2× or 4× binned data first (fast, converges quickly), then upsample the shifts and continue at full resolution. Expose this as a bin factor in the align panel; when the bin factor changes, scale the existing shifts accordingly. This is the single biggest usability win for large datasets and should not be an afterthought.

---

## 11. Testing (do this before wiring up the GUI)

`examples/make_phantom.py` must generate a synthetic ground-truth dataset:
- build a 3D phantom (TomoPy's `shepp3d`, plus a version with some off-centre asymmetric blobs so the COM sinusoid is non-trivial),
- forward-project it with `tomopy.sim.project.project` over e.g. 180 angles,
- apply **known** random per-angle shifts (`sx_true`, `sy_true`), plus optionally a slow drift and a linear phase ramp,
- add noise.

Then `tests/test_engine.py` must assert that after ~20 iterations the recovered shifts match the truth to better than 0.1 px RMS (after removing the global degenerate modes — you cannot compare absolute shifts, only shifts modulo a constant offset, so subtract the mean from both before comparing). **This test is the definition of done for the core.** If the sign convention in step 5 is backwards, this test fails immediately and loudly, which is exactly what we want.

`tests/test_com.py`: feed it a phantom with a known applied sinusoid + known random jitter and assert the recovered `center` and the recovered `sx` residuals are right.

`tests/test_preprocess.py`: apply a known linear ramp to a projection and assert `remove_phase_ramp` removes it to machine precision.

---

## 12. Documentation

`README.md` must contain: install instructions (conda is the sane route for TomoPy — note that), a quickstart that runs `examples/make_phantom.py` and then opens the GUI on the result, a screenshot placeholder, a short explanation of the algorithm with the citation to Gürsoy et al., *Sci. Rep.* 7, 11818 (2017), and a "troubleshooting: my alignment is diverging" section covering the usual suspects (residual phase ramp, wrong sign convention, bad centre, too few inner reconstruction iterations, missing-wedge artifacts biasing the reprojection).

---

## 13. Build order (follow this)

1. `core/dataset.py`, `core/preprocess.py`, `examples/make_phantom.py`, `tests/test_preprocess.py`.
2. `core/com.py` + `tests/test_com.py`.
3. `core/state.py`, `core/engine.py` + `tests/test_engine.py`. **Do not proceed until the engine test passes.**
4. `gui/worker.py` and a minimal `main_window.py` with just the action bar and the tomogram view — prove that step-wise operation works end to end.
5. Add the remaining viewers, then the docks, then session I/O, then the binned-preview mode.

Commit at each stage. Keep `core/` importable and usable headlessly at all times.