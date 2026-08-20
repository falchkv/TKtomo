---
name: phasetomo-alignment
description: Align phase-contrast (ptychographic) tomography projections with TKtomo — the mandatory stage order (ramp/offset → rotation centre → vertical → horizontal → geometry → residual check → non-rigid), the executable artifact diagnostics, the ground-truth benchmark and its ⅓-voxel target, plus the traps and the already-measured dead ends. Use when aligning any phase-tomography dataset, or when developing tktomo.ptycho_align / tktomo.diagnostics / benchmarks.
---

# Phase-contrast (ptychographic) tomography alignment

Two jobs in one skill: **(A) run** the alignment pipeline on a new phase-tomography
dataset in the mandatory order, and **(B) develop** the alignment code without
repeating work that has already been measured and rejected.

Reference implementation: the **TKtomo** repository, branch
`feature/phasetomo-alignment`. Everything below names a real module, a real test or a
real command; nothing here is aspirational. Deeper documents:
`docs/alignment_roadmap.md` (method landscape + artifact table),
`IMPLEMENTATION_NOTES.md` (checklist + decision log), `docs/diagnostics.md`,
`docs/benchmark_results.md`, `docs/joint_gd.md`, `docs/nonrigid.md`,
`benchmarks/README.md`.

## The problem in one paragraph

Ptychographic phase retrieval determines each projection's phase only up to a
**constant offset** and a **linear ramp**, and a linear ramp across a projection is
*mathematically identical to a lateral translation of it*. So the alignment problem is
not "find the shift" — it is "find the shift given that the data carries a per-view
gauge freedom that looks exactly like a shift". On top of that, the vertical and
horizontal directions are different problems: vertical is a well-posed 1-D
registration, horizontal is only defined against a reconstructed volume. The accuracy
target is **residual alignment error ≤ ⅓ of the target voxel**, and it must be
demonstrated against injected ground truth or against reprojection residuals — an FSC
alone cannot certify it (rule 6 below).

---

## 1. The order of operations. Never violated.

Each stage's fix **invalidates the measurements of every later stage**, so this is not
advice about priority — it is a statement about which numbers are valid at all.

| # | stage | why it must come first | module | proof |
|---|---|---|---|---|
| **0** | **ramp / offset removal** | a residual ramp is indistinguishable from a lateral shift; leaving it in means the loop "corrects" a shift that is not there | `ptycho_align.core.preprocess.remove_phase_ramp`, `remove_phase_offset` | `tests/test_ptycho_preprocess.py` |
| **1** | **rotation centre** | a wrong centre is *one number*; letting a per-view loop absorb it spends N free parameters on it and invites drift, and it inflates every reprojection residual downstream | `ptycho_align.core.com.find_center`; `diagnostics.probe_center_sweep`, `probe_center_consistency` | `tests/test_ptycho_com.py`, `tests/test_diagnostics.py` |
| **2** | **vertical** | decoupled and free: rotation about a vertical axis maps every voxel inside its own detector row, so the row-summed mass profile is angle-invariant. No projector, no volume | `ptycho_align.core.vertical.align_vertical` | `tests/test_vertical_alignment.py` |
| **3** | **horizontal** | entangled with the rotation angle, so it can only be solved against a reconstructed volume: reconstruct → reproject → compare → update → repeat | `odstrcil.OdstrcilEngine` (stage 2), `engine.AlignmentEngine`, `joint_gd.JointGDAligner` | `tests/test_odstrcil.py`, `test_ptycho_engine.py`, `test_joint_gd.py` |
| **4** | **geometry refinement** (axis tilt in-plane / out-of-plane, angle readback, magnification) | these are *not* per-view translations; no rigid shift can represent them, and a shift loop asked to absorb them produces a converged wrong answer | detection: `diagnostics.probe_axis_tilt`, `probe_angle_readback`, `probe_scale_drift`. Landmark fit + ASTRA `parallel3d_vec` export: `tktomo.tracking.model` | `tests/test_diagnostics.py`, `tests/test_tracking_model.py` |
| **5** | **residual check** | the only honest stopping criterion; also the gate for whether stage 6 is warranted at all | `benchmarks.metrics.reprojection_residual`, `residual_plateau`, `score_shifts`; `nonrigid_gate.evaluate_gate` | `tests/test_benchmark.py`, `tests/test_nonrigid_gate.py` |
| **6** | **non-rigid** | ~1000× more free parameters than a rigid shift; it will absorb any unfixed earlier error and return a sharp, plausible, **wrong** volume | `ptycho_align.core.nonrigid.NonRigidAligner`, `deformation.py` | `tests/test_nonrigid.py`, `test_deformation.py` |

`NonRigidAligner` **refuses to start** on data that does not look rigidly aligned
(`RigidAlignmentRequired`) and refuses a missing wedge in any time subset. That is
deliberate: the caller is exactly who gets this wrong.

---

## 2. Diagnose before you align (ten minutes, no reconstruction)

```python
from tktomo.diagnostics import triage, diagnose, format_verdict

verdict = triage(projections, theta_deg)     # (n_theta, n_v, n_u) phase; h5py dataset is fine
print(format_verdict(verdict))
verdict.top.mode, verdict.top.confidence, verdict.top.fix
verdict.to_json()
```

`triage()` runs the probes in the order above and **stops at the first firing**;
`diagnose()` runs all eleven and ranks them (confidence halved per earlier stage that
also fired). The stack may be an `h5py` dataset or a memmap — every pass is chunked, so
a ~10 GB stack is diagnosed without loading it. Ptycho phase is negative in material;
the sign flip is detected and recorded, not assumed.

**How to read it honestly.**

- `ProbeStatus.NOT_APPLICABLE` carries a *reason* (no vacuum border, span too small, no
  reconstruction). `Verdict.coverage` says what fraction actually ran. **An empty
  finding list means nothing until you have read the coverage.**
- **Always run `diagnose()` as well as `triage()`.** Measured on the benchmark
  scenarios, `diagnose()` named the injected cause first in 6 of 7 cases but `triage()`
  only in 3 of 7 — it stops at the first stage that fires *however weakly*, and will
  return a 0.05-confidence `tilt_axis_angle` in preference to a 0.43-confidence
  `vertical_drift`. Known issue; fix is item 7 in `docs/benchmark_results.md` §14.
- Modes 1 (wrong centre) and 5 (lateral axis shift) are **one equivalence class** and
  no detector can separate them. Both are reported at equal confidence.
- The three-slice arc test (modes 4/5/6) can produce a *confident* axis-tilt verdict
  that an independent per-band entropy centre sweep contradicts **in sign**. A
  model-strain guard halves the confidence and prints the cross-check to run. **Do not
  act on an axis-tilt number until the per-projection horizontal residual is small** —
  see the dead ends, item D2.

Catalogue of the twelve failure modes with slice/sinogram signature, confirmation and
fix: `format_catalogue()`, or `docs/alignment_roadmap.md` §3.

---

## 3. Running the pipeline

```python
import numpy as np
from tktomo.io import ProjectionData
from tktomo.ptycho_align.core import preprocess, com, vertical, odstrcil, engine

# [0] ramp + offset, on a presumed-vacuum border. Confirm the border IS vacuum first.
prj = preprocess.remove_phase_ramp(prj, border=8)      # fits a+bu+cv, subtracts over the frame

# [1] rotation centre from the centroid sinusoid (offset c IS the axis position)
pre = com.com_prealign(prj, angles_rad)                # ComResult: sx, sy, center, fit_residual
center = pre.center                                    # cross-check: diagnostics.probe_center_sweep

# [2] vertical — free, no projector, run to convergence, then leave it alone
vres = vertical.align_vertical(prj, vertical.VerticalConfig())
assert vres.truncation_reason is None                  # or handle it; see rule 3
sy = vres.sy

# [3] horizontal — against a reconstructed volume, one inspectable outer iteration at a time
data = ProjectionData(data=prj, angles=angles_rad, metadata={"pixel_size_nm": px_nm})
align_cfg, ods_cfg = odstrcil.default_odstrcil_config(data)   # SIRT, 2 inner iters, joint
eng = odstrcil.OdstrcilEngine(dataset=data, config=align_cfg, odstrcil=ods_cfg,
                              sx0=pre.sx, center=center, com_amplitude=pre.amplitude)
for it in eng.run(12):
    print(it.iteration, it.residual, np.abs(it.dsx).max())
sy, sx = eng.state.sy, eng.state.sx

# [5] stop when the residual plateaus, not when the pictures look nice
from benchmarks.metrics import reprojection_residual, residual_plateau
```

Notes that are not optional:

- `OdstrcilEngine` **ignores `sy0`** when `run_vertical=True` — stage 1 solves the
  vertical axis outright rather than refining a guess, and it logs that it did. Pass
  `OdstrcilConfig(run_vertical=False)` to keep your own `sy`.
- `AlignmentEngine` (the incumbent Gürsoy loop) has the identical surface and needs
  **tomopy** (`shift_images`, `blur_edges`); `OdstrcilEngine` and `JointGDAligner` are
  numpy+scipy only. Pick the backend through `AlignConfig.backend` / `tktomo.recon`.
- `JointGDAligner` (`joint_gd.py`) solves volume and shifts together by multi-resolution
  gradient descent; its answer must be read through `finalize()`, which does the median
  centring and the MAD outlier rejection. Use it at realistic scale with the ASTRA
  projector — that is where it wins (`docs/joint_gd.md`).
- Non-rigid, last, and only if the gate says so:
  ```python
  from tktomo.ptycho_align.core.nonrigid_gate import gate_from_engine, format_gate
  v = gate_from_engine(eng, acquisition_index=acq); print(format_gate(v))
  if v:  # True only for RUN_NONRIGID
      ...
  ```

---

## 4. The rules (each one is a bug if you get it wrong)

1. **Centre of mass is structurally wrong for the horizontal direction.** Horizontal
   shift is entangled with rotation angle, so it is not a well-posed pairwise
   registration problem at all. CoM is defensible **only vertically**, and only when the
   sample is fully inside the field of view. Horizontally, use its sinusoid fit for the
   *rotation centre* and as a warm start — never as the answer.
2. **Register on the phase gradient, not the phase — but know what that buys.**
   Differentiating sends the constant offset to exactly zero and the ramp to a constant
   the mean-subtraction removes: pairwise ramp invariance is exact (4e-16 px across
   0–100 rad, vs −25.7 px for the value domain at 10 rad). **What it did not buy** is an
   end-to-end win: measured inside the same engine with one knob changed, the gradient
   domain was 1.06–1.50× *worse* than the value domain at every ramp from 0.25 to 2 rad,
   and better only at 4 rad, because differentiating puts the comparison's weight where
   a half-converged reprojection is least faithful. Treat it as **insurance against a
   phase background that step [0] cannot model**, not as a free improvement. Read
   `docs/benchmark_results.md` §3 before quoting either claim.
3. **Vertical alignment assumes the sample is fully inside the frame.** If it is
   truncated, mass enters and leaves as the sample rotates and the profile is no longer
   angle-invariant. `align_vertical` detects it and warns; `on_truncation="raise"` makes
   it fatal; `row_range=(v0, v1)` is the escape hatch. Under truncation both the
   centroid and the profile correlation fail, and the **correlation fails worse**
   (1.16 px vs 0.74 px) — this is a detection problem, not something to power through.
4. **Never initialise the loop from FBP, and never use GridRec with a limited angular
   range.** FBP's streaks are reprojected into the simulated stack and the registration
   then measures the streaks; GridRec interpolates onto a Cartesian Fourier grid, so an
   unsampled wedge is filled with interpolation noise instead of being left empty. SIRT
   (or MLEM on non-negative data) is the default. `odstrcil.check_reconstruction_choice`
   raises `LimitedAngleError` on both; gridrec+limited-angle stays fatal even under the
   `"warn"` policy.
5. **Median-centre the shifts every run, and handle the gauge when scoring.** The mean
   `dx` is degenerate with the rotation-axis position and the mean `dy` with the z
   origin — the objective cannot see them, so they random-walk and two runs of the same
   data disagree by a constant that is not an error. When comparing to truth, project
   out `{1}` from `dy` and `{1, sin, cos}` from `dx` (`metrics.remove_gauge`); removing
   only the mean leaves two of three horizontal modes in and makes a *perfect* aligner
   score ~0.2 px.
6. **FSC cannot detect systematic geometric bias. Always pair it with reprojection
   residuals.** A rigid-but-wrong geometry applied identically to both half-sets gives a
   deceptively good FSC — the common-mode factors `exp(∓2πik·d)` cancel exactly in the
   cross term, and `tests/test_benchmark.py` proves it to machine precision. Measured:
   split-half FSC read **19.51 px for everything** across a 1198× range of alignment
   error (0.114 % spread); on our own phantom the half-bit FRC read *exactly* 508.6 nm
   at centring errors of 0, 4, 8, 16, 32 and 64 px while the true edge blur grew to
   128 px. And the residual is not a free pass either: on the non-rigid benchmark case
   it ranked a wrong rigid answer (0.218) **better than the true shifts** (0.268).
7. **Sign conventions, pinned by tests, not by intuition.** `apply_shifts(prj, sy, sx)`
   gives `corrected(v,u) = measured(v+sy, u+sx)` — a feature at row 10 with `sy=+3`
   lands on row 7. `sy`/`sx` are the *correction to apply*, not the displacement the
   object has; the two differ by a minus sign. `phase_cross_correlation(measured,
   simulated)` in that order. `(dy, dx)` row-first everywhere. A sign-flipped aligner
   scores **exactly twice** the injected RMS with a negative correlation to truth —
   that is the signature, and it is more common than a genuinely 2×-worse algorithm.
8. **Never re-shift shifted data.** The cumulative shift is always applied to the
   pristine stack. Repeated warping compounds the interpolation and blurs the data away.
   Same rule for deformation fields: compose the coarse fields, apply once.

---

## 5. Prove it: the benchmark and the ⅓-voxel target

An aligner is measured against **injected ground truth**, never by eye and never by FSC.

```bash
# fully synthetic, all aligners, ~seconds; no GPU, no tomopy, no measured data
python -m benchmarks.runner --size 64 --slices 12 --angles 60 --iterations 12

# one perturbation at a time — this is where methods separate
python -m benchmarks.runner --catalogue --size 48 --slices 6 --angles 48

# the adjudicated multi-seed comparison (ablations, paired per-seed stats)
python -m benchmarks.run_three_way --out benchmarks/results --seeds 5 \
    --size 64 --slices 12 --angles 60 --margin 20 --iterations 12 \
    --gd-iterations 80 --ramp 1.0 --sweep 0.25,0.5,1,2,4 --sweep-seeds 3

# real sample statistics with exact truth: forward-project YOUR reconstructed volume
python -m benchmarks.runner --volume /path/to/volume --slice-range 700:716 --bin 4 \
    --angles-file /path/to/angles.h5 --angles-dataset exchange/theta \
    --pixel-size 74.5 --jitter-dy 25 --jitter-dx 7.5 --iterations 12 --out $SCRATCH/results
```

**Target: RMS residual shift error ≤ ⅓ of the target voxel** (= 0.333 px when
reconstructing on the detector grid; pass `voxel_nm` when it is not). Judge it on the
**worst seed**, not the mean — a method that fails one draw in five has not met it.

**Read these two rows first, every time.** `oracle` returns the truth and must score
~1e-16 px; if it does not, the *scorer* is broken and no other row means anything.
`null` returns zeros and its score *is* the injected misalignment; any aligner that does
not beat it is doing harm.

Baseline to beat on the synthetic case (5 seeds, gauge-removed RMS px, dy / dx):
`null` 2.405 / 0.726 · `jirr` 0.013 / 0.026 · `odstrcil` 0.006 / 0.026 · `joint_gd`
0.401 / 0.051. On **real sample structure** the ordering changes: `jirr` 0.022 / 0.379
(the only method that misses the target), `odstrcil` 0.022 / 0.238, `joint_gd`
0.017 / 0.230.

Never quote a benchmark number without its environment: on a machine without tomopy the
incumbent runs under `benchmarks.runner.tomopy_shim` (scipy stand-ins for
`shift_images`/`blur_edges`), declared in `environment.tomopy_shim` of every report.

---

## 6. Which method

| method | module | use it when | do not |
|---|---|---|---|
| `AlignmentEngine` (JIRR, Gürsoy 2017) | `core/engine.py` | the incumbent; interactive, one inspectable iteration at a time | it needs tomopy; it aborts under a strong (≥2 rad) residual ramp |
| `OdstrcilEngine` | `core/odstrcil.py` | **default recommendation.** Decoupled: free vertical stage, gradient-domain horizontal. numpy+scipy only | it replaces your `sy0`; its stage-2 default domain is an open decision (§7 / notes) |
| `JointGDAligner` | `core/joint_gd.py` | large real stacks with ASTRA on a GPU, large shifts (≈100 px regime), multi-resolution | on small clean phantoms it has a bimodal `dy` failure (2 of 5 seeds) and it blew up silently at a 4 rad ramp with `status="ok"` |
| `NonRigidAligner` | `core/nonrigid.py` | only after the gate returns `RUN_NONRIGID` | never as a substitute for an earlier stage |

---

## 7. Known dead ends — do not repeat these

Measured, not assumed. Full numbers and provenance in `IMPLEMENTATION_NOTES.md`
(decision log).

- **D1 — Shift-interpolation kernel.** Bilinear vs cubic vs Lanczos vs exact Fourier:
  **0.0 nm** difference in the reconstructed resolution on a real sample. Not a tuning
  axis. (Injection in the *benchmark* still uses a Fourier shift, deliberately a
  different implementation from any aligner's.)
- **D2 — Rotation-axis tilt correction.** Two calibrated estimators disagreed **in
  sign**; applying the correction made 2 of 3 measured rows *worse*. Independently, the
  diagnostics' arc test and an entropy centre sweep contradicted each other in
  magnitude and sign on the same real stack. Do not correct an axis tilt you cannot
  confirm with a second, model-free estimator.
- **D3 — A full-resolution alignment stage** on top of a binned ladder: no gain.
- **D4 — Reconstruction algorithm choice for the *final* volume.** FBP 188.4 nm vs
  SIRT-300 190.6 nm interior FRC — one FRC bin apart. CGLS-15 was adopted purely for
  cost (14 min vs 134 min per volume). This is *not* licence to use FBP or GridRec
  **inside** the alignment loop; see rule 4.
- **D5 — Edge-feature alignment with an air clamp.** Looked visibly cleaner and
  **halved the resolution**, 216 nm → 473 nm. A cleaner-looking projection is not a
  better-aligned one.
- **D6 — POCS Fourier completion.** With genuinely complete 180° coverage there is
  nothing to complete: a controlled hold-out raised in-sample error by **1.61×**.
- **D7 — `gradient-x` as the stage-2 domain.** The a-priori argument (the shift acts
  along x, it is cheaper) loses to `gradient-both` at every ramp amplitude
  (0.256 vs 0.167 px clean; 0.389 vs 0.263 px at 4 rad) and misses the ⅓-voxel target.
- **D8 — "Profile registration degrades more gracefully than the centroid under
  truncation."** False, and measured to be false — see rule 3. What registration buys
  is immunity to *localised* corruption (a vacuum artifact drags the centroid 1.45 px
  and moves the correlation peak 0.002 px), not truncation tolerance.
- **D9 — Enforcing a zero vacuum plane per projection**, beyond step [0]. The effect is
  real and measurable (per-view tilts up to 1.29 rad across a frame, wandering 0.22 rad
  rms between adjacent angles) and bought **0–9.7 nm** of resolution, because it is ~3 %
  of the sample's own contrast. Worth knowing about; not worth prioritising.
- **D10 — Reaching for non-rigid because the residual is high.** On a real 52-hour scan,
  **96.4 %** of the reprojection residual was unexplainable by any rigid or 3×3
  block-rigid displacement, was localised inside the sample and was *incoherent in
  angle* — the signature of per-projection reconstruction error, not deformation
  (a real deformation is smooth in acquisition time). Run the gate.

---

## 8. Development guide

**Module map.**

| file | what it owns |
|---|---|
| `core/preprocess.py` | step [0]: `remove_phase_ramp`, `remove_phase_offset`, masks, binning, padding |
| `core/com.py` | centroid pre-alignment + `find_center` (the sinusoid offset *is* the axis) |
| `core/vertical.py` | stage [2]: 1-D vertical mass profiles, sub-pixel registration, truncation detection |
| `core/gradient.py` | the gradient-domain pairwise estimator; domains, taper, the mean-before-taper trap |
| `core/odstrcil.py` | the two-stage series aligner; `check_reconstruction_choice`; `SeriesAligner` registry |
| `core/engine.py` | the incumbent JIRR loop, `AlignConfig`, `apply_shifts`, divergence/runaway guards |
| `core/joint_gd.py` | joint volume+shift gradient descent, projector layer (ASTRA / numpy), `clean_shifts` |
| `core/deformation.py`, `nonrigid.py`, `nonrigid_gate.py` | stage [6] and its evidential decision gate |
| `tktomo/diagnostics/` | the 12-mode catalogue, 11 probes, `diagnose`/`triage`, report + figure |
| `benchmarks/` | phantom + perturbation catalogue, metrics, runner, three-way adjudication |

**Seams to use rather than fork.**

- `OdstrcilEngine` subclasses `AlignmentEngine` and overrides **only `step()`**;
  chunked reconstruction, warm start, cancellation, `_condition_update`, divergence
  detection, `run`/`revert_to`/history all come from the parent by call. The clean fix
  is a registration-*estimator* hook on the parent — if you add it, this class shrinks
  to an estimator plus the stage-1 pre-pass.
- Registries mirror the repo's existing pattern: `tktomo.align.base` (pairwise 2-D),
  `tktomo.recon.backend` (reconstruct/reproject), `joint_gd.register_projector` (raw
  adjoint pair P/Pᵀ), `odstrcil.register_series_aligner` (series-level). **The pairwise
  `Aligner` protocol in `tktomo/align/base.py` is *not* the home for series-level
  tomographic alignment** — that is the engines.
- New aligners plug into the benchmark through `EngineAligner` (engine-shaped),
  a bespoke adapter, or `ModuleAligner` (a module-level `align(...)` function). An
  import failure must yield `status="skipped"`, never an exception.

**House style (the repo is unusually well written — match it).** `from __future__ import
annotations`, type hints, dataclasses for configuration, module docstrings that explain
*why* and name the traps ("three conventions, each of which is a bug if you get it
wrong"). Core is **numpy + scipy only**; skimage / tomopy / astra / matplotlib /
pyqtgraph are optional and imported lazily *inside functions*, with a clear error naming
the alternative. Honest failure: raise or return a documented sentinel, never silently
degrade. `pytest` tests for everything, runnable **without a GPU and without measured
data**.

```bash
python -m pytest -q                                    # (add --continue-on-collection-errors)
python -m pytest tests/test_odstrcil.py tests/test_vertical_alignment.py \
                 tests/test_gradient_registration.py tests/test_joint_gd.py \
                 tests/test_diagnostics.py tests/test_benchmark.py -q
```

**When you change a documented number, change the docstring table in the same commit,
and add the test that pins it.** Several docstring tables here are regenerated by the
test that asserts them; that is on purpose, so the prose cannot drift from the
measurement.

**Data policy.** Beamtime data is never committed. Benchmarks read a *user-supplied*
path; the repository ships a fully synthetic fallback phantom so its own tests run for
anyone. No `.h5` / `.npy` / `.tiff` of measured data, and no absolute site paths in
library code or in this skill — parameters only.
