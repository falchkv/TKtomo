# tktomo.diagnostics — the artifact-to-cause table, executable

Twelve stereotyped ways a tomographic reconstruction goes wrong, each with a slice
signature, a sinogram signature, a way to confirm it and a fix. Everyone who does
tomography has a version of that table on a wall somewhere. A table on a wall cannot
be acted on by code, so here it is a **catalogue of records** plus **one probe per
row**, and the answer comes back as a ranked, JSON-serialisable verdict.

```python
from tktomo.diagnostics import triage, format_verdict

verdict = triage(projections, theta_deg)      # (n_theta, n_v, n_u) phase, angles
print(format_verdict(verdict))

if verdict.top is not None:
    print(verdict.top.mode)          # FailureMode.PHASE_RAMP
    print(verdict.top.confidence)    # 0.56
    print(verdict.top.evidence)      # {'vacuum_offset_frac': 0.061, 'ramp_frac': 0.12}
    print(verdict.top.fix)           # what to do about it
```

`projections` may be an `h5py` dataset or a memmap: every pass over the data is
chunked, so a ~1000 × 1500 × 1800 float32 stack (around 10 GB) is diagnosed without
loading it. The sign is handled for you — ptychographic phase is negative inside
material and every moment here needs it positive, so the stack's total decides a global
flip and records it in the verdict context.

## Two entry points

| | |
|---|---|
| `triage(projections, theta)` | runs the probes in the roadmap's **order of operations** and **stops at the first one that fires**. This is the actionable answer. |
| `diagnose(projections, theta)` | runs all of them and ranks what it finds. This is the survey. |

The order is: **ramp/offset and truncation → angular coverage → rotation centre →
vertical → horizontal → non-rigid.** It is not advice about what to fix first, it is a
statement about which measurements are *valid*. A residual phase ramp moves every
centroid; truncation invalidates every moment; a wrong rotation centre inflates every
reprojection residual. Fix what fired, run again.

`diagnose` honours the same order in its ranking: a finding's confidence is **halved
for each earlier stage that also fired**, with the original preserved in the finding's
evidence as `confidence_before_stage_discount`. Within a stage, `INVALIDATED_BY`
encodes the one hard dependency: truncation (mode 12) invalidates the phase-ramp fit
(mode 11), because the ramp is fitted on a border that mode 12 says is object.

## The twelve modes

`format_catalogue()` prints the whole table; `CATALOGUE[FailureMode.X]` is the record.

| # | mode | stage | probe | what fires it |
|---|------|-------|-------|---------------|
| 1 | `wrong_center` | rotation centre | `center_consistency`, `center_sweep` | axis estimate differs from the assumed centre |
| 2 | `jitter` | horizontal | `shift_jitter` | residual large **and white** in acquisition order |
| 3 | `vertical_drift` | vertical | `vertical_drift` | smooth trend in the registered vertical mass profile |
| 4 | `tilt_axis_angle` | rotation centre | `axis_tilt` | sinusoid **offset** walks with detector row |
| 5 | `tilt_axis_lateral` | rotation centre | `axis_tilt` | offset flat in z but displaced |
| 6 | `out_of_plane_tilt` | rotation centre | `axis_tilt` | sinusoid **amplitude** walks with z **and** `com_v` is modulated |
| 7 | `angle_readback` | horizontal | `angle_readback` | free-gain refit of the centroid sinusoid prefers g ≠ 1 |
| 8 | `scale_drift` | horizontal | `scale_drift` | secular trend in the projected second moment |
| 9 | `deformation` | non-rigid | `deformation` | projected mass not conserved; localised reprojection residual |
| 10 | `missing_wedge` | coverage | `angular_coverage` | largest gap in θ mod 180° |
| 11 | `phase_ramp` | data integrity | `vacuum_phase` | fitted vacuum offset/ramp large or fluctuating |
| 12 | `local_tomography` | data integrity | `truncation` | profile still sloping at both frame edges |

## The probes

Every probe is independently callable, returns a `ProbeResult`, and is honest about
its preconditions. A probe that cannot run returns `ProbeStatus.NOT_APPLICABLE` **with
the reason** — never a score. `Verdict.coverage` is the fraction that actually ran,
and an empty finding list means nothing until you have read it.

### The ten-minute check: `probe_vacuum_phase` (mode 11)

Run this first and most often. It calls
`tktomo.ptycho_align.core.preprocess.remove_phase_ramp` and measures the plane that
was removed, so there is exactly one ramp-fitting implementation in the repo. It needs
no angles and no reconstruction, and it reports the fitted offset and ramp relative to
the projection's own contrast, plus how much they fluctuate projection to projection.

A residual ramp is *mathematically indistinguishable from a lateral shift*, so it does
not merely add artifacts — it poisons the alignment that is supposed to remove them.

### The three-slice arc test: `probe_axis_tilt` (modes 4, 5, 6)

Split the **object's row support** into three bands (not the detector: a sample filling
half a tall frame leaves an empty band and the test then correctly, uselessly, refuses
to run), fit `com_u = a sin θ + b cos θ + c` in each, and regress the coefficients
against the band's **mass-weighted** height z. The three axis errors are algebraically
distinct:

| defect | signature | why |
|---|---|---|
| in-plane tilt α (4) | `c(z) = c₀ + z tan α` | material at height z rotates about a laterally displaced point; **the constant term is the axis position whatever the sample looks like** |
| lateral shift (5) | `c` flat in z, displaced | the axis is simply somewhere else |
| out-of-plane tilt β (6) | `u += z sin β sin θ`: the **amplitude** walks with z | the axis leans toward the beam |

The trap: **mode 6's amplitude signature is exactly what a tilted or sheared sample
produces.** A slice whose centre of mass sits at `(X_z, Y_z)` has `b = X_z`,
`a = −Y_z`, so any sample whose centroid walks with height mimics an out-of-plane
tilt. On the test phantom that false signal is 0.84 px, nearly three times the
firing threshold.

The discriminator is the vertical channel. A rigid sample rotating about a **truly
vertical** axis has an *exactly* rotation-invariant vertical mass profile, however
tilted the sample itself is — the row sum is the density integrated over the whole
x–y plane. A non-vertical axis breaks that: `com_v` picks up `−X sin β sin θ`. So mode
6 fires only when the amplitude walks with z **and** `com_v` is modulated beyond what
the measured in-plane tilt already explains, and the modulation then gives an
independent estimate `sin β = −V_sin / b`. Swept over injected tilts of 2–6° that
estimate reads 2.04°, 3.09°, 3.85° and 6.07° — within 4% — and the probe starts firing
at 3°.

Only the `sin θ` component of the modulation is used: over a monotone 0–180° sweep the
`cos θ` component is nearly collinear with a vertical drift (mode 3), and reading a
drift as a tilt would be worse than missing the tilt. The modulation is also gated
against its own standard error, because 1.2 px of vertical jitter on 60 views
otherwise manufactures an out-of-plane tilt at 0.65 confidence.

**The arc test polices its own model.** Everything above rests on the rigid-object
sinusoid, and over a 180° span its basis `{sin θ, cos θ, 1}` is not orthogonal — so a
residual that is *correlated with θ*, which is exactly what leftover per-projection
misalignment looks like, leaks into the constant term the test regresses, differently
for each band, and manufactures a slope. So the probe compares the per-band fit residual
with the axis walk it is about to claim (`band_residual_over_effect`) and, when the
residual wins, halves its confidence and prints a caveat naming the cross-check:
`probe_center_sweep(rows=(r0, r1))` on a top and a bottom band estimates the same
quantity from a reconstruction and shares none of these assumptions.

This is not a hypothetical. On a real ptycho-tomography stack the arc test reported a
confident in-plane tilt of 3.9° (0.068 px per detector row, and it survived removing the
phase ramp) while the per-band entropy sweep put the walk at −0.009 px/row between the
first two bands and nowhere near a straight line across the third: the two estimators
disagreed in magnitude **and in sign**, on a stack whose band residual was 1.3× the walk
being claimed. Without the guard the tool would have reported a 0.96-confidence tilt that
is an artifact of its own fit.

### `probe_deformation` (mode 9) — two statistics, one sharp

1. **Projected mass conservation.** A parallel-beam projection of an enclosed object
   has a line integral whose total is the object's mass — exactly, at every angle,
   independent of alignment, centre, tilt and angle readback. The non-smooth part of
   that series (a smooth trend is removed first: that is intensity or scale drift, not
   deformation) is a rigorous rigidity test costing one pass. It separates a clean
   phantom from an injected deformation by a factor of **128**.
2. **Reprojection residual**, as the roadmap prescribes: reconstruct, forward project,
   remove the best rigid shift + gain + offset per angle, and ask whether what is left
   is large *and localised*. On the same injected deformation this separates clean from
   deformed by a factor of **1.1** — filtered backprojection absorbs inconsistent data
   into streaks that reproject back onto the measurement. It is reported, but it is not
   the statistic that catches anything.

### The reconstruction-based probes

`probe_center_sweep` and the second half of `probe_deformation` need a reconstruction.
The package carries its own minimal filtered backprojection (`fbp_slice` /
`forward_project_slice`, numpy + `scipy.ndimage.rotate`, lazily imported) rather than
requiring astra or tomopy — *a diagnostic that only runs when the heavy stack is
installed is a diagnostic nobody runs*. Its convention matches
`skimage.transform.radon` and it is its own adjoint pair, so reprojection residuals
computed with it mean something. It is for one small binned slice, not for producing a
volume anybody keeps. Pass `volume=` to use a real reconstruction instead, or
`allow_reconstruction=False` to skip them (`probe_deformation` still reports its mass
statistic).

### Registration is done on gradients

`_register_1d` correlates the **derivatives** of the two profiles. This is the
roadmap's central trick in one dimension and it is not a refinement — measured on the
phantom's vertical mass profile at shifts of 0.25–3 px:

| | error |
|---|---|
| plain correlation | 0.15–0.35 px **low**, and the bias varies with the shift |
| gradient correlation | exact to the 1/20 px interpolation grid, still exact at 2% noise |

A 0.3 px bias is the entire error budget of a ⅓-voxel alignment target.

## Thresholds

`DiagnosticConfig` is a frozen dataclass; every threshold is documented at its
definition and every one is in physical units — pixels for geometry, fractions of the
projection's own contrast for phase. Two of them are deliberately **not** fractional:

* mode 7 fires on the *tangential smear at the object's rim*, not on the centroid
  residual. A 6% angle error hides in 0.05 px of centroid residual (a half-period
  sinusoid absorbs almost all of it) while smearing a 35 px object's rim by 3 px.
* mode 8 fires on the *radial displacement at the rim*, not on the fractional size
  change. 1% on a 10 px object is nothing; 1% on a 1000 px object is 10 px of blur.

Both use the same definition of "the rim" (`_object_rim_px`) so that their findings
are comparable — and they do get compared.

## Validation: injected vs detected

`tests/test_diagnostics.py` injects every mode into a synthetic phantom and checks the
verdict. Modes 4, 5 and 6 are injected by **rotating the phantom volume about a
genuinely tilted 3-D axis** and projecting, not by shearing finished projections: a
shear reproduces the horizontal signature of an out-of-plane tilt but not its vertical
one, and the vertical one is the whole discriminator.

| injected | triage stops at | rank-1 in `diagnose` | injected mode's rank |
|---|---|---|---|
| *(clean)* | — | *(nothing fires)* | — |
| *(clean + 2% noise)* | — | *(nothing above 0.25)* | — |
| wrong_center | `center_consistency` | wrong_center | 1 |
| jitter | `shift_jitter` | jitter | 1 |
| vertical_drift | `vertical_drift` | vertical_drift | 1 |
| tilt_axis_angle (8°) | `axis_tilt` | tilt_axis_angle | 1 |
| tilt_axis_lateral (3 px) | `center_consistency` | wrong_center | 3 (tied, see below) |
| out_of_plane_tilt (6°) | `axis_tilt` | out_of_plane_tilt | 1 |
| angle_readback (g=1.06) | `angle_readback` | angle_readback | 1 |
| scale_drift (2%) | `scale_drift` | scale_drift | 1 |
| deformation | `center_sweep` | wrong_center | 2 |
| missing_wedge (50°) | `angular_coverage` | missing_wedge | 1 |
| phase_ramp | `vacuum_phase` | phase_ramp | 1 |
| local_tomography | `truncation` | local_tomography | 1 |

### What is *not* separable, and why

* **Modes 1 and 5 are one equivalence class.** A laterally displaced rotation axis and
  a constant lateral shift of the data are the same stack of numbers. Both are reported
  at equal confidence; which is "the" cause is a question about the instrument, not
  about the data. No detector can do better.
* **Random angle-readback noise is jitter.** Both enter the centroid as
  `amplitude × δ`. Only the *systematic* part of an angle error (a gain, a wrong span)
  is separable this cheaply, and that is what `probe_angle_readback` targets.
* **A tilted sample and an out-of-plane tilted axis** look identical in the horizontal
  channel. Separated here only by the vertical channel — and if the object's centre of
  mass sits on the axis (`b ≈ 0`), the vertical channel is silent too and the probe
  says so.
* **Deformation moves the entropy-minimising rotation centre**, so `center_sweep`
  fires on the deformation case at 0.66 while the model-free `center_consistency`
  reports 0.00 px. That disagreement is itself the tell, and it is visible in the probe
  log — but the ranking puts `wrong_center` first, which is wrong. Check the probe log
  before acting on a lone `center_sweep` finding.
* **The in-plane tilt angle is a lower bound.** Under tilt, material leaks between row
  bands as θ turns, diluting the fitted `c(z)` slope by a factor that was a remarkably
  steady 0.74 across injected tilts of 2–8° on this phantom (2° → 1.48°, 8° → 5.69°) but
  that depends on how the object's mass is distributed in z, so it is not a correction
  factor worth applying. Detection is reliable from about 3° over a 34 px object height;
  the calibration is not.
* **The scale-drift magnitude is right to about 40%**, not better. Enough to detect and
  to size the problem; not a calibration. Its 2% injection is also detected right at the
  sensitivity limit (0.65 px of rim displacement against a 0.5 px threshold) — on a 22 px
  object 2% simply is not much, which is the point of measuring the rim rather than the
  fraction.
* **The arc test can be fooled by its own model** on data that is not yet rigidly
  aligned; see the caveat above. It says so when the residual is large, but the honest
  order of operations remains: reduce the per-projection residual first, then re-run.

## Running it on a large HDF5 stack

Nothing needs to be in memory. `stack_moments` and `probe_vacuum_phase` read the stack
in chunks, and the reconstruction probes read single rows, so an `h5py` dataset can be
handed in directly:

```python
import h5py
from tktomo.diagnostics import triage, format_verdict, save_verdict

with h5py.File("scan.h5", "r") as f:
    verdict = triage(f["exchange/data"], f["exchange/theta"][:], chunk=8)
print(format_verdict(verdict))
save_verdict(verdict, "verdict.json")
```

`chunk` is the number of projections held at once (they are promoted to float64 for the
moments, so 8 frames of a 1500 × 1800 detector is about 170 MB). A full `diagnose` over
a ten-gigabyte stack takes a couple of minutes on one core, most of it in the ramp fit.

Anything array-like with `.shape` and slicing works, which is a useful trick for
answering "does this finding survive the fix?" without writing a new file — wrap the
dataset in a view that applies the correction to each chunk it serves, and diagnose the
wrapper.

## The verdict object

```python
verdict.to_dict()            # plain dicts, floats, strings — json.dumps-safe
verdict.to_json()            # curves excluded by default; include_curves=True adds them
save_verdict(verdict, path)  # the same, to a file
verdict.top                  # highest-ranked Finding, or None
verdict.by_mode(FailureMode.PHASE_RAMP)
verdict.probe("axis_tilt")   # the ProbeResult, incl. metrics and per-projection curves
verdict.coverage             # fraction of probes that actually ran
```

`format_verdict` prints the ranked findings with the numbers that produced them and
the fix, then a log line for **every** probe including those that did not run and why —
because a diagnosis that shows only what fired lets a stack of `NOT_APPLICABLE` read
as a clean bill of health. `plot_verdict` builds an optional matplotlib figure
(confidence bars plus the curves behind them); matplotlib is imported inside that
function alone, so the package needs numpy and nothing else.
