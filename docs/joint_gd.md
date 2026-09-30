# joint_gd — joint gradient-descent alignment

`tktomo/ptycho_align/core/joint_gd.py` solves for the tomogram **and** the
per-projection shifts as one optimisation problem, instead of alternating a
reconstruction with a pairwise registration.

```
min_{v, s}  sum_i  w_i || P_{theta_i} v  -  T_{s_i} d_i ||^2
```

`v` is the volume, `s_i = (dy, dx)` the shift of projection `i`, `d_i` the measured
phase projection, `T` a translation, `P` the parallel-beam projector, `w_i` a
per-projection quality weight. Both gradients are written out by hand, which is why
this needs no autograd framework:

| unknown | gradient | step |
| --- | --- | --- |
| volume `v` | `P^T (P v − T_s d)` — a backprojection of the residual | preconditioned SIRT, `R = P(1)`, `C = P^T(1)`, plus Nesterov momentum |
| shift `s_i` | `⟨res_i, ∇(T_{s_i} d_i)⟩` — an image-gradient inner product | Gauss-Newton, `÷ ⟨∇, ∇⟩`, damped and capped |

The shift half costs two inner products per projection per iteration, so essentially
nothing next to the two projections. **The cost of this method is the projector**, and
that is why ASTRA on a GPU is the production backend.

> Odstrčil, M. *et al.* Alignment methods for nanotomography with deep subpixel
> accuracy. **Opt. Express 27**, 36637 (2019). — the objective.
>
> The optimiser machinery (analytic adjoints instead of autograd, Nesterov momentum,
> coarse-to-fine multi-resolution, a low-pass on early volume updates) is carried over
> from the ASRM / "Dora" work, **Opt. Express 32**, 10801 (2024).

## Where this came from

A port of `joint_align_gd.py` from a 2026 P06 ptycho-tomography campaign pipeline, where it ran as the top-end alignment method on 907–918-projection lens-1
and lens-2 phase stacks. The original was a single `main()` configured entirely by
environment variables. The port keeps the numerics and changes the shape:

| original | here |
| --- | --- |
| `JOINT_SMOKE`, `JOINT_LONG`, `JOINT_REFINE` | `STAGES_SMOKE`, `STAGES_LONG_JITTER`, `STAGES_REFINE` |
| `JOINT_SCAP` | `JointGDConfig.shift_cap_px` |
| module-level `STAGES`, `LR_V`, `LR_S`, `WARMUP`, `MOM` | fields of `JointGDConfig` |
| whole schedule inside `main()` | `JointGDAligner.step()` — one iteration, returns |
| `astra` imported at module top | optional import inside `AstraProjector3D`, plus a pure-numpy CPU fallback |
| shift cleaning lived in a separate `finalize_joint.py` | `JointGDAligner.finalize()` / `clean_shifts()` |
| angles in degrees | radians, the repo-wide convention |
| shifts in `scipy.ndimage.shift`'s sign | **negated**, into `apply_shifts`' sign (see below) |

Exposing one iteration at a time is what lets a benchmark driver run this, the
reprojection engine (`core/engine.py`) and a vertical-mass-fluctuation pass through the
same loop, and what lets a GUI show a tomogram between iterations.

## Using it

```python
import numpy as np
from tktomo.ptycho_align.core.joint_gd import JointGDAligner, JointGDConfig, quality_weights

config = JointGDConfig(projector="astra")           # the default schedule: bin 16/8/4
aligner = JointGDAligner(
    projections,                                    # (n_angles, n_rows, n_cols) float32
    angles,                                         # RADIANS
    config,
    weights=quality_weights(steepness),             # optional, de-weights bad frames
    initial_shifts=np.column_stack([com.sy, com.sx]),   # optional, from com_prealign
)

for result in aligner.run():                        # or: while not aligner.done: aligner.step()
    print(result.iteration, result.binning, result.loss, result.max_abs_shift)

answer = aligner.finalize()                         # median-centred, outliers rejected
print(answer.summary())
aligned = aligner.aligned_projections(answer.shifts)
```

`finalize()` is not optional bookkeeping — see below. `aligner.shifts` is the raw
solution and applying it directly is a bug.

### Sign convention — the trap this port fell into

Shifts are `(dy, dx)`, row first, **in `engine.apply_shifts`' sign**. TKtomo's
`apply_shifts(prj, sy, sx)` is `scipy.ndimage.shift(prj, (-sy, -sx))`: a feature at row
10 given `sy=+3` lands on row **7**, which `tests/test_ptycho_engine.py` pins
deliberately. `com_prealign`'s `sy`/`sx` are in the same sign, so
`initial_shifts=np.column_stack([com.sy, com.sx])` is correct as written.

The original P06 script's `shifts_joint.tsv` used the *opposite*, ndimage sense. A
faithful transliteration therefore produced a perfectly converged, sign-inverted answer
— and every accuracy test written alongside it passed, because the injected
displacement and the recovered shift were flipped together. What caught it was the
benchmark harness, which applies each aligner's answer with `apply_shifts` and scored
this one at **4.49 px** while reporting that negating it would score **0.009 px**.

The lesson is in the test suite now: `test_a_positive_shift_moves_rows_up_like_apply_shifts`
asserts against the engine's own numbers rather than against another number this module
produced, and it is the only test here that cannot pass on a sign-inverted aligner.
If you export a TSV to compare against the original script, negate it.

### Backends

`config.projector` is `"astra"` (default), `"numpy"`, `"auto"`, or any name passed to
`register_projector(name, factory)`.

* **`astra`** — `parallel3d` FP/BP on the GPU. Optional dependency
  (`conda install -c conda-forge astra-toolbox`), imported inside the class, and it
  raises with a message naming the alternative if it is missing. It does **not** fall
  back silently: a run that quietly switched to the CPU projector would take days and
  nobody would know why.
* **`numpy`** — rotate-and-sum in numpy/scipy. Exists so the unit tests, and anyone
  evaluating the method before installing CUDA, can exercise the real optimiser. Usable
  to roughly 128³, hopeless on real data.
* **`auto`** — prefers ASTRA, logs loudly when it falls back.

`tktomo.recon.backend.ReconBackend` is deliberately *not* the interface here: it offers
`reconstruct`/`reproject`, and this optimiser needs the raw adjoint pair `P` / `Pᵀ`. A
reconstruction is a whole inner solve, not a backprojection. `register_projector` is the
hook for a native TKtomo projector when one exists.

## The schedule

Multi-resolution, coarse to fine. Each stage bins the detector axes, rescales the
shifts into the new pixel grid, rebuilds the projector and the preconditioners, and
restarts the volume from zero.

| preset | stages `(binning, iterations, smooth σ)` | for |
| --- | --- | --- |
| `STAGES_STANDARD` | (16, 150, 2.0), (8, 150, 1.0), (4, 100, 0.0) | the default; what the P06 lens-1/lens-2 stacks were aligned with |
| `STAGES_SMOKE` | (16, 60, 2.0) | a numeric smoke test, not an alignment |
| `STAGES_LONG_JITTER` | (16, 250, 2.0), (8, 200, 1.0), (4, 150, 0.0) | large jitter — lens-2 ran ~120 px rms and needed the extra iterations at every scale to walk in |
| `STAGES_REFINE` | (4, 100, 0.5), (2, 80, 0.0) | refining an already-aligned, normalised stack: skip the coarse stages, add the binning-2 stage where sub-4-px sinogram edge jitter lives |

`smooth σ` is a Gaussian low-pass on the **volume update**, in binned voxels — never on
the data. The finest stage runs with 0.

## Load-bearing numerics

These were learned the hard way on real data. The method fails without them, and
"cleaning them up" is how it gets broken.

1. **Median-centre the shifts, every run.** The global mean shift is invisible to the
   objective — in `dx` it is the rotation-axis position, in `dy` the `z` origin — so the
   optimiser lets it random-walk. Skipping the centring makes a long run drift and makes
   two runs of the same data disagree by a constant that is not an error. The *median*,
   not the mean: one projection that has slid out of frame moves a mean far enough to
   shift every other projection.
2. **The MAD outlier fallback is essential.** A projection too degraded to register does
   not stall — it slides out of frame, because a ramp-dominated projection has an almost
   constant image gradient, so the Gauss-Newton step points the same way every
   iteration. Nothing in the loss says anything is wrong. **20 of 918 projections did
   exactly this on the first real run.** Anything more than `outlier_mad` (5) MADs from
   the median, or further than `outlier_abs_px` (150 px) in total, is reset and reported
   in `FinalizedShifts.outliers`. `tests/test_joint_gd.py` reproduces the failure on the
   synthetic phantom: a ramped projection reaches ~47 px on a 48 px detector while every
   healthy one stays under 4 px.
3. **Damping (`lr_shift = 0.5`) and the per-iteration cap (`shift_cap_px = 0.5` binned
   px).** Before these were added this method diverged outright: loss to NaN, shifts to
   1e4 px. The Gauss-Newton denominator `⟨∇, ∇⟩` collapses wherever a projection is
   flat, so an undamped, uncapped step is unbounded exactly where it is least
   trustworthy. The `1e-6` floor on the denominator is part of the same defence.
4. **Volume warm-up (`warmup_iters = 15`), at the start of *every* stage.** Registering
   against a volume that has not formed yet is registering against noise — the same
   failure `engine.shift_update_is_runaway` exists to catch on the other method.
5. **SIRT preconditioners** `R = max(P(1), 1)`, `C = max(Pᵀ(1), 1)`. The raw
   backprojection is wildly non-uniform (the edge of the field of view sees a handful of
   rays, the centre sees all of them) and a plain gradient step on it is unusable.
6. **Loss back-off.** If the loss rises above 3× the best seen, halve `lr_volume` and
   clear the momentum buffer, in place, mid-stage. Cheaper and more reliable than
   picking a safe learning rate up front.

Deviations from the original, all behaviour-preserving on the default path: shifts
accumulate in float64; the three separable `gaussian_filter1d` calls are one
`gaussian_filter` (identical result); the reported shifts are negated into TKtomo's sign
convention; non-finite losses and runaway shifts raise
`JointGDDivergence` instead of being logged and continued; and the MAD test measures
deviation *from the median* rather than absolute magnitude — identical whenever
`median_center` is on, which is the default and was the original's only behaviour, but
correct rather than degenerate when it is off.

## Measured behaviour

### On the P06 beamtime data

**The data itself is not in this repo and must not be** — these are the numbers the
original script produced there, quoted so the port can be checked against them. Voxel
74.51 nm, 907 lens-1 phase projections at 1488 × 1816, 180.0° span.

* Converged **monotonically**: loss 17.81 → 0.2852 at binning 4, and 17.60 → 0.3115 at
  binning 2 on the refinement schedule.
* Maximum absolute shift 70.7 full-resolution px.
* 16 MAD outliers rejected at finalisation.
* Shifts from the pipeline the method feeds: rms `dy` 25.0 px, `dx` 7.5 px.

A caveat worth carrying into any write-up: on this dataset the **vertical** direction is
the harder one, which is the opposite of the textbook expectation. Projection-space FRC
for lens-1 is 199 nm horizontal against **359 nm vertical**, and the vertical shifts are
3.3× larger than the horizontal ones. Whatever is driving the vertical jitter here
(stage drift, most likely) is bigger than the horizontal problem the reprojection loop
is designed around.

I did **not** re-run the method on the beamtime data as part of this port; the numbers
above come from the original script's runs and the port is verified against synthetic
ground truth instead.

### On synthetic ground truth (what the tests check)

`tests/test_joint_gd.py`, numpy backend, no GPU, no ASTRA, no beamtime data. A 20 × 48 ×
48 blobbed ellipsoid, 32 angles over 180°, random shifts up to ±2.5 px injected with
`scipy.ndimage.fourier_shift` — deliberately a different implementation from the loop's
`scipy.ndimage.shift`, so recovering the truth proves the loop works rather than that it
agrees with itself.

| seed | recovered error (RMS, observable modes) | baseline (no alignment) |
| --- | --- | --- |
| 1 | 0.044 px | 1.995 px |
| 2 | 0.036 px | 1.830 px |
| 3 | 0.048 px | 1.840 px |
| 7 | 0.038 px | 1.986 px |
| 11 | 0.012 px | 2.095 px |

So **0.01–0.05 px** against a ~2 px baseline, in ~9 s of CPU. The gate in the test suite
is set at 0.15 px so a real regression trips it and interpolation noise does not.

### In the benchmark harness

Driven through `benchmarks/runner.py` on its 64³ synthetic case (60 angles, numpy
projector, COM pre-alignment, 40 iterations per stage over binnings 2 and 1):

| aligner | rms dy | rms dx | max dy | max dx | ≤ 1/3 voxel? |
| --- | --- | --- | --- | --- | --- |
| `null` (no alignment) | 2.244 px | 0.752 px | 6.006 | 1.807 | no |
| `joint_gd` | **0.009 px** | **0.025 px** | 0.014 | 0.048 | **yes** |

after removing the unobservable gauge modes, against a 0.333 px target. 66 s on CPU.

### The ASTRA path, checked on a GPU node

The unit tests cannot exercise `AstraProjector3D`, so it was validated separately on the
same synthetic phantom on an A100 node (astra-toolbox 2.5.0, synthetic data only):

* ASTRA's FP/BP pair is adjoint to **0.000%**; the numpy pair to 5% (interpolation
  asymmetry, which the SIRT preconditioners absorb).
* The two backends forward-project the same phantom to within **0.54%** relative — so
  they agree on geometry and handedness, not just on array shape.
* Recovery error **0.038 px** (ASTRA) against 0.048 px (numpy) on the same data, in
  **0.9 s against 16.6 s** — an 18× speed-up on a phantom small enough to be almost all
  overhead. On real data the gap is the whole reason ASTRA is the default.
* The degraded-projection failure reproduces identically on the ASTRA backend
  (projection 7 escapes to 46 px, MAD rule catches it), and the loss back-off fires
  three times mid-run without derailing convergence.

"Observable modes" matters: the objective is blind to any shift pattern a rigid motion
of the volume reproduces — a constant `dy`, and `{sin θ, cos θ, 1}` in `dx`. Comparing
raw shifts would fail a *correct* alignment. `observable_error()` in the test file
projects those out, and is the same yardstick `tests/test_ptycho_engine.py` uses for the
reprojection engine, which is what makes the two methods comparable at all.

## Known weaknesses

Stated plainly, because this is one of three benchmark entries and the comparison is
only worth anything if each entry's limits are on the table.

1. **It treats vertical and horizontal symmetrically, in one joint loop.** This is the
   biggest structural criticism. Vertical alignment is *decoupled*: the object's vertical
   extent is invariant under rotation about a vertical axis, so it can be solved by
   registering 1-D row-sum profiles, with no forward or back projection at all. Spending
   full 3-D projections on it — inside the same loop that needs them for the horizontal
   problem — is waste, and worse, it lets a vertical error and a horizontal error trade
   against each other in a single residual. The roadmap's order of operations is
   ramp/offset → rotation centre → **vertical** → **horizontal** → non-rigid, and this
   method collapses the middle two. Mitigation is available but not the default:
   `JointGDConfig(align_vertical=False)` freezes `dy` at its initial value and leaves
   this loop the horizontal problem it is actually needed for.
2. **It registers on the phase, not on the phase gradient.** This is the single
   highest-value change not made in this port. Ptychographic phase retrieval leaves two
   ambiguities behind — an arbitrary constant offset and a linear ramp — and
   differentiating kills both (the constant goes to zero, the ramp goes to a constant
   absorbed by the comparison). That is the whole reason Odstrčil is the ptycho-tomo
   reference method. This port computes its residual on the phase, so it inherits a hard
   dependence on `preprocess.remove_phase_ramp` having done its job on a genuinely
   vacuum border. Where the border is not vacuum, a residual ramp is
   *indistinguishable* from a lateral shift and the loop will happily "correct" it
   forever. Weakness 2 and weakness 1 are the two things a follow-up should fix.
3. **Rigid translation only.** Two parameters per projection. No rotation-axis tilt, no
   magnification drift, no non-rigid deformation. If the axis tilts, that error goes
   somewhere — into the shifts, where it will not fit.
4. **The volume restarts from zero at every stage boundary.** This matches the original
   (`vol0=None`) and is preserved deliberately, but it throws away a converged coarse
   volume that could have been upsampled into the next stage. It costs the warm-up
   iterations at every stage. Changing it changes the numbers, so it was not changed
   here.
5. **The quality weights damp the volume update only, never the shift update.** That is
   the original's behaviour and it is defensible — a bad projection still needs its own
   shift estimated — but it means a badly-weighted projection is held in place by
   nothing except the MAD rule at the end.
6. **The whole stack lives in RAM** and is re-binned at every stage boundary. Peak use is
   roughly `stack + stack/binning² + 3 volumes`; a 907 × 1488 × 1816 float32 stack is
   9.8 GB before anything else. This is a batch job, not an interactive one.
7. **No convergence criterion.** The schedule runs its iteration count and stops. The
   loss is reported per iteration and the caller can stop early, but nothing decides for
   you that it is done.
8. **It reports no accuracy estimate.** The residual alignment error should be at or
   below 1/3 of the target voxel and that has to be verified separately, by split-data
   FSC — *and* by reprojection-residual maps, because **FSC cannot detect a systematic
   geometric bias**. A rigid-but-wrong geometry applied identically to both half-sets
   gives a deceptively good FSC: on a phantom, the half-bit FRC read exactly 508.6 nm at
   centring errors of 0, 4, 8, 16, 32 **and** 64 px while the true edge blur grew to
   128 px. Never sign off on this method — or any of them — on FSC alone.

## Where it sits among the three

| | `core/engine.py` (Gürsoy JIRR) | `core/joint_gd.py` (this) | vertical mass fluctuation |
| --- | --- | --- | --- |
| solves | both directions | both directions (or horizontal only) | vertical only |
| needs a volume | yes, one full recon per outer iteration | yes, one FP + one BP per iteration | **no** |
| registration | phase cross-correlation on the phase | analytic gradient on the phase | 1-D profile registration |
| robustness | divergence + runaway detection | damping, cap, warm-up, MAD | trivially robust |
| cost | high | high, GPU-bound | negligible |

The intended composition, following the roadmap: ramp/offset removal → rotation centre →
vertical by mass fluctuation → **horizontal by this method with `align_vertical=False`**
→ verification. Running this method on both directions at once, as the original did, is
the benchmark's control, not its recommendation.
