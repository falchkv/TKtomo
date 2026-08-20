# Implementation notes — phase-contrast tomography alignment

**Branch:** `feature/phasetomo-alignment` · **Last updated:** 2026-08-20 ·
**Status:** living document — update the checklist and add a decision-log row in the
same commit that changes what the code does.

This is the answer to "are we on track?". It has four parts:

1. a **checklist** keyed to the roadmap's section-4 pipeline, steps [0]–[6], each with
   the file that implements it and the test that proves it;
2. a **decision log** — what was tried, what the numbers said, what was adopted, what
   was rejected and *why*. The rejections are the valuable half: six of them are
   negative results from a full measurement campaign and each one saves the next
   person days to weeks;
3. **open questions** and the next decision point;
4. **what is not done**, and what it would take.

Rules for keeping it honest: every number here is measured and says where; a claim that
was made and then contradicted stays in, with both measurements. Where the code and a
measurement disagree, the disagreement is stated rather than resolved by editing the
prose.

> **Data policy.** No measured data is in this repository and none may be. Numbers below
> that come from a real 907-projection P06 ptycho-tomography scan are labelled
> *(internal dataset)*; the dataset itself is not committed, not referenced by path, and
> the benchmark reads a user-supplied volume plus a fully synthetic fallback phantom so
> the repo's own tests run for anyone.

---

## 1. Checklist — the roadmap's section-4 pipeline

| # | stage | status | implemented in | proven by |
|---|---|---|---|---|
| **0** | ramp / offset removal | **done** | `core/preprocess.py` — `remove_phase_ramp`, `remove_phase_offset`, `background_mask`; measured by `diagnostics.probe_vacuum_phase`, which *calls* it so there is one ramp-fitting implementation | `tests/test_ptycho_preprocess.py` (6 passed), `tests/test_diagnostics.py` (56 passed) |
| **1** | rotation centre | **partial** | `core/com.py` — `find_center` (the centroid sinusoid's offset *is* the axis), `center_is_plausible`; `diagnostics.probe_center_sweep` (entropy sweep) and `probe_center_consistency` (θ vs θ+180 flip); `AlignConfig.refine_center` | `tests/test_ptycho_com.py` (5 passed), `tests/test_diagnostics.py` |
| **2** | vertical | **done** | `core/vertical.py` — `align_vertical`, 1-D mass profiles, upsampled-DFT sub-pixel, truncation detection; wired as stage 1 of `core/odstrcil.py` | `tests/test_vertical_alignment.py` (23 passed), `tests/test_odstrcil.py` (28 passed, 1 skipped) |
| **3** | horizontal | **done** | three methods: `core/engine.py` (`AlignmentEngine`, incumbent JIRR), `core/odstrcil.py` (`OdstrcilEngine`, gradient domain), `core/joint_gd.py` (`JointGDAligner`, joint optimiser + ASTRA/numpy projectors) | `tests/test_odstrcil.py`, `tests/test_joint_gd.py` (42 passed, 1 skipped), `tests/test_benchmark.py` (32), `tests/test_run_three_way.py` (28); `tests/test_ptycho_engine.py` **skips entirely** without tomopy |
| **4** | geometry refinement (axis tilt in-plane / out-of-plane, angle readback, magnification) | **partial — detection only** | detection: `diagnostics.probe_axis_tilt` (modes 4/5/6, three-slice arc test), `probe_angle_readback`, `probe_scale_drift`. Correction: only via manual landmarks — `tktomo/tracking/model.py` fits rotation axis + tilt drift + per-view shifts and exports ASTRA `parallel3d_vec` | `tests/test_diagnostics.py`, `tests/test_tracking_model.py` (19 passed) |
| **5** | residual check / validation | **partial** | `benchmarks/metrics.py` — `score_shifts` (gauge-aware), `reprojection_residual` (+ lag-1, peakiness), `residual_plateau`, `fourier_shell_correlation` (with its caveat attached to every report); `nonrigid_gate.measure_plateau` | `tests/test_benchmark.py`, `tests/test_run_three_way.py`, `tests/test_nonrigid_gate.py` (42 passed, 11 failed — see §1.1) |
| **6** | non-rigid | **done — and gated** | `core/deformation.py` (DVF, warp, compose, invert, optical flow), `core/nonrigid.py` (`NonRigidAligner`), `core/nonrigid_gate.py` (`evaluate_gate`, five outcomes), `benchmarks/scenario_nonrigid.py` | `tests/test_nonrigid.py` (24 passed), `tests/test_deformation.py` (23 passed, 4 failed), `tests/test_nonrigid_gate.py` |

Cross-cutting, not a pipeline stage:

| capability | status | where | proven by |
|---|---|---|---|
| executable artifact-to-cause table (12 modes, 11 probes, `diagnose` / `triage`) | **done** | `tktomo/diagnostics/` | `tests/test_diagnostics.py` — 12/12 injected modes detected, 11/12 rank first, 12/12 triage stops at the right probe, 2 null controls fire nothing |
| ground-truth benchmark + perturbation catalogue | **done** | `benchmarks/` | `tests/test_benchmark.py`, `tests/test_run_three_way.py`; `docs/benchmark_results.md` |
| decision gate for stage [6] | **done** | `core/nonrigid_gate.py` | `tests/test_nonrigid_gate.py` |

### 1.1 Why stages [5] and [6] read "partial"/"failed" today

The 15 failures in `tests/test_deformation.py` (4) and `tests/test_nonrigid_gate.py` (11)
are **not** logic errors in that code. They are caused by an interpreter/NumPy defect on
this cluster's default Python, diagnosed in decision-log entry **E1**: on
CPython 3.14.0 + NumPy 2.2.6, `a * b` inside a **function scope** writes its result into
`a` when `a` is a live local array above the elision threshold. `deformation._horn_schunck_level`
does exactly `gradient * averaged` on a live `gradient`, so the Horn–Schunck iteration
corrupts its own gradient, overflows and returns all-NaN. Measured directly, same code,
same input:

| interpreter | `estimate_flow` result |
|---|---|
| CPython 3.14.0 / NumPy 2.2.6 | **NaN fraction 1.00** |
| CPython 3.11.15 / NumPy 1.26.4 | NaN fraction 0.00, 0.815 px rms field |

So stage [6] is implemented and correct, and is **unrunnable on the default
interpreter** until either the module defends itself with explicit ufuncs
(`np.multiply(..., out=...)`) or the environment moves to NumPy ≥ 2.3. That is a
shipping blocker for the non-rigid stage and is open question **Q5**.

Stage [5] is "partial" for a different reason: it is complete against *injected ground
truth*, and there is no ground truth on real data. What ships for a real dataset is the
reprojection-residual map plus a plateau test plus a caveated FSC — a self-consistency
argument, not a certificate. See **Q3**.

---

## 2. Decision log

### 2.1 Adopted

**A1 — The vertical mass-distribution stage. Adopted; this is the strongest result of
the contribution.**
Row-sum each projection to a 1-D profile and register the profiles; no forward or back
projection, no volume. Measured on the synthetic benchmark, 5 seeds, gauge-removed RMS:
`odstrcil` **0.006 ± 0.000 px** vs incumbent `jirr` **0.013 ± 0.001 px** — 2.3×, decisive
(5/5 seeds, paired −0.0076 ± 0.0011 px), and the ablation attributes it to the vertical
stage (turning stage 1 off gives 0.016 px, ratio 0.356). It is also the difference
between converging and aborting under a strong ramp: at ≥2 rad `jirr`'s runaway guard
fired on 3/3 seeds while `odstrcil` finished, and with stage 1 disabled the same engine
aborts too.
*Caveats measured, not assumed:* the 2.3× **does not transfer to real sample structure**
(three-way tie at 0.022 px), and "vertical is the easy direction" is a statement about
well-posedness, not about magnitude — on the internal dataset vertical shifts are 3.3×
*larger* than horizontal and projection-space vertical FRC is 1.8× *worse*
(359 nm vs 199 nm).

**A2 — Gradient-domain registration: implemented and kept, but the claim is narrowed.**
The pairwise property is exactly as advertised and is the cleanest measurement in the
campaign: a linear ramp of 0–100 rad across the frame moves the gradient-domain estimate
by **4.4e-16 px** and the value-domain estimate by **−0.59 px at 1 rad and −25.7 px at
10 rad**. The value-domain baseline is not a straw man — it reproduces
`skimage.registration.phase_cross_correlation` to 1e-9 px.
**What did not survive is the end-to-end benefit.** Measured inside the *same engine*
with one knob changed (`OdstrcilConfig.gradient.domain`), on 3–5 seeds per point:

| ramp across frame | gradient / value `dx` error ratio |
|---|---|
| 0 rad | 1.04 |
| 0.25 rad | **1.50** |
| 0.5 rad | **1.35** |
| 1 rad | **1.19** |
| 2 rad | **1.06** |
| 4 rad | 0.54 |

and on real sample structure 0.238 px (gradient) vs 0.210 px (value). Mechanism:
differentiating is a high-pass, and it puts the comparison's weight exactly where a
half-converged SIRT reprojection is least faithful. **Decision:** keep the estimator,
keep `domain="gradient-both"` as the `OdstrcilConfig` default *for now*, and state the
narrowed claim in the docstrings — it is insurance against a phase background that
step [0] cannot model, not a free improvement. Flipping the default is **Q1** and is
blocked on **Q2**.

**A3 — Both derivatives, not just `d/dx`.** The a-priori argument (the shift acts along
x, one derivative is cheaper) was wrong when measured: `gradient-x` 0.256 px clean /
0.389 px at 4 rad vs `gradient-both` 0.167 / 0.263. `gradient-x` misses the ⅓-voxel
target at 4 rad; `gradient-both` does not. Adopted `gradient-both`, and the test asserts
the ranking so the docstring table cannot drift from the measurement.

**A4 — The joint gradient-descent optimiser, ported rather than transliterated.**
`core/joint_gd.py`: env-var configuration became `JointGDConfig` + named stage schedules,
the loop became one-iteration-at-a-time, projectors became a registry (ASTRA / numpy).
The port **found and fixed a real sign bug**: the original script's shift convention is
the negative of TKtomo's `apply_shifts`, and a faithful transliteration produced a
perfectly converged, sign-inverted answer that passed every self-consistent accuracy test
(the harness scored it 4.49 px while reporting that negating would score 0.009 px).
Adopted with the negation in one documented place and a test that asserts against the
engine's own pinned numbers rather than against another number the module produced.
**Cost verdict reverses with scale**: on the small synthetic phantom it is 175.8 s
against 1.8–2.1 s for the engines, but on the real-scale case with ASTRA on one V100 it
is **41.6 s against 86–91 s** for the engines on 32 CPU cores, with the best `dy` and
second-best `dx`. Do not quote the synthetic cost comparison.

**A5 — Refusing GridRec with limited angular range, and FBP initialisation, in code.**
`odstrcil.check_reconstruction_choice` raises `LimitedAngleError`; the policy is
three-valued but gridrec + limited angle stays fatal even under `"warn"`. The benchmark's
own numpy backend deliberately has **no gridrec** and says so rather than degrading.

**A6 — Median-centring and explicit gauge handling everywhere.** The mean `dx` is
degenerate with the rotation-axis position and the mean `dy` with the z origin, so both
random-walk; `joint_gd.clean_shifts` centres and MAD-rejects, and `metrics.score_shifts`
projects out `{1}` from `dy` and `{1, sin, cos}` from `dx`. Removing only the mean —
the intuitive choice — leaves two of three horizontal modes in and makes a *perfect*
aligner score ~0.2 px.

**A7 — The artifact table as executable code.** 12 catalogue rows, 11 probes,
`diagnose()` (survey, stage-discounted ranking) and `triage()` (stops at the first
firing). Validated by injection: 12/12 modes detected, 11/12 rank first, 12/12 triage
stops at the correct probe, and two null controls (clean; clean + 2 % noise) fire
nothing. Ran end-to-end on a real 907 × 1488 × 1816 stack in 108 s without loading the
9.8 GB.

**A8 — Ground truth as the primary benchmark metric; FSC demoted to a caveated
secondary.** See R7.

### 2.2 Rejected — measured, and not to be repeated

Six of these are negative results from a measurement campaign on the internal dataset;
four are from this repository's own benchmark. Every one of them is a thing that looks
obviously worth doing.

**R1 — Shift-interpolation kernel. No effect. 0.0 nm.**
Bilinear vs cubic vs Lanczos vs exact Fourier resampling of the aligned stack: **zero
measurable difference** in reconstructed resolution *(internal dataset)*. Not a tuning
axis; stop optimising it. (The *benchmark* still injects with `scipy.ndimage.fourier_shift`
deliberately, because it must use a different implementation from any aligner's, and a
5th-order spline shift of a hard edge is not an exact translation — ~0.1 px of apparent
displacement, enough to swamp a 0.333 px target.)

**R2 — Rotation-axis tilt correction. Rejected: it made things worse, and the estimators
disagree in sign.**
Two independently calibrated estimators of the in-plane axis tilt **disagreed in SIGN**,
and applying the correction made **2 of 3** measured rows worse *(internal dataset)*.
This repository reproduced the pathology independently: `diagnostics.probe_axis_tilt`
returned a confident mode-4 verdict (3.9° in-plane tilt, raw confidence 0.96) on a real
stack, and an independent per-band entropy centre sweep contradicted it **in magnitude
and in sign** (−0.009 px/row between the first two bands, not linear across the third).
A model-strain guard now halves the confidence (0.96 → 0.48) and prints the cross-check.
**Rule: do not correct an axis tilt you cannot confirm with a second, model-free
estimator**, and do not read any axis-geometry number off a dataset whose per-projection
horizontal residual is still ~13–20 px.

**R3 — A full-resolution alignment stage** on top of the binned multi-resolution ladder.
**No gain** *(internal dataset)*. The information that fixes a shift is not in the top
octave.

**R4 — Choosing the final reconstruction algorithm for resolution. It is a cost
decision, not a quality one.** Interior FRC: FBP **188.4 nm** vs SIRT-300 **190.6 nm** —
one FRC bin apart *(internal dataset)*. CGLS-15 was adopted purely for cost, **14 min vs
134 min** per volume. **This is not licence to use FBP or GridRec *inside* the alignment
loop**, where the streaks are reprojected into the simulated stack and the registration
measures the streaks (see A5). Different question, different answer.

**R5 — Edge-feature alignment with an air clamp. Rejected: it halved the resolution.**
It produced visibly cleaner projections and moved the resolution from **216 nm to
473 nm** *(internal dataset)*. The canonical example of why "looks cleaner" is not a
metric.

**R6 — POCS Fourier completion. Rejected: there is nothing to complete.**
Coverage is a genuinely complete 180°, and a controlled hold-out raised in-sample error
by **1.61×** *(internal dataset)*. Fourier completion buys something only against a real
missing wedge; run `diagnostics.probe_angular_coverage` before considering it.

**R7 — FSC as an alignment metric. Rejected outright, with proof.**
A rigid-but-wrong geometry applied identically to both half-sets gives a deceptively good
FSC: the common-mode factors `exp(∓2πik·d)` cancel exactly in the cross term, and
`tests/test_benchmark.py::test_fsc_is_exactly_blind_to_common_mode_geometric_error`
proves it to machine precision. Measured three ways:
half-bit FRC read **exactly 508.6 nm** at centring errors of 0, 4, 8, 16, 32 and 64 px
while the true edge blur grew to 128 px *(phantom)*; split-half FSC read **11.97 /
11.97 / 11.98 px** for a perfectly aligned, an unaligned and a 4.5 px-misaligned
reconstruction; and in the three-way run **19.51 px for every row of every scenario** — a
1198× range of alignment error compressed into 0.114 % of FSC.
**And the replacement is not clean either**: on the non-rigid case the reprojection
residual ranked a *wrong* rigid answer (0.218) better than the *true* shifts (0.268),
because rigid shifts absorb part of a deformation. Use both, and never quote either
alone.

**R8 — `gradient-x` as the stage-2 default.** See A3.

**R9 — "Profile registration degrades more gracefully than the centroid under
truncation." Written, then measured, then retracted.**
RMS recovery error, same synthetic profiles into both estimators:

| regime | centroid | profile registration |
|---|---|---|
| sample fully inside the FOV | **0.000 px** | 0.002 px |
| cut off at one edge | **0.74 px** | 1.16 px |
| cut off at both edges | **0.97 px** | 1.67 px |
| fully inside, artifact in the vacuum | 1.45 px | **0.002 px** |

With the sample fully inside, the centroid is *exact* and cannot be beaten. Under
truncation both fail and the correlation fails **worse**, because the fixed detector
window becomes a stationary feature the correlation partly locks onto. What registration
actually buys is immunity to *localised* corruption. Truncation is therefore something to
**detect**, not to survive — hence `truncation_flags` and a default of warn-loudly. The
test asserts this ranking so the docstring cannot drift back.

**R10 — Reaching for non-rigid because the reprojection residual is high.** On the
internal 52-hour scan, **96.4 %** of the reprojection residual could not be explained by
any rigid or 3×3 block-rigid displacement; the residual was strongly localised (top 10 %
of pixels held 83.6 % of its power, 5.2× stronger inside the sample support than in air)
but **incoherent in angle** (spatial coherence 0.11, angular coherence 0.15). A 52-hour
deformation is smooth in acquisition time; this was not. Diagnosis: per-projection
ptychographic reconstruction error, not deformation. This is precisely the case
`nonrigid_gate` returns `ACCEPT_RIGID` for, and it is why the gate exists.

**R11 — Enforcing a zero vacuum phase plane per projection, beyond step [0].**
The effect is real and was measured: per-view horizontal tilt ranging −1.29 to +0.56 rad
across the frame, wandering 0.219 rad rms between angles 0.08° apart, of which only ~60 %
is a smooth function of angle *(internal dataset)*. It bought **0 / 0 / 9.7 nm** of
interior FRC at three rows, and 2.217 % → 2.174 % of unexplained variance. Real, but ~3 %
of the sample's own 11 rad contrast, so not what limits that dataset. Deprioritised, not
disproven.

### 2.3 The one place two measurements disagree

**The gradient data term helped on real data and hurt on the benchmark.** Both are
measured; they are not the same experiment and neither is wrong.

| | TKtomo benchmark (A2) | internal-dataset ablation |
|---|---|---|
| estimator | correlation peak of `∇` images, inside `OdstrcilEngine` | Gauss–Newton shift step on `‖∇FP(v) − ∇T_s d‖²`, inside a multi-resolution joint optimiser |
| data | 64 px synthetic phantom, numpy SIRT, 12 outer iterations | 1816 px real projections, 907 views, binning ladder |
| result | gradient 1.06–1.50× **worse** at ramps 0.25–2 rad | interior FRC **267/188/196 → 239/151/156 nm** (−28 to −40 nm) |
| independently checked? | ablation is same-engine, one knob | yes — vertical mass profile (no reconstruction) and reprojection residual both prefer it; and the FSC trap was caught live, two under-converged gradient runs scored *better* on FRC while the vertical-mass estimator refuted them (regression slope 0.40 and 0.71) |

The real-data gain also required conditioning the estimator: at `GRAD_SIGMA=1` the run
was under-converged and returned a *shrunken* solution (regression slope 0.40 on `dy`),
because noise in the second derivatives inflates the Gauss–Newton denominator and
attenuates every step — textbook regression dilution. At `GRAD_SIGMA=3` it converged onto
and past the phase solution. **Conclusion carried forward: differentiating helps when it
is the *data term of an optimiser* whose denominator is properly conditioned, and hurts
when it is the *similarity measure of a correlation* against a half-converged
reprojection.** That distinction is not currently expressed anywhere in the code, and it
is the sharpest open technical question in the project.

### 2.4 Environment findings that are not about the method

**E1 — CPython 3.14.0 + NumPy 2.2.6 corrupt live local arrays. Reproduced here,
function-scope only.** Two agents reported this and one could not reproduce it; the
disagreement is now resolved — **it fires only inside a function**, which is why a
module-scope canary comes back clean. Canary
(`np.float32`, shape (48, 28, 64) = 344 KB):

| construct | function scope | module scope | py3.11 / numpy 1.26 |
|---|---|---|---|
| `a * b` (`a` a live local) | **mutated** | clean | clean |
| `a - b` | **mutated** | clean | clean |
| `np.linalg.norm(a - b)` | **mutated** | clean | clean |
| `np.sum((a - b)**2, axis=…)` | **mutated** | clean | clean |
| `np.subtract(a, b)` | clean | clean | clean |

A 16 × 16 array is unaffected — it is below NumPy's elision size threshold. Cause:
CPython 3.14's `LOAD_FAST_BORROW` does not incref, so NumPy judges an ordinary named
array to be a dead temporary and writes into it. No error, no warning.
**Known impact in this repository:**
- `core/engine.py:470` does `float(np.linalg.norm(prj_aligned - sim) / denominator)`
  inside `step()`, and line 505 then caches `self._last_aligned = prj_aligned`. On this
  interpreter the GUI's cached aligned stack is silently the *difference* every
  iteration. `core/odstrcil.py:507` already defends with `np.subtract` and says why;
  `engine.py` should do the same. **Not fixed here — engine.py is not this document's
  file to edit; raised as Q5.**
- `core/deformation.py:562–563` (`gradient * averaged`, `gradient * flow`) — this is
  what makes `estimate_flow` return all-NaN and takes 15 non-rigid tests with it (§1.1).
**Mitigations:** use explicit ufuncs (`np.subtract`, `np.multiply`) at every point where
an operand is a live local that is used again; or run stage [6] and any GUI session under
CPython 3.11 / NumPy 1.26; the class of bug disappears on NumPy ≥ 2.3.

**E2 — tomopy is installed in no environment on this cluster.** `AlignmentEngine.step`
hard-requires `tomopy.prep.alignment.shift_images` and `blur_edges`, so
`tests/test_ptycho_engine.py` **skips entirely** and the incumbent cannot run without
`benchmarks.runner.tomopy_shim` (scipy stand-ins, installed only when tomopy is genuinely
absent, removed on exit, declared in every report's `environment.tomopy_shim`). Every
`jirr` number in `docs/benchmark_results.md` was produced under that shim and should be
re-measured on a machine with tomopy before being quoted externally. The interpolation is
the same 5th-order spline but skimage's `warp` clips to the input range and scipy's
`shift` does not, a sub-0.01 px difference — the same order as the clean-case number
itself.

**E3 — Two environments, neither complete.** scikit-image (needed by the incumbent) and
ASTRA (needed by `joint_gd`'s production path) are not installed in the same conda
environment, so the real-scale three-way comparison needed two SLURM jobs in two
interpreters. The case was bit-identical (null and oracle rows agree to ten decimals) and
the shift scores are pure numpy, but the FSC column is **not comparable across them** —
the same FSC code on the same data gave 32.50 px under py3.14/NumPy 2.2 and 7.78 px under
py3.11/NumPy 1.26. `benchmarks.metrics._first_crossing` is worth a look.

---

## 3. Open questions, and the next decision point

**Q1 — Should `OdstrcilConfig.gradient.domain` default to `"value"`?**
On the evidence in A2 the value domain wins in 10 of 11 headline scenarios and in all 6
no-vacuum runs, and loses only at a 4 rad ramp. Against that, §2.3 shows a *differently
formulated* gradient objective gaining 28–40 nm on real data. **Do not flip the default
on one phantom.** Blocked on Q2 and Q4.

**Q2 — The benchmark cannot currently express the failure the gradient trick exists
for.** A linear ramp lies exactly in the span of the plane `remove_phase_ramp` fits, so
least squares removes it *whatever the mask contains* — measured: after step [0],
projections injected with 0.5 rad and 2.0 rad of ramp differ by **1.9e-6 rad on a 15 rad
scale**, even with the vacuum border cropped away. Until `PerturbationSpec` grows a
**non-planar** phase background (low-order polynomial or a smooth random field per
projection), the gradient claim is *untested*, not refuted. This is the single most
informative thing to add.

**Q3 — How is the ⅓-voxel target certified on real data, where there is no truth?**
Today: reprojection-residual map + plateau + a caveated FSC. That is self-consistency,
not a certificate, and R7 shows the residual is itself gameable. A candidate answer is a
hold-out: withhold every k-th projection from the alignment and score the residual on the
withheld views. Nothing in the repo does this yet.

**Q4 — Does the gradient/value crossover move with detector size?** The crossover sat at
4 rad on a 64 px detector and appeared to move toward lower amplitudes as the loop got
better. Rerunning §3.1 of `docs/benchmark_results.md` at 128 and 256 px decides Q1.

**Q5 — Fix `engine.py` (and `deformation.py`) against E1, or pin the environment?**
Cheapest correct answer: explicit ufuncs at the three known sites plus a canary test in
the suite (`test_odstrcil.py` already carries one). Otherwise stage [6] does not run at
all on the default interpreter. **This is the next decision point** — it blocks the only
stage that is implemented but unrunnable.

**Q6 — Should the reference projector be promoted out of the tests?**
`tests/test_odstrcil.py` carries an exact-adjoint numpy/scipy parallel-beam projector and
`benchmarks/runner.py` carries a sparse one. Neither is in `tktomo/recon/`, so the
library still has no dependency-free reconstruction backend and
`tests/test_ptycho_engine.py` skips without tomopy. Promoting one would make the repo's
own definition-of-done test runnable in CI.

**Q7 — Where does the series-aligner registry live?** It is currently inside
`odstrcil.py` (mirroring `tktomo/align/base.py` and `tktomo/recon/backend.py`) because a
package-root registry is a shared file. If a canonical one is created, the two
registrations move there and nothing but the import path changes.

**Q8 — `joint_gd`'s bimodal `dy` basin.** Two of five *clean* seeds converge to a `dy`
100× worse than the other three (0.875 and 1.102 px vs 0.008–0.009) and stay there at
double the iteration budget, so it is a basin, not under-convergence. Reported, not
diagnosed. It did not recur on the real-structure case (a single draw). Also: at a 4 rad
ramp `joint_gd` returned **52.8 px** of `dx` error with `status="ok"` while both engines
refused — MAD outlier rejection cannot see a failure that moves every projection
together, so it needs a consensus guard before it is offered as a default.

**Q9 — `triage()` stops too eagerly.** It returns a 0.05-confidence `tilt_axis_angle` in
preference to a 0.43-confidence `vertical_drift`: 3 of 7 scenarios correct, against 6 of
7 for `diagnose()`. It needs a minimum confidence before it is allowed to stop at a
stage.

**Q10 — The vertical stage's truncation verdict is a perfect detector with the wrong
label.** It fired on 20/20 ramped seeds and 0/25 unramped ones — perfect separation — and
reports "the sample is not fully inside the field of view" when the sample is entirely
inside and the actual cause is a residual ramp (which adds a linear-in-v term to every
profile). Rename it, or defer to `probe_vacuum_phase`. The remedy is the same in both
cases (fix the preprocessing), which is why it was left.

---

## 4. What is NOT done, and what it would take

| gap | what is missing | what it would take |
|---|---|---|
| **Stage [4] correction** | detection exists for axis tilt (in-plane and out-of-plane), angle readback and magnification; there is **no automatic corrector**. The only path today is manual landmarks through `tktomo/tracking/` | a geometry-refinement step that fits the axis direction jointly with the shifts and hands a vector geometry (ASTRA `parallel3d_vec`) to the projector. Note R2: it must ship with a second, model-free estimator, or it will make things worse |
| **A full-resolution certificate** | nothing certifies any method at full detector size. The real-scale case ran at bin 8 with 130 of 907 views, because the benchmark's sparse projector caps at 40 M non-zeros. ⅓ of the full-resolution voxel is 24.84 nm = 0.042 binned px; the best measured row is 125 nm — **5× outside** | a GPU/ASTRA-backed `ReconBackend` shared by the harness and all methods, so one run covers every method at full size (Q6, and item 10 of `docs/benchmark_results.md` §14) |
| **Non-planar phase background** | `PerturbationSpec` has `phase_ramp_rms`/`phase_offset_rms` only, and a linear ramp is exactly removable by step [0] | Q2 |
| **Real-data validation of the new methods** | `OdstrcilEngine` and the TKtomo `JointGDAligner` have never been run on a real stack end to end. The real-structure benchmark case is *forward-projected from* a reconstructed volume, not measured projections | one SLURM run per method on a real stack, scored by reprojection residual + vertical-mass cross-check (there is no truth), plus a rebuilt volume and a 3-D FSC |
| **Stage [6] on the default interpreter** | see §1.1 / E1 | Q5 |
| **Seeds and error bars on the decision-relevant number** | the 1.59× `odstrcil`-over-`jirr` margin in `dx` on real structure is a **single draw, single binning, single angular subsampling**, and at bin 8 with 130 views it is Crowther-undersampled for its own detector width | repeat over several seeds and at bin 4 before quoting it externally |
| **CI** | the suite does not run clean in any single environment here (tomopy, msgpack, pyqtgraph all missing) | pin an environment with tomopy + msgpack + pyqtgraph, or make the tests that need them skip rather than error |
| **A resolution/coverage calculator in the library** | Crowther, ⅓-voxel and FRC arithmetic are done ad hoc in scripts | a small `tktomo` module so the target is computed the same way everywhere |

### Calibration worth carrying forward

On the internal dataset, injecting known random per-projection error and measuring the
interior FRC gives `R(σ)² = R₀² + c²σ²` with **R² = 0.9999** and
**c = 157.5 nm per px of rms horizontal error, 45.8 nm per px vertical** — the metric is
**3.4× more sensitive to horizontal than to vertical error**. Two consequences:

1. At the ⅓-voxel target (0.333 px) perfect alignment would buy about **7.5 nm**; at
   0.5 px, 17.3 nm. On that dataset alignment residual is **not** the binding constraint
   — per-projection ptychographic quality is (see R10).
2. **The interior FRC is a horizontal metric and is nearly blind to vertical error.**
   Scaling `dy` by 0.8–1.2 moved the measured resolution by ≤3 nm; scaling `dx` by 0.8
   cost 63 nm. Any campaign that measured its resolution story that way measured it with
   an instrument that cannot see the axis it should worry about most.
   *(The calibration cannot be inverted to bound the residual error of a fitted solution:
   injected error is independent of the data, whereas the error surviving a joint fit
   lies in the null space of the objective — and the odd/even FRC lives in the same null
   space.)*

---

## 5. Repository coherence check — 2026-08-20

Run: `python -m pytest -q --continue-on-collection-errors`, CPython 3.14.0 / NumPy 2.2.6
/ SciPy 1.16.3 / scikit-image 0.26.0.

```
66 failed, 474 passed, 6 skipped, 33 warnings, 85 errors in 141.71s
```

Attribution of every failure — nothing is unexplained:

| count | tests | cause | ours? |
|---|---|---|---|
| 78 errors + 2 failed | `test_session_conformance.py` | `ModuleNotFoundError: tomopy` (and `msgpack`) | no — environment |
| 44 failed | `test_session_codec.py` | `ModuleNotFoundError: msgpack` | no — environment |
| 6 errors + 2 failed | `test_session_remote.py` | `msgpack` / `tomopy` | no — environment |
| 1 collection error | `test_plane_cache.py` | `ModuleNotFoundError: pyqtgraph` | no — environment |
| 2 failed | `test_messaging.py` | `msgpack` | no — environment |
| 1 failed | `test_apply.py` | `msgpack` | no — environment |
| **11 failed** | `test_nonrigid_gate.py` | **E1** (NaN from `estimate_flow`) | **yes — blocked, §1.1** |
| **4 failed** | `test_deformation.py` | **E1** | **yes — blocked, §1.1** |

Missing modules across the run: `tomopy` ×84, `msgpack` ×53, `pyqtgraph` ×1.

Per-file, for the alignment work specifically (same interpreter):

| file | result |
|---|---|
| `test_ptycho_preprocess.py` | 6 passed |
| `test_ptycho_com.py` | 5 passed |
| `test_vertical_alignment.py` | 23 passed |
| `test_gradient_registration.py` | 22 passed |
| `test_odstrcil.py` | 28 passed, 1 skipped (tomopy-gated) |
| `test_joint_gd.py` | 42 passed, 1 skipped (tomopy-gated) |
| `test_ptycho_engine.py` | **1 skipped — the whole file, no tomopy** |
| `test_diagnostics.py` | 56 passed |
| `test_benchmark.py` | 32 passed |
| `test_run_three_way.py` | 28 passed |
| `test_nonrigid.py` | 24 passed |
| `test_deformation.py` | 23 passed, **4 failed** (E1) |
| `test_nonrigid_gate.py` | 42 passed, **11 failed** (E1) |
| `test_tracking_model.py` | 19 passed |

**Data policy: clean.** `find` over the whole tree returns **no** `.h5`, `.hdf5`, `.npy`,
`.npz`, `.tif`, `.tiff` or `.cxi` anywhere, and **no absolute path rooted at the
beamtime mount point** appears in any `.py`, `.md`, `.toml`, `.sh` or `.json` in the
repository (grep for the mount prefix returns nothing but this sentence's own
description). The benchmark's
provenance string for a user-supplied volume is literally
`"user-supplied volume (not committed)"`.

**Licence: present** — `LICENSE` is BSD-3-Clause, "Copyright (c) 2026, Kim Vegard Falch
and TKtomo contributors".
**Inconsistency to fix (not this document's file):** `pyproject.toml` declares
`license = { text = "MIT" }` while `LICENSE` is BSD-3-Clause, and the `authors` field
reads "Ken Vidar Falch" against the LICENSE's "Kim Vegard Falch". One of each pair is
wrong.

**Other things noticed and not papered over:**
- `benchmarks/results/` (≈70 files: gzipped per-case records, JSON, PNG) is committed
  inside the repo. It contains scores and curves only — no sample imagery, no volume
  path — but it is generated output in a source tree, and `.gitignore` does not cover it.
  Worth a deliberate decision.
- `.idea/` and `.pytest_cache/` are present in the working tree; `.gitignore` covers
  `.pytest_cache/` and `.idea/workspace.xml` but not `.idea/` itself.
- `README.md`'s build order and architecture section predate `ptycho_align/`,
  `diagnostics/`, `benchmarks/` and `tracking/`; nothing links to
  `docs/alignment_roadmap.md`, `docs/diagnostics.md`, `docs/nonrigid.md` or this file.

---

## Q5 — RESOLVED (2026-08-20)

The elision defect was chased to its actual root and closed in three moves:

1. **Library hardening at the measured sites.** `deformation._horn_schunck_level`
   (all-NaN flow, 4 test failures) and `nonrigid_gate.measure_localisation` /
   `measure_temporal_change` now use explicit ufuncs (`np.multiply`, `np.square`).
   The gate site was subtle and worth recording: `residual**2` elided INTO
   `residual`, so the later `values = residual[:, padded]**2` built the
   permutation null from **fourth-power** values — null 0.302 instead of 0.202 on
   the deformation fixture — which flipped the verdict from RUN_NONRIGID to
   ACCEPT_RIGID while every individually-inspected intermediate looked correct.
2. **The remaining 3 "failures" were the defect corrupting the TEST code itself.**
   `measured = simulated + noise` inside a test function returned `simulated`
   (the addition was elided into it via CPython 3.14's LOAD_FAST_BORROW making a
   live local look like a refcount-1 temporary), so the test asserted on an
   identically-zero residual. Measured: `noise.std = 0.0100`,
   `(measured - simulated).std = 0.0`. No library hardening can defend
   arbitrary arithmetic in tests or user scripts.
3. **A canary conftest** (`tests/conftest.py`) probes the defect with this
   project's minimal reproducer (strided view × fresh local) and, when it fires,
   skips the entire suite with instructions, because on a defective interpreter a
   test can also silently PASS while checking nothing — worse than failing.

Verified after the fix: **335 passed, 2 skipped (tomopy-gated)** across all ten
alignment/diagnostics/benchmark test files on CPython 3.12 / NumPy 1.26.4;
CPython 3.14 / NumPy 2.2.6 skips with the canary message. Supported interpreters:
CPython <= 3.13 with any NumPy, or NumPy >= 2.3.

**Rule for this repo:** never write `a <op> b` where the left operand is a live
local ndarray you use again — use explicit ufuncs in library code, and treat any
interpreter where the canary fires as unusable for numerics, full stop.
