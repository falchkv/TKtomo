# Benchmark results

Measured 2026-08-20 on Maxwell (`allcpu`, one core used), Python 3.14 / NumPy 2.2.6 /
SciPy 1.16.3, scikit-image 0.26.0, **no tomopy** (the incumbent ran under
`benchmarks.runner.tomopy_shim`, see the README), **no GPU**.

Everything below is from the fully synthetic phantom. No measured data was used and
none is committed. Reproduce with:

```bash
python -m benchmarks.runner --name synthetic_main --size 64 --slices 12 --angles 60 \
    --jitter-dy 2.5 --jitter-dx 0.75 --seed 0 --iterations 12 --gd-iterations 80 --fsc
python -m benchmarks.runner --catalogue --size 48 --slices 6 --angles 48 \
    --iterations 10 --gd-iterations 60 --aligners null,oracle,jirr,odstrcil,joint_gd
```

---

## 1. The main case

60 projections over 180°, detector 40 × 92 px (a 64 px object inside a 14 px zero
margin), injected jitter 2.25 px RMS vertical / 0.76 px RMS horizontal — the 3:1
vertical:horizontal asymmetry our own lens-1 scan shows (25.0 / 7.5 px on 1816 px).
Target 0.333 px = ⅓ voxel.

| aligner | rms dy | rms dx | max dy | max dx | ⅓-voxel | iters | wall clock |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| `null` (do nothing) | 2.244 | 0.752 | 6.006 | 1.807 | **no** | 0 | — |
| `oracle` (truth) | 0.000 | 0.000 | 0.000 | 0.000 | yes | 0 | — |
| **`jirr` (incumbent)** | **0.013** | **0.031** | 0.026 | 0.073 | **yes** | 12 | **2.5 s** |
| `odstrcil` | 0.006 | 0.033 | 0.015 | 0.129 | yes | 12 | 3.0 s |
| `joint_gd` | 0.009 | 0.037 | 0.015 | 0.258 | yes | 160 | 202 s |
| `joint_gd` negated (sign control) | 4.489 | 1.506 | 12.013 | 3.627 | no | 160 | 203 s |

All in pixels, after removing the unobservable gauge modes (`dy`: {1}; `dx`: {1, sin,
cos}). The oracle row reads 0.000 to 1e-16, so the scoring path itself is sound.

**The incumbent baseline the other two must beat: `jirr` = 0.013 px dy / 0.031 px dx**,
i.e. 176× and 24× better than doing nothing, and 26× / 11× inside the ⅓-voxel target.
On this case none of the three is the bottleneck — all three clear the target by more
than an order of magnitude. Where they differ is shown in §3.

The negated row is a deliberate control, not a result: it is the same estimate with its
sign flipped, and it scores **exactly twice the injected RMS** (2 × 2.244 = 4.489,
2 × 0.753 = 1.506). That is the signature the harness looks for, and every row carries a
`sign_check` block reporting the flipped score and the correlation with truth.

---

## 2. FSC certifies the misaligned reconstruction as readily as the aligned one

Split-half FSC (odd/even angle subsets, half-bit criterion) and the per-angle
reprojection residual, on exactly the same runs:

| aligner | shift error dy/dx (px) | reprojection residual | residual lag-1 | **split-half FSC** |
| --- | ---: | ---: | ---: | ---: |
| `null` | 2.244 / 0.752 | 0.451 | −0.03 | **11.97 px** |
| `oracle` | 0.000 / 0.000 | 0.052 | +0.92 | **11.97 px** |
| `jirr` | 0.013 / 0.031 | 0.052 | +0.92 | **11.97 px** |
| `odstrcil` | 0.006 / 0.033 | 0.052 | +0.92 | **11.97 px** |
| `joint_gd` | 0.009 / 0.037 | 0.052 | +0.89 | **11.97 px** |
| `joint_gd` negated | 4.489 / 1.506 | 0.641 | −0.05 | **11.98 px** |

Read the last column and the first together. **The FSC is 11.97 px for a perfectly
aligned reconstruction and 11.97 px for one that was never aligned at all** — and
11.98 px for one misaligned by 4.5 px RMS, which is 340× the residual error of the
`odstrcil` row it is indistinguishable from. A ⅓-voxel-accurate alignment and a
0-effort one get the same resolution certificate to three significant figures.

This is the roadmap's claim 7, reproduced on this harness. It is not a subtlety of the
threshold or the masking: FSC measures whether two half-sets *agree*, and per-projection
jitter drawn from the same distribution blurs both half-sets the same way, so they
agree just as well about the wrong answer. `tests/test_benchmark.py::
test_fsc_is_exactly_blind_to_common_mode_geometric_error` proves the rigid case
analytically — a common translation contributes conjugate phase factors that cancel
exactly in the cross term, leaving the correlation bit-identical. It matches what we
measured independently on a phantom before this harness existed: a half-bit FRC of
exactly 508.6 nm at centring errors of 0, 4, 8, 16, 32 and 64 px while the true edge
blur grew to 128 px.

The two columns that *do* work are the reprojection residual (0.052 aligned vs 0.451
unaligned, an 8.7× separation) and its lag-1 autocorrelation across angle (+0.92
aligned vs −0.03 unaligned — a well-aligned stack leaves a residual that is a smooth
function of the geometry, a misaligned one leaves angular noise). Both are reported for
every row, and neither is a substitute for the shift-recovery error against truth.

---

## 3. Diagnostic sweep: one perturbation at a time

48 projections, detector 26 × 68 px, base jitter 1.32 / 0.51 px RMS, one extra
perturbation per case. `rms dy / rms dx` in px; **bold** = misses the ⅓-voxel target.

| perturbation | `jirr` | `odstrcil` | `joint_gd` |
| --- | ---: | ---: | ---: |
| jitter only | 0.015 / 0.029 | **0.006** / 0.039 | 0.016 / 0.044 |
| constant centre offset (3 px) | 0.013 / **0.334** | 0.006 / **0.422** | 0.016 / 0.260 |
| vertical drift (6 px) | 0.018 / 0.031 | **0.006** / 0.035 | 0.018 / 0.045 |
| axis tilt in plane (0.5°) | 0.021 / 0.031 | 0.017 / 0.047 | 0.028 / 0.043 |
| out-of-plane tilt (0.5°) | 0.016 / 0.039 | 0.007 / 0.040 | 0.020 / **0.016** |
| magnification drift (2 %) | 0.014 / 0.034 | 0.006 / 0.039 | 0.017 / 0.048 |
| angle readback error (0.3°) | 0.015 / 0.054 | 0.005 / 0.052 | 0.017 / 0.028 |
| **phase ramp + offset** | 0.020 / 0.222 | 0.014 / **0.124** | 0.029 / **0.344** |
| FOV truncation (6 px/side) | 0.015 / 0.022 | 0.006 / 0.035 | 0.017 / 0.047 |
| **non-rigid deformation (1 px)** | **0.871 / 0.830** | **0.768 / 1.408** | **0.841 / 0.995** |
| detector noise (5 %) | 0.017 / 0.026 | 0.007 / 0.033 | 0.016 / 0.044 |

Four things this says that the single-case table cannot.

**Vertical is where the methods actually separate, and Odstrcil wins it.** Across ten of
the eleven cases `odstrcil` recovers `dy` to 0.005–0.017 px against `jirr`'s 0.013–0.021
and `joint_gd`'s 0.016–0.029 — a consistent 2–3× advantage, obtained from a row-sum with
no reconstruction in the loop at all. Its stage-1 report converges in 2 iterations. That
is roadmap claim 2 confirmed: the vertical direction is decoupled from the rotation
angle, and solving it with the vertical mass distribution is both cheaper and better
than solving it by reprojection.

**The phase ramp is where the gradient trick earns its keep.** With a per-projection
linear ramp plus constant offset injected, horizontal recovery goes `jirr` 0.222 px →
`odstrcil` 0.124 px, a 1.8× improvement and the largest margin `odstrcil` shows in `dx`
anywhere in the sweep, while `joint_gd` degrades to 0.344 px and misses the target. This
is roadmap claim 3: differentiating sends the constant offset to zero and the ramp to a
constant, and a registration on the phase gradient is therefore blind to exactly the two
ambiguities ptychographic phase retrieval leaves behind. It is also the one perturbation
in this catalogue that a real ptycho-tomography dataset is guaranteed to have.

**No rigid aligner survives a non-rigid deformation, and all three fail identically.**
1 px RMS of smooth zero-mean warp costs every method 0.77–1.41 px — 50–100× worse than
any other row and far outside the target. That is the correct behaviour, not a defect:
the model has no term for it. It is also why the roadmap puts non-rigid last in the order
of operations, and why the reprojection-residual map matters — it is what tells you the
plateau you are stuck on is deformation rather than a rigid error you could still fix.

**A constant centre offset is not free.** All three lose an order of magnitude in `dx`
(0.26–0.42 px) when a 3 px constant offset is injected, and two of them miss the target,
even though a constant `dx` is nominally pure gauge. The gauge removal takes the
constant back out of the *score*; what it cannot undo is that the loop reconstructed
from a stack whose rotation axis was 3 px from where it was told, so every iteration's
reprojection was compared against a slightly wrong model. Practical reading: get the
rotation centre right before the shift loop, exactly as the roadmap's order of
operations says — it is step 2 for a reason.

---

## 4. Cost

| aligner | iterations | wall clock (main case) | per iteration |
| --- | ---: | ---: | ---: |
| `jirr` | 12 outer | 2.5 s | 0.20 s |
| `odstrcil` | 12 outer | 3.0 s | 0.25 s |
| `joint_gd` | 160 (2 stages × 80) | 202 s | 1.26 s |

`joint_gd` is ~70× more expensive here for no accuracy gain, but the comparison is not
fair to it and should not be quoted as if it were: it is a multi-resolution optimiser
whose production schedule bins by 16/8/4 and runs 400 iterations, aimed at the ~120 px
RMS lens-2 regime, and it is being asked here to walk 2 px on a 92 px frame where its
coarse stages are meaningless. It is also the only one of the three carrying MAD outlier
rejection and median centring, which cost nothing on clean synthetic data and are what
kept 20 of 918 real projections from sliding out of frame. Benchmark it again at
realistic scale before drawing a conclusion about it.

The `odstrcil` and `jirr` timings are dominated by the same SIRT reconstruction, which
is why they are within 20 % of each other despite quite different registration stages.

---

## 5. Caveats

- **The incumbent ran under a tomopy shim.** tomopy is not installed on this cluster and
  `AlignmentEngine.step` needs `tomopy.prep.alignment.shift_images` and `blur_edges`;
  `benchmarks.runner.tomopy_shim` supplies scipy equivalents for the duration of the run
  only. The interpolation is the same 5th-order spline, but skimage's `warp` clips to the
  input range and scipy's `shift` does not, so `jirr` numbers may differ from a real
  tomopy run at the sub-0.01 px level — which is the same order as the number itself.
  **Re-measure the `jirr` baseline on a machine with tomopy before quoting it in a
  paper.** `environment.tomopy_shim` in every report says whether the shim was used.
- **The reconstruction backend is `benchmarks.runner.NumpyProjectorBackend`**, not
  tomopy or astra. All three aligners used the same one, so the comparison between them
  is fair, but absolute residuals are not comparable to a tomopy-based run.
- **One phantom, one seed, one size.** These are single-draw numbers, not distributions.
  Nothing here has error bars; a claim about which of `jirr` and `odstrcil` is better in
  `dx` (0.031 vs 0.033) is not supported by one draw.
- **The synthetic-from-real generator has not been run against our volume yet.** The code
  path (`load_volume` → `volume_case`) is implemented and unit-tested on synthetic
  input, but a run against `rec_lens1_v4` at realistic scale is still outstanding, and
  that is the one that would test the methods at the jitter amplitude and object
  complexity they were designed for.
