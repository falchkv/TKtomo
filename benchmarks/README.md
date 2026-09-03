# Alignment benchmarks

A harness for measuring tomographic **series** aligners against **known injected
ground truth**, plus a fully synthetic phantom so the whole thing runs with no
external data, no GPU, no tomopy and no astra.

```bash
python -m benchmarks.runner --size 64 --slices 12 --angles 60 --iterations 12
```

That builds a synthetic case, runs every aligner in the repo on it, prints a
comparison table, and writes `benchmark_results/<case>.json` and `.png`.

---

## Why ground truth, and not FSC

An aligner cannot be benchmarked by FSC. A rigid-but-**wrong** geometry applied
identically to both half-sets gives a deceptively good FSC, because a common-mode
translation multiplies the two transforms by `exp(-2πik·d)` and `exp(+2πik·d)` and the
factors cancel exactly in the cross term. The correlation is not merely insensitive to
common-mode geometric error — it is **bit-for-bit invariant** to it.
`tests/test_benchmark.py::test_fsc_is_exactly_blind_to_common_mode_geometric_error`
proves this to machine precision rather than asserting it.

We hit this on a phantom before writing any of this code: the half-bit FRC read
**exactly 508.6 nm** at centring errors of 0, 4, 8, 16, 32 and 64 px, while the true
edge blur grew to 128 px.

So the metrics are ranked:

| rank | metric | what it sees | what it misses |
| --- | --- | --- | --- |
| **primary** | shift-recovery error vs injected truth (`score_shifts`) | everything a per-projection rigid model can express | nothing, on this phantom — that is the point |
| secondary | reprojection residual vs angle (`reprojection_residual`) | common-mode geometry error, non-rigid deformation, angular structure | absolute scale; it is a self-consistency measure |
| secondary | split-half FSC (`fourier_shell_correlation`) | random per-projection error | **all systematic geometric bias** |
| bookkeeping | wall clock, iteration count | cost | — |

Never quote the FSC alone. The harness attaches the caveat text to every FSC block it
writes into a report, for exactly this reason.

---

## Three conventions, each of which is a bug if you get it wrong

**1. Sign.** `truth.dy[i]` / `truth.dx[i]` are the *content displacement injected*:
projection *i* was moved **down** by `dy` rows and **right** by `dx` columns.
`tktomo.ptycho_align.core.engine.apply_shifts` moves content by `-s`, so **a working
aligner reports `sy = +dy`, `sx = +dx`** — the same sign as the truth. This matches
`examples/make_phantom.py` and `tests/test_ptycho_engine.py`.

The failure signature is unmistakable: a sign-flipped estimate scores **exactly twice**
the injected RMS, and its correlation with the truth is negative. The runner computes
both and writes them into every row as `sign_check`. It does **not** silently negate a
result — auto-correcting would turn a shipping-blocking interface bug into an invisible
one.

**2. Gauge.** Some of the answer is physically unobservable and must be removed before
comparing, or a *perfect* aligner scores non-zero:

- **`dy`** — translating the object along the rotation axis shifts every projection by
  the same amount. One mode: `{1}`.
- **`dx`** — translating the object in the rotation plane by `(X, Y)` shifts projection
  *i* by `X·cos θᵢ + Y·sin θᵢ`, and a constant `dx` *is* the rotation-axis position.
  Three modes: `{1, sin, cos}`.

Removing only the mean, the intuitive choice, leaves two of the three horizontal modes
in; TKtomo's own engine test measured a perfectly correct alignment scoring ~0.2 px
that way. `score_shifts` reports the gauge-removed number as primary **and** the
mean-only and raw numbers beside it **and** the RMS amplitude of what was forgiven, so
the choice hides nothing.

**3. Injection order.** Geometry (axis tilt, out-of-plane tilt, angle readback error) is
baked into the forward projection; then magnification, then the non-rigid deformation,
then the rigid shift, then the phase ramp/offset, then FOV truncation, then noise. The
rigid shift is applied *after* every other geometric distortion, so the recorded truth
is exactly what was applied. It is injected with `scipy.ndimage.fourier_shift` —
deliberately a different implementation from any aligner's — because a Fourier shift is
an exact translation of a band-limited signal while a 5th-order spline shift of a hard
edge is not (~0.1 px of apparent displacement, enough to swamp the target). Fourier
shifts wrap, so `perturb` **refuses to run** unless the zero margin exceeds the largest
shift it is about to apply.

---

## The perturbation catalogue

Every entry is independently switchable, defaults to off (0), and is recorded as ground
truth. They map one-to-one onto the diagnostic catalogue, deliberately: each exists so
a diagnostic can be shown to fire on it and only on it.

| `PerturbationSpec` field | what it models | rigid-fixable? |
| --- | --- | --- |
| `jitter_dy_rms`, `jitter_dx_rms` | per-projection stage jitter | yes — this is the thing being measured |
| `center_dy`, `center_dx` | constant offset / rotation-axis position | pure gauge; injected to prove the scorer removes it |
| `drift_dy`, `drift_dx`, `drift_shape` | thermal drift across the scan | yes |
| `axis_tilt_deg` | rotation axis tilted *in* the detector plane | no — rigid per projection, but not a translation |
| `out_of_plane_tilt_deg` | rotation axis tilted *toward the beam* | no — a true 3D geometry error |
| `magnification_drift` | isotropic scale drift across the scan | no |
| `angle_error_rms_deg` | angle readback error (projector uses truth, aligner is told otherwise) | no |
| `phase_ramp_rms`, `phase_offset_rms` | the two ambiguities ptycho phase retrieval leaves behind | no — but a *gradient-domain* aligner is invariant to them |
| `truncation_px` | object wider than the field of view | no |
| `deformation_px`, `deformation_scale` | smooth zero-mean non-rigid warp | no |
| `noise_rms` | detector noise | no |

`cases_from_catalogue()` builds one case per entry with that perturbation alone on top
of a modest jitter. Running an aligner across the sweep says *which* geometry error it
cannot handle, which a single all-perturbations-on case never can.

---

## The two generators

Both return a `BenchmarkCase` and are interchangeable everywhere downstream.

**(a) Fully synthetic** — `synthetic_case(...)`. A 3D ellipsoid phantom (the same
construction as `tktomo.io.phantom.generate_volume`, plus two off-axis blobs so the COM
sinusoid is not trivially zero), forward-projected with a numpy+scipy parallel-beam
projector. No external data, no optional dependencies. This is what the test suite and
any outside user run.

**(b) Synthetic-from-real** — `load_volume(path) → volume_case(...)`. Forward-projects a
**user-supplied** reconstructed volume at user-supplied angles, then injects the same
known perturbations. Real sample statistics — the streak texture, the noise, the actual
phase distribution — with exact ground truth, which no real dataset can give you.

`load_volume` accepts a directory of per-slice TIFFs, a `.npy`, or an `.h5` with a
dataset path, and takes `slices=` and `bin_factor=` because a 1488 × 1816² volume is
19 GB in float32.

> **Note on what this measures.** The volume was itself reconstructed from projections
> that were aligned by *something*. Residual misalignment in that reconstruction shows
> up here as extra blur, not as extra truth: the recovered shifts are still exact, but
> the difficulty of the case is understated by however good the original alignment was.

### Our own case (a 2026 P06 ptycho-tomography campaign, lens 1)

**No measured data is committed to this repository and none may be.** The benchmark
ships as code that reads a path you supply. For our own runs, with `$TOMO` pointing at
the processed directory:

```bash
python -m benchmarks.runner \
  --name lens1_from_real \
  --volume $TOMO/rec_lens1_v4 --slice-range 700:716 --bin 4 \
  --angles-file $TOMO/lens1_v4_best.h5 --angles-dataset exchange/theta \
  --pixel-size 74.50973137 \
  --jitter-dy 25.0 --jitter-dx 7.5 \
  --iterations 12 --out $SCRATCH/results
```

Measured properties of that scan, for context when reading such a run: 907 projections
over 180.000°, median step 0.080°, 99 gaps wider than 1°, worst 1.284°; voxel
74.50973137 nm; object diameter 1377 px (102.6 µm); the pipeline's own shifts had RMS
`dy` 25.0 px and `dx` 7.5 px.

Two things about that scan are worth stating because they cut against the roadmap's
framing. Projection-space FRC per lens gives lens-1 **264 nm isotropic, 199 nm
horizontal, 359 nm vertical** — vertical is 1.8× *worse* — and the vertical shifts are
3.3× *larger* than the horizontal ones. The roadmap's "vertical is the easy, decoupled
direction" is a statement about the geometry, not about our stage: on this instrument
the vertical axis carries most of the misalignment and ends up the worse-resolved one.
That is exactly the kind of claim a benchmark exists to test, and `--jitter-dy 25
--jitter-dx 7.5` reproduces the asymmetry.

---

## Plugging in an aligner

The runner drives three shapes of thing, in decreasing order of preference.

**1. An `AlignmentEngine`-compatible engine** (`EngineAligner`). Anything with the
constructor `(dataset, config, sx0, sy0, center, **extra)` and a `run(n)` returning
`IterationResult`s with `.sx`/`.sy`. Both `AlignmentEngine` and `OdstrcilEngine` present
this surface, so the comparison between them is genuinely like-for-like: same case, same
`AlignConfig`, same reconstruction backend, same rotation centre, same warm start.

```python
EngineAligner(name="mine", engine_module="my.module", engine_class="MyEngine")
```

**2. A bespoke adapter**, when the surface differs — `JointGdAligner` is one, because
joint-GD takes the raw stack, runs a multi-resolution *schedule* rather than N equal
outer iterations, and its answer must be read through `finalize()`.

**3. A module-level function** (`ModuleAligner`), for anything not yet integrated. The
preferred contract is:

```python
def align(projections, angles, *, center=None, iterations=10, pixel_size_nm=1.0, **kw):
    """-> object with .sy and .sx, px, in the apply_shifts sign convention."""
```

`ModuleAligner` probes a few plausible entry points and call signatures and records
which worked. An import failure yields `status="skipped"` with a message, never an
exception — a module that does not exist yet must not take the harness down.

Every adapter returns an `AlignerResult` with a `status`:

- `"ok"` — scored.
- `"skipped"` — not available (missing module or optional dependency). Not a failure of
  the method.
- `"error"` — it ran and blew up. That **is** a failure of the method, and the message
  is the evidence.

### The two reference rows, which are not optional

- **`null`** returns zeros. Its score *is* the injected misalignment. Any aligner that
  does not beat it is doing harm.
- **`oracle`** returns the ground truth. Its score must be ~1e-16 px. **Read this row
  first, always.** If it is not ~0, the *scorer* is broken — wrong gauge, wrong sign,
  wrong shape — and no other row means anything.

---

## Running it

```bash
# fully synthetic, everything
python -m benchmarks.runner --size 64 --slices 12 --angles 60 --iterations 12 --fsc

# one aligner, one perturbation, for a diagnosis
python -m benchmarks.runner --aligners null,odstrcil --phase-ramp 1.0 --phase-offset 1.0

# the geometry errors no rigid aligner can fix
python -m benchmarks.runner --out-of-plane-tilt 0.5 --magnification 0.02

# the full diagnostic sweep: one case per perturbation, that perturbation alone
python -m benchmarks.runner --catalogue --size 48 --slices 6 --angles 48
```

The projector and reconstruction backend are `NumpyProjectorBackend`: SIRT / FBP / BP on
`scipy.ndimage.rotate`, registered into TKtomo's backend registry as `benchmark-numpy`.
It exists so the benchmark runs anywhere, not because it is fast. It deliberately has
**no gridrec** — gridrec fails outright on a limited angular range, which is the regime
this harness exists to measure — and raises a message saying so rather than degrading.

### Environment caveats

- **tomopy is not required, and on this cluster is not installed.**
  `AlignmentEngine.step` calls `tomopy.prep.alignment.shift_images` and `blur_edges`;
  without them the incumbent cannot run at all and there is no baseline. `tomopy_shim()`
  installs a scipy re-implementation of exactly those two functions into `sys.modules`
  **for the duration of the run and only when tomopy is genuinely absent**, and removes
  them on exit so a stub can never leak into another test file's
  `pytest.importorskip("tomopy")`. Every report declares it in
  `environment.tomopy_shim`. The 5th-order spline is the same as tomopy's; results may
  differ at the sub-0.01 px level (skimage's `warp` clips to the input range, scipy's
  `shift` does not), which is far below anything measured here.
- **scikit-image** is needed only by the incumbent JIRR aligner
  (`phase_cross_correlation`). The Odstrcil and joint-GD paths are numpy+scipy only.
- **astra / GPU** is never required. `JointGdAligner` overrides joint-GD's own
  `projector="astra"` default with `"numpy"`.

---

## Measured baseline

Full tables, provenance and caveats in **`RESULTS.md`** beside this file. The headline,
on the fully synthetic case (60 projections, 40 x 92 px detector, 2.25 / 0.76 px RMS
injected jitter, target 0.333 px = 1/3 voxel):

| aligner | rms dy | rms dx | wall clock |
| --- | ---: | ---: | ---: |
| `null` (do nothing) | 2.244 | 0.752 | — |
| **`jirr` (incumbent baseline)** | **0.013** | **0.031** | 2.5 s |
| `odstrcil` | 0.006 | 0.033 | 3.0 s |
| `joint_gd` | 0.009 | 0.037 | 202 s |

All three clear the target by more than an order of magnitude on the easy case; the
diagnostic sweep in `RESULTS.md` §3 is where they separate. Two results worth carrying
forward:

- With a per-projection **phase ramp** injected, horizontal recovery goes `jirr`
  0.222 px -> `odstrcil` 0.124 px. That is the gradient trick doing exactly what the
  roadmap says it does, on the one perturbation every real ptycho dataset has.
- **Vertical**: `odstrcil` beats both others by 2-3x in almost every case, from a
  row-sum with no reconstruction in the loop.

And the negative result, which is the reason this harness scores shifts and not FSC:
**the split-half FSC read 11.97 px for the perfectly aligned reconstruction, 11.97 px
for the unaligned one, and 11.98 px for one misaligned by 4.5 px RMS.** Same certificate,
340x different error. Numbers to quote are the gauge-removed `rms dy` / `rms dx` in px.
