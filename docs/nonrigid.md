# Non-rigid alignment — when to use it, and (mostly) when not to

Rigid alignment assumes one global transform per projection. When the sample deforms
*during* the scan — radiation damage, thermal drift, a 52-hour acquisition — no rigid
alignment satisfies all the projections at once. The residual plateaus above zero and
stays there, and more iterations of anything rigid will not touch it.

The method implemented here replaces straight projection lines with paths curved by a
**deformation vector field** (DVF), estimated by 3D optical flow between partial
reconstructions taken at different times. It follows Odstrčil *et al.*, *Ab initio
nonrigid X-ray nanotomography*, **Nat. Commun. 10**, 4778 (2019), who reported a beetle
sample going from 53 nm to 27 nm — a factor of two rigid alignment could not reach — with
recovered DVFs matching a simulated ground truth to about 0.8 px rms.

It is also, by a wide margin, the most dangerous stage in the pipeline. A DVF has of
order a thousand times more parameters than a per-projection rigid shift. It will absorb
an unfixed phase ramp, a wrong rotation centre, an unconverged rigid alignment, or plain
detector noise, and hand back a **sharp, plausible, wrong** volume that nothing
downstream can detect. Everything below is organised around that fact.

| file | what it is |
| --- | --- |
| `tktomo/ptycho_align/core/deformation.py` | the DVF: `DeformationField`, `DeformationSequence`, `warp_volume`, `estimate_flow`, `compose`, `invert` |
| `tktomo/ptycho_align/core/nonrigid.py` | `NonRigidAligner` — one outer iteration per `step()`, the same contract as `AlignmentEngine` |
| `tktomo/ptycho_align/core/nonrigid_gate.py` | **the decision gate**: should this dataset get a non-rigid model at all? |
| `benchmarks/scenario_nonrigid.py` | the benchmark: the rigid floor, the non-rigid answer, and the negative control |
| `tests/test_deformation.py`, `test_nonrigid.py`, `test_nonrigid_gate.py` | 104 tests, numpy + scipy only, no GPU and no beamtime data |

---

## 1. Non-rigid is the LAST stage. It is never a substitute for anything earlier.

The roadmap's order of operations, which is not negotiable:

**ramp / offset removal → rotation centre → vertical → horizontal → geometry
refinement → non-rigid.**

Each stage's fix invalidates the measurements of every stage after it. A residual phase
ramp is *mathematically indistinguishable from a lateral shift*, so it does not merely
add artifacts — it poisons the alignment that is supposed to remove them, and a
deformation field fitted on top of it will happily model the ramp as sample motion. A
wrong rotation centre inflates every reprojection residual, and a deformation field will
absorb that too.

So the aligner does not take your word for it:

- `NonRigidAligner` **refuses to start** on data that does not look rigidly aligned. The
  check is on the *data* — a centre-of-mass consistency test, constant vertical centroid
  and sinusoidal horizontal (`rigid_alignment_is_plausible`) — not a promise from the
  caller, because the caller is exactly who gets this wrong. It raises
  `RigidAlignmentRequired`.
- It **refuses a missing wedge** in any acquisition-time subset. A partial reconstruction
  of an angular wedge is elongated along the missing direction, and optical flow will
  measure that elongation and return a smooth, entirely plausible-looking field. This is
  the single most dangerous failure mode of the method.
- The decision to go non-rigid at all is **evidential**, and that is the next section.

If you are reading this because a reconstruction looks bad and you are hoping non-rigid
will fix it: run `tktomo.diagnostics.triage(projections, theta)` first. It stops at the
first thing that fires, in the order above, and non-rigid is last for a reason.

---

## 2. The decision gate

```python
from tktomo.ptycho_align.core.nonrigid_gate import gate_from_engine, format_gate

verdict = gate_from_engine(engine, acquisition_index=acq)   # after the rigid stage
print(format_gate(verdict))

if verdict:                       # True only for RUN_NONRIGID
    ...
```

`evaluate_gate(...)` is the same thing without an engine: pass `residual_history`,
`shift_history`, and the final `measured` / `simulated` stacks. The verdict is a frozen
dataclass, `to_json()`-able, and it records the numbers **and the alternatives that were
excluded**, so a decision to run a method with this much freedom leaves a record that can
be checked later by someone who was not in the room.

### The five outcomes

| recommendation | means |
| --- | --- |
| `FIX_UPSTREAM` | the leftover residual is explained by a per-projection **phase ramp** or a **rotation-centre** error. Non-rigid would pave over it permanently. |
| `MORE_RIGID_ITERATIONS` | the residual has not plateaued, or it has but the leftover is a per-angle shift that is **smooth in acquisition time** — drift the rigid stage has not finished. |
| `ACCEPT_RIGID` | plateaued, nothing upstream, and what is left is **spread and angle-random**. That is jitter and noise. Fitting a DVF to it fabricates structure. |
| `RUN_NONRIGID` | plateaued, nothing upstream, and the leftover is beyond any rigid model, **localised**, **angle-consistent** *and* **changing over acquisition time**. All four. |
| `INSUFFICIENT_EVIDENCE` | there is no residual history, no residual maps, or no `acquisition_index` (without it the residual cannot be asked whether it *changes* — see (d)). `bool(verdict)` is `False`, because the default for a model that can invent structure has to be the conservative one. |

### The four criteria, and why they are not the obvious ones

**(a) The plateau is measured, not asserted.** "The residual stopped falling" is a claim
about a trend against the scatter of that trend, so both are estimated: a log-linear
decay rate over the trailing window and the RMS scatter about it. A curve improving by
0.5 % per iteration is *under* any sensible tolerance and is still convergence — four
more iterations buy 2 %, and rigid iterations are cheap. A curve improving by 0.02 % per
iteration with a perfectly clean trend is a plateau, however clean the trend is. The gate
separates the two on the **projected gain from continuing**
(`GateConfig.min_projected_gain`, default 0.5 %), not on the slope alone. A *rising*
residual is reported as divergence, not as a plateau — testing only
`improvement < tolerance` waves a diverging alignment straight into the non-rigid stage.

**(b) The statistic is computed on the residual no rigid model can reach.** The raw
reprojection residual of a real dataset is dominated by things rigid alignment handles
perfectly: on the benchmark phantom **89 % of its energy is per-angle offset and gain** — a
consequence of reconstructing inconsistent data, present with no normalisation error
anywhere in it. Measure concentration on that and you measure the offset.
`rigid_reduced_residual()` fits out the best per-angle offset, gain and rigid shift first;
what remains is by construction *what no rigid alignment can fix*, which is exactly what
the roadmap's criterion is about. A dataset where almost nothing survives that reduction
gets `ACCEPT_RIGID` on those grounds alone (`min_residual_beyond_rigid`, default 5 %),
before localisation is consulted.

**(c) Localisation is measured against a null, not against a threshold.** The natural
statistic — the share of residual energy in the hottest 10 % of blocks — has no fixed
meaning. Two measurements from `tests/test_nonrigid_gate.py`:

| residual | raw concentration | permutation null | z | verdict |
| --- | --- | --- | --- | --- |
| six hot pixels per frame at random places | **0.77** | 0.77 | −0.4 | not localised ✓ |
| a genuine localised blob, same place every angle | **0.29** | 0.14 | +115 | localised ✓ |

Any fixed threshold fires on the first and misses the second. The null is realised
explicitly: the residual pixels are permuted within each frame, which destroys spatial
organisation while preserving the amplitude distribution — including its tails — exactly.
The observed concentration is then reported as a z-score and an empirical p-value against
that null.

Two more things the gate does that a naive version does not:

- **The concentration is measured inside the object's own footprint.** A frame that is
  mostly vacuum makes *every* residual look localised: all the energy sits on the object,
  which is a tenth of the frame. Same residual, measured both ways: z = 158 without a
  support mask, z = 0.5 with one. The support is derived from the *measured* projections,
  never from the residual, which would be circular.
- **The gauge is removed before any leftover shift is called an error.** A leftover
  per-angle horizontal shift of the form `X cos θ + Y sin θ` is a translation of the
  reconstructed object: unobservable and harmless. A leftover *constant* horizontal shift
  is exactly the rotation-axis position. The gate reports them separately, with a standard
  error on the constant — the mean of a jittery leftover shift is itself a random number
  of order `du_rms / √n_angles`, so on a short scan a "centre error" of a quarter pixel can
  be nothing but the noise in one's own estimate of it.

**(d) The residual must CHANGE during the scan.** This criterion is not in the roadmap and
it is not optional: the benchmark forced it into existence. A residual that is localised
*and* angle-consistent passes both of the roadmap's tests and can still have nothing to do
with the sample moving — an FBP streak, an edge the band-limited projector cannot
reproduce, an unmodelled static feature, all sit in the same place at every angle for the
whole scan. Measured on the benchmark phantom with **no deformation whatsoever**:

| | undeformed phantom | deformed phantom |
| --- | --- | --- |
| localisation z | 108 | 110 |
| angle-consistency z | **305** | 162 |
| **temporal-change z** | **−1.6** | **+7.8** |
| gate verdict | ACCEPT_RIGID | RUN_NONRIGID |

Without row 3 the gate said "go non-rigid" to the undeformed phantom, confidently, with
better numbers than the real case. Deformation is *by definition* something that happens
during the scan, so `measure_temporal_change` groups the projections into acquisition-time
blocks, forms each block's mean residual-energy map, and asks whether those maps
decorrelate more than randomly grouped projections of the same sizes do — a permutation
null on the acquisition labels, which preserves the noise exactly and destroys only the
time ordering.

This is why **`acquisition_index` is required for a `yes`**. Without it the verdict is
`INSUFFICIENT_EVIDENCE`, not `RUN_NONRIGID` — the same requirement `NonRigidAligner` makes,
for the same reason.

### Distinguishing the confusable alternatives

The residual is regressed, per angle, onto a small physical basis:

```
R_i  ~  c_i + g_i·S_i                 offset and gain    -> NUISANCES, fitted and removed
     +  e_i·u + f_i·v                 a phase ramp       -> stage 0
     +  a_i·∂S_i/∂u + b_i·∂S_i/∂v     a leftover shift   -> stages 2-4
```

Each group is credited with its **unique** contribution — what the full model explains
minus what it still explains without that group — and the part that several groups explain
equally well is credited to none of them and reported as `shared_fraction`. Two findings
made this necessary, and both are in the tests:

- Under the obvious *sequential* attribution in roadmap order, a **pure 0.8 px shift** was
  credited with 16 % "ramp" — enough to fire the ramp test and send someone off to remove
  a phase ramp that was never in the data. Its unique ramp contribution is ~0.
- The **per-angle offset and gain are nuisances, never diagnosed.** A deforming sample
  makes the reconstruction inconsistent; reprojecting that inconsistent volume mismatches
  the measurement in overall scale and level at every angle — 15 % of the residual energy
  on the benchmark phantom, with no normalisation error anywhere in the data. Diagnosing
  it would send every genuine deformation off to fix an imaginary flat-field. They are also
  the harmless part: a constant and a global scale move nothing, so neither can be confused
  with a misalignment. (`tktomo.diagnostics` treats them the same way.)

**Jitter vs deformation**, the confusion that costs a paper: jitter is a per-angle rigid
shift that is *angle-random*. The gate calls it jitter when the shift group explains more
of the residual than anything outside the rigid basis does **and** the leftover shift has
low lag-1 autocorrelation **in acquisition order**. Deformation is the opposite: it is not
shift-like at all, and it is smooth in time. Acquisition order matters and is not angle
order for an interlaced scan or a repeated series — reading a drift in angle order makes it
look like jitter and gets the opposite verdict, which is why nothing here ever guesses the
ordering from the angles.

### `tktomo.diagnostics` integration

When the package is importable, `verdict.to_finding()` returns a
`Finding(mode=FailureMode.DEFORMATION, ...)` and `verdict.to_probe_result()` a
`ProbeResult(probe="nonrigid_gate", stage=TriageStage.NON_RIGID, ...)`, so a gate verdict
drops straight into a diagnostics report. Both return `None` when the package is absent
and everything else works unchanged. A finding is emitted **only** for `RUN_NONRIGID`: the
other recommendations are statements about this stage's evidence, not claims about the
earlier failure modes, which have their own probes.

`verdict.as_rigid_evidence(residuals, shifts)` converts to the `RigidEvidence` type
`NonRigidAligner` uses as its own precondition, so the gate and the aligner cannot
disagree about a dataset — a pinned test.

---

## 3. What the gate cannot do

**It vetoes a purely global deformation, by design — and this is measured, not
hypothetical.** Uniform swelling or bulk thermal expansion is genuinely non-rigid and
produces a residual spread over the whole frame, which this scores as "not localised". At
`--local-fraction 0` the benchmark injects exactly that: the aligner gains **+7.8 % on
held-out projections**, so the deformation is real and worth modelling, and the gate says
**no**. That is the roadmap's criterion implemented faithfully, not a bug in the code, and
it is the gate's known false-negative mode. If you have independent reason to expect a
global deformation — a long scan of a beam-sensitive sample — the localisation test is the
wrong instrument and its "no" is not evidence of rigidity.

**A `yes` does not mean the deformation is worth modelling.** At `--local-fraction 0.7` the
gate correctly detects a real deformation that is nonetheless too weak to help: the
aligner's held-out monitor declines it and `run()` stops after one iteration. The gate
answers *"is there evidence of deformation?"*; the held-out residual answers *"did
modelling it help?"*. Two questions, two instruments, and the second one is the verdict.

**It cannot prove there is no deformation.** `ACCEPT_RIGID` is a statement about the
evidence.

**It is a gate, not a validator.** A `yes` licenses running the method, not believing its
output. See §5.

---

## 4. The overfitting guards, and why each exists

In the order they bind:

1. **A coarse DVF grid** (`grid_spacing`, in **voxels**). The strongest of the three
   regularisers and a hard restriction of the model space: `3·(N/spacing)³` parameters per
   subset against `N³` voxels. The aligner logs the parameter-to-voxel ratio and warns
   above 1 %. The default of 16 voxels is for the few-hundred-voxel volumes this gets
   prototyped on; on a 1488³ tomogram it would be 2.5 M parameters per subset. **Aim for
   8–20 nodes per axis** — on lens-1 that is a spacing of 75–190 voxels.
2. **Temporal smoothing and interpolation** (`time_sigma`). K fields describe N
   projections, interpolated linearly in acquisition time and never extrapolated. A
   deformation that is not smooth in time cannot be represented at all. This is the
   regulariser that makes the problem tractable.
3. **A magnitude cap** (`max_dvf_px`). Optical flow that has locked onto a streak returns
   tens of voxels, and a warp that large makes anything match anything.
4. **A held-out set** — 15 % of projections excluded from every partial reconstruction and
   from the reference volume, so they are *predicted*, never fitted. The fitted residual
   always improves; more parameters always fit better; it is not evidence. Three rules,
   and `run()` stops when any fails: the fitted residual must improve at all; the held-out
   gain must keep improving iteration to iteration; the held-out residual must not worsen
   nor lag the fitted gain.

> **`run()`, not a loop of `step()`.** The refinement composes an increment onto the field
> each iteration, which is what recovers amplitude — and on undeformed data the field
> grows too (0.18 px rms at iteration 1 to 0.60 px at iteration 6 in an early run without
> the stall guard). `run()` is where the stopping rule lives. A caller who loops `step()`
> and ignores `result.overfitting` will get a growing fiction.

---

## 5. Verify by split-data FSC — and never by FSC alone

**FSC is exactly blind to common-mode geometric error.** A translation `d` applied to both
half-sets multiplies their transforms by `exp(∓2πik·d)`; the two cancel in the cross term
and every shell's correlation is bit-identical. The same argument covers a wrong rotation
centre, a wrong tilt, a wrong magnification. This repo measured it directly: the half-bit
FRC read 508.6 nm at centring errors of 0, 4, 8, 16, 32 **and** 64 px while the true edge
blur grew to 128 px.

So: split-data FSC **paired with a reprojection-residual map**, which does see common-mode
error, plus the held-out residual from the aligner itself. Never eyeball slices — a
deformation field can make any slice look sharp, which is the whole problem.

Two details that matter for a moving sample:

- Split **even/odd in angle within each acquisition-time block**, not first-half /
  second-half of the scan. A time split gives two limited-angle reconstructions whose FSC
  measures the missing wedge.
- Carry each block's volume back into the common frame through the **inverse** of its own
  field before averaging (`warp_volume(partial_k, invert(u_k)) ≈ reference`). Without that,
  a split-half FSC of a moving sample measures how much the sample moved between blocks —
  which is precisely the quantity non-rigid alignment is supposed to remove, so the rigid
  and non-rigid rows would not be the same metric. `benchmarks/scenario_nonrigid.half_volumes`
  does this and is the reference implementation.

---

## 6. Measured numbers

### From the core validation (SLURM 24048281)

Synthetic phantom, 32 × 64 × 64 voxels, six sub-tomograms × 60 angles = 360 projections,
coarse grid (3, 5, 5) = 225 parameters per subset, six iterations, library defaults.

| | |
| --- | --- |
| recovered DVF vs known truth, **inside the object support** | **0.748 px rms** |
| ...over the whole coarse grid (includes air, where flow is unconstrained) | 1.460 px rms |
| truth DVF, gauge-fixed | 1.777 px rms |
| correlation with truth | **0.97** |
| amplitude relative to truth | **0.59** |
| reprojection residual, fitted | 0.1718 → 0.1393 (**+18.9 %**) |
| reprojection residual, **held out** | 0.1750 → 0.1424 (**+18.6 %**) |
| Odstrčil *et al.* reported DVF accuracy | ≈ 0.8 px rms |

The fitted and held-out gains **move together**. That is what makes the deformation
credible rather than fitted, and it is the only number of the set that could not have been
manufactured.

**Negative control** — identical pipeline, same phantom, *no* deformation: `run()` stops at
iteration 1 with the guard firing ("does not improve even the projections it was fitted
to, +0.0 %"), having invented **0.125 px rms / 0.329 px max**. Held-out gain +0.13 %,
against +18.6 % for the real case.

**A deliberately under-regularised configuration on rigid data** invents 1.35 px rms /
2.82 px max — and the held-out guard catches it (fitted residual −5.1 %). That contrast is
the argument that the guard, and not luck, is what keeps the method honest.

### The `flow_alpha` trade

Recovered amplitude and invented-from-nothing move together, and this is not tunable away:

| `flow_alpha` | error vs truth | amplitude | correlation | invented on rigid data |
| --- | --- | --- | --- | --- |
| 0.3 | 0.62 px | 0.64 | 0.68 | 0.40 px rms |
| 0.5 | 0.51 px | 0.63 | 0.78 | 0.26 px rms |
| 1.0 | 0.46 px | 0.59 | 0.84 | 0.12 px rms |
| **2.0 (default)** | **0.51 px** | **0.50** | **0.78** | **0.05 px rms** |
| 4.0 | 0.60 px | 0.37 | 0.69 | 0.03 px rms |
| 8.0 | 0.67 px | 0.25 | 0.58 | 0.01 px rms |

The default sits where the method still finds most of the deformation while inventing
under a tenth of a pixel from nothing.

> **Amplitude bias, by design.** The recovered DVF has correlation 0.97 with the truth and
> only 0.59 of its magnitude. Read a recovered DVF as *where, when and in which direction
> the sample moved*. **Do not quote it as a calibrated strain field.**

### From the benchmark scenario (SLURM 24050011)

`python -m benchmarks.scenario_nonrigid`, headline configuration: 32 × 64 × 64 voxels,
6 sub-tomograms × 60 angles, deformation 2 px peak with `local_fraction 0.5`, 2 % noise,
grid spacing 16, 6 rigid + 5 non-rigid iterations. **163 s wall clock, single core.**

| row | residual | held out | DVF err (px) | corr | FSC (px) | NRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| no alignment | 0.1293 | | | | | 0.194 |
| **rigid, to plateau** | **0.1273** | | | | 2.56 | 0.198 |
| **non-rigid** | **0.1250** | 0.1289 | **0.332** | **0.75** | | 0.203 |
| CONTROL no alignment | 0.1246 | | | | | 0.183 |
| **CONTROL rigid** | **0.1245** | | | | | 0.184 |
| **CONTROL non-rigid** | 0.1235 | 0.1276 | *0.155 invented* | −0.19 | | 0.187 |

1. **RIGID FLOOR.** The rigid loop plateaus at 0.1273 on deformed data and 0.1245 on the
   identical undeformed phantom. The gap, **+2.2 % of the floor**, is what no rigid model
   can remove.
2. **NON-RIGID** closes **80 %** of that gap. Against the aligner's own baseline: **+0.8 %
   fitted and +0.8 % held out** — moving together, which is the whole argument. DVF
   recovered to **0.332 px rms** in support, correlation 0.75, amplitude 0.59.
3. **NEGATIVE CONTROL [PASS].** With no deformation, `run()` **stopped itself at iteration
   1** — "the deformation model does not improve even the projections it was fitted to
   (−0.2 %, needs +0.2 %)" — having invented 0.155 px rms. Held-out gain −0.1 % against
   +0.8 % for the real case.

**The gate across a global-to-local sweep** (48³, 5 × 40 angles), with the aligner's
held-out gain beside it as the ground truth for whether the gate was right:

| `local_fraction` | temporal z | gate verdict | aligner held-out gain | gate was |
| --- | --- | --- | --- | --- |
| 0.00 (pure global) | 2.9 | ACCEPT_RIGID | **+7.8 %** | **wrong — false negative** |
| 0.35 | 3.7 | RUN_NONRIGID | +3.3 % | right |
| 0.50 (headline) | 7.8 | RUN_NONRIGID | +0.8 % | right |
| 0.70 | 4.4 | RUN_NONRIGID | **−0.5 %** | **wrong — false positive** |
| 1.00 (pure local) | −0.9 | ACCEPT_RIGID | −1.4 % | right |
| jitter 1.5 px, no deformation | — | MORE_RIGID_ITERATIONS | −0.7 % | right |
| every control arm | ≤ 2.9 | ACCEPT_RIGID | ≤ −0.1 % | right |

Four right, one wrong each way, and both errors are worth more than the successes:

- **`local_fraction 0` is the documented blind spot, measured.** A purely global drift and
  shear is genuinely non-rigid, the aligner gains 7.8 % held out on it — and the gate says
  no, missing the temporal threshold by 0.1 (z = 2.9 against 3.0). This is the roadmap's
  localisation criterion doing what it says, and it is a real cost. The thresholds were
  **not** tuned to make this case pass; fitting a gate to one synthetic phantom is exactly
  how a gate stops meaning anything.
- **`local_fraction 0.7` is why the gate is a screen and not a verdict.** The deformation is
  real and the gate correctly detects it, but it is too weak to help: the held-out monitor
  inside the aligner declines it and `run()` stops. The gate answers "is there evidence for
  deformation?"; only the held-out residual answers "did modelling it help?". Both are
  needed and they are not the same question.

---

## 7. Running it

```bash
# the benchmark, small enough for a laptop (about 5 s)
python -m benchmarks.scenario_nonrigid --size 32 --slices 16 --subtomos 4 --angles 24 \
    --rigid-iterations 4 --nonrigid-iterations 3 --grid-spacing 8

# the headline numbers in section 6 (163 s, one core)
python -m benchmarks.scenario_nonrigid --size 64 --slices 32 --subtomos 6 --angles 60 \
    --local-fraction 0.5 --rigid-iterations 6 --nonrigid-iterations 5 --grid-spacing 16

# the gate's blind spot, made visible: a purely GLOBAL deformation
python -m benchmarks.scenario_nonrigid --local-fraction 0.0

# jitter instead of deformation: the gate must not say yes
python -m benchmarks.scenario_nonrigid --deformation 0 --jitter 1.5
```

Use an interpreter with numpy < 2 — see the environment note at the end. On Maxwell,
submit it rather than running it on the login node.

The scenario runs the whole chain — build the scan, rigid loop to its plateau, the gate,
then the non-rigid stage — on the deformed data **and** on an identical undeformed
phantom, and prints three verdict lines. The third is the negative control, and it is the
row that makes the other two mean anything.

`--out PATH` writes the report as JSON. There is no default and it must be outside the
repository: results belong beside the data, not in the source tree.

### On real data

```python
from tktomo.ptycho_align.core.nonrigid import NonRigidAligner, NonRigidConfig
from tktomo.ptycho_align.core.nonrigid_gate import gate_from_engine, format_gate

verdict = gate_from_engine(engine, acquisition_index=acq)
print(format_gate(verdict))
if not verdict:
    raise SystemExit(verdict.headline)          # do the thing it told you to do

aligner = NonRigidAligner.from_engine(
    engine, acquisition_index=acq,
    config=NonRigidConfig.from_align_config(engine.config, grid_spacing=100.0),
)
results = aligner.run(6)                        # run(), not a loop of step()
if results[-1].overfitting:
    aligner.revert_to(len(results) - 1)
```

`acquisition_index` is **required** and has no default: it is the order in which the
projections were *acquired*, which is not the order of `angles` unless the scan was a
single monotonic sweep. Pass the scan number, the file index, or the timestamp.

### The subset trade, which the method lives or dies by

`subset_mode="time_blocks"` (default) makes each subset one contiguous block of
acquisition time, so each subset is **one deformation state** — which is what makes a DVF
between it and the reference mean anything. It is correct only if each block is *angularly
complete*: an interlaced or golden-ratio scheme, or a series that repeats whole tomograms.
That is Odstrčil's geometry and it is the geometry of a P06 series swept through 0–180
several times over many hours.

`subset_mode="interleaved"` takes every K-th projection in time order. Every subset then
spans the full angular range *and* the full time range, so every partial shows the same
time-averaged sample and recovers almost nothing. It exists as an escape hatch and is
measured to be nearly useless.

**A single sequential 0–180 sweep is not supported, deliberately.** Time blocks are then
angular wedges, the aligner raises `RigidAlignmentRequired` rather than producing a
plausible field from missing-wedge elongation, and the honest answer is that a time-varying
deformation is simply **not identifiable** from that scan. Say so in the paper instead of
running this.

---

## 8. Applying this to lens-1 — what has to be true first

Nothing here has been run on beamtime data; every number above is from synthetic phantoms.
Before `lens1_v4_best.h5` (907 projections, 74.51 nm voxel, 158.3 nm volume FSC, 52 hours
of scanning) goes anywhere near this stage:

1. **The rigid residual must have plateaued.** Run the gate. If it says
   `MORE_RIGID_ITERATIONS` or `FIX_UPSTREAM`, do that instead.
2. **The acquisition-time ordering of the 907 projections must be available** — scan
   number or timestamp per projection. Without it there is no non-rigid stage. The angles
   will not do.
3. **Each acquisition-time block must be angularly complete.** If the series is
   sub-tomogram-per-sweep, this holds; if it is one long sweep, it does not, and the
   aligner will refuse.
4. **Bin first.** The aligner holds K partial volumes plus the reference simultaneously; at
   1488 × 1816 × 1816 float32 that is ~19.6 GB *per volume*, so a six-subset run is not
   feasible at full resolution. It logs a warning above 8 GB. The deformation is smooth by
   construction and does not need full resolution to estimate — estimate it binned, apply it
   at full resolution.
5. **Set `grid_spacing` in voxels for the binned volume**, aiming at 8–20 nodes per axis.
   The default of 16 is far too fine for anything of this size.

52 hours of total scanning makes sample deformation a live hypothesis here, not a corner
case. That is a reason to *measure* it with the gate, not a reason to assume it.

---

## Known limitations, collected

- **Recovered amplitude is ~0.6 of truth**, by design (§6). Not a calibrated strain field.
- **The localisation criterion vetoes global deformation** (§3): measured false negative at
  `--local-fraction 0`, where the aligner gains 7.8 % held out and the gate says no (temporal
  z = 2.9 against a threshold of 3.0). The thresholds were deliberately not tuned to fix it.
- **The angle-consistency measure compares residual maps across the whole scan**, so a
  deformation that *evolves* in time scores lower consistency for a physically correct
  reason. Compute it within one acquisition-time block to see the effect cleanly.
- **The temporal test needs the projections spread over enough acquisition-time blocks**
  (`GateConfig.n_time_blocks`, default 4, each needing at least two projections) and it is
  the criterion most sensitive to a mislabelled acquisition order. A wrong ordering makes a
  real deformation look static and produces a false `ACCEPT_RIGID`.
- **`NonRigidAligner.from_engine()` has not been exercised end to end** — it calls
  `engine.aligned_projections()`, which needs tomopy, installed in none of the available
  interpreters. `RigidEvidence.from_engine` is tested; the rest is code-reviewed only. It
  is the path most likely to have a signature mismatch when someone runs it with tomopy
  present.
- **The benchmark is a partial inverse crime**: the projector that makes the data
  reconstructs it. Mitigated (truth at interpolation order 3, reconstruction at order 1),
  not removed. Read the *relative* numbers.
- **The benchmark's rigid loop is a compact re-implementation**, not `AlignmentEngine`, so
  the scenario runs without tomopy. It establishes the rigid floor, which is a property of
  the data; it is not a benchmark *of* the engine.
- **The new modules are not re-exported** from `tktomo/ptycho_align/core/__init__.py` —
  import them by full path.

## Environment note (not about this stage, but it invalidates results)

`python` — Python 3.14.0 with NumPy 2.2.6, and the
**default interpreter in `tomo/job_cpu_template.sh`** — silently corrupts arrays. For
arrays above NumPy's elision threshold, inside a function, an expression like `a - b` where
`a` is a local writes the result *into* `a`:

```python
def f():
    a = np.ones((1000, 1000), np.float32)
    b = np.full_like(a, 2.0)
    d = a - b
    return a.max()      # returns -1.0; must be 1.0
```

CPython 3.14's borrowed stack references leave locals at refcount 1, which defeats NumPy's
temporary-elision heuristic. Correct at module scope, correct for small arrays, correct on
numpy 1.26.4. `benchmarks/metrics.py` already carries a workaround for a symptom of this
without naming the cause. Every number in this document was produced with
`ptypy_v8` (3.12 / numpy 1.26.4) or `tomo_2026` (3.11 / numpy 1.26.4). **Any result
produced through that template without `USE_TOMO_ENV=1` is suspect.**

---

## References

> Odstrčil, M. *et al.* Ab initio nonrigid X-ray nanotomography. **Nat. Commun. 10**, 4778
> (2019). — the method.
>
> Gürsoy, D. *et al.* Rapid alignment of nanotomography data using joint iterative
> reconstruction and reprojection. **Sci. Rep. 7**, 11818 (2017). — the rigid stage this
> one follows, implemented in `core/engine.py`.
