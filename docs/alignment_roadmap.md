# Alignment for phase-contrast (ptychographic) tomography

The technical rationale for the `feature/phasetomo-alignment` branch: what the problem
is, which methods exist and what each is for, the order they must be applied in, the
artifact-to-cause table, and how an alignment is validated. Condensed to be citable and
to be read once before touching the code.

Companion documents: `IMPLEMENTATION_NOTES.md` (checklist + decision log — what is
actually built and what was measured), `docs/benchmark_results.md` (the three-way
comparison), `docs/diagnostics.md`, `docs/joint_gd.md`, `docs/nonrigid.md`,
`benchmarks/README.md`, and `skills/phasetomo-alignment/SKILL.md` (the operational
short form).

---

## 1. Why this is not ordinary tomographic alignment

**1.1 The data carries a gauge freedom that looks exactly like the answer.**
Ptychographic phase retrieval determines each projection's phase only up to a **constant
offset** and a **linear ramp** — different projections of the same scan come back with
different, unknowable values of both [7, 8]. A linear phase ramp across a projection is
*mathematically identical to a lateral translation of it*. So a registration performed on
the phase cannot distinguish "the sample moved" from "the phase retrieval picked a
different gauge", and will spend iterations correcting a shift that does not exist. Two
independent defences exist, and they attack it from opposite ends:

* **remove it** — fit and subtract offset + plane on a presumed-vacuum border
  (`preprocess.remove_phase_ramp`). Exact for a *planar* background, and — a result worth
  knowing — exact **whatever the mask contains**, because a plane fitted by least squares
  recovers a plane exactly and a contaminated border only adds a fixed object-dependent
  plane. Measured: after this step, projections injected with 0.5 rad and 2.0 rad of ramp
  differ by 1.9 × 10⁻⁶ rad on a 15 rad scale. It does *not* handle a non-planar (curved,
  higher-order) background.
* **be invariant to it** — register on the phase **gradient**. Differentiating sends the
  constant to exactly zero and the ramp to a constant that mean-subtraction removes:
  `∂/∂x [φ + ax + by + c] = ∂φ/∂x + a`. Measured pairwise invariance: 4 × 10⁻¹⁶ px across
  ramps of 0–100 rad, against −25.7 px for the value domain at 10 rad.
  **What the gradient does *not* buy** is an automatic end-to-end improvement inside a
  reprojection loop — see §5.3 and `docs/benchmark_results.md` §3.

**1.2 The two transverse directions are different problems.**
The rotation axis is vertical, so a rotation maps every voxel to another voxel *in the
same detector row*. The vertical mass distribution `m(v) = Σ_u p(v, u)` is therefore
**invariant under the rotation angle**, and "find the vertical shift" is an ordinary,
well-conditioned 1-D registration with a unique answer — no forward projection, no
volume, milliseconds [3].

Nothing of the sort holds horizontally. The horizontal mass distribution is *supposed* to
change with angle — that variation is the tomographic signal — so no pairwise comparison
of two projections can separate "the sample moved sideways" from "the sample turned".
The horizontal shift is only defined against the object itself, which is why it costs a
reconstruction per iteration and vertical does not.

**Consequence, and it is a hard one: centre of mass is structurally wrong for the
horizontal direction.** It is defensible **only vertically**, and only when the sample is
entirely inside the field of view (§4, mode 12). Horizontally its value is the *fit*: for
a mass-conserving projection the column centroid traces `com_u(θ) = a sin θ + b cos θ + c`,
and the offset `c` **is** the rotation-axis position. Use it for the centre and as a warm
start; never as the alignment.

**1.3 "Easy" is a statement about well-posedness, not magnitude.** On a real
ptycho-tomography scan the vertical shifts were 3.3× *larger* than the horizontal ones,
and the projection-space FRC was 1.8× *worse* vertically (359 nm) than horizontally
(199 nm). Vertical being the tractable direction does not make it the small one.

---

## 2. The order of operations

**ramp / offset removal → rotation centre → vertical → horizontal → geometry refinement
→ residual check → non-rigid.**

This is not a priority list. **Each stage's fix invalidates the measurements of every
later stage**, so applying them out of order does not merely waste time — it produces
confidently wrong numbers:

| step | why nothing after it is valid until it is done |
|---|---|
| **[0] ramp / offset** | a residual ramp is indistinguishable from a lateral shift and moves *every* centroid, every profile and every registration. It does not add artifacts, it poisons the alignment meant to remove them. Truncation (mode 12) must be checked **before** the ramp fit, because a truncated frame has no vacuum border to fit on |
| **[1] rotation centre** | a wrong centre inflates every reprojection residual, and it is *one number*: letting a per-view loop absorb it spends N free parameters on a single degree of freedom and invites drift |
| **[2] vertical** | decoupled and essentially free; solving it removes a whole axis of variation from the expensive stage that follows |
| **[3] horizontal** | requires a volume, so it requires everything above to be right first |
| **[4] geometry refinement** | axis tilt, out-of-plane tilt, angle-readback and magnification errors are **not** per-view translations. A shift loop asked to absorb them converges to a wrong answer that looks converged |
| **[5] residual check** | the only honest stopping criterion, and the evidence for whether [6] is warranted |
| **[6] non-rigid** | ~10³× more free parameters than a rigid shift. It will absorb any unfixed error above and return a sharp, plausible, wrong volume that nothing downstream can detect |

`tktomo.diagnostics.triage()` enforces this order and stops at the first firing;
`tktomo.ptycho_align.core.nonrigid_gate` enforces the entry condition for [6].

---

## 3. The artifact-to-cause table

Twelve stereotyped failures. `tktomo.diagnostics` is this table as executable code: one
`ArtifactSpec` record and one probe per row, with `diagnose()` (survey, ranked) and
`triage()` (stops at the first firing). `format_catalogue()` prints the full text; the
table below is the condensed form.

| # | failure (stage) | in a slice | in a sinogram | how to confirm | fix |
|---|---|---|---|---|---|
| 1 | **Wrong rotation centre** (centre) | tuning-fork doubled edges, identical in every slice | all sinusoids offset by the same constant in `u` | entropy sweep over the assumed centre; independently, register θ against the mirrored θ+180 partner | set the centre — do **not** let the loop absorb it into per-view shifts |
| 2 | **Random per-projection jitter** (horizontal) | uniform blur, *not* doubled edges | high-frequency scatter about the ideal sinusoid, white in acquisition order | residual of the `a sinθ + b cosθ + c` fit has lag-1 autocorrelation ≈ 0 | iterative reprojection alignment — jitter is exactly what the loop is for |
| 3 | **Vertical drift** (vertical) | slice-to-slice smearing along z | smooth low-order drift in a vertical sinogram | register the row-summed 1-D profiles; split the shift series into trend + white residual | 1-D vertical mass alignment [3] — no projector needed |
| 4 | **Tilt-axis ANGLE error** (centre) | arcs curving in *opposite* directions above and below mid-plane; mid-plane looks fine | sinusoid *offset* is linear in detector row: `c(z) = c₀ + z tanα` | three-slice arc test: regress `c` on `z`; non-zero slope, zero mean offset | rotate projections by −α, or pass the tilt to the projector geometry |
| 5 | **Tilt-axis LATERAL shift** (centre) | arcs in the *same* direction in all slices | offset constant in z but displaced | three-slice arc test: `c` flat in z, mean ≠ assumed centre | correct the centre — geometric twin of mode 1 |
| 6 | **Out-of-plane tilt** (centre) | apparent axis position drifts monotonically with z | sinusoid *amplitude* grows linearly with z: `u = z sinβ sinθ` | regress `(a, b)` on z with `c` flat; gate on the vertical channel to exclude a tilted *sample* | solve for the full axis direction and use a vector geometry — a 2-D shift cannot represent it |
| 7 | **Angle readback error** (horizontal) | azimuthal (tangential) smearing | the sinusoid does not close over the reported span | refit `a sin(gθ) + b cos(gθ) + c` over a grid of gain `g`; best `g ≠ 1` halving the residual | recalibrate the angle axis before aligning anything |
| 8 | **Magnification / scale drift** (horizontal) | radial blur growing with radius; centre stays sharp | sinusoid amplitude wrong in proportion to the true radius | `σ_u² = A + B cos2θ + C sin2θ` is exact for a rigid object; look for a secular trend in its residual vs acquisition index | rescale each projection, or fix the cause (detector distance, energy drift) |
| 9 | **Deformation / radiation damage** (non-rigid) | no single rigid alignment works; some regions sharp, others smeared | individually inconsistent sinusoids; features appear, move, vanish | residual stays high **after** the best-fit rigid shift is removed **and** is localised, not spread; projected mass not conserved | non-rigid alignment [4] or time-resolved reconstruction; last resort, drop the damaged range |
| 10 | **Missing wedge** (coverage) | elongation along the missing direction | wedge-shaped gap; a wedge of unmeasured Fourier frequencies | angles only: reduce θ mod 180°, sort, measure the largest gap | acquire it, or accept it with a regularised reconstruction. **GridRec fails outright here**; SIRT/MLEM degrade gracefully |
| 11 | **Phase ramp / offset / unwrap failure** (data integrity) | cupping, background gradients, a floor that is not zero | per-view residual ramp; non-zero vacuum mean fluctuating view to view | fit offset+ramp on the vacuum border of every projection, look at the spread of the coefficients. Ten minutes, no reconstruction | remove ramp and offset **and** register on the gradient. **Run this first and most often** |
| 12 | **Local / interior tomography** (data integrity) | cupping plus a bright rim at the reconstruction border | projections do not return to vacuum at both edges | column profile minus its own minimum; compare both edge values with the peak | pad and extrapolate, or use an interior-tomography method. **Centre of mass and every moment estimate are invalid on truncated data** |

Two structural caveats, both encoded in the code rather than left as folklore:

* **Modes 1 and 5 are one equivalence class.** A laterally displaced axis and a constant
  lateral shift of the data are the same stack of numbers; no detector can separate them.
  Which one is "the" cause is a question about the instrument.
* **Random angle-readback noise is indistinguishable from lateral jitter** at the
  centroid level — both enter as amplitude × δ. Only the *systematic* gain or span error
  is separable this cheaply, which is what mode 7's probe targets.

---

## 4. The method landscape

| family | what it solves | cost | when it is the right tool |
|---|---|---|---|
| **Neighbour cross-correlation** (register projection *i* to *i*−1) | drift | trivial | never on its own: errors accumulate along the chain, and it cannot see a global geometry error |
| **Centroid / sinogram sinusoid** [1] | rotation centre; vertical warm start | one pass | **[1]** and a warm start for **[2]**. Structurally wrong for horizontal alignment (§1.2) |
| **1-D vertical mass distribution** [3] | the vertical axis, outright | one row-sum + 1-D registration | **[2]**, always, first. No projector, no volume |
| **Joint iterative reprojection (JIRR)** [1, 2] | per-view rigid shifts against the object | one reconstruction + one reprojection per outer iteration | **[3]**, the general-purpose loop. Registers the *phase*, so it inherits the gauge problem |
| **Gradient-domain reprojection** [3] | the same, made invariant to offset and ramp | as JIRR, plus one FFT pair per view | **[3]** when the phase background is not exactly planar; see §5.3 for the measured caveat |
| **Joint optimisation of volume *and* shifts** [3, 6] | both unknowns together, coarse-to-fine | many cheap gradient steps; needs a GPU projector at scale | **[3]** for large shifts (≈10²  px) and large stacks; fastest of the three at realistic scale |
| **Landmark / feature tracking** | axis direction, tilt drift, per-view shifts, from hand-labelled features | manual | **[4]**, and as an independent cross-check on any automatic geometry number (see §5.4) |
| **Non-rigid / deformation vector field** [4] | sample change *during* the scan | partial reconstructions + 3-D optical flow, iterated | **[6]**, only after the gate. Reported 53 → 27 nm on a beetle sample [4] |
| **Consistency conditions** (Helgason–Ludwig moment conditions) | detecting inconsistent geometry without a reconstruction | cheap | as a *detector*; the projected-mass conservation used in mode 9's probe is the zeroth-order case, and separated clean from deformed data by 128× where the reprojection residual separated them by 1.1× |

### 4.1 Reconstruction *inside* the alignment loop

A separate question from which algorithm produces the final volume.

* **SIRT / MLEM** — robust, degrade gracefully with limited angular range. The default.
* **GridRec** — **fails outright with limited angular range.** It interpolates onto a
  Cartesian Fourier grid, so an unsampled wedge is filled with interpolation noise
  instead of being left empty. `odstrcil.check_reconstruction_choice` raises on this and
  the refusal is fatal even under a permissive policy.
* **FBP initialisation** — converges **more slowly**, because FBP's streak artifacts are
  reprojected into the simulated stack and the registration then measures the streaks
  along with the sample, feeding them back into the shift estimate.

The final volume is a different trade: measured on a real dataset, FBP and SIRT-300 gave
188.4 nm and 190.6 nm interior FRC — one FRC bin apart — and the production choice was
made on cost (14 min vs 134 min per volume). That result is **not** licence to use a
direct method inside the loop.

---

## 5. Validating an alignment

### 5.1 The target

**Residual alignment error at or below ⅓ of the target voxel.** When reconstructing on
the detector grid that is 0.333 px. Judge it on the *worst* draw, not the mean.

### 5.2 What can and cannot certify it

| metric | sees | blind to |
|---|---|---|
| shift error vs **injected ground truth** | everything a per-view rigid model can express | only available on synthetic or forward-projected data |
| **reprojection residual** vs angle (+ lag-1 autocorrelation, localisation) | common-mode geometry error, deformation, angular structure | absolute scale; and it is gameable — see below |
| **split-half FSC / FRC** [9] | random per-view error | **all systematic geometric bias** |

**FSC cannot detect a common-mode geometric error, and this is exact, not approximate.**
A rigid-but-wrong geometry applied identically to both half-sets multiplies the two
transforms by `exp(−2πik·d)` and `exp(+2πik·d)`, and the factors cancel in the cross
term. Measured three independent ways: half-bit FRC read *exactly* 508.6 nm at centring
errors of 0, 4, 8, 16, 32 and 64 px while the true edge blur grew to 128 px; split-half
FSC read 11.97 / 11.97 / 11.98 px for a perfectly aligned, an unaligned and a
4.5 px-misaligned reconstruction; and a five-seed benchmark returned 19.51 px for every
row of every scenario — a 1198× range of alignment error compressed into 0.114 % of FSC.
`tests/test_benchmark.py` proves the invariance to machine precision.

**The replacement is not clean either.** On a non-rigid benchmark case the reprojection
residual ranked a *wrong* rigid answer better than the *true* shifts, because rigid
shifts absorb part of a deformation. **Always pair the two, and never quote either
alone.**

A related trap, observed live: two *under-converged* runs scored **better** on interior
FRC than the converged one, while an independent estimator that involves no
reconstruction at all (the vertical mass profile) and the data-consistency residual both
ranked them worst. FRC alone would have accepted a demonstrably wrong geometry.

### 5.3 The gradient trick, stated at the strength the evidence supports

The pairwise ramp invariance is exact and holds (§1.1). The **end-to-end** benefit inside
a reprojection loop was measured with the same engine and one knob changed, and the
gradient domain was **1.06–1.50× worse** than the value domain at every ramp from 0.25 to
2 rad, better only at 4 rad. Mechanism: differentiation is a high-pass filter, and it
weights the comparison exactly where a half-converged reprojection is least faithful.
Meanwhile, the *same idea used as the data term of a properly conditioned Gauss–Newton
optimiser* on real data improved interior FRC by 28–40 nm, confirmed by two independent
checks. The honest statement is therefore: **differentiate the residual of an optimiser,
not necessarily the similarity measure of a correlation** — and the perturbation
catalogue cannot yet express a non-planar phase background, which is the regime the trick
was designed for. See `IMPLEMENTATION_NOTES.md` §2.3.

### 5.4 Geometry numbers need a second, model-free estimator

Two calibrated estimators of the same in-plane axis tilt disagreed **in sign** on a real
dataset, and applying the correction made 2 of 3 measured rows worse. This repository
reproduced the pathology: the three-slice arc test returned a 0.96-confidence 3.9° tilt
that an independent per-band entropy centre sweep contradicted in magnitude *and* sign. A
model-strain guard now halves such a verdict and names the cross-check to run. Do not act
on an axis-geometry number that a second, structurally different estimator has not
confirmed.

### 5.5 Sampling limits, so the target is read against something

For an object of diameter `D` reconstructed from `N` views over 180°, the Crowther
criterion [10] gives a resolution limit of order `πD/N` (`πD/2N` for the optimistic
convention). Compute it before blaming alignment: a dataset already at its sampling limit
cannot be improved by better shifts, and on one real scan the whole remaining
rigid-alignment prize was calibrated at **≈7.5 nm at the ⅓-voxel target** against a
158 nm volume FSC — i.e. alignment was not the binding constraint at all.

---

## 6. References

Cited where the corresponding module docstring cites them; module docstrings are the
authority for implementation detail.

1. D. Gürsoy, Y. P. Hong, K. He, K. Hujsak, S. Yoo, S. Chen, Y. Li, M. Ge,
   L. M. Miller, Y. S. Chu, V. De Andrade, K. He, O. Cossairt, A. K. Katsaggelos and
   C. Jacobsen, *Rapid alignment of nanotomography data using joint iterative
   reconstruction and reprojection*, **Sci. Rep. 7**, 11818 (2017).
2. D. Gürsoy, F. De Carlo, X. Xiao and C. Jacobsen, *TomoPy: a framework for the analysis
   of synchrotron tomographic data*, **J. Synchrotron Radiat. 21**, 1188 (2014).
3. M. Odstrčil, M. Holler, J. Raabe and M. Guizar-Sicairos, *Alignment methods for
   nanotomography with deep subpixel accuracy*, **Opt. Express 27**, 36637 (2019).
4. M. Odstrčil, M. Holler, J. Raabe, A. Sepe, X. Sheng, S. Vignolini, C. G. Schroer and
   M. Guizar-Sicairos, *Ab initio nonrigid X-ray nanotomography*, **Nat. Commun. 10**,
   4778 (2019).
5. M. Guizar-Sicairos, S. T. Thurman and J. R. Fienup, *Efficient subpixel image
   registration algorithms*, **Opt. Lett. 33**, 156 (2008). — the upsampled-DFT sub-pixel
   refinement used in `vertical.py` and `gradient.py`.
6. Aligned-sinogram / joint reconstruction-and-motion refinement, **Opt. Express 32**,
   10801 (2024) — as cited in `core/joint_gd.py`, the source of the optimiser machinery
   (analytic adjoints, Nesterov momentum, multi-resolution schedule).
7. P. Thibault and M. Guizar-Sicairos, *Maximum-likelihood refinement for coherent
   diffractive imaging*, **New J. Phys. 14**, 063004 (2012). — the phase ambiguities that
   ptychographic reconstruction leaves behind.
8. M. Holler, M. Guizar-Sicairos, E. H. R. Tsai, R. Dinapoli, E. Müller, O. Bunk,
   J. Raabe and G. Aeppli, *High-resolution non-destructive three-dimensional imaging of
   integrated circuits*, **Nature 543**, 402 (2017). — ptychographic tomography at the
   scale where alignment dominates.
9. M. van Heel and M. Schatz, *Fourier shell correlation threshold criteria*,
   **J. Struct. Biol. 151**, 250 (2005).
10. R. A. Crowther, D. J. DeRosier and A. Klug, *The reconstruction of a three-dimensional
    structure from projections and its application to electron microscopy*,
    **Proc. R. Soc. Lond. A 317**, 319 (1970).
11. B. K. P. Horn and B. G. Schunck, *Determining optical flow*, **Artif. Intell. 17**,
    185 (1981). — the flow estimator in `core/deformation.py`.
12. M. Donath, F. Beckmann and A. Schreyer, *Automated determination of the center of
    rotation in tomography data*, **J. Opt. Soc. Am. A 23**, 1048 (2006). — the entropy
    centre sweep used by `diagnostics.probe_center_sweep`.
13. W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski,
    J. De Beenhouwer, K. J. Batenburg and J. Sijbers, *Fast and flexible X-ray tomography
    using the ASTRA toolbox*, **Opt. Express 24**, 25129 (2016).
