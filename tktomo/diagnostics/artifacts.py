"""The artifact-to-cause table as *data*, plus the verdict types the probes return.

A tomographic reconstruction fails in a small number of stereotyped ways, and the
table that maps "what the slice looks like" to "what is actually wrong" is the most
reused page of any alignment write-up. Prose cannot be acted on by code, so here the
table is a catalogue of :class:`ArtifactSpec` records keyed by :class:`FailureMode`,
and the probes in :mod:`tktomo.diagnostics.tests_geometry` return
:class:`Finding` objects that *point at a row of it*.

Three conventions, each of which is a bug if you get it wrong:

1. **A probe that cannot run must say so, not score zero.** ``ProbeStatus.CLEAR``
   means "I ran and found nothing"; ``ProbeStatus.NOT_APPLICABLE`` means "I could not
   run, here is why". Collapsing the two turns a missing precondition (no vacuum
   border, no 180 deg span, no reconstruction) into a clean bill of health, which is
   the single most dangerous thing a diagnostic can do.

2. **Confidence is monotone in evidence, never a probability.** It is
   :func:`confidence_from_ratio` of (statistic / threshold): 0 at the threshold, 0.5
   at twice it, 0.9 at ten times it. Two findings are comparable only within the same
   probe. Do not read it as P(mode is present).

3. **Modes are not mutually exclusive and the ranking is not a classification.**
   A wrong rotation centre (mode 1) *is* a lateral axis offset (mode 5) seen from the
   other side of the geometry, and a residual phase ramp (mode 11) manufactures an
   apparent lateral shift. The verdict is a ranked list on purpose; the caller acts on
   the top finding, fixes it, and runs again.

The 12 modes are the roadmap's table, in its order. Rows 1-10 are generic parallel-beam
tomography; row 11 is specific to *ptychographic* phase projections and, being both the
cheapest test and the most common cause, is the one to run first and most often.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "CATALOGUE",
    "INVALIDATED_BY",
    "STAGE_ORDER",
    "ArtifactSpec",
    "DiagnosticConfig",
    "FailureMode",
    "Finding",
    "ProbeResult",
    "ProbeStatus",
    "TriageStage",
    "Verdict",
    "apply_stage_discount",
    "confidence_from_ratio",
    "rank",
    "spec_for",
    "stage_of",
]


class FailureMode(str, Enum):
    """The 12 rows of the artifact-to-cause table.

    ``str`` mixin on purpose: the value is a stable slug, so ``json.dumps`` of a
    verdict needs no encoder and a UI can round-trip a mode through a config file.
    """

    WRONG_CENTER = "wrong_center"  # 1
    JITTER = "jitter"  # 2
    VERTICAL_DRIFT = "vertical_drift"  # 3
    TILT_AXIS_ANGLE = "tilt_axis_angle"  # 4
    TILT_AXIS_LATERAL = "tilt_axis_lateral"  # 5
    OUT_OF_PLANE_TILT = "out_of_plane_tilt"  # 6
    ANGLE_READBACK = "angle_readback"  # 7
    SCALE_DRIFT = "scale_drift"  # 8
    DEFORMATION = "deformation"  # 9
    MISSING_WEDGE = "missing_wedge"  # 10
    PHASE_RAMP = "phase_ramp"  # 11
    LOCAL_TOMOGRAPHY = "local_tomography"  # 12


class TriageStage(str, Enum):
    """The roadmap's order of operations. Never to be violated.

    Ramp/offset removal, then rotation centre, then vertical, then horizontal, and
    only then non-rigid. Each stage's fix invalidates the measurements of every later
    stage, which is why :func:`~tktomo.diagnostics.tests_geometry.triage` stops at the
    first thing that fires instead of reporting everything it can find.

    ``COVERAGE`` sits second because it is pure geometry (angles only, no fix in the
    alignment loop at all): knowing there is a 40 deg wedge changes how you read every
    number that follows.
    """

    DATA_INTEGRITY = "data_integrity"
    COVERAGE = "coverage"
    ROTATION_CENTRE = "rotation_centre"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    NON_RIGID = "non_rigid"


#: Stages in the order the roadmap prescribes.
STAGE_ORDER: tuple[TriageStage, ...] = (
    TriageStage.DATA_INTEGRITY,
    TriageStage.COVERAGE,
    TriageStage.ROTATION_CENTRE,
    TriageStage.VERTICAL,
    TriageStage.HORIZONTAL,
    TriageStage.NON_RIGID,
)


class ProbeStatus(str, Enum):
    """Outcome of running one probe.

    ``NOT_APPLICABLE`` is the load-bearing one: several probes need a reconstruction,
    a vacuum border, or a full 180 deg span, and a probe that lacks its precondition
    reports *why* rather than inventing a score. ``ERROR`` is for an unexpected
    exception inside a probe -- the runner records it and carries on, so one broken
    probe cannot take the whole diagnosis down with it.
    """

    FIRED = "fired"
    CLEAR = "clear"
    NOT_APPLICABLE = "not_applicable"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class ArtifactSpec:
    """One row of the table: what it looks like, how to confirm it, how to fix it."""

    mode: FailureMode
    number: int  # the roadmap's row number, 1-12
    title: str
    stage: TriageStage
    slice_signature: str
    sinogram_signature: str
    confirm: str
    fix: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "number": self.number,
            "title": self.title,
            "stage": self.stage.value,
            "slice_signature": self.slice_signature,
            "sinogram_signature": self.sinogram_signature,
            "confirm": self.confirm,
            "fix": self.fix,
            "notes": self.notes,
        }


def _catalogue() -> dict[FailureMode, ArtifactSpec]:
    rows = [
        ArtifactSpec(
            mode=FailureMode.WRONG_CENTER,
            number=1,
            title="Constant lateral shift / wrong rotation centre",
            stage=TriageStage.ROTATION_CENTRE,
            slice_signature=(
                "Tuning-fork doubled edges: every sharp feature splits into a pair of "
                "arcs opening away from the axis. Uniform over the whole slice and "
                "identical in every slice."
            ),
            sinogram_signature="All sinusoids offset by the same constant in u.",
            confirm=(
                "Sweep the assumed centre and minimise slice entropy (or maximise "
                "gradient energy); independently, register projection theta against the "
                "left-right flip of its theta+180 partner -- the axis is at "
                "((n_u - 1) - shift) / 2."
            ),
            fix=(
                "Set the rotation centre. Do NOT let the alignment loop absorb it into "
                "per-projection shifts: a constant lateral offset is one number, and "
                "spending 900 free parameters on it invites the loop to drift."
            ),
            notes=(
                "Same physical defect as mode 5 seen from the other side of the "
                "geometry (data shifted vs axis moved); the two probes cross-reference."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.JITTER,
            number=2,
            title="Random per-projection jitter",
            stage=TriageStage.HORIZONTAL,
            slice_signature=(
                "Uniform blurring, NOT doubled edges. Resolution loss without a "
                "structure you can point at."
            ),
            sinogram_signature=(
                "High-frequency scatter about the ideal sinusoid; the residual is white "
                "in acquisition order (lag-1 autocorrelation near zero)."
            ),
            confirm=(
                "Fit com_u(theta) = a sin + b cos + c and look at the residual's "
                "autocorrelation: white means jitter, smooth means drift or an angle "
                "error. Same test on the vertical mass profile."
            ),
            fix=(
                "Iterative reprojection alignment on the phase GRADIENT (Odstrcil "
                "2019). Jitter is exactly what the loop is for."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.VERTICAL_DRIFT,
            number=3,
            title="Vertical drift",
            stage=TriageStage.VERTICAL,
            slice_signature="Slice-to-slice smearing along z; features leak between slices.",
            sinogram_signature="Smooth low-order drift of the object in a y-z (vertical) sinogram.",
            confirm=(
                "Row-sum each projection to a 1D vertical mass profile, register the "
                "profiles, and split the resulting shift series into a smooth trend and "
                "a white residual. A large trend is drift."
            ),
            fix=(
                "Align the vertical direction with the 1D vertical mass distribution "
                "(Odstrcil 2019). Vertical extent is invariant under rotation about a "
                "vertical axis, so this needs no forward or back projection at all."
            ),
            notes=(
                "The one direction where centre of mass is defensible -- and only when "
                "the sample is fully inside the field of view."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.TILT_AXIS_ANGLE,
            number=4,
            title="Tilt-axis ANGLE error (axis tilted within the detector plane)",
            stage=TriageStage.ROTATION_CENTRE,
            slice_signature=(
                "Arcing that curves in OPPOSITE directions above and below the mid-plane; "
                "the mid-plane slice looks fine."
            ),
            sinogram_signature=(
                "The sinusoid offset c is a linear function of detector row: c(z) = c0 + "
                "z tan(alpha), and is independent of theta."
            ),
            confirm=(
                "Three-slice arc test: fit the centroid sinusoid in a top, middle and "
                "bottom row band and regress the constant term c against z. A non-zero "
                "slope with a zero mean offset is an axis angle error."
            ),
            fix="Rotate the projections by -alpha (or pass the tilt to the projector geometry).",
        ),
        ArtifactSpec(
            mode=FailureMode.TILT_AXIS_LATERAL,
            number=5,
            title="Tilt-axis LATERAL shift",
            stage=TriageStage.ROTATION_CENTRE,
            slice_signature="Arcing in the SAME direction in all slices, top to bottom.",
            sinogram_signature="Sinusoid offset c constant in z but displaced from the assumed axis.",
            confirm=(
                "Three-slice arc test: constant term c flat in z (slope ~ 0) but its mean "
                "differs from the assumed centre."
            ),
            fix="Correct the rotation centre; see mode 1, of which this is the geometric twin.",
        ),
        ArtifactSpec(
            mode=FailureMode.OUT_OF_PLANE_TILT,
            number=6,
            title="Out-of-plane tilt (axis leaning toward or away from the beam)",
            stage=TriageStage.ROTATION_CENTRE,
            slice_signature="The apparent axis position drifts smoothly and MONOTONICALLY with z.",
            sinogram_signature=(
                "A point at height z on the nominal axis lands at u = z sin(beta) "
                "sin(theta): the sinusoid's AMPLITUDE, not its offset, grows linearly "
                "with z."
            ),
            confirm=(
                "Three-slice arc test: regress the sinusoid coefficients (a, b) against "
                "z. A linear trend there, with the constant term c flat, is an "
                "out-of-plane tilt."
            ),
            fix=(
                "Solve for the full axis direction (both tilt angles) and hand it to a "
                "vector geometry -- a 2D shift correction cannot represent it."
            ),
            notes=(
                "Confounded with a genuinely tilted or sheared SAMPLE, whose slice "
                "centroids also walk linearly with z. Distinguishing them needs the "
                "reprojection residual, not the centroids."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.ANGLE_READBACK,
            number=7,
            title="Angle readback error",
            stage=TriageStage.HORIZONTAL,
            slice_signature=(
                "Azimuthal smearing: features spread TANGENTIALLY (around the axis), not "
                "radially."
            ),
            sinogram_signature=(
                "The sinusoid does not close on itself over the reported span; refitting "
                "with a free angular gain g reduces the residual sharply."
            ),
            confirm=(
                "Refit com_u = a sin(g theta) + b cos(g theta) + c over a grid of g. A "
                "best g away from 1 that halves the residual is a systematic readback "
                "error (a stage scale error or a wrong span)."
            ),
            fix="Recalibrate the angle axis (or use the fitted gain) before aligning anything.",
            notes=(
                "RANDOM readback noise is mathematically indistinguishable from lateral "
                "jitter (mode 2) at the centroid level; only the systematic part is "
                "separable this cheaply."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.SCALE_DRIFT,
            number=8,
            title="Magnification / scale drift",
            stage=TriageStage.HORIZONTAL,
            slice_signature="Radial blur that grows with distance from the axis; the centre stays sharp.",
            sinogram_signature="Sinusoid AMPLITUDE wrong in proportion to the true radius.",
            confirm=(
                "For a rigid object the projected second moment is exactly harmonic in "
                "2 theta: sigma_u^2 = A + B cos 2theta + C sin 2theta. Fit it and look "
                "for a secular trend in the residual with acquisition index; a scale "
                "s(i) shows up as sigma^2 ~ s(i)^2."
            ),
            fix="Rescale each projection (or fix the detector distance / energy drift that caused it).",
        ),
        ArtifactSpec(
            mode=FailureMode.DEFORMATION,
            number=9,
            title="Sample deformation or radiation damage",
            stage=TriageStage.NON_RIGID,
            slice_signature="No single rigid alignment works; some regions sharp, others smeared.",
            sinogram_signature="Sinusoids that are individually inconsistent -- features appear, move, vanish.",
            confirm=(
                "Reproject the best rigid-aligned reconstruction and look at the "
                "residual: it stays high AFTER the best-fit rigid shift is removed, and "
                "it is LOCALISED to sub-regions rather than spread over the detector."
            ),
            fix=(
                "Non-rigid / per-region alignment, or time-resolved reconstruction. Last "
                "resort: drop the damaged angular range."
            ),
            notes=(
                "Only trust this after modes 1, 3, 4, 5 are cleared -- every uncorrected "
                "rigid error also leaves a high reprojection residual."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.MISSING_WEDGE,
            number=10,
            title="Missing wedge / limited angular range",
            stage=TriageStage.COVERAGE,
            slice_signature="Elongation along the missing direction; edges perpendicular to it fade.",
            sinogram_signature="A wedge-shaped gap; in Fourier space a wedge of unmeasured frequencies.",
            confirm="Angles only: reduce theta modulo 180 deg, sort, and measure the largest gap.",
            fix=(
                "Acquire the missing range, or accept it and use a regularised / "
                "constrained reconstruction. GridRec fails outright here; SIRT and MLEM "
                "degrade gracefully."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.PHASE_RAMP,
            number=11,
            title="Phase ramp / offset / unwrap failure (ptycho-specific)",
            stage=TriageStage.DATA_INTEGRITY,
            slice_signature="Cupping, background gradients, bowl artifacts; a floor that is not zero.",
            sinogram_signature=(
                "A per-projection residual ramp and a NONZERO VACUUM MEAN that fluctuates "
                "from projection to projection."
            ),
            confirm=(
                "Fit offset + ramp on a presumed-vacuum border of every projection and "
                "look at the spread of the fitted coefficients across the stack. Ten "
                "minutes, no reconstruction."
            ),
            fix=(
                "Remove the ramp and offset (see "
                "tktomo.ptycho_align.core.preprocess.remove_phase_ramp), and register on "
                "the phase GRADIENT rather than the phase: differentiating sends the "
                "constant offset to zero and the linear ramp to a constant, which is "
                "exactly the pair of ambiguities ptychographic phase retrieval leaves "
                "behind."
            ),
            notes=(
                "Run this FIRST and most often. A residual ramp is mathematically "
                "indistinguishable from a lateral shift, so it does not merely add "
                "artifacts -- it poisons the alignment that is supposed to fix them."
            ),
        ),
        ArtifactSpec(
            mode=FailureMode.LOCAL_TOMOGRAPHY,
            number=12,
            title="Local / interior tomography (truncated projections)",
            stage=TriageStage.DATA_INTEGRITY,
            slice_signature="Cupping across the field plus a bright rim at the reconstruction border.",
            sinogram_signature="Projections truncated: the object does not return to vacuum at both edges.",
            confirm=(
                "Per projection, take the column profile, subtract its own minimum, and "
                "compare the two edge values with the peak. Both edges elevated means "
                "the object exceeds the field of view."
            ),
            fix=(
                "Pad and extrapolate the projections before reconstruction, or "
                "reconstruct with an interior-tomography method. Note that centre of "
                "mass and every moment-based estimate are invalid on truncated data."
            ),
        ),
    ]
    return {row.mode: row for row in rows}


#: The table, keyed by mode. Ordered by the roadmap's row number.
CATALOGUE: dict[FailureMode, ArtifactSpec] = _catalogue()


def spec_for(mode: FailureMode | str) -> ArtifactSpec:
    """The catalogue row for ``mode``. Raises ``KeyError`` on an unknown mode."""
    key = FailureMode(mode)
    return CATALOGUE[key]


def stage_of(mode: FailureMode | str) -> TriageStage:
    """Which triage stage owns ``mode``."""
    return spec_for(mode).stage


def confidence_from_ratio(value: float, threshold: float) -> float:
    """Map a statistic and its threshold onto a monotone confidence in ``[0, 1)``.

    ``0`` at or below the threshold, ``0.5`` at twice it, ``0.9`` at ten times it::

        confidence = 1 - threshold / value

    It is deliberately NOT a probability (see the module docstring). Both arguments
    must be finite and the threshold strictly positive; a non-finite statistic returns
    ``0.0`` rather than propagating a NaN into a ranking.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if not math.isfinite(value) or value <= threshold:
        return 0.0
    return float(1.0 - threshold / value)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Thresholds, all in physical units, all documented, all overridable.

    The defaults are set to the roadmap's accuracy target: a residual alignment error at
    or below 1/3 of the target voxel. That puts the shift thresholds at a few tenths of a
    pixel -- tight enough that a firing is worth acting on, loose enough that
    discretisation noise on a well-aligned stack does not trip it.

    Phase thresholds are FRACTIONS of the projection's own contrast (the 1-99
    percentile spread of the interior), never absolute radians, because the phase scale
    of a projection depends entirely on the sample.
    """

    #: Vacuum-mean spread across the stack, as a fraction of projection contrast (mode 11).
    vacuum_mean_frac: float = 0.02
    #: Peak-to-valley of the fitted background ramp, as a fraction of contrast (mode 11).
    vacuum_ramp_frac: float = 0.05
    #: Border width in pixels used as the presumed-vacuum frame.
    border: int = 8
    #: Gradient of the column profile at BOTH frame edges, as a fraction of the
    #: profile's own peak gradient, before the object counts as truncated (mode 12).
    #: The gradient, not the level, because it needs no vacuum reference: an enclosed
    #: object's profile is flat at the edges whatever the background offset is, while a
    #: truncated one is still on a slope where the field of view cut it off.
    truncation_frac: float = 0.15
    #: Largest gap in theta modulo 180 deg, in degrees, before it is a missing wedge (mode 10).
    wedge_gap_deg: float = 10.0
    #: Rotation-centre disagreement in pixels that matters (modes 1, 5).
    center_tol_px: float = 0.5
    #: Apparent axis travel between the mid-plane and the top of the OBJECT (not of the
    #: detector), in pixels (modes 4, 6). Measured noise floor on a clean synthetic
    #: phantom: 0.001 px, so this is a threshold against real geometry, not against noise.
    tilt_tol_px: float = 0.3
    #: How large the per-band centroid-sinusoid residual may be relative to the axis walk
    #: the arc test claims, before that claim is downgraded (modes 4, 6). The rigid-object
    #: sinusoid is the arc test's whole model, and over a 180 deg span its basis
    #: {sin, cos, 1} is not orthogonal -- so a residual that is CORRELATED with theta (as
    #: leftover per-projection misalignment is) leaks into the constant term, differently
    #: for each band, and manufactures a slope. Measured on real data: a stack with 19 px
    #: rms of band residual produced a confident 3.9 deg tilt that an independent
    #: per-band entropy centre sweep contradicted in both magnitude and sign. A standard
    #: error would not have caught it -- that residual is structured, not white -- so the
    #: comparison is against the raw residual, deliberately conservative.
    arc_residual_ratio: float = 1.0
    #: Amplitude of the sin(theta) modulation of the vertical mass centroid, in pixels,
    #: before the rotation axis counts as non-vertical (mode 6). This is the gate on the
    #: out-of-plane finding and it is what makes the finding trustworthy: a rigid sample
    #: rotating about a truly vertical axis has an EXACTLY rotation-invariant vertical
    #: mass profile, while a sample that is merely tilted or sheared -- which produces the
    #: same z-walking sinusoid amplitude as an out-of-plane tilt -- leaves it invariant.
    #: Only the sin component is used: over a monotone 0-180 deg sweep the cos component
    #: is degenerate with a vertical drift (mode 3).
    vertical_modulation_tol_px: float = 0.2
    #: Peak-to-peak of the smooth part of the vertical shift series, in pixels (mode 3).
    vertical_drift_tol_px: float = 1.0
    #: RMS of the white part of either shift series, in pixels (mode 2).
    jitter_tol_px: float = 0.3
    #: Lag-1 autocorrelation below which a residual series counts as white (mode 2).
    jitter_autocorr_max: float = 0.4
    #: Displacement of the object's OUTERMOST material implied by the fitted scale drift,
    #: in pixels, before it matters (mode 8). A fractional threshold would be the wrong
    #: knob: 1% on a 10 px object is nothing and 1% on a 1000 px object is 10 px of blur,
    #: and mode 8's whole signature is that the blur grows with distance from the axis.
    scale_rim_tol_px: float = 0.5
    #: Fractional reduction in centroid-fit residual required of a free angular gain (mode 7).
    angle_gain_gain_ratio: float = 0.5
    #: Tangential displacement at the object's rim, in pixels, implied by the fitted
    #: angular gain, before a readback error matters (mode 7). This rather than the
    #: centroid residual is the firing criterion: a 6% angle error leaves a centroid
    #: residual of only ~0.05 px (a half-period sinusoid absorbs almost all of it) while
    #: smearing the rim of a 35 px object by 3 px. Measured on the synthetic phantom.
    angle_tangential_tol_px: float = 1.0
    #: Reprojection residual, relative to the measured projection's own RMS (mode 9).
    deformation_residual_frac: float = 0.15
    #: Fraction of residual energy in the busiest 10% of detector columns before the
    #: residual counts as localised rather than spread (mode 9). Uniform noise gives 0.1.
    deformation_locality: float = 0.35
    #: Non-smooth variation of the total projected mass, as a fraction of its mean,
    #: before the object counts as non-rigid (mode 9). Parallel-beam projections of an
    #: enclosed object conserve mass EXACTLY at every angle, so this is the sharpest
    #: rigidity test available and it costs one pass: on the synthetic phantom a clean
    #: stack gives 2e-4 and an injected deformation 2.6e-2, a factor of 128.
    deformation_mass_frac: float = 5e-3
    #: Width in pixels above which single-slice probes bin the sinogram before reconstructing.
    max_recon_width: int = 256
    #: Angles above which single-slice probes subsample the scan before reconstructing.
    #: 180 views is already past the Crowther limit for a 256 px slice, so the extra
    #: angles buy the sharpness metric nothing and cost it linear time.
    max_recon_angles: int = 180
    #: Half-width, in pixels, of the rotation-centre sweep.
    center_sweep_px: float = 10.0
    #: Step, in pixels, of the rotation-centre sweep.
    center_sweep_step: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict  # noqa: PLC0415

        return asdict(self)


@dataclass(frozen=True)
class Finding:
    """One probe pointing at one row of the catalogue, with the numbers that did it."""

    mode: FailureMode
    confidence: float
    probe: str
    detail: str
    evidence: Mapping[str, float] = field(default_factory=dict)

    @property
    def spec(self) -> ArtifactSpec:
        return CATALOGUE[self.mode]

    @property
    def fix(self) -> str:
        return self.spec.fix

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "number": self.spec.number,
            "title": self.spec.title,
            "stage": self.spec.stage.value,
            "confidence": round(float(self.confidence), 4),
            "probe": self.probe,
            "detail": self.detail,
            "evidence": {k: _jsonable(v) for k, v in self.evidence.items()},
            "fix": self.spec.fix,
        }


@dataclass(frozen=True)
class ProbeResult:
    """What one probe did: its status, its numbers, and any findings it raised.

    ``metrics`` is scalars only, so the whole verdict round-trips through JSON.
    ``curves`` holds the per-projection arrays a plot needs; they are dropped from
    :meth:`to_dict` unless asked for, because a thousand-point array per probe is not a
    report, it is a data file.
    """

    probe: str
    stage: TriageStage
    status: ProbeStatus
    reason: str = ""
    detail: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)
    findings: tuple[Finding, ...] = ()
    curves: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    @property
    def fired(self) -> bool:
        return self.status is ProbeStatus.FIRED

    @property
    def ran(self) -> bool:
        """True when the probe actually produced numbers (fired or clear)."""
        return self.status in (ProbeStatus.FIRED, ProbeStatus.CLEAR)

    def to_dict(self, *, include_curves: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "probe": self.probe,
            "stage": self.stage.value,
            "status": self.status.value,
            "reason": self.reason,
            "detail": self.detail,
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
            "findings": [f.to_dict() for f in self.findings],
        }
        if include_curves:
            out["curves"] = {k: np.asarray(v).tolist() for k, v in self.curves.items()}
        return out


@dataclass(frozen=True)
class Verdict:
    """The machine-readable answer: ranked findings plus every probe's status.

    ``findings`` is sorted by confidence, highest first, and is empty when nothing
    fired -- which is a real, meaningful answer only if you also read ``probes`` and
    check how many of them were ``NOT_APPLICABLE``. :attr:`coverage` is that number.
    """

    findings: tuple[Finding, ...]
    probes: tuple[ProbeResult, ...]
    stopped_at: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> Finding | None:
        return self.findings[0] if self.findings else None

    @property
    def coverage(self) -> float:
        """Fraction of probes that actually ran. Read the verdict in this light."""
        if not self.probes:
            return 0.0
        return sum(1 for p in self.probes if p.ran) / len(self.probes)

    def by_mode(self, mode: FailureMode | str) -> Finding | None:
        key = FailureMode(mode)
        for finding in self.findings:
            if finding.mode is key:
                return finding
        return None

    def probe(self, name: str) -> ProbeResult:
        for result in self.probes:
            if result.probe == name:
                return result
        raise KeyError(f"no probe named {name!r}; ran: {[p.probe for p in self.probes]}")

    def to_dict(self, *, include_curves: bool = False) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "probes": [p.to_dict(include_curves=include_curves) for p in self.probes],
            "stopped_at": self.stopped_at,
            "coverage": round(self.coverage, 4),
            "context": {k: _jsonable(v) for k, v in self.context.items()},
        }

    def to_json(self, *, indent: int = 2, include_curves: bool = False) -> str:
        return json.dumps(self.to_dict(include_curves=include_curves), indent=indent)


def rank(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Sort findings by confidence (descending), then by triage stage, then by row number.

    The confidence is rounded to three decimals *in the sort key only*: a difference of
    0.0004 between two monotone evidence scores is not a reason to reorder them, and
    without the rounding a near-tie would silently outrank the roadmap's order of
    operations. Ties fall to the earlier stage, so the finding you are supposed to act
    on first comes first.
    """
    def key(f: Finding) -> tuple[float, int, int]:
        return (-round(float(f.confidence), 3), STAGE_ORDER.index(f.spec.stage), f.spec.number)

    return tuple(sorted(findings, key=key))


#: Modes whose firing invalidates the *measurement* another mode's finding rests on,
#: within the same triage stage. Truncation is the only one so far and it is a real
#: dependency, not a heuristic: the ramp fit of mode 11 is performed on a presumed-vacuum
#: border, and mode 12 says that border is object.
INVALIDATED_BY: dict[FailureMode, tuple[FailureMode, ...]] = {
    FailureMode.PHASE_RAMP: (FailureMode.LOCAL_TOMOGRAPHY,),
}


def apply_stage_discount(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Halve a finding's confidence for every EARLIER triage stage that also fired.

    The roadmap's order of operations is not advice about what to fix first, it is a
    statement about which measurements are valid: a residual phase ramp moves every
    centroid, truncation invalidates every moment, and a wrong rotation centre inflates
    every reprojection residual. So a later-stage finding measured in the presence of an
    earlier-stage failure is evidence about a quantity that has not been measured
    correctly yet, and it is discounted rather than reported at face value.

    :data:`INVALIDATED_BY` adds the same treatment *within* a stage where one mode's
    firing invalidates another's measurement -- truncation versus the ramp fit.

    A stage counts as fired when something in it reaches confidence 0.25. The original
    number is preserved in the finding's evidence as
    ``confidence_before_stage_discount`` -- nothing is thrown away, it is re-weighted.
    """
    findings = list(findings)
    fired_stages = {
        f.spec.stage for f in findings if float(f.confidence) >= 0.25
    }
    fired_modes = {f.mode for f in findings if float(f.confidence) >= 0.25}
    out: list[Finding] = []
    for finding in findings:
        index = STAGE_ORDER.index(finding.spec.stage)
        earlier = sum(1 for stage in fired_stages if STAGE_ORDER.index(stage) < index)
        reasons = [s.value for s in fired_stages if STAGE_ORDER.index(s) < index]
        invalidators = [
            m.value for m in INVALIDATED_BY.get(finding.mode, ()) if m in fired_modes
        ]
        earlier += len(invalidators)
        if earlier == 0:
            out.append(finding)
            continue
        names = ", ".join(sorted(reasons + invalidators))
        factor = 0.5**earlier
        evidence = dict(finding.evidence)
        evidence["confidence_before_stage_discount"] = float(finding.confidence)
        out.append(
            Finding(
                mode=finding.mode,
                confidence=float(finding.confidence) * factor,
                probe=finding.probe,
                detail=finding.detail
                + f" [confidence x{factor:g}: the earlier stage(s) {names} fired, so the "
                "measurement behind this finding is not yet trustworthy]",
                evidence=evidence,
            )
        )
    return tuple(out)


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars / arrays to plain Python so ``json.dumps`` never chokes."""
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)
