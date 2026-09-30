"""Executable diagnostics: which tomographic failure mode is in this data?

The roadmap's artifact-to-cause table lists twelve stereotyped ways a tomographic
reconstruction goes wrong, each with a slice signature, a sinogram signature, a way to
confirm it and a fix. This package is that table turned into code:
:mod:`~tktomo.diagnostics.artifacts` holds the catalogue and the verdict types,
:mod:`~tktomo.diagnostics.tests_geometry` holds one probe per row, and
:mod:`~tktomo.diagnostics.report` renders the answer.

Start here::

    from tktomo.diagnostics import triage, format_verdict

    verdict = triage(projections, theta_deg)     # stops at the first thing that fires
    print(format_verdict(verdict))
    if verdict.top is not None:
        print(verdict.top.mode, verdict.top.fix)

:func:`~tktomo.diagnostics.tests_geometry.triage` runs the probes in the roadmap's
prescribed order -- ramp/offset, coverage, rotation centre, vertical, horizontal, and
only then non-rigid -- and stops at the first firing, because each stage's fix
invalidates every measurement after it. :func:`~tktomo.diagnostics.tests_geometry.diagnose`
runs all of them and ranks what it finds. Both return a
:class:`~tktomo.diagnostics.artifacts.Verdict`, which is dataclasses all the way down
and serialises with ``verdict.to_json()``.

The cheapest and most important single probe is
:func:`~tktomo.diagnostics.tests_geometry.probe_vacuum_phase` (mode 11, a residual
phase ramp): it needs no angles, no reconstruction and one pass over the data, and a
residual ramp is mathematically indistinguishable from a lateral shift -- so it does
not merely add artifacts, it poisons the alignment that is supposed to remove them.
Run it first and most often.

Importing this package pulls in numpy only; ``scipy.ndimage`` (the reconstruction
probes) and ``matplotlib`` (the optional figure) are imported inside the functions
that need them, and there are no other dependencies.
"""

from __future__ import annotations

from tktomo.diagnostics.artifacts import (
    CATALOGUE,
    INVALIDATED_BY,
    STAGE_ORDER,
    ArtifactSpec,
    DiagnosticConfig,
    FailureMode,
    Finding,
    ProbeResult,
    ProbeStatus,
    TriageStage,
    Verdict,
    apply_stage_discount,
    confidence_from_ratio,
    rank,
    spec_for,
    stage_of,
)
from tktomo.diagnostics.report import (
    format_catalogue,
    format_verdict,
    plot_verdict,
    save_verdict,
)
from tktomo.diagnostics.tests_geometry import (
    PROBES,
    StackMoments,
    diagnose,
    fbp_slice,
    forward_project_slice,
    probe_angle_readback,
    probe_angular_coverage,
    probe_axis_tilt,
    probe_center_consistency,
    probe_center_sweep,
    probe_deformation,
    probe_scale_drift,
    probe_shift_jitter,
    probe_truncation,
    probe_vacuum_phase,
    probe_vertical_drift,
    stack_moments,
    triage,
)

__all__ = [
    "CATALOGUE",
    "INVALIDATED_BY",
    "PROBES",
    "STAGE_ORDER",
    "ArtifactSpec",
    "DiagnosticConfig",
    "FailureMode",
    "Finding",
    "ProbeResult",
    "ProbeStatus",
    "StackMoments",
    "TriageStage",
    "Verdict",
    "apply_stage_discount",
    "confidence_from_ratio",
    "diagnose",
    "fbp_slice",
    "format_catalogue",
    "format_verdict",
    "forward_project_slice",
    "plot_verdict",
    "probe_angle_readback",
    "probe_angular_coverage",
    "probe_axis_tilt",
    "probe_center_consistency",
    "probe_center_sweep",
    "probe_deformation",
    "probe_scale_drift",
    "probe_shift_jitter",
    "probe_truncation",
    "probe_vacuum_phase",
    "probe_vertical_drift",
    "rank",
    "save_verdict",
    "spec_for",
    "stack_moments",
    "stage_of",
    "triage",
]
