"""Three-way alignment benchmark: incumbent JIRR vs Odstrcil-decoupled vs joint-GD.

This is the *adjudication* layer on top of :mod:`benchmarks.runner`. The runner knows
how to score one aligner on one case; this module decides what cases the three methods
are compared on, runs them under conditions that are identical by construction, and
repeats every case over several seeds so that a difference between two methods can be
told apart from a difference between two random draws.

Read ``docs/benchmark_results.md`` for the measured answer. Read this docstring first
for what the answer is and is not allowed to mean.

Four rules the comparison is built around, each of which is a way to get a flattering
but worthless result if you break it
==================================================================================

**1. One case object, three aligners.** A scenario is generated exactly once per
(scenario, seed) and the same :class:`~benchmarks.phantom.BenchmarkCase` -- the same
projections, the same reported angles, the same injected truth -- is handed to every
method, including ``null`` and ``oracle``. Regenerating the case per method would
re-draw the jitter and silently compare methods on different data. This is why the
parallelism is over *cases* and never over methods.

**2. Same preprocessing, same backend, same centre, same warm start.** All three get
the benchmark's own numpy SIRT backend, the *true* rotation centre, and the same
centre-of-mass warm start, because :class:`~benchmarks.runner.EngineAligner` and
:class:`~benchmarks.runner.JointGdAligner` are both wired that way. The one thing that
differs between ``phase_ramp`` and ``phase_ramp_step0`` is whether
:func:`~tktomo.ptycho_align.core.preprocess.remove_phase_ramp` ran first, and it runs
on the case, before any method sees it -- so all three get it or none do. That pair
exists because a benchmark of a ramp-invariant method against a ramp-sensitive one is
only honest if it also shows what happens when the ramp is properly removed first,
which is what the roadmap's step [0] is for.

**3. Seeds, not a seed.** Every scenario runs at ``--seeds`` independent draws and the
tables carry mean +- sample standard deviation. A single draw cannot support a claim
like "0.031 vs 0.033"; :func:`paired_comparison` therefore also reports the *paired*
per-seed difference and a win count, which is the only form in which two methods that
are close should ever be compared.

**4. The two reference rows are not optional.** ``oracle`` must score ~1e-16 px or the
scorer is broken and nothing else on the page means anything; ``null`` must score the
injected RMS, and any method that does not beat it is doing harm. Both are cheap.
They are included in every scenario and printed first.

What the scenarios are for
==========================

======================  =====================================================
``clean_jitter``        the control: per-projection jitter and nothing else.
``phase_ramp``          jitter + a per-projection linear phase ramp and constant
                        offset. **This is the contribution's central claim**: the
                        two ambiguities ptychographic phase retrieval leaves
                        behind, which a gradient-domain registration is blind to
                        and a value-domain one is not.
``phase_ramp_step0``    the same case with ``remove_phase_ramp`` applied first --
                        the fairness control for the row above. Measured, it makes
                        the ramp vanish entirely, which reframes the claim: the
                        gradient trick is a defence for the case where step [0]
                        *cannot* be run.
``ramp_no_vacuum``      exactly that case. The vacuum margin is cropped away after
                        the shift is injected, so the object reaches the frame edge
                        and the plane fit has no vacuum to fit on. Paired with
                        ``ramp_no_vacuum_step0``, which attempts step [0] anyway on
                        a border that is mostly object -- which is what a user who
                        does not check will actually do.
``ramp_sweep_*``        the same claim as a dose-response curve: horizontal
                        recovery against ramp amplitude. A single amplitude is an
                        anecdote; a monotone separation is a mechanism.
``wrong_center``        jitter + a constant rotation-axis offset. Nominally pure
                        gauge, and the scorer removes it from the *score* -- what
                        it cannot remove is that every reprojection in the loop was
                        compared against a volume reconstructed about the wrong
                        axis.
``vertical_drift``      jitter + a smooth vertical drift across the scan: the
                        decoupled direction, where the roadmap says a row-sum
                        beats a reprojection loop.
``nonrigid``            jitter + a smooth zero-mean non-rigid warp. **Every rigid
                        method must fail here.** It is in the suite as a positive
                        control on the benchmark itself: a harness that cannot
                        detect a model failure cannot certify a model success.
======================  =====================================================

Cross-validation with the diagnostics
=====================================

:func:`diagnose_scenario` runs :func:`tktomo.diagnostics.triage` on the *unaligned*
projections of each scenario and records whether the failure mode it names is the one
that was injected. That is two deliverables checking each other: a diagnostic that
fires on the perturbation the benchmark knows it injected is evidence for both, and a
disagreement is a finding rather than a nuisance.

Running it
==========

::

    # the headline, reproducible by anyone, no data and no GPU
    python -m benchmarks.run_three_way --seeds 5 --out benchmarks/results

    # add the synthetic-from-real case (volume path is always the caller's)
    python -m benchmarks.run_three_way --out benchmarks/results \\
        --volume $TOMO/rec_lens1_v4 --volume-name from_real_lens1 \\
        --volume-slices 512:1024 --bin 8 --z-bin 8 \\
        --angles-file $TOMO/lens1_v4_best.h5 --angle-subsample 7 \\
        --pixel-size 74.50973137 --jitter-dy 3.125 --jitter-dx 0.9375 \\
        --margin-volume 17 --iterations 12 --gd-iterations 80

The binning in that second command is not cosmetic. ``benchmarks.phantom`` builds an
explicit sparse system matrix and refuses anything over 40 M non-zeros, which is
``n_angles x n_u^2 x 4``; at bin 8 and a 261 px padded detector that caps the scan at
about 147 angles. Bin less or use more angles and it raises a ``MemoryError`` that
says so rather than thrashing. ``--z-bin`` matches the in-plane binning because
``load_volume`` deliberately leaves the rotation-axis rows alone -- without it the
injected ``dy`` is in unbinned rows while ``dx`` is in binned columns and the
vertical:horizontal asymmetry under test is wrong by the binning factor.

No measured data is read unless ``--volume`` is given, and none is ever written into
this repository: the outputs are JSON summaries and figures of *synthetic* cases.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks import phantom, runner

logger = logging.getLogger("benchmarks.run_three_way")

#: The methods under test, plus the two reference rows and two ablations. Order is the
#: reading order of every table this module writes: references first, incumbent next,
#: challengers, then the ablations that say *which part* of a challenger did the work.
METHOD_ORDER = (
    "null",
    "oracle",
    "jirr",
    "odstrcil",
    "joint_gd",
    "odstrcil_value",
    "odstrcil_novert",
)

#: The three rows a summary table is really about.
HEADLINE_METHODS = ("jirr", "odstrcil", "joint_gd")

#: Our own lens-1 scan measured rms dy 25.0 px and dx 7.5 px on a 1816 px detector.
#: The synthetic cases keep that 3.3:1 vertical:horizontal asymmetry at a scale that
#: fits a 64 px phantom, because it is the asymmetry -- not the absolute amplitude --
#: that decides whether "vertical is the easy direction" is true on this instrument.
BASE_JITTER_DY = 2.5
BASE_JITTER_DX = 0.75


@dataclass(frozen=True)
class Scenario:
    """One benchmark condition: a perturbation, an expectation, and a reason."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    #: Apply ``remove_phase_ramp`` to the case before any aligner sees it (roadmap
    #: step [0]). Applied to the case, so all three methods get it or none do.
    step0_ramp_removal: bool = False
    #: Crop this many rows *and* columns from each side after the shift is injected,
    #: so the object reaches the frame edge and there is no clean vacuum border left
    #: to fit a ramp on. This is the one regime in which the gradient trick is the
    #: only available defence, and it is therefore the fair test of it.
    crop_margin: int = 0
    #: True when *every* rigid method is expected to miss the target. A scenario
    #: flagged this way that quietly passes is a bug in the harness, not a triumph.
    expect_all_fail: bool = False
    #: The failure mode ``tktomo.diagnostics.triage`` ought to name, or None.
    expected_diagnosis: str | None = None
    note: str = ""

    def spec(self, seed: int) -> phantom.PerturbationSpec:
        return phantom.PerturbationSpec(
            jitter_dy_rms=BASE_JITTER_DY,
            jitter_dx_rms=BASE_JITTER_DX,
            seed=seed,
            **self.overrides,
        )


def core_scenarios(*, ramp: float = 1.0) -> list[Scenario]:
    """The five conditions the task asks for, plus the step-0 fairness control."""
    return [
        Scenario(
            name="clean_jitter",
            expected_diagnosis="jitter",
            note="control: per-projection jitter only",
        ),
        Scenario(
            name="phase_ramp",
            overrides={"phase_ramp_rms": ramp, "phase_offset_rms": ramp},
            expected_diagnosis="phase_ramp",
            note="the central claim: the two ambiguities phase retrieval leaves behind",
        ),
        Scenario(
            name="phase_ramp_step0",
            overrides={"phase_ramp_rms": ramp, "phase_offset_rms": ramp},
            step0_ramp_removal=True,
            expected_diagnosis="jitter",
            note="the same ramp, removed first by remove_phase_ramp (roadmap step 0)",
        ),
        # The expectation on these two was originally "phase_ramp" and was CHANGED
        # after the first run, because the diagnostics were right and the label was
        # wrong: with the object reaching all four frame edges the data is locally
        # tomographic, and mode 12 is exactly the condition that
        # ``tktomo.diagnostics.INVALIDATED_BY`` says invalidates the phase-ramp fit.
        # ``diagnose`` and ``triage`` both returned local_tomography at 0.68-0.71 in
        # all six runs. The change is recorded here and in docs/benchmark_results.md
        # rather than made quietly, because moving a target after seeing the arrow
        # land is only defensible if it is visible.
        Scenario(
            name="ramp_no_vacuum",
            overrides={"phase_ramp_rms": ramp, "phase_offset_rms": ramp},
            crop_margin=18,
            expected_diagnosis="local_tomography",
            note="the same ramp with the vacuum border cropped away: step [0] cannot run",
        ),
        Scenario(
            name="ramp_no_vacuum_step0",
            overrides={"phase_ramp_rms": ramp, "phase_offset_rms": ramp},
            crop_margin=18,
            step0_ramp_removal=True,
            expected_diagnosis="local_tomography",
            note="step [0] attempted anyway, on a border that is mostly object",
        ),
        Scenario(
            name="wrong_center",
            overrides={"center_dx": 3.0},
            expected_diagnosis="wrong_center",
            note="rotation axis 3 px from where the loop is told it is",
        ),
        Scenario(
            name="vertical_drift",
            overrides={"drift_dy": 6.0, "drift_shape": "linear"},
            expected_diagnosis="vertical_drift",
            note="the decoupled direction: a smooth 6 px vertical walk",
        ),
        Scenario(
            name="nonrigid",
            overrides={"deformation_px": 1.0, "deformation_scale": 8.0},
            expect_all_fail=True,
            expected_diagnosis="deformation",
            note="positive control: no rigid model can express this",
        ),
    ]


def ramp_sweep_scenarios(amplitudes: Sequence[float]) -> list[Scenario]:
    """The central claim as a dose-response curve rather than a single point."""
    out = []
    for amplitude in amplitudes:
        out.append(
            Scenario(
                name=f"ramp_sweep_{amplitude:g}",
                overrides={"phase_ramp_rms": float(amplitude), "phase_offset_rms": float(amplitude)},
                expected_diagnosis="phase_ramp" if amplitude > 0 else "jitter",
                note=f"phase ramp {amplitude:g} rad RMS across the frame",
            )
        )
    return out


# --------------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Everything that fixes the *size* of a case, held constant across scenarios.

    ``margin`` is pinned rather than derived per spec (which is what
    :func:`benchmarks.phantom.synthetic_case` does by default) so that every scenario
    has the same detector, the same number of vacuum rows for the ramp fit, and the
    same reconstruction cost. Otherwise the drift scenario would quietly get a wider
    frame than the jitter one and the two columns would not be comparable.
    """

    size: int = 64
    n_slices: int = 12
    n_angles: int = 60
    margin: int = 20
    pixel_size_nm: float = 1.0
    iterations: int = 12
    gd_iterations: int = 80
    #: Projector for ``joint_gd`` only. ``"numpy"`` keeps the whole run
    #: dependency-free; ``"astra"`` needs a GPU and is the only way this method is
    #: affordable at realistic scale (its numpy projector is O(n_angles) array
    #: rotations per projection). The two engines are unaffected either way -- they
    #: always use the benchmark's own SIRT backend, so a run that switches this is
    #: still comparing methods on one case with one reconstruction backend for the
    #: metrics, but joint_gd's *internal* projector differs and must be declared.
    gd_projector: str = "numpy"
    with_fsc: bool = True

    def validate(self, scenarios: Sequence[Scenario]) -> None:
        """Refuse to run if the zero margin is too small for the shift about to be
        injected -- the Fourier shift wraps, and a wrapped case is not a case."""
        for scenario in scenarios:
            needed = scenario.spec(0).max_rigid_shift
            if self.margin < needed:
                raise ValueError(
                    f"scenario {scenario.name!r} injects up to {needed:.1f} px but the "
                    f"margin is {self.margin} px; the Fourier shift would wrap. "
                    "Raise --margin."
                )
            if scenario.crop_margin:
                rows = self.n_slices + 2 * self.margin
                if 2 * scenario.crop_margin >= rows:
                    raise ValueError(
                        f"scenario {scenario.name!r} crops {scenario.crop_margin} px from "
                        f"each side of a {rows}-row frame, which removes all of it. "
                        "Raise --slices or --margin."
                    )


def build_case(scenario: Scenario, seed: int, geom: Geometry) -> phantom.BenchmarkCase:
    """Generate one case, and apply roadmap step [0] to it when the scenario says so.

    The step-0 pass is applied to ``case.projections`` *and* to ``case.clean``, so the
    residual metric keeps comparing like with like; the injected truth is untouched
    because subtracting a plane does not move content.
    """
    case = phantom.synthetic_case(
        name=f"{scenario.name}_seed{seed}",
        size=geom.size,
        n_slices=geom.n_slices,
        n_angles=geom.n_angles,
        spec=scenario.spec(seed),
        margin=geom.margin,
        pixel_size_nm=geom.pixel_size_nm,
    )
    border = min(8, geom.margin)
    if scenario.crop_margin:
        # Crop the vacuum margin away AFTER the shift has been injected. Symmetric, so
        # it moves no content and the recorded truth stays exact -- the same argument
        # the harness's own ``truncation_px`` relies on. What it destroys is the clean
        # vacuum border that ``remove_phase_ramp`` fits its plane on, which is the
        # whole point: this is the regime where step [0] cannot be trusted.
        cut = int(scenario.crop_margin)
        n_v, n_u = case.projections.shape[1:]
        if 2 * cut >= min(n_v, n_u):
            raise ValueError(f"crop_margin={cut} removes the whole {n_v}x{n_u} frame")
        case.projections = np.ascontiguousarray(
            case.projections[:, cut : n_v - cut, cut : n_u - cut]
        )
        if case.clean is not None:
            case.clean = np.ascontiguousarray(case.clean[:, cut : n_v - cut, cut : n_u - cut])
        border = max(1, min(3, (min(n_v, n_u) - 2 * cut) // 4))
        case.metadata["crop_margin_px"] = cut
    if scenario.step0_ramp_removal:
        from tktomo.ptycho_align.core.preprocess import remove_phase_ramp

        case.projections = np.asarray(
            remove_phase_ramp(case.projections, border=border), dtype=np.float64
        )
        if case.clean is not None:
            case.clean = np.asarray(
                remove_phase_ramp(case.clean, border=border), dtype=np.float64
            )
        case.metadata["step0_ramp_removal"] = {"border_px": border}
    return case


def aligners_for(geom: Geometry, only: Sequence[str] | None = None) -> list[Any]:
    """The five rows plus two ablations, constructed identically for every case.

    ``jirr`` and ``odstrcil`` are the same adapter with a different engine class, so
    the only thing that differs between them is the algorithm. ``joint_gd`` needs its
    own adapter because it runs a multi-resolution schedule; it is given the same
    warm start and the same numpy projector.

    The two ablations exist because "odstrcil beat jirr" is not a claim about the
    gradient trick -- ``OdstrcilEngine`` differs from ``AlignmentEngine`` in *two*
    ways at once (a vertical stage that never reconstructs, and a registration on the
    phase gradient), and a two-engine comparison cannot say which one paid.

    ``odstrcil_value`` is the same engine with ``GradientConfig.domain="value"``: the
    registration goes back to comparing the phase itself and everything else -- the
    vertical stage, the chunking, the update conditioning, the divergence guard -- is
    bit-for-bit identical. The difference between the ``odstrcil`` and
    ``odstrcil_value`` rows is the gradient trick and nothing else. That knob exists
    in ``OdstrcilConfig`` precisely so this comparison can be made; using it is the
    difference between measuring a claim and illustrating one.

    ``odstrcil_novert`` turns stage 1 off and keeps the centre-of-mass warm start, so
    the vertical axis is solved the way the incumbent solves it. The difference
    between ``odstrcil`` and ``odstrcil_novert`` is the vertical mass stage alone.

    ``only`` keeps just the named rows. It exists for one reason: ``joint_gd``'s numpy
    projector is O(n_angles) array rotations per projection and costs ~80 s per
    iteration on a 130 x 98 x 261 stack, so a realistic-size case with it in takes
    hours and the same case without it takes minutes. Dropping it changes nothing
    about the other rows -- each aligner sees the same case object either way -- but a
    run that used ``only`` must say so, because a missing row is not a zero.
    """
    from tktomo.ptycho_align.core.gradient import GradientConfig
    from tktomo.ptycho_align.core.odstrcil import OdstrcilConfig

    built = [
        runner.NullAligner(),
        runner.OracleAligner(),
        runner.JirrAligner(iterations=geom.iterations),
        runner.OdstrcilAligner(iterations=geom.iterations),
        runner.JointGdAligner(
            iterations_per_stage=geom.gd_iterations, projector=geom.gd_projector
        ),
        runner.OdstrcilAligner(
            name="odstrcil_value",
            iterations=geom.iterations,
            engine_kwargs={
                "odstrcil": OdstrcilConfig(gradient=GradientConfig(domain="value"))
            },
        ),
        runner.OdstrcilAligner(
            name="odstrcil_novert",
            iterations=geom.iterations,
            engine_kwargs={"odstrcil": OdstrcilConfig(run_vertical=False)},
        ),
    ]
    if only is None:
        return built
    wanted = set(only)
    unknown = sorted(wanted - {a.name for a in built})
    if unknown:
        raise ValueError(
            f"unknown aligner name(s) {unknown}; have {[a.name for a in built]}"
        )
    return [a for a in built if a.name in wanted]


# --------------------------------------------------------------------------------
# One case, all methods -- the unit of parallelism
# --------------------------------------------------------------------------------


def run_case(task: tuple[Scenario, int, Geometry, tuple[str, ...] | None]) -> dict[str, Any]:
    """Run every method on one (scenario, seed) and return a JSON-ready record.

    Runs in a worker process. Every failure is caught and returned as data: one
    scenario blowing up must not cost the other twenty-nine their results.
    """
    scenario, seed, geom, only = task
    started = time.time()
    try:
        case = build_case(scenario, seed, geom)
        report = runner.run_benchmark(
            case,
            aligners_for(geom, only),
            with_residual=True,
            with_fsc=geom.with_fsc,
        )
        report.case["truth_dy"] = case.truth.dy.tolist()
        report.case["truth_dx"] = case.truth.dx.tolist()
        payload = report.to_dict()
    except Exception as exc:  # noqa: BLE001 - a dead case is data, not a crash
        logger.exception("case %s seed %d failed", scenario.name, seed)
        payload = {"case": {"name": f"{scenario.name}_seed{seed}"}, "results": [], "error": f"{type(exc).__name__}: {exc}"}
    payload["scenario"] = {
        "name": scenario.name,
        "seed": seed,
        "overrides": scenario.overrides,
        "step0_ramp_removal": scenario.step0_ramp_removal,
        "expect_all_fail": scenario.expect_all_fail,
        "expected_diagnosis": scenario.expected_diagnosis,
        "note": scenario.note,
    }
    payload["wallclock_case_s"] = time.time() - started
    return payload


def object_contrast(geom: Geometry) -> float:
    """Peak-to-peak of the *clean* projections, in radians.

    The phase ramp is injected in absolute radians, which is meaningless on its own:
    1 rad of ramp is nothing on an object that spans 100 rad and fatal on one that
    spans 2. Every ramp amplitude in the tables is therefore also quoted as a fraction
    of this number, which is the only form in which it transfers to another dataset.
    """
    case = phantom.synthetic_case(
        size=geom.size,
        n_slices=geom.n_slices,
        n_angles=geom.n_angles,
        spec=phantom.PerturbationSpec(seed=0),
        margin=geom.margin,
    )
    return float(np.ptp(np.asarray(case.clean)))


# --------------------------------------------------------------------------------
# Aggregation across seeds
# --------------------------------------------------------------------------------

#: Scalars lifted out of each row and averaged across seeds.
_SCALARS = (
    ("rms_dy", ("shift_recovery", "rms_dy")),
    ("rms_dx", ("shift_recovery", "rms_dx")),
    ("max_dy", ("shift_recovery", "max_dy")),
    ("max_dx", ("shift_recovery", "max_dx")),
    ("rms_dy_raw", ("shift_recovery", "rms_dy_raw")),
    ("rms_dx_raw", ("shift_recovery", "rms_dx_raw")),
    ("target_px", ("shift_recovery", "target_px")),
    ("injected_rms_dy", ("shift_recovery", "injected_rms_dy")),
    ("injected_rms_dx", ("shift_recovery", "injected_rms_dx")),
    ("wallclock_s", ("wallclock_s",)),
    ("iterations", ("iterations",)),
    ("residual_total", ("reprojection_residual", "total")),
    ("residual_lag1", ("reprojection_residual", "lag1")),
    ("residual_peakiness", ("reprojection_residual", "peakiness")),
    ("residual_vs_clean", ("reprojection_residual", "vs_clean")),
    ("fsc_resolution_px", ("fsc", "resolution_px")),
    ("plateau_iteration", ("plateau", "iteration")),
)


def _dig(row: dict[str, Any], path: tuple[str, ...]) -> float:
    node: Any = row
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return math.nan
        node = node[key]
    if node is None:
        return math.nan
    try:
        return float(node)
    except (TypeError, ValueError):
        return math.nan


def _stat(values: Sequence[float]) -> dict[str, float]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n": int(finite.size),
    }


def aggregate(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-seed payloads into per-(scenario, method) statistics."""
    by_scenario: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        scenario = payload["scenario"]["name"]
        entry = by_scenario.setdefault(
            scenario,
            {
                "scenario": scenario,
                "note": payload["scenario"]["note"],
                "expect_all_fail": payload["scenario"]["expect_all_fail"],
                "expected_diagnosis": payload["scenario"]["expected_diagnosis"],
                "overrides": payload["scenario"]["overrides"],
                "step0_ramp_removal": payload["scenario"]["step0_ramp_removal"],
                "seeds": [],
                "case_shape": payload.get("case", {}).get("shape"),
                "methods": {},
            },
        )
        entry["seeds"].append(payload["scenario"]["seed"])
        for row in payload.get("results", []):
            method = entry["methods"].setdefault(
                row["name"], {"status": [], "message": [], "samples": {k: [] for k, _ in _SCALARS},
                              "sign_flipped": [], "per_seed_rms_dy": {}, "per_seed_rms_dx": {}}
            )
            method["status"].append(row.get("status"))
            method["message"].append(row.get("message"))
            extras = row.get("extras", {}) or {}
            # A run cut short by the engine's own runaway guard is not a bad score, it
            # is a refusal, and the two must not be averaged together silently.
            method.setdefault("runaway", []).append(bool(extras.get("runaway")))
            method.setdefault("diverging", []).append(bool(extras.get("diverging")))
            vertical = extras.get("vertical_stage") or {}
            method.setdefault("vertical_truncation_warning", []).append(
                vertical.get("truncation_reason") is not None
            )
            method["sign_flipped"].append(
                bool(row.get("sign_check", {}).get("flipped_is_better", False))
            )
            for key, path in _SCALARS:
                method["samples"][key].append(_dig(row, path))
            seed = payload["scenario"]["seed"]
            method["per_seed_rms_dy"][str(seed)] = _dig(row, ("shift_recovery", "rms_dy"))
            method["per_seed_rms_dx"][str(seed)] = _dig(row, ("shift_recovery", "rms_dx"))

    for entry in by_scenario.values():
        for method, data in entry["methods"].items():
            stats = {key: _stat(values) for key, values in data["samples"].items()}
            target = stats["target_px"]["mean"]
            data["stats"] = stats
            data["meets_target"] = bool(
                np.isfinite(target)
                and np.isfinite(stats["rms_dy"]["mean"])
                and np.isfinite(stats["rms_dx"]["mean"])
                and stats["rms_dy"]["max"] <= target
                and stats["rms_dx"]["max"] <= target
            )
            data["meets_target_mean"] = bool(
                np.isfinite(target)
                and stats["rms_dy"]["mean"] <= target
                and stats["rms_dx"]["mean"] <= target
            )
            data["ok"] = all(s == "ok" for s in data["status"])
            data["n_runaway"] = int(sum(data.get("runaway", [])))
            data["n_diverging"] = int(sum(data.get("diverging", [])))
            data["n_vertical_truncation_warning"] = int(
                sum(data.get("vertical_truncation_warning", []))
            )
            del data["samples"]
    return by_scenario


def paired_comparison(
    summary: dict[str, Any], left: str, right: str, axis: str = "rms_dx"
) -> dict[str, dict[str, Any]]:
    """Per-seed paired difference between two methods, per scenario.

    Two methods measured on the *same* draws are compared by pairing, never by
    comparing two means with overlapping spreads. ``wins`` counts the seeds on which
    ``left`` was better; ``ratio`` is the geometric mean of ``left/right``, which is
    the scale-free version of "how much better".
    """
    key = {"rms_dy": "per_seed_rms_dy", "rms_dx": "per_seed_rms_dx"}[axis]
    out: dict[str, dict[str, Any]] = {}
    for name, entry in summary.items():
        a = entry["methods"].get(left, {}).get(key, {})
        b = entry["methods"].get(right, {}).get(key, {})
        seeds = sorted(set(a) & set(b), key=int)
        pairs = [(a[s], b[s]) for s in seeds if np.isfinite(a[s]) and np.isfinite(b[s])]
        if not pairs:
            continue
        diffs = np.array([x - y for x, y in pairs])
        ratios = np.array([x / y for x, y in pairs if y > 0])
        out[name] = {
            "n_seeds": len(pairs),
            "mean_diff": float(diffs.mean()),
            "std_diff": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
            "left_wins": int(np.sum(diffs < 0)),
            "geo_mean_ratio": float(np.exp(np.mean(np.log(ratios)))) if ratios.size else math.nan,
            "decisive": bool(diffs.size > 1 and abs(diffs.mean()) > 2.0 * diffs.std(ddof=1)),
        }
    return out


# --------------------------------------------------------------------------------
# Diagnostics cross-check
# --------------------------------------------------------------------------------


def diagnose_scenario(scenario: Scenario, seed: int, geom: Geometry) -> dict[str, Any]:
    """Run ``tktomo.diagnostics.triage`` on the unaligned stack of one scenario.

    The point is not to produce a diagnosis; it is to check the diagnosis against a
    perturbation we know we injected. A probe suite that names the right cause on a
    case with known ground truth is evidence for the probe suite *and* for the case
    generator. Returns a record with ``matches_expected`` so a disagreement shows up
    in the table rather than in a footnote.
    """
    record: dict[str, Any] = {
        "scenario": scenario.name,
        "seed": seed,
        "expected": scenario.expected_diagnosis,
    }
    try:
        from tktomo.diagnostics import diagnose, triage
    except ImportError as exc:
        record["status"] = "skipped"
        record["message"] = f"tktomo.diagnostics unavailable: {exc}"
        return record

    try:
        case = build_case(scenario, seed, geom)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(diagnose_case(case, scenario.expected_diagnosis))
    return record


def diagnose_case(case: phantom.BenchmarkCase, expected: str | None) -> dict[str, Any]:
    """Triage one already-built case. Split out so the from-real case gets it too."""
    record: dict[str, Any] = {"expected": expected}
    try:
        from tktomo.diagnostics import diagnose, triage
    except ImportError as exc:
        return {**record, "status": "skipped", "message": f"tktomo.diagnostics: {exc}"}
    try:
        theta_deg = np.degrees(case.angles)
        verdict = triage(case.projections, theta_deg)
        survey = diagnose(case.projections, theta_deg)
    except Exception as exc:  # noqa: BLE001
        return {**record, "status": "error", "message": f"{type(exc).__name__}: {exc}"}

    def _mode(finding: Any) -> str | None:
        if finding is None:
            return None
        mode = getattr(finding, "mode", None)
        return getattr(mode, "value", str(mode)) if mode is not None else None

    top = getattr(verdict, "top", None)
    record["status"] = "ok"
    record["triage_top"] = _mode(top)
    record["triage_confidence"] = float(getattr(top, "confidence", math.nan)) if top else math.nan
    record["triage_fix"] = getattr(top, "fix", None) if top else None
    record["triage_evidence"] = runner._jsonable(getattr(top, "evidence", {})) if top else {}
    record["triage_coverage"] = float(getattr(verdict, "coverage", math.nan))
    record["survey_ranked"] = [
        {
            "mode": _mode(f),
            "confidence": float(getattr(f, "confidence", math.nan)),
        }
        for f in list(getattr(survey, "findings", []))[:5]
    ]
    record["survey_modes"] = [entry["mode"] for entry in record["survey_ranked"]]
    record["matches_expected"] = bool(expected is not None and record["triage_top"] == expected)
    record["expected_in_survey"] = bool(expected is not None and expected in record["survey_modes"])
    return record


# --------------------------------------------------------------------------------
# Tables and figures
# --------------------------------------------------------------------------------


def _fmt(value: float, places: int = 3) -> str:
    return "  n/a" if not np.isfinite(value) else f"{value:.{places}f}"


def scenario_table(summary: dict[str, Any], order: Sequence[str] = METHOD_ORDER) -> str:
    """The headline table: one block per scenario, one row per method."""
    lines: list[str] = []
    for name, entry in summary.items():
        target = math.nan
        for method in entry["methods"].values():
            target = method["stats"]["target_px"]["mean"]
            if np.isfinite(target):
                break
        lines.append("")
        lines.append(f"### {name}   ({entry['note']})")
        lines.append(
            f"    seeds={sorted(entry['seeds'])}  shape={entry['case_shape']}  "
            f"target={_fmt(target)} px"
        )
        header = (
            f"{'method':<16}{'rms dy':>16}{'rms dx':>16}{'max dy':>9}{'max dx':>9}"
            f"{'1/3 vox':>9}{'FSC px':>9}{'resid':>8}{'lag1':>7}{'iters':>7}{'sec':>8}"
            f"{'abort':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for method in order:
            data = entry["methods"].get(method)
            if data is None:
                continue
            s = data["stats"]
            if not data["ok"]:
                lines.append(f"{method:<16}  {data['status'][0]}: {str(data['message'][0])[:70]}")
                continue
            dy = f"{_fmt(s['rms_dy']['mean'])}+-{_fmt(s['rms_dy']['std'])}"
            dx = f"{_fmt(s['rms_dx']['mean'])}+-{_fmt(s['rms_dx']['std'])}"
            flag = "yes" if data["meets_target"] else "NO"
            aborts = data.get("n_runaway", 0)
            lines.append(
                f"{method:<16}{dy:>16}{dx:>16}"
                f"{_fmt(s['max_dy']['mean']):>9}{_fmt(s['max_dx']['mean']):>9}"
                f"{flag:>9}{_fmt(s['fsc_resolution_px']['mean'], 2):>9}"
                f"{_fmt(s['residual_total']['mean']):>8}{_fmt(s['residual_lag1']['mean'], 2):>7}"
                f"{s['iterations']['mean']:>7.1f}{_fmt(s['wallclock_s']['mean'], 1):>8}"
                f"{(str(aborts) + '/' + str(len(data['status']))) if aborts else '-':>7}"
            )
    lines.append("")
    lines.append(
        "px, gauge-removed (dy: {1}; dx: {1, sin, cos}). mean +- sample sd over seeds. "
        "'1/3 vox' is yes only when the WORST seed meets the target. 'abort' counts the "
        "seeds the engine's own runaway guard stopped early -- a refusal, not a score, "
        "and the numbers on that row are what it had reached when it stopped."
    )
    return "\n".join(lines)


def summary_figure(
    summary: dict[str, Any],
    path: str | Path,
    *,
    sweep_prefix: str = "ramp_sweep_",
    contrast: float = math.nan,
):
    """Six panels. Look at it before you describe it.

    1. horizontal recovery per scenario per method, log scale, target line;
    2. vertical recovery, same layout -- the two axes separate different methods;
    3. the ramp sweep: horizontal error against ramp amplitude, the central claim;
    4. the two ablations, as ratios: gradient-vs-value in ``dx`` and
       vertical-stage-vs-none in ``dy``. Below 1 means the feature paid;
    5. FSC against shift error, the negative control: a flat cloud means the FSC
       certificate is independent of how well the stack was actually aligned;
    6. wall clock, because an accuracy table with no cost column is an advertisement.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("matplotlib is missing; no figure written")
        return None

    core = [n for n in summary if not n.startswith(sweep_prefix)]
    sweep = sorted(
        (n for n in summary if n.startswith(sweep_prefix)),
        key=lambda n: float(n[len(sweep_prefix):]),
    )
    methods = [m for m in HEADLINE_METHODS if any(m in summary[n]["methods"] for n in summary)]
    sweep_methods = [
        m
        for m in ("jirr", "odstrcil", "odstrcil_value", "joint_gd")
        if any(m in summary[n]["methods"] for n in summary)
    ]
    colors = {
        "jirr": "#444444",
        "odstrcil": "#c1121f",
        "odstrcil_value": "#f4a261",
        "odstrcil_novert": "#7b2cbf",
        "joint_gd": "#0466c8",
        "null": "#bbbbbb",
    }

    fig, axes = plt.subplots(2, 3, figsize=(20.0, 10.0))
    target = math.nan
    for entry in summary.values():
        for data in entry["methods"].values():
            if np.isfinite(data["stats"]["target_px"]["mean"]):
                target = data["stats"]["target_px"]["mean"]
                break
        if np.isfinite(target):
            break

    for ax, axis, label in (
        (axes[0][0], "rms_dx", "horizontal  rms dx"),
        (axes[0][1], "rms_dy", "vertical  rms dy"),
    ):
        width = 0.8 / (len(methods) + 1)
        x = np.arange(len(core))
        for k, method in enumerate(methods + ["null"]):
            means = [summary[n]["methods"].get(method, {}).get("stats", {}).get(axis, {}).get("mean", math.nan) for n in core]
            errs = [summary[n]["methods"].get(method, {}).get("stats", {}).get(axis, {}).get("std", 0.0) for n in core]
            ax.bar(
                x + (k - len(methods) / 2) * width,
                means,
                width,
                yerr=errs,
                capsize=2,
                label=method,
                color=colors.get(method, "#888888"),
                alpha=0.9 if method != "null" else 0.45,
            )
        ax.axhline(target, color="green", ls="--", lw=1.2, label=f"1/3 voxel = {target:.3f} px")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(core, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(f"{label}  (px, gauge-removed)")
        ax.set_title(label + " by scenario")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)

    ax = axes[0][2]
    if sweep:
        amps = [float(n[len(sweep_prefix):]) for n in sweep]
        for method in sweep_methods:
            means = [summary[n]["methods"].get(method, {}).get("stats", {}).get("rms_dx", {}).get("mean", math.nan) for n in sweep]
            errs = [summary[n]["methods"].get(method, {}).get("stats", {}).get("rms_dx", {}).get("std", 0.0) for n in sweep]
            ax.errorbar(amps, means, yerr=errs, marker="o", capsize=3, label=method, color=colors.get(method))
        ax.axhline(target, color="green", ls="--", lw=1.2, label="1/3 voxel")
        suffix = (
            f"   [object contrast {contrast:.1f} rad p-p]" if np.isfinite(contrast) else ""
        )
        ax.set_xlabel("injected phase ramp + offset (rad RMS across the frame)" + suffix)
        ax.set_ylabel("rms dx (px)")
        ax.set_yscale("log")
        ax.set_title("the central claim: horizontal error vs residual phase ramp")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no ramp sweep in this run", ha="center", va="center")

    # -- panel 4: the ablations, as ratios ------------------------------------------
    ax = axes[1][0]
    all_names = core + sweep
    ablations = (
        ("odstrcil", "odstrcil_value", "rms_dx", "gradient vs value  (dx)", "#c1121f"),
        ("odstrcil", "odstrcil_novert", "rms_dy", "vertical stage vs none  (dy)", "#7b2cbf"),
    )
    width = 0.38
    x = np.arange(len(all_names))
    drew = False
    for k, (left, right, axis, label, color) in enumerate(ablations):
        ratios = []
        for name in all_names:
            a = summary[name]["methods"].get(left, {}).get("stats", {}).get(axis, {}).get("mean", math.nan)
            b = summary[name]["methods"].get(right, {}).get("stats", {}).get(axis, {}).get("mean", math.nan)
            ratios.append(a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else math.nan)
        if np.any(np.isfinite(ratios)):
            drew = True
        ax.bar(x + (k - 0.5) * width, ratios, width, label=label, color=color, alpha=0.9)
    if drew:
        ax.axhline(1.0, color="black", lw=1.2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("error ratio  (< 1 = the feature paid)")
        ax.set_title("ablations: same engine, one knob changed")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no ablation rows in this run", ha="center", va="center")

    ax = axes[1][1]
    xs, ys, cs, labels = [], [], [], []
    for name, entry in summary.items():
        for method, data in entry["methods"].items():
            fsc = data["stats"].get("fsc_resolution_px", {}).get("mean", math.nan)
            err = math.hypot(
                data["stats"]["rms_dy"]["mean"] if np.isfinite(data["stats"]["rms_dy"]["mean"]) else 0.0,
                data["stats"]["rms_dx"]["mean"] if np.isfinite(data["stats"]["rms_dx"]["mean"]) else 0.0,
            )
            if np.isfinite(fsc) and np.isfinite(err) and err > 0:
                xs.append(err)
                ys.append(fsc)
                cs.append(colors.get(method, "#888888"))
                labels.append(method)
    if xs:
        ax.scatter(xs, ys, c=cs, s=26, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("total shift-recovery error  sqrt(dy^2+dx^2)  (px)")
        ax.set_ylabel("split-half FSC resolution (px)")
        ax.set_title("FSC cannot see alignment error (flat cloud = blind)")
        ax.grid(alpha=0.3)
        # The narrowness of the y range is the entire point, and a reader who does not
        # check the axis limits will misread this panel as a scatter of real variation.
        spread = (max(ys) - min(ys)) / float(np.median(ys)) * 100.0
        span = max(xs) / min(xs)
        ax.text(
            0.03,
            0.04,
            f"x spans {span:,.0f}x in alignment error;\ny spans {spread:.2f}% in FSC",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
        )
        for method in set(labels):
            ax.scatter([], [], c=colors.get(method, "#888888"), label=method)
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no FSC in this run (--no-fsc)", ha="center", va="center")

    # -- panel 6: cost ---------------------------------------------------------------
    ax = axes[1][2]
    cost_methods = [m for m in METHOD_ORDER if m not in ("null", "oracle")]
    seconds, names = [], []
    for method in cost_methods:
        values = [
            summary[n]["methods"].get(method, {}).get("stats", {}).get("wallclock_s", {}).get("mean", math.nan)
            for n in core
        ]
        values = [v for v in values if np.isfinite(v)]
        if values:
            seconds.append(float(np.mean(values)))
            names.append(method)
    if seconds:
        ax.barh(names, seconds, color=[colors.get(n, "#888888") for n in names])
        ax.set_xscale("log")
        ax.set_xlabel("mean wall clock per case (s), benchmark numpy backend, 1 core")
        ax.set_title("cost")
        ax.grid(axis="x", alpha=0.3)
        for i, value in enumerate(seconds):
            ax.text(value, i, f" {value:.1f}s", va="center", fontsize=8)
    else:
        ax.set_axis_off()

    fig.suptitle("TKtomo three-way alignment benchmark (synthetic phantom, ground truth)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def collect_figure(
    directories: Sequence[Path],
    x_values: Sequence[float],
    path: str | Path,
    *,
    x_label: str = "sweep parameter",
    methods: Sequence[str] = ("jirr", "odstrcil", "odstrcil_value", "joint_gd"),
):
    """Plot ``rms dx`` against a parameter that varies *between separate runs*.

    :func:`summary_figure`'s sweep panel can only show a parameter that varies
    between scenarios inside one run. Some experiments -- the no-vacuum ramp series,
    for instance -- vary something that is a command-line argument rather than a
    scenario field, so they come out as several result directories. This reads their
    summaries back and puts them on one axis, which is the only honest way to look at
    a dose-response curve.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - environment dependent
        return None

    bundles = [json.loads((Path(d) / "three_way_summary.json").read_text()) for d in directories]
    scenarios = list(bundles[0]["summary"])
    colors = {
        "jirr": "#444444",
        "odstrcil": "#c1121f",
        "odstrcil_value": "#f4a261",
        "odstrcil_novert": "#7b2cbf",
        "joint_gd": "#0466c8",
    }
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 5.0), squeeze=False)
    target = math.nan
    for ax, scenario in zip(axes[0], scenarios):
        for method in methods:
            means, errs = [], []
            for bundle in bundles:
                stats = bundle["summary"][scenario]["methods"].get(method, {}).get("stats", {})
                means.append(stats.get("rms_dx", {}).get("mean", math.nan))
                errs.append(stats.get("rms_dx", {}).get("std", 0.0))
                if np.isfinite(stats.get("target_px", {}).get("mean", math.nan)):
                    target = stats["target_px"]["mean"]
            ax.errorbar(
                x_values, means, yerr=errs, marker="o", capsize=3, label=method,
                color=colors.get(method, "#888888"),
            )
        if np.isfinite(target):
            ax.axhline(target, color="green", ls="--", lw=1.2, label="1/3 voxel")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("rms dx (px, gauge-removed)")
        ax.set_title(scenario)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------
# Synthetic-from-real
# --------------------------------------------------------------------------------


def build_volume_case(args: argparse.Namespace, geom: Geometry) -> phantom.BenchmarkCase:
    """Forward-project a caller-supplied reconstruction and inject known truth.

    Nothing about the path is defaulted and nothing measured is written back out --
    only the scores. The volume is negated when ``--invert`` is set, which is the
    default, because our phase convention is air ~ 0 and material *negative* while
    every mass-based estimator in the repo (centre of mass, vertical mass profile)
    needs the object to be the positive part.
    """
    slices = None
    if args.volume_slices:
        start, _, stop = args.volume_slices.partition(":")
        slices = slice(int(start), int(stop))
    volume = phantom.load_volume(
        args.volume, dataset=args.volume_dataset, slices=slices, bin_factor=args.bin
    )
    if args.invert:
        volume = -volume
    volume = np.clip(volume, 0.0, None)
    if args.z_bin > 1:
        # ``load_volume``'s bin_factor deliberately leaves the rotation-axis rows
        # alone. For a benchmark we want an isotropically binned volume, or the
        # injected vertical shift is in unbinned rows while the horizontal one is in
        # binned columns and the 3.3:1 asymmetry we are trying to reproduce turns into
        # 26:1 by accident.
        n_z = volume.shape[0] - volume.shape[0] % args.z_bin
        volume = volume[:n_z].reshape(n_z // args.z_bin, args.z_bin, *volume.shape[1:]).mean(axis=1)

    if args.angles_file:
        angles = phantom.load_angles(
            args.angles_file, dataset=args.angles_dataset, subsample=args.angle_subsample
        )
    else:
        angles = np.deg2rad(np.linspace(0.0, 180.0, geom.n_angles, endpoint=False))

    spec = phantom.PerturbationSpec(
        jitter_dy_rms=args.jitter_dy,
        jitter_dx_rms=args.jitter_dx,
        phase_ramp_rms=args.volume_ramp,
        phase_offset_rms=args.volume_ramp,
        seed=args.seed,
    )
    margin = args.margin_volume or (int(math.ceil(spec.max_rigid_shift)) + 4)
    return phantom.volume_case(
        volume,
        angles,
        name=args.volume_name,
        spec=spec,
        margin=margin,
        pixel_size_nm=args.pixel_size * args.bin,
        metadata={
            "source": "user-supplied volume (not committed)",
            "bin_factor": args.bin,
            "z_bin_factor": args.z_bin,
            "volume_shape": list(volume.shape),
            "n_angles": int(angles.size),
            "angle_subsample": args.angle_subsample,
            "angle_span_deg": float(np.degrees(np.ptp(angles))),
            "inverted": bool(args.invert),
            "isotropic_binning": bool(args.bin == args.z_bin),
        },
    )


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def _worker_count(requested: int | None) -> int:
    if requested:
        return max(1, requested)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm))
    return max(1, (os.cpu_count() or 2) // 2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument("--seeds", type=int, default=5, help="independent draws per scenario")
    parser.add_argument("--seed0", type=int, default=0, help="first seed")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--angles", type=int, default=60)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--gd-iterations", type=int, default=80)
    parser.add_argument(
        "--gd-projector",
        default="numpy",
        choices=("numpy", "astra"),
        help="joint_gd's internal projector; astra needs a GPU and is declared in the report",
    )
    parser.add_argument("--ramp", type=float, default=1.0, help="ramp amplitude for the phase_ramp scenario")
    parser.add_argument("--sweep", default="0.25,0.5,1,2,4", help="ramp amplitudes, comma separated ('' disables)")
    parser.add_argument("--sweep-seeds", type=int, default=3)
    parser.add_argument("--no-fsc", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--scenarios", default=None, help="comma-separated subset of core scenario names")
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated subset of aligner rows (default: all seven). A run that uses this must say so in its write-up: a missing row is not a zero.",
    )
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="redraw the figure and table from an existing three_way_summary.json in --out",
    )
    parser.add_argument(
        "--collect",
        default=None,
        help="comma-separated result directories to put on one axis (see collect_figure)",
    )
    parser.add_argument("--collect-x", default=None, help="comma-separated x values for --collect")
    parser.add_argument("--collect-label", default="sweep parameter")
    parser.add_argument("--collect-out", default="collected.png")

    real = parser.add_argument_group("synthetic-from-real (a volume path YOU supply)")
    real.add_argument("--volume", default=None)
    real.add_argument("--volume-name", default="from_real")
    real.add_argument("--volume-dataset", default="/tomogram/data")
    real.add_argument("--volume-slices", default=None, help="START:STOP")
    real.add_argument("--bin", type=int, default=1, help="in-plane binning of the volume")
    real.add_argument("--z-bin", type=int, default=1, help="binning along the rotation axis")
    real.add_argument(
        "--voxel-nm",
        type=float,
        default=None,
        help="target voxel for the 1/3-voxel gate; defaults to the (binned) detector pixel",
    )
    real.add_argument("--angles-file", default=None)
    real.add_argument("--angles-dataset", default="exchange/theta")
    real.add_argument("--angle-subsample", type=int, default=1)
    real.add_argument("--pixel-size", type=float, default=1.0, help="nm per unbinned pixel")
    real.add_argument("--jitter-dy", type=float, default=6.25)
    real.add_argument("--jitter-dx", type=float, default=1.875)
    real.add_argument("--volume-ramp", type=float, default=0.0)
    real.add_argument("--margin-volume", type=int, default=None)
    real.add_argument("--seed", type=int, default=0)
    real.add_argument("--no-invert", dest="invert", action="store_false")
    real.set_defaults(invert=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)

    if args.collect:
        dirs = [Path(d.strip()) for d in args.collect.split(",") if d.strip()]
        xs = [float(v) for v in (args.collect_x or "").split(",") if v.strip()]
        if len(xs) != len(dirs):
            parser.error("--collect-x must give one value per --collect directory")
        figure = collect_figure(
            dirs, xs, out / args.collect_out, x_label=args.collect_label
        )
        print(f"wrote {figure}")
        return 0

    if args.figure_only:
        # The runs are the expensive part and the plotting is the part that gets
        # fiddled with, so they are deliberately separable.
        bundle = json.loads((out / "three_way_summary.json").read_text())
        table = scenario_table(bundle["summary"])
        print(table)
        (out / "three_way_table.txt").write_text(table + "\n")
        figure = summary_figure(
            bundle["summary"],
            out / "three_way_summary.png",
            contrast=bundle.get("object_contrast_rad", math.nan),
        )
        print(f"wrote {figure}")
        return 0

    geom = Geometry(
        size=args.size,
        n_slices=args.slices,
        n_angles=args.angles,
        margin=args.margin,
        iterations=args.iterations,
        gd_iterations=args.gd_iterations,
        gd_projector=args.gd_projector,
        with_fsc=not args.no_fsc,
    )

    # ---- the synthetic-from-real branch is a single case, run in this process ----
    if args.volume:
        case = build_volume_case(args, geom)
        logger.info("volume case %s: %s", case.name, case.projections.shape)
        selected = tuple(m.strip() for m in args.methods.split(",") if m.strip()) if args.methods else None
        report = runner.run_benchmark(
            case,
            aligners_for(geom, selected),
            with_residual=True,
            with_fsc=geom.with_fsc,
            voxel_nm=args.voxel_nm,
        )
        report.case["methods_requested"] = list(selected) if selected else "all"
        report.case["truth_dy"] = case.truth.dy.tolist()
        report.case["truth_dx"] = case.truth.dx.tolist()
        if not args.no_diagnostics:
            report.case["diagnostics"] = diagnose_case(case, None)
        json_path = report.to_json(out / f"{case.name}.json")
        figure = runner.comparison_figure(report, out / f"{case.name}.png")
        print(report.table())
        if not args.no_diagnostics:
            record = report.case["diagnostics"]
            print(
                f"\ntriage on the unaligned stack: {record.get('triage_top')} "
                f"(confidence {record.get('triage_confidence', float('nan')):.2f}); "
                f"survey ranked {record.get('survey_modes')}"
            )
        print(f"\nwrote {json_path}")
        if figure:
            print(f"wrote {figure}")
        return 0

    # ---- the synthetic sweep ----
    scenarios = core_scenarios(ramp=args.ramp)
    if args.scenarios:
        wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s.name in wanted]
    sweep = [float(v) for v in args.sweep.split(",") if v.strip()] if args.sweep else []
    sweep_scenarios = ramp_sweep_scenarios(sweep)
    geom.validate(scenarios + sweep_scenarios)

    only = tuple(m.strip() for m in args.methods.split(",") if m.strip()) if args.methods else None
    tasks: list[tuple[Scenario, int, Geometry, tuple[str, ...] | None]] = []
    for scenario in scenarios:
        for k in range(args.seeds):
            tasks.append((scenario, args.seed0 + k, geom, only))
    for scenario in sweep_scenarios:
        for k in range(args.sweep_seeds):
            tasks.append((scenario, args.seed0 + k, geom, only))

    workers = min(_worker_count(args.workers), len(tasks))
    logger.info("%d cases over %d worker(s)", len(tasks), workers)
    started = time.time()
    if workers == 1:
        payloads = [run_case(task) for task in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            payloads = pool.map(run_case, tasks, chunksize=1)
    elapsed = time.time() - started
    logger.info("all cases done in %.1f s", elapsed)

    for payload in payloads:
        name = payload["case"].get("name", "case")
        # Gzipped: a full per-case record is ~128 kB of JSON and there are dozens of
        # them, which is more than a source repository should carry uncompressed.
        # ``gzip.open(...).read()`` round-trips it and every plotting tool here reads
        # the aggregate summary instead.
        with gzip.open(out / "cases" / f"{name}.json.gz", "wt") as handle:
            json.dump(payload, handle, indent=1)

    summary = aggregate(payloads)

    diagnostics: list[dict[str, Any]] = []
    if not args.no_diagnostics:
        for scenario in scenarios:
            logger.info("diagnostics on %s", scenario.name)
            diagnostics.append(diagnose_scenario(scenario, args.seed0, geom))

    comparisons = {
        "odstrcil_vs_jirr_dx": paired_comparison(summary, "odstrcil", "jirr", "rms_dx"),
        "odstrcil_vs_jirr_dy": paired_comparison(summary, "odstrcil", "jirr", "rms_dy"),
        "joint_gd_vs_jirr_dx": paired_comparison(summary, "joint_gd", "jirr", "rms_dx"),
        "joint_gd_vs_jirr_dy": paired_comparison(summary, "joint_gd", "jirr", "rms_dy"),
        "odstrcil_vs_joint_gd_dx": paired_comparison(summary, "odstrcil", "joint_gd", "rms_dx"),
        # The two ablations: same engine, one knob. These are the mechanism claims.
        "gradient_trick_dx": paired_comparison(summary, "odstrcil", "odstrcil_value", "rms_dx"),
        "gradient_trick_dy": paired_comparison(summary, "odstrcil", "odstrcil_value", "rms_dy"),
        "vertical_stage_dy": paired_comparison(summary, "odstrcil", "odstrcil_novert", "rms_dy"),
        "vertical_stage_dx": paired_comparison(summary, "odstrcil", "odstrcil_novert", "rms_dx"),
    }

    bundle = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {
            **runner.environment_report(),
            "host": platform.node(),
            "workers": workers,
            "wallclock_total_s": elapsed,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "geometry": asdict(geom),
        "object_contrast_rad": object_contrast(geom),
        "n_seeds": args.seeds,
        "sweep_seeds": args.sweep_seeds,
        "summary": summary,
        "paired": comparisons,
        "diagnostics": diagnostics,
    }
    (out / "three_way_summary.json").write_text(json.dumps(bundle, indent=1))

    table = scenario_table(summary)
    print(table)
    (out / "three_way_table.txt").write_text(table + "\n")
    figure = summary_figure(
        summary, out / "three_way_summary.png", contrast=bundle["object_contrast_rad"]
    )

    if diagnostics:
        print("\n### diagnostics cross-check (triage on the UNALIGNED stack)")
        print(f"{'scenario':<20}{'expected':<16}{'triage says':<16}{'match':<7}{'conf':>6}")
        for record in diagnostics:
            print(
                f"{record['scenario']:<20}{str(record.get('expected')):<16}"
                f"{str(record.get('triage_top')):<16}"
                f"{'yes' if record.get('matches_expected') else 'no':<7}"
                f"{record.get('triage_confidence', float('nan')):>6.2f}"
            )

    print(f"\nwrote {out/'three_way_summary.json'}")
    if figure:
        print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
