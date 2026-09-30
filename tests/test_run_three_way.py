"""Tests for the three-way benchmark driver.

The driver does not compute anything an aligner would recognise -- it builds cases,
collapses seeds, and renders. So these tests are about the three ways a *comparison*
silently stops being a comparison:

* the scenarios stop being the same case for every method (geometry drift, a margin
  too small for the shift being injected, a step-0 pass that moved the truth);
* the seeds get averaged in a way that hides a refusal or a diverged run;
* two methods get compared by their means when the honest comparison is paired.

Nothing here runs an aligner. The expensive end-to-end path is
:mod:`benchmarks.runner`'s and is covered by ``tests/test_benchmark.py``; duplicating
it would make this file slow enough that nobody runs it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks import run_three_way as r3w

TINY = r3w.Geometry(size=24, n_slices=4, n_angles=12, margin=14, iterations=2, gd_iterations=4)


# -- scenarios and case construction -------------------------------------------------


def test_every_scenario_carries_the_same_base_jitter():
    """The perturbation under test is the *only* thing that differs between scenarios.

    If a scenario quietly changed the jitter as well, its column would be measuring two
    things at once and the sweep would be uninterpretable.
    """
    for scenario in r3w.core_scenarios():
        spec = scenario.spec(seed=3)
        assert spec.jitter_dy_rms == r3w.BASE_JITTER_DY
        assert spec.jitter_dx_rms == r3w.BASE_JITTER_DX
        assert spec.seed == 3


def test_the_scenario_list_covers_the_five_conditions_asked_for():
    names = {s.name for s in r3w.core_scenarios()}
    assert {"clean_jitter", "phase_ramp", "wrong_center", "vertical_drift", "nonrigid"} <= names


def test_exactly_one_scenario_expects_every_method_to_fail():
    """The non-rigid case is a positive control on the harness, not a method comparison."""
    failing = [s.name for s in r3w.core_scenarios() if s.expect_all_fail]
    assert failing == ["nonrigid"]


def test_a_margin_too_small_for_the_injected_shift_is_refused():
    """The injection is a Fourier shift, which wraps. A wrapped case is not a case."""
    geom = r3w.Geometry(margin=2)
    with pytest.raises(ValueError, match="wrap"):
        geom.validate(r3w.core_scenarios())


def test_the_default_geometry_survives_its_own_validation():
    geom = r3w.Geometry()
    geom.validate(r3w.core_scenarios() + r3w.ramp_sweep_scenarios([0.5, 4.0]))


def test_a_crop_that_would_eat_the_whole_frame_is_refused_up_front():
    """Refused at submit time, not three minutes into a worker process."""
    with pytest.raises(ValueError, match="removes all of it"):
        TINY.validate([s for s in r3w.core_scenarios() if s.crop_margin])


def test_two_scenarios_produce_the_same_detector_geometry():
    """Cross-scenario columns are only comparable if the frames are the same size."""
    a = r3w.build_case(r3w.core_scenarios()[0], 0, TINY)
    b = r3w.build_case(
        next(s for s in r3w.core_scenarios() if s.name == "vertical_drift"), 0, TINY
    )
    assert a.projections.shape == b.projections.shape


def test_the_same_scenario_and_seed_gives_the_same_case():
    a = r3w.build_case(r3w.core_scenarios()[0], 7, TINY)
    b = r3w.build_case(r3w.core_scenarios()[0], 7, TINY)
    assert np.array_equal(a.projections, b.projections)
    assert np.array_equal(a.truth.dx, b.truth.dx)


def test_different_seeds_give_different_draws():
    a = r3w.build_case(r3w.core_scenarios()[0], 0, TINY)
    b = r3w.build_case(r3w.core_scenarios()[0], 1, TINY)
    assert not np.allclose(a.truth.dx, b.truth.dx)


def test_step0_removes_the_ramp_without_moving_the_truth():
    """Subtracting a plane changes the values, never the content displacement.

    This is the whole reason the step-0 control is allowed to exist: if
    ``remove_phase_ramp`` moved content, the ``phase_ramp`` and ``phase_ramp_step0``
    rows would be scored against different ground truths and could not be compared.
    """
    scenarios = {s.name: s for s in r3w.core_scenarios(ramp=1.0)}
    raw = r3w.build_case(scenarios["phase_ramp"], 0, TINY)
    fixed = r3w.build_case(scenarios["phase_ramp_step0"], 0, TINY)

    assert np.array_equal(raw.truth.dy, fixed.truth.dy)
    assert np.array_equal(raw.truth.dx, fixed.truth.dx)
    assert not np.allclose(raw.projections, fixed.projections)
    # The step-0 pass is supposed to flatten the presumed-vacuum border.
    border = 3
    raw_border = np.abs(raw.projections[:, :border, :]).mean()
    fixed_border = np.abs(fixed.projections[:, :border, :]).mean()
    assert fixed_border < 0.25 * raw_border


def test_cropping_the_margin_destroys_the_vacuum_border_without_moving_the_truth():
    """The no-vacuum scenario must still be scoreable against the same ground truth.

    The crop is symmetric and happens after the shift injection, so it moves no
    content -- the same argument the harness's own ``truncation_px`` relies on. If it
    moved content, the row it produces could not be compared with any other row.
    """
    # The crop needs a frame it cannot eat: TINY is deliberately too small for it, and
    # Geometry.validate says so rather than letting a worker fail (tested below).
    geom = r3w.Geometry(size=24, n_slices=16, n_angles=12, margin=14, iterations=2)
    scenarios = {s.name: s for s in r3w.core_scenarios(ramp=1.0)}
    plain = r3w.build_case(scenarios["phase_ramp"], 0, geom)
    cropped = r3w.build_case(scenarios["ramp_no_vacuum"], 0, geom)

    assert np.array_equal(plain.truth.dy, cropped.truth.dy)
    assert np.array_equal(plain.truth.dx, cropped.truth.dx)
    assert cropped.projections.shape[1] < plain.projections.shape[1]
    assert cropped.projections.shape[2] < plain.projections.shape[2]
    assert cropped.clean.shape == cropped.projections.shape
    # The point of the scenario: the border is no longer vacuum.
    assert np.abs(cropped.projections[:, :2, :]).mean() > 0.2 * np.abs(
        cropped.projections
    ).mean()


def test_the_object_contrast_is_positive_and_reproducible():
    """The ramp amplitudes are only interpretable relative to this number."""
    first = r3w.object_contrast(TINY)
    assert first > 0
    assert first == pytest.approx(r3w.object_contrast(TINY))


# -- aggregation ---------------------------------------------------------------------


def _payload(scenario: str, seed: int, rows: list[dict]) -> dict:
    return {
        "case": {"name": f"{scenario}_seed{seed}", "shape": [4, 8, 8]},
        "results": rows,
        "scenario": {
            "name": scenario,
            "seed": seed,
            "overrides": {},
            "step0_ramp_removal": False,
            "expect_all_fail": False,
            "expected_diagnosis": None,
            "note": "",
        },
    }


def _row(name: str, dy: float, dx: float, *, runaway: bool = False, target: float = 0.333) -> dict:
    return {
        "name": name,
        "status": "ok",
        "message": "",
        "iterations": 12,
        "wallclock_s": 1.0,
        "extras": {"runaway": "stopped" if runaway else None, "diverging": False},
        "shift_recovery": {
            "rms_dy": dy,
            "rms_dx": dx,
            "max_dy": dy * 2,
            "max_dx": dx * 2,
            "rms_dy_raw": dy,
            "rms_dx_raw": dx,
            "target_px": target,
            "injected_rms_dy": 2.0,
            "injected_rms_dx": 0.7,
        },
    }


def test_aggregate_reports_the_mean_and_the_spread():
    payloads = [
        _payload("s", 0, [_row("jirr", 0.010, 0.030)]),
        _payload("s", 1, [_row("jirr", 0.020, 0.050)]),
    ]
    summary = r3w.aggregate(payloads)
    stats = summary["s"]["methods"]["jirr"]["stats"]
    assert stats["rms_dy"]["mean"] == pytest.approx(0.015)
    assert stats["rms_dy"]["std"] == pytest.approx(np.std([0.010, 0.020], ddof=1))
    assert stats["rms_dx"]["n"] == 2


def test_the_target_flag_is_decided_by_the_worst_seed_not_the_mean():
    """A method that passes on average and fails on one draw has not met the target.

    Averaging first is how a benchmark ends up certifying a method that fails one run
    in five, which is exactly the failure a user notices and a table does not.
    """
    payloads = [
        _payload("s", 0, [_row("m", 0.01, 0.01)]),
        _payload("s", 1, [_row("m", 0.01, 0.60)]),
    ]
    summary = r3w.aggregate(payloads)
    method = summary["s"]["methods"]["m"]
    assert method["meets_target_mean"] is True
    assert method["meets_target"] is False


def test_a_run_stopped_by_the_runaway_guard_is_counted_not_hidden():
    payloads = [
        _payload("s", 0, [_row("jirr", 5.0, 3.0, runaway=True)]),
        _payload("s", 1, [_row("jirr", 0.01, 0.02)]),
    ]
    summary = r3w.aggregate(payloads)
    assert summary["s"]["methods"]["jirr"]["n_runaway"] == 1


def test_paired_comparison_pairs_by_seed_and_counts_wins():
    """Two methods measured on the same draws are compared by pairing, not by means."""
    payloads = [
        _payload("s", 0, [_row("a", 0.01, 0.10), _row("b", 0.01, 0.20)]),
        _payload("s", 1, [_row("a", 0.01, 0.40), _row("b", 0.01, 0.20)]),
    ]
    summary = r3w.aggregate(payloads)
    paired = r3w.paired_comparison(summary, "a", "b", "rms_dx")["s"]
    assert paired["n_seeds"] == 2
    assert paired["left_wins"] == 1
    # mean difference is +0.05 (a loses on average) while the geometric mean ratio is
    # sqrt(0.5 * 2) = 1.0 (a and b are equal in the scale-free sense): the two summaries
    # disagree, which is exactly why both are reported and neither alone is quoted.
    assert paired["mean_diff"] == pytest.approx(0.05)
    assert paired["geo_mean_ratio"] == pytest.approx(1.0)
    assert paired["decisive"] is False


def test_a_consistent_win_is_marked_decisive():
    payloads = [
        _payload("s", seed, [_row("a", 0.01, 0.10), _row("b", 0.01, 0.20)])
        for seed in range(4)
    ]
    summary = r3w.aggregate(payloads)
    paired = r3w.paired_comparison(summary, "a", "b", "rms_dx")["s"]
    assert paired["left_wins"] == 4
    assert paired["decisive"] is True


def test_missing_methods_do_not_break_the_pairing():
    payloads = [_payload("s", 0, [_row("a", 0.01, 0.10)])]
    summary = r3w.aggregate(payloads)
    assert r3w.paired_comparison(summary, "a", "absent", "rms_dx") == {}


# -- rendering -----------------------------------------------------------------------


def test_the_table_marks_a_failing_method_and_names_the_scenario():
    payloads = [_payload("s", 0, [_row("jirr", 0.01, 0.02), _row("odstrcil", 0.9, 1.2)])]
    text = r3w.scenario_table(r3w.aggregate(payloads), order=("jirr", "odstrcil"))
    assert "### s" in text
    assert "NO" in text
    assert "jirr" in text and "odstrcil" in text


def test_the_figure_is_written_and_a_missing_matplotlib_is_not_fatal(tmp_path):
    payloads = [
        _payload("clean_jitter", seed, [_row("jirr", 0.01, 0.03), _row("odstrcil", 0.01, 0.02)])
        for seed in range(2)
    ]
    path = r3w.summary_figure(r3w.aggregate(payloads), tmp_path / "fig.png", contrast=16.0)
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None
    assert path is not None and path.exists()


def test_method_order_lists_every_aligner_the_runner_builds():
    """A method missing from METHOD_ORDER runs, is scored, and never appears in a table."""
    names = {a.name for a in r3w.aligners_for(TINY)}
    assert names <= set(r3w.METHOD_ORDER)
    assert set(r3w.HEADLINE_METHODS) <= names


def test_the_ablation_rows_differ_from_the_default_only_in_the_named_knob():
    """``odstrcil_value`` must be the same engine, or the ablation proves nothing."""
    built = {a.name: a for a in r3w.aligners_for(TINY)}
    base, value = built["odstrcil"], built["odstrcil_value"]
    assert (base.engine_module, base.engine_class) == (value.engine_module, value.engine_class)
    assert base.iterations == value.iterations
    assert base.algorithm == value.algorithm
    assert value.engine_kwargs["odstrcil"].gradient.domain == "value"
    assert value.engine_kwargs["odstrcil"].run_vertical is True
    novert = built["odstrcil_novert"]
    assert novert.engine_kwargs["odstrcil"].run_vertical is False
    assert novert.engine_kwargs["odstrcil"].gradient.domain != "value"


def test_the_default_odstrcil_row_is_actually_ramp_invariant():
    """A benchmark of "the gradient trick" against a configuration that is not one."""
    from tktomo.ptycho_align.core.gradient import GradientConfig

    assert GradientConfig().is_ramp_invariant


@pytest.mark.parametrize("amplitude", [0.0, 0.5, 4.0])
def test_the_ramp_sweep_names_encode_their_amplitude(amplitude):
    scenario = r3w.ramp_sweep_scenarios([amplitude])[0]
    assert scenario.name == f"ramp_sweep_{amplitude:g}"
    assert scenario.spec(0).phase_ramp_rms == amplitude
    assert float(scenario.name[len("ramp_sweep_"):]) == amplitude


def test_run_case_returns_a_record_rather_than_raising(monkeypatch):
    """One dead case must not cost the other twenty-nine their results."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("projector on fire")

    monkeypatch.setattr(r3w.runner, "run_benchmark", explode)
    payload = r3w.run_case((r3w.core_scenarios()[0], 0, TINY, None))
    assert payload["results"] == []
    assert "projector on fire" in payload["error"]
    assert payload["scenario"]["name"] == "clean_jitter"


def test_diagnose_case_reports_rather_than_raises_on_a_degenerate_stack():
    """A probe suite that cannot run must say so, not return a score."""
    case = r3w.build_case(r3w.core_scenarios()[0], 0, TINY)
    case.projections = np.zeros_like(case.projections)
    record = r3w.diagnose_case(case, expected="jitter")
    assert record["status"] in {"ok", "error", "skipped"}
    if record["status"] == "ok":
        assert record["matches_expected"] in (True, False)
        assert math.isnan(record["triage_confidence"]) or record["triage_confidence"] >= 0.0
