"""Ground-truth round trips for the tracking model solver.

Everything here is synthetic: generate a known model, sample labels from
it, fit, and demand the truth back. Gauge directions are compared after
regauging BOTH sides, because two parameterizations that differ by pure
gauge produce identical data and only the canonical representative is
comparable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.diagnostics import (
    holdout_error,
    per_view_spread,
    regauge_condition,
    run_diagnostics,
    shift_split,
)
from tktomo.tracking.model import (
    AxisModel,
    FreeMask,
    fill_missing_shifts,
    poly_basis,
    residuals,
    solve_model,
)


def make_truth(n_feat=12, n_view=60, degrees=(1, 1, 0), seed=1,
               span_deg=180.0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, np.deg2rad(span_deg), n_view)
    model = AxisModel.blank(theta, np.arange(n_feat), degrees)
    r = 40.0 + 60.0 * rng.random(n_feat)
    phi = 2 * np.pi * rng.random(n_feat)
    model.a = r * np.cos(phi)
    model.b = r * np.sin(phi)
    model.y = 80.0 * (rng.random(n_feat) - 0.5)
    model.c_coef = np.array([435.0, 6.0])[: degrees[0] + 1]
    model.alpha_coef = np.array([-0.01, 0.004])[: degrees[1] + 1]
    model.beta_coef = np.array([0.006, -0.002])[: degrees[2] + 1]
    # smooth wobble plus jitter, then regauge so truth is canonical
    tt = np.linspace(0, 6 * np.pi, n_view)
    model.dx = 1.5 * np.sin(tt) + 0.5 * rng.standard_normal(n_view)
    model.dy = 1.0 * np.cos(1.7 * tt) + 0.4 * rng.standard_normal(n_view)
    canonical(model)
    return model


def canonical(model):
    """Regauge a model in place against the all-free gauge basis."""
    kc = model.degrees[0]
    pc = poly_basis(model.theta, kc, model.theta_ref, model.theta_scale)
    g = np.column_stack([pc, np.cos(model.theta), np.sin(model.theta)])
    p, *_ = np.linalg.lstsq(g, model.dx, rcond=None)
    model.c_coef += p[: kc + 1]
    model.a += p[kc + 1]
    model.b += p[kc + 2]
    model.dx -= g @ p
    m = float(np.mean(model.dy))
    model.dy -= m
    model.y += m
    return model


def sample_labels(model, frac=0.6, noise=0.05, seed=2, min_obs=8):
    rng = np.random.default_rng(seed)
    u, v = model.predict()
    n_feat, n_view = u.shape
    valid = rng.random((n_feat, n_view)) < frac
    for f in range(n_feat):
        while valid[f].sum() < min_obs:
            valid[f, rng.integers(n_view)] = True
    u = u + noise * rng.standard_normal(u.shape)
    v = v + noise * rng.standard_normal(v.shape)
    return u, v, valid


def test_round_trip_recovers_truth():
    truth = make_truth()
    u, v, valid = sample_labels(truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees)
    fit = solve_model(u, v, valid, start)
    m = fit.model

    assert fit.rms_u < 0.15
    assert fit.rms_v < 0.15
    # coefficients compare canonically (truth is already canonical)
    assert np.allclose(m.c_coef, truth.c_coef, atol=0.5)
    assert np.allclose(m.alpha_coef, truth.alpha_coef, atol=2e-3)
    assert np.allclose(m.beta_coef, truth.beta_coef, atol=2e-3)
    assert np.allclose(m.a, truth.a, atol=0.5)
    assert np.allclose(m.b, truth.b, atol=0.5)
    assert np.allclose(m.y, truth.y, atol=0.5)
    obs = fit.observed_views
    assert np.allclose(m.dx[obs], truth.dx[obs], atol=0.5)
    assert np.allclose(m.dy[obs], truth.dy[obs], atol=0.5)


def test_predictions_match_truth_predictions():
    truth = make_truth()
    u, v, valid = sample_labels(truth, noise=0.02)
    fit = solve_model(u, v, valid,
                      AxisModel.blank(truth.theta, truth.feature_ids,
                                      truth.degrees))
    u_t, v_t = truth.predict()
    u_f, v_f = fit.model.predict()
    i, j = np.nonzero(valid)
    assert float(np.sqrt(np.mean((u_f[i, j] - u_t[i, j]) ** 2))) < 0.1
    assert float(np.sqrt(np.mean((v_f[i, j] - v_t[i, j]) ** 2))) < 0.1


def test_gauge_shift_is_invisible_and_regauged():
    truth = make_truth()
    u, v, valid = sample_labels(truth, noise=0.0)
    fit = solve_model(u, v, valid,
                      AxisModel.blank(truth.theta, truth.feature_ids,
                                      truth.degrees))
    m = fit.model
    u_before, v_before = m.predict()

    # apply a pure gauge transformation by hand
    shifted = m.copy()
    kc = m.degrees[0]
    pc = poly_basis(m.theta, kc, m.theta_ref, m.theta_scale)
    g_content = 3.0 * pc[:, 0] + 0.7 * pc[:, 1] + 2.0 * np.cos(m.theta) \
        - 1.5 * np.sin(m.theta)
    shifted.dx = m.dx + g_content
    shifted.c_coef = m.c_coef - np.array([3.0, 0.7])
    shifted.a = m.a - 2.0
    shifted.b = m.b + 1.5
    u_after, v_after = shifted.predict()
    assert np.allclose(u_before, u_after, atol=1e-9)

    # the canonical representative comes back after regauging
    canonical(shifted)
    assert np.allclose(shifted.c_coef, m.c_coef, atol=1e-8)
    assert np.allclose(shifted.a, m.a, atol=1e-8)
    assert np.allclose(shifted.dx, m.dx, atol=1e-8)


def test_dx_fixed_identifies_center_exactly():
    truth = make_truth(degrees=(0, 0, 0))
    u, v, valid = sample_labels(truth, noise=0.0)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees)
    start.dx = truth.dx.copy()          # fixed AT the true values
    start.dy = truth.dy.copy()
    mask = FreeMask.all_free(start)
    mask.dx = False
    mask.dy = False
    fit = solve_model(u, v, valid, start, mask)
    assert abs(fit.model.center_at_mean_theta()
               - truth.center_at_mean_theta()) < 1e-6
    assert np.allclose(fit.model.alpha_coef, truth.alpha_coef, atol=1e-8)


def test_fixed_c_with_free_dx_changes_nothing_and_warns():
    truth = make_truth(degrees=(0, 0, 0))
    u, v, valid = sample_labels(truth)
    free = solve_model(u, v, valid,
                       AxisModel.blank(truth.theta, truth.feature_ids,
                                       truth.degrees))

    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees)
    start.c_coef[0] = 9999.0            # absurd, and it must not matter
    mask = FreeMask.all_free(start)
    mask.c[0] = False
    fixed = solve_model(u, v, valid, start, mask)

    assert abs(fixed.rms_u - free.rms_u) < 1e-6
    assert any("W3" in w for w in fixed.warnings)
    # dx swallowed the offset: predictions agree with the free fit
    u_a, _ = free.model.predict()
    u_b, _ = fixed.model.predict()
    i, j = np.nonzero(valid)
    assert np.allclose(u_a[i, j], u_b[i, j], atol=1e-4)


def test_pinned_feature_breaks_gauge_and_recovers_center():
    truth = make_truth(degrees=(0, 0, 0))
    u, v, valid = sample_labels(truth, noise=0.02)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees)
    start.a[0], start.b[0], start.y[0] = truth.a[0], truth.b[0], truth.y[0]
    mask = FreeMask.all_free(start)
    mask.features[0] = False
    fit = solve_model(u, v, valid, start, mask)
    # with the gauge pinned by one true feature, the center is real again
    assert abs(fit.model.center_at_mean_theta()
               - truth.center_at_mean_theta()) < 0.5


def test_outliers_are_downweighted():
    truth = make_truth()
    u, v, valid = sample_labels(truth, noise=0.05, seed=3)
    rng = np.random.default_rng(4)
    i, j = np.nonzero(valid)
    bad = rng.choice(i.size, size=max(1, i.size // 20), replace=False)
    u_corrupt = u.copy()
    u_corrupt[i[bad], j[bad]] += 20.0
    fit = solve_model(u_corrupt, v, valid,
                      AxisModel.blank(truth.theta, truth.feature_ids,
                                      truth.degrees))
    order = {(fi, fj): k for k, (fi, fj) in enumerate(zip(*fit.obs))}
    bad_w = [fit.weight_u[order[(i[k], j[k])]] for k in bad]
    assert np.median(bad_w) < 0.2
    assert np.allclose(fit.model.c_coef, truth.c_coef, atol=1.0)


def test_fill_missing_shifts():
    theta = np.linspace(0, np.pi, 11)
    values = np.sin(theta)
    observed = np.ones(11, bool)
    observed[[0, 4, 5, 10]] = False
    filled = fill_missing_shifts(values, observed, theta)
    assert np.allclose(filled[observed], values[observed])
    assert abs(filled[4] - np.sin(theta[4])) < 0.1
    # outside the observed span: edge value held
    assert filled[0] == filled[1]
    assert filled[10] == filled[9]


def test_residuals_only_no_solve():
    truth = make_truth()
    u, v, valid = sample_labels(truth, noise=0.0)
    fit = residuals(u, v, valid, truth)
    assert fit.rms_u < 1e-9
    assert fit.rms_v < 1e-9
    nudged = truth.copy()
    nudged.c_coef[0] += 2.0
    fit2 = residuals(u, v, valid, nudged)
    assert fit2.rms_u == pytest.approx(2.0, rel=0.05)


def test_degree_upgrade_finds_zero_extra_coefficient():
    # Truth canonical in the Kc=1 gauge (dx has no linear-in-theta content),
    # with the linear drift then removed: a Kc=1 fit must report c_1 ~ 0.
    # Note the premise matters: if truth dx DID carry smooth linear content,
    # moving it into c_1 would be the fit's correct gauge choice, not a bug.
    truth = make_truth(degrees=(1, 0, 0))
    truth.c_coef[1] = 0.0
    u, v, valid = sample_labels(truth, noise=0.02)
    fit = solve_model(u, v, valid,
                      AxisModel.blank(truth.theta, truth.feature_ids,
                                      (1, 0, 0)))
    assert abs(fit.model.c_coef[1]) < 0.5


def test_per_view_spread_and_warnings():
    truth = make_truth()
    u, v, valid = sample_labels(truth)
    fit = solve_model(u, v, valid,
                      AxisModel.blank(truth.theta, truth.feature_ids,
                                      truth.degrees))
    j = fit.obs[1]
    spread = per_view_spread(fit.residual_u, j, truth.theta.size)
    assert np.nanmedian(spread) < 0.2
    assert not any("W1" in w for w in fit.warnings)   # 180 deg arc

    short = make_truth(span_deg=30.0, degrees=(0, 0, 0))
    u2, v2, valid2 = sample_labels(short)
    fit2 = solve_model(u2, v2, valid2,
                       AxisModel.blank(short.theta, short.feature_ids,
                                       short.degrees))
    assert any("W1" in w for w in fit2.warnings)


def test_huber_scale_survives_zero_inflated_residuals():
    """Exact zeros from single-label views must not veto the real labels.

    A free dx fits every single-label view exactly, so with staggered
    manual labeling most residuals are exactly zero, the MAD collapses,
    and un-floored Huber weights discard every remaining genuine label.
    """
    from tktomo.tracking.model import _huber_weights

    zero_inflated = np.concatenate([np.zeros(60), np.full(40, 5.0)])
    w = _huber_weights(zero_inflated, 3.0)
    assert w[60:].min() > 0.5

    # ordinary robustness is untouched: few large outliers still go
    rng = np.random.default_rng(0)
    normal = np.concatenate([0.1 * rng.standard_normal(95),
                             np.full(5, 30.0)])
    w2 = _huber_weights(normal, 3.0)
    assert np.median(w2[:95]) > 0.9
    assert w2[95:].max() < 0.2


def test_staggered_labels_warn_unconstrained_geometry():
    truth = make_truth(n_feat=2, degrees=(0, 0, 0))
    u, v, valid = sample_labels(truth, noise=0.0)
    # staggered: feature 0 only in even views, feature 1 only in odd ones
    valid[:] = False
    valid[0, 0::2] = True
    valid[1, 1::2] = True
    fit = solve_model(u, v, valid,
                      AxisModel.blank(truth.theta, truth.feature_ids,
                                      truth.degrees))
    assert any("W6" in w for w in fit.warnings)

    # co-labeled views: the same features, now constrained, no W6
    valid[:, ::3] = True
    fit2 = solve_model(u, v, valid,
                       AxisModel.blank(truth.theta, truth.feature_ids,
                                       truth.degrees))
    assert not any("W6" in w for w in fit2.warnings)


def test_holdout_and_shift_split_run():
    truth = make_truth(n_feat=16)
    u, v, valid = sample_labels(truth, noise=0.05)
    model = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees)
    mask = FreeMask.all_free(model)
    ho = holdout_error(u, v, valid, model, mask)
    assert ho["rms_u"] < 0.5
    assert ho["center_split"] < 2.0
    sp = shift_split(u, v, valid, model, mask)
    assert np.isfinite(sp["dx_rms"])
    assert sp["dx_detrended_rms"] <= sp["dx_rms"] + 0.2

    fit = solve_model(u, v, valid, model, mask)
    diag = run_diagnostics(u, v, valid, model, mask, fit)
    assert diag["center_reliable"]
    assert np.isfinite(diag["regauge_condition"])


def test_regauge_condition_flags_short_arc_high_degree():
    long_arc = make_truth(degrees=(1, 0, 0))
    fit_cond = regauge_condition(long_arc, np.ones(long_arc.theta.size, bool))
    short = make_truth(span_deg=10.0, degrees=(2, 0, 0))
    short_cond = regauge_condition(short, np.ones(short.theta.size, bool))
    assert short_cond > 100 * fit_cond


def test_coordinate_chain_round_trip():
    chain = CoordinateChain(binning=2, crop=(8, 690, 10, 1942))
    # the slogger convention: center 435.7 on the 966 preproc grid
    u_raw, v_raw = chain.to_parent(435.7, 100.0)
    assert u_raw == pytest.approx(10 + 435.7 * 2 + 0.5)
    u_back, v_back = chain.from_parent(u_raw, v_raw)
    assert u_back == pytest.approx(435.7)
    assert v_back == pytest.approx(100.0)
    # parent -> preproc grid reproduces the loaded coordinate
    assert chain.parent_to_grid(u_raw, 2) == pytest.approx(435.7)
    assert chain.grid_to_parent(435.7, 2) == pytest.approx(u_raw)


def test_coordinate_chain_view_origin_and_extra_crop():
    origin = np.array([[5, 20], [7, 30]])
    chain = CoordinateChain(binning=2, crop=(8, 690, 10, 1942),
                            extra_crop=(2, 60, 3, 70), view_origin=origin)
    u, v = 11.0, 4.0
    u_raw, v_raw = chain.to_parent(u, v, view=1)
    assert u_raw == pytest.approx(10 + (11 + 3 + 30) * 2 + 0.5)
    assert v_raw == pytest.approx(8 + (4 + 2 + 7) * 2 + 0.5)
    u2, v2 = chain.from_parent(u_raw, v_raw, view=1)
    assert u2 == pytest.approx(u)
    assert v2 == pytest.approx(v)
    with pytest.raises(ValueError):
        chain.to_parent(1.0, 1.0)   # view index required


def test_shift_scaling():
    chain = CoordinateChain(binning=4, crop=(0, 0, 100, 0))
    assert chain.shift_to_parent(2.5) == pytest.approx(10.0)
    assert chain.shift_from_parent(10.0) == pytest.approx(2.5)
