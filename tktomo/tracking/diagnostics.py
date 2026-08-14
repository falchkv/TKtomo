"""Empirical error bars and agreement checks for the tracking model fit.

The fit residual is the wrong number to trust: a model with two free
parameters per view can always make it small. Everything here measures the
same geometry twice in ways that share no data, and reports the
disagreement. Two disjoint halves of the features are two independent
measurements of the center, the tilts, and the shift curves; their split is
an error bar that assumes nothing about noise or model correctness. These
checks were built and burned in on the slogger graphite-ball pipeline,
where the parametric significance number said 57 sigma for a tilt the
half-split showed to be noise.
"""

from __future__ import annotations

import numpy as np

from tktomo.tracking.model import (
    AxisModel,
    FreeMask,
    poly_basis,
    solve_model,
)


def per_view_spread(residual: np.ndarray, obs_j: np.ndarray,
                    n_view: int) -> np.ndarray:
    """MAD of the residual within each view, across the features seeing it.

    If this stays sub-pixel over all views, the per-view displacement really
    is one number every feature agrees on. Views with fewer than 3 labels
    return NaN: agreement of two points is not evidence.
    """
    out = np.full(n_view, np.nan)
    for k in range(n_view):
        m = obs_j == k
        if m.sum() >= 3:
            r = residual[m]
            out[k] = float(np.median(np.abs(r - np.median(r))))
    return out


def tilt_significance(res_v: np.ndarray, s: np.ndarray,
                      alpha0: float) -> float:
    """|alpha| over its standard error. DO NOT trust this number alone.

    It assumes independent residuals and a correct model, and neither holds
    for short tracks. The half-split in `holdout_error` is the empirical
    check and wins every disagreement.
    """
    sd = float(np.std(s))
    if sd <= 0:
        return 0.0
    sigma = float(np.sqrt(np.mean(res_v ** 2)))
    se = sigma / (sd * np.sqrt(res_v.size))
    return abs(alpha0) / se if se > 0 else 0.0


def _split_features(valid: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(valid.sum(axis=1) > 0)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return idx[::2], idx[1::2]


def holdout_error(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                  model: AxisModel, mask: FreeMask, *, seed: int = 0,
                  iters: int = 4, huber: float = 3.0) -> dict:
    """Fit on half the features, predict the other half. The acceptance test.

    Also reports the half-split of the center, alpha and beta constants:
    the disagreement between two disjoint halves is the number that catches
    an ill-posed fit, which a residual never will.
    """
    a_idx, b_idx = _split_features(valid, seed)
    out = {"n_a": int(a_idx.size), "n_b": int(b_idx.size),
           "rms_u": float("nan"), "rms_v": float("nan"),
           "center_split": float("nan"), "alpha_split": float("nan"),
           "beta_split": float("nan")}
    if a_idx.size < 2 or b_idx.size < 2:
        return out

    fit_a = solve_model(u[a_idx], v[a_idx], valid[a_idx],
                        model.subset(a_idx), mask.subset(a_idx),
                        iters=iters, huber=huber)
    fit_b = solve_model(u[b_idx], v[b_idx], valid[b_idx],
                        model.subset(b_idx), mask.subset(b_idx),
                        iters=iters, huber=huber)
    ma, mb = fit_a.model, fit_b.model
    out["center_split"] = abs(ma.center_at_mean_theta()
                              - mb.center_at_mean_theta()) / 2.0
    out["alpha_split"] = abs(float(ma.alpha_coef[0] - mb.alpha_coef[0])) / 2.0
    out["beta_split"] = abs(float(ma.beta_coef[0] - mb.beta_coef[0])) / 2.0

    # Hold half A's per-view alignment fixed; refit only each held-out
    # feature's own 3 parameters (properties of the object, not alignment).
    theta = model.theta
    ct, sn = np.cos(theta), np.sin(theta)
    c_of, alpha_of, beta_of = ma.axis_curves()
    ru, rv = [], []
    for f in b_idx:
        m = valid[f]
        if m.sum() < 4:
            continue
        du = u[f, m] - c_of[m] - ma.dx[m]
        g = np.column_stack([ct[m], sn[m]])
        coef, *_ = np.linalg.lstsq(g, du, rcond=None)
        ru.append(du - g @ coef)
        s = g @ coef
        t = -coef[0] * sn[m] + coef[1] * ct[m]
        dv = v[f, m] - ma.dy[m] - alpha_of[m] * s - beta_of[m] * t
        rv.append(dv - np.mean(dv))
    if ru:
        out["rms_u"] = float(np.sqrt(np.mean(np.concatenate(ru) ** 2)))
        out["rms_v"] = float(np.sqrt(np.mean(np.concatenate(rv) ** 2)))
    return out


def shift_split(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                model: AxisModel, mask: FreeMask, *, seed: int = 0,
                order: int = 3, iters: int = 4, huber: float = 3.0) -> dict:
    """Solve on disjoint feature halves; compare the shift curves.

    Reported raw and after removing an order-`order` polynomial in theta:
    if detrending collapses the disagreement, the drift lives in the smooth
    part and the per-view jitter is real. Each half has sqrt(2) the error
    of the full solve and the difference sqrt(2) of one half, so the full
    solution's error is about rms(difference)/2.
    """
    a_idx, b_idx = _split_features(valid, seed)
    out = {"dx_rms": float("nan"), "dy_rms": float("nan"),
           "dx_detrended_rms": float("nan"), "dy_detrended_rms": float("nan"),
           "order": order}
    if a_idx.size < 2 or b_idx.size < 2:
        return out

    fit_a = solve_model(u[a_idx], v[a_idx], valid[a_idx],
                        model.subset(a_idx), mask.subset(a_idx),
                        iters=iters, huber=huber)
    fit_b = solve_model(u[b_idx], v[b_idx], valid[b_idx],
                        model.subset(b_idx), mask.subset(b_idx),
                        iters=iters, huber=huber)
    both = fit_a.observed_views & fit_b.observed_views
    if both.sum() <= order + 1:
        return out
    theta = model.theta[both]
    ddx = fit_a.model.dx[both] - fit_b.model.dx[both]
    ddy = fit_a.model.dy[both] - fit_b.model.dy[both]

    g = poly_basis(theta, order, float(theta.mean()),
                   float(np.ptp(theta)) or 1.0)

    def detrend(x):
        return x - g @ np.linalg.lstsq(g, x, rcond=None)[0]

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)) / 2.0)

    out["dx_rms"] = rms(ddx)
    out["dy_rms"] = rms(ddy)
    out["dx_detrended_rms"] = rms(detrend(ddx))
    out["dy_detrended_rms"] = rms(detrend(ddy))
    return out


def regauge_condition(model: AxisModel, observed: np.ndarray) -> float:
    """Condition number of the {P_k, cos, sin} gauge basis on observed views.

    On a short arc cos(theta) is nearly a low-order polynomial in theta, so
    higher polynomial degrees collide with the cos/sin gauge directions and
    with the feature amplitudes. Above ~1e4 the split between "axis drift"
    and "object position" is numerical fiction (warning W2).
    """
    if not observed.any():
        return float("inf")
    kc = model.degrees[0]
    pc = poly_basis(model.theta, kc, model.theta_ref, model.theta_scale)
    g = np.column_stack([pc[observed],
                         np.cos(model.theta[observed]),
                         np.sin(model.theta[observed])])
    try:
        return float(np.linalg.cond(g))
    except np.linalg.LinAlgError:
        return float("inf")


def run_diagnostics(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                    model: AxisModel, mask: FreeMask, fit) -> dict:
    """The full on-demand panel: splits, holdout, spreads, significance.

    `fit` is the current FitResult (for residual-based numbers). Costs four
    extra solves; still fast at manual-label scale.
    """
    i, j = fit.obs
    ct, sn = np.cos(model.theta), np.sin(model.theta)
    s = fit.model.a[i] * ct[j] + fit.model.b[i] * sn[j]
    ho = holdout_error(u, v, valid, model, mask)
    sp = shift_split(u, v, valid, model, mask)
    spread_u = per_view_spread(fit.residual_u, j, model.theta.size)
    spread_v = per_view_spread(fit.residual_v, j, model.theta.size)
    cond = regauge_condition(fit.model, fit.observed_views)
    out = {
        "holdout": ho,
        "shift_split": sp,
        "tilt_significance_sigma": tilt_significance(
            fit.residual_v, s, float(fit.model.alpha_coef[0])),
        "per_view_spread_u": spread_u,
        "per_view_spread_v": spread_v,
        "regauge_condition": cond,
        "center_estimate_raw_px": fit.model.center_at_mean_theta(),
        "center_split_px": ho["center_split"],
        "center_reliable": bool(np.isfinite(ho["center_split"])
                                and ho["center_split"] <= 5.0),
        "fit_rms_u": fit.rms_u,
        "fit_rms_v": fit.rms_v,
    }
    warnings = list(fit.warnings)
    if cond > 1e4:
        warnings.append(
            f"W2: gauge basis condition {cond:.2e}. The polynomial drift "
            f"degrees collide with the cos/sin gauge on this arc; the split "
            f"between axis drift and object position is not trustworthy.")
    if np.isfinite(ho["center_split"]) and ho["center_split"] > 5.0:
        warnings.append(
            f"W1: center half-split {ho['center_split']:.1f} px (> 5 px). "
            f"The center estimate is unreliable; label longer arcs or pin "
            f"the center from an independent estimate.")
    out["warnings"] = warnings
    return out
