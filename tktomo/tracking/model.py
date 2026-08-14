"""Two-stage linear tomography model fit from sparse manual feature labels.

The model, for feature i at view j (all detector quantities in raw px):

    s_ij = a_i*cos(theta_j) + b_i*sin(theta_j)
    t_ij = -a_i*sin(theta_j) + b_i*cos(theta_j)
    u_ij = s_ij + c(theta_j) + dx_j
    v_ij = y_i + alpha(theta_j)*s_ij + beta(theta_j)*t_ij + dy_j

c (axis position), alpha (in-plane tilt) and beta (out-of-plane tilt) are
polynomials in the normalized angle tau = (theta - theta_ref)/theta_scale,
degree 0 (a constant) by default. No nonlinear optimizer is needed: the
u-system is linear in (a, b, c_k, dx); once solved, s and t are known
numbers and the v-system is linear in (y, alpha_k, beta_k, dy). Each solve
is wrapped in IRLS with Huber weights so a mislabeled point is downweighted
rather than dragging the geometry.

Every parameter can be held fixed at its current value ("fixed" and "fixed
at zero" are the same mechanism: a value plus a mask). Fixed columns move
to the right-hand side, so a fit with everything fixed is just a residual
evaluation.

GAUGE. With dx free, adding f(theta_j) to dx is invisible whenever f is
compensated by another column with the same per-view profile shared by all
features: f = P_k(tau) by the free c_k, f = cos by a uniform shift of every
a_i, f = sin by every b_i. Those directions are the object's position in
the reconstruction frame, not an error, and `solve_model` projects them out
of dx into (c, a, b) after each solve so that the reported c is canonical.
Pinning any feature's (a_i, b_i) breaks the cos/sin gauge, and fixing dx
breaks all of it: both are legitimate ways to inject external knowledge of
the center. Vertically only the constant of dy is gauge (into y): the
alpha_k and beta_k columns are scaled by each feature's own s or t, so
features at different radius respond differently and the tilt separates
from dy. That per-feature lever is the reason tilts are identifiable here
at all.

alpha and beta stay identifiable per polynomial order by the same argument,
but a warning is due when a coefficient is FIXED while its absorbing shift
group is FREE: the shifts then soak up whatever the fixed value gets wrong,
the residual cannot react, and the fixed value is decorative. `solve_model`
returns that warning rather than silently accepting the combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def poly_basis(theta: np.ndarray, degree: int, theta_ref: float,
               theta_scale: float) -> np.ndarray:
    """Columns tau^0 .. tau^degree of the normalized angle, shape (V, degree+1)."""
    tau = (np.asarray(theta, float) - theta_ref) / (theta_scale or 1.0)
    return np.column_stack([tau ** k for k in range(degree + 1)])


@dataclass
class AxisModel:
    """All parameters of the fit, plus the angle normalization they refer to."""

    theta: np.ndarray                # (V,) rad
    c_coef: np.ndarray               # (Kc+1,) raw px
    alpha_coef: np.ndarray           # (Ka+1,) rad (small-angle slope dv/du)
    beta_coef: np.ndarray            # (Kb+1,) rad
    dx: np.ndarray                   # (V,) raw px
    dy: np.ndarray                   # (V,) raw px
    feature_ids: np.ndarray          # (F,) int
    a: np.ndarray                    # (F,) raw px
    b: np.ndarray                    # (F,) raw px
    y: np.ndarray                    # (F,) raw px
    theta_ref: float = 0.0
    theta_scale: float = 1.0

    @classmethod
    def blank(cls, theta: np.ndarray, feature_ids,
              degrees: tuple[int, int, int] = (0, 0, 0)) -> "AxisModel":
        theta = np.asarray(theta, float)
        ids = np.asarray(feature_ids, int)
        kc, ka, kb = degrees
        return cls(
            theta=theta,
            c_coef=np.zeros(kc + 1), alpha_coef=np.zeros(ka + 1),
            beta_coef=np.zeros(kb + 1),
            dx=np.zeros(theta.size), dy=np.zeros(theta.size),
            feature_ids=ids,
            a=np.zeros(ids.size), b=np.zeros(ids.size), y=np.zeros(ids.size),
            theta_ref=float(theta.mean()) if theta.size else 0.0,
            theta_scale=float(np.ptp(theta)) or 1.0 if theta.size else 1.0,
        )

    @property
    def degrees(self) -> tuple[int, int, int]:
        return (self.c_coef.size - 1, self.alpha_coef.size - 1,
                self.beta_coef.size - 1)

    def with_degrees(self, kc: int, ka: int, kb: int) -> "AxisModel":
        """Resize the polynomials, keeping low-order coefficients."""
        def resize(coef, k):
            out = np.zeros(k + 1)
            n = min(coef.size, k + 1)
            out[:n] = coef[:n]
            return out
        return AxisModel(
            theta=self.theta.copy(),
            c_coef=resize(self.c_coef, kc),
            alpha_coef=resize(self.alpha_coef, ka),
            beta_coef=resize(self.beta_coef, kb),
            dx=self.dx.copy(), dy=self.dy.copy(),
            feature_ids=self.feature_ids.copy(),
            a=self.a.copy(), b=self.b.copy(), y=self.y.copy(),
            theta_ref=self.theta_ref, theta_scale=self.theta_scale,
        )

    def copy(self) -> "AxisModel":
        return AxisModel(
            theta=self.theta.copy(), c_coef=self.c_coef.copy(),
            alpha_coef=self.alpha_coef.copy(), beta_coef=self.beta_coef.copy(),
            dx=self.dx.copy(), dy=self.dy.copy(),
            feature_ids=self.feature_ids.copy(),
            a=self.a.copy(), b=self.b.copy(), y=self.y.copy(),
            theta_ref=self.theta_ref, theta_scale=self.theta_scale,
        )

    def subset(self, index: np.ndarray) -> "AxisModel":
        """The same model restricted to the features in `index`."""
        m = self.copy()
        m.feature_ids = self.feature_ids[index]
        m.a, m.b, m.y = self.a[index], self.b[index], self.y[index]
        return m

    # -- evaluation -------------------------------------------------------

    def axis_curves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """c(theta), alpha(theta), beta(theta) evaluated per view."""
        kc, ka, kb = self.degrees
        pc = poly_basis(self.theta, kc, self.theta_ref, self.theta_scale)
        pa = poly_basis(self.theta, ka, self.theta_ref, self.theta_scale)
        pb = poly_basis(self.theta, kb, self.theta_ref, self.theta_scale)
        return pc @ self.c_coef, pa @ self.alpha_coef, pb @ self.beta_coef

    def center_at_mean_theta(self) -> float:
        """c(theta_ref) in raw px. tau(theta_ref) = 0, so this is c_coef[0]."""
        return float(self.c_coef[0])

    def st(self) -> tuple[np.ndarray, np.ndarray]:
        """s and t for every (feature, view), shape (F, V)."""
        ct, sn = np.cos(self.theta), np.sin(self.theta)
        s = self.a[:, None] * ct[None, :] + self.b[:, None] * sn[None, :]
        t = -self.a[:, None] * sn[None, :] + self.b[:, None] * ct[None, :]
        return s, t

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Model (u, v) for every (feature, view), shape (F, V)."""
        c_of, alpha_of, beta_of = self.axis_curves()
        s, t = self.st()
        u = s + c_of[None, :] + self.dx[None, :]
        v = (self.y[:, None] + alpha_of[None, :] * s + beta_of[None, :] * t
             + self.dy[None, :])
        return u, v


@dataclass
class FreeMask:
    """Which parameters the next solve may move. Everything else is data."""

    dx: bool = True
    dy: bool = True
    c: np.ndarray = field(default_factory=lambda: np.ones(1, bool))
    alpha: np.ndarray = field(default_factory=lambda: np.ones(1, bool))
    beta: np.ndarray = field(default_factory=lambda: np.ones(1, bool))
    features: np.ndarray = field(default_factory=lambda: np.ones(0, bool))

    @classmethod
    def all_free(cls, model: AxisModel) -> "FreeMask":
        kc, ka, kb = model.degrees
        return cls(
            dx=True, dy=True,
            c=np.ones(kc + 1, bool), alpha=np.ones(ka + 1, bool),
            beta=np.ones(kb + 1, bool),
            features=np.ones(model.feature_ids.size, bool),
        )

    def subset(self, index: np.ndarray) -> "FreeMask":
        return FreeMask(dx=self.dx, dy=self.dy, c=self.c.copy(),
                        alpha=self.alpha.copy(), beta=self.beta.copy(),
                        features=self.features[index])

    def matches(self, model: AxisModel) -> bool:
        kc, ka, kb = model.degrees
        return (self.c.size == kc + 1 and self.alpha.size == ka + 1
                and self.beta.size == kb + 1
                and self.features.size == model.feature_ids.size)


@dataclass
class FitResult:
    """A solved (or merely evaluated) model plus per-observation residuals."""

    model: AxisModel
    obs: tuple[np.ndarray, np.ndarray]   # (i, j) feature/view index per obs
    residual_u: np.ndarray
    residual_v: np.ndarray
    weight_u: np.ndarray
    weight_v: np.ndarray
    observed_views: np.ndarray           # (V,) bool: dx/dy measured, not filled
    warnings: list[str] = field(default_factory=list)

    @property
    def rms_u(self) -> float:
        return float(np.sqrt(np.mean(self.residual_u ** 2))) if self.residual_u.size else float("nan")

    @property
    def rms_v(self) -> float:
        return float(np.sqrt(np.mean(self.residual_v ** 2))) if self.residual_v.size else float("nan")

    def feature_rms(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature rms of (residual_u, residual_v), NaN when unobserved."""
        n_feat = self.model.feature_ids.size
        i, _ = self.obs
        out_u = np.full(n_feat, np.nan)
        out_v = np.full(n_feat, np.nan)
        for f in range(n_feat):
            m = i == f
            if m.any():
                out_u[f] = float(np.sqrt(np.mean(self.residual_u[m] ** 2)))
                out_v[f] = float(np.sqrt(np.mean(self.residual_v[m] ** 2)))
        return out_u, out_v


# ---------------------------------------------------------------------------
# solver internals
# ---------------------------------------------------------------------------

def _huber_weights(r: np.ndarray, k: float) -> np.ndarray:
    """Huber weights on a robustly-scaled residual. Constant scale, no runaway.

    The scale is the MAD floored by percentile-90/2. The floor matters for
    MANUAL labels: views holding a single label are fitted exactly by their
    free dx/dy, so with staggered labeling a majority of residuals are
    exactly zero, the MAD collapses, and every remaining genuine label
    would be vetoed as an "outlier". The p90 floor keeps the scale at the
    real spread in that regime while barely moving it for well-behaved
    residuals (p90/2 ~ 0.8 sigma for a normal distribution), so a few
    true mis-clicks are still downweighted hard.
    """
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    s = max(s, float(np.percentile(np.abs(r), 90.0)) / 2.0)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(r)
    a = np.abs(r) / (k * s)
    return np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))


def _masked_irls(rows, cols, vals, target, ncol, x0, free, *,
                 iters, huber, damp, base_w=None):
    """IRLS least squares with fixed columns moved to the right-hand side.

    rows/cols/vals describe the UNWEIGHTED sparse design matrix in COO form,
    with `rows` indexing observations (so per-observation weights broadcast
    to entries as w[rows]). `base_w` is a per-observation prior weight
    (only RELATIVE values matter) multiplied into the Huber weights each
    iteration; it carries the feature-size prior, where a click on a large
    diffuse feature localizes it worse than one on a small sharp feature.
    Returns (x, residual, weight).
    """
    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import lsqr  # noqa: PLC0415

    n_obs = target.size
    a_full = sp.csr_matrix((vals, (rows, cols)), shape=(n_obs, ncol))
    free = np.asarray(free, bool)
    free_idx = np.flatnonzero(free)
    x = np.asarray(x0, float).copy()

    if free_idx.size == 0:
        r = target - a_full @ x
        return x, r, np.ones(n_obs)

    x_fixed = x.copy()
    x_fixed[free_idx] = 0.0
    fixed_pred = a_full @ x_fixed

    w0 = np.ones(n_obs) if base_w is None else np.asarray(base_w, float)
    w = w0.copy()
    for _ in range(max(1, iters)):
        a_w = sp.csr_matrix((vals * w[rows], (rows, cols)),
                            shape=(n_obs, ncol))
        sol = lsqr(a_w[:, free_idx], w * (target - fixed_pred),
                   damp=damp, atol=1e-12, btol=1e-12, iter_lim=5000)[0]
        x[free_idx] = sol
        r = target - a_full @ x
        w = w0 * _huber_weights(r, huber)
    return x, r, w


def _regauge_horizontal(model: AxisModel, mask: FreeMask,
                        observed: np.ndarray) -> None:
    """Project the gauge content of dx into (c, a, b). In place, exact.

    Gauge directions exist only where BOTH sides of the degeneracy are free:
    P_k needs a free c_k, cos/sin need every (a_i, b_i) free. A direction
    whose partner is fixed is a real degeneracy of the fit, not a gauge
    choice, and is left alone (solve_model warns about it instead).
    """
    if not mask.dx or not observed.any():
        return
    kc = model.degrees[0]
    pc = poly_basis(model.theta, kc, model.theta_ref, model.theta_scale)
    cols = []
    targets = []
    for k in range(kc + 1):
        if mask.c[k]:
            cols.append(pc[observed, k])
            targets.append(("c", k))
    all_features_free = bool(mask.features.size) and bool(mask.features.all())
    if all_features_free:
        cols.append(np.cos(model.theta[observed]))
        targets.append(("a", None))
        cols.append(np.sin(model.theta[observed]))
        targets.append(("b", None))
    if not cols:
        return
    g = np.column_stack(cols)
    p, *_ = np.linalg.lstsq(g, model.dx[observed], rcond=None)
    for (kind, k), coef in zip(targets, p):
        if kind == "c":
            model.c_coef[k] += coef
        elif kind == "a":
            model.a += coef
        else:
            model.b += coef
    model.dx[observed] -= g @ p


def _regauge_vertical(model: AxisModel, mask: FreeMask,
                      observed: np.ndarray) -> None:
    """Move the mean of dy into y. Only gauge when every feature's y is free."""
    if not mask.dy or not observed.any():
        return
    if not (mask.features.size and mask.features.all()):
        return
    m = float(np.mean(model.dy[observed]))
    model.dy[observed] -= m
    model.y += m


def fill_missing_shifts(values: np.ndarray, observed: np.ndarray,
                        theta: np.ndarray) -> np.ndarray:
    """Interpolate per-view shifts over theta where no view was labeled.

    PCHIP (shape-preserving, no overshoot) inside the observed span, edge
    values held constant outside it. The result is a modeling choice, not a
    measurement: callers must keep the `observed` mask alongside so plots
    and exports can say which is which.
    """
    values = np.asarray(values, float)
    observed = np.asarray(observed, bool)
    out = values.copy()
    missing = ~observed
    if not missing.any():
        return out
    n_obs = int(observed.sum())
    if n_obs == 0:
        out[:] = 0.0
        return out
    t_obs = theta[observed]
    v_obs = values[observed]
    if n_obs == 1:
        out[missing] = v_obs[0]
        return out
    order = np.argsort(t_obs)
    t_obs, v_obs = t_obs[order], v_obs[order]
    inside = missing & (theta >= t_obs[0]) & (theta <= t_obs[-1])
    outside = missing & ~inside
    if inside.any():
        from scipy.interpolate import PchipInterpolator  # noqa: PLC0415
        out[inside] = PchipInterpolator(t_obs, v_obs)(theta[inside])
    if outside.any():
        out[outside] = np.where(theta[outside] < t_obs[0], v_obs[0], v_obs[-1])
    return out


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def _structural_warnings(model: AxisModel, mask: FreeMask,
                         valid: np.ndarray) -> list[str]:
    out = []
    theta = model.theta
    span = float(np.rad2deg(theta.max() - theta.min())) if theta.size else 0.0
    if mask.dx and span < 120.0:
        out.append(
            f"W1: only {span:.0f} deg of angular arc. The center and the "
            f"feature amplitudes are barely separable on a short arc; do not "
            f"trust the center however small the residual is.")
    if mask.dx:
        fixed_c = [k for k in range(mask.c.size) if not mask.c[k]]
        if fixed_c:
            out.append(
                f"W3: c coefficient(s) {fixed_c} are fixed while dx is free. "
                f"dx absorbs the corresponding drift, so the fixed value "
                f"cannot affect the fit. Fix dx or pin a feature to make it "
                f"meaningful.")
    if mask.dy:
        fixed_ab = ([f"alpha[{k}]" for k in range(mask.alpha.size)
                     if not mask.alpha[k]]
                    + [f"beta[{k}]" for k in range(mask.beta.size)
                       if not mask.beta[k]])
        if fixed_ab and not (mask.features.size and mask.features.any()):
            out.append(f"W3: {', '.join(fixed_ab)} fixed with nothing pinned.")
    n_feat = valid.shape[0]
    if n_feat < 2:
        out.append("W5: fewer than 2 features. Per-view shifts are not "
                   "constrained by a single feature.")
    elif n_feat < 3:
        out.append("W5: fewer than 3 features. Tilts (alpha, beta) need "
                   "features at different radii to separate from dy.")
    labels_per_view = valid.sum(axis=0)
    thin = int(((labels_per_view == 1)).sum())
    if thin and (mask.dx or mask.dy):
        out.append(
            f"W4: {thin} view(s) carry exactly one label. Their dx/dy fit "
            f"that label exactly and mean nothing on their own.")
    # A free per-view shift eats the observation of any view it alone
    # explains. When MOST observations sit in single-label views, the
    # geometry (a, b, c, tilts) is left nearly unconstrained and the
    # fitted track can sit far from every marker while the residual is
    # still tiny. This is the staggered-labeling trap.
    n_obs_total = int(labels_per_view.sum())
    n_obs_single = int(labels_per_view[labels_per_view == 1].sum())
    if (mask.dx or mask.dy) and n_obs_total \
            and n_obs_single / n_obs_total > 0.5:
        out.append(
            f"W6: {n_obs_single} of {n_obs_total} labels are the ONLY "
            f"label in their view, so free dx/dy absorb them and the "
            f"track geometry is mostly unconstrained. Label several "
            f"features in the SAME views (or fix dx/dy) to pin it down.")
    return out


def residuals(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
              model: AxisModel) -> FitResult:
    """Evaluate the model against the labels without solving anything.

    This is what makes manual parameter overrides responsive: editing a
    value re-runs this, not the solver.
    """
    i, j = np.nonzero(valid)
    u_pred, v_pred = model.predict()
    observed = np.zeros(model.theta.size, bool)
    observed[j] = True
    return FitResult(
        model=model, obs=(i, j),
        residual_u=u[i, j] - u_pred[i, j],
        residual_v=v[i, j] - v_pred[i, j],
        weight_u=np.ones(i.size), weight_v=np.ones(i.size),
        observed_views=observed,
    )


def solve_model(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                model: AxisModel, mask: FreeMask | None = None, *,
                iters: int = 4, huber: float = 3.0,
                damp: float = 1e-8,
                feature_weight: np.ndarray | None = None) -> FitResult:
    """Fit the free parameters to the labels; fixed ones stay put.

    u, v: (F, V) label coordinates in RAW px (NaN/garbage where not valid).
    valid: (F, V) bool. The input `model` is not modified; the result
    carries an updated copy with dx/dy on unlabeled views filled by
    interpolation (flagged via `observed_views`). `feature_weight` (F,)
    is a relative per-feature prior weight, e.g. 1/size for hand labels
    whose localization scales with the feature's size.
    """
    model = model.copy()
    if mask is None:
        mask = FreeMask.all_free(model)
    if not mask.matches(model):
        raise ValueError("mask shape does not match model degrees/features")
    n_feat, n_view = valid.shape
    if u.shape != valid.shape or v.shape != valid.shape:
        raise ValueError("u, v, valid must share shape (n_features, n_views)")
    if n_feat != model.feature_ids.size or n_view != model.theta.size:
        raise ValueError("label arrays do not match model dimensions")

    i, j = np.nonzero(valid)
    n_obs = i.size
    if n_obs == 0:
        raise ValueError("no valid observations")
    observed = np.zeros(n_view, bool)
    observed[j] = True
    view_col = -np.ones(n_view, int)
    view_col[observed] = np.arange(int(observed.sum()))
    n_ov = int(observed.sum())

    theta = model.theta
    ct, sn = np.cos(theta), np.sin(theta)
    kc, ka, kb = model.degrees
    pc = poly_basis(theta, kc, model.theta_ref, model.theta_scale)
    pa = poly_basis(theta, ka, model.theta_ref, model.theta_scale)
    pb = poly_basis(theta, kb, model.theta_ref, model.theta_scale)

    # ---- stage 1: u is linear in (a, b, c_k, dx) ------------------------
    # column layout: [a (F), b (F), c (kc+1), dx (n_ov)]
    n1 = 2 * n_feat + (kc + 1) + n_ov
    per1 = 2 + (kc + 1) + 1
    rows1 = np.repeat(np.arange(n_obs), per1)
    cols1 = np.column_stack(
        [i, n_feat + i]
        + [np.full(n_obs, 2 * n_feat + k) for k in range(kc + 1)]
        + [2 * n_feat + (kc + 1) + view_col[j]]).ravel()
    vals1 = np.column_stack(
        [ct[j], sn[j]]
        + [pc[j, k] for k in range(kc + 1)]
        + [np.ones(n_obs)]).ravel()
    x0 = np.concatenate([model.a, model.b, model.c_coef, model.dx[observed]])
    free1 = np.concatenate([
        mask.features, mask.features,          # a_i and b_i follow the pin
        mask.c,
        np.full(n_ov, bool(mask.dx)),
    ])
    base_w = (None if feature_weight is None
              else np.asarray(feature_weight, float)[i])
    x1, res_u, w_u = _masked_irls(rows1, cols1, vals1, u[i, j], n1, x0, free1,
                                  iters=iters, huber=huber, damp=damp,
                                  base_w=base_w)
    model.a = x1[:n_feat]
    model.b = x1[n_feat:2 * n_feat]
    model.c_coef = x1[2 * n_feat:2 * n_feat + kc + 1]
    model.dx[observed] = x1[2 * n_feat + kc + 1:]
    _regauge_horizontal(model, mask, observed)

    # residuals after the regauge (predictions are gauge-invariant, but be
    # exact rather than clever)
    c_of = pc @ model.c_coef
    res_u = u[i, j] - (model.a[i] * ct[j] + model.b[i] * sn[j]
                       + c_of[j] + model.dx[j])

    # ---- stage 2: with (a, b) known, v is linear in (y, alpha_k, beta_k, dy)
    s = model.a[i] * ct[j] + model.b[i] * sn[j]
    t = -model.a[i] * sn[j] + model.b[i] * ct[j]
    # column layout: [y (F), alpha (ka+1), beta (kb+1), dy (n_ov)]
    n2 = n_feat + (ka + 1) + (kb + 1) + n_ov
    per2 = 1 + (ka + 1) + (kb + 1) + 1
    rows2 = np.repeat(np.arange(n_obs), per2)
    cols2 = np.column_stack(
        [i]
        + [np.full(n_obs, n_feat + k) for k in range(ka + 1)]
        + [np.full(n_obs, n_feat + ka + 1 + k) for k in range(kb + 1)]
        + [n_feat + (ka + 1) + (kb + 1) + view_col[j]]).ravel()
    vals2 = np.column_stack(
        [np.ones(n_obs)]
        + [s * pa[j, k] for k in range(ka + 1)]
        + [t * pb[j, k] for k in range(kb + 1)]
        + [np.ones(n_obs)]).ravel()
    y0 = np.concatenate([model.y, model.alpha_coef, model.beta_coef,
                         model.dy[observed]])
    free2 = np.concatenate([
        mask.features, mask.alpha, mask.beta,
        np.full(n_ov, bool(mask.dy)),
    ])
    x2, res_v, w_v = _masked_irls(rows2, cols2, vals2, v[i, j], n2, y0, free2,
                                  iters=iters, huber=huber, damp=damp,
                                  base_w=base_w)
    model.y = x2[:n_feat]
    model.alpha_coef = x2[n_feat:n_feat + ka + 1]
    model.beta_coef = x2[n_feat + ka + 1:n_feat + ka + 1 + kb + 1]
    model.dy[observed] = x2[n_feat + (ka + 1) + (kb + 1):]
    _regauge_vertical(model, mask, observed)

    alpha_of = pa @ model.alpha_coef
    beta_of = pb @ model.beta_coef
    res_v = v[i, j] - (model.y[i] + alpha_of[j] * s + beta_of[j] * t
                       + model.dy[j])

    # ---- fill the unlabeled views so exports have a complete curve ------
    if mask.dx:
        model.dx = fill_missing_shifts(model.dx, observed, theta)
    if mask.dy:
        model.dy = fill_missing_shifts(model.dy, observed, theta)

    return FitResult(
        model=model, obs=(i, j),
        residual_u=res_u, residual_v=res_v,
        weight_u=w_u, weight_v=w_v,
        observed_views=observed,
        warnings=_structural_warnings(model, mask, valid),
    )
