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

PER-VIEW ROTATIONS. Sometimes the whole object tilts during the scan,
which no smooth tilt polynomial and no shift can express. Three per-view
angles (radians) describe the acquisition geometry, the beam frame
(e_s across the beam, e_t along it, e_z the rotation axis) with the
detector, rotated relative to the object by the small rotation vector
w_j = (rot_horiz, rot_beam, rot_axis) in the beam frame's own axes:

    p'_ij = R(-w_j) p_ij        with p = (s, t, y), R the Rodrigues rotation
    u_ij  = s' + c(theta_j) + dx_j
    v_ij  = y' + alpha(theta_j) s' + beta(theta_j) t' + dy_j

To first order s' = s + rot_axis t - rot_beam y, t' = t - rot_axis s +
rot_horiz y, y' = y - rot_horiz t + rot_beam s. rot_axis alone gives
s' = a cos(theta + rot_axis) + b sin(theta + rot_axis) exactly: it is an
increment of the projection angle, added to the nominal one. rot_beam is
an in-plane rotation of the image about the point (c, 0), rot_horiz an
out-of-plane tilt. With the rotations at zero the model is exactly the
one above, bit for bit.

With any rotation free the fit is no longer two separable linear
stages: rot_beam moves u (through y) and v (through s) at once, so the
solver switches to a JOINT Gauss-Newton on every free parameter, the u
and v residuals stacked, three passes, Huber weights per stage as before.
Each rotation carries a Gaussian prior N(0, sigma^2) that enters as
extra least-squares rows scaled by the assumed label noise: minimise
sum r^2 + noise_px^2 sum (w/sigma)^2. The prior is what makes them well
posed: a constant rot_axis over all views is degenerate with rotating
every (a_i, b_i), a constant rot_beam with alpha_0, a constant rot_horiz
with beta_0, and a per-view rot_beam or rot_horiz with dy_j whenever a
view's labels share the same s or t. The ridge picks the minimum-norm
representative, so the unpenalised partner takes the constant exactly
and no extra regauge step exists. (Measured on synthetic truth: solving
the rotations inside the two stages instead, rot_beam in v only, let
the stages trade error back and forth as the prior loosened.)

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


#: default Gaussian prior rms on each per-view rotation (rot_horiz, rot_beam,
#: rot_axis), radians
DEFAULT_ROT_SIGMA = (np.deg2rad(1.0),) * 3


def rotation_matrices(w: np.ndarray) -> np.ndarray:
    """R(w) = exp([w]_x) for rotation vectors `w` of shape (V, 3): (V, 3, 3).

    Rodrigues' formula, numpy only. Rows whose vector is exactly zero get
    the exact identity (no 0/0 branch), so a model without rotations
    reproduces the plain formulas bit for bit.
    """
    w = np.asarray(w, float).reshape(-1, 3)
    n = w.shape[0]
    out = np.tile(np.eye(3), (n, 1, 1))
    angle = np.linalg.norm(w, axis=1)
    nz = angle > 0
    if nz.any():
        k = w[nz] / angle[nz, None]
        kx = np.zeros((int(nz.sum()), 3, 3))
        kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
        kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
        kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
        sn = np.sin(angle[nz])[:, None, None]
        cs = np.cos(angle[nz])[:, None, None]
        out[nz] = np.eye(3) + sn * kx + (1.0 - cs) * (kx @ kx)
    return out


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
    #: per-view rotations of the beam frame, rad (module docstring); None
    #: at construction means zeros, so older callers need not pass them
    rot_horiz: np.ndarray | None = None   # (V,) about e_s, out-of-plane
    rot_beam: np.ndarray | None = None    # (V,) about e_t, in-plane
    rot_axis: np.ndarray | None = None    # (V,) about e_z, added to theta

    def __post_init__(self) -> None:
        n = np.asarray(self.theta).size
        for name in ("rot_horiz", "rot_beam", "rot_axis"):
            val = getattr(self, name)
            if val is None:
                setattr(self, name, np.zeros(n))
            else:
                setattr(self, name, np.asarray(val, float))

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
            rot_horiz=self.rot_horiz.copy(), rot_beam=self.rot_beam.copy(),
            rot_axis=self.rot_axis.copy(),
        )

    def copy(self) -> "AxisModel":
        return AxisModel(
            theta=self.theta.copy(), c_coef=self.c_coef.copy(),
            alpha_coef=self.alpha_coef.copy(), beta_coef=self.beta_coef.copy(),
            dx=self.dx.copy(), dy=self.dy.copy(),
            feature_ids=self.feature_ids.copy(),
            a=self.a.copy(), b=self.b.copy(), y=self.y.copy(),
            theta_ref=self.theta_ref, theta_scale=self.theta_scale,
            rot_horiz=self.rot_horiz.copy(), rot_beam=self.rot_beam.copy(),
            rot_axis=self.rot_axis.copy(),
        )

    @property
    def rotations(self) -> np.ndarray:
        """(V, 3) rotation vectors (rot_horiz, rot_beam, rot_axis)."""
        return np.column_stack([self.rot_horiz, self.rot_beam, self.rot_axis])

    @property
    def has_rotations(self) -> bool:
        return bool(np.any(self.rotations != 0.0))

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

    def beam_rotation(self, views=None) -> np.ndarray:
        """R(-w_j) per view, (V', 3, 3): its rows turn the nominal beam-frame
        coordinates (s, t, y) into the rotated ones (s', t', y')."""
        w = self.rotations
        if views is not None:
            w = w[np.asarray(views, int)]
        return rotation_matrices(-w)

    def project(self, a, b, y, views=None) -> tuple[np.ndarray, np.ndarray]:
        """The forward model for object points (a, b, y), shape (F', V').

        The one place the model is evaluated: `predict`, the window's
        probe and predicted markers, and the diagnostics all come here.
        With the rotations at zero this is bit for bit the plain formula.
        """
        rm = self.beam_rotation(views)
        views = slice(None) if views is None else np.asarray(views, int)
        theta = self.theta[views]
        ct, sn = np.cos(theta), np.sin(theta)
        a = np.atleast_1d(np.asarray(a, float))
        b = np.atleast_1d(np.asarray(b, float))
        y = np.atleast_1d(np.asarray(y, float))
        s = a[:, None] * ct[None, :] + b[:, None] * sn[None, :]
        t = -a[:, None] * sn[None, :] + b[:, None] * ct[None, :]
        yy = y[:, None] * np.ones_like(ct)[None, :]
        s_p = rm[None, :, 0, 0] * s + rm[None, :, 0, 1] * t + rm[None, :, 0, 2] * yy
        t_p = rm[None, :, 1, 0] * s + rm[None, :, 1, 1] * t + rm[None, :, 1, 2] * yy
        y_p = rm[None, :, 2, 0] * s + rm[None, :, 2, 1] * t + rm[None, :, 2, 2] * yy
        c_of, alpha_of, beta_of = self.axis_curves()
        c_of, alpha_of, beta_of = c_of[views], alpha_of[views], beta_of[views]
        u = s_p + c_of[None, :] + self.dx[views][None, :]
        v = (y_p + alpha_of[None, :] * s_p + beta_of[None, :] * t_p
             + self.dy[views][None, :])
        return u, v

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Model (u, v) for every (feature, view), shape (F, V)."""
        return self.project(self.a, self.b, self.y)


@dataclass
class FreeMask:
    """Which parameters the next solve may move. Everything else is data.

    The per-view rotations are opt-in extras with a prior: they default to
    fixed (at whatever the model holds, zero unless fitted), so a mask
    built by `all_free` or omitted altogether gives the plain fit.
    """

    dx: bool = True
    dy: bool = True
    c: np.ndarray = field(default_factory=lambda: np.ones(1, bool))
    alpha: np.ndarray = field(default_factory=lambda: np.ones(1, bool))
    beta: np.ndarray = field(default_factory=lambda: np.ones(1, bool))
    features: np.ndarray = field(default_factory=lambda: np.ones(0, bool))
    rot_horiz: bool = False
    rot_beam: bool = False
    rot_axis: bool = False

    @classmethod
    def all_free(cls, model: AxisModel, rotations: bool = False) -> "FreeMask":
        kc, ka, kb = model.degrees
        return cls(
            dx=True, dy=True,
            c=np.ones(kc + 1, bool), alpha=np.ones(ka + 1, bool),
            beta=np.ones(kb + 1, bool),
            features=np.ones(model.feature_ids.size, bool),
            rot_horiz=rotations, rot_beam=rotations, rot_axis=rotations,
        )

    @property
    def any_rotation(self) -> bool:
        return bool(self.rot_horiz or self.rot_beam or self.rot_axis)

    def subset(self, index: np.ndarray) -> "FreeMask":
        return FreeMask(dx=self.dx, dy=self.dy, c=self.c.copy(),
                        alpha=self.alpha.copy(), beta=self.beta.copy(),
                        features=self.features[index],
                        rot_horiz=self.rot_horiz, rot_beam=self.rot_beam,
                        rot_axis=self.rot_axis)

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

    The scale is the MAD floored by percentile-90/2, over the residuals a
    free parameter did not absorb. The floor matters for MANUAL labels:
    views holding a single label are fitted exactly by their free dx/dy,
    so with staggered labeling a majority of residuals are exactly zero,
    the MAD collapses, and every remaining genuine label would be vetoed
    as an "outlier". The p90 floor keeps the scale at the real spread in
    that regime while barely moving it for well-behaved residuals (p90/2
    ~ 0.8 sigma for a normal distribution), so a few true mis-clicks are
    still downweighted hard. Leaving the absorbed zeros out of the
    estimate extends that to the case where they outnumber 9 in 10.
    """
    # Residuals a free parameter absorbed exactly (a single-label view
    # under a free dx/dy or rotation) say nothing about the noise, and
    # when they are the majority even the p90 floor sits at zero: the
    # scale collapses, every real residual becomes an "outlier" and the
    # fit turns into an L1 problem driven by numerical dust. Measured on
    # a session with 82 of 84 views single-labeled. So the scale comes
    # from the residuals that are not (numerically) zero.
    r = np.asarray(r, float)
    live = r[np.abs(r) > 1e-9 * max(1.0, float(np.abs(r).max()))]
    base = live if live.size >= 3 else r
    s = 1.4826 * np.median(np.abs(base - np.median(base)))
    s = max(s, float(np.percentile(np.abs(base), 90.0)) / 2.0)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(r)
    a = np.abs(r) / (k * s)
    return np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))


def _masked_irls(rows, cols, vals, target, ncol, x0, free, *,
                 iters, huber, damp, base_w=None, n_data=None,
                 huber_split=None):
    """IRLS least squares with fixed columns moved to the right-hand side.

    rows/cols/vals describe the UNWEIGHTED sparse design matrix in COO form,
    with `rows` indexing observations (so per-observation weights broadcast
    to entries as w[rows]). `base_w` is a per-observation prior weight
    (only RELATIVE values matter) multiplied into the Huber weights each
    iteration; it carries the feature-size prior, where a click on a large
    diffuse feature localizes it worse than one on a small sharp feature.

    Rows from `n_data` on are PRIOR rows (Gaussian priors written as
    pseudo-observations): they keep weight 1, never enter the Huber scale,
    and are not part of the returned residual. So that "only relative
    values matter" stays true for `base_w` next to absolute prior rows, the
    data weights are normalised by their median whenever prior rows exist.
    `huber_split` = n means the data rows are two stacked groups (u rows
    then v rows, n each) whose Huber scales are estimated separately.
    Returns (x, residual, weight) over the data rows.
    """
    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import lsqr  # noqa: PLC0415

    n_rows = target.size
    n_data = n_rows if n_data is None else int(n_data)
    n_prior = n_rows - n_data
    a_full = sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, ncol))
    free = np.asarray(free, bool)
    free_idx = np.flatnonzero(free)
    x = np.asarray(x0, float).copy()

    if free_idx.size == 0:
        r = target - a_full @ x
        return x, r[:n_data], np.ones(n_data)

    x_fixed = x.copy()
    x_fixed[free_idx] = 0.0
    fixed_pred = a_full @ x_fixed

    w0 = np.ones(n_rows)
    if base_w is not None:
        w0[:n_data] = np.asarray(base_w, float)
        if n_prior:
            w0[:n_data] /= float(np.median(w0[:n_data]))
    w = w0.copy()
    for _ in range(max(1, iters)):
        a_w = sp.csr_matrix((vals * w[rows], (rows, cols)),
                            shape=(n_rows, ncol))
        sol = lsqr(a_w[:, free_idx], w * (target - fixed_pred),
                   damp=damp, atol=1e-12, btol=1e-12, iter_lim=5000)[0]
        x[free_idx] = sol
        r = target - a_full @ x
        w = w0.copy()
        if huber_split is None:
            w[:n_data] *= _huber_weights(r[:n_data], huber)
        else:
            k = int(huber_split)
            w[:k] *= _huber_weights(r[:k], huber)
            w[k:n_data] *= _huber_weights(r[k:n_data], huber)
    return x, r[:n_data], w[:n_data]


def _regauge_horizontal(model: AxisModel, mask: FreeMask,
                        observed: np.ndarray, profiles=None) -> None:
    """Project the gauge content of dx into (c, a, b). In place, exact.

    Gauge directions exist only where BOTH sides of the degeneracy are free:
    P_k needs a free c_k, cos/sin need every (a_i, b_i) free. A direction
    whose partner is fixed is a real degeneracy of the fit, not a gauge
    choice, and is left alone (solve_model warns about it instead).

    `profiles` = (per-view profile of a uniform a shift, of a uniform b
    shift), which are cos and sin of theta without rotations and the
    rotated versions with them (what the u stage's a and b columns
    actually were); None means cos, sin.
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
        if profiles is None:
            prof_a, prof_b = np.cos(model.theta), np.sin(model.theta)
        else:
            prof_a, prof_b = profiles
        cols.append(np.asarray(prof_a, float)[observed])
        targets.append(("a", None))
        cols.append(np.asarray(prof_b, float)[observed])
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


def _joint_pass(u, v, i, j, observed, view_col, model, mask, ct, sn,
                pc, pa, pb, base_w, prior_rows, rot_sigma, noise_px, *,
                iters, huber, damp):
    """One Gauss-Newton pass on every free parameter at once, in place.

    Rows are the u residuals, then the v residuals, then the prior rows;
    columns are increments of [a, b, c_k, dx, y, alpha_k, beta_k, dy,
    rot_horiz, rot_beam, rot_axis] (the rotation blocks only when free).
    The Jacobian is exact at the current model: with p' = R(-w) p, the
    derivative of p' with respect to a rotation increment dw is p' x dw,
    and the derivatives with respect to (a, b, y) are the rows of R
    applied to (cos, sin, 0), (sin, cos, 0) and (0, 0, 1). Returns the
    Huber weights of the u and v rows; residuals are recomputed exactly
    by the caller after the last pass.
    """
    n_feat = model.feature_ids.size
    n_obs = i.size
    n_ov = int(observed.sum())
    kc, ka, kb = model.degrees
    rm = model.beam_rotation()
    r = rm[j]
    a_i, b_i, y_i = model.a[i], model.b[i], model.y[i]
    s = a_i * ct[j] + b_i * sn[j]
    t = -a_i * sn[j] + b_i * ct[j]
    sp = r[:, 0, 0] * s + r[:, 0, 1] * t + r[:, 0, 2] * y_i
    tp = r[:, 1, 0] * s + r[:, 1, 1] * t + r[:, 1, 2] * y_i
    yp = r[:, 2, 0] * s + r[:, 2, 1] * t + r[:, 2, 2] * y_i
    c_of = (pc @ model.c_coef)[j]
    al = (pa @ model.alpha_coef)[j]
    be = (pb @ model.beta_coef)[j]
    # d(s', t', y') / d(a, b, y): the rows of R applied to the derivative
    # of the nominal (s, t, y)
    d_da = [r[:, k, 0] * ct[j] - r[:, k, 1] * sn[j] for k in range(3)]
    d_db = [r[:, k, 0] * sn[j] + r[:, k, 1] * ct[j] for k in range(3)]
    d_dy = [r[:, k, 2] for k in range(3)]
    res_u = u[i, j] - (sp + c_of + model.dx[j])
    res_v = v[i, j] - (yp + al * sp + be * tp + model.dy[j])

    oa, ob = 0, n_feat
    oc = 2 * n_feat
    odx = oc + kc + 1
    oy = odx + n_ov
    oal = oy + n_feat
    obe = oal + ka + 1
    ody = obe + kb + 1
    orh = ody + n_ov
    orb = orh + (n_ov if mask.rot_horiz else 0)
    ora = orb + (n_ov if mask.rot_beam else 0)
    ncol = ora + (n_ov if mask.rot_axis else 0)
    vc = view_col[j]
    ones = np.ones(n_obs)

    def v_of(q):
        """dv/dq from d(s', t', y')/dq."""
        return q[2] + al * q[0] + be * q[1]

    u_cols = [oa + i, ob + i] + [np.full(n_obs, oc + k) for k in range(kc + 1)] \
        + [odx + vc, oy + i]
    u_vals = [d_da[0], d_db[0]] + [pc[j, k] for k in range(kc + 1)] \
        + [ones, d_dy[0]]
    v_cols = [oa + i, ob + i, oy + i] \
        + [np.full(n_obs, oal + k) for k in range(ka + 1)] \
        + [np.full(n_obs, obe + k) for k in range(kb + 1)] + [ody + vc]
    v_vals = [v_of(d_da), v_of(d_db), v_of(d_dy)] \
        + [sp * pa[j, k] for k in range(ka + 1)] \
        + [tp * pb[j, k] for k in range(kb + 1)] + [ones]
    # rotation increments dw = (dh, db, da): d(s', t', y') = p' x dw. The
    # unknown of each block is the increment in units of sigma/noise_px
    # (see prior_rows), so its data columns carry that factor.
    unit_h, unit_b, unit_a = (sig / noise_px for sig in rot_sigma)
    if mask.rot_horiz:
        v_cols.append(orh + vc)
        v_vals.append(unit_h * v_of((np.zeros(n_obs), yp, -tp)))
    if mask.rot_beam:
        u_cols.append(orb + vc)
        u_vals.append(unit_b * -yp)
        v_cols.append(orb + vc)
        v_vals.append(unit_b * v_of((-yp, np.zeros(n_obs), sp)))
    if mask.rot_axis:
        u_cols.append(ora + vc)
        u_vals.append(unit_a * tp)
        v_cols.append(ora + vc)
        v_vals.append(unit_a * v_of((tp, -sp, np.zeros(n_obs))))
    rows = np.concatenate([np.repeat(np.arange(n_obs), len(u_cols)),
                           np.repeat(n_obs + np.arange(n_obs), len(v_cols))])
    cols = np.concatenate([np.column_stack(u_cols).ravel(),
                           np.column_stack(v_cols).ravel()])
    vals = np.concatenate([np.column_stack(u_vals).ravel(),
                           np.column_stack(v_vals).ravel()])
    target = np.concatenate([res_u, res_v])
    row_off = 2 * n_obs
    for flag, current, sigma, off in (
            (mask.rot_horiz, model.rot_horiz, rot_sigma[0], orh),
            (mask.rot_beam, model.rot_beam, rot_sigma[1], orb),
            (mask.rot_axis, model.rot_axis, rot_sigma[2], ora)):
        if not flag:
            continue
        pr, pcn, pv, pt = prior_rows(current, sigma, off, row_off)
        rows = np.concatenate([rows, pr])
        cols = np.concatenate([cols, pcn])
        vals = np.concatenate([vals, pv])
        target = np.concatenate([target, pt])
        row_off += n_ov
    free = np.concatenate(
        [mask.features, mask.features, mask.c, np.full(n_ov, bool(mask.dx)),
         mask.features, mask.alpha, mask.beta, np.full(n_ov, bool(mask.dy))]
        + [np.ones(n_ov, bool) for flag in (mask.rot_horiz, mask.rot_beam,
                                            mask.rot_axis) if flag])
    bw = None if base_w is None else np.concatenate([base_w, base_w])
    x, _res, w = _masked_irls(rows, cols, vals, target, ncol, np.zeros(ncol),
                              free, iters=iters, huber=huber, damp=damp,
                              base_w=bw, n_data=2 * n_obs, huber_split=n_obs)
    model.a = model.a + x[oa:ob]
    model.b = model.b + x[ob:oc]
    model.c_coef = model.c_coef + x[oc:odx]
    model.dx[observed] += x[odx:oy]
    model.y = model.y + x[oy:oal]
    model.alpha_coef = model.alpha_coef + x[oal:obe]
    model.beta_coef = model.beta_coef + x[obe:ody]
    model.dy[observed] += x[ody:orh]
    if mask.rot_horiz:
        model.rot_horiz[observed] += unit_h * x[orh:orh + n_ov]
    if mask.rot_beam:
        model.rot_beam[observed] += unit_b * x[orb:orb + n_ov]
    if mask.rot_axis:
        model.rot_axis[observed] += unit_a * x[ora:ora + n_ov]
    prof_a = rm[:, 0, 0] * ct - rm[:, 0, 1] * sn
    prof_b = rm[:, 0, 0] * sn + rm[:, 0, 1] * ct
    _regauge_horizontal(model, mask, observed, profiles=(prof_a, prof_b))
    _regauge_vertical(model, mask, observed)
    return w[:n_obs], w[n_obs:]


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
    # A per-view rotation needs features at different s, t or y in the
    # SAME view to be measured at all (that per-feature lever is what
    # separates it from a shift). With one or two labels the free shift
    # plus the angle fit them exactly and only the prior sets the angle.
    if mask.any_rotation:
        labelled = labels_per_view[labels_per_view > 0]
        few = int((labelled < 3).sum())
        if labelled.size and few / labelled.size > 0.5:
            out.append(
                f"W7: {few} of {labelled.size} labeled views carry fewer "
                f"than 3 labels while a per-view rotation is free. In those "
                f"views only the prior sets the angle, the labels carry no "
                f"information about it. Label three or more features in "
                f"the same views, or tighten the prior.")
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
                feature_weight: np.ndarray | None = None,
                rot_sigma=DEFAULT_ROT_SIGMA, noise_px: float = 1.0,
                passes: int | None = None) -> FitResult:
    """Fit the free parameters to the labels; fixed ones stay put.

    u, v: (F, V) label coordinates in RAW px (NaN/garbage where not valid).
    valid: (F, V) bool. The input `model` is not modified; the result
    carries an updated copy with dx/dy (and free rotations) on unlabeled
    views filled by interpolation (flagged via `observed_views`).
    `feature_weight` (F,) is a relative per-feature prior weight, e.g.
    1/size for hand labels whose localization scales with the feature's
    size.

    `rot_sigma` = (sigma_horiz, sigma_beam, sigma_axis) in radians is the
    Gaussian prior rms of each per-view rotation and `noise_px` the label
    noise it is weighed against (module docstring). `passes` is the number
    of outer Gauss-Newton passes around the two linear stages; None means
    1 without free rotations (the plain fit, unchanged) and 3 with them.
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
    rot_sigma = tuple(float(x) for x in rot_sigma)
    if len(rot_sigma) != 3 or any(not (x > 0) for x in rot_sigma):
        raise ValueError("rot_sigma must be three positive radians")
    noise_px = float(noise_px)
    if not noise_px > 0:
        raise ValueError("noise_px must be positive")
    n_pass = (3 if mask.any_rotation else 1) if passes is None else int(passes)

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
    base_w = (None if feature_weight is None
              else np.asarray(feature_weight, float)[i])
    ones = np.ones(n_obs)

    def prior_rows(current, sigma, col_off, row_off):
        """Pseudo-observations for N(0, sigma^2) on the TOTAL angle of one
        rotation block. The block's unknown is the increment in units of
        sigma/noise_px (the data columns are scaled by that factor), so the
        prior row is an identity row with target minus the current angle in
        those units. This whitening is what keeps the system conditioned:
        with the raw angle as unknown a tight sigma puts a huge weight on
        the prior rows, the iterative solver stops short, and residuals
        that should be exactly zero are not, which wrecks the Huber scale.
        """
        unit = sigma / noise_px
        rows = row_off + np.arange(n_ov)
        cols = col_off + np.arange(n_ov)
        return rows, cols, np.ones(n_ov), -current[observed] / unit

    if mask.any_rotation:
        for _ in range(max(1, n_pass)):
            w_u, w_v = _joint_pass(
                u, v, i, j, observed, view_col, model, mask, ct, sn,
                pc, pa, pb, base_w, prior_rows, rot_sigma, noise_px,
                iters=iters, huber=huber, damp=damp)
        u_pred, v_pred = model.predict()
        res_u = u[i, j] - u_pred[i, j]
        res_v = v[i, j] - v_pred[i, j]
        n_pass = 0                                   # stages below skipped

    for _ in range(max(0, n_pass)):
        # ---- stage 1: u is linear in (a, b, c_k, dx) given the rotations;
        # rot_axis enters as a Gauss-Newton increment column (value t')
        rm = model.beam_rotation()             # (V, 3, 3) = R(-w)
        r = rm[j]
        y_cur = model.y[i]
        s_nom = model.a[i] * ct[j] + model.b[i] * sn[j]
        t_nom = -model.a[i] * sn[j] + model.b[i] * ct[j]
        t_pr = r[:, 1, 0] * s_nom + r[:, 1, 1] * t_nom + r[:, 1, 2] * y_cur
        prof_a = rm[:, 0, 0] * ct - rm[:, 0, 1] * sn      # per view
        prof_b = rm[:, 0, 0] * sn + rm[:, 0, 1] * ct
        # column layout: [a (F), b (F), c (kc+1), dx (n_ov), d_rot_axis (n_ov)?]
        off_ra = 2 * n_feat + (kc + 1) + n_ov
        n1 = off_ra + (n_ov if mask.rot_axis else 0)
        col_blocks = ([i, n_feat + i]
                      + [np.full(n_obs, 2 * n_feat + k) for k in range(kc + 1)]
                      + [2 * n_feat + (kc + 1) + view_col[j]])
        val_blocks = ([prof_a[j], prof_b[j]]
                      + [pc[j, k] for k in range(kc + 1)]
                      + [ones])
        if mask.rot_axis:
            col_blocks.append(off_ra + view_col[j])
            val_blocks.append(t_pr * (rot_sigma[2] / noise_px))
        per1 = len(col_blocks)
        rows1 = np.repeat(np.arange(n_obs), per1)
        cols1 = np.column_stack(col_blocks).ravel()
        vals1 = np.column_stack(val_blocks).ravel()
        target1 = u[i, j] - r[:, 0, 2] * y_cur
        x0 = np.concatenate([model.a, model.b, model.c_coef,
                             model.dx[observed]]
                            + ([np.zeros(n_ov)] if mask.rot_axis else []))
        free1 = np.concatenate([
            mask.features, mask.features,          # a_i and b_i follow the pin
            mask.c,
            np.full(n_ov, bool(mask.dx)),
        ] + ([np.ones(n_ov, bool)] if mask.rot_axis else []))
        if mask.rot_axis:
            pr, pcn, pv, pt = prior_rows(model.rot_axis, rot_sigma[2],
                                         off_ra, n_obs)
            rows1 = np.concatenate([rows1, pr])
            cols1 = np.concatenate([cols1, pcn])
            vals1 = np.concatenate([vals1, pv])
            target1 = np.concatenate([target1, pt])
        x1, res_u, w_u = _masked_irls(rows1, cols1, vals1, target1, n1, x0,
                                      free1, iters=iters, huber=huber,
                                      damp=damp, base_w=base_w, n_data=n_obs)
        model.a = x1[:n_feat]
        model.b = x1[n_feat:2 * n_feat]
        model.c_coef = x1[2 * n_feat:2 * n_feat + kc + 1]
        model.dx[observed] = x1[2 * n_feat + kc + 1:off_ra]
        if mask.rot_axis:
            model.rot_axis[observed] += (rot_sigma[2] / noise_px) \
                * x1[off_ra:off_ra + n_ov]
        _regauge_horizontal(model, mask, observed, profiles=(prof_a, prof_b))

        # residuals after the regauge (predictions are gauge-invariant, but
        # be exact rather than clever)
        u_pred, _ = model.predict()
        res_u = u[i, j] - u_pred[i, j]

        # ---- stage 2: with (a, b) known, v is linear in (y, alpha_k,
        # beta_k, dy) given the rotations; rot_horiz and rot_beam enter as
        # increment columns (values -t' and s', with the tilt coupling)
        rm = model.beam_rotation()
        r = rm[j]
        s_nom = model.a[i] * ct[j] + model.b[i] * sn[j]
        t_nom = -model.a[i] * sn[j] + model.b[i] * ct[j]
        s_pr = r[:, 0, 0] * s_nom + r[:, 0, 1] * t_nom + r[:, 0, 2] * y_cur
        t_pr = r[:, 1, 0] * s_nom + r[:, 1, 1] * t_nom + r[:, 1, 2] * y_cur
        y_pr = r[:, 2, 0] * s_nom + r[:, 2, 1] * t_nom + r[:, 2, 2] * y_cur
        alpha_cur = (pa @ model.alpha_coef)[j]
        beta_cur = (pb @ model.beta_coef)[j]
        # column layout: [y (F), alpha (ka+1), beta (kb+1), dy (n_ov),
        #                 d_rot_horiz (n_ov)?, d_rot_beam (n_ov)?]
        off_dy = n_feat + (ka + 1) + (kb + 1)
        off_rh = off_dy + n_ov
        off_rb = off_rh + (n_ov if mask.rot_horiz else 0)
        n2 = off_rb + (n_ov if mask.rot_beam else 0)
        col_blocks = ([i]
                      + [np.full(n_obs, n_feat + k) for k in range(ka + 1)]
                      + [np.full(n_obs, n_feat + ka + 1 + k)
                         for k in range(kb + 1)]
                      + [off_dy + view_col[j]])
        val_blocks = ([r[:, 2, 2]]
                      + [s_pr * pa[j, k] for k in range(ka + 1)]
                      + [t_pr * pb[j, k] for k in range(kb + 1)]
                      + [ones])
        if mask.rot_horiz:
            col_blocks.append(off_rh + view_col[j])
            val_blocks.append((rot_sigma[0] / noise_px)
                              * (-t_pr + beta_cur * y_pr))
        if mask.rot_beam:
            col_blocks.append(off_rb + view_col[j])
            val_blocks.append((rot_sigma[1] / noise_px)
                              * (s_pr - alpha_cur * y_pr))
        per2 = len(col_blocks)
        rows2 = np.repeat(np.arange(n_obs), per2)
        cols2 = np.column_stack(col_blocks).ravel()
        vals2 = np.column_stack(val_blocks).ravel()
        target2 = v[i, j] - (r[:, 2, 0] * s_nom + r[:, 2, 1] * t_nom)
        y0 = np.concatenate([model.y, model.alpha_coef, model.beta_coef,
                             model.dy[observed]]
                            + ([np.zeros(n_ov)] if mask.rot_horiz else [])
                            + ([np.zeros(n_ov)] if mask.rot_beam else []))
        free2 = np.concatenate([
            mask.features, mask.alpha, mask.beta,
            np.full(n_ov, bool(mask.dy)),
        ] + ([np.ones(n_ov, bool)] if mask.rot_horiz else [])
          + ([np.ones(n_ov, bool)] if mask.rot_beam else []))
        row_off = n_obs
        for flag, current, sigma, off in (
                (mask.rot_horiz, model.rot_horiz, rot_sigma[0], off_rh),
                (mask.rot_beam, model.rot_beam, rot_sigma[1], off_rb)):
            if not flag:
                continue
            pr, pcn, pv, pt = prior_rows(current, sigma, off, row_off)
            rows2 = np.concatenate([rows2, pr])
            cols2 = np.concatenate([cols2, pcn])
            vals2 = np.concatenate([vals2, pv])
            target2 = np.concatenate([target2, pt])
            row_off += n_ov
        x2, res_v, w_v = _masked_irls(rows2, cols2, vals2, target2, n2, y0,
                                      free2, iters=iters, huber=huber,
                                      damp=damp, base_w=base_w, n_data=n_obs)
        model.y = x2[:n_feat]
        model.alpha_coef = x2[n_feat:n_feat + ka + 1]
        model.beta_coef = x2[n_feat + ka + 1:n_feat + ka + 1 + kb + 1]
        model.dy[observed] = x2[off_dy:off_rh]
        if mask.rot_horiz:
            model.rot_horiz[observed] += (rot_sigma[0] / noise_px) \
                * x2[off_rh:off_rh + n_ov]
        if mask.rot_beam:
            model.rot_beam[observed] += (rot_sigma[1] / noise_px) \
                * x2[off_rb:off_rb + n_ov]
        _regauge_vertical(model, mask, observed)

        _, v_pred = model.predict()
        res_v = v[i, j] - v_pred[i, j]

    # ---- fill the unlabeled views so exports have a complete curve ------
    if mask.dx:
        model.dx = fill_missing_shifts(model.dx, observed, theta)
    if mask.dy:
        model.dy = fill_missing_shifts(model.dy, observed, theta)
    for flag, name in ((mask.rot_horiz, "rot_horiz"),
                       (mask.rot_beam, "rot_beam"),
                       (mask.rot_axis, "rot_axis")):
        if flag:
            setattr(model, name,
                    fill_missing_shifts(getattr(model, name), observed, theta))

    return FitResult(
        model=model, obs=(i, j),
        residual_u=res_u, residual_v=res_v,
        weight_u=w_u, weight_v=w_v,
        observed_views=observed,
        warnings=_structural_warnings(model, mask, valid),
    )
