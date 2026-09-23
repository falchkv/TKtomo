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

TANGENT FEATURES. Besides a plain point, a feature can be one of the four
silhouette tangents of a sphere: laos and raos at the smallest and largest
u, taos and baos at the smallest and largest v. Features linked into one
BODY share a centre (a, b, y) and a radius R, which is what makes a pair
worth more than two separate labels.

The model reads an object point out as u = m1.p + c + dx and
v = m2.p + dy with m1 = e_s', m2 = alpha e_s' + beta e_t' + e_z' in the
rotated beam basis (the dual basis `astra_parallel3d_vectors` is built
from). A linear functional is extremal over a sphere of radius R at
p = centre +- R m/|m|, so with |m1| = 1, |m2| = sqrt(1 + alpha^2 + beta^2)
and m1.m2 = alpha:

    laos/raos (sign s = -1/+1):  u = u_c + s R        v = v_c + s R alpha
    taos/baos (sign s = -1/+1):  v = v_c + s R |m2|   u = u_c + s R alpha/|m2|

Exact, not first order, and invariant to the per-view rotations because m1
and m2 are already the rotated basis vectors: a sphere projects to a disc
of radius R under any orientation, so no tilt or rotation in this model
changes R. `tangent_scales` is the one place those four coefficients live.

Two consequences worth knowing. The midpoint of two opposite tangents is
the projected centre for ANY centrally symmetric convex body, sphere or
not, so a linked pair is an exact point feature whatever the void really
looks like. The constant radius is the extra assumption, and what it buys
is that a view showing only one edge still constrains the centre.

A tangent feature constrains ONE coordinate. The other one is the centre's,
which is exactly true for a sphere and false for a lopsided void, so it is
used only where `use_center` says so. The two coordinates therefore have
their own observation lists (`split_validity`, `FitResult.obs_u` and
`obs_v`), and a view carrying only tangent labels gets a dx column and no
dy one. A body with no vertical observation has no identifiable height at
all: its y reaches u only through rot_beam, so it is seeded from the
clicked v as a nuisance value and held fixed (warning W11).

The radius sits in the u system for a horizontal tangent and in the v
system for a vertical one, so a sphere labelled on both axes shares a
parameter between the two stages, which a two-stage solve cannot own. That
case goes to the joint Gauss-Newton pass, the same one the rotations use
(`radius_couples_stages`). Everything else stays on the staged solve.

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
from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# feature kinds
# ---------------------------------------------------------------------------

#: a fixed object point: both coordinates of a label observe it directly
POINT = 0
#: tangent points of a sphere's silhouette, named for the side they sit on.
#: u is the detector column and v the detector row, so "top" is the smaller v.
LAOS = 1       # left apex of sphere, smallest u
RAOS = 2       # right apex of sphere, largest u
TAOS = 3       # top apex of sphere, smallest v
BAOS = 4       # bottom apex of sphere, largest v

KIND_NAMES = {POINT: "point", LAOS: "laos", RAOS: "raos",
              TAOS: "taos", BAOS: "baos"}
KIND_BY_NAME = {name: k for k, name in KIND_NAMES.items()}
TANGENT_KINDS = (LAOS, RAOS, TAOS, BAOS)

#: the signed multiple of the radius each kind rides on, along u and along v
#: before the tilt correction (see TANGENT FEATURES in the module docstring)
SIGN_U = np.array([0.0, -1.0, +1.0, 0.0, 0.0])
SIGN_V = np.array([0.0, 0.0, 0.0, -1.0, +1.0])


def compact_bodies(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renumber body labels to 0..B-1 keeping first-appearance order.

    Returns (compact labels, the original label of each new one), so a
    caller can carry per-body arrays across the renumbering.
    """
    body = np.asarray(body, int).ravel()
    if body.size == 0:
        return body.copy(), np.zeros(0, int)
    order = {}
    for lab in body:
        if int(lab) not in order:
            order[int(lab)] = len(order)
    out = np.array([order[int(lab)] for lab in body], int)
    back = np.zeros(len(order), int)
    for lab, k in order.items():
        back[k] = lab
    return out, back


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
    #: feature kinds and the bodies they belong to. None means "every
    #: feature is a plain point and its own body", i.e. the model as it
    #: was before tangent features existed.
    kind: np.ndarray | None = None        # (F,) int, one of POINT..BAOS
    body: np.ndarray | None = None        # (F,) int, compact 0..B-1
    use_center: np.ndarray | None = None  # (F,) bool
    radius: np.ndarray | None = None      # (B,) raw px

    def __post_init__(self) -> None:
        n = np.asarray(self.theta).size
        for name in ("rot_horiz", "rot_beam", "rot_axis"):
            val = getattr(self, name)
            if val is None:
                setattr(self, name, np.zeros(n))
            else:
                setattr(self, name, np.asarray(val, float))
        f = np.asarray(self.feature_ids).size
        self.kind = (np.zeros(f, int) if self.kind is None
                     else np.asarray(self.kind, int))
        self.use_center = (np.zeros(f, bool) if self.use_center is None
                           else np.asarray(self.use_center, bool))
        if self.body is None:
            self.body = np.arange(f)
        else:
            self.body = compact_bodies(self.body)[0]
        n_body = int(self.body.max()) + 1 if f else 0
        if self.radius is None:
            self.radius = np.zeros(n_body)
        else:
            self.radius = np.asarray(self.radius, float)
            if self.radius.size != n_body:
                r = np.zeros(n_body)
                r[:min(n_body, self.radius.size)] = \
                    self.radius[:min(n_body, self.radius.size)]
                self.radius = r

    @classmethod
    def blank(cls, theta: np.ndarray, feature_ids,
              degrees: tuple[int, int, int] = (0, 0, 0),
              kind=None, body=None, use_center=None,
              radius=None) -> "AxisModel":
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
            kind=kind, body=body, use_center=use_center, radius=radius,
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
            kind=self.kind.copy(), body=self.body.copy(),
            use_center=self.use_center.copy(), radius=self.radius.copy(),
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
            kind=self.kind.copy(), body=self.body.copy(),
            use_center=self.use_center.copy(), radius=self.radius.copy(),
        )

    @property
    def rotations(self) -> np.ndarray:
        """(V, 3) rotation vectors (rot_horiz, rot_beam, rot_axis)."""
        return np.column_stack([self.rot_horiz, self.rot_beam, self.rot_axis])

    @property
    def has_rotations(self) -> bool:
        return bool(np.any(self.rotations != 0.0))

    def subset(self, index: np.ndarray) -> "AxisModel":
        """The same model restricted to the features in `index`.

        Bodies are renumbered compactly and their radii carried across, so
        a subset that keeps only one member of a sphere keeps that sphere's
        radius (the half-split diagnostics split by body, so that case
        should not arise, but a subset must not silently lose a radius).
        """
        m = self.copy()
        m.feature_ids = self.feature_ids[index]
        m.a, m.b, m.y = self.a[index], self.b[index], self.y[index]
        m.kind = self.kind[index]
        m.use_center = self.use_center[index]
        compact, back = compact_bodies(self.body[index])
        m.body = compact
        m.radius = self.radius[back] if back.size else np.zeros(0)
        return m

    # -- feature kinds ----------------------------------------------------

    @property
    def n_bodies(self) -> int:
        return int(self.radius.size)

    @property
    def has_tangents(self) -> bool:
        return bool(np.any(self.kind != POINT))

    def body_radius_per_feature(self) -> np.ndarray:
        """(F,) the radius of each feature's body."""
        return (self.radius[self.body] if self.body.size
                else np.zeros(0))

    def tangent_scales_for(self, kind, views=None
                           ) -> tuple[np.ndarray, np.ndarray]:
        """`tangent_scales` for an arbitrary kind array, (len(kind), V').

        The diagnostics hold one half's geometry fixed and refit the other
        half's features against it, so they need these coefficients for
        features that are not rows of this model.
        """
        _, alpha_of, beta_of = self.axis_curves()
        if views is not None:
            views = np.asarray(views, int)
            alpha_of, beta_of = alpha_of[views], beta_of[views]
        nm2 = np.sqrt(1.0 + alpha_of ** 2 + beta_of ** 2)
        kind = np.atleast_1d(np.asarray(kind, int))
        su = SIGN_U[kind][:, None]
        sv = SIGN_V[kind][:, None]
        return (su + sv * (alpha_of / nm2)[None, :],
                su * alpha_of[None, :] + sv * nm2[None, :])

    def tangent_scales(self, views=None) -> tuple[np.ndarray, np.ndarray]:
        """(du/dR, dv/dR) per (feature, view), shape (F, V').

        The read-out functionals of the model are u = m1.p + c + dx and
        v = m2.p + dy with |m1| = 1, |m2| = sqrt(1 + alpha^2 + beta^2) and
        m1.m2 = alpha (the dual basis `astra_parallel3d_vectors` is built
        from). The silhouette of a sphere of radius R is extremal in u at
        p = centre +- R m1/|m1| and in v at p = centre +- R m2/|m2|, which
        gives these four coefficients exactly, per the module docstring.
        They are invariant under the per-view rotations because m1 and m2
        are already expressed in the rotated beam basis.
        """
        return self.tangent_scales_for(self.kind, views)

    def tangent_offsets(self, views=None) -> tuple[np.ndarray, np.ndarray]:
        """(du, dv) added to the body centre for each feature, (F, V')."""
        du, dv = self.tangent_scales(views)
        r = self.body_radius_per_feature()[:, None]
        return du * r, dv * r

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

    def predict_centers(self, views=None) -> tuple[np.ndarray, np.ndarray]:
        """Model (u, v) of each feature's BODY centre, shape (F, V')."""
        return self.project(self.a, self.b, self.y, views)

    def predict(self, views=None) -> tuple[np.ndarray, np.ndarray]:
        """Model (u, v) of each feature's own marker, shape (F, V').

        For a point feature that is the projected object point. For a
        tangent feature it is the silhouette tangent point, which is what
        the user clicked, so the predicted cross and the residuals are
        about the same thing for every kind.
        """
        u, v = self.project(self.a, self.b, self.y, views)
        if self.has_tangents:
            du, dv = self.tangent_offsets(views)
            u = u + du
            v = v + dv
        return u, v


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
    #: one flag per BODY. None means "free wherever the labels constrain
    #: it", which `solve_model` works out from the feature kinds.
    radius: np.ndarray | None = None

    @classmethod
    def all_free(cls, model: AxisModel, rotations: bool = False) -> "FreeMask":
        kc, ka, kb = model.degrees
        return cls(
            dx=True, dy=True,
            c=np.ones(kc + 1, bool), alpha=np.ones(ka + 1, bool),
            beta=np.ones(kb + 1, bool),
            features=np.ones(model.feature_ids.size, bool),
            rot_horiz=rotations, rot_beam=rotations, rot_axis=rotations,
            radius=np.ones(model.n_bodies, bool),
        )

    @property
    def any_rotation(self) -> bool:
        return bool(self.rot_horiz or self.rot_beam or self.rot_axis)

    def subset(self, index: np.ndarray, model: AxisModel | None = None
               ) -> "FreeMask":
        radius = self.radius
        if radius is not None and model is not None:
            _, back = compact_bodies(model.body[index])
            radius = radius[back] if back.size else np.zeros(0, bool)
        return FreeMask(dx=self.dx, dy=self.dy, c=self.c.copy(),
                        alpha=self.alpha.copy(), beta=self.beta.copy(),
                        features=self.features[index],
                        rot_horiz=self.rot_horiz, rot_beam=self.rot_beam,
                        rot_axis=self.rot_axis,
                        radius=None if radius is None else np.asarray(radius))

    def matches(self, model: AxisModel) -> bool:
        kc, ka, kb = model.degrees
        return (self.c.size == kc + 1 and self.alpha.size == ka + 1
                and self.beta.size == kb + 1
                and self.features.size == model.feature_ids.size
                and (self.radius is None
                     or self.radius.size == model.n_bodies))


@dataclass
class FitResult:
    """A solved (or merely evaluated) model plus per-observation residuals."""

    model: AxisModel
    #: (i, j) feature/view index per observation, separately for the two
    #: coordinates: a tangent feature constrains only one of them, so the
    #: two lists differ as soon as any tangent feature exists.
    obs_u: tuple[np.ndarray, np.ndarray]
    obs_v: tuple[np.ndarray, np.ndarray]
    residual_u: np.ndarray
    residual_v: np.ndarray
    weight_u: np.ndarray
    weight_v: np.ndarray
    observed_dx: np.ndarray              # (V,) bool: dx measured, not filled
    observed_dy: np.ndarray              # (V,) bool: dy measured, not filled
    warnings: list[str] = field(default_factory=list)

    @property
    def obs(self) -> tuple[np.ndarray, np.ndarray]:
        """The u observations. Kept for callers that predate the split."""
        return self.obs_u

    @property
    def observed_views(self) -> np.ndarray:
        """Views carrying any observation at all."""
        return self.observed_dx | self.observed_dy

    @property
    def rms_u(self) -> float:
        return float(np.sqrt(np.mean(self.residual_u ** 2))) if self.residual_u.size else float("nan")

    @property
    def rms_v(self) -> float:
        return float(np.sqrt(np.mean(self.residual_v ** 2))) if self.residual_v.size else float("nan")

    def feature_rms(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature rms of (residual_u, residual_v), NaN when unobserved.

        A tangent feature has no residual in the coordinate it does not
        constrain, so that entry stays NaN rather than reporting a number
        about a coordinate the fit never used.
        """
        n_feat = self.model.feature_ids.size
        out_u = np.full(n_feat, np.nan)
        out_v = np.full(n_feat, np.nan)
        for out, (i, _), res in ((out_u, self.obs_u, self.residual_u),
                                 (out_v, self.obs_v, self.residual_v)):
            for f in range(n_feat):
                m = i == f
                if m.any():
                    out[f] = float(np.sqrt(np.mean(res[m] ** 2)))
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
    if r.size == 0:
        return np.ones_like(r)
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
                      observed: np.ndarray, y_free=None) -> None:
    """Move the mean of dy into y. Only gauge when every feature's y is free.

    `y_free` is the per-body freedom of y, which differs from the pin mask
    when a sphere carries no vertical observation at all: its height is
    then a nuisance value held fixed, and shifting it would move a number
    the labels never measured.
    """
    if not mask.dy or not observed.any():
        return
    if not (mask.features.size and mask.features.all()):
        return
    if y_free is not None and not np.all(y_free):
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


def _joint_pass(u, v, iu, ju, iv, jv, obs_dx, obs_dy, obs_any,
                col_u, col_v, col_r, model, mask, ct, sn,
                pc, pa, pb, base_wu, base_wv, prior_rows, rot_sigma,
                noise_px, free_body, free_y, free_r, *,
                iters, huber, damp):
    """One Gauss-Newton pass on every free parameter at once, in place.

    Rows are the u residuals, then the v residuals, then the prior rows;
    columns are increments of [a, b, c_k, dx, y, alpha_k, beta_k, dy, R,
    rot_horiz, rot_beam, rot_axis] (the radius block only with tangent
    features, the rotation blocks only when free). The (a, b, y, R)
    blocks are per BODY, so the members of a sphere share a centre and a
    diameter by construction rather than through a soft tie.

    The Jacobian is exact at the current model: with p' = R(-w) p, the
    derivative of p' with respect to a rotation increment dw is p' x dw,
    and the derivatives with respect to (a, b, y) are the rows of R
    applied to (cos, sin, 0), (sin, cos, 0) and (0, 0, 1). A tangent
    feature sits R m1 or R m2/|m2| off its body centre, so the radius
    enters both coordinates and the tilt coefficients pick up a term of
    order R (`AxisModel.tangent_scales`), which is why the u rows carry
    alpha and beta columns here and did not before. Returns the Huber
    weights of the u and v rows; residuals are recomputed exactly by the
    caller after the last pass.
    """
    tang = model.has_tangents
    n_body = model.n_bodies
    n_u, n_v = iu.size, iv.size
    n_ov_u, n_ov_v = int(obs_dx.sum()), int(obs_dy.sum())
    n_ov_r = int(obs_any.sum())
    kc, ka, kb = model.degrees
    rm = model.beam_rotation()
    body = model.body
    alpha_of = pa @ model.alpha_coef
    beta_of = pb @ model.beta_coef
    nm2_of = np.sqrt(1.0 + alpha_of ** 2 + beta_of ** 2)
    c_of = pc @ model.c_coef

    def geometry(i, j):
        g = SimpleNamespace()
        r = rm[j]
        s = model.a[i] * ct[j] + model.b[i] * sn[j]
        t = -model.a[i] * sn[j] + model.b[i] * ct[j]
        y_i = model.y[i]
        g.sp = r[:, 0, 0] * s + r[:, 0, 1] * t + r[:, 0, 2] * y_i
        g.tp = r[:, 1, 0] * s + r[:, 1, 1] * t + r[:, 1, 2] * y_i
        g.yp = r[:, 2, 0] * s + r[:, 2, 1] * t + r[:, 2, 2] * y_i
        g.d_da = [r[:, k, 0] * ct[j] - r[:, k, 1] * sn[j] for k in range(3)]
        g.d_db = [r[:, k, 0] * sn[j] + r[:, k, 1] * ct[j] for k in range(3)]
        g.d_dy = [r[:, k, 2] for k in range(3)]
        g.al, g.be = alpha_of[j], beta_of[j]
        su, sv = SIGN_U[model.kind[i]], SIGN_V[model.kind[i]]
        nm = nm2_of[j]
        rad = model.radius[body[i]]
        g.rad = rad
        g.du = su + sv * g.al / nm
        g.dv = su * g.al + sv * nm
        # d(tangent offset)/d(alpha, beta): of order R, so not negligible
        g.du_al = sv * rad * (1.0 + g.be ** 2) / nm ** 3
        g.dv_al = su * rad + sv * rad * g.al / nm
        g.du_be = -sv * rad * g.al * g.be / nm ** 3
        g.dv_be = sv * rad * g.be / nm
        g.v_of = lambda q: q[2] + g.al * q[0] + g.be * q[1]
        return g

    gu, gv = geometry(iu, ju), geometry(iv, jv)
    res_u = u[iu, ju] - (gu.sp + c_of[ju] + model.dx[ju] + gu.du * gu.rad)
    res_v = v[iv, jv] - (gv.yp + gv.al * gv.sp + gv.be * gv.tp
                         + model.dy[jv] + gv.dv * gv.rad)

    n_rad = n_body if tang else 0
    oa = 0
    ob = oa + n_body
    oc = ob + n_body
    odx = oc + kc + 1
    oy = odx + n_ov_u
    oal = oy + n_body
    obe = oal + ka + 1
    ody = obe + kb + 1
    orad = ody + n_ov_v
    orh = orad + n_rad
    orb = orh + (n_ov_r if mask.rot_horiz else 0)
    ora = orb + (n_ov_r if mask.rot_beam else 0)
    ncol = ora + (n_ov_r if mask.rot_axis else 0)
    bu, bv = body[iu], body[iv]
    ones_u, ones_v = np.ones(n_u), np.ones(n_v)

    u_cols = ([oa + bu, ob + bu]
              + [np.full(n_u, oc + k) for k in range(kc + 1)]
              + [odx + col_u[ju], oy + bu])
    u_vals = ([gu.d_da[0], gu.d_db[0]]
              + [pc[ju, k] for k in range(kc + 1)]
              + [ones_u, gu.d_dy[0]])
    v_cols = ([oa + bv, ob + bv, oy + bv]
              + [np.full(n_v, oal + k) for k in range(ka + 1)]
              + [np.full(n_v, obe + k) for k in range(kb + 1)]
              + [ody + col_v[jv]])
    v_vals = ([gv.v_of(gv.d_da), gv.v_of(gv.d_db), gv.v_of(gv.d_dy)]
              + [gv.sp * pa[jv, k] + gv.dv_al * pa[jv, k]
                 for k in range(ka + 1)]
              + [gv.tp * pb[jv, k] + gv.dv_be * pb[jv, k]
                 for k in range(kb + 1)]
              + [ones_v])
    if tang:
        u_cols.append(orad + bu)
        u_vals.append(gu.du)
        u_cols += [np.full(n_u, oal + k) for k in range(ka + 1)]
        u_vals += [gu.du_al * pa[ju, k] for k in range(ka + 1)]
        u_cols += [np.full(n_u, obe + k) for k in range(kb + 1)]
        u_vals += [gu.du_be * pb[ju, k] for k in range(kb + 1)]
        v_cols.append(orad + bv)
        v_vals.append(gv.dv)
    # rotation increments dw = (dh, db, da): d(s', t', y') = p' x dw. The
    # unknown of each block is the increment in units of sigma/noise_px
    # (see prior_rows), so its data columns carry that factor.
    unit_h, unit_b, unit_a = (sig / noise_px for sig in rot_sigma)
    if mask.rot_horiz:
        v_cols.append(orh + col_r[jv])
        v_vals.append(unit_h * gv.v_of((np.zeros(n_v), gv.yp, -gv.tp)))
    if mask.rot_beam:
        u_cols.append(orb + col_r[ju])
        u_vals.append(unit_b * -gu.yp)
        v_cols.append(orb + col_r[jv])
        v_vals.append(unit_b * gv.v_of((-gv.yp, np.zeros(n_v), gv.sp)))
    if mask.rot_axis:
        u_cols.append(ora + col_r[ju])
        u_vals.append(unit_a * gu.tp)
        v_cols.append(ora + col_r[jv])
        v_vals.append(unit_a * gv.v_of((gv.tp, -gv.sp, np.zeros(n_v))))
    rows = np.concatenate([np.repeat(np.arange(n_u), len(u_cols)),
                           np.repeat(n_u + np.arange(n_v), len(v_cols))])
    cols = np.concatenate([np.column_stack(u_cols).ravel(),
                           np.column_stack(v_cols).ravel()])
    vals = np.concatenate([np.column_stack(u_vals).ravel(),
                           np.column_stack(v_vals).ravel()])
    target = np.concatenate([res_u, res_v])
    row_off = n_u + n_v
    for flag, current, sigma, off in (
            (mask.rot_horiz, model.rot_horiz, rot_sigma[0], orh),
            (mask.rot_beam, model.rot_beam, rot_sigma[1], orb),
            (mask.rot_axis, model.rot_axis, rot_sigma[2], ora)):
        if not flag:
            continue
        pr, pcn, pv, pt = prior_rows(current, sigma, off, row_off, obs_any)
        rows = np.concatenate([rows, pr])
        cols = np.concatenate([cols, pcn])
        vals = np.concatenate([vals, pv])
        target = np.concatenate([target, pt])
        row_off += n_ov_r
    free = ([free_body, free_body, mask.c, np.full(n_ov_u, bool(mask.dx)),
             free_y, mask.alpha, mask.beta, np.full(n_ov_v, bool(mask.dy))]
            + ([free_r] if tang else [])
            + [np.ones(n_ov_r, bool) for flag in (mask.rot_horiz,
                                                  mask.rot_beam,
                                                  mask.rot_axis) if flag])
    free = np.concatenate(free)
    bw = (None if base_wu is None
          else np.concatenate([base_wu, base_wv]))
    x, _res, w = _masked_irls(rows, cols, vals, target, ncol, np.zeros(ncol),
                              free, iters=iters, huber=huber, damp=damp,
                              base_w=bw, n_data=n_u + n_v, huber_split=n_u)
    model.a = model.a + x[oa:ob][body]
    model.b = model.b + x[ob:oc][body]
    model.c_coef = model.c_coef + x[oc:odx]
    model.dx[obs_dx] += x[odx:oy]
    model.y = model.y + x[oy:oal][body]
    model.alpha_coef = model.alpha_coef + x[oal:obe]
    model.beta_coef = model.beta_coef + x[obe:ody]
    model.dy[obs_dy] += x[ody:orad]
    if tang:
        model.radius = model.radius + x[orad:orad + n_body]
    if mask.rot_horiz:
        model.rot_horiz[obs_any] += unit_h * x[orh:orh + n_ov_r]
    if mask.rot_beam:
        model.rot_beam[obs_any] += unit_b * x[orb:orb + n_ov_r]
    if mask.rot_axis:
        model.rot_axis[obs_any] += unit_a * x[ora:ora + n_ov_r]
    prof_a = rm[:, 0, 0] * ct - rm[:, 0, 1] * sn
    prof_b = rm[:, 0, 0] * sn + rm[:, 0, 1] * ct
    _regauge_horizontal(model, mask, obs_dx, profiles=(prof_a, prof_b))
    _regauge_vertical(model, mask, obs_dy, y_free=free_y)
    return w[:n_u], w[n_u:]


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def _body_label(model, b):
    """Name a body by the feature ids in it, which is what the table shows."""
    ids = model.feature_ids[np.flatnonzero(model.body == b)]
    return "+".join(str(int(f)) for f in ids)


def _one_sided(model, rows, sign_table):
    """(B,) bool: this body's observations all ride on the same side.

    A body observed only through, say, its left apex has u = u_c - R at
    every view, so moving the axis and the radius together is invisible.
    Two opposite members, or any member on the centre, break that.
    """
    n_body = model.n_bodies
    seen = [set() for _ in range(n_body)]
    for f in (np.unique(rows) if rows.size else ()):
        seen[model.body[f]].add(float(sign_table[model.kind[f]]))
    return np.array([len(s) == 1 and 0.0 not in s for s in seen], bool)


def _structural_warnings(model: AxisModel, mask: FreeMask,
                         valid_u: np.ndarray, valid_v: np.ndarray,
                         in_u: np.ndarray, in_v: np.ndarray) -> list[str]:
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
    n_feat = valid_u.shape[0]
    if n_feat < 2:
        out.append("W5: fewer than 2 features. Per-view shifts are not "
                   "constrained by a single feature.")
    elif n_feat < 3:
        out.append("W5: fewer than 3 features. Tilts (alpha, beta) need "
                   "features at different radii to separate from dy.")
    same = np.array_equal(valid_u, valid_v)
    axes = ((valid_u, ""),) if same else ((valid_u, " in u"), (valid_v, " in v"))
    for valid, where in axes:
        labels_per_view = valid.sum(axis=0)
        thin = int((labels_per_view == 1).sum())
        if thin and (mask.dx or mask.dy):
            out.append(
                f"W4: {thin} view(s) carry exactly one label{where}. Their "
                f"dx/dy fit that label exactly and mean nothing on their own.")
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
                f"W6: {n_obs_single} of {n_obs_total} labels{where} are the "
                f"ONLY label in their view, so free dx/dy absorb them and the "
                f"track geometry is mostly unconstrained. Label several "
                f"features in the SAME views (or fix dx/dy) to pin it down.")
    # A per-view rotation needs features at different s, t or y in the
    # SAME view to be measured at all (that per-feature lever is what
    # separates it from a shift). With one or two labels the free shift
    # plus the angle fit them exactly and only the prior sets the angle.
    if mask.any_rotation:
        per_view = (valid_u | valid_v).sum(axis=0)
        labelled = per_view[per_view > 0]
        few = int((labelled < 3).sum())
        if labelled.size and few / labelled.size > 0.5:
            out.append(
                f"W7: {few} of {labelled.size} labeled views carry fewer "
                f"than 3 labels while a per-view rotation is free. In those "
                f"views only the prior sets the angle, the labels carry no "
                f"information about it. Label three or more features in "
                f"the same views, or tighten the prior.")
    if not model.has_tangents:
        return out

    # -- tangent features -------------------------------------------------
    iu = np.nonzero(valid_u)[0]
    iv = np.nonzero(valid_v)[0]
    one_u = _one_sided(model, iu, SIGN_U)
    one_v = _one_sided(model, iv, SIGN_V)
    bodies_u = np.unique(model.body[iu]) if iu.size else np.zeros(0, int)
    if mask.dx and bodies_u.size and bool(np.all(one_u[bodies_u])):
        out.append(
            "W8: every horizontal observation comes from a sphere with "
            "tangents on ONE side only, so shifting the axis and every "
            "radius together is invisible and c is not a measurement. "
            "Label the opposite apex of at least one sphere, or add a "
            "point feature, or fix dx.")
    loose = [b for b in range(model.n_bodies)
             if (in_u[b] and one_u[b]) or (in_v[b] and one_v[b])]
    if loose:
        names = ", ".join(_body_label(model, b) for b in loose[:8])
        more = "" if len(loose) <= 8 else f" and {len(loose) - 8} more"
        out.append(
            f"W9: sphere(s) {names}{more} are labelled on one side only, so "
            f"their diameter and their centre trade off exactly. They still "
            f"constrain the shifts. Label the opposite apex, or type the "
            f"diameter into the R column, to make them count for the axis.")
    # a negative radius only MEANS anything where two opposite members
    # fixed its sign: on a one-sided sphere the sign is arbitrary and
    # saying "swapped" would be a false alarm (W9 already covers it)
    two_sided = ((in_u & ~one_u) | (in_v & ~one_v))
    neg = [b for b in range(model.n_bodies)
           if model.radius[b] < 0 and two_sided[b]]
    if neg:
        names = ", ".join(_body_label(model, b) for b in neg[:8])
        out.append(
            f"W10: sphere(s) {names} came out with a NEGATIVE radius, which "
            f"means their members are the wrong way round. laos is the apex "
            f"at the smaller u and taos the one at the smaller v.")
    if mask.rot_beam:
        body_has_v = np.zeros(model.n_bodies, bool)
        if iv.size:
            body_has_v[model.body[iv]] = True
        blind = [b for b in range(model.n_bodies) if not body_has_v[b]]
        if blind:
            names = ", ".join(_body_label(model, b) for b in blind[:8])
            out.append(
                f"W11: rot beam is free while sphere(s) {names} carry no "
                f"vertical observation. Their height is a nominal value read "
                f"off your clicks, not a measurement, and rot beam multiplies "
                f"it into u. Tick their centre box or fix rot beam.")
    return out


def split_validity(valid: np.ndarray, model: AxisModel,
                   measured_across: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(valid_u, valid_v): which labels observe u, and which observe v.

    A point label observes both. A tangent label observes its own axis
    always, and the across axis only when the feature's `use_center` is
    set: the extreme of a projected sphere sits exactly at the centre in
    the other coordinate, which is true for a sphere and false for a
    lopsided void, so it is opt in.

    `measured_across` (F, V) bool marks labels whose across coordinate was
    really measured. The 1D edge completer writes its own prediction into
    that coordinate instead of measuring it, so its labels must never
    become centre observations however `use_center` is set.
    """
    valid = np.asarray(valid, bool)
    kind = np.asarray(model.kind, int)
    su = SIGN_U[kind] != 0
    sv = SIGN_V[kind] != 0
    point = ~(su | sv)
    across = np.asarray(model.use_center, bool)[:, None]
    if measured_across is not None:
        across = across & np.asarray(measured_across, bool)
    use_u = point[:, None] | su[:, None] | (sv[:, None] & across)
    use_v = point[:, None] | sv[:, None] | (su[:, None] & across)
    return valid & use_u, valid & use_v


def _radius_presence(model: AxisModel, iu: np.ndarray, iv: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(in_u, in_v): (B,) bool, does each body's radius get a column here.

    Only a tangent feature carries the radius into an equation, and only
    in the stages where it actually has observations.
    """
    n_body = model.n_bodies
    in_u = np.zeros(n_body, bool)
    in_v = np.zeros(n_body, bool)
    tangent = np.asarray(model.kind, int) != POINT
    for rows, out in ((iu, in_u), (iv, in_v)):
        for f in np.unique(rows) if rows.size else ():
            if tangent[f]:
                out[model.body[f]] = True
    return in_u, in_v


def radius_couples_stages(model: AxisModel, valid_u: np.ndarray,
                          valid_v: np.ndarray) -> bool:
    """Does any sphere's radius appear in BOTH the u and the v system.

    It does when a sphere carries tangents on both axes, or when a tangent
    contributes a centre observation across its own axis. The two-stage
    solve cannot own a parameter shared by its stages, so that case goes
    to the joint Gauss-Newton pass instead.
    """
    iu = np.nonzero(valid_u)[0]
    iv = np.nonzero(valid_v)[0]
    in_u, in_v = _radius_presence(model, iu, iv)
    return bool(np.any(in_u & in_v))


def residuals(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
              model: AxisModel, *, valid_v: np.ndarray | None = None
              ) -> FitResult:
    """Evaluate the model against the labels without solving anything.

    This is what makes manual parameter overrides responsive: editing a
    value re-runs this, not the solver. `valid` is the u validity and
    `valid_v` the v one, which differ once tangent features exist.
    """
    valid_u = np.asarray(valid, bool)
    valid_v = valid_u if valid_v is None else np.asarray(valid_v, bool)
    iu, ju = np.nonzero(valid_u)
    iv, jv = np.nonzero(valid_v)
    u_pred, v_pred = model.predict()
    obs_dx = np.zeros(model.theta.size, bool)
    obs_dy = np.zeros(model.theta.size, bool)
    obs_dx[ju] = True
    obs_dy[jv] = True
    return FitResult(
        model=model, obs_u=(iu, ju), obs_v=(iv, jv),
        residual_u=u[iu, ju] - u_pred[iu, ju],
        residual_v=v[iv, jv] - v_pred[iv, jv],
        weight_u=np.ones(iu.size), weight_v=np.ones(iv.size),
        observed_dx=obs_dx, observed_dy=obs_dy,
    )


def _seed_nominal_heights(model: AxisModel, v: np.ndarray,
                          valid_u: np.ndarray,
                          body_has_v: np.ndarray) -> None:
    """Give a sphere with no vertical observation a nominal height.

    Nothing measures such a body's y: it reaches u only through rot_beam.
    Left at zero it would put a systematic rot_beam * y error into u, so
    it is seeded from the clicked v of its own labels minus the current
    dy, and then held fixed. It is a nuisance value, not a measurement,
    and a value already entered by hand is left alone.
    """
    for b in np.flatnonzero(~body_has_v):
        rows = np.flatnonzero(model.body == b)
        if not np.all(model.y[rows] == 0.0):
            continue
        m = valid_u[rows]
        if not m.any():
            continue
        cols = np.nonzero(m)[1]
        model.y[rows] = float(np.median(v[rows][m] - model.dy[cols]))


def solve_model(u: np.ndarray, v: np.ndarray, valid: np.ndarray,
                model: AxisModel, mask: FreeMask | None = None, *,
                valid_v: np.ndarray | None = None,
                iters: int = 4, huber: float = 3.0,
                damp: float = 1e-8,
                feature_weight: np.ndarray | None = None,
                rot_sigma=DEFAULT_ROT_SIGMA, noise_px: float = 1.0,
                passes: int | None = None) -> FitResult:
    """Fit the free parameters to the labels; fixed ones stay put.

    u, v: (F, V) label coordinates in RAW px (NaN/garbage where not valid).
    `valid` is the (F, V) validity of the u coordinate and `valid_v` that
    of the v coordinate; None means they are the same, which is the case
    for a model of plain point features. With tangent features they
    differ, and `split_validity` builds the pair from the feature kinds.
    The input `model` is not modified; the result carries an updated copy
    with dx/dy (and free rotations) on unlabeled views filled by
    interpolation (flagged via `observed_dx` / `observed_dy`).
    `feature_weight` (F,) is a relative per-feature prior weight, e.g.
    1/size for hand labels whose localization scales with the feature's
    size.

    `rot_sigma` = (sigma_horiz, sigma_beam, sigma_axis) in radians is the
    Gaussian prior rms of each per-view rotation and `noise_px` the label
    noise it is weighed against (module docstring). `passes` is the number
    of outer Gauss-Newton passes around the two linear stages; None means
    1 for the plain staged fit and 3 when a free rotation or a radius
    shared between the stages forces the joint pass.
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
    valid_u = np.asarray(valid, bool)
    valid_v = valid_u if valid_v is None else np.asarray(valid_v, bool)
    if valid_v.shape != valid_u.shape:
        raise ValueError("valid and valid_v must share shape")
    rot_sigma = tuple(float(x) for x in rot_sigma)
    if len(rot_sigma) != 3 or any(not (x > 0) for x in rot_sigma):
        raise ValueError("rot_sigma must be three positive radians")
    noise_px = float(noise_px)
    if not noise_px > 0:
        raise ValueError("noise_px must be positive")

    # The two coordinates have their own observation lists: a tangent
    # feature constrains one of them and says nothing about the other, so
    # a view carrying only tangent labels gets a dx column and no dy one.
    iu, ju = np.nonzero(valid_u)
    iv, jv = np.nonzero(valid_v)
    n_u, n_v = iu.size, iv.size
    if n_u + n_v == 0:
        raise ValueError("no valid observations")
    obs_dx = np.zeros(n_view, bool)
    obs_dy = np.zeros(n_view, bool)
    obs_dx[ju] = True
    obs_dy[jv] = True
    obs_any = obs_dx | obs_dy
    n_ov_u, n_ov_v = int(obs_dx.sum()), int(obs_dy.sum())
    n_ov_r = int(obs_any.sum())
    col_u = -np.ones(n_view, int)
    col_v = -np.ones(n_view, int)
    col_r = -np.ones(n_view, int)
    col_u[obs_dx] = np.arange(n_ov_u)
    col_v[obs_dy] = np.arange(n_ov_v)
    col_r[obs_any] = np.arange(n_ov_r)

    body = model.body
    n_body = model.n_bodies
    tang = model.has_tangents
    n_rad = n_body if tang else 0
    first_of_body = np.zeros(n_body, int)
    if n_feat:
        first_of_body[body[::-1]] = np.arange(n_feat)[::-1]
    in_u, in_v = _radius_presence(model, iu, iv)
    coupled = bool(np.any(in_u & in_v))
    # a body is free when every one of its members is unpinned
    free_body = np.ones(n_body, bool)
    np.logical_and.at(free_body, body, mask.features)
    body_has_v = np.zeros(n_body, bool)
    if n_v:
        body_has_v[body[iv]] = True
    free_y = free_body & body_has_v
    free_r = (in_u | in_v)
    if mask.radius is not None:
        free_r = free_r & np.asarray(mask.radius, bool)
    _seed_nominal_heights(model, v, valid_u, body_has_v)

    n_pass = ((3 if (mask.any_rotation or coupled) else 1)
              if passes is None else int(passes))

    theta = model.theta
    ct, sn = np.cos(theta), np.sin(theta)
    kc, ka, kb = model.degrees
    pc = poly_basis(theta, kc, model.theta_ref, model.theta_scale)
    pa = poly_basis(theta, ka, model.theta_ref, model.theta_scale)
    pb = poly_basis(theta, kb, model.theta_ref, model.theta_scale)
    fw = None if feature_weight is None else np.asarray(feature_weight, float)
    base_wu = None if fw is None else fw[iu]
    base_wv = None if fw is None else fw[iv]

    def prior_rows(current, sigma, col_off, row_off, observed):
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
        n = int(observed.sum())
        rows = row_off + np.arange(n)
        cols = col_off + np.arange(n)
        return rows, cols, np.ones(n), -current[observed] / unit

    u_pred, v_pred = model.predict()
    res_u = u[iu, ju] - u_pred[iu, ju]
    res_v = v[iv, jv] - v_pred[iv, jv]
    w_u, w_v = np.ones(n_u), np.ones(n_v)

    if mask.any_rotation or coupled:
        for _ in range(max(1, n_pass)):
            w_u, w_v = _joint_pass(
                u, v, iu, ju, iv, jv, obs_dx, obs_dy, obs_any,
                col_u, col_v, col_r, model, mask, ct, sn, pc, pa, pb,
                base_wu, base_wv, prior_rows, rot_sigma, noise_px,
                free_body, free_y, free_r,
                iters=iters, huber=huber, damp=damp)
        u_pred, v_pred = model.predict()
        res_u = u[iu, ju] - u_pred[iu, ju]
        res_v = v[iv, jv] - v_pred[iv, jv]
        n_pass = 0                                   # stages below skipped

    for _ in range(max(0, n_pass)):
        # ---- stage 1: u is linear in (a, b, c_k, dx, R) given the
        # rotations; rot_axis enters as a Gauss-Newton increment column
        # (value t'). The radius column is the exact du/dR of
        # `tangent_scales`, which is +-1 for a horizontal tangent.
        # A model of vertical tangents only has no u system at all.
        if n_u:
            rm = model.beam_rotation()
            r = rm[ju]
            y_cur = model.y[iu]
            s_nom = model.a[iu] * ct[ju] + model.b[iu] * sn[ju]
            t_nom = -model.a[iu] * sn[ju] + model.b[iu] * ct[ju]
            t_pr = r[:, 1, 0] * s_nom + r[:, 1, 1] * t_nom + r[:, 1, 2] * y_cur
            prof_a = rm[:, 0, 0] * ct - rm[:, 0, 1] * sn      # per view
            prof_b = rm[:, 0, 0] * sn + rm[:, 0, 1] * ct
            # column layout: [a (B), b (B), c (kc+1), dx (n_ov_u),
            #                 d_rot_axis (n_ov_u)?, R (B)?]
            off_ra = 2 * n_body + (kc + 1) + n_ov_u
            off_rad = off_ra + (n_ov_u if mask.rot_axis else 0)
            n1 = off_rad + n_rad
            bu = body[iu]
            ones_u = np.ones(n_u)
            col_blocks = ([bu, n_body + bu]
                          + [np.full(n_u, 2 * n_body + k) for k in range(kc + 1)]
                          + [2 * n_body + (kc + 1) + col_u[ju]])
            val_blocks = ([prof_a[ju], prof_b[ju]]
                          + [pc[ju, k] for k in range(kc + 1)]
                          + [ones_u])
            if mask.rot_axis:
                col_blocks.append(off_ra + col_u[ju])
                val_blocks.append(t_pr * (rot_sigma[2] / noise_px))
            if tang:
                col_blocks.append(off_rad + bu)
                val_blocks.append(model.tangent_scales()[0][iu, ju])
            per1 = len(col_blocks)
            rows1 = np.repeat(np.arange(n_u), per1)
            cols1 = np.column_stack(col_blocks).ravel()
            vals1 = np.column_stack(val_blocks).ravel()
            target1 = u[iu, ju] - r[:, 0, 2] * y_cur
            x0 = np.concatenate([model.a[first_of_body], model.b[first_of_body],
                                 model.c_coef, model.dx[obs_dx]]
                                + ([np.zeros(n_ov_u)] if mask.rot_axis else [])
                                + ([model.radius] if tang else []))
            free1 = np.concatenate([
                free_body, free_body,                  # a and b follow the pin
                mask.c,
                np.full(n_ov_u, bool(mask.dx)),
            ] + ([np.ones(n_ov_u, bool)] if mask.rot_axis else [])
              + ([free_r & in_u] if tang else []))
            if mask.rot_axis:
                pr, pcn, pv, pt = prior_rows(model.rot_axis, rot_sigma[2],
                                             off_ra, n_u, obs_dx)
                rows1 = np.concatenate([rows1, pr])
                cols1 = np.concatenate([cols1, pcn])
                vals1 = np.concatenate([vals1, pv])
                target1 = np.concatenate([target1, pt])
            x1, res_u, w_u = _masked_irls(rows1, cols1, vals1, target1, n1, x0,
                                          free1, iters=iters, huber=huber,
                                          damp=damp, base_w=base_wu, n_data=n_u)
            model.a = x1[:n_body][body]
            model.b = x1[n_body:2 * n_body][body]
            model.c_coef = x1[2 * n_body:2 * n_body + kc + 1]
            model.dx[obs_dx] = x1[2 * n_body + kc + 1:off_ra]
            if mask.rot_axis:
                model.rot_axis[obs_dx] += (rot_sigma[2] / noise_px) \
                    * x1[off_ra:off_ra + n_ov_u]
            if tang:
                model.radius = x1[off_rad:off_rad + n_body]
            _regauge_horizontal(model, mask, obs_dx, profiles=(prof_a, prof_b))

            # residuals after the regauge (predictions are gauge-invariant, but
            # be exact rather than clever)
            u_pred, _ = model.predict()
            res_u = u[iu, ju] - u_pred[iu, ju]

        # ---- stage 2: with (a, b) known, v is linear in (y, alpha_k,
        # beta_k, dy, R) given the rotations; rot_horiz and rot_beam enter
        # as increment columns (values -t' and s', with the tilt coupling).
        # A model of horizontal tangents only has no v system at all.
        if n_v:
            rm = model.beam_rotation()
            r = rm[jv]
            y_cur = model.y[iv]
            s_nom = model.a[iv] * ct[jv] + model.b[iv] * sn[jv]
            t_nom = -model.a[iv] * sn[jv] + model.b[iv] * ct[jv]
            s_pr = r[:, 0, 0] * s_nom + r[:, 0, 1] * t_nom + r[:, 0, 2] * y_cur
            t_pr = r[:, 1, 0] * s_nom + r[:, 1, 1] * t_nom + r[:, 1, 2] * y_cur
            y_pr = r[:, 2, 0] * s_nom + r[:, 2, 1] * t_nom + r[:, 2, 2] * y_cur
            alpha_cur = (pa @ model.alpha_coef)[jv]
            beta_cur = (pb @ model.beta_coef)[jv]
            # column layout: [y (B), alpha (ka+1), beta (kb+1), dy (n_ov_v),
            #                 d_rot_horiz (n_ov_v)?, d_rot_beam (n_ov_v)?, R (B)?]
            off_dy = n_body + (ka + 1) + (kb + 1)
            off_rh = off_dy + n_ov_v
            off_rb = off_rh + (n_ov_v if mask.rot_horiz else 0)
            off_rad2 = off_rb + (n_ov_v if mask.rot_beam else 0)
            n2 = off_rad2 + n_rad
            bv = body[iv]
            ones_v = np.ones(n_v)
            col_blocks = ([bv]
                          + [np.full(n_v, n_body + k) for k in range(ka + 1)]
                          + [np.full(n_v, n_body + ka + 1 + k)
                             for k in range(kb + 1)]
                          + [off_dy + col_v[jv]])
            val_blocks = ([r[:, 2, 2]]
                          + [s_pr * pa[jv, k] for k in range(ka + 1)]
                          + [t_pr * pb[jv, k] for k in range(kb + 1)]
                          + [ones_v])
            if mask.rot_horiz:
                col_blocks.append(off_rh + col_v[jv])
                val_blocks.append((rot_sigma[0] / noise_px)
                                  * (-t_pr + beta_cur * y_pr))
            if mask.rot_beam:
                col_blocks.append(off_rb + col_v[jv])
                val_blocks.append((rot_sigma[1] / noise_px)
                                  * (s_pr - alpha_cur * y_pr))
            if tang:
                col_blocks.append(off_rad2 + bv)
                val_blocks.append(model.tangent_scales()[1][iv, jv])
            per2 = len(col_blocks)
            rows2 = np.repeat(np.arange(n_v), per2)
            cols2 = np.column_stack(col_blocks).ravel()
            vals2 = np.column_stack(val_blocks).ravel()
            target2 = v[iv, jv] - (r[:, 2, 0] * s_nom + r[:, 2, 1] * t_nom)
            y0 = np.concatenate([model.y[first_of_body], model.alpha_coef,
                                 model.beta_coef, model.dy[obs_dy]]
                                + ([np.zeros(n_ov_v)] if mask.rot_horiz else [])
                                + ([np.zeros(n_ov_v)] if mask.rot_beam else [])
                                + ([model.radius] if tang else []))
            free2 = np.concatenate([
                free_y, mask.alpha, mask.beta,
                np.full(n_ov_v, bool(mask.dy)),
            ] + ([np.ones(n_ov_v, bool)] if mask.rot_horiz else [])
              + ([np.ones(n_ov_v, bool)] if mask.rot_beam else [])
              + ([free_r & in_v] if tang else []))
            row_off = n_v
            for flag, current, sigma, off in (
                    (mask.rot_horiz, model.rot_horiz, rot_sigma[0], off_rh),
                    (mask.rot_beam, model.rot_beam, rot_sigma[1], off_rb)):
                if not flag:
                    continue
                pr, pcn, pv, pt = prior_rows(current, sigma, off, row_off, obs_dy)
                rows2 = np.concatenate([rows2, pr])
                cols2 = np.concatenate([cols2, pcn])
                vals2 = np.concatenate([vals2, pv])
                target2 = np.concatenate([target2, pt])
                row_off += n_ov_v
            x2, res_v, w_v = _masked_irls(rows2, cols2, vals2, target2, n2, y0,
                                          free2, iters=iters, huber=huber,
                                          damp=damp, base_w=base_wv, n_data=n_v)
            model.y = x2[:n_body][body]
            model.alpha_coef = x2[n_body:n_body + ka + 1]
            model.beta_coef = x2[n_body + ka + 1:n_body + ka + 1 + kb + 1]
            model.dy[obs_dy] = x2[off_dy:off_rh]
            if mask.rot_horiz:
                model.rot_horiz[obs_dy] += (rot_sigma[0] / noise_px) \
                    * x2[off_rh:off_rh + n_ov_v]
            if mask.rot_beam:
                model.rot_beam[obs_dy] += (rot_sigma[1] / noise_px) \
                    * x2[off_rb:off_rb + n_ov_v]
            if tang:
                model.radius = x2[off_rad2:off_rad2 + n_body]
            _regauge_vertical(model, mask, obs_dy, y_free=free_y)

            _, v_pred = model.predict()
            res_v = v[iv, jv] - v_pred[iv, jv]

    # ---- fill the unlabeled views so exports have a complete curve ------
    if mask.dx:
        model.dx = fill_missing_shifts(model.dx, obs_dx, theta)
    if mask.dy:
        model.dy = fill_missing_shifts(model.dy, obs_dy, theta)
    for flag, name in ((mask.rot_horiz, "rot_horiz"),
                       (mask.rot_beam, "rot_beam"),
                       (mask.rot_axis, "rot_axis")):
        if flag:
            setattr(model, name,
                    fill_missing_shifts(getattr(model, name), obs_any, theta))

    return FitResult(
        model=model, obs_u=(iu, ju), obs_v=(iv, jv),
        residual_u=res_u, residual_v=res_v,
        weight_u=w_u, weight_v=w_v,
        observed_dx=obs_dx, observed_dy=obs_dy,
        warnings=_structural_warnings(model, mask, valid_u, valid_v,
                                      in_u, in_v),
    )
