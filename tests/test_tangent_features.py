"""Sphere tangent features: the geometry, the solver, the degeneracies.

The claim these tests defend is that a tangent label is an exact
observation of its body's centre plus a signed radius, under the tilts and
the per-view rotations alike, and that two opposite members of one sphere
are worth exactly as much as a point feature at its centre.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.tracking.model import (
    BAOS,
    poly_basis,
    LAOS,
    POINT,
    RAOS,
    TAOS,
    AxisModel,
    FreeMask,
    radius_couples_stages,
    solve_model,
    split_validity,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_model(kinds, bodies, radii, *, n_view=48, span_deg=180.0,
               degrees=(1, 1, 0), rot_rms_deg=0.0, seed=1):
    """A ground-truth model with the given feature kinds and bodies."""
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(np.linspace(0.0, span_deg, n_view, endpoint=False))
    n_feat = len(kinds)
    m = AxisModel.blank(theta, np.arange(n_feat), degrees,
                        kind=np.asarray(kinds, int),
                        body=np.asarray(bodies, int),
                        radius=np.asarray(radii, float))
    n_body = m.n_bodies
    phi = rng.uniform(0.0, 2 * np.pi, n_body)
    r = rng.uniform(40.0, 100.0, n_body)
    m.a = (r * np.cos(phi))[m.body]
    m.b = (r * np.sin(phi))[m.body]
    m.y = rng.uniform(-40.0, 40.0, n_body)[m.body]
    m.c_coef = np.array([435.0, 6.0])[:m.c_coef.size]
    m.alpha_coef = np.array([-0.01, 0.004])[:m.alpha_coef.size]
    m.beta_coef = np.array([0.006, -0.002])[:m.beta_coef.size]
    m.dx = 3.0 * np.sin(2.3 * theta) + rng.normal(0.0, 0.4, n_view)
    m.dy = 2.0 * np.cos(1.7 * theta) + rng.normal(0.0, 0.3, n_view)
    if rot_rms_deg:
        s = np.deg2rad(rot_rms_deg)
        for name in ("rot_horiz", "rot_beam", "rot_axis"):
            w = rng.normal(0.0, s, n_view)
            setattr(m, name, w - w.mean())
    return m


def canonical(model):
    """Regauge a model in place against the all-free gauge basis.

    The truth has to be in the same gauge as the fit before their a, b and
    c can be compared at all (the solver regauges after every solve).
    """
    kc = model.degrees[0]
    pc = poly_basis(model.theta, kc, model.theta_ref, model.theta_scale)
    g = np.column_stack([pc, np.cos(model.theta), np.sin(model.theta)])
    p, *_ = np.linalg.lstsq(g, model.dx, rcond=None)
    model.c_coef += p[:kc + 1]
    model.a += p[kc + 1]
    model.b += p[kc + 2]
    model.dx -= g @ p
    m = float(np.mean(model.dy))
    model.dy -= m
    model.y += m
    return model


def sample_labels(model, *, frac=0.7, noise=0.02, seed=2):
    """Labels from the model's own markers, with a random validity mask."""
    rng = np.random.default_rng(seed)
    u, v = model.predict()
    valid = rng.random(u.shape) < frac
    valid[:, 0] = True                       # keep every feature observed
    return (u + rng.normal(0.0, noise, u.shape),
            v + rng.normal(0.0, noise, v.shape), valid)


def unit_sphere(n):
    """n roughly uniform directions (Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rot_rms_deg", [0.0, 1.5])
def test_tangent_points_are_the_silhouette_extremes(rot_rms_deg):
    """The closed form equals the extremum derived from `project` alone.

    u and v are affine in the object coordinates (a, b, y), which are an
    orthonormal frame, so a sphere of radius R has u extremal at
    centre +- R g_u/|g_u| with g_u the gradient, and v there follows. The
    gradients come from finite differences of `project`, so this derives
    the tangent point without using the formula it checks, exactly, with
    the tilts and the per-view rotations on.
    """
    radius = 37.0
    m = make_model([LAOS, RAOS, TAOS, BAOS], [0, 0, 0, 0], [radius],
                   n_view=9, rot_rms_deg=rot_rms_deg, seed=5)
    views = np.arange(m.theta.size)
    a0, b0, y0 = m.a[0], m.b[0], m.y[0]
    u0, v0 = m.project(a0, b0, y0, views=views)
    gu, gv = [], []
    for da, db, dy in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        uu, vv = m.project(a0 + da, b0 + db, y0 + dy, views=views)
        gu.append(uu[0] - u0[0])
        gv.append(vv[0] - v0[0])
    gu, gv = np.asarray(gu), np.asarray(gv)          # (3, V)

    def extreme(g, other, sign):
        """(value, other coordinate) at the extremum of g over the sphere."""
        d = sign * radius * g / np.linalg.norm(g, axis=0)
        return (g * d).sum(axis=0), (other * d).sum(axis=0)

    u_pred, v_pred = m.predict()
    for row, (g, other, sign, pred, cross) in enumerate((
            (gu, gv, -1.0, u_pred, v_pred),          # laos
            (gu, gv, +1.0, u_pred, v_pred),          # raos
            (gv, gu, -1.0, v_pred, u_pred),          # taos
            (gv, gu, +1.0, v_pred, u_pred))):        # baos
        along, across = extreme(g, other, sign)
        base_along = u0[0] if row < 2 else v0[0]
        base_across = v0[0] if row < 2 else u0[0]
        assert pred[row] == pytest.approx(base_along + along, abs=1e-9)
        assert cross[row] == pytest.approx(base_across + across, abs=1e-9)


def test_tangent_points_match_a_sampled_silhouette():
    """A coarse sanity check against real points on the sphere."""
    radius = 37.0
    m = make_model([LAOS, RAOS, TAOS, BAOS], [0, 0, 0, 0], [radius],
                   n_view=5, rot_rms_deg=1.5, seed=5)
    n = unit_sphere(200_000)
    views = np.arange(m.theta.size)
    u_s, v_s = m.project(m.a[0] + radius * n[:, 0],
                         m.b[0] + radius * n[:, 1],
                         m.y[0] + radius * n[:, 2], views=views)
    u_pred, v_pred = m.predict()
    for j in range(views.size):
        assert u_pred[0, j] == pytest.approx(u_s[:, j].min(), abs=5e-3)
        assert u_pred[1, j] == pytest.approx(u_s[:, j].max(), abs=5e-3)
        assert v_pred[2, j] == pytest.approx(v_s[:, j].min(), abs=5e-3)
        assert v_pred[3, j] == pytest.approx(v_s[:, j].max(), abs=5e-3)


def test_radius_is_untouched_by_tilts_and_rotations():
    """A sphere projects to a disc of radius R whatever the orientation."""
    m = make_model([LAOS, RAOS], [0, 0], [25.0], n_view=12,
                   rot_rms_deg=2.0, seed=7)
    u, _ = m.predict()
    half_width = (u[1] - u[0]) / 2.0
    assert np.allclose(half_width, 25.0, atol=1e-12)


def test_point_model_has_no_tangents_and_no_coupling():
    m = make_model([POINT] * 4, [0, 1, 2, 3], [0.0] * 4)
    assert not m.has_tangents
    valid = np.ones((4, m.theta.size), bool)
    vu, vv = split_validity(valid, m)
    assert np.array_equal(vu, valid)
    assert np.array_equal(vv, valid)
    assert not radius_couples_stages(m, vu, vv)


# ---------------------------------------------------------------------------
# validity split
# ---------------------------------------------------------------------------

def test_split_validity_follows_the_kinds():
    m = make_model([POINT, LAOS, RAOS, TAOS], [0, 1, 1, 2], [0.0, 10.0, 8.0])
    valid = np.ones((4, m.theta.size), bool)
    vu, vv = split_validity(valid, m)
    assert vu[0].all() and vv[0].all()          # point: both
    assert vu[1].all() and not vv[1].any()      # laos: u only
    assert vu[2].all() and not vv[2].any()
    assert vv[3].all() and not vu[3].any()      # taos: v only


def test_use_center_opens_the_across_axis():
    m = make_model([LAOS, RAOS], [0, 0], [12.0])
    m.use_center = np.array([True, False])
    valid = np.ones((2, m.theta.size), bool)
    vu, vv = split_validity(valid, m)
    assert vv[0].all()
    assert not vv[1].any()
    # an auto label that never measured the across coordinate cannot
    # become a centre observation however use_center is set
    measured = np.ones_like(valid)
    measured[0, :5] = False
    vu2, vv2 = split_validity(valid, m, measured_across=measured)
    assert not vv2[0, :5].any()
    assert vv2[0, 5:].all()
    assert np.array_equal(vu2, vu)


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def test_linked_pair_recovers_centre_radius_and_axis():
    kinds = [LAOS, RAOS] * 5
    bodies = [b for b in range(5) for _ in range(2)]
    radii = [12.0, 31.0, 47.0, 8.0, 25.0]
    truth = canonical(make_model(kinds, bodies, radii, n_view=60, seed=3))
    u, v, valid = sample_labels(truth, noise=0.02)
    vu, vv = split_validity(valid, truth)

    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    m = fit.model
    assert np.abs(fit.residual_u).max() < 0.2
    assert m.radius == pytest.approx(truth.radius, abs=0.05)
    assert m.center_at_mean_theta() == pytest.approx(
        truth.center_at_mean_theta(), abs=0.2)
    assert m.a == pytest.approx(truth.a, abs=0.3)
    assert m.b == pytest.approx(truth.b, abs=0.3)


def test_pair_is_worth_the_same_as_a_point_at_the_centre():
    """The midpoint of two opposite tangents IS the projected centre."""
    kinds = [LAOS, RAOS] * 4
    bodies = [b for b in range(4) for _ in range(2)]
    radii = [15.0, 40.0, 9.0, 28.0]
    truth = make_model(kinds, bodies, radii, n_view=60, seed=11)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)

    # the same object, labelled as four point features at the centres
    pts = AxisModel.blank(truth.theta, np.arange(4), truth.degrees)
    u_mid = 0.5 * (u[1::2] + u[0::2])
    v_mid = 0.5 * (v[1::2] + v[0::2])
    ok = np.ones_like(u_mid, bool)
    fit_p = solve_model(u_mid, v_mid, ok, pts, FreeMask.all_free(pts))

    assert fit.model.center_at_mean_theta() == pytest.approx(
        fit_p.model.center_at_mean_theta(), abs=1e-6)
    assert fit.model.dx == pytest.approx(fit_p.model.dx, abs=1e-6)
    assert fit.model.a[0::2] == pytest.approx(fit_p.model.a, abs=1e-6)


def test_one_side_labelled_still_fits_the_shifts():
    """A lone apex carries dx and loses the axis, which is W9 and W8."""
    kinds = [LAOS] * 5
    bodies = list(range(5))
    truth = make_model(kinds, bodies, [10.0, 20.0, 30.0, 14.0, 25.0],
                       n_view=60, seed=4)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    assert np.abs(fit.residual_u).max() < 1e-6
    codes = " ".join(fit.warnings)
    assert "W8" in codes
    assert "W9" in codes


def test_opposite_pair_does_not_warn_about_the_radius():
    kinds = [LAOS, RAOS, LAOS, RAOS]
    truth = make_model(kinds, [0, 0, 1, 1], [18.0, 33.0], n_view=60, seed=6)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    codes = " ".join(fit.warnings)
    assert "W8" not in codes
    assert "W9" not in codes


def test_swapped_members_give_a_negative_radius_and_warn():
    kinds = [LAOS, RAOS, LAOS, RAOS]
    truth = make_model(kinds, [0, 0, 1, 1], [18.0, 33.0], n_view=60, seed=6)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    u[[0, 1]] = u[[1, 0]]                    # the user clicked them the wrong
    v[[0, 1]] = v[[1, 0]]                    # way round on the first sphere
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    assert fit.model.radius[0] < 0
    assert any("W10" in w for w in fit.warnings)


def test_one_sided_sphere_does_not_claim_its_members_are_swapped():
    """A lone apex has no sign to get wrong, so W10 must stay quiet."""
    truth = make_model([LAOS] * 4, list(range(4)), [10.0, 20.0, 30.0, 14.0],
                       n_view=60, seed=14)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    assert any("W9" in w for w in fit.warnings)
    assert not any("W10" in w for w in fit.warnings)


def test_vertical_pair_recovers_height_and_radius():
    kinds = [TAOS, BAOS] * 4
    bodies = [b for b in range(4) for _ in range(2)]
    radii = [11.0, 26.0, 40.0, 17.0]
    truth = make_model(kinds, bodies, radii, n_view=60, seed=8)
    u, v, valid = sample_labels(truth, noise=0.02)
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    # nothing measures u here, so (a, b) are given rather than fitted
    start.a, start.b = truth.a.copy(), truth.b.copy()
    mask = FreeMask.all_free(start)
    mask.dx = False                  # no horizontal observation at all
    fit = solve_model(u, v, vu, start, mask, valid_v=vv)
    assert np.abs(fit.residual_v).max() < 0.2
    assert fit.model.radius == pytest.approx(truth.radius, abs=0.1)
    assert not fit.observed_dx.any()
    assert fit.observed_dy.all()


def test_all_four_tangents_share_one_radius_through_the_joint_pass():
    kinds = [LAOS, RAOS, TAOS, BAOS] * 3
    bodies = [b for b in range(3) for _ in range(4)]
    radii = [19.0, 34.0, 12.0]
    truth = make_model(kinds, bodies, radii, n_view=60, seed=9)
    u, v, valid = sample_labels(truth, noise=0.02)
    vu, vv = split_validity(valid, truth)
    assert radius_couples_stages(truth, vu, vv)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    start.a, start.b, start.y = truth.a.copy(), truth.b.copy(), truth.y.copy()
    start.radius = truth.radius.copy()
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    assert fit.model.radius == pytest.approx(truth.radius, abs=0.15)
    assert np.abs(fit.residual_u).max() < 0.25
    assert np.abs(fit.residual_v).max() < 0.25


def test_a_view_with_only_tangent_labels_gets_no_dy_column():
    kinds = [POINT, LAOS, RAOS]
    truth = make_model(kinds, [0, 1, 1], [0.0, 22.0], n_view=30, seed=10)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    valid[0, 5] = False                      # view 5 keeps only the sphere
    vu, vv = split_validity(valid, truth)
    start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                            kind=truth.kind, body=truth.body)
    fit = solve_model(u, v, vu, start, FreeMask.all_free(start), valid_v=vv)
    assert fit.observed_dx[5]
    assert not fit.observed_dy[5]
    assert fit.observed_dy.sum() == truth.theta.size - 1


def test_centre_observation_recovers_the_sphere_height():
    kinds = [LAOS, RAOS] * 3
    bodies = [b for b in range(3) for _ in range(2)]
    truth = make_model(kinds, bodies, [14.0, 29.0, 9.0], n_view=60, seed=12)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    off = np.zeros(6, bool)
    on = np.ones(6, bool)
    for use, expect_height in ((off, False), (on, True)):
        start = AxisModel.blank(truth.theta, truth.feature_ids, truth.degrees,
                                kind=truth.kind, body=truth.body,
                                use_center=use)
        vu, vv = split_validity(valid, start)
        mask = FreeMask.all_free(start)
        fit = solve_model(u, v, vu, start, mask, valid_v=vv)
        if expect_height:
            assert fit.obs_v[0].size > 0
            assert np.abs(fit.residual_v).max() < 1e-6
        else:
            assert fit.obs_v[0].size == 0
            assert not fit.observed_dy.any()


def test_pinning_one_member_pins_the_whole_sphere():
    kinds = [LAOS, RAOS, POINT, POINT]
    truth = make_model(kinds, [0, 0, 1, 2], [21.0, 0.0, 0.0], n_view=60,
                       seed=13)
    u, v, valid = sample_labels(truth, frac=1.0, noise=0.0)
    vu, vv = split_validity(valid, truth)
    start = truth.copy()
    start.dx = np.zeros_like(start.dx)
    start.dy = np.zeros_like(start.dy)
    mask = FreeMask.all_free(start)
    mask.features[0] = False                 # pin only the laos
    fit = solve_model(u, v, vu, start, mask, valid_v=vv)
    assert fit.model.a[0] == pytest.approx(truth.a[0], abs=1e-9)
    assert fit.model.a[1] == pytest.approx(truth.a[0], abs=1e-9)
