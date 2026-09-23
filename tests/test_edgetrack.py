"""The 1D edge matcher for sphere tangent features.

A tangent label has one real coordinate. These tests check that the
matcher finds it, that it declines when there is nothing to find, and that
the coordinate it does not measure comes back as the caller's own
prediction rather than an invention.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.tracking.autotrack import (
    AutoTrackJob,
    AutoTrackParams,
    highpass2d,
    run_autotrack,
)
from tktomo.tracking.edgetrack import (
    EdgeMatcher,
    axis_for_kind,
    edge_profile,
)
from tktomo.tracking.model import BAOS, LAOS, POINT, RAOS, TAOS


def disc(frame, cv, cu, radius, amp=1.0, wall=1.5):
    """A bright shell, the way a bubble looks in a phase projection."""
    ny, nx = frame.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cv, xx - cu)
    frame += amp * np.exp(-0.5 * ((r - radius) / wall) ** 2)
    return frame


def disc_scan(n_views=40, ny=96, nx=160, centre=(48.0, 80.0), amp=25.0,
              radius=18.0, seed=0):
    """A sphere going round: its edges move on a sinusoid, its size does not."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, np.pi, n_views, endpoint=False)
    cu = centre[1] + amp * np.cos(theta)
    cv = np.full(n_views, centre[0])
    frames = np.zeros((n_views, ny, nx), np.float32)
    for j in range(n_views):
        disc(frames[j], cv[j], cu[j], radius)
        frames[j] += rng.normal(0.0, 0.01, (ny, nx)).astype(np.float32)
    return theta, frames, cv, cu, radius


def highpassed(frames, sigma=12.0):
    return np.stack([highpass2d(f, sigma) for f in frames])


def test_axis_for_kind():
    assert axis_for_kind(POINT) is None
    assert axis_for_kind(LAOS) == "u"
    assert axis_for_kind(RAOS) == "u"
    assert axis_for_kind(TAOS) == "v"
    assert axis_for_kind(BAOS) == "v"


def test_edge_profile_shape_and_bounds():
    frame = np.arange(40 * 60, dtype=float).reshape(40, 60)
    prof, origin = edge_profile(frame, 20, 30, "u", 21, 5)
    assert prof.shape == (21,)
    assert origin == 20.0
    prof, origin = edge_profile(frame, 20, 30, "v", 11, 3)
    assert prof.shape == (11,)
    assert origin == 15.0
    assert edge_profile(frame, 1, 30, "u", 21, 9)[0] is None
    assert edge_profile(frame, 20, 2, "u", 21, 5)[0] is None


@pytest.mark.parametrize("kind,sign", [(LAOS, -1.0), (RAOS, +1.0)])
def test_matcher_recovers_the_horizontal_apex(kind, sign):
    theta, frames, cv, cu, radius = disc_scan()
    hp = highpassed(frames)
    matcher = EdgeMatcher(axis="u")
    seed_view = 0
    seeds = [(seed_view, cu[seed_view] + sign * radius, cv[seed_view])]
    errors = []
    for view in range(1, 8):
        truth_u = cu[view] + sign * radius
        # predict badly on purpose: the matcher has to find the edge
        pred = (cv[view], truth_u - 1.3)
        hit = matcher(hp, seeds, view, pred, 6.0)
        assert hit is not None
        v_hit, u_hit, q = hit
        assert v_hit == pytest.approx(pred[0])       # never measured
        assert q > 0.5
        errors.append(u_hit - truth_u)
    assert np.abs(errors).max() < 0.4


@pytest.mark.parametrize("kind,sign", [(TAOS, -1.0), (BAOS, +1.0)])
def test_matcher_recovers_the_vertical_apex(kind, sign):
    theta, frames, cv, cu, radius = disc_scan()
    hp = highpassed(frames)
    matcher = EdgeMatcher(axis="v")
    seeds = [(0, cu[0], cv[0] + sign * radius)]
    for view in (3, 6):
        truth_v = cv[view] + sign * radius
        pred = (truth_v - 1.1, cu[view])
        hit = matcher(hp, seeds, view, pred, 6.0)
        assert hit is not None
        v_hit, u_hit, _ = hit
        assert u_hit == pytest.approx(pred[1])       # never measured
        assert v_hit == pytest.approx(truth_v, abs=0.5)


def test_matcher_declines_a_flat_patch():
    theta, frames, cv, cu, radius = disc_scan()
    flat = np.zeros_like(frames)
    matcher = EdgeMatcher(axis="u")
    seeds = [(0, cu[0] - radius, cv[0])]
    assert matcher(flat, seeds, 3, (cv[3], cu[3] - radius), 6.0) is None


def test_matcher_refuses_a_match_outside_the_search_box():
    theta, frames, cv, cu, radius = disc_scan()
    hp = highpassed(frames)
    matcher = EdgeMatcher(axis="u")
    seeds = [(0, cu[0] - radius, cv[0])]
    view = 12
    truth_u = cu[view] - radius
    assert matcher(hp, seeds, view, (cv[view], truth_u - 9.0), 2.0) is None


def test_run_autotrack_uses_the_edge_matcher_for_a_tangent():
    theta, frames, cv, cu, radius = disc_scan(n_views=24)
    seeds = tuple((j, float(cu[j] - radius), float(cv[j]))
                  for j in (0, 8, 16))
    params = AutoTrackParams(patch=40, search_radius=6.0, min_corr=0.3,
                             hp_sigma=12.0)

    def refuse(*_args, **_kw):           # the blob matcher must not be used
        raise AssertionError("the blob matcher was called for a tangent")

    job = AutoTrackJob(fid=0, seeds=seeds, params=params, track_bin=1,
                       kind=LAOS)
    (fid, result), = run_autotrack(frames, theta, [job], hp_sigma=12.0,
                                   matcher=refuse)
    assert fid == 0
    assert result.stats["matcher"] == "edge-1d"
    assert len(result.labels) > 8
    err = [al.u - (cu[al.view] - radius) for al in result.labels]
    assert np.abs(err).max() < 1.0
    # the across coordinate is the prediction, held at the seed height
    assert np.allclose([al.v for al in result.labels], cv[0], atol=1e-6)
