"""Ground-truth tests for the semi-automatic track completion matcher.

Synthetic scans of gaussian blobs following known sinusoid tracks; the
matcher must recover the truth, and every rejection gate must fire on the
scenario it exists for. The single-pass bias test pins the re-cut lesson
so nobody "simplifies" it away.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from tktomo.tracking.autotrack import (  # noqa: E402
    AutoTrackParams,
    complete_track,
    coherence,
    highpass2d,
    match_patch,
    patch_size,
    structure_tensor,
)


def blob(frame, v, u, amplitude=1.0, sigma=2.5):
    ny, nx = frame.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    frame += amplitude * np.exp(-((yy - v) ** 2 + (xx - u) ** 2)
                                / (2 * sigma ** 2))


def synthetic_scan(n_views=60, ny=96, nx=160, jitter=0.0, seed=0,
                   tracks=(((40.0, -25.0, 80.0), 40.0),)):
    """Frames with blobs on tracks u = a cosT + b sinT + c at height y.

    tracks: (((a, b, c), y), ...). Returns (frames, theta, truth_u (F,V),
    truth_v (F,V)). A textured background exercises the highpass.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, np.pi, n_views)
    truth_u = np.zeros((len(tracks), n_views))
    truth_v = np.zeros((len(tracks), n_views))
    frames = []
    texture = 0.02 * rng.standard_normal((ny, nx))
    for j in range(n_views):
        frame = texture.copy()
        for f, ((a, b, c), y) in enumerate(tracks):
            u = a * np.cos(theta[j]) + b * np.sin(theta[j]) + c
            v = y + (jitter * rng.standard_normal() if jitter else 0.0)
            truth_u[f, j], truth_v[f, j] = u, v
            blob(frame, v, u)
        frames.append(frame.astype(np.float32))
    return frames, theta, truth_u, truth_v


def test_match_patch_subpixel_recovery():
    # noise INDEPENDENT between frames: a shared static background adds a
    # coherent zero-lag correlation peak that biases sub-pixel shifts
    # toward zero (the static-structure trap the slogger pipeline hit on
    # the real data; highpass and static-band cuts existed to fight it)
    rng = np.random.default_rng(1)
    a = 0.02 * rng.standard_normal((80, 80)).astype(np.float32)
    blob(a, 40.0, 40.0)
    for dv, du in ((0.3, -0.7), (1.6, 2.1), (-2.4, 0.9)):
        b = 0.02 * rng.standard_normal((80, 80)).astype(np.float32)
        blob(b, 40.0 + dv, 40.0 + du)
        hit = match_patch(a, b, (40.0, 40.0), (40.0, 40.0), 32, 6.0)
        assert hit is not None
        assert hit[0] == pytest.approx(40.0 + dv, abs=0.1)
        assert hit[1] == pytest.approx(40.0 + du, abs=0.1)
        assert hit[2] > 0.9


def test_match_patch_single_pass_bias():
    """The re-cut is load-bearing: a single pass underestimates a 3 px
    motion measurably, the iterated match does not."""
    a = np.zeros((80, 80), np.float32)
    blob(a, 40.0, 40.0)
    b = np.zeros((80, 80), np.float32)
    blob(b, 40.0, 43.0)
    one = match_patch(a, b, (40.0, 40.0), (40.0, 40.0), 32, 6.0, iters=1)
    full = match_patch(a, b, (40.0, 40.0), (40.0, 40.0), 32, 6.0, iters=3)
    assert one is not None and full is not None
    assert abs(one[1] - 43.0) > 0.15          # single pass: biased short
    assert abs(full[1] - 43.0) < 0.05         # re-cut converges


def test_match_patch_rejects_outside_search_box():
    a = np.zeros((80, 80), np.float32)
    blob(a, 40.0, 40.0)
    b = np.zeros((80, 80), np.float32)
    blob(b, 40.0, 52.0)                       # moved 12 px
    assert match_patch(a, b, (40.0, 40.0), (40.0, 40.0), 32, 5.0) is None


def test_highpass2d_kills_border_ramp():
    ny, nx = 60, 90
    ramp = (np.linspace(0, 30, ny)[:, None]
            + np.linspace(0, 20, nx)[None, :]).astype(np.float32)
    out = highpass2d(ramp, sigma=8.0)
    assert float(np.abs(out).max()) < 0.1     # including the borders
    # blur-only, for contrast, leaves a border artefact
    from scipy.ndimage import gaussian_filter
    blur_only = ramp - gaussian_filter(ramp, 8.0)
    assert float(np.abs(blur_only).max()) > 1.0


def test_structure_tensor_and_coherence():
    yy, xx = np.mgrid[0:40, 0:40]
    edge = (xx > 20).astype(float)            # vertical edge: gx only
    lam1, lam2, gxx, gyy = structure_tensor(edge)
    assert coherence(lam1, lam2) > 0.95
    assert gxx > 100 * max(gyy, 1e-12)
    corner = np.zeros((40, 40))
    blob(corner, 20.0, 20.0)
    lam1, lam2, _, _ = structure_tensor(corner)
    assert coherence(lam1, lam2) < 0.2


def test_patch_size_rule():
    assert patch_size(10.0) == 40
    assert patch_size(1.0) == 16              # floor
    assert patch_size(100.0) == 96            # ceiling


def default_params(**kw):
    return AutoTrackParams(**{"patch": 32, "search_radius": 6.0, **kw})


def _pc(params):
    """Single-anchor phase-correlation matcher, in the complete_track hook
    shape. Reproduces the removed default branch so the march, coherence
    gate, fb check and stop logic stay covered."""
    import numpy as np

    from tktomo.tracking.autotrack import match_patch

    def matcher(frames_hp, seeds_usable, view, pred, max_step):
        seeds = np.asarray(seeds_usable, float)
        a = seeds[int(np.argmin(np.abs(seeds[:, 0] - view)))]
        hit = match_patch(frames_hp[int(a[0])], frames_hp[view], (a[2], a[1]),
                          pred, params.patch, max_step,
                          upsample=params.upsample, iters=params.iters,
                          tol=params.tol)
        return None if hit is None else (hit[0], hit[1], hit[2])
    return matcher



def test_complete_track_recovers_sinusoid():
    frames, theta, truth_u, truth_v = synthetic_scan()
    hp = [highpass2d(f, 8.0) for f in frames]
    seed_views = [5, 30, 55]
    seeds = [(w, truth_u[0, w], truth_v[0, w]) for w in seed_views]
    result = complete_track(hp, theta, seeds, default_params(), matcher=_pc(default_params()))
    assert not result.warnings
    got = {al.view: al for al in result.labels}
    assert len(got) >= 0.8 * (theta.size - len(seed_views))
    errs = [np.hypot(al.u - truth_u[0, al.view], al.v - truth_v[0, al.view])
            for al in result.labels]
    assert float(np.sqrt(np.mean(np.square(errs)))) < 0.3
    assert all(al.quality >= 0.30 for al in result.labels)


def test_complete_track_two_seeds_linear_fallback():
    frames, theta, truth_u, truth_v = synthetic_scan(n_views=30)
    hp = [highpass2d(f, 8.0) for f in frames]
    seeds = [(8, truth_u[0, 8], truth_v[0, 8]),
             (20, truth_u[0, 20], truth_v[0, 20])]
    result = complete_track(hp, theta, seeds, default_params(), matcher=_pc(default_params()))
    # between the seeds the linear prediction is close enough to track
    inner = [al for al in result.labels if 8 < al.view < 20]
    assert len(inner) >= 8


def test_coherence_gate_refuses_edge():
    # a long straight filament instead of a blob
    theta = np.linspace(0, np.pi, 20)
    frames = []
    for _ in range(20):
        frame = np.zeros((80, 120), np.float32)
        frame[38:42, :] = 1.0                 # horizontal filament
        frames.append(frame)
    seeds = [(2, 60.0, 40.0), (10, 60.0, 40.0)]
    result = complete_track(frames, theta, seeds, default_params(), matcher=_pc(default_params()))
    assert not result.labels
    assert any("not trackable" in w for w in result.warnings)
    assert all(not rep["used"] for rep in result.seed_report)


def test_forward_backward_gate(monkeypatch):
    """The fb gate mechanics, tested directly: a backward round trip that
    misses the seed (or correlates poorly) removes the label; a clean
    round trip keeps it.

    Deliberately NOT an end-to-end lookalike scenario: on smooth
    synthetic gaussians, phase-correlation mixtures are forgiving and the
    Hann-windowed backward search cannot land far from its prediction, so
    such a test proves nothing about real slips (whose signature on
    textured data is a cratered backward correlation). Building one that
    "passes" would overstate what fb can do; the gate logic is what this
    module owns, so the gate logic is what gets pinned.
    """
    import tktomo.tracking.autotrack as at

    frames, theta, truth_u, truth_v = synthetic_scan(n_views=12)
    hp = [highpass2d(f, 8.0) for f in frames]
    seeds = [(3, truth_u[0, 3], truth_v[0, 3]),
             (8, truth_u[0, 8], truth_v[0, 8])]

    real = at.match_patch
    behavior = {"back_miss": 0.0, "back_q": 0.9}

    def fake(prev, cur, p_yx, q_yx, P, max_step, **kw):
        hit = real(prev, cur, p_yx, q_yx, P, max_step, **kw)
        # backward calls have the SEED frame as `cur`; sabotage those
        is_backward = any(cur is hp[s[0]] for s in seeds)
        if hit is not None and is_backward:
            return (q_yx[0] + behavior["back_miss"], q_yx[1],
                    behavior["back_q"])
        return hit

    monkeypatch.setattr(at, "match_patch", fake)
    params = default_params()

    behavior.update(back_miss=0.0, back_q=0.9)
    clean = at.complete_track(hp, theta, seeds, params, matcher=_pc(params))
    assert clean.labels                       # good round trips keep labels

    behavior.update(back_miss=5.0, back_q=0.9)
    missed = at.complete_track(hp, theta, seeds, params, matcher=_pc(params))
    assert not missed.labels                  # round trip misses: rejected

    behavior.update(back_miss=0.0, back_q=0.05)
    cratered = at.complete_track(hp, theta, seeds, params, matcher=_pc(params))
    assert not cratered.labels                # backward corr craters: rejected

    behavior.update(back_miss=5.0, back_q=0.9)
    off_params = AutoTrackParams(**{**params.__dict__, "fb_check": False})
    off = at.complete_track(hp, theta, seeds, off_params,
                            matcher=_pc(off_params))
    assert off.labels                         # gate off: forward rules


def test_stop_after_consecutive_failures():
    frames, theta, truth_u, truth_v = synthetic_scan()
    hp = [highpass2d(f, 8.0) for f in frames]
    # corrupt three consecutive frames to the right of the middle seed
    for k in (40, 41, 42):
        hp[k] = np.zeros_like(hp[k])
    seeds = [(5, truth_u[0, 5], truth_v[0, 5]),
             (30, truth_u[0, 30], truth_v[0, 30])]
    result = complete_track(hp, theta, seeds, default_params(), matcher=_pc(default_params()))
    got = {al.view for al in result.labels}
    assert not any(v in got for v in (40, 41, 42))
    assert not any(v in got for v in range(43, 60))   # direction ended
    assert 39 in got                                   # tracked up to it

    # a SINGLE corrupted frame is skipped, the march continues past it
    hp2 = [highpass2d(f, 8.0) for f in frames]
    hp2[40] = np.zeros_like(hp2[40])
    result2 = complete_track(hp2, theta, seeds, default_params(), matcher=_pc(default_params()))
    got2 = {al.view for al in result2.labels}
    assert 40 not in got2
    assert 41 in got2 and 45 in got2


def test_complete_track_cancel():
    frames, theta, truth_u, truth_v = synthetic_scan()
    hp = [highpass2d(f, 8.0) for f in frames]
    seeds = [(5, truth_u[0, 5], truth_v[0, 5]),
             (30, truth_u[0, 30], truth_v[0, 30])]
    result = complete_track(hp, theta, seeds, default_params(),
                            matcher=_pc(default_params()),
                            cancelled=lambda: True)
    assert result.cancelled
    assert not result.labels


def test_seed_report_records_stops_and_coherence():
    frames, theta, truth_u, truth_v = synthetic_scan()
    hp = [highpass2d(f, 8.0) for f in frames]
    seeds = [(5, truth_u[0, 5], truth_v[0, 5]),
             (30, truth_u[0, 30], truth_v[0, 30])]
    result = complete_track(hp, theta, seeds, default_params(), matcher=_pc(default_params()))
    assert len(result.seed_report) == 2
    for rep in result.seed_report:
        assert rep["used"]
        assert rep["coherence"] < 0.4
        assert "left_stop" in rep and "right_stop" in rep


def test_complete_track_custom_matcher_is_used_and_thresholded():
    """A matcher gets all usable seeds and its quality is gated by min_corr."""
    frames, theta, tu, tv = synthetic_scan(n_views=30)
    views = [0, 10, 20, 29]
    seeds = [(j, tu[0, j], tv[0, j]) for j in views]
    calls = []

    def oracle(frames_hp, seeds_usable, view, pred, max_step):
        calls.append(len(seeds_usable))
        return tv[0, view], tu[0, view], 0.9 if view % 2 else 0.1

    params = default_params(min_corr=0.5, fb_check=False)
    frames_hp = [highpass2d(f) for f in frames]
    res = complete_track(frames_hp, theta, seeds, params, matcher=oracle)
    assert calls and all(c == len(views) for c in calls)
    got = {al.view for al in res.labels}
    assert got and all(j % 2 == 1 for j in got)
    for al in res.labels:
        assert abs(al.u - tu[0, al.view]) < 1e-9
        assert abs(al.v - tv[0, al.view]) < 1e-9


def test_complete_track_requires_a_matcher():
    frames, theta, tu, tv = synthetic_scan(n_views=20)
    hp = [highpass2d(f) for f in frames]
    seeds = [(2, tu[0, 2], tv[0, 2]), (10, tu[0, 10], tv[0, 10])]
    import pytest

    with pytest.raises(ValueError, match="requires a matcher"):
        complete_track(hp, theta, seeds, default_params())
