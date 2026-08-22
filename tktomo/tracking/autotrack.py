"""Semi-automatic track completion from manual seed labels.

The user's manual labels are the anchors; this module fills the views
between and around them by template matching, guided by the sinusoid the
seeds imply, and STOPS honestly when the template no longer resembles the
image. The core matcher (`match_patch`, `highpass2d`, `structure_tensor`)
is ported from the slogger graphite-ball pipeline's `feature_tracks.py` /
`prealign.py`, where every design choice was measured on the same
dataset this app was built for. The load-bearing lessons, kept verbatim
in the docstrings below:

- Iterative RE-CUT of the search patch is not optional (a single pass
  systematically underestimates motion by 13 percent).
- Anchored templates decorrelate honestly (correlation 0.65 at one view
  of separation, 0.23 at ten, 0.06 at forty), so tracks END rather than
  wander; chained templates drift smoothly and plausibly, which is the
  dangerous kind of wrong. Templates here are always cut at a MANUAL
  seed, never updated.
- A patch on an edge or filament can slide along it with a confident
  correlation peak; the structure-tensor coherence (> 0.4) names those
  templates untrackable up front.
- Bounded prediction beats unbiased-but-noisy: the seed sinusoid (or its
  edge-held linear fallback) feeds the search box; nothing extrapolates
  through jitter.

The forward-backward check is the one addition beyond the port (KLT
practice): every accepted match is tracked BACK to its seed frame with
the found position as the template, and rejected if the round trip
misses the seed or the backward correlation falls below `min_corr`. Be
honest about what it can catch: the backward search patch is cut at the
seed position and Hann-windowed, so a slip whose content truly lives
elsewhere shows up as a CRATERED backward correlation (different
context), not as a confidently displaced landing; on smooth low-texture
content whose phase-correlation mixtures are forgiving, the check adds
little beyond the forward threshold. It is cheap (one extra match per
accepted label) and one-sided: it only ever removes labels.

Everything is Qt-free and operates in LOADED-frame pixels; the caller
converts through `CoordinateChain`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tktomo.tracking.labels import interpolate_track


def highpass2d(frame: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Frame minus a least-squares plane, then minus a heavy blur.

    The blur alone is not enough. Gaussian smoothing reproduces a linear
    ramp in the interior but not at the edges, so a per-frame ramp
    survives the high pass as a border artefact, and a border artefact is
    a strong, frame-varying feature that a correlator will happily lock
    onto. Removing the plane by least squares first kills the ramp
    everywhere including the border. (This is deliberately NOT the
    display filter's blur-only high-pass.)
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    frame = np.asarray(frame, np.float32)
    ny, nx = frame.shape
    y = np.linspace(-1, 1, ny)[:, None] * np.ones((1, nx))
    x = np.ones((ny, 1)) * np.linspace(-1, 1, nx)[None, :]
    basis = np.column_stack([np.ones(frame.size), y.ravel(), x.ravel()])
    coef, *_ = np.linalg.lstsq(basis, frame.ravel(), rcond=None)
    flat = frame - (basis @ coef).reshape(ny, nx).astype(np.float32)
    return flat - gaussian_filter(flat, sigma)


def _patch(img, cy, cx, P):
    """P x P patch centred on the nearest integer to (cy, cx). None if it
    does not fit entirely inside the image."""
    h = P // 2
    iy, ix = int(round(cy)), int(round(cx))
    if iy - h < 0 or ix - h < 0 or iy - h + P > img.shape[0] \
            or ix - h + P > img.shape[1]:
        return None, None, None
    return img[iy - h:iy - h + P, ix - h:ix - h + P], float(iy), float(ix)


def _hann2d(P, _cache={}):  # noqa: B006 - deliberate module-lifetime cache
    """Separable Hann window, cached: the same P is used thousands of times."""
    if P not in _cache:
        w = np.hanning(P + 2)[1:-1]
        _cache[P] = np.outer(w, w)
    return _cache[P]


def match_patch(prev, cur, p_yx, q_yx, P, max_step, upsample=20, iters=3,
                tol=0.02):
    """Where the feature at p_yx in `prev` has moved to, searched near q_yx.

    Returns (new_y, new_x, quality) or None. `quality` is the plain
    correlation coefficient between the template and the matched patch
    after alignment: an interpretable number.

    RE-CUT AND ITERATE, which is not optional. Phase correlation assumes
    the two images differ by a CIRCULAR shift. Cutting both patches at the
    same fixed location breaks that: as the feature moves, content leaves
    one edge and nothing comes in to replace it, and the estimate is
    dragged toward zero. Measured on a clean synthetic blob the gain was
    0.87, i.e. a 3 px motion read as 2.65 px, a 13 percent systematic
    underestimate, not noise. The cure is to re-cut the search patch at
    the position just estimated and correlate again; the bias is
    proportional to the residual displacement, so two passes take it
    below a hundredth of a pixel.

    Patches are Hann-windowed for the correlation (a hard patch edge
    leaks across the spectrum) but quality is computed on the unwindowed
    pair, so it still means "how alike are these".
    """
    from scipy.ndimage import shift as ndshift  # noqa: PLC0415
    from skimage.registration import phase_cross_correlation  # noqa: PLC0415

    a, ry, rx = _patch(prev, p_yx[0], p_yx[1], P)
    if a is None or a.std() < 1e-9:
        return None
    w = _hann2d(P)
    aw = (a - a.mean()) * w

    cy, cx = float(q_yx[0]), float(q_yx[1])
    b = d = None
    for it in range(max(1, iters)):
        b, qy, qx = _patch(cur, cy, cx, P)
        if b is None or b.std() < 1e-9:
            return None
        s, _, _ = phase_cross_correlation(aw, (b - b.mean()) * w,
                                          upsample_factor=upsample,
                                          normalization=None)
        d = -np.asarray(s, float)      # displacement of b's content vs a's
        if not np.all(np.isfinite(d)):
            return None
        if it == 0 and np.abs(d).max() > max_step:
            return None                # outside the search box: wrong feature
        # the feature sat at (p - r) from the template centre; it is that
        # far from the matched patch centre too, plus the displacement
        ny = qy + (p_yx[0] - ry) + d[0]
        nx = qx + (p_yx[1] - rx) + d[1]
        moved = max(abs(ny - cy), abs(nx - cx))
        cy, cx = ny, nx
        if moved < tol:
            break

    # the answer must lie inside the search box drawn around the PREDICTION
    if max(abs(cy - q_yx[0]), abs(cx - q_yx[1])) > max_step:
        return None

    bb = ndshift(b, -d, order=1, mode="nearest")
    q = float(np.corrcoef(a.ravel(), bb.ravel())[0, 1])
    if not np.isfinite(q):
        return None
    return cy, cx, q


def structure_tensor(patch):
    """(lam1, lam2, gxx, gyy) of the gradient second-moment matrix.

    THE APERTURE PROBLEM, made into a number. A patch lying on a filament
    or an edge can slide ALONG that structure without changing the
    correlation at all, so its displacement in that direction is not
    measured, but phase correlation still returns a confident-looking
    peak. lam1 >> lam2 is an edge (position known across it, unknown
    along it); lam1 ~ lam2 is a corner or blob; both small is no
    structure. lam2 is the Shi-Tomasi good-features score.
    """
    gy, gx = np.gradient(np.asarray(patch, float))
    gxx = float((gx * gx).mean())
    gyy = float((gy * gy).mean())
    gxy = float((gx * gy).mean())
    tr, det = gxx + gyy, gxx * gyy - gxy * gxy
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    return tr / 2.0 + disc, tr / 2.0 - disc, gxx, gyy


def coherence(lam1, lam2):
    """0 = isotropic (corner), 1 = pure edge. The elongation measure."""
    s = lam1 + lam2
    return 0.0 if s <= 0 else (lam1 - lam2) / s


def patch_size(feature_size: float) -> int:
    """Template size from the feature's marker size: clip(4*size, 16, 96).

    The default marker size 10 gives P = 40, close to the 48 the slogger
    tracker settled on for the same data at the same binning.
    """
    return int(np.clip(int(4 * float(feature_size)), 16, 96))


# ---------------------------------------------------------------------------
# track completion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoTrackParams:
    """All measured defaults; see the module docstring for provenance."""

    patch: int = 40
    search_radius: float = 8.0          # loaded px around the prediction
    radius_growth: float = 0.25         # px per view of distance to the seed
    min_corr: float = 0.30
    fb_check: bool = True
    fb_tol: float = 1.0                 # loaded px round-trip miss
    hp_sigma: float = 12.0
    upsample: int = 20
    iters: int = 3
    tol: float = 0.02
    max_consecutive_failures: int = 3
    max_coherence: float = 0.4


@dataclass(frozen=True)
class AutoLabel:
    view: int
    u: float
    v: float
    quality: float


@dataclass
class TrackResult:
    labels: list = field(default_factory=list)          # [AutoLabel]
    seed_report: list = field(default_factory=list)     # per-seed dicts
    warnings: list = field(default_factory=list)
    cancelled: bool = False


def complete_track(frames_hp, theta, seeds, params: AutoTrackParams, *,
                   progress=None, cancelled=None,
                   matcher=None) -> TrackResult:
    """Fill unlabeled views around manual seeds by anchored matching.

    frames_hp: sequence of ALREADY-HIGHPASSED float32 2D frames.
    theta: (V,) angles in radians.
    seeds: [(view, u, v)] manual labels in LOADED-frame px, at least 2.
    progress: optional callable(views_done); cancelled: optional
    callable() -> bool, checked per view.
    matcher (REQUIRED): the per-view matcher,
    `matcher(frames_hp, seeds_usable, view, pred_vu, max_step)
    -> (v, u, quality) | None`, given ALL usable seeds so it may use
    more than the nearest one. `quality` is thresholded by
    `params.min_corr`, so set that to whatever the matcher's confidence
    scale means. `tktomo.tracking.learned_match.LearnedMatcher` is the
    shipped one. The forward-backward check stays phase-correlation based
    (it is a consistency check on the accepted match, not the tracker).

    Each unlabeled view is assigned to the nearest seed that passed the
    coherence gate; the template is always cut at that seed's manual
    position (no chaining, no updates: the honest failure is a stopped
    track, never a wandering one). The march runs outward from each seed
    per direction, skipping isolated failures and stopping a direction
    after `max_consecutive_failures` in a row.
    """
    if matcher is None:
        raise ValueError(
            "complete_track requires a matcher; pass "
            "tktomo.tracking.learned_match.LearnedMatcher() "
            "(the single-anchor phase-correlation completer was removed)")
    theta = np.asarray(theta, float)
    n_views = theta.size
    result = TrackResult()
    if len(seeds) < 2:
        result.warnings.append("need at least 2 manual labels to auto-track")
        return result

    # -- coherence gate on every seed's template --------------------------
    usable = []
    for view, u, v in sorted(seeds):
        patch, _, _ = _patch(frames_hp[view], v, u, params.patch)
        if patch is None:
            result.warnings.append(
                f"seed at view {view} refused: template does not fit "
                f"inside the frame")
            result.seed_report.append({"view": view, "used": False,
                                       "coherence": float("nan")})
            continue
        lam1, lam2, _, _ = structure_tensor(patch)
        coh = coherence(lam1, lam2)
        used = coh <= params.max_coherence
        result.seed_report.append({"view": view, "used": used,
                                   "coherence": float(coh)})
        if used:
            usable.append((int(view), float(u), float(v)))
        else:
            result.warnings.append(
                f"seed at view {view} refused: coherence {coh:.2f} > "
                f"{params.max_coherence:.2f} (edge-like, position along "
                f"the structure would be fiction)")
    if not usable:
        result.warnings.append(
            "feature not trackable: every seed template failed the "
            "coherence gate")
        return result

    # -- bounded prediction from the manual seeds only --------------------
    seed_views = [s[0] for s in usable]
    u_pred, v_pred = interpolate_track(
        theta, seed_views, [s[1] for s in usable], [s[2] for s in usable],
        u_mode="sinusoid", v_mode="linear")
    # With >= 3 seeds the sinusoid extrapolates physically beyond the seed
    # span. With 2 the interpolation edge-holds there, which goes stale at
    # the feature's real speed (px per view) and kills the march within a
    # few steps; the bounded alternative is a zeroth-order hold on the
    # LAST ACCEPTED position (the slogger prediction lesson: bounded beats
    # unbiased-but-noisy when the answer feeds a search box).
    sinusoid_active = len(usable) >= 3
    span_lo, span_hi = min(seed_views), max(seed_views)

    # -- assign each view to its nearest usable seed ----------------------
    seed_arr = np.asarray(seed_views)
    assignment = {}
    for j in range(n_views):
        if j in seed_views:
            continue
        assignment[j] = int(seed_arr[np.argmin(np.abs(seed_arr - j))])

    # -- outward march per seed, per direction ----------------------------
    seed_pos = {s[0]: (s[1], s[2]) for s in usable}
    done = 0
    labelled_views = set(seed_views)
    for seed_view in seed_views:
        su, sv = seed_pos[seed_view]
        for direction in (-1, 1):
            failures = 0
            j = seed_view + direction
            stop_at = None
            last_accepted = (sv, su)
            while 0 <= j < n_views and failures < \
                    params.max_consecutive_failures:
                if cancelled is not None and cancelled():
                    result.cancelled = True
                    return result
                if assignment.get(j) != seed_view or j in labelled_views:
                    j += direction
                    continue
                dist = abs(j - seed_view)
                max_step = min(
                    params.search_radius + params.radius_growth * dist,
                    params.patch // 2 - 2)
                if sinusoid_active or span_lo <= j <= span_hi:
                    pred = (v_pred[j], u_pred[j])
                else:
                    pred = last_accepted        # bounded hold beyond span
                hit = matcher(frames_hp, usable, j, pred, max_step)
                ok = hit is not None and hit[2] >= params.min_corr
                if ok and params.fb_check:
                    # the round trip must land on the seed AND look like
                    # the seed: a slip onto a persistent lookalike tracks
                    # back to the lookalike's own position instead
                    back = match_patch(
                        frames_hp[j], frames_hp[seed_view],
                        (hit[0], hit[1]), (sv, su), params.patch, max_step,
                        upsample=params.upsample, iters=params.iters,
                        tol=params.tol)
                    ok = (back is not None
                          and back[2] >= params.min_corr
                          and np.hypot(back[0] - sv, back[1] - su)
                          <= params.fb_tol)
                if ok:
                    result.labels.append(AutoLabel(
                        view=int(j), u=float(hit[1]), v=float(hit[0]),
                        quality=float(hit[2])))
                    labelled_views.add(j)
                    failures = 0
                    last_accepted = (hit[0], hit[1])
                else:
                    failures += 1
                    stop_at = j
                done += 1
                if progress is not None:
                    progress(done)
                j += direction
            key = "left_stop" if direction < 0 else "right_stop"
            for rep in result.seed_report:
                if rep["view"] == seed_view:
                    rep[key] = stop_at if failures >= \
                        params.max_consecutive_failures else None
    result.labels.sort(key=lambda al: al.view)
    return result
