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
accepted label) and one-sided: it only ever removes labels. Measured on a
second sample (lens1_v11_upper, 2026-09-04) it removed half the good
ones too, because a template cut far from its seed does not round-trip
to within a pixel of it, so it is off by default and the residual
rejection after the fit does its job.

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


#: The template size the learned matcher was trained at, in TRACK px. The
#: tracking grid is chosen so that this many pixels cover about four times
#: the feature (`choose_track_bin`); the patch itself never changes.
TRACK_PATCH = 40


def patch_size(feature_size: float) -> int:
    """Template size from the feature's marker size: clip(4*size, 16, 96).

    The default marker size 10 gives P = 40, close to the 48 the slogger
    tracker settled on for the same data at the same binning. The learned
    matcher ignores this and uses `TRACK_PATCH`; the grid is what adapts.
    """
    return int(np.clip(int(4 * float(feature_size)), 16, 96))


def choose_track_bin(feature_size_file_px: float, *, max_bin: int = 8) -> int:
    """The grid to track on, from the feature's size on the file's grid.

    The largest bin b (1..max_bin) at which a `TRACK_PATCH` template still
    covers at least four times the feature, i.e. the coarsest grid on which
    the feature is at least a quarter of the patch. A 12 px particle tracks
    at bin 1, a 25 px one at bin 2, a 45 px one at bin 4. Measured on a
    12 raw px particle (lens1_v11_upper, 2026-09-04): at the display bin 4
    the app used to track on, the 40 px patch spanned 160 raw px of sample
    edges and the coherence gate refused 17 of 18 seeds; at bin 1 the same
    tracker covered 79 percent of the views at 95 percent within 4 px.
    """
    size = float(feature_size_file_px)
    if not np.isfinite(size) or size <= 0:
        return 1
    return int(max(1, min(int(max_bin), int(4 * size // TRACK_PATCH))))


def max_search_radius(patch: int) -> float:
    """The widest search box a `patch` px template can see: half the patch
    minus a margin. Beyond it the phase correlation wraps."""
    return float(int(patch) // 2 - 2)


# ---------------------------------------------------------------------------
# track completion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoTrackParams:
    """All measured defaults; see the module docstring for provenance.

    Every length is in px of the grid `complete_track` runs on (the TRACK
    grid, see `run_autotrack`). `min_corr` thresholds whatever the matcher
    returns as quality (a probability for the learned matcher, a
    correlation coefficient for a plain phase-correlation one), while
    `fb_min_corr` thresholds the backward match's correlation coefficient,
    which is always a plain correlation. They used to be one number.

    `fb_check` defaults to off. Measured on lens1_v11_upper (feature 0, 18
    seeds every ~50 views, 2026-09-04): on, the round trip ended enough
    marches to halve the coverage, 50 against 79 percent of the views,
    for 100 against 95 percent of the labels within 4 raw px, and the
    result did not change between a backward correlation threshold of
    0.0 and 0.2, so it is the round-trip distance that fails: a template
    cut many views from its seed does not land back on it within
    `fb_tol`. The residual rejection after the fit catches the lock-ons
    the round trip was meant to catch, without the cost.
    """

    patch: int = 40
    search_radius: float = 8.0          # track px around the prediction
    radius_growth: float = 0.25         # px per view of distance to the seed
    min_corr: float = 0.30
    fb_check: bool = False
    fb_min_corr: float = 0.10           # backward correlation coefficient
    fb_tol: float = 1.0                 # track px round-trip miss
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
    """What one feature's completion produced, and where every view went.

    `outcomes` maps each view that was not labelled to a short code:
    ``none`` (no anchor matched inside the search box), ``low_p`` (matched
    but below `min_corr`), ``fb_corr`` / ``fb_miss`` (the backward match
    correlated below `fb_min_corr` / landed off the seed), ``stopped``
    (behind a march that ended, never attempted). `stats` carries the
    counts and the grid the run used, so a report can say which gate ate
    the views rather than "some".
    """

    labels: list = field(default_factory=list)          # [AutoLabel]
    seed_report: list = field(default_factory=list)     # per-seed dicts
    warnings: list = field(default_factory=list)
    cancelled: bool = False
    stats: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)        # view -> code


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
    # Kept fatal on purpose. Measured on lens1_v11_upper (2026-09-04): with
    # the gate off, the seeds it would have refused (a particle riding on
    # the sample's edge, coherence 0.91) produced labels 11 to 16 raw px
    # off that the learned confidence still passed. A refused seed is a
    # region the user labels by hand, and the report says so.
    usable = []
    for view, u, v in sorted(seeds):
        patch, _, _ = _patch(frames_hp[view], v, u, params.patch)
        if patch is None:
            reason = "refused: template outside frame"
            result.warnings.append(
                f"seed at view {view} refused: a {params.patch} px "
                f"template around it does not fit inside the frame. "
                f"Views near the border stay manual.")
            result.seed_report.append({"view": view, "used": False,
                                       "coherence": float("nan"),
                                       "reason": reason})
            continue
        lam1, lam2, _, _ = structure_tensor(patch)
        coh = coherence(lam1, lam2)
        used = coh <= params.max_coherence
        result.seed_report.append({
            "view": view, "used": used, "coherence": float(coh),
            "reason": "" if used else f"refused: coherence {coh:.2f}"})
        if used:
            usable.append((int(view), float(u), float(v)))
        else:
            result.warnings.append(
                f"seed at view {view} refused: coherence {coh:.2f} > "
                f"{params.max_coherence:.2f}, the template sits on an edge "
                f"and its position along it would be fiction. Views near "
                f"it stay manual: label them by hand, the tracker cannot "
                f"follow a feature that sits on an edge.")
    if not usable:
        result.warnings.append(
            "feature not trackable: every seed template failed the "
            "coherence gate. Label it by hand, or pick a feature that is "
            "a blob rather than a piece of an edge.")
        _finish_stats(result, n_views, [s[0] for s in seeds], params)
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
                if hit is None:
                    code = "none"
                elif hit[2] < params.min_corr:
                    code = "low_p"
                elif params.fb_check:
                    # the round trip must land on the seed AND look like
                    # the seed: a slip onto a persistent lookalike tracks
                    # back to the lookalike's own position instead. The
                    # correlation threshold is fb_min_corr, a correlation
                    # coefficient, never min_corr (see AutoTrackParams).
                    back = match_patch(
                        frames_hp[j], frames_hp[seed_view],
                        (hit[0], hit[1]), (sv, su), params.patch, max_step,
                        upsample=params.upsample, iters=params.iters,
                        tol=params.tol)
                    if back is None or back[2] < params.fb_min_corr:
                        code = "fb_corr"
                    elif np.hypot(back[0] - sv, back[1] - su) > params.fb_tol:
                        code = "fb_miss"
                    else:
                        code = "ok"
                else:
                    code = "ok"
                if code == "ok":
                    result.labels.append(AutoLabel(
                        view=int(j), u=float(hit[1]), v=float(hit[0]),
                        quality=float(hit[2])))
                    labelled_views.add(j)
                    result.outcomes.pop(j, None)
                    failures = 0
                    last_accepted = (hit[0], hit[1])
                else:
                    result.outcomes[j] = code
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
    _finish_stats(result, n_views, seed_views, params)
    return result


def _finish_stats(result: TrackResult, n_views: int, seed_views,
                  params: AutoTrackParams) -> None:
    """Account for every view: labelled, one of the gate codes, or behind a
    stopped march. Fills `result.stats` and completes `result.outcomes`."""
    seeds = set(int(v) for v in seed_views)
    labelled = {al.view for al in result.labels}
    for j in range(n_views):
        if j in seeds or j in labelled or j in result.outcomes:
            continue
        result.outcomes[j] = "stopped"
    counts = {}
    for code in result.outcomes.values():
        counts[code] = counts.get(code, 0) + 1
    unlabelled = [j for j in range(n_views) if j not in seeds
                  and j not in labelled]
    gap, best = None, 0
    if unlabelled:
        run_lo = prev = unlabelled[0]
        for j in unlabelled[1:] + [None]:
            if j is not None and j == prev + 1:
                prev = j
                continue
            if prev - run_lo + 1 > best:
                best, gap = prev - run_lo + 1, (run_lo, prev)
            if j is not None:
                run_lo = prev = j
    report = result.seed_report
    result.stats.update({
        "n_views": int(n_views),
        "n_unlabelled": int(n_views - len(seeds)),
        "n_seeds": len(report),
        "n_seeds_used": sum(1 for r in report if r.get("used")),
        "n_seeds_refused": sum(1 for r in report if not r.get("used")),
        "n_attempted": sum(v for k, v in counts.items() if k != "stopped")
        + len(labelled),
        "n_accepted": len(labelled),
        "n_none": counts.get("none", 0),
        "n_low_p": counts.get("low_p", 0),
        "n_fb_miss": counts.get("fb_miss", 0),
        "n_fb_corr": counts.get("fb_corr", 0),
        "n_stopped": counts.get("stopped", 0),
        "largest_gap": gap,
        "min_corr": float(params.min_corr),
        "patch_track_px": int(params.patch),
        "search_radius_track_px": float(params.search_radius),
    })


# ---------------------------------------------------------------------------
# batch driver, shared by the local worker thread and the remote stack host
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoTrackJob:
    """One feature to complete: its manual seeds in SERVED-frame px.

    `track_bin` is the grid to track on, as a mean-pool factor of the base
    stack (the file's grid); 0 means the served grid, which is what a
    client that predates the field asked for. Choose it with
    `choose_track_bin` from the feature's size in file px.
    """

    fid: int
    seeds: tuple                    # ((view, u, v), ...)
    params: AutoTrackParams
    track_bin: int = 0


class BinnedFrames:
    """A stack mean-pooled by `factor`, one frame at a time on access.

    `HighpassCache.frames` reads frames one by one, so with this in front
    of the base stack the pooled copy never exists in memory: only the
    high-passed frames do.
    """

    def __init__(self, base, factor: int) -> None:
        self._base = base
        self._factor = int(factor)

    def __len__(self) -> int:
        return len(self._base)

    @property
    def shape(self) -> tuple[int, int, int]:
        n, ny, nx = np.shape(self._base)[:3]
        return (int(n), int(ny) // self._factor, int(nx) // self._factor)

    def __getitem__(self, k):
        from tktomo.ptycho_align.core.preprocess import bin_stack  # noqa: PLC0415

        frame = np.asarray(self._base[k], np.float32)
        return bin_stack(frame[None], self._factor)[0]


class HighpassCache:
    """The high-passed copy of one stack, kept so repeated runs skip ~7 s.

    Keyed by the stack's identity and sigma (or an explicit `key`); at most
    one stack is cached. The stack is NOT copied: frames are read one at a
    time from whatever `stack[k]` returns.
    """

    def __init__(self) -> None:
        self._key = None
        self._frames: list | None = None

    def clear(self) -> None:
        self._key, self._frames = None, None

    @property
    def key(self):
        return self._key

    def frames(self, stack, sigma: float, *, key=None, progress=None,
               cancelled=None) -> list | None:
        """The high-passed frames, or None if cancelled before finishing.

        `progress(done, total, None)` every ten frames; `cancelled()` is
        polled per frame. `key` replaces the default `(id(stack), sigma)`
        when the caller knows a better identity (the base stack plus a
        binning factor: a lazily binned view is a new object every call).
        """
        key = (id(stack), float(sigma)) if key is None else key
        if self._key == key and self._frames is not None:
            return self._frames
        self.clear()
        n = len(stack)
        hp = []
        for k in range(n):
            if cancelled is not None and cancelled():
                return None
            hp.append(highpass2d(np.asarray(stack[k], np.float32), sigma))
            if progress is not None and k % 10 == 0:
                progress(k + 1, n, None)
        self._key, self._frames = key, hp
        return hp


def run_autotrack(base, theta, jobs, *, hp_sigma: float, matcher,
                  cache: HighpassCache | None = None, progress=None,
                  cancelled=None, served_bin: int = 1) -> list:
    """Complete every job's track; returns [(fid, TrackResult)].

    `base` is the stack on the file's grid and `served_bin` the mean-pool
    factor of the grid the job's seeds are expressed in (what the window
    shows). Each job tracks on its own `track_bin` grid: the seeds are
    regridded there, `search_radius` (given in served px) is converted and
    capped at what the template can see, `hp_sigma` is applied on the
    track grid, and the labels come back in served px so the caller's
    coordinate chain applies unchanged.

    Returns [] when cancelled (during the high-pass or mid-track), so the
    caller never applies a half-finished batch. `progress(done, total, fid)`
    is called once per job with its index, and with fid=None during the
    high-pass; `cancelled() -> bool` is polled throughout.
    """
    from dataclasses import replace  # noqa: PLC0415

    from tktomo.tracking.coords import regrid_uv  # noqa: PLC0415

    cache = cache if cache is not None else HighpassCache()
    theta = np.asarray(theta, float)
    served_bin = int(served_bin)
    out = [None] * len(jobs)
    # jobs sharing a grid run back to back so the one-entry cache serves them
    order = sorted(range(len(jobs)),
                   key=lambda i: int(jobs[i].track_bin) or served_bin)
    for idx in order:
        job = jobs[idx]
        b = int(job.track_bin) or served_bin
        frames = base if b == 1 else BinnedFrames(base, b)
        hp = cache.frames(frames, hp_sigma, key=(id(base), b, float(hp_sigma)),
                          progress=progress, cancelled=cancelled)
        if hp is None:
            return []
        if progress is not None:
            progress(idx, len(jobs), job.fid)
        seeds = []
        for view, u, v in job.seeds:
            tu, tv = regrid_uv(u, v, served_bin, b)
            seeds.append((int(view), float(tu), float(tv)))
        radius = min(float(job.params.search_radius) * served_bin / b,
                     max_search_radius(job.params.patch))
        params = replace(job.params, search_radius=radius)
        result = complete_track(hp, theta, seeds, params,
                                cancelled=cancelled, matcher=matcher)
        if result.cancelled:
            return []
        result.labels = [
            replace(al, u=float(su), v=float(sv))
            for al in result.labels
            for su, sv in [regrid_uv(al.u, al.v, b, served_bin)]]
        result.stats.update({"track_bin": b, "served_bin": served_bin,
                             "hp_sigma_track_px": float(hp_sigma)})
        out[idx] = (job.fid, result)
    return out
