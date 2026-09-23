"""One-dimensional edge matching for sphere tangent features.

The blob tracker in `autotrack` cannot do this job and refuses to try: its
structure-tensor gate throws out edge-like seeds on purpose, because a
patch sitting on a filament slides along it while reporting a confident
correlation, and the position along the edge would be fiction. A sphere's
apex is exactly that geometry. The answer is not to switch the gate off
and hope, it is to stop asking for the coordinate that is not there.

So this matcher measures ONE coordinate. For a laos or a raos it cuts a
short profile ALONG u, averaged over a few rows, and phase-correlates it
against the same profile cut at the nearest manual seed. The across
coordinate it returns is the caller's own prediction, untouched, and the
caller must treat it as unmeasured: an auto label on a tangent feature can
never become a centre observation (`split_validity`'s `measured_across`).

Two things about the averaging window. It has to be short, because the
edge is curved: averaging the silhouette over +-H around the apex pulls
the measured edge inward by about H^2 / (6 R), which is 0.3 px at H = 6
and R = 20 and less for a bigger bubble. And it mostly does not matter,
because the template and the target are cut the same way, so what the
correlation measures is the DISPLACEMENT of the edge and the bias cancels
to first order. The absolute position comes from the manual seed.

The quality reported is a plain correlation coefficient. The learned
classifier the blob tracker uses does not apply: its 43-feature vector is
a contract with a trained model and half of it describes the vertical
agreement of five anchors, which does not exist here. A field of bubbles
offers many similar edges, so a second gate asks that the correlation peak
beat its best rival by `peak_ratio`, which a lock-on to the neighbouring
bubble's wall usually fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tktomo.tracking.model import LAOS, POINT, RAOS

#: which coordinate each feature kind constrains, for picking a matcher
AXIS_OF_KIND = {LAOS: "u", RAOS: "u"}


def axis_for_kind(kind: int) -> str | None:
    """"u", "v", or None for a plain point feature."""
    if int(kind) == POINT:
        return None
    return AXIS_OF_KIND.get(int(kind), "v")


def sign_for_kind(kind: int) -> float:
    """-1 for the apex at the smaller coordinate along its axis, +1 above."""
    from tktomo.tracking.model import SIGN_U, SIGN_V  # noqa: PLC0415

    k = int(kind)
    return float(SIGN_U[k] if SIGN_U[k] else SIGN_V[k])


def edge_profile(frame, v, u, axis, length, window):
    """A 1D profile along `axis`, averaged over `window` across it.

    Returns (profile, origin) with `origin` the coordinate of the profile's
    first sample along the axis, or (None, None) when the cut does not fit
    inside the frame.
    """
    frame = np.asarray(frame)
    half_l = int(length) // 2
    half_w = int(window) // 2
    iv, iu = int(round(v)), int(round(u))
    if axis == "u":
        r0, r1 = iv - half_w, iv + half_w + 1
        c0, c1 = iu - half_l, iu - half_l + int(length)
    else:
        r0, r1 = iv - half_l, iv - half_l + int(length)
        c0, c1 = iu - half_w, iu + half_w + 1
    if r0 < 0 or c0 < 0 or r1 > frame.shape[0] or c1 > frame.shape[1]:
        return None, None
    block = np.asarray(frame[r0:r1, c0:c1], float)
    if axis == "u":
        return block.mean(axis=0), float(c0)
    return block.mean(axis=1), float(r0)


def _hann(n, _cache={}):  # noqa: B006 - deliberate module-lifetime cache
    if n not in _cache:
        w = np.hanning(n + 2)[1:-1]
        _cache[n] = w
    return _cache[n]


def _peak_ratio(a, b):
    """Best normalised cross-correlation over its best rival lag.

    A rival is a lag at least a quarter of the profile away, which is
    where a neighbouring bubble's wall would sit. 1.0 means the match is
    no better than a lookalike.
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom <= 0:
        return 0.0
    cc = np.correlate(a, b, mode="full") / denom
    k = int(np.argmax(cc))
    far = max(3, cc.size // 8)
    rival = np.concatenate([cc[:max(0, k - far)], cc[k + far:]])
    if rival.size == 0:
        return float("inf")
    best_rival = float(np.max(rival))
    if best_rival <= 0:
        return float("inf")
    return float(cc[k] / best_rival)


@dataclass(frozen=True)
class EdgeMatcher:
    """A `complete_track` matcher that measures one coordinate only.

    `axis` is the coordinate the feature's kind constrains: "u" for a laos
    or a raos, "v" for a taos or a baos.
    """

    axis: str = "u"
    length: int = 41            # profile length along the axis, track px
    window: int = 7             # averaged across the axis, track px
    upsample: int = 20
    iters: int = 3
    tol: float = 0.02
    peak_ratio: float = 1.15
    min_contrast: float = 1e-6

    def __call__(self, frames_hp, seeds, view, pred_vu, max_step):
        from skimage.registration import (  # noqa: PLC0415
            phase_cross_correlation,
        )

        if not seeds:
            return None
        seed = min(seeds, key=lambda s: abs(int(s[0]) - int(view)))
        seed_view, seed_u, seed_v = int(seed[0]), float(seed[1]), float(seed[2])
        tmpl, t0 = edge_profile(frames_hp[seed_view], seed_v, seed_u,
                                self.axis, self.length, self.window)
        if tmpl is None or float(np.std(tmpl)) < self.min_contrast:
            return None
        seed_pos = (seed_u if self.axis == "u" else seed_v)
        pred_v, pred_u = float(pred_vu[0]), float(pred_vu[1])
        pos = pred_u if self.axis == "u" else pred_v
        start = pos
        w = _hann(tmpl.size)
        tw = (tmpl - tmpl.mean()) * w

        cur = shift = None
        for it in range(max(1, self.iters)):
            cv = pred_v if self.axis == "u" else pos
            cu = pos if self.axis == "u" else pred_u
            cur, c0 = edge_profile(frames_hp[int(view)], cv, cu,
                                   self.axis, self.length, self.window)
            if cur is None or float(np.std(cur)) < self.min_contrast:
                return None
            s, _, _ = phase_cross_correlation(
                tw, (cur - cur.mean()) * w,
                upsample_factor=self.upsample, normalization=None)
            shift = -float(np.asarray(s, float).ravel()[0])
            if not np.isfinite(shift):
                return None
            if it == 0 and abs(shift) > max_step:
                return None              # outside the search box
            # the apex sat at (seed_pos - t0) into the template; it is that
            # far into the matched profile too, plus the displacement
            new = c0 + (seed_pos - t0) + shift
            moved = abs(new - pos)
            pos = new
            if moved < self.tol:
                break
        if abs(pos - start) > max_step:
            return None
        if _peak_ratio(tmpl, cur) < self.peak_ratio:
            return None
        rolled = np.interp(np.arange(cur.size) + shift,
                           np.arange(cur.size), cur)
        q = float(np.corrcoef(tmpl, rolled)[0, 1])
        if not np.isfinite(q):
            return None
        if self.axis == "u":
            return pred_v, pos, q
        return pos, pred_u, q


class DetectionCache:
    """One detector pass per (frame, axis), shared by every feature.

    Detecting apexes costs a multiscale ridge filter over the whole frame,
    and `complete_track` asks per feature per view. Without this, ten
    features on the same views would each pay for the same detection. One
    cache is built per `run_autotrack` call, so it never outlives the
    frames it was filled from.
    """

    def __init__(self, detector, frames, max_frames: int = 512) -> None:
        self.detector = detector
        self.frames = frames
        self._by_view: dict = {}
        self._max = int(max_frames)

    def __call__(self, view: int, axis: str) -> np.ndarray:
        from tktomo.tracking.apexdetect import detections_array  # noqa: PLC0415

        key = (int(view), str(axis))
        hit = self._by_view.get(key)
        if hit is None:
            hit = detections_array(
                self.detector.detect(np.asarray(self.frames[int(view)]), axis))
            if len(self._by_view) >= self._max:
                self._by_view.clear()
            self._by_view[key] = hit
        return hit


@dataclass
class ApexMatcher:
    """A `complete_track` matcher that snaps to a DETECTED apex.

    Where `EdgeMatcher` follows the edge by correlation and knows nothing
    about what it is following, this asks a detector trained on the sample
    where the apexes in this frame are, and takes the one nearest the
    prediction that sits on the right side of its body. So it cannot drift
    along a wall, and its quality is the detector's own confidence rather
    than a correlation, which means the "min p" box thresholds something
    with a meaning again.

    Nearest to the prediction, not highest scoring: inside a search box
    this small the anchored prediction is the better evidence, and the
    score is then an independent report on what was chosen rather than the
    reason it was chosen. `across_tol` gates the coordinate the feature
    does not constrain, generously, because the apex of a body sits at its
    centre in that coordinate and moves slowly, so a detection far off it
    belongs to a different body.
    """

    cache: DetectionCache
    axis: str = "u"
    sign: float = -1.0
    min_score: float = 0.0
    across_tol: float = 12.0

    def __call__(self, frames_hp, seeds, view, pred_vu, max_step):
        det = self.cache(int(view), self.axis)
        if det.shape[0] == 0:
            return None
        u, v, score, sign = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
        pred_v, pred_u = float(pred_vu[0]), float(pred_vu[1])
        if self.axis == "u":
            along, across = u - pred_u, v - pred_v
        else:
            along, across = v - pred_v, u - pred_u
        ok = ((np.sign(sign) == np.sign(self.sign))
              & (np.abs(along) <= float(max_step))
              & (np.abs(across) <= float(self.across_tol))
              & (score >= float(self.min_score)))
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)
        best = idx[int(np.argmin(np.abs(along[idx])))]
        q = float(score[best])
        if self.axis == "u":
            return pred_v, float(u[best]), q
        return float(v[best]), pred_u, q
