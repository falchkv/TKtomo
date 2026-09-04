"""Multi-anchor matching with a learned confidence, for `complete_track`.

The phase-correlation completer finds good positions (1.8 raw px median
from 10-view anchors, near the human floor) but its correlation quality is
close to chance at telling the 17 to 27 percent bad answers from the good
ones (AUC 0.6). This module keeps the positions and replaces the quality:

- position: `match_patch` against the K=5 manual labels nearest in angle,
  fused by quality-weighted median (halves the p90 over a single anchor);
- confidence: a gradient-boosted classifier on 43 features describing HOW
  the answer was reached (the five answers and their qualities, how they
  agree, the fused NCC map, patch structure, residual against the sinusoid),
  trained leave-one-feature-out to predict "within 4 raw px". AUC 0.86.

Measured on the graphite-ball stack; the classifier was trained on that one
dataset and transfers across features, not (yet) proven across samples. The
strongest cues are consistency ones (sinusoid residual, spread of the vote),
not appearance, which is the best transfer prospect 844 labels allow.

Shape of the matcher is the `complete_track(matcher=...)` hook:

    matcher(frames_hp, seeds_usable, view, pred_vu, max_step) -> (v, u, p)

`p` is a probability; threshold it with `AutoTrackParams.min_corr`, 0.20 is
the measured operating point (keeps ~90 percent of answers at 10-view
anchors, about 1 percent worse than 10 raw px).

Optional dependencies: scikit-learn and joblib (the `learned` extra). The
feature extraction itself needs only numpy, scipy and scikit-image. Feature
ORDER is a contract with the shipped classifier: `NAMES` must not change
without retraining (`tomo_feature_tracking_training/completion/`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tktomo.tracking.autotrack import (
    TRACK_PATCH,
    coherence,
    match_patch,
    structure_tensor,
)

P = TRACK_PATCH   # the classifier was trained at this size, on the track grid
K = 5             # anchors per target
TAU = 10.0        # views: angular weighting scale of the fused NCC map
DEFAULT_MODEL = Path(__file__).with_name("data") / "confidence_gboost.joblib"

NAMES = (
    "k_hits q_max q_mean q_min q_nearest spread spread_max n_identical "
    "sep_nearest sep_mean_hit resid_u resid_v resid ncc_peak ncc_psr "
    "ncc_second_ratio ncc_vote_dist coh_anchor coh_target lam2_anchor "
    "lam2_target contrast_ratio corr_nearest "
    + " ".join(f"hit{i}_du hit{i}_dv hit{i}_q hit{i}_sep" for i in range(K))
).split()


def available() -> tuple[bool, str]:
    """(usable, reason). The UI greys the option out on False."""
    try:
        import joblib  # noqa: F401, PLC0415
        import sklearn  # noqa: F401, PLC0415
    except ImportError as exc:
        return False, f"needs scikit-learn and joblib ({exc.name} missing)"
    if not DEFAULT_MODEL.exists():
        return False, f"classifier file missing: {DEFAULT_MODEL}"
    return True, ""


# ---------------------------------------------------------------------------
# helpers, identical to the training code
# ---------------------------------------------------------------------------

def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def _hann(size, _c={}):  # noqa: B006 - module-lifetime cache
    if size not in _c:
        h = np.hanning(size + 2)[1:-1]
        _c[size] = np.outer(h, h)
    return _c[size]


def _cut(img, cv, cu, size):
    h = size // 2
    iv, iu = int(round(cv)), int(round(cu))
    if iv - h < 0 or iu - h < 0 or iv - h + size > img.shape[0] \
            or iu - h + size > img.shape[1]:
        return None, None, None
    return (np.asarray(img[iv - h:iv - h + size, iu - h:iu - h + size],
                       np.float32), iv, iu)


def _mp(frames, anchor, view, pred_uv, max_step):
    """match_patch from one anchor, returned as (u, v, q) or None."""
    av, au, avv = int(anchor[0]), anchor[1], anchor[2]
    hit = match_patch(frames[av], frames[view], (avv, au),
                      (pred_uv[1], pred_uv[0]), P, max_step)
    return None if hit is None else (hit[1], hit[0], hit[2])


def _tensor_feats(patch):
    if patch is None or patch.std() < 1e-9:
        return np.nan, np.nan
    l1, l2, _, _ = structure_tensor(patch)
    return coherence(l1, l2), np.log10(max(l2, 1e-12))


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------

def extract_features(frames_hp, anchors, view, pred_uv, max_step):
    """Fused position and the 43-vector the classifier scores.

    anchors: (K, 3) [view, u, v] in loaded px, NEAREST IN ANGLE FIRST.
    pred_uv: (u, v) sinusoid prediction the search box is centred on.
    Returns (u, v, features) or None when no anchor matched.
    """
    from skimage.feature import match_template  # noqa: PLC0415

    anchors = np.asarray(anchors, float)[:K]
    hits = [(a, h) for a, h in ((a, _mp(frames_hp, a, view, pred_uv, max_step))
                                for a in anchors) if h]
    if not hits:
        return None
    u, v, q = np.array([h for _, h in hits]).T
    seps = np.array([abs(a[0] - view) for a, _ in hits])
    w = np.clip(q, 1e-3, None)
    fu, fv = _wmedian(u, w), _wmedian(v, w)
    d = np.hypot(u - fu, v - fv)
    sep_nearest = int(abs(anchors[0, 0] - view))
    f = dict(k_hits=len(hits), q_max=q.max(), q_mean=q.mean(), q_min=q.min(),
             spread=_wmedian(d, w), spread_max=d.max(),
             n_identical=int((d < 0.05).sum()),
             sep_nearest=sep_nearest, sep_mean_hit=seps.mean(),
             resid_u=fu - pred_uv[0], resid_v=fv - pred_uv[1],
             resid=np.hypot(fu - pred_uv[0], fv - pred_uv[1]))
    near = anchors[0]
    hit_near = _mp(frames_hp, near, view, pred_uv, max_step)
    f["q_nearest"] = hit_near[2] if hit_near else np.nan

    # fused normalized cross-correlation over the search region
    R = int(np.ceil(max_step))
    region, _, _ = _cut(frames_hp[view], pred_uv[1], pred_uv[0], P + 2 * R)
    fused, wsum = None, 0.0
    if region is not None:
        for a in anchors:
            tpl, _, _ = _cut(frames_hp[int(a[0])], a[2], a[1], P)
            if tpl is None or tpl.std() < 1e-9:
                continue
            wa = float(np.exp(-abs(a[0] - view) / TAU))
            ncc = match_template(region, (tpl - tpl.mean()) * _hann(P))
            fused = ncc * wa if fused is None else fused + ncc * wa
            wsum += wa
    if fused is not None:
        fused /= wsum
        iy, ix = np.unravel_index(np.argmax(fused), fused.shape)
        peak = fused[iy, ix]
        yy, xx = np.mgrid[:fused.shape[0], :fused.shape[1]]
        far = np.hypot(yy - iy, xx - ix) > 3
        side = fused[far]
        f["ncc_peak"] = peak
        f["ncc_psr"] = ((peak - side.mean()) / (side.std() + 1e-9)
                        if side.size else np.nan)
        f["ncc_second_ratio"] = (side.max() / peak
                                 if side.size and peak > 0 else np.nan)
        f["ncc_vote_dist"] = np.hypot((fv - pred_uv[1]) - (iy - R),
                                      (fu - pred_uv[0]) - (ix - R))
    else:
        f.update(ncc_peak=np.nan, ncc_psr=np.nan, ncc_second_ratio=np.nan,
                 ncc_vote_dist=np.nan)

    pa, _, _ = _cut(frames_hp[int(near[0])], near[2], near[1], P)
    pt, _, _ = _cut(frames_hp[view], fv, fu, P)
    f["coh_anchor"], f["lam2_anchor"] = _tensor_feats(pa)
    f["coh_target"], f["lam2_target"] = _tensor_feats(pt)
    if (pa is not None and pt is not None
            and pa.std() > 1e-9 and pt.std() > 1e-9):
        f["contrast_ratio"] = pt.std() / pa.std()
        f["corr_nearest"] = float(np.corrcoef(pa.ravel(), pt.ravel())[0, 1])
    else:
        f["contrast_ratio"] = f["corr_nearest"] = np.nan

    order = np.argsort(seps)
    for i in range(K):
        if i < len(hits):
            j = order[i]
            f[f"hit{i}_du"], f[f"hit{i}_dv"] = u[j] - fu, v[j] - fv
            f[f"hit{i}_q"], f[f"hit{i}_sep"] = q[j], seps[j]
        else:
            f[f"hit{i}_du"] = f[f"hit{i}_dv"] = np.nan
            f[f"hit{i}_q"] = f[f"hit{i}_sep"] = np.nan
    return fu, fv, np.array([f[n] for n in NAMES], float)


# ---------------------------------------------------------------------------
# the matcher
# ---------------------------------------------------------------------------

class LearnedMatcher:
    """Callable in the `complete_track(matcher=...)` shape."""

    def __init__(self, path: str | Path | None = None, *, model=None):
        if model is None:
            import joblib  # noqa: PLC0415

            bundle = joblib.load(Path(path) if path else DEFAULT_MODEL)
            if list(bundle.get("names", NAMES)) != NAMES:
                raise ValueError("classifier feature set does not match "
                                 "learned_match.NAMES; retrain")
            model = bundle["model"]
        self.model = model

    def __call__(self, frames_hp, seeds_usable, view, pred_vu, max_step):
        seeds = np.asarray(seeds_usable, float)             # (S, 3) view, u, v
        near = np.argsort(np.abs(seeds[:, 0] - view))[:K]
        out = extract_features(frames_hp, seeds[near], int(view),
                               (float(pred_vu[1]), float(pred_vu[0])),
                               float(max_step))
        if out is None:
            return None
        u, v, x = out
        p = float(self.model.predict_proba(x[None])[0, 1])
        return v, u, p
