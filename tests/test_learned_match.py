"""The learned matcher: feature contract, hook shape, and the app wiring."""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.tracking.autotrack import AutoTrackParams, complete_track, highpass2d
from tktomo.tracking.learned_match import (
    DEFAULT_MODEL,
    K,
    NAMES,
    LearnedMatcher,
    available,
    extract_features,
)

from .test_autotrack import synthetic_scan


class _Stub:
    """predict_proba that rewards a small sinusoid residual, like the real one."""

    def predict_proba(self, X):
        resid = X[:, NAMES.index("resid")]
        p = 1.0 / (1.0 + resid)
        return np.column_stack([1 - p, p])


def _scene():
    frames, theta, tu, tv = synthetic_scan(n_views=40, jitter=0.0)
    hp = [highpass2d(f) for f in frames]
    views = [0, 9, 19, 29, 39]
    seeds = [(j, float(tu[0, j]), float(tv[0, j])) for j in views]
    return hp, theta, tu, tv, seeds


def test_feature_vector_matches_names_contract():
    hp, theta, tu, tv, seeds = _scene()
    anchors = np.array(sorted(seeds, key=lambda s: abs(s[0] - 14)), float)
    out = extract_features(hp, anchors, 14, (tu[0, 14] + 1.0, tv[0, 14]), 8.0)
    assert out is not None
    u, v, x = out
    assert x.shape == (len(NAMES),) == (23 + 4 * K,)
    assert abs(u - tu[0, 14]) < 0.5 and abs(v - tv[0, 14]) < 0.5
    assert x[NAMES.index("k_hits")] == K
    assert x[NAMES.index("sep_nearest")] == 5
    assert np.isfinite(x[NAMES.index("ncc_peak")])
    # unused hit slots are NaN, used ones are finite
    assert np.isfinite(x[NAMES.index("hit0_q")])


def test_matcher_hook_shape_and_threshold():
    hp, theta, tu, tv, seeds = _scene()
    m = LearnedMatcher(model=_Stub())
    hit = m(hp, seeds, 14, (tv[0, 14], tu[0, 14]), 8.0)
    assert hit is not None
    v, u, p = hit
    assert abs(u - tu[0, 14]) < 0.5 and abs(v - tv[0, 14]) < 0.5
    assert 0.0 <= p <= 1.0

    res = complete_track(hp, theta, seeds, AutoTrackParams(min_corr=0.5),
                         matcher=m)
    assert res.labels
    for al in res.labels:
        assert al.quality >= 0.5
        assert abs(al.u - tu[0, al.view]) < 1.0


@pytest.mark.skipif(not available()[0], reason=available()[1])
def test_shipped_classifier_loads_and_scores():
    hp, theta, tu, tv, seeds = _scene()
    m = LearnedMatcher(DEFAULT_MODEL)
    hit = m(hp, seeds, 14, (tv[0, 14], tu[0, 14]), 8.0)
    assert hit is not None and 0.0 <= hit[2] <= 1.0


def test_app_uses_learned_matcher_only(qtbot):
    pytest.importorskip("PySide6")
    from tktomo.ui.track_model_app import TrackModelWindow

    win = TrackModelWindow()
    qtbot.addWidget(win)
    # the phase-corr dropdown is gone; the learned matcher is the only tracker
    assert not hasattr(win, "auto_matcher")
    assert win.auto_thr_label.text() == "min p"
    assert win.auto_min_corr.value() == pytest.approx(0.20)
    # with scikit-learn, joblib and the shipped model present it is available
    assert win._learned_ok is available()[0]
