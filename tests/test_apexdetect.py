"""The apex detector registry and the matcher that snaps to a detection.

A detector for a real sample lives with that sample, so everything here
runs against a synthetic one: the contract is what is being tested, not
anybody's trained model.
"""

from __future__ import annotations

import numpy as np
import pytest

from tktomo.tracking import apexdetect
from tktomo.tracking.apexdetect import (
    ApexDetector,
    available_apex_detectors,
    detections_array,
    get_apex_detector,
    register_apex_detector,
)
from tktomo.tracking.autotrack import (
    AutoTrackJob,
    AutoTrackParams,
    run_autotrack,
)
from tktomo.tracking.edgetrack import (
    ApexMatcher,
    DetectionCache,
    sign_for_kind,
)
from tktomo.tracking.model import BAOS, LAOS, POINT, RAOS, TAOS


class FakeDetector:
    """Apexes on a known sinusoid, plus a decoy on the far side."""

    native_bin = 2

    def __init__(self, name="fake", amp=25.0, centre=(48.0, 80.0),
                 radius=18.0, n_views=24, full_nx=160):
        self.name = name
        self.calls = 0
        theta = np.linspace(0.0, np.pi, n_views, endpoint=False)
        self.cu = centre[1] + amp * np.cos(theta)
        self.cv = np.full(n_views, centre[0])
        self.radius = radius
        self.full_nx = full_nx

    def detect(self, frame, axis):
        from tktomo.tracking.coords import regrid_uv

        self.calls += 1
        b = self.full_nx // frame.shape[1]    # the grid it was handed
        j = int(round(float(frame[0, 0]) ))   # the view index, planted below
        cu, cv = regrid_uv(self.cu[j], self.cv[j], 1, b)
        r = self.radius / b
        if axis == "u":
            rows = [[cu - r, cv, 0.9, -1.0], [cu + r, cv, 0.8, +1.0],
                    [cu - r + 40.0, cv, 0.7, -1.0]]     # a decoy, far away
        else:
            rows = [[cu, cv - r, 0.9, -1.0], [cu, cv + r, 0.8, +1.0],
                    [cu, cv - r, 0.2, -1.0]]            # a low-score twin
        return np.asarray(rows, float)


def planted_stack(n_views=24, ny=96, nx=160):
    """Frames carrying the view index in a corner, for FakeDetector.

    A whole 8 x 8 block, so the index survives being mean-pooled onto a
    coarser tracking grid.
    """
    frames = np.zeros((n_views, ny, nx), np.float32)
    for j in range(n_views):
        frames[j, :8, :8] = j
    return frames


# ---------------------------------------------------------------- registry

def test_register_and_get(monkeypatch):
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})
    monkeypatch.setattr(apexdetect, "_PLUGINS_LOADED", True)
    det = FakeDetector(name="unit test detector")
    register_apex_detector(det)
    assert "unit test detector" in available_apex_detectors()
    assert get_apex_detector("unit test detector") is det


def test_unknown_detector_says_what_is_available(monkeypatch):
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})
    monkeypatch.setattr(apexdetect, "_PLUGINS_LOADED", True)
    register_apex_detector(FakeDetector(name="only one"))
    with pytest.raises(KeyError) as excinfo:
        get_apex_detector("nope")
    assert "only one" in str(excinfo.value)
    assert "TKTOMO_APEX_PLUGINS" in str(excinfo.value)


def test_a_detector_must_satisfy_the_protocol(monkeypatch):
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})

    class NotOne:
        name = "broken"

    assert not isinstance(NotOne(), ApexDetector)
    with pytest.raises(TypeError):
        register_apex_detector(NotOne())


def test_plugins_load_from_the_environment(monkeypatch, tmp_path):
    module = tmp_path / "tktomo_plugin_under_test.py"
    module.write_text(
        "from tktomo.tracking.apexdetect import register_apex_detector\n"
        "class D:\n"
        "    name = 'from a plugin'\n"
        "    native_bin = 1\n"
        "    def detect(self, frame, axis):\n"
        "        import numpy as np\n"
        "        return np.zeros((0, 4))\n"
        "register_apex_detector(D())\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})
    monkeypatch.setattr(apexdetect, "_PLUGINS_LOADED", False)
    monkeypatch.setenv("TKTOMO_APEX_PLUGINS",
                       "tktomo_plugin_under_test, does.not.exist")
    names = available_apex_detectors()
    assert "from a plugin" in names
    # a broken plugin is reported, never raised: the window must still open
    assert any("does.not.exist" in p for p in apexdetect.plugin_problems())


def test_detections_array_checks_the_contract():
    assert detections_array([]).shape == (0, 4)
    assert detections_array([[1, 2, 0.5, -1]]).shape == (1, 4)
    with pytest.raises(ValueError, match="N, 4"):
        detections_array([[1, 2, 3]])


# ----------------------------------------------------------------- matcher

def test_cache_detects_once_per_frame_and_axis():
    frames = planted_stack()
    det = FakeDetector()
    cache = DetectionCache(det, frames)
    cache(3, "u")
    cache(3, "u")
    assert det.calls == 1
    cache(3, "v")
    cache(4, "u")
    assert det.calls == 3


@pytest.mark.parametrize("kind,axis", [(LAOS, "u"), (RAOS, "u"),
                                       (TAOS, "v"), (BAOS, "v")])
def test_matcher_snaps_to_the_apex_on_its_own_side(kind, axis):
    frames = planted_stack()
    det = FakeDetector()
    cache = DetectionCache(det, frames)
    sign = sign_for_kind(kind)
    m = ApexMatcher(cache=cache, axis=axis, sign=sign)
    view = 7
    truth_u = det.cu[view] + sign * det.radius if axis == "u" else det.cu[view]
    truth_v = det.cv[view] + sign * det.radius if axis == "v" else det.cv[view]
    hit = m(frames, [], view, (truth_v - 1.7, truth_u - 1.7), 6.0)
    assert hit is not None
    v_hit, u_hit, q = hit
    if axis == "u":
        assert u_hit == pytest.approx(truth_u)
        assert v_hit == pytest.approx(truth_v - 1.7)     # never measured
    else:
        assert v_hit == pytest.approx(truth_v)
        assert u_hit == pytest.approx(truth_u - 1.7)     # never measured
    assert 0.0 < q <= 1.0


def test_matcher_ignores_a_detection_outside_the_box():
    frames = planted_stack()
    det = FakeDetector()
    m = ApexMatcher(cache=DetectionCache(det, frames), axis="u", sign=-1.0)
    view = 7
    truth_u = det.cu[view] - det.radius
    assert m(frames, [], view, (det.cv[view], truth_u - 9.0), 2.0) is None
    # inside the box NEAREST wins, so a wide box plus a bad prediction takes
    # the decoy. That is the honest failure mode: keep the box small.
    hit = m(frames, [], view, (det.cv[view], truth_u + 25.0), 30.0)
    assert hit is not None
    assert hit[1] == pytest.approx(truth_u + 40.0)
    hit = m(frames, [], view, (det.cv[view], truth_u + 5.0), 30.0)
    assert hit[1] == pytest.approx(truth_u)


def test_matcher_gates_the_unconstrained_coordinate():
    frames = planted_stack()
    det = FakeDetector()
    m = ApexMatcher(cache=DetectionCache(det, frames), axis="u", sign=-1.0,
                    across_tol=3.0)
    view = 7
    truth_u = det.cu[view] - det.radius
    assert m(frames, [], view, (det.cv[view] + 20.0, truth_u), 6.0) is None
    assert m(frames, [], view, (det.cv[view] + 1.0, truth_u), 6.0) is not None


def test_matcher_min_score_drops_a_weak_detection():
    frames = planted_stack()
    det = FakeDetector()
    view = 7
    truth_v = det.cv[view] - det.radius
    loose = ApexMatcher(cache=DetectionCache(det, frames), axis="v",
                        sign=-1.0, min_score=0.1)
    assert loose(frames, [], view, (truth_v, det.cu[view]), 6.0) is not None
    strict = ApexMatcher(cache=DetectionCache(det, frames), axis="v",
                         sign=-1.0, min_score=0.95)
    assert strict(frames, [], view, (truth_v, det.cu[view]), 6.0) is None


# ------------------------------------------------------------ run_autotrack

def test_run_autotrack_uses_the_detector_and_its_grid(monkeypatch):
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})
    monkeypatch.setattr(apexdetect, "_PLUGINS_LOADED", True)
    det = FakeDetector(name="grid detector")
    register_apex_detector(det)

    frames = planted_stack()
    theta = np.linspace(0.0, np.pi, frames.shape[0], endpoint=False)
    seeds = tuple((j, float(det.cu[j] - det.radius), float(det.cv[j]))
                  for j in (0, 8, 16))

    def refuse(*_a, **_k):
        raise AssertionError("the blob matcher was called for a tangent")

    job = AutoTrackJob(fid=0, seeds=seeds,
                       params=AutoTrackParams(min_corr=0.3, hp_sigma=12.0,
                                              search_radius=14.0),
                       track_bin=4, kind=LAOS, detector="grid detector")
    (fid, result), = run_autotrack(frames, theta, [job], hp_sigma=12.0,
                                   matcher=refuse)
    assert fid == 0
    assert result.stats["matcher"] == "apex:grid detector"
    # the detector's grid wins over the track_bin the job asked for
    assert result.stats["track_bin"] == det.native_bin
    assert len(result.labels) > 8
    assert all(al.quality == pytest.approx(0.9) for al in result.labels)


def test_run_autotrack_shares_one_detection_pass(monkeypatch):
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})
    monkeypatch.setattr(apexdetect, "_PLUGINS_LOADED", True)
    det = FakeDetector(name="shared")
    register_apex_detector(det)

    frames = planted_stack()
    theta = np.linspace(0.0, np.pi, frames.shape[0], endpoint=False)
    params = AutoTrackParams(min_corr=0.3, hp_sigma=12.0)
    jobs = []
    for fid, kind, sign in ((0, LAOS, -1.0), (1, RAOS, +1.0)):
        seeds = tuple((j, float(det.cu[j] + sign * det.radius),
                       float(det.cv[j])) for j in (0, 8, 16))
        jobs.append(AutoTrackJob(fid=fid, seeds=seeds, params=params,
                                 track_bin=1, kind=kind, detector="shared"))
    run_autotrack(frames, theta, jobs, hp_sigma=12.0, matcher=None)
    # two features over the same views, one detector pass per view
    assert det.calls <= frames.shape[0]


def test_point_features_never_reach_the_detector(monkeypatch):
    monkeypatch.setattr(apexdetect, "_REGISTRY", {})
    monkeypatch.setattr(apexdetect, "_PLUGINS_LOADED", True)
    det = FakeDetector(name="unused")
    register_apex_detector(det)
    frames = planted_stack()
    theta = np.linspace(0.0, np.pi, frames.shape[0], endpoint=False)
    calls = []

    def matcher(frames_hp, seeds, view, pred, max_step):
        calls.append(view)
        return None

    job = AutoTrackJob(fid=0, seeds=((0, 60.0, 48.0), (8, 62.0, 48.0)),
                       params=AutoTrackParams(hp_sigma=12.0), track_bin=1,
                       kind=POINT, detector="unused")
    run_autotrack(frames, theta, [job], hp_sigma=12.0, matcher=matcher)
    assert det.calls == 0
    assert calls                       # the blob matcher ran instead
