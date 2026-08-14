"""Round trips for label storage, keyframe interpolation and stack formats."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from tktomo.tracking.coords import CoordinateChain  # noqa: E402
from tktomo.tracking.labels import (  # noqa: E402
    LabelStore,
    interpolate_track,
    sinusoid_fit_info,
)
from tktomo.tracking.stackio import (  # noqa: E402
    compose_view_origin,
    crop_track_windows,
    detect_format,
    load_tracking_stack,
    write_feature_crop,
)


def test_label_store_basics():
    store = LabelStore()
    store.set(3, 10, 100.0, 50.0)
    store.set(3, 10, 101.0, 51.0)          # move, not duplicate
    store.set(1, 10, 200.0, 60.0)
    store.set(1, 12, 205.0, 61.0)
    assert len(store) == 3
    assert store.get(3, 10) == (101.0, 51.0)
    assert store.feature_ids() == [1, 3]
    assert store.counts() == {1: 2, 3: 1}
    assert store.views_of(1) == [10, 12]
    assert store.in_view(10) == [(1, 200.0, 60.0), (3, 101.0, 51.0)]

    fid, dist = store.nearest(10, 102.0, 51.0)
    assert fid == 3
    assert dist == pytest.approx(1.0)

    assert store.remove(3, 10)
    assert not store.remove(3, 10)
    assert store.clear_feature(1) == 2
    assert len(store) == 0


def test_label_store_arrays_and_table():
    store = LabelStore()
    store.set(5, 0, 10.0, 20.0)
    store.set(2, 3, 30.0, 40.0)
    u, v, valid, ids = store.to_arrays(n_views=5)
    assert ids.tolist() == [2, 5]
    assert valid[0, 3] and valid[1, 0]
    assert valid.sum() == 2
    assert u[0, 3] == 30.0 and v[1, 0] == 20.0

    # explicit id order keeps pinned-feature row indices stable
    u2, _, valid2, ids2 = store.to_arrays(5, feature_ids=[5, 2, 9])
    assert ids2.tolist() == [5, 2, 9]
    assert valid2[0, 0] and valid2[1, 3] and not valid2[2].any()

    table = store.to_table()
    restored = LabelStore.from_table(table)
    assert restored.to_table().tolist() == table.tolist()


def test_interpolate_track_sinusoid_recovery():
    theta = np.linspace(0, np.pi, 40)
    a, b, c = 120.0, -60.0, 430.0
    u_true = a * np.cos(theta) + b * np.sin(theta) + c
    v_true = 90.0 + 3.0 * theta
    keys = [2, 11, 22, 35]
    u_all, v_all = interpolate_track(theta, keys, u_true[keys], v_true[keys],
                                     u_mode="sinusoid", v_mode="linear")
    assert np.allclose(u_all, u_true, atol=1e-9)   # exact: 4 keys, 3 params
    inside = slice(keys[0], keys[-1] + 1)
    assert np.allclose(v_all[inside], v_true[inside], atol=1e-9)

    info = sinusoid_fit_info(theta, keys, u_true[keys])
    assert info["amplitude"] == pytest.approx(np.hypot(a, b))
    assert info["rms"] == pytest.approx(0.0, abs=1e-9)


def test_interpolate_track_fallbacks():
    theta = np.linspace(0, np.pi, 20)
    # 2 keyframes: sinusoid falls back to linear, edges held outside
    u_all, v_all = interpolate_track(theta, [5, 10], [100.0, 110.0],
                                     [50.0, 55.0])
    assert u_all[0] == 100.0 and u_all[-1] == 110.0
    assert u_all[7] == pytest.approx(np.interp(theta[7],
                                               theta[[5, 10]], [100, 110]))
    # 1 keyframe: constant
    u1, v1 = interpolate_track(theta, [4], [77.0], [33.0])
    assert np.all(u1 == 77.0) and np.all(v1 == 33.0)
    with pytest.raises(ValueError):
        interpolate_track(theta, [], [], [])
    assert sinusoid_fit_info(theta, [1, 2], [1.0, 2.0]) is None


def test_crop_track_windows_and_clamping():
    rng = np.random.default_rng(0)
    proj = rng.random((6, 40, 60)).astype(np.float32)
    track_u = np.array([30.0, 3.0, 57.0, 30.0, 30.0, 30.0])
    track_v = np.array([20.0, 20.0, 20.0, 2.0, 38.0, 20.0])
    out, origin = crop_track_windows(proj, track_u, track_v, (16, 20))
    assert out.shape == (6, 16, 20)
    # interior view: window centred on the track
    assert origin[0].tolist() == [12, 20]
    assert np.array_equal(out[0], proj[0, 12:28, 20:40])
    # clamped at every edge
    assert origin[1].tolist() == [12, 0]
    assert origin[2].tolist() == [12, 40]
    assert origin[3].tolist() == [0, 20]
    assert origin[4].tolist() == [24, 20]
    with pytest.raises(ValueError):
        crop_track_windows(proj, track_u, track_v, (64, 20))


def _write_slogger_preproc(path, proj, theta):
    with h5py.File(path, "w") as f:
        f.create_dataset("proj", data=proj)
        f.create_dataset("theta_rad", data=theta)
        f.attrs["binning"] = 2
        f.attrs["extra_bin"] = 1
        f.attrs["crop"] = [8, 690, 10, 1942]
        f.attrs["sign"] = 1


def test_detect_and_load_slogger_preproc(tmp_path):
    proj = np.zeros((4, 12, 16), np.float32)
    theta = np.linspace(0, np.pi, 4)
    path = tmp_path / "preproc.h5"
    _write_slogger_preproc(path, proj, theta)

    assert detect_format(path) == "slogger_preproc"
    data, chain = load_tracking_stack(path)
    assert data.data.shape == (4, 12, 16)
    assert chain.binning == 2
    assert chain.crop == (8, 690, 10, 1942)
    assert chain.view_origin is None
    # the slogger center convention comes out right through this chain
    assert chain.grid_to_parent(435.7, 2) == pytest.approx(10 + 435.7 * 2 + 0.5)


def test_feature_crop_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    proj = rng.random((5, 30, 40)).astype(np.float32)
    theta = np.linspace(0, np.pi, 5)
    source_chain = CoordinateChain(binning=2, crop=(8, 690, 10, 1942))

    track_u = np.linspace(12, 25, 5)
    track_v = np.linspace(18, 10, 5)
    cropped, origin = crop_track_windows(proj, track_u, track_v, (12, 14))

    path = tmp_path / "crop.h5"
    write_feature_crop(path, cropped, theta, origin, source_chain,
                       window=(12, 14), source={"path": "orig.h5"},
                       keyframes=[[0, 12.0, 18.0]], u_mode="sinusoid")

    assert detect_format(path) == "feature_crop"
    data, chain = load_tracking_stack(path)
    assert data.data.shape == (5, 12, 14)
    assert chain.binning == 2
    assert chain.view_origin is not None

    # THE point of the format: a label clicked in the cropped frame maps to
    # the same raw coordinate as the same pixel in the original frame.
    view = 3
    u_in_crop = track_u[view] - origin[view, 1]
    v_in_crop = track_v[view] - origin[view, 0]
    raw_a = chain.to_parent(u_in_crop, v_in_crop, view=view)
    raw_b = source_chain.to_parent(track_u[view], track_v[view])
    assert raw_a[0] == pytest.approx(raw_b[0])
    assert raw_a[1] == pytest.approx(raw_b[1])
    assert data.metadata["u_mode"] == "sinusoid"


def test_compose_view_origin_with_extra_crop():
    chain = CoordinateChain(binning=2, crop=(0, 0, 0, 0),
                            extra_crop=(5, 50, 7, 70),
                            view_origin=np.array([[1.0, 2.0], [3.0, 4.0]]))
    window = np.array([[10, 20], [30, 40]], float)
    composed = compose_view_origin(chain, window)
    assert composed.tolist() == [[10 + 5 + 1, 20 + 7 + 2],
                                 [30 + 5 + 3, 40 + 7 + 4]]
    with pytest.raises(ValueError):
        compose_view_origin(chain, np.zeros((3, 2)))


def test_load_rejects_foreign_h5(tmp_path):
    path = tmp_path / "foreign.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("something", data=np.zeros(3))
    assert detect_format(path) is None
    with pytest.raises(ValueError):
        load_tracking_stack(path)
