"""Convention-pinning tests for the tracking exports.

Signs and grids in this file are contracts with external consumers (the
slogger pipeline, ASTRA, any reconstruction of the aligned stack). If one
of these tests fails after an edit, the edit changed a convention, not a
detail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tktomo.align.feature import forward_points  # noqa: E402
from tktomo.tracking.coords import CoordinateChain  # noqa: E402
from tktomo.tracking.export import (  # noqa: E402
    aligned_view_transforms,
    astra_parallel3d_vectors,
    export_aligned_stack,
    read_model_h5,
    write_model_h5,
    write_slogger_shifts,
)
from tktomo.tracking.labels import LabelStore  # noqa: E402
from tktomo.tracking.model import AxisModel, FreeMask, residuals  # noqa: E402

from test_tracking_model import make_truth, sample_labels  # noqa: E402


def make_fit(truth=None, **kw):
    truth = truth or make_truth(**kw)
    u, v, valid = sample_labels(truth, noise=0.0)
    return residuals(u, v, valid, truth), u, v, valid


def test_astra_vectors_reproduce_model_exactly():
    truth = make_truth(degrees=(1, 1, 1), n_view=24)
    det_shape = (341, 966)
    vec = astra_parallel3d_vectors(truth, det_shape)
    assert vec.shape == (24, 12)

    u_model, v_model = truth.predict()
    h, w = det_shape
    rng = np.random.default_rng(0)
    for j in rng.choice(truth.theta.size, 6, replace=False):
        ray, d = vec[j, 0:3], vec[j, 3:6]
        u_vec, v_vec = vec[j, 6:9], vec[j, 9:12]
        basis = np.column_stack([u_vec, v_vec, ray])
        for f in rng.choice(truth.feature_ids.size, 4, replace=False):
            point = np.array([truth.a[f], truth.b[f], truth.y[f]])
            p, q, _ = np.linalg.solve(basis, point - d)
            assert p + (w - 1) / 2.0 == pytest.approx(u_model[f, j], abs=1e-9)
            assert q + (h - 1) / 2.0 == pytest.approx(v_model[f, j], abs=1e-9)


def test_slogger_shifts_signs_and_center(tmp_path):
    # hand-built binning-2 case on the graphite preproc grid convention
    theta = np.linspace(0, np.pi, 8)
    model = AxisModel.blank(theta, np.arange(3), (1, 0, 0))
    chain = CoordinateChain(binning=2, crop=(8, 690, 10, 1942))
    center_grid = 435.7
    model.c_coef[0] = 10 + center_grid * 2 + 0.5     # raw px
    model.c_coef[1] = 4.0                            # drift folds into sx
    model.dx = np.linspace(-2.0, 2.0, 8)             # raw px
    model.dy = np.full(8, 1.0)

    fit, *_ = make_fit_from_model(model)
    path = tmp_path / "shifts.h5"
    write_slogger_shifts(path, fit, chain, target_binning=2,
                         source="unit-test", center_split_raw_px=2.0)
    with h5py.File(path, "r") as f:
        sx, sy = f["sx"][()], f["sy"][()]
        attrs = dict(f.attrs)

    c_of, _, _ = model.axis_curves()
    assert np.allclose(sx, -(model.dx + c_of - model.c_coef[0]) / 2.0)
    assert np.allclose(sy, -model.dy / 2.0)
    assert attrs["center_estimate"] == pytest.approx(center_grid)
    assert attrs["center_split_px"] == pytest.approx(1.0)
    assert attrs["center_reliable"]
    assert attrs["axis_tilt_rad"] == pytest.approx(0.0)
    assert attrs["stage"] == "track_model_app"


def make_fit_from_model(model):
    u, v = model.predict()
    valid = np.ones(u.shape, bool)
    return residuals(u, v, valid, model), u, v, valid


def test_aligned_transforms_remove_shifts_drift_and_tilt():
    theta = np.linspace(0, np.pi, 12)
    model = AxisModel.blank(theta, np.arange(4), (1, 0, 0))
    rng = np.random.default_rng(2)
    model.a = np.array([60.0, -40.0, 10.0, 90.0])
    model.b = np.array([-30.0, 70.0, -80.0, 5.0])
    model.y = np.array([40.0, 60.0, 80.0, 100.0])
    model.c_coef = np.array([120.0, 6.0])
    model.alpha_coef = np.array([-0.01])
    model.dx = 2.0 * rng.standard_normal(12)
    model.dy = 1.5 * rng.standard_normal(12)

    chain = CoordinateChain(binning=1)          # loaded frame == raw frame
    image_shape = (161, 241)
    transforms = aligned_view_transforms(model, chain)
    u_model, v_model = model.predict()

    aligned_u = np.empty_like(u_model)
    aligned_v = np.empty_like(v_model)
    for j, t in enumerate(transforms):
        pts = np.column_stack([u_model[:, j], v_model[:, j]])
        out = forward_points(pts, t, image_shape)
        aligned_u[:, j], aligned_v[:, j] = out[:, 0], out[:, 1]

    # After alignment each feature must trace a PURE sinusoid about a fixed
    # per-feature center and sit at constant height. The per-feature center
    # is c_ref only to within |alpha|*(y - v_center): derotating the tilt
    # genuinely moves u by alpha*(v - v_center), that offset is a property
    # of the geometry, not an alignment error.
    ct, sn = np.cos(theta), np.sin(theta)
    s = model.a[:, None] * ct + model.b[:, None] * sn
    du = aligned_u - s
    assert np.max(np.std(du, axis=1)) < 0.005         # sinusoid + constant
    assert np.max(np.std(aligned_v, axis=1)) < 0.005  # flat height
    alpha = model.alpha_coef[0]
    max_off = abs(alpha) * image_shape[0] / 2 + 0.05
    assert np.max(np.abs(du.mean(axis=1) - model.c_coef[0])) < max_off


def test_aligned_transforms_reject_moving_crop():
    theta = np.linspace(0, np.pi, 4)
    model = AxisModel.blank(theta, np.arange(2))
    chain = CoordinateChain(binning=1, view_origin=np.zeros((4, 2)))
    with pytest.raises(ValueError):
        aligned_view_transforms(model, chain)


def test_export_aligned_stack_moves_a_dot():
    from tktomo.io import ProjectionData

    theta = np.linspace(0, np.pi, 5)
    model = AxisModel.blank(theta, np.arange(1))
    model.a[0], model.b[0], model.y[0] = 20.0, 0.0, 40.0
    model.c_coef[0] = 60.0
    model.dx = np.array([0.0, 4.0, -3.0, 2.0, 0.0])
    model.dy = np.array([0.0, -2.0, 3.0, 1.0, 0.0])
    u_model, v_model = model.predict()

    stack = np.zeros((5, 81, 121), np.float32)
    for j in range(5):
        stack[j, int(round(v_model[0, j])), int(round(u_model[0, j]))] = 1.0
    data = ProjectionData(data=stack, angles=theta, metadata={})
    chain = CoordinateChain(binning=1)

    aligned = export_aligned_stack(data, model, chain)
    ct, sn = np.cos(theta), np.sin(theta)
    for j in range(5):
        r, c = np.unravel_index(np.argmax(aligned.data[j]),
                                aligned.data[j].shape)
        assert abs(c - (20.0 * ct[j] + 60.0)) <= 1.0
        assert abs(r - 40.0) <= 1.0
    assert aligned.metadata["center_raw_px"] == pytest.approx(60.0)
    assert "beta_coef" in aligned.metadata["not_applied"]


def test_export_aligned_stack_progress_cancel():
    from tktomo.io import ProjectionData

    theta = np.linspace(0, np.pi, 3)
    model = AxisModel.blank(theta, np.arange(1))
    data = ProjectionData(data=np.zeros((3, 8, 8), np.float32),
                          angles=theta, metadata={})
    with pytest.raises(RuntimeError):
        export_aligned_stack(data, model, CoordinateChain(),
                             progress=lambda done, total: False)


def test_model_h5_round_trip(tmp_path):
    truth = make_truth(degrees=(1, 1, 0), n_feat=5, n_view=16)
    fit, u, v, valid = make_fit(truth)
    mask = FreeMask.all_free(truth)
    mask.dx = False
    mask.features[2] = False
    labels = LabelStore()
    i, j = np.nonzero(valid)
    for k in range(i.size):
        labels.set(int(truth.feature_ids[i[k]]), int(j[k]),
                   u[i[k], j[k]], v[i[k], j[k]])
    chain = CoordinateChain(binning=2, crop=(8, 690, 10, 1942))

    path = tmp_path / "model.h5"
    write_model_h5(path, fit, mask, labels, chain,
                   source={"path": "stack.h5"},
                   diagnostics={"center_split_px": np.float64(0.4)},
                   det_shape=(341, 966))
    out = read_model_h5(path)

    m = out["model"]
    assert np.allclose(m.c_coef, truth.c_coef)
    assert np.allclose(m.a, truth.a)
    assert np.allclose(m.dx, truth.dx)
    assert m.degrees == truth.degrees
    assert not out["mask"].dx
    assert not out["mask"].features[2]
    assert out["mask"].features[0]
    assert len(out["labels"]) == len(labels)
    assert out["labels"].to_table().tolist() == labels.to_table().tolist()
    assert out["provenance"]["chain"]["binning"] == 2
    assert out["diagnostics"]["center_split_px"] == pytest.approx(0.4)
    with h5py.File(path, "r") as f:
        assert f["astra_parallel3d_vec"].shape == (16, 12)
