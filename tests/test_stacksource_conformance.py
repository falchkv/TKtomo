"""Every StackSource verb, against the in-process source and a real socket.

The remote source must be indistinguishable from the local one to the
window, so each test runs on both. One documented exception: frames come
back quantised to display precision over the wire (`FramePacket`), so pixel
comparisons allow the half-level that costs, and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers_tracking import write_blob_file
from tktomo.tracking.autotrack import AutoTrackJob, AutoTrackParams, patch_size
from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.model import AxisModel
from tktomo.tracking.recon import plan_slice
from tktomo.tracking.stacksource import (
    AlignedExportRequest,
    LocalStackSource,
    StackInfo,
)


def frame_tol(source, frame) -> float:
    """How far a returned frame may sit from the file.

    Half a quantisation level for a packed source, plus the float32 rounding
    of undoing the scale. Zero for the local source, which is exact.
    """
    if getattr(source, "packs_frames", False):
        span = float(frame.max() - frame.min())
        peak = float(np.abs(frame).max())
        return span / (2 * 65535.0) + float(np.spacing(np.float32(peak)))
    return 0.0


@pytest.fixture(scope="module")
def blob_file(tmp_path_factory):
    return write_blob_file(tmp_path_factory.mktemp("blob") / "blob.h5")


@pytest.fixture(params=["local", "remote"])
def source(request):
    if request.param == "local":
        src = LocalStackSource()
        yield src
        src.close()
        return
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    from tktomo.tracking.remote import RemoteStackSource, make_server

    server = make_server("tcp://127.0.0.1:*").start()
    src = RemoteStackSource(server.endpoint, timeout=60.0)
    yield src
    src.close()
    server.stop()


def test_no_stack_before_open(source):
    assert source.info() is None
    assert source.shape == (0, 0, 0)
    assert source.angles.size == 0
    with pytest.raises(Exception):
        source.view(0)


def test_detect_and_open(source, blob_file):
    path, stack, theta, _, _ = blob_file
    assert source.detect_format(path) == "slogger_preproc"
    reports = []
    info = source.open_stack(path, progress=lambda f, m: reports.append(f))
    assert isinstance(info, StackInfo)
    assert info.shape == stack.shape
    assert info.kind == "slogger_preproc"
    assert info.binning == 2 and info.crop == (10, 154, 20, 244)
    assert info.path == path
    assert np.allclose(info.angles, theta)
    assert source.info() is info or source.info().epoch == info.epoch
    assert source.n_views == stack.shape[0]
    assert np.allclose(source.angles, theta)


def test_views_are_copies_and_bounds_checked(source, blob_file):
    path, stack, *_ = blob_file
    source.open_stack(path)
    frame = source.view(7)
    tol = frame_tol(source, stack[7])
    assert frame.dtype == np.float32
    assert np.allclose(frame, stack[7], rtol=0, atol=tol)
    assert frame.flags.writeable
    frame[:] = 0
    assert np.allclose(source.view(7), stack[7], rtol=0, atol=tol)   # untouched
    a, b = source.views([3, 9])
    assert np.allclose(a, stack[3], rtol=0, atol=tol)
    assert np.allclose(b, stack[9], rtol=0, atol=tol)
    with pytest.raises(IndexError):
        source.view(stack.shape[0])


def test_epoch_bumps_on_reopen(source, blob_file):
    path, *_ = blob_file
    first = source.open_stack(path)
    second = source.open_stack(path)
    assert second.epoch > first.epoch


def test_set_angles(source, blob_file):
    path, stack, *_ = blob_file
    source.open_stack(path)
    new = np.linspace(0.1, 2.0, stack.shape[0])
    info = source.set_angles(new)
    assert np.allclose(info.angles, new)
    assert np.allclose(source.angles, new)
    with pytest.raises(ValueError):
        source.set_angles(new[:-1])


def test_list_hdf5(source, blob_file):
    path, *_ = blob_file
    names = {e.path for e in source.list_hdf5(path)}
    assert "/proj" in names and "/theta_rad" in names


def test_gridrec_slice_matches_direct(source, blob_file):
    pytest.importorskip("tomopy")
    from tktomo.tracking.recon import reconstruct_slice

    path, stack, theta, *_ = blob_file
    source.open_stack(path)
    model = AxisModel.blank(theta, np.arange(1))
    model.alpha_coef[0] = -0.01
    req = plan_slice(model, CoordinateChain(binning=2), stack.shape[1],
                     stack.shape[2], 36, 1)
    got = source.gridrec_slice(req)
    want = reconstruct_slice(stack[:, req.lo:req.hi, :], theta, req)
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-5)


def test_autotrack_and_cancel(source, blob_file):
    path, stack, theta, u, v = blob_file
    ok, why = source.autotrack_available()
    if not ok:
        pytest.skip(why)
    source.open_stack(path)
    seeds = tuple((w, float(u[w]), float(v[w])) for w in (5, 20, 35))
    job = AutoTrackJob(fid=0, seeds=seeds,
                       params=AutoTrackParams(patch=patch_size(10.0),
                                              search_radius=8.0,
                                              min_corr=0.2))
    seen = []
    out = source.autotrack([job], hp_sigma=12.0,
                           progress=lambda d, t, f: seen.append((d, t, f)))
    assert len(out) == 1 and out[0][0] == 0
    labels = out[0][1].labels
    assert len(labels) >= 20
    for al in labels:
        assert abs(al.u - u[al.view]) < 2.0 and abs(al.v - v[al.view]) < 2.0
    assert any(f is None for _, _, f in seen)      # high-pass progress
    assert any(f == 0 for _, _, f in seen)          # per-feature progress

    cancelled = source.autotrack([job], hp_sigma=12.0, cancelled=lambda: True)
    assert cancelled == []


def test_export_aligned(source, blob_file, tmp_path):
    from tktomo.io import load_projections

    path, stack, theta, *_ = blob_file
    source.open_stack(path)
    n = stack.shape[0]
    req = AlignedExportRequest(dx=np.full(n, 1.0), dy=np.zeros(n),
                               rot_deg=np.zeros(n),
                               metadata={"center_loaded_px": 55.5})
    out = tmp_path / f"aligned_{source.is_remote}.h5"
    reports = []
    written = source.export_aligned(req, str(out),
                                    progress=lambda f, m: reports.append(f))
    assert written == str(out) and out.exists()
    data = load_projections(out)
    assert data.data.shape == stack.shape
    assert reports and reports[-1] == pytest.approx(1.0)
    # shifted by one column: content moved, so column 60 now holds column 59
    assert np.allclose(data.data[0, :, 60], stack[0, :, 59], atol=1e-4)
