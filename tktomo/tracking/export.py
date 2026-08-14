"""Exports of a fitted tracking model: model file, shifts, vectors, stack.

Four consumers, four conventions, all pinned by unit tests:

- `write_model_h5` / `read_model_h5`: the complete fit (coefficients,
  shifts, feature positions, labels, masks, provenance) in one HDF5 file,
  arrays as datasets and free-form dicts as JSON attrs.
- `write_slogger_shifts`: the shift table the slogger graphite-ball
  pipeline consumes (`sy`/`sx` datasets + center/tilt attrs). Sign
  convention is that of the pipeline's `save_shifts`: sy/sx are the
  correction to APPLY, and the center obeys `c_grid = (c_raw - u0 -
  (b-1)/2)/b`.
- `astra_parallel3d_vectors`: per-view 12-vectors for ASTRA's
  `parallel3d_vec` geometry, for an exact geometry-aware GPU
  reconstruction. The detector axes are the DUAL basis of the model's
  (u, v) read-out, not the physical detector edges: that is what makes the
  ASTRA forward projection reproduce the fitted model exactly instead of
  to first order in the tilts. Verifiable without astra installed.
- `aligned_view_transforms` / `export_aligned_stack`: best-effort 2D
  correction of the stack itself: undo dx/dy, fold the c(theta) drift into
  per-view shifts, rotate out the constant in-plane tilt. The out-of-plane
  tilt beta and any theta-dependence of alpha cannot be expressed as 2D
  image transforms and stay metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.labels import LabelStore
from tktomo.tracking.model import AxisModel, FitResult, FreeMask

MODEL_VERSION = 1


# ---------------------------------------------------------------------------
# model file
# ---------------------------------------------------------------------------

def write_model_h5(path: str | Path, fit: FitResult, mask: FreeMask,
                   labels: LabelStore, chain: CoordinateChain, *,
                   source: dict | None = None,
                   diagnostics: dict | None = None,
                   det_shape: tuple[int, int] | None = None) -> None:
    """The complete fit in one file. Everything is in RAW-grid px.

    `det_shape` (H, W in raw px) additionally stores the astra
    `parallel3d_vec` vectors so a cluster-side reconstruction job needs no
    reimplementation of the model.
    """
    import h5py  # noqa: PLC0415

    m = fit.model
    i, j = fit.obs
    with h5py.File(path, "w") as f:
        f.attrs["model_version"] = MODEL_VERSION
        f.attrs["model"] = json.dumps({
            "frame": "raw_px",
            "degrees": list(m.degrees),
            "theta_ref": m.theta_ref,
            "theta_scale": m.theta_scale,
        })
        f.attrs["provenance"] = json.dumps(
            {"chain": chain.to_dict(), "source": source or {}})
        if diagnostics is not None:
            f.attrs["diagnostics"] = json.dumps(_jsonable(diagnostics))
        f.attrs["free_dx"] = bool(mask.dx)
        f.attrs["free_dy"] = bool(mask.dy)

        f.create_dataset("theta_rad", data=m.theta)
        for name in ("c_coef", "alpha_coef", "beta_coef", "dx", "dy",
                     "a", "b", "y"):
            f.create_dataset(name, data=getattr(m, name))
        f.create_dataset("feature_id", data=m.feature_ids)
        f.create_dataset("observed_views", data=fit.observed_views)
        f.create_dataset("label_table", data=labels.to_table())
        f.create_dataset("obs_i", data=i)
        f.create_dataset("obs_j", data=j)
        f.create_dataset("residual_u", data=fit.residual_u)
        f.create_dataset("residual_v", data=fit.residual_v)
        f.create_dataset("weight_u", data=fit.weight_u)
        f.create_dataset("weight_v", data=fit.weight_v)
        f.create_dataset("free_c", data=mask.c)
        f.create_dataset("free_alpha", data=mask.alpha)
        f.create_dataset("free_beta", data=mask.beta)
        f.create_dataset("free_features", data=mask.features)
        if det_shape is not None:
            vec = astra_parallel3d_vectors(m, det_shape)
            d = f.create_dataset("astra_parallel3d_vec", data=vec)
            d.attrs["det_shape"] = [int(det_shape[0]), int(det_shape[1])]
            d.attrs["note"] = (
                "rows are (rayX rayY rayZ dX dY dZ uX uY uZ vX vY vZ) in the "
                "rotating object frame, units raw px; volume z axis is the "
                "detector row direction. Detector axes are the dual basis "
                "of the model read-out (exact, oblique), not unit vectors.")


def read_model_h5(path: str | Path) -> dict:
    """Inverse of `write_model_h5`; returns model/mask/labels/provenance."""
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("model_version", -1))
        if version != MODEL_VERSION:
            raise ValueError(f"model_version {version} is not {MODEL_VERSION}")
        info = json.loads(f.attrs["model"])
        model = AxisModel(
            theta=f["theta_rad"][()],
            c_coef=f["c_coef"][()], alpha_coef=f["alpha_coef"][()],
            beta_coef=f["beta_coef"][()],
            dx=f["dx"][()], dy=f["dy"][()],
            feature_ids=f["feature_id"][()].astype(int),
            a=f["a"][()], b=f["b"][()], y=f["y"][()],
            theta_ref=float(info["theta_ref"]),
            theta_scale=float(info["theta_scale"]),
        )
        mask = FreeMask(
            dx=bool(f.attrs["free_dx"]), dy=bool(f.attrs["free_dy"]),
            c=f["free_c"][()].astype(bool),
            alpha=f["free_alpha"][()].astype(bool),
            beta=f["free_beta"][()].astype(bool),
            features=f["free_features"][()].astype(bool),
        )
        out = {
            "model": model,
            "mask": mask,
            "labels": LabelStore.from_table(f["label_table"][()]),
            "observed_views": f["observed_views"][()].astype(bool),
            "provenance": json.loads(f.attrs["provenance"]),
        }
        if "diagnostics" in f.attrs:
            out["diagnostics"] = json.loads(f.attrs["diagnostics"])
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# slogger shift table
# ---------------------------------------------------------------------------

def write_slogger_shifts(path: str | Path, fit: FitResult,
                         chain: CoordinateChain, *, target_binning: int,
                         source: str = "",
                         center_split_raw_px: float = float("nan")) -> None:
    """The shift table the slogger pipeline's recon consumes.

    Written on the grid that shares the chain's raw crop but is binned
    `target_binning` from raw (2 = the graphite preproc grid). Signs follow
    the pipeline's track_bundle: sy/sx are the CORRECTION to apply, so
    they are minus the fitted per-view displacement; the drift part of
    c(theta) is folded in so the remaining center is the single number
    `center_estimate`.
    """
    import h5py  # noqa: PLC0415

    m = fit.model
    b = int(target_binning)
    c_of, _, _ = m.axis_curves()
    c_ref = m.center_at_mean_theta()
    sx = -(m.dx + (c_of - c_ref)) / b
    sy = -m.dy / b
    center = float(chain.parent_to_grid(c_ref, b))
    split = float(center_split_raw_px) / b
    with h5py.File(path, "w") as f:
        f.create_dataset("sy", data=sy.astype(np.float64))
        f.create_dataset("sx", data=sx.astype(np.float64))
        f.create_dataset("measured_view", data=fit.observed_views)
        f.attrs["stage"] = "track_model_app"
        f.attrs["center_estimate"] = center
        f.attrs["center_split_px"] = split
        f.attrs["center_reliable"] = bool(np.isfinite(split) and split <= 5.0)
        f.attrs["axis_tilt_rad"] = float(m.alpha_coef[0])
        f.attrs["n_tracks_used"] = int(m.feature_ids.size)
        f.attrs["source"] = source
        f.attrs["binning"] = b


# ---------------------------------------------------------------------------
# astra parallel3d_vec
# ---------------------------------------------------------------------------

def astra_parallel3d_vectors(model: AxisModel,
                             det_shape: tuple[int, int]) -> np.ndarray:
    """(V, 12) `parallel3d_vec` rows reproducing the fitted model EXACTLY.

    The volume frame is the rotating object frame, (x, y, z) = (a, b, y)
    in raw px; z is the detector row direction. The model reads a point
    P = s*e_s + t*e_t + y*e_z out as

        u = P.e_s + c + dx        v = P.(alpha*e_s + beta*e_t + e_z) + dy

    i.e. the read-out functionals are m1 = e_s and m2 = alpha*e_s +
    beta*e_t + e_z. ASTRA maps P to detector indices by solving
    P = d + p*u_vec + q*v_vec + tau*ray, so (u_vec, v_vec, ray) must be the
    basis DUAL to (m1, m2, m3): then p = m1.(P-d) and q = m2.(P-d) hold
    identically and the forward projection equals the model to machine
    precision, tilts included. Choosing the "physical" v_vec = e_z +
    alpha*e_s + beta*e_t instead would be wrong at second order in the
    tilts (~0.03 px here), which is exactly the kind of silent
    approximation this export exists to avoid.

    With m3 = ray/|ray|^2 and ray = m1 x m2, the dual basis works out to
    (gamma = 1 + beta^2):

        ray   = -e_t + beta*e_z
        u_vec = e_s - (alpha*beta/gamma)*e_t - (alpha/gamma)*e_z
        v_vec = (beta/gamma)*e_t + (1/gamma)*e_z
        d     = -(c + dx - (W-1)/2)*u_vec - (dy - (H-1)/2)*v_vec

    Row layout: (rayX rayY rayZ, dX dY dZ, uX uY uZ, vX vY vZ).
    """
    h, w = int(det_shape[0]), int(det_shape[1])
    theta = model.theta
    n_view = theta.size
    ct, sn = np.cos(theta), np.sin(theta)
    zeros = np.zeros(n_view)
    e_s = np.stack([ct, sn, zeros], axis=1)
    e_t = np.stack([-sn, ct, zeros], axis=1)
    e_z = np.stack([zeros, zeros, np.ones(n_view)], axis=1)

    c_of, alpha_of, beta_of = model.axis_curves()
    gamma = (1.0 + beta_of ** 2)[:, None]
    alpha = alpha_of[:, None]
    beta = beta_of[:, None]

    ray = -e_t + beta * e_z
    u_vec = e_s - (alpha * beta / gamma) * e_t - (alpha / gamma) * e_z
    v_vec = (beta / gamma) * e_t + (1.0 / gamma) * e_z
    p0 = (c_of + model.dx - (w - 1) / 2.0)[:, None]
    q0 = (model.dy - (h - 1) / 2.0)[:, None]
    d = -p0 * u_vec - q0 * v_vec

    return np.concatenate([ray, d, u_vec, v_vec], axis=1)


# ---------------------------------------------------------------------------
# aligned stack
# ---------------------------------------------------------------------------

def aligned_view_transforms(model: AxisModel, chain: CoordinateChain,
                            c_ref: float | None = None) -> list:
    """Per-view `tktomo.align.Transform` (in the LOADED frame) that undoes
    dx/dy, folds the c(theta) drift into shifts, and derotates the constant
    in-plane tilt.

    Positive `Transform.rotation` moves content right of the image center
    down (pinned by test), so undoing v = y + alpha*(u - c) takes
    rotation = -degrees(alpha_0). The rotation is about the image center,
    not the axis column; the difference is a constant vertical offset,
    identical for every view, hence harmless to any reconstruction.
    """
    from tktomo.align.transform import Transform  # noqa: PLC0415

    if chain.view_origin is not None:
        raise ValueError(
            "aligned export of a per-view-cropped stack is not meaningful: "
            "the crop windows already move with the feature")
    c_of, _, _ = model.axis_curves()
    if c_ref is None:
        c_ref = model.center_at_mean_theta()
    rot = -float(np.rad2deg(model.alpha_coef[0]))
    # apply_transform moves content to p' = R(p - c) + c + R d, i.e. it
    # rotates the translation along with the content. That is exactly
    # "shift in the un-rotated frame, then derotate": p' = R(p + d - c) + c.
    # So the fitted shifts go in unmodified; pre-rotating them here would
    # apply the rotation to them twice (found the hard way by the v-flatness
    # tolerance in the aligned-transform test).
    out = []
    for j in range(model.theta.size):
        tx = -float(chain.shift_from_parent(model.dx[j] + (c_of[j] - c_ref)))
        ty = -float(chain.shift_from_parent(model.dy[j]))
        out.append(Transform(dx=tx, dy=ty, rotation=rot))
    return out


def export_aligned_stack(data, model: AxisModel, chain: CoordinateChain, *,
                         order: int = 1, progress=None):
    """Warp every projection by its `aligned_view_transforms` transform.

    `data` is a `ProjectionData` in the chain's loaded frame; returns a new
    `ProjectionData` (one affine resample per view: shift and rotation are
    composed, not applied twice). Metadata records the fixed center on both
    grids and what was NOT applied (beta, alpha drift), so a downstream
    reconstruction cannot mistake best-effort for exact.
    """
    from tktomo.align.transform import apply_transform  # noqa: PLC0415
    from tktomo.io import ProjectionData  # noqa: PLC0415

    transforms = aligned_view_transforms(model, chain)
    n = data.data.shape[0]
    if len(transforms) != n:
        raise ValueError("model and stack view counts differ")
    out = np.empty_like(data.data, dtype=np.float32)
    for k in range(n):
        out[k] = apply_transform(np.asarray(data.data[k], np.float32),
                                 transforms[k], order=order)
        if progress is not None and not progress(k + 1, n):
            raise RuntimeError("aligned-stack export cancelled")

    c_ref = model.center_at_mean_theta()
    metadata = dict(data.metadata)
    metadata.update({
        "aligned_by": "track_model_app",
        "center_raw_px": float(c_ref),
        "center_loaded_px": float(chain.from_parent(c_ref, 0.0)[0]),
        "alpha_applied_rad": float(model.alpha_coef[0]),
        "rotation_center": "image center (constant vertical offset vs axis)",
        "not_applied": json.dumps({
            "beta_coef": model.beta_coef.tolist(),
            "alpha_coef_higher": model.alpha_coef[1:].tolist(),
        }),
    })
    return ProjectionData(data=out, angles=data.angles.copy(),
                          metadata=metadata)
