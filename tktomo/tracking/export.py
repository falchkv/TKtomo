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
  per-view shifts, rotate out the constant in-plane tilt and the per-view
  rotation about the beam. The out-of-plane tilt beta, any
  theta-dependence of alpha, and the per-view rotations about the axis
  and the horizontal cannot be expressed as 2D image transforms and stay
  metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.labels import LabelStore
from tktomo.tracking.model import AxisModel, FitResult, FreeMask

# version 2: label_table gained provenance columns (kind, quality);
# version-1 files are still readable (LabelStore.from_table accepts both).
# version 3: per-view rotations and their free flags; older files load
# with the rotations at zero and fixed.
MODEL_VERSION = 3
ROTATIONS = ("rot_horiz", "rot_beam", "rot_axis")


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
        for name in ROTATIONS:
            f.attrs[f"free_{name}"] = bool(getattr(mask, name))

        f.create_dataset("theta_rad", data=m.theta)
        for name in ("c_coef", "alpha_coef", "beta_coef", "dx", "dy",
                     "a", "b", "y", *ROTATIONS):
            f.create_dataset(name, data=getattr(m, name))
        f.create_dataset("feature_id", data=m.feature_ids)
        f.create_dataset("observed_views", data=fit.observed_views)
        table = f.create_dataset("label_table", data=labels.to_table())
        table.attrs["columns"] = ("feature_id view u_raw v_raw kind "
                                  "quality (kind: 0 manual, 1 auto)")
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
        if version not in (1, 2, MODEL_VERSION):
            raise ValueError(
                f"model_version {version} is not 1, 2 or {MODEL_VERSION}")
        info = json.loads(f.attrs["model"])
        theta = f["theta_rad"][()]
        model = AxisModel(
            theta=theta,
            c_coef=f["c_coef"][()], alpha_coef=f["alpha_coef"][()],
            beta_coef=f["beta_coef"][()],
            dx=f["dx"][()], dy=f["dy"][()],
            feature_ids=f["feature_id"][()].astype(int),
            a=f["a"][()], b=f["b"][()], y=f["y"][()],
            theta_ref=float(info["theta_ref"]),
            theta_scale=float(info["theta_scale"]),
            **{name: (f[name][()] if name in f else np.zeros(theta.size))
               for name in ROTATIONS},
        )
        mask = FreeMask(
            dx=bool(f.attrs["free_dx"]), dy=bool(f.attrs["free_dy"]),
            c=f["free_c"][()].astype(bool),
            alpha=f["free_alpha"][()].astype(bool),
            beta=f["free_beta"][()].astype(bool),
            features=f["free_features"][()].astype(bool),
            **{name: bool(f.attrs.get(f"free_{name}", False))
               for name in ROTATIONS},
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
        for name in ROTATIONS:
            f.create_dataset(f"{name}_rad", data=getattr(m, name))
        f.attrs["rotations_in_shifts"] = False
        f.attrs["rotations_note"] = (
            "sx and sy do not contain the per-view rotations; they are "
            "written as rot_*_rad for provenance only")
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

    The per-view rotations rotate the beam frame itself: the model reads
    p'_k = sum_l Rm[k, l] (P . e_l) with Rm = R(-w) (`AxisModel.beam_rotation`),
    which is P . e_k' with e_k' = sum_l Rm[k, l] e_l, the basis rotated
    by +w in the lab. The closed forms above then hold in the primed basis
    unchanged, so the export stays exact with rotations too.

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
    if model.has_rotations:
        basis = np.stack([e_s, e_t, e_z], axis=1)          # (V, 3 rows, xyz)
        rotated = np.einsum("vkl,vlx->vkx", model.beam_rotation(), basis)
        e_s, e_t, e_z = rotated[:, 0], rotated[:, 1], rotated[:, 2]

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
                            c_ref: float | None = None,
                            det_shape_loaded: tuple[int, int] | None = None
                            ) -> list:
    """Per-view `tktomo.align.Transform` (in the LOADED frame) that undoes
    dx/dy, folds the c(theta) drift into shifts, and derotates the constant
    in-plane tilt plus the per-view rotation about the beam.

    Positive `Transform.rotation` moves content right of the image center
    down (pinned by test), so undoing v = y + alpha*(u - c) takes
    rotation = -degrees(alpha_0). The rotation is about the image center,
    not the axis column; for the constant alpha the difference is a
    constant offset, identical for every view, hence harmless to any
    reconstruction. A per-view rot_beam changes that offset from view to
    view, so it is compensated exactly (`rotation_centre_shift`), which
    needs the loaded image shape `det_shape_loaded` (H, W).
    """
    from tktomo.align.transform import Transform  # noqa: PLC0415

    if chain.view_origin is not None:
        raise ValueError(
            "aligned export of a per-view-cropped stack is not meaningful: "
            "the crop windows already move with the feature")
    c_of, _, _ = model.axis_curves()
    if c_ref is None:
        c_ref = model.center_at_mean_theta()
    alpha0 = float(model.alpha_coef[0])
    has_beam = bool(np.any(model.rot_beam != 0.0))
    if has_beam and det_shape_loaded is None:
        raise ValueError("per-view rot_beam needs det_shape_loaded to place "
                         "the rotation about the axis column")
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
        rot = -float(np.rad2deg(alpha0 + model.rot_beam[j]))
        if has_beam:
            eu, ev = rotation_centre_shift(
                alpha0, float(model.rot_beam[j]), float(c_of[j]), chain,
                det_shape_loaded)
            tx += float(chain.shift_from_parent(eu))
            ty += float(chain.shift_from_parent(ev))
        out.append(Transform(dx=tx, dy=ty, rotation=rot))
    return out


def rotation_centre_shift(alpha0: float, rot_beam: float, c_view: float,
                          chain: CoordinateChain,
                          det_shape_loaded: tuple[int, int],
                          centre_loaded: tuple[float, float] | None = None
                          ) -> tuple[float, float]:
    """The raw-px (u, v) shift that makes an image rotation about the image
    centre act like the model's rotation about the axis column at the top
    row, for the per-view part of the angle.

    The model rotates content about (c, 0) in raw px by alpha0 + rot_beam,
    `Transform` about the image centre. The difference is the translation
    (R(phi) - I)(c0 - c_img); the constant part for alpha0 alone is the
    same in every view and is left in (as before), the per-view part
    (R(alpha0 + rot_beam) - R(alpha0))(c0 - c_img) is returned, zero when
    rot_beam is zero.
    """
    h, w = det_shape_loaded
    if centre_loaded is None:
        centre_loaded = ((w - 1) / 2.0, (h - 1) / 2.0)
    cu, cv = chain.to_parent(centre_loaded[0], centre_loaded[1])
    du, dv = float(c_view) - float(cu), 0.0 - float(cv)

    def rot(phi):
        c, s = np.cos(phi), np.sin(phi)
        return c * du - s * dv, s * du + c * dv

    ru1, rv1 = rot(alpha0 + rot_beam)
    ru0, rv0 = rot(alpha0)
    return ru1 - ru0, rv1 - rv0


def warp_stack(frames, transforms, *, order: int = 1, progress=None) -> np.ndarray:
    """Resample every frame by its transform; `frames[k]` is read one at a time.

    `progress(done, total) -> bool` returning False cancels (RuntimeError).
    """
    from tktomo.align.transform import apply_transform  # noqa: PLC0415

    n = len(frames)
    if len(transforms) != n:
        raise ValueError("model and stack view counts differ")
    out = None
    for k in range(n):
        frame = np.asarray(frames[k], np.float32)
        if out is None:
            out = np.empty((n, *frame.shape), np.float32)
        out[k] = apply_transform(frame, transforms[k], order=order)
        if progress is not None and not progress(k + 1, n):
            raise RuntimeError("aligned-stack export cancelled")
    return out if out is not None else np.empty((0, 0, 0), np.float32)


def aligned_metadata(base: dict, model: AxisModel, chain: CoordinateChain) -> dict:
    """The provenance the aligned stack carries: what was applied and what was NOT."""
    c_ref = model.center_at_mean_theta()
    metadata = dict(base)
    metadata.update({
        "aligned_by": "track_model_app",
        "center_raw_px": float(c_ref),
        "center_loaded_px": float(chain.from_parent(c_ref, 0.0)[0]),
        "alpha_applied_rad": float(model.alpha_coef[0]),
        "rot_beam_applied_rad": model.rot_beam.tolist(),
        "rotation_center": ("image center (constant offset vs axis for "
                            "alpha, per-view part compensated for rot_beam)"),
        "not_applied": json.dumps({
            "beta_coef": model.beta_coef.tolist(),
            "alpha_coef_higher": model.alpha_coef[1:].tolist(),
            "rot_axis_rad": model.rot_axis.tolist(),
            "rot_horiz_rad": model.rot_horiz.tolist(),
        }),
    })
    return metadata


def export_aligned_stack(data, model: AxisModel, chain: CoordinateChain, *,
                         order: int = 1, progress=None):
    """Warp every projection by its `aligned_view_transforms` transform.

    `data` is a `ProjectionData` in the chain's loaded frame; returns a new
    `ProjectionData` (one affine resample per view: shift and rotation are
    composed, not applied twice). Metadata records the fixed center on both
    grids and what was NOT applied (beta, alpha drift), so a downstream
    reconstruction cannot mistake best-effort for exact.
    """
    from tktomo.io import ProjectionData  # noqa: PLC0415

    transforms = aligned_view_transforms(
        model, chain, det_shape_loaded=tuple(np.shape(data.data)[1:3]))
    out = warp_stack(data.data, transforms, order=order, progress=progress)
    return ProjectionData(data=out, angles=np.asarray(data.angles).copy(),
                          metadata=aligned_metadata(data.metadata, model, chain))
