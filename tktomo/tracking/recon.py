"""The live gridrec slice, split into a pixel-free plan and a pixel-consuming run.

`plan_slice` is what the track-model window computes from the fitted model:
which detector rows are needed and how each view must be shifted and
derotated before one row is handed to gridrec. `reconstruct_slice` takes
that plan plus the row slab and does the work. The split exists so the slab
never has to travel: a remote stack host receives the `SliceRequest`, cuts
the slab out of the stack it already holds, and returns the 2-D slice.

Both halves are numpy-only (tomopy and scipy imported lazily), so the same
function runs in the local worker thread and on a cluster node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.model import AxisModel

__all__ = ["SliceRequest", "plan_slice", "reconstruct_slice"]


@dataclass(frozen=True)
class SliceRequest:
    """Everything a slice reconstruction needs except the pixels.

    Rows and shifts are in the LOADED frame at bin 1; `extra_bin` is applied
    inside `reconstruct_slice`, which rescales shifts, center and row by the
    pixel-center rule. `row` is the requested detector row; `lo:hi` is the
    slab that must be read so shifting and derotating still leave that row
    valid.
    """

    row: int
    lo: int
    hi: int
    sy: np.ndarray          # (n,) per-view vertical shift, loaded px
    sx: np.ndarray          # (n,) per-view horizontal shift, loaded px
    rot_deg: np.ndarray     # (n,) per-view derotation, degrees
    center: float           # rotation axis, loaded px
    extra_bin: int = 1
    #: (n,) per-view increment of the projection angle, rad (the model's
    #: rot_axis), added to the stack's angles before gridrec. Appended
    #: with a default so an older window still talks to a newer server;
    #: the reverse needs the server to be at least as new as the window.
    dtheta: np.ndarray | None = None

    @property
    def row_in_slab(self) -> int:
        return int(self.row - self.lo)


def plan_slice(model: AxisModel, chain: CoordinateChain, n_rows: int,
               width: int, row: int, extra_bin: int = 1) -> SliceRequest:
    """Turn the fitted model into the per-view corrections for one row.

    Full per-view 2D correction, same semantics as the aligned-stack
    export: shift AND derotation by the in-plane tilt alpha(theta) plus the
    per-view rotation about the beam, so the slice responds to the tilt
    parameters. (The rotation is about the slab center; versus the axis
    column that is one constant offset for alpha, and the per-view part
    for rot_beam is compensated, as in the aligned export.) The per-view
    rotation about the axis goes to gridrec as an angle increment
    (`dtheta`). beta and rot_horiz need 3D geometry and stay out, as
    everywhere in 2D.
    """
    from tktomo.tracking.export import rotation_centre_shift  # noqa: PLC0415

    c_of, alpha_of, _ = model.axis_curves()
    c_ref = model.center_at_mean_theta()
    alpha0 = float(model.alpha_coef[0])
    phi = alpha_of + model.rot_beam
    sx = -chain.shift_from_parent(model.dx + (c_of - c_ref))
    sy = -chain.shift_from_parent(model.dy)
    rot_deg = -np.rad2deg(phi)
    # the slab must be tall enough that shifting AND derotating still
    # leaves the middle row valid: rotation moves rows by up to
    # |phi| * width/2 at the image edges
    margin = 4 + int(np.ceil(np.abs(phi).max() * width / 2.0
                             + np.abs(sy).max()))
    row = int(np.clip(row, 0, n_rows - 1))
    lo = max(0, row - margin)
    hi = min(n_rows, row + margin + 1)
    if np.any(model.rot_beam != 0.0):
        centre = ((width - 1) / 2.0, (lo + hi - 1) / 2.0)
        for j in range(model.theta.size):
            eu, ev = rotation_centre_shift(alpha0, float(model.rot_beam[j]),
                                           float(c_of[j]), chain,
                                           (hi - lo, width), centre)
            sx[j] += chain.shift_from_parent(eu)
            sy[j] += chain.shift_from_parent(ev)
        extra = int(np.ceil(np.abs(sy).max())) + 1
        lo = max(0, row - margin - extra)
        hi = min(n_rows, row + margin + extra + 1)
    center = float(chain.from_parent(c_ref, 0.0)[0])
    dtheta = (model.rot_axis.copy() if np.any(model.rot_axis != 0.0)
              else None)
    return SliceRequest(row=row, lo=lo, hi=hi,
                        sy=np.asarray(sy, float), sx=np.asarray(sx, float),
                        rot_deg=np.asarray(rot_deg, float), center=center,
                        extra_bin=int(extra_bin), dtheta=dtheta)


def reconstruct_slice(slab: np.ndarray, theta: np.ndarray,
                      req: SliceRequest) -> np.ndarray:
    """gridrec of `req.row` from `slab = stack[:, req.lo:req.hi, :]`.

    Returns the 2-D slice. Binning happens BEFORE warping: isotropic
    mean-pooling preserves the rotation angle, and shifts/center rescale by
    the pixel-center rule. The slice lands within half a binned pixel of the
    requested row, which is what a speed preview is for.
    """
    from tktomo.align.transform import (  # noqa: PLC0415
        Transform,
        apply_transform,
    )
    from tktomo.recon import get_backend  # noqa: PLC0415

    slab = np.array(slab, np.float32)
    theta = np.asarray(theta, float)
    if req.dtheta is not None:
        theta = theta + np.asarray(req.dtheta, float)
    sy, sx = np.asarray(req.sy, float), np.asarray(req.sx, float)
    rot_deg = np.asarray(req.rot_deg, float)
    center = float(req.center)
    row_in_slab = req.row_in_slab
    extra_bin = int(req.extra_bin)
    if extra_bin > 1:
        from tktomo.ptycho_align.core.preprocess import (  # noqa: PLC0415
            bin_stack,
        )
        slab = bin_stack(slab, extra_bin)
        sy = sy / extra_bin
        sx = sx / extra_bin
        center = (center - (extra_bin - 1) / 2.0) / extra_bin
        row_in_slab = min(row_in_slab // extra_bin, slab.shape[1] - 1)

    for k in range(slab.shape[0]):
        t = Transform(dx=sx[k], dy=sy[k], rotation=rot_deg[k])
        if not t.is_identity():
            slab[k] = apply_transform(slab[k], t, order=1)
    # the requested detector row's position inside the slab; NOT the slab
    # middle, which drifts when the slab is clipped at the top or bottom
    sino = slab[:, row_in_slab:row_in_slab + 1, :]
    volume = get_backend("tomopy").reconstruct(
        sino, theta, center=center, algorithm="gridrec")
    return np.asarray(volume)[0]
