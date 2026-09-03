"""Deformation vector fields: representation, warping, composition, optical flow.

The machinery under the non-rigid aligner of Odstrcil et al., *Ab initio nonrigid
X-ray nanotomography*, Nat. Commun. **10**, 4778 (2019). Rigid alignment gives one
translation per projection; when the sample itself changes shape during a long scan
no set of translations can satisfy every projection at once, and the reprojection
residual plateaus above zero. The fix is to let the projection lines bend, which is
implemented here as *warping the volume before projecting it* -- mathematically the
same thing, and far easier to bolt onto an existing projector than a curved-ray
forward model.

Everything in this module is numpy + scipy. scipy is imported inside the functions
that need it, matching the rest of ``ptycho_align.core``.

THREE CONVENTIONS, each of which is a bug if you get it wrong. They are the direct
analogue of the three in :mod:`~tktomo.ptycho_align.core.engine`, and they bite
harder here because a sign error in a vector field does not announce itself as a
diverging number -- it quietly produces a plausible-looking, wrong volume.

1. **Pull-back, not push-forward.** A field ``u`` is applied as

       warp_volume(V, u)[p] = V[p + u(p)]

   i.e. ``u(p)`` says *where in the input to read* for output voxel ``p``. It is a
   backward / pull-back warp, the convention ``scipy.ndimage.map_coordinates``
   implements natively, so applying a field never needs it inverted first. The other
   convention (``warped[p + u(p)] = V[p]``, a forward splat) needs scattered writes,
   leaves holes, and is the classic sign bug: it differs from this one by the
   *inverse* of the field, which for small deformations looks like a minus sign and
   therefore produces a result that is wrong by exactly twice the deformation.

2. **Flow direction.** ``estimate_flow(reference, moving)`` returns the field ``u``
   with ``warp_volume(moving, u) ~= reference``. "Reference" is the fixed image,
   "moving" is the one that gets warped, as in every registration package. Swapping
   the arguments returns (approximately) the inverse field, so the aligner would warp
   its volume the wrong way and *increase* the residual.

3. **Never re-warp warped data.** Interpolation is lossy; two trilinear warps blur a
   volume measurably more than one. So fields are chained with :func:`compose` and
   :func:`invert` -- which resample the *coarse vector field*, a few thousand numbers,
   not the volume -- and the resulting single total field is applied once to the
   pristine volume. This is the same rule as the engine's "never re-shift shifted
   data", and it matters more here: a shift is a single interpolation of the data, a
   deformation is one per iteration if you are careless.

**The coarse grid, and the bias-variance trade it controls.** A DVF is stored on a
regular coarse grid of ``(gz, gy, gx)`` nodes spanning the volume, not per voxel.
``grid_spacing`` (in voxels) is the knob:

* Small spacing -> many nodes -> the field can follow fine detail, but the number of
  free parameters approaches the number of voxels and the field will happily absorb
  noise, streaks, and genuine sample structure. It can *invent* features: a dense
  field can warp any volume into any other. This is the failure mode that makes
  non-rigid methods untrustworthy.
* Large spacing -> few nodes -> only smooth, large-scale deformation is
  representable. Real local deformation is under-fitted and leaks into the residual,
  but nothing is fabricated.

Because the coarse grid is a *hard* restriction of the model space rather than a
penalty term, it is the strongest and most predictable of the regularisers here. The
default (``spacing=16`` voxels) puts of order ``(N/16)^3 * 3`` parameters against
``N^3`` voxels of data per subset, i.e. about a 1500-fold reduction. Quote that
number when someone asks whether the deformation is real.

The other two regularisers, in decreasing order of how much work they do:

* **Temporal smoothing** (:meth:`DeformationSequence.smoothed_in_time`). One field
  per time subset, smoothed and then linearly interpolated in acquisition time, so N
  projections share K << N fields. This is what makes the problem tractable at all --
  a per-projection field would be under-determined by a single projection each.
* **Spatial smoothing inside the flow solver** (``alpha``). Real, but largely
  redundant once the field is projected onto the coarse grid; treat it as
  conditioning for the solver rather than as the thing keeping you honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DeformationField",
    "DeformationSequence",
    "coarse_support_mask",
    "compose",
    "estimate_flow",
    "invert",
    "sequence_rms_difference",
    "warp_volume",
]

# Coordinate arrays are built in blocks so a warp of a large volume does not need
# 3 x its size in float32 coordinates all at once. 32 MiB per component.
_COORD_BLOCK_BYTES = 32 << 20


def _interp_order(grid_shape: Sequence[int]) -> int:
    """Cubic where the coarse grid can support it, linear where it cannot.

    ``map_coordinates`` needs at least 4 samples per axis for a cubic spline; a 2- or
    3-node axis (a very coarse grid, or a thin volume) must fall back to linear rather
    than silently mirroring itself into nonsense.
    """
    return 3 if min(grid_shape) >= 4 else 1


def _axis_to_coarse(coords: np.ndarray, extent: int, nodes: int) -> np.ndarray:
    """Full-resolution voxel coordinate -> fractional index into the coarse grid.

    Node ``i`` sits at voxel ``i * (extent - 1) / (nodes - 1)``: the grid spans the
    volume inclusive of both end faces, so extrapolation at the border is never needed.
    """
    if nodes <= 1 or extent <= 1:
        return np.zeros_like(coords, dtype=np.float64)
    return coords * ((nodes - 1) / (extent - 1))


@dataclass(frozen=True)
class DeformationField:
    """A deformation vector field on a coarse grid over a volume.

    ``vectors`` has shape ``(3, gz, gy, gx)``: component 0 is the z (slice / detector
    row) displacement, 1 is y, 2 is x, matching the volume's own axis order, and all
    three are in units of **full-resolution voxels** regardless of how coarse the grid
    is. ``shape`` is the full-resolution volume the field describes.

    Frozen because every operation returns a new field: a field that is mutated after
    a volume has been warped with it is a debugging nightmare, and the aligner keeps
    old fields in its history.
    """

    vectors: np.ndarray
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        if vectors.ndim != 4 or vectors.shape[0] != 3:
            raise ValueError(
                f"vectors must have shape (3, gz, gy, gx), got {vectors.shape}. "
                "Component order is (z, y, x), matching the volume axes."
            )
        shape = tuple(int(s) for s in self.shape)
        if len(shape) != 3:
            raise ValueError(f"shape must be a 3-tuple (nz, ny, nx), got {self.shape!r}")
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "shape", shape)

    # -- construction ---------------------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(s) for s in self.vectors.shape[1:])  # type: ignore[return-value]

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Node spacing in full-resolution voxels, per axis."""
        return tuple(
            (extent - 1) / (nodes - 1) if nodes > 1 else float(extent)
            for extent, nodes in zip(self.shape, self.grid_shape)
        )  # type: ignore[return-value]

    @classmethod
    def zeros(
        cls, shape: Sequence[int], grid_shape: Sequence[int] | None = None, *, spacing: float = 16.0
    ) -> "DeformationField":
        """The identity field (no deformation) on the given volume."""
        shape = tuple(int(s) for s in shape)
        if grid_shape is None:
            grid_shape = cls.grid_for(shape, spacing)
        return cls(np.zeros((3, *grid_shape), dtype=np.float32), shape)

    @staticmethod
    def grid_for(shape: Sequence[int], spacing: float) -> tuple[int, int, int]:
        """Node counts giving approximately ``spacing`` voxels between nodes.

        Never fewer than 2 nodes per axis: a single node is a global translation, which
        is the *rigid* model and must not be reachable from here by accident.
        """
        if spacing <= 0:
            raise ValueError(f"grid spacing must be > 0 voxels, got {spacing}")
        return tuple(max(2, int(round((int(s) - 1) / spacing)) + 1) for s in shape)  # type: ignore[return-value]

    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray,
        *,
        spacing: float = 16.0,
        grid_shape: Sequence[int] | None = None,
        smooth: bool = True,
    ) -> "DeformationField":
        """Project a full-resolution field ``(3, nz, ny, nx)`` onto the coarse grid.

        This is the step that turns a dense optical-flow answer into a *regularised*
        model. ``smooth`` low-passes each component at half the node spacing first, so
        the node values are local averages rather than point samples -- without it, a
        single noisy voxel at a node position becomes a node value and the coarse grid
        stops protecting anything.
        """
        from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: PLC0415

        dense = np.asarray(dense, dtype=np.float32)
        if dense.ndim != 4 or dense.shape[0] != 3:
            raise ValueError(f"dense field must have shape (3, nz, ny, nx), got {dense.shape}")
        shape = tuple(int(s) for s in dense.shape[1:])
        if grid_shape is None:
            grid_shape = cls.grid_for(shape, spacing)
        grid_shape = tuple(int(g) for g in grid_shape)

        node_coords = [
            np.linspace(0.0, extent - 1, nodes) if nodes > 1 else np.array([(extent - 1) / 2.0])
            for extent, nodes in zip(shape, grid_shape)
        ]
        mesh = np.stack(np.meshgrid(*node_coords, indexing="ij"), axis=0).reshape(3, -1)

        sigma = [
            max(0.0, 0.5 * (extent - 1) / (nodes - 1)) if nodes > 1 else 0.0
            for extent, nodes in zip(shape, grid_shape)
        ]
        out = np.empty((3, *grid_shape), dtype=np.float32)
        for c in range(3):
            component = gaussian_filter(dense[c], sigma, mode="nearest") if smooth else dense[c]
            out[c] = map_coordinates(component, mesh, order=1, mode="nearest").reshape(grid_shape)
        return cls(out, shape)

    # -- evaluation -----------------------------------------------------------------

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary full-resolution coordinates.

        ``points`` has shape ``(..., 3)`` in ``(z, y, x)`` voxel units; the result has
        shape ``(3, ...)``. Used by :func:`compose` and :func:`invert`, which is the
        point: chaining fields resamples this small array, never the volume.
        """
        from scipy.ndimage import map_coordinates  # noqa: PLC0415

        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError(f"points must have a trailing axis of 3, got {points.shape}")
        batch = points.shape[:-1]
        coarse = np.empty((3, int(np.prod(batch)) if batch else 1), dtype=np.float64)
        flat = points.reshape(-1, 3)
        for a in range(3):
            coarse[a] = _axis_to_coarse(flat[:, a], self.shape[a], self.grid_shape[a])

        order = _interp_order(self.grid_shape)
        out = np.empty((3, coarse.shape[1]), dtype=np.float32)
        for c in range(3):
            out[c] = map_coordinates(self.vectors[c], coarse, order=order, mode="nearest")
        return out.reshape((3, *batch))

    def dense(self) -> np.ndarray:
        """The field interpolated to every voxel: ``(3, nz, ny, nx)`` float32.

        Costs ``3 * nz * ny * nx * 4`` bytes -- 1.6 GB at 512^3. Prefer
        :func:`warp_volume`, which builds the same information one slab at a time.
        """
        from scipy.ndimage import map_coordinates  # noqa: PLC0415

        order = _interp_order(self.grid_shape)
        axes = [
            _axis_to_coarse(np.arange(extent, dtype=np.float64), extent, nodes)
            for extent, nodes in zip(self.shape, self.grid_shape)
        ]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0).reshape(3, -1)
        out = np.empty((3, *self.shape), dtype=np.float32)
        for c in range(3):
            out[c] = map_coordinates(
                self.vectors[c], mesh, order=order, mode="nearest"
            ).reshape(self.shape)
        return out

    # -- regularisation and statistics -----------------------------------------------

    @property
    def magnitude(self) -> np.ndarray:
        """Per-node displacement length, in voxels."""
        return np.sqrt((self.vectors.astype(np.float64) ** 2).sum(axis=0))

    @property
    def rms_magnitude(self) -> float:
        return float(np.sqrt(np.mean(self.magnitude**2)))

    @property
    def max_magnitude(self) -> float:
        return float(self.magnitude.max()) if self.vectors.size else 0.0

    def clipped(self, max_px: float) -> "DeformationField":
        """Scale down any node whose displacement exceeds ``max_px`` voxels.

        A magnitude cap, not a component cap: clipping components independently would
        rotate the vector, which is a different (and unphysical) deformation. This is
        the last line of defence -- an optical flow that has locked onto a streak can
        return tens of voxels, and a warp that large will make anything match anything.
        """
        if max_px <= 0:
            raise ValueError(f"max_px must be > 0, got {max_px}")
        magnitude = self.magnitude
        scale = np.ones_like(magnitude)
        hot = magnitude > max_px
        if hot.any():
            scale[hot] = max_px / magnitude[hot]
            logger.debug(
                "clipped %d/%d DVF nodes to %.2f px (max was %.2f px)",
                int(hot.sum()),
                magnitude.size,
                max_px,
                float(magnitude.max()),
            )
        return DeformationField(self.vectors * scale.astype(np.float32), self.shape)

    def scaled(self, factor: float) -> "DeformationField":
        """The same field with every displacement multiplied by ``factor``."""
        return DeformationField(self.vectors * np.float32(factor), self.shape)

    def smoothed(self, sigma_nodes: float) -> "DeformationField":
        """Gaussian-smooth the field across the coarse grid (sigma in *nodes*)."""
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415

        if sigma_nodes <= 0:
            return self
        out = np.stack(
            [gaussian_filter(self.vectors[c], sigma_nodes, mode="nearest") for c in range(3)]
        )
        return DeformationField(out, self.shape)


def _node_points(field: DeformationField) -> np.ndarray:
    """The coarse grid's node positions in full-resolution voxel coordinates."""
    axes = [
        np.linspace(0.0, extent - 1, nodes) if nodes > 1 else np.array([(extent - 1) / 2.0])
        for extent, nodes in zip(field.shape, field.grid_shape)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def warp_volume(
    volume: np.ndarray,
    field: DeformationField,
    *,
    order: int = 1,
    mode: str = "nearest",
    cval: float = 0.0,
) -> np.ndarray:
    """Deform ``volume`` by ``field``: ``out[p] = volume[p + field(p)]``.

    A **pull-back** warp (convention 1 in the module docstring). ``field`` says, for
    each output voxel, where to read in the input. So ``field`` describes the state the
    sample was in when those projections were taken, expressed in the coordinates of
    the reference volume.

    ``order=1`` (trilinear) by default. Cubic is more accurate on smooth data but rings
    at the hard edges phase tomograms are full of, and the accuracy gain is not worth
    it when -- following convention 3 -- the volume is warped exactly once from
    pristine data per iteration rather than repeatedly.

    ``mode="nearest"`` extends the border rather than filling with zeros: a field that
    reaches slightly outside the volume at the edge is normal, and a ring of zeros
    there would be registered as structure by the next flow estimate.
    """
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {volume.shape}")
    if tuple(volume.shape) != field.shape:
        raise ValueError(
            f"field describes a {field.shape} volume but was handed {volume.shape}. "
            "A DVF is tied to the grid it was estimated on; resample it, do not reuse it."
        )

    from scipy.ndimage import map_coordinates, spline_filter  # noqa: PLC0415

    source: np.ndarray = volume.astype(np.float32, copy=False)
    prefilter = True
    if order > 1:
        # Prefilter once for the whole volume rather than once per slab.
        source = spline_filter(source, order=order, mode=mode, output=np.float32)
        prefilter = False

    nz, ny, nx = field.shape
    plane = max(1, ny * nx)
    slab = max(1, min(nz, _COORD_BLOCK_BYTES // (4 * plane)))

    out = np.empty(field.shape, dtype=np.float32)
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        grid = np.stack(
            np.meshgrid(
                np.arange(z0, z1, dtype=np.float32),
                np.arange(ny, dtype=np.float32),
                np.arange(nx, dtype=np.float32),
                indexing="ij",
            ),
            axis=0,
        )
        displacement = field.sample(np.moveaxis(grid, 0, -1))
        np.add(grid, displacement, out=grid)
        out[z0:z1] = map_coordinates(
            source, grid, order=order, mode=mode, cval=cval, prefilter=prefilter
        )
    return out


def compose(outer: DeformationField, inner: DeformationField) -> DeformationField:
    """The single field equivalent to applying ``inner`` and then ``outer``.

    ``warp_volume(V, compose(outer, inner)) ~= warp_volume(warp_volume(V, inner), outer)``,
    but with **one** interpolation of the volume instead of two. Derivation, from the
    pull-back definition::

        warp(warp(V, i), o)[p] = warp(V, i)[p + o(p)] = V[p + o(p) + i(p + o(p))]

    so the composed field is ``w(p) = o(p) + i(p + o(p))``. The right-hand side is
    evaluated by resampling the *coarse vector arrays* -- a few thousand numbers -- so
    chaining fields across iterations costs nothing and blurs nothing (convention 3).

    The result lives on the finer of the two grids, per axis, so composing a coarse
    field into a fine one does not throw the fine one's detail away.
    """
    if outer.shape != inner.shape:
        raise ValueError(
            f"cannot compose fields over different volumes: {outer.shape} vs {inner.shape}"
        )
    grid_shape = tuple(max(a, b) for a, b in zip(outer.grid_shape, inner.grid_shape))
    target = DeformationField.zeros(outer.shape, grid_shape)
    points = _node_points(target)  # (gz, gy, gx, 3)

    o = np.moveaxis(outer.sample(points), 0, -1)  # (gz, gy, gx, 3)
    i = np.moveaxis(inner.sample(points + o), 0, -1)
    return DeformationField(np.moveaxis(o + i, -1, 0), outer.shape)


def invert(
    field: DeformationField, *, iterations: int = 25, tol: float = 1e-2
) -> DeformationField:
    """The field ``v`` with ``compose(field, v) ~= 0``, i.e. the inverse warp.

    Needed to carry a partial reconstruction, which lives in its own deformed state,
    back into the common reference frame. Solved by the standard fixed point
    ``v(p) = -u(p + v(p))``, which converges for deformations whose Jacobian stays
    positive-definite -- true for anything the magnitude cap lets through, and the
    reason the cap exists.

    Raises :class:`ValueError` if the fixed point has not converged below ``tol``
    voxels: a field that cannot be inverted is a field that folds space onto itself,
    and silently returning the last iterate would hand the caller a plausible-looking
    volume built on a non-invertible warp. The default ``tol`` of 0.01 voxels is two
    orders of magnitude below anything that matters downstream, but loose enough that a
    large-but-still-diffeomorphic field is not rejected for converging slowly.
    """
    points = _node_points(field)
    v = -np.moveaxis(field.sample(points), 0, -1)
    residual = float("inf")
    for _ in range(iterations):
        update = -np.moveaxis(field.sample(points + v), 0, -1)
        residual = float(np.abs(update - v).max())
        v = update
        if residual < tol:
            break
    if residual >= tol:
        raise ValueError(
            f"DVF inversion did not converge: last update {residual:.3g} px after "
            f"{iterations} iterations (tol {tol:g}). The field is probably folding "
            f"(max magnitude {field.max_magnitude:.2f} px); lower max_dvf_px or "
            "coarsen the grid."
        )
    return DeformationField(np.moveaxis(v, -1, 0), field.shape)


# ---------------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------------


def _normalise_pair(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put both volumes on a common zero-mean, unit-variance scale.

    Brightness constancy is an assumption about the *difference* of the two volumes, so
    they must share one offset and one scale -- normalising them separately would
    manufacture a brightness change out of a genuine density difference. Doing it at
    all is what makes ``alpha`` a dimensionless, transferable number instead of a magic
    constant tuned to one dataset's units.
    """
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    offset = 0.5 * (float(reference.mean()) + float(moving.mean()))
    scale = float(np.sqrt(0.5 * (reference.var() + moving.var())))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "both volumes are constant; optical flow has nothing to register. "
            "Check that the partial reconstructions actually contain the object."
        )
    return (reference - offset) / scale, (moving - offset) / scale


def _downsample(volume: np.ndarray) -> np.ndarray:
    from scipy.ndimage import zoom  # noqa: PLC0415

    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    smoothed = gaussian_filter(volume, 1.0, mode="nearest")
    return zoom(smoothed, 0.5, order=1, mode="nearest").astype(np.float32)


def _upsample_flow(flow: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    from scipy.ndimage import zoom  # noqa: PLC0415

    factors = [t / s for t, s in zip(shape, flow.shape[1:])]
    out = np.empty((3, *shape), dtype=np.float32)
    for c in range(3):
        resized = zoom(flow[c], factors, order=1, mode="nearest")
        # The displacement is measured in the grid it lives on, so it scales with it.
        out[c] = resized[: shape[0], : shape[1], : shape[2]] * factors[c]
    return out


def _warp_dense(volume: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Pull-back warp with a dense field -- the inner loop of the flow solver."""
    from scipy.ndimage import map_coordinates  # noqa: PLC0415

    grid = np.stack(
        np.meshgrid(*[np.arange(s, dtype=np.float32) for s in volume.shape], indexing="ij"), axis=0
    )
    np.add(grid, flow, out=grid)
    return map_coordinates(volume, grid, order=1, mode="nearest").astype(np.float32)


def _horn_schunck_level(
    reference: np.ndarray,
    moving: np.ndarray,
    flow: np.ndarray,
    *,
    alpha: float,
    iterations: int,
    warps: int,
) -> np.ndarray:
    """Smoothed-Jacobi Horn-Schunck at one pyramid level, with re-linearisation.

    Minimises ``sum (grad J . w + b)^2 + alpha^2 ||grad w||^2`` where ``J`` is the
    moving volume warped by the current flow. Classic Horn-Schunck averages the flow
    with a centre-excluding kernel; a Gaussian is used instead, which makes the update
    a damped Jacobi sweep with slightly stronger smoothing per iteration and converges
    in fewer of them. ``warps`` outer re-warps handle displacements larger than the
    linearisation is good for, which is also what the pyramid is for.
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    # Every product below whose left operand is a live, reused local goes through
    # an explicit ufunc call (np.multiply / np.square), never the * operator.
    # On CPython 3.14.0 + NumPy 2.2.6 the temporary-elision optimisation writes
    # the result of ``a * b`` INTO ``a`` when ``a`` is a function-local array of
    # refcount 1 above the elision threshold — here that corrupted ``gradient``
    # in-place, the Horn-Schunck iteration overflowed, and estimate_flow
    # returned all-NaN (NaN fraction 1.00 vs 0.00 on CPython 3.11/NumPy 1.26,
    # same code, same input). Ufunc calls do not take the elision path. The same
    # defect was found and fixed in this project's polish_stack.py; the class of
    # bug disappears on NumPy >= 2.3, but this module must not depend on that.
    for _ in range(max(1, warps)):
        warped = _warp_dense(moving, flow)
        gradient = np.stack(np.gradient(warped)).astype(np.float32)
        it = (warped - reference).astype(np.float32)
        b = it - np.multiply(gradient, flow).sum(axis=0)
        denominator = alpha**2 + np.square(gradient).sum(axis=0)
        for _ in range(max(1, iterations)):
            averaged = np.stack(
                [gaussian_filter(flow[c], 1.0, mode="nearest") for c in range(3)]
            )
            common = (np.multiply(gradient, averaged).sum(axis=0) + b) / denominator
            flow = averaged - np.multiply(gradient, common)
    return flow.astype(np.float32)


def _horn_schunck(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    alpha: float,
    levels: int,
    iterations: int,
    warps: int,
) -> np.ndarray:
    """Pyramidal 3D Horn-Schunck. Returns a dense ``(3, nz, ny, nx)`` field."""
    pyramid_ref = [reference]
    pyramid_mov = [moving]
    for _ in range(max(0, levels - 1)):
        if min(pyramid_ref[-1].shape) < 8:
            break
        pyramid_ref.append(_downsample(pyramid_ref[-1]))
        pyramid_mov.append(_downsample(pyramid_mov[-1]))

    flow = np.zeros((3, *pyramid_ref[-1].shape), dtype=np.float32)
    for level in range(len(pyramid_ref) - 1, -1, -1):
        if flow.shape[1:] != pyramid_ref[level].shape:
            flow = _upsample_flow(flow, pyramid_ref[level].shape)
        flow = _horn_schunck_level(
            pyramid_ref[level],
            pyramid_mov[level],
            flow,
            alpha=alpha,
            iterations=iterations,
            warps=warps,
        )
    return flow


def _tvl1(reference: np.ndarray, moving: np.ndarray, *, attachment: float) -> np.ndarray:
    try:
        from skimage.registration import optical_flow_tvl1  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "flow_method='tvl1' needs scikit-image (skimage.registration."
            "optical_flow_tvl1). Install it, or use the default "
            "flow_method='horn_schunck', which is scipy-only."
        ) from exc

    # skimage's convention matches ours: the returned flow warps `moving` onto
    # `reference` as a pull-back. Verified by its own docstring example.
    return np.stack(optical_flow_tvl1(reference, moving, attachment=attachment)).astype(np.float32)


def estimate_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    spacing: float = 16.0,
    grid_shape: Sequence[int] | None = None,
    method: str = "horn_schunck",
    alpha: float = 1.0,
    levels: int = 3,
    iterations: int = 40,
    warps: int = 3,
    prefilter_sigma: float = 0.0,
    max_px: float | None = None,
) -> DeformationField:
    """3D optical flow: the field that warps ``moving`` onto ``reference``.

    Returns a :class:`DeformationField` on a coarse grid such that
    ``warp_volume(moving, field) ~= reference`` (convention 2). ``reference`` is the
    fixed volume; ``moving`` is the one that gets deformed.

    ``alpha`` is the smoothness weight of the Horn-Schunck data/smoothness trade-off
    and is dimensionless because both volumes are normalised to unit variance first
    (see :func:`_normalise_pair`). Larger ``alpha`` -> smoother, more conservative
    flow. It is a parameter, deliberately, because its right value depends on how
    noisy the partial reconstructions are, which depends on how many angles went into
    each subset.

    ``method``:

    * ``"horn_schunck"`` (default) -- scipy only, pyramidal, no optional dependency.
    * ``"tvl1"`` -- ``skimage.registration.optical_flow_tvl1``. Better at
      discontinuous deformation, but an **optional** import: a clear ImportError names
      the fallback rather than the method silently changing under the caller.

    ``prefilter_sigma`` low-passes both volumes before the flow is estimated, and it is
    worth more than it looks. The deformation being sought is smooth by construction --
    it lives on a coarse grid -- so nothing about it is carried by the finest scales,
    while the *differences* between subset reconstructions at those scales are almost
    entirely streak artefacts from their different angular sampling. Flow estimated
    against those streaks is the main source of deformation invented from rigid data.
    Set it to roughly half the coarse-grid spacing when the partials are streaky.

    The dense solver output is then projected onto the coarse grid, which is where most
    of the regularisation happens; ``spacing`` (voxels) is that knob, and ``max_px``
    caps the magnitude afterwards.
    """
    if method not in ("horn_schunck", "tvl1"):
        raise ValueError(f"unknown flow_method {method!r}; expected 'horn_schunck' or 'tvl1'")
    reference = np.asarray(reference, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    if reference.shape != moving.shape:
        raise ValueError(
            f"reference {reference.shape} and moving {moving.shape} must have the same shape"
        )
    if reference.ndim != 3:
        raise ValueError(f"estimate_flow works on 3D volumes, got {reference.ndim}D")

    reference_n, moving_n = _normalise_pair(reference, moving)
    if prefilter_sigma > 0:
        from scipy.ndimage import gaussian_filter  # noqa: PLC0415

        reference_n = gaussian_filter(reference_n, prefilter_sigma, mode="nearest")
        moving_n = gaussian_filter(moving_n, prefilter_sigma, mode="nearest")

    if method == "horn_schunck":
        dense = _horn_schunck(
            reference_n, moving_n, alpha=alpha, levels=levels, iterations=iterations, warps=warps
        )
    else:
        dense = _tvl1(reference_n, moving_n, attachment=1.0 / max(alpha, 1e-6))

    field = DeformationField.from_dense(dense, spacing=spacing, grid_shape=grid_shape)
    if max_px is not None:
        field = field.clipped(max_px)
    return field


# ---------------------------------------------------------------------------------
# Time sequences of fields
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeformationSequence:
    """One :class:`DeformationField` per time subset, plus the times they belong to.

    ``times`` are in whatever units the caller's acquisition clock uses (a scan index
    is fine); they only need to be monotonically increasing and on the same scale as
    the times passed to :meth:`at`. All fields must share the same volume shape and
    the same coarse grid, so the whole sequence is one ``(K, 3, gz, gy, gx)`` array
    and smoothing across time is a single filter call.

    **This is the regulariser that does the real work.** K fields stand in for N
    projections, and every projection's deformation is a smooth interpolation of its
    neighbours in *time*. Without it each projection would carry its own field, which
    a single projection cannot possibly determine.
    """

    fields: tuple[DeformationField, ...]
    times: np.ndarray

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("a DeformationSequence needs at least one field")
        shape = fields[0].shape
        grid = fields[0].grid_shape
        for f in fields[1:]:
            if f.shape != shape or f.grid_shape != grid:
                raise ValueError(
                    "every field in a sequence must share the volume shape and coarse "
                    f"grid: {shape}/{grid} vs {f.shape}/{f.grid_shape}"
                )
        times = np.asarray(self.times, dtype=np.float64)
        if times.shape != (len(fields),):
            raise ValueError(f"times must have one entry per field, got {times.shape}")
        if np.any(np.diff(times) < 0):
            raise ValueError("times must be non-decreasing (sort the subsets by time first)")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "times", times)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.fields[0].shape

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.fields[0].grid_shape

    @property
    def node_array(self) -> np.ndarray:
        """All fields as one ``(K, 3, gz, gy, gx)`` array."""
        return np.stack([f.vectors for f in self.fields])

    def _rebuilt(self, nodes: np.ndarray) -> "DeformationSequence":
        shape = self.shape
        return DeformationSequence(
            tuple(DeformationField(nodes[k], shape) for k in range(nodes.shape[0])), self.times
        )

    def smoothed_in_time(self, sigma: float) -> "DeformationSequence":
        """Gaussian-smooth each node's trajectory across subsets (sigma in subsets).

        Deformation caused by dose, drift or creep is slow compared with one subset, so
        a field that jumps between neighbouring subsets is fitting noise. ``mode
        ="nearest"`` rather than reflect: the deformation at the ends of the scan is
        extrapolated flat, not mirrored, because mirroring would invent a reversal.
        """
        if sigma <= 0 or len(self.fields) < 2:
            return self
        from scipy.ndimage import gaussian_filter1d  # noqa: PLC0415

        return self._rebuilt(gaussian_filter1d(self.node_array, sigma, axis=0, mode="nearest"))

    def zero_mean(self) -> "DeformationSequence":
        """Remove the time-averaged deformation -- a gauge fix, not a cosmetic step.

        A deformation common to *every* projection is indistinguishable from having
        reconstructed a differently-shaped reference volume: the data cannot tell them
        apart, so the common mode is unobservable and, left alone, random-walks from
        iteration to iteration. Exactly the reason the rigid engine subtracts the mean
        from ``sx``/``sy``. Fixing the gauge here is also what makes a recovered
        sequence comparable with a ground-truth one at all.
        """
        nodes = self.node_array
        return self._rebuilt(nodes - nodes.mean(axis=0, keepdims=True))

    def clipped(self, max_px: float) -> "DeformationSequence":
        return DeformationSequence(tuple(f.clipped(max_px) for f in self.fields), self.times)

    def at(self, time: float) -> DeformationField:
        """The field at an arbitrary acquisition time, by linear interpolation.

        Linear rather than spline: the sequence has already been Gaussian-smoothed in
        time, and a spline through smoothed knots can overshoot into deformation that
        no subset ever showed. Times outside the range are clamped to the end fields --
        extrapolating a deformation trend past the data is exactly how a non-rigid
        model invents structure at the start and end of a scan.
        """
        times = self.times
        if len(self.fields) == 1:
            return self.fields[0]
        t = float(np.clip(time, times[0], times[-1]))
        j = int(np.searchsorted(times, t, side="right"))
        j = max(1, min(j, len(times) - 1))
        span = times[j] - times[j - 1]
        w = 0.0 if span <= 0 else (t - times[j - 1]) / span
        vectors = (1.0 - w) * self.fields[j - 1].vectors + w * self.fields[j].vectors
        return DeformationField(vectors.astype(np.float32), self.shape)

    @property
    def rms_magnitude(self) -> float:
        return float(np.sqrt(np.mean(np.sum(self.node_array.astype(np.float64) ** 2, axis=1))))

    @property
    def max_magnitude(self) -> float:
        return float(np.sqrt(np.sum(self.node_array.astype(np.float64) ** 2, axis=1)).max())

    @classmethod
    def zeros_like(cls, other: "DeformationSequence") -> "DeformationSequence":
        return cls(
            tuple(DeformationField.zeros(other.shape, other.grid_shape) for _ in other.fields),
            other.times,
        )


def sequence_rms_difference(
    a: DeformationSequence, b: DeformationSequence, mask: np.ndarray | None = None
) -> float:
    """RMS difference between two sequences, in voxels, at the coarse nodes.

    Both are gauge-fixed with :meth:`DeformationSequence.zero_mean` first, because the
    time-averaged deformation is unobservable (see that method) and comparing without
    removing it measures a quantity neither sequence can be blamed for.

    ``mask`` is a boolean array over the coarse grid; pass the object support to get
    the number that means something. Flow in empty air is determined by the smoothness
    term alone -- the data says nothing there -- so a whole-grid RMS mostly measures
    how far the regulariser extrapolated, not how well the deformation was recovered.
    """
    if a.grid_shape != b.grid_shape or len(a.fields) != len(b.fields):
        raise ValueError("sequences must share a grid and a length to be compared")
    difference = a.zero_mean().node_array - b.zero_mean().node_array  # (K, 3, gz, gy, gx)
    squared = (difference.astype(np.float64) ** 2).sum(axis=1)  # (K, gz, gy, gx)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.grid_shape:
            raise ValueError(f"mask shape {mask.shape} != coarse grid {a.grid_shape}")
        if not mask.any():
            raise ValueError("mask selects no nodes")
        squared = squared[:, mask]
    return float(np.sqrt(squared.mean()))


def coarse_support_mask(
    volume: np.ndarray, grid_shape: Sequence[int], *, threshold: float = 0.05, sigma: float = 3.0
) -> np.ndarray:
    """Boolean mask over the coarse grid selecting nodes that sit inside the object.

    Optical flow in empty space is determined entirely by the smoothness term -- the
    data says nothing there -- so any error statistic computed over the whole grid is
    dominated by how far the regulariser extrapolated, not by how well the deformation
    was recovered. Every honest comparison against a known field should be restricted
    to this mask (and the whole-grid number quoted alongside it, not instead of it).

    ``threshold`` is a fraction of the smoothed volume's peak magnitude; ``sigma`` is
    the smoothing applied first, so an isolated bright voxel does not open the mask.
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    volume = np.asarray(volume, dtype=np.float32)
    support = gaussian_filter(np.abs(volume), sigma, mode="nearest")
    peak = float(support.max())
    if peak <= 0:
        raise ValueError("the volume is empty; there is no object support to mask with")
    field = DeformationField.from_dense(
        np.stack([support, support, support]), grid_shape=grid_shape, smooth=False
    )
    return field.vectors[0] > threshold * peak
