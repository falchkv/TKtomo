"""Synthetic phantom data so UIs and tests run without real acquisitions.

Builds a genuine 3D ellipsoid phantom (numpy only) and forward-projects each
z-slice with scikit-image's ``radon`` (no reconstruction backend required), so
scrolling through detector rows in the sinogram viewer shows structurally
different slices. Optionally injects random per-projection shifts to emulate the
misalignment the alignment UI must correct.
"""

from __future__ import annotations

import numpy as np

from tktomo.io.data import ProjectionData

# Ellipsoids defining the synthetic 3D phantom, in normalized ``[-1, 1]``
# coordinates: ``(value, cx, cy, cz, ax, ay, az)``. Several sit at different
# ``cz`` so each z-slice is a distinct cross-section. In-plane semi-axes are
# kept <= ~0.85 so ``radon(circle=True)`` does not clip / warn.
_ELLIPSOIDS = [
    (0.20, 0.00, 0.00, 0.00, 0.80, 0.80, 0.95),  # outer body, all slices
    (0.30, -0.30, 0.10, -0.55, 0.25, 0.30, 0.30),  # low-z blob
    (-0.20, 0.35, -0.10, 0.00, 0.20, 0.20, 0.35),  # mid, subtractive
    (0.40, 0.00, 0.40, 0.45, 0.30, 0.15, 0.30),  # high-z
    (0.35, -0.20, -0.35, 0.60, 0.18, 0.18, 0.25),  # top small
    (-0.15, 0.10, 0.10, -0.20, 0.12, 0.12, 0.15),  # tiny core
]


def _import_radon():
    try:
        from skimage.transform import radon  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Phantom generation requires scikit-image. Install with: pip install scikit-image"
        ) from exc
    return radon


def generate_volume(size: int = 64, n_slices: int = 32) -> np.ndarray:
    """Generate a genuine 3D ellipsoid phantom ``(n_slices, size, size)``.

    Ellipsoids are placed at varying depth, so each z-slice is a structurally
    different cross-section (unlike scaled copies of one image). Useful for
    exercising the tomogram viewer / reprojection without running a
    reconstruction backend. Pure numpy; no scikit-image required.
    """
    ys, xs = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]  # (size, size)
    zs = np.array([0.0]) if n_slices == 1 else np.linspace(-0.8, 0.8, n_slices)
    vol = np.zeros((n_slices, size, size), dtype=np.float64)
    for value, cx, cy, cz, ax, ay, az in _ELLIPSOIDS:
        d = (
            ((xs[None] - cx) / ax) ** 2
            + ((ys[None] - cy) / ay) ** 2
            + ((zs[:, None, None] - cz) / az) ** 2
        )  # (n_slices, size, size)
        vol[d <= 1.0] += value
    np.clip(vol, 0.0, None, out=vol)
    return vol


def generate_phantom(
    n_angles: int = 90,
    size: int = 64,
    n_slices: int = 32,
    *,
    max_shift: float = 0.0,
    seed: int | None = 0,
) -> ProjectionData:
    """Generate a synthetic projection stack from a genuine 3D phantom.

    Builds a 3D ellipsoid volume (see :func:`generate_volume`) and forward-
    projects each z-slice, so each detector row's sinogram is structurally
    distinct. Cost is ``O(n_slices * n_angles * size**2)`` — linear in
    ``n_slices``.

    Parameters
    ----------
    n_angles:
        Number of projection angles, evenly spaced over ``[0, pi)``.
    size:
        In-plane size (pixels) of each square slice / projection width.
    n_slices:
        Number of detector rows (distinct z-slices of the 3D phantom).
    max_shift:
        If > 0, apply a random (dy, dx) shift up to this many pixels to each
        projection, to emulate misalignment for testing the alignment tool.
    seed:
        RNG seed for reproducible shifts.
    """
    radon = _import_radon()
    rng = np.random.default_rng(seed)

    volume = generate_volume(size=size, n_slices=n_slices)  # (n_slices, size, size)
    angles_deg = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    projections = np.zeros((n_angles, n_slices, size), dtype=np.float64)
    for s in range(n_slices):
        sino = radon(volume[s], theta=angles_deg, circle=True)  # (size, n_angles)
        projections[:, s, :] = sino.T

    proj = ProjectionData(
        data=projections,
        angles=np.deg2rad(angles_deg),
        metadata={"source": "synthetic_ellipsoid_3d", "size": size, "n_slices": n_slices},
    )

    if max_shift > 0:
        proj = _apply_random_shifts(proj, max_shift, rng)
    return proj


def _apply_random_shifts(
    proj: ProjectionData, max_shift: float, rng: np.random.Generator
) -> ProjectionData:
    from scipy.ndimage import shift as ndi_shift  # noqa: PLC0415

    shifted = np.empty_like(proj.data)
    shifts = rng.uniform(-max_shift, max_shift, size=(proj.n_angles, 2))
    for i in range(proj.n_angles):
        shifted[i] = ndi_shift(proj.data[i], shifts[i], order=1, mode="nearest")
    meta = dict(proj.metadata)
    meta["applied_shifts"] = shifts
    return ProjectionData(data=shifted, angles=proj.angles, metadata=meta)
