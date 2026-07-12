"""Generate a synthetic misaligned dataset with known ground-truth shifts.

This is what ``tests/test_ptycho_engine.py`` aligns, and what the README quickstart
opens in the GUI.

The misalignment is applied with ``scipy.ndimage.fourier_shift`` -- deliberately a
*different* implementation from the engine's ``apply_shifts`` -- so recovering the
truth proves the loop actually works rather than merely agreeing with itself.

Why Fourier and not ``ndimage.shift``: a spline shift of a shepp3d projection is not
an exact translation. The phantom has hard edges, and cubic interpolation of a hard
edge distorts it by ~0.1 px worth of apparent displacement. That is enough to swamp
the 0.1 px tolerance the engine is judged against, and it is a defect of the *test
article*, not of the algorithm -- measured: the identical loop scores 0.128 px on
spline-shifted data and 0.023 px on Fourier-shifted data. A Fourier shift is an exact
translation of a band-limited periodic signal, so the ground truth is really the
truth. (The wrap-around it implies is harmless because ``margin`` leaves a zero
border.)

    python examples/make_phantom.py --output phantom.h5

Ground-truth shifts are written alongside as ``<output>_truth.npz`` (HDF5 attrs only
take scalars, so the arrays cannot ride along in the projection file).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_phantom_volume(size: int = 64) -> np.ndarray:
    """TomoPy's shepp3d plus off-centre asymmetric blobs.

    The blobs matter: a perfectly centred, symmetric phantom has a near-zero COM
    sinusoid, so the sinusoid fit in ``core/com.py`` would be trivially satisfied and
    would not actually be tested.
    """
    import tomopy

    volume = np.asarray(tomopy.shepp3d(size=size), dtype=np.float32)

    z, y, x = np.mgrid[0:size, 0:size, 0:size].astype(np.float32)
    for (cz, cy, cx), radius, amplitude in (
        ((0.55, 0.30, 0.35), 0.10, 0.8),
        ((0.40, 0.70, 0.62), 0.07, 1.0),
        ((0.62, 0.58, 0.28), 0.05, 0.6),
    ):
        blob = (
            (z - cz * size) ** 2 + (y - cy * size) ** 2 + (x - cx * size) ** 2
        ) < (radius * size) ** 2
        volume[blob] += amplitude

    return volume


def make_misaligned_dataset(
    *,
    size: int = 64,
    n_angles: int = 90,
    max_shift: float = 3.0,
    margin: int = 8,
    drift: float = 0.0,
    ramp: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
):
    """Forward-project the phantom, then corrupt it with known shifts.

    Returns ``(ProjectionData, sx_true, sy_true, volume)``. The shifts are the
    misalignment that was *applied*; the engine's job is to recover them.

    ``margin`` zero-pads the clean projections *before* the shift is applied, and it
    is not optional. A shepp3d projection fills the frame edge to edge, so shifting
    it in place would push real mass off one side and pull zeros in at the other --
    translation plus truncation. The corrupted data would then not be a shifted copy
    of any consistent object, and no alignment algorithm could recover the truth. The
    margin must exceed ``max_shift`` (plus ``drift``).
    """
    import tomopy
    from scipy.ndimage import fourier_shift

    from tktomo.io import ProjectionData

    volume = make_phantom_volume(size)
    angles = tomopy.angles(n_angles)  # radians, 0..pi
    clean = np.asarray(tomopy.project(volume, angles, pad=False), dtype=np.float32)

    required = max_shift + abs(drift)
    if margin <= required:
        raise ValueError(
            f"margin={margin} px must exceed max_shift+drift={required:g} px, or the "
            "shift truncates the object and the ground truth becomes unrecoverable."
        )
    clean = np.pad(clean, ((0, 0), (margin, margin), (margin, margin)), mode="constant")

    rng = np.random.default_rng(seed)
    sx_true = rng.uniform(-max_shift, max_shift, n_angles)
    sy_true = rng.uniform(-max_shift, max_shift, n_angles)

    if drift:
        # A slow linear walk across the scan, on top of the random jitter -- the
        # thing a per-projection registration must not mistake for real structure.
        ramp_line = np.linspace(0.0, drift, n_angles)
        sx_true = sx_true + ramp_line
        sy_true = sy_true + 0.5 * ramp_line

    projections = np.empty_like(clean)
    for i in range(n_angles):
        spectrum = fourier_shift(np.fft.fftn(clean[i]), (sy_true[i], sx_true[i]))
        projections[i] = np.fft.ifftn(spectrum).real

    if ramp:
        n_v, n_u = projections.shape[1:]
        v, u = np.mgrid[0:n_v, 0:n_u].astype(np.float32)
        for i in range(n_angles):
            a, b = rng.uniform(-ramp, ramp, 2)
            projections[i] += a * u / n_u + b * v / n_v

    if noise:
        projections += rng.normal(0.0, noise * clean.std(), projections.shape).astype(
            np.float32
        )

    data = ProjectionData(
        data=projections,
        angles=angles,
        metadata={"name": "phantom", "pixel_size_nm": 10.0},
    )
    return data, sx_true, sy_true, volume


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="phantom.h5", help="HDF5 file to write")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--angles", type=int, default=90)
    parser.add_argument("--max-shift", type=float, default=3.0)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--ramp", type=float, default=0.0, help="linear phase ramp amplitude")
    parser.add_argument("--noise", type=float, default=0.0, help="noise sigma, x data std")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from tktomo.io import save_projections

    data, sx_true, sy_true, _volume = make_misaligned_dataset(
        size=args.size,
        n_angles=args.angles,
        max_shift=args.max_shift,
        drift=args.drift,
        ramp=args.ramp,
        noise=args.noise,
        seed=args.seed,
    )

    output = Path(args.output)
    save_projections(output, data)

    truth = output.with_name(f"{output.stem}_truth.npz")
    np.savez(truth, sx_true=sx_true, sy_true=sy_true, angles=data.angles)

    print(f"Wrote {data.data.shape} projections to {output}")
    print(f"Ground-truth shifts -> {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
