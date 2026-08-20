"""Ground-truth phantoms for benchmarking tomographic alignment.

An aligner cannot be benchmarked by FSC alone. A rigid-but-*wrong* geometry applied
identically to both half-sets gives a deceptively good FSC -- the cross term
``F_a conj(F_b)`` picks up ``exp(+i phi) exp(-i phi) = 1`` from a common translation,
so the correlation is *exactly* invariant to it (see
:func:`benchmarks.metrics.fourier_shell_correlation` and the test that pins it).
Measured on a phantom here: the half-bit FRC read 508.6 nm at centring errors of 0,
4, 8, 16, 32 and 64 px while the true edge blur grew to 128 px. So the primary
metric has to be **shift recovery against known injected truth**, which means the
truth has to be injected by this module.

Two generators share one interface and return a :class:`BenchmarkCase`:

* :func:`synthetic_case` -- self-contained ellipsoid phantom, numpy+scipy only, so
  the repo's own tests and any outside user can run the benchmark with no data.
* :func:`volume_case` / :func:`load_volume` -- forward-projects a **user-supplied**
  reconstructed volume at user-supplied angles, giving real sample statistics with
  exact ground truth. The volume path is always a parameter; no dataset path is
  hard-coded and no measured data is shipped with the repo.

Three conventions, each of which is a bug if you get it wrong:

1. **Truth sign.** ``truth.dy`` / ``truth.dx`` are the *content displacement* that
   was injected: projection ``i`` was moved down by ``dy[i]`` rows and right by
   ``dx[i]`` columns. TKtomo's
   :func:`~tktomo.ptycho_align.core.engine.apply_shifts` moves content by ``-s``, so
   an aligner that works reports ``sy = +dy`` and ``sx = +dx`` -- the same sign as
   the truth, not its negative. This matches ``examples/make_phantom.py`` and
   ``tests/test_ptycho_engine.py``. Flip it and a *perfect* aligner scores twice the
   injection RMS instead of zero; the tell is a recovery RMS of almost exactly
   ``2x`` the injected RMS.

2. **Injection order.** Geometry (axis tilt, angle readback error) is baked into the
   forward projection; then magnification, then the non-rigid deformation, then the
   rigid shift, then the phase ramp/offset, then FOV truncation, then noise. The
   rigid shift goes on *after* every other geometric distortion so the recorded
   ``dy``/``dx`` are exactly what was applied -- if a deformation were applied on top
   of the shift, the recorded truth would be a lie by a fraction of a pixel and no
   aligner could ever score zero.

3. **Shift by Fourier, pad first.** The rigid jitter is injected with
   ``scipy.ndimage.fourier_shift``, deliberately a *different* implementation from
   the one any aligner uses to correct it, so recovering the truth proves the
   algorithm works rather than that it agrees with itself. A Fourier shift is an
   exact translation of a band-limited periodic signal (a 5th-order spline shift of
   a hard edge is not -- it costs ~0.1 px of apparent displacement), but it wraps
   around, so :func:`perturb` refuses to run unless ``margin`` exceeds the largest
   shift it is about to apply.

scipy is imported lazily inside functions; importing this module costs numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "P06_LENS1_SPEC",
    "BenchmarkCase",
    "GroundTruth",
    "PerturbationSpec",
    "back_project",
    "cases_from_catalogue",
    "circular_mask",
    "forward_project",
    "load_angles",
    "load_volume",
    "perturb",
    "synthetic_case",
    "synthetic_volume",
    "volume_case",
]


# --------------------------------------------------------------------------------
# Geometry: a parallel-beam projector in numpy + scipy.ndimage only
# --------------------------------------------------------------------------------
#
# Array conventions, fixed once here and used by every function in the package:
#
#   volume      (n_z, n_y, n_x)   n_z indexes detector ROWS (the rotation axis),
#                                 n_y is along the beam, n_x across it.
#   projections (n_theta, n_v, n_u) with n_v == n_z and n_u == n_x.
#
# The rotation axis sits at in-plane index ``(n_x - 1) / 2`` and projects to detector
# column ``(n_u - 1) / 2``. That "-1" is not a typo and not tomopy's default of
# ``width / 2``: ``scipy.ndimage.rotate`` rotates about the array's geometric centre,
# which for an N-sample axis is ``(N - 1) / 2``. Handing an aligner ``width / 2``
# instead costs a constant half-pixel of ``dx``, which is pure gauge (see
# :func:`benchmarks.metrics.remove_gauge`) and so does not corrupt a score -- but it
# does blur the reconstruction, so :func:`benchmarks.runner.default_center` passes
# the honest value.


#: Largest number of non-zeros the sparse projection matrix may hold before
#: :func:`projection_matrix` refuses to build it. 40e6 entries is about 320 MB as CSR
#: (float32 data + int32 indices).
_MAX_MATRIX_ENTRIES = 40_000_000


def projection_matrix(n_u: int, angles: np.ndarray, *, cache: dict | None = None):
    """The parallel-beam system matrix ``M`` for one slice geometry, as CSR.

    Rows are ``(angle, detector column)``, columns are ``(y, x)`` voxels of one square
    slice; the geometry is slice-independent, so the *same* matrix projects every
    detector row of the volume.

    **Why a matrix at all, when ``scipy.ndimage.rotate`` is right there.** Because
    SIRT needs a true adjoint pair and rotate-then-sum does not have one. The transpose
    of an interpolating *gather* is a *scatter*; ``rotate(-theta)`` is another gather,
    which is close to the inverse but not the transpose. With that pair ``A^T A`` is
    not positive semi-definite, some eigenvalues go negative, and ``x <- x + s A^T(...)``
    grows without bound along those modes -- and no relaxation factor can fix it,
    because ``|1 - s*lambda| > 1`` for every ``s > 0`` when ``lambda < 0``.

    That failure is nasty because it is *data-dependent*: divergence only shows once
    the data has a component along the bad mode. Measured here at 92 px, 60 angles,
    with the identical geometry: a 4-slice phantom converged to a relative residual of
    0.059 by iteration 20 while a 6-slice one blew up to 26.8, and the measured
    spectral radius was 0.9986 in both cases -- because power iteration reports
    ``|lambda_max|`` and says nothing about its sign.

    With ``M`` built explicitly, ``forward = M @ x`` and ``back = M.T @ y`` are exactly
    adjoint by construction, ``M^T M`` is PSD, and SIRT is unconditionally stable
    again. It is also *faster* than the rotate-based pair at these sizes, because a
    sparse matvec beats ``n_angles`` spline resamplings.

    Rays are sampled at ``n_u`` unit-spaced points, matching the rotate-based
    projector's sum over one array axis, so the two agree in scale as well as in
    handedness (pinned by ``tests/test_benchmark.py``).
    """
    key = (int(n_u), angles.tobytes())
    if cache is not None and key in cache:
        return cache[key]

    from scipy.sparse import csr_matrix  # noqa: PLC0415

    n_theta = int(angles.size)
    entries = n_theta * n_u * n_u * 4
    if entries > _MAX_MATRIX_ENTRIES:
        raise MemoryError(
            f"A {n_theta} x {n_u} x {n_u} projection matrix needs ~{entries / 1e6:.0f}M "
            f"non-zeros (~{entries * 8 / 1e9:.1f} GB), over the "
            f"{_MAX_MATRIX_ENTRIES / 1e6:.0f}M limit. This projector is for benchmark-"
            "sized problems: reduce the volume with load_volume(bin_factor=...) or "
            "slices=..., or use a real reconstruction backend (tomopy, astra) for a "
            "volume this large."
        )

    centre = (n_u - 1) / 2.0
    # Detector coordinate s and along-ray coordinate t, both centred on the axis.
    s = np.arange(n_u, dtype=np.float64) - centre
    t = np.arange(n_u, dtype=np.float64) - centre
    # Handedness chosen to match `_rotate(volume, +theta, axes=(1, 2))` followed by a
    # sum over axis 1; `test_projectors_agree` is what actually pins it.
    cos = np.cos(angles)[:, None, None]
    sin = np.sin(angles)[:, None, None]
    ss = s[None, :, None]
    tt = t[None, None, :]
    x = centre + ss * cos - tt * sin  # (n_theta, n_u, n_u)
    y = centre + ss * sin + tt * cos

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0

    rows = np.repeat(
        (np.arange(n_theta)[:, None] * n_u + np.arange(n_u)[None, :]).ravel(), n_u
    ).reshape(n_theta, n_u, n_u)

    row_list, col_list, val_list = [], [], []
    for dy in (0, 1):
        for dx in (0, 1):
            yy = y0 + dy
            xx = x0 + dx
            weight = (fy if dy else 1.0 - fy) * (fx if dx else 1.0 - fx)
            inside = (yy >= 0) & (yy < n_u) & (xx >= 0) & (xx < n_u) & (weight > 0)
            row_list.append(rows[inside])
            col_list.append((yy[inside] * n_u + xx[inside]))
            val_list.append(weight[inside])

    matrix = csr_matrix(
        (
            np.concatenate(val_list).astype(np.float32),
            (np.concatenate(row_list), np.concatenate(col_list)),
        ),
        shape=(n_theta * n_u, n_u * n_u),
    )
    if cache is not None:
        cache.clear()  # one geometry at a time; an alignment loop reuses one
        cache[key] = matrix
    return matrix


_MATRIX_CACHE: dict = {}


def _rotate(array: np.ndarray, angle_deg: float, axes: tuple[int, int], order: int) -> np.ndarray:
    from scipy.ndimage import rotate as ndi_rotate  # noqa: PLC0415

    if angle_deg == 0.0:
        return array
    return ndi_rotate(
        array,
        angle_deg,
        axes=axes,
        reshape=False,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )


def circular_mask(n_y: int, n_x: int, *, radius_fraction: float = 1.0) -> np.ndarray:
    """The inscribed disc of an ``(n_y, n_x)`` slice.

    ``rotate(..., reshape=False)`` clips the corners of a square array, so anything
    outside the inscribed circle is destroyed by rotation and the forward projector
    stops being a linear operator on it. Masking the volume to the disc *before*
    projecting makes forward and back projection mutually consistent, which is what
    lets SIRT converge instead of chasing corner artifacts.
    """
    ys, xs = np.ogrid[:n_y, :n_x]
    dy = ys - (n_y - 1) / 2.0
    dx = xs - (n_x - 1) / 2.0
    radius = radius_fraction * min(n_y, n_x) / 2.0
    return (dy * dy + dx * dx) <= radius * radius


def forward_project(
    volume: np.ndarray,
    angles: np.ndarray,
    *,
    center: float | None = None,
    axis_tilt_deg: float = 0.0,
    out_of_plane_tilt_deg: float = 0.0,
    order: int = 1,
) -> np.ndarray:
    """Parallel-beam forward projection of ``(n_z, n_y, n_x)`` at ``angles`` (radians).

    Returns ``(n_theta, n_z, n_x)``.

    ``axis_tilt_deg`` tilts the rotation axis *in the detector plane* (a rotation
    about the beam). Because that rotation does not mix the beam coordinate, it
    commutes with the line integral and is applied to the finished projection --
    which is both exact and ``n_theta`` rotations cheaper than doing it in 3D.

    ``out_of_plane_tilt_deg`` tips the axis *toward the beam*. That one does mix the
    beam coordinate, so it has to be applied to the rotated volume before the
    integral: it is a genuine 3D geometry error, not a per-projection 2D transform,
    which is exactly why no per-projection rigid aligner can remove it. Enabling it
    triples the cost of the projector.

    With no out-of-plane tilt and a square slice this runs through
    :func:`projection_matrix`, which is both faster and -- crucially -- exactly
    adjoint to :func:`back_project`. The rotate-based path is kept for the tilted and
    non-square cases, which only ever *generate* data and never feed a reconstruction.
    """
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D (n_z, n_y, n_x); got shape {volume.shape}")
    angles = np.asarray(angles, dtype=np.float64)

    n_z, n_y, n_x = volume.shape
    if not out_of_plane_tilt_deg and n_y == n_x:
        matrix = projection_matrix(n_x, angles, cache=_MATRIX_CACHE)
        flat = matrix @ volume.reshape(n_z, n_x * n_x).T  # (n_theta * n_u, n_z)
        out = np.ascontiguousarray(
            flat.reshape(angles.size, n_x, n_z).transpose(0, 2, 1), dtype=np.float32
        )
        if axis_tilt_deg:
            out = np.stack([_rotate(frame, axis_tilt_deg, axes=(0, 1), order=order) for frame in out])
    else:
        out = np.empty((angles.size, n_z, n_x), dtype=np.float32)
        for i, theta in enumerate(angles):
            rotated = _rotate(volume, math.degrees(theta), axes=(1, 2), order=order)
            if out_of_plane_tilt_deg:
                rotated = _rotate(rotated, out_of_plane_tilt_deg, axes=(0, 1), order=order)
            frame = rotated.sum(axis=1)
            if axis_tilt_deg:
                frame = _rotate(frame, axis_tilt_deg, axes=(0, 1), order=order)
            out[i] = frame

    if center is not None:
        out = _recenter(out, from_center=(n_x - 1) / 2.0, to_center=float(center))
    return out


def back_project(
    projections: np.ndarray,
    angles: np.ndarray,
    *,
    center: float | None = None,
    order: int = 1,
) -> np.ndarray:
    """**Exact** adjoint of :func:`forward_project`: ``(n_theta, n_z, n_u)`` -> ``(n_z, n_u, n_u)``.

    Unfiltered backprojection, computed as ``M.T @ y`` with the same ``M`` the forward
    projector uses -- so ``<A x, y> == <x, A^T y>`` to floating-point rounding, and
    ``A^T A`` is positive semi-definite. That is what makes SIRT converge rather than
    quietly explode; :func:`projection_matrix` documents what happens without it.

    The reconstructed volume is square in-plane with the rotation axis at
    ``(n_u - 1) / 2``; a ``center`` other than that is undone first, so the volume is
    always returned on the centred grid.
    """
    projections = np.asarray(projections, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float64)
    n_theta, n_z, n_u = projections.shape
    del order  # the matrix path is bilinear by construction

    if center is not None:
        projections = _recenter(projections, from_center=float(center), to_center=(n_u - 1) / 2.0)

    matrix = projection_matrix(n_u, angles, cache=_MATRIX_CACHE)
    flat = np.ascontiguousarray(projections.transpose(0, 2, 1)).reshape(n_theta * n_u, n_z)
    volume = matrix.T @ flat  # (n_u * n_u, n_z)
    return np.ascontiguousarray(volume.T.reshape(n_z, n_u, n_u), dtype=np.float32)


def _recenter(projections: np.ndarray, *, from_center: float, to_center: float) -> np.ndarray:
    """Slide the detector so the rotation axis moves from ``from_center`` to ``to_center``."""
    delta = to_center - from_center
    if abs(delta) < 1e-9:
        return projections
    from scipy.ndimage import shift as ndi_shift  # noqa: PLC0415

    return ndi_shift(projections, (0.0, 0.0, delta), order=3, mode="constant", cval=0.0)


# --------------------------------------------------------------------------------
# The perturbation catalogue
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbationSpec:
    """Every perturbation the harness can inject, each independently switchable.

    Zero disables a term. The list maps one-to-one onto the roadmap's diagnostic
    catalogue, which is deliberate: each entry exists so a diagnostic can be shown to
    fire on it and only on it.

    Attributes
    ----------
    jitter_dy_rms, jitter_dx_rms:
        Per-projection independent Gaussian jitter, px RMS. Our lens-1 scan measured
        ``dy`` 25.0 px and ``dx`` 7.5 px RMS on a 1816 px detector -- note that is the
        *opposite* of the roadmap's "vertical is the easy direction", and worth saying
        out loud in any write-up.
    center_dy, center_dx:
        A constant offset added to every projection. Pure gauge in both axes (a
        translation of the reconstructed volume; in ``dx`` it is also exactly the
        rotation-axis position), so it is injected to prove the scorer removes it
        rather than to be recovered.
    drift_dy, drift_dx:
        Amplitude of a smooth drift across the scan, px. ``drift_shape`` picks
        ``"linear"`` (thermal walk) or ``"cosine"`` (one full period).
    axis_tilt_deg:
        Rotation-axis tilt *in* the detector plane. Rigid per projection but not a
        translation: a per-projection shift aligner cannot remove it.
    out_of_plane_tilt_deg:
        Rotation-axis tilt *toward the beam*. A true 3D geometry error; no
        per-projection 2D transform can remove it. Triples projector cost.
    magnification_drift:
        Fractional magnification change from first to last projection (e.g. 0.01 =
        1 %), applied as an isotropic scale about the frame centre.
    angle_error_rms_deg:
        Gaussian error between the true projection angle and the angle handed to the
        aligner. The aligner sees ``case.angles``; the projector used
        ``truth.angles_true``.
    phase_ramp_rms, phase_offset_rms:
        Per-projection linear phase ramp (rad across the full frame, RMS over
        projections, independent in u and v) and constant offset (rad RMS). These are
        the two ambiguities ptychographic phase retrieval leaves behind, and the
        reason the roadmap insists on comparing *phase gradients*: differentiating
        sends the offset to zero and the ramp to a constant.
    truncation_px:
        Columns cut from each side of the detector, so the object no longer fits the
        field of view. The rigid truth is unaffected (a symmetric crop does not move
        content) but the line integrals are now incomplete.
    deformation_px:
        RMS amplitude of a smooth, per-projection, zero-mean non-rigid warp, px.
        ``deformation_scale`` is its correlation length in px. Zero-mean is
        load-bearing: a field with a net translation would silently contaminate the
        rigid ground truth.
    noise_rms:
        Gaussian detector noise, as a fraction of the clean projection standard
        deviation.
    seed:
        Everything random in this spec is drawn from ``default_rng(seed)`` in a fixed
        order, so a case is exactly reproducible from its spec.
    """

    jitter_dy_rms: float = 0.0
    jitter_dx_rms: float = 0.0
    center_dy: float = 0.0
    center_dx: float = 0.0
    drift_dy: float = 0.0
    drift_dx: float = 0.0
    drift_shape: str = "linear"  # "linear" | "cosine"
    axis_tilt_deg: float = 0.0
    out_of_plane_tilt_deg: float = 0.0
    magnification_drift: float = 0.0
    angle_error_rms_deg: float = 0.0
    phase_ramp_rms: float = 0.0
    phase_offset_rms: float = 0.0
    truncation_px: int = 0
    deformation_px: float = 0.0
    deformation_scale: float = 8.0
    noise_rms: float = 0.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PerturbationSpec":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown PerturbationSpec field(s): {', '.join(unknown)}")
        return cls(**raw)

    @property
    def max_rigid_shift(self) -> float:
        """A conservative bound on the largest rigid displacement this spec injects.

        Used to size the zero margin. 4 sigma covers a 1000-projection draw with room
        to spare; the drift and the constant offset add on top because they are not
        random.
        """
        jitter = 4.0 * max(self.jitter_dy_rms, self.jitter_dx_rms)
        drift = max(abs(self.drift_dy), abs(self.drift_dx))
        offset = max(abs(self.center_dy), abs(self.center_dx))
        return jitter + drift + offset


#: The perturbation that reproduces our own P06 lens-1 scan's measured misalignment,
#: for benchmarking against a user-supplied reconstruction of it. The numbers are
#: measurements (rms dy 25.0 px, dx 7.5 px at 74.50973137 nm/px); the *data* is not
#: shipped and the volume path is always supplied by the caller.
P06_LENS1_SPEC = PerturbationSpec(
    jitter_dy_rms=25.0,
    jitter_dx_rms=7.5,
    seed=11023330,
)


@dataclass
class GroundTruth:
    """Exactly what was injected. The scorer compares against this and nothing else."""

    dy: np.ndarray  # per-projection content displacement, rows (px)
    dx: np.ndarray  # per-projection content displacement, columns (px)
    angles_true: np.ndarray  # angles the projector actually used (radians)
    angles_reported: np.ndarray  # angles the aligner is told about (radians)
    magnification: np.ndarray  # per-projection isotropic scale factor
    phase_ramp: np.ndarray  # (n, 2) rad across the full frame, [u, v]
    phase_offset: np.ndarray  # (n,) rad
    axis_tilt_deg: float
    out_of_plane_tilt_deg: float
    truncation_px: int
    deformation_rms_px: float
    deformation_field: np.ndarray | None = None  # (n, 2, n_v, n_u) if kept
    noise_rms: float = 0.0

    @property
    def n_angles(self) -> int:
        return int(self.dy.size)

    @property
    def rigid_rms(self) -> tuple[float, float]:
        """RMS of the injected rigid displacement, ``(dy, dx)`` in px."""
        return float(np.sqrt(np.mean(self.dy**2))), float(np.sqrt(np.mean(self.dx**2)))

    def to_dict(self, *, arrays: bool = True) -> dict[str, Any]:
        """JSON-friendly. ``arrays=False`` keeps only the scalars (for a report header)."""
        out: dict[str, Any] = {
            "n_angles": self.n_angles,
            "rigid_rms_dy_px": self.rigid_rms[0],
            "rigid_rms_dx_px": self.rigid_rms[1],
            "axis_tilt_deg": self.axis_tilt_deg,
            "out_of_plane_tilt_deg": self.out_of_plane_tilt_deg,
            "truncation_px": self.truncation_px,
            "deformation_rms_px": self.deformation_rms_px,
            "noise_rms": self.noise_rms,
            "angle_error_rms_deg": float(
                np.degrees(np.sqrt(np.mean((self.angles_true - self.angles_reported) ** 2)))
            ),
            "magnification_span": float(self.magnification.max() - self.magnification.min()),
        }
        if arrays:
            out["dy"] = self.dy.tolist()
            out["dx"] = self.dx.tolist()
        return out


@dataclass
class BenchmarkCase:
    """A perturbed projection stack plus everything needed to score an aligner on it.

    ``projections`` and ``angles`` are all an aligner may look at. ``clean`` (the
    unperturbed projections at the same reported angles) and ``truth`` are the
    scorer's, and handing them to an aligner is cheating -- which is precisely what
    :class:`~benchmarks.runner.OracleAligner` does on purpose, as the upper bound
    that proves the scoring path itself is sound.
    """

    name: str
    projections: np.ndarray  # (n_theta, n_v, n_u) -- what the aligner sees
    angles: np.ndarray  # radians, as REPORTED (may carry readback error)
    truth: GroundTruth
    spec: PerturbationSpec
    clean: np.ndarray | None = None  # unperturbed projections, same shape
    volume: np.ndarray | None = None  # source volume, when the case had one
    pixel_size_nm: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_angles(self) -> int:
        return int(self.projections.shape[0])

    @property
    def width(self) -> int:
        return int(self.projections.shape[2])

    @property
    def height(self) -> int:
        return int(self.projections.shape[1])

    @property
    def center(self) -> float:
        """The true rotation-axis column of :attr:`projections`."""
        return (self.width - 1) / 2.0

    def as_projection_data(self):
        """Wrap as a :class:`tktomo.io.ProjectionData` for TKtomo's own machinery."""
        from tktomo.io import ProjectionData  # noqa: PLC0415

        return ProjectionData(
            data=np.asarray(self.projections, dtype=np.float32),
            angles=np.asarray(self.angles, dtype=np.float64),
            metadata={
                "name": self.name,
                "pixel_size_nm": self.pixel_size_nm,
                **self.metadata,
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.projections.shape),
            "pixel_size_nm": self.pixel_size_nm,
            "angle_span_deg": float(np.degrees(self.angles.max() - self.angles.min())),
            "spec": self.spec.to_dict(),
            "truth": self.truth.to_dict(arrays=False),
            **self.metadata,
        }


# --------------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------------


def _drift(n: int, amplitude: float, shape: str) -> np.ndarray:
    if amplitude == 0.0:
        return np.zeros(n)
    t = np.linspace(0.0, 1.0, n)
    if shape == "linear":
        return amplitude * (t - t.mean())
    if shape == "cosine":
        return amplitude * np.cos(2.0 * np.pi * t)
    raise ValueError(f"drift_shape must be 'linear' or 'cosine', got {shape!r}")


def _smooth_field(
    rng: np.random.Generator, shape: tuple[int, int], scale: float, amplitude: float
) -> np.ndarray:
    """A zero-mean smooth displacement field of the requested RMS amplitude."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    raw = rng.standard_normal(shape)
    field_ = gaussian_filter(raw, sigma=max(scale, 1e-3), mode="wrap")
    field_ -= field_.mean()  # a net translation here would corrupt the rigid truth
    rms = float(np.sqrt(np.mean(field_**2)))
    if rms == 0.0:
        return np.zeros(shape)
    return field_ * (amplitude / rms)


def _fourier_shift_stack(stack: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    from scipy.ndimage import fourier_shift  # noqa: PLC0415

    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        spectrum = np.fft.fftn(stack[i])
        out[i] = np.fft.ifftn(fourier_shift(spectrum, (dy[i], dx[i]))).real
    return out


def perturb(
    clean: np.ndarray,
    angles: np.ndarray,
    spec: PerturbationSpec,
    *,
    margin: int = 0,
    store_deformation: bool = False,
) -> tuple[np.ndarray, GroundTruth]:
    """Apply ``spec`` to a clean projection stack and return ``(perturbed, truth)``.

    ``clean`` must already carry the geometry-level perturbations that cannot be
    expressed per-projection (axis tilt, out-of-plane tilt, angle readback error) --
    :func:`_project_case` bakes those into the forward projection and passes the
    resulting stack here. Everything expressible in the projection domain is applied
    here, in the order documented in the module docstring.

    ``margin`` is the width of the guaranteed-zero border already present in
    ``clean``. The Fourier shift wraps, so a shift larger than the margin drags real
    mass off one edge and back in at the other; the corrupted stack is then not a
    shifted copy of *any* consistent object and no aligner could recover the truth.
    This raises rather than letting that happen silently.
    """
    clean = np.asarray(clean, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float64)
    n, n_v, n_u = clean.shape
    rng = np.random.default_rng(spec.seed)

    # -- 1. rigid displacement, assembled but not yet applied -----------------------
    dy = rng.normal(0.0, spec.jitter_dy_rms, n) if spec.jitter_dy_rms else np.zeros(n)
    dx = rng.normal(0.0, spec.jitter_dx_rms, n) if spec.jitter_dx_rms else np.zeros(n)
    dy = dy + spec.center_dy + _drift(n, spec.drift_dy, spec.drift_shape)
    dx = dx + spec.center_dx + _drift(n, spec.drift_dx, spec.drift_shape)

    required = float(np.max(np.abs(np.concatenate([dy, dx])))) if n else 0.0
    if margin > 0 and required >= margin:
        raise ValueError(
            f"margin={margin} px is too small for this spec: it injects up to "
            f"{required:.2f} px of rigid shift. A Fourier shift wraps, so the object "
            "would be truncated on one edge and duplicated on the other, and the "
            "ground truth would no longer be recoverable by any algorithm. Increase "
            "`margin` or reduce the jitter/drift/offset."
        )

    out = clean.copy()

    # -- 2. magnification drift -----------------------------------------------------
    magnification = np.ones(n)
    if spec.magnification_drift:
        magnification = 1.0 + np.linspace(
            -spec.magnification_drift / 2.0, spec.magnification_drift / 2.0, n
        )
        out = _apply_magnification(out, magnification)

    # -- 3. smooth non-rigid deformation (zero-mean, so the rigid truth stays exact) -
    deformation_field = None
    if spec.deformation_px:
        from scipy.ndimage import map_coordinates  # noqa: PLC0415

        grid_v, grid_u = np.mgrid[0:n_v, 0:n_u].astype(np.float32)
        fields = np.empty((n, 2, n_v, n_u), dtype=np.float32)
        for i in range(n):
            fv = _smooth_field(rng, (n_v, n_u), spec.deformation_scale, spec.deformation_px)
            fu = _smooth_field(rng, (n_v, n_u), spec.deformation_scale, spec.deformation_px)
            fields[i, 0] = fv
            fields[i, 1] = fu
            out[i] = map_coordinates(
                out[i], [grid_v + fv, grid_u + fu], order=3, mode="constant", cval=0.0
            )
        if store_deformation:
            deformation_field = fields

    # -- 4. the rigid shift, last of the geometric terms ----------------------------
    if np.any(dy) or np.any(dx):
        out = _fourier_shift_stack(out, dy, dx)

    # -- 5. phase ramp + constant offset --------------------------------------------
    ramp = np.zeros((n, 2))
    offset = np.zeros(n)
    if spec.phase_ramp_rms or spec.phase_offset_rms:
        if spec.phase_ramp_rms:
            ramp = rng.normal(0.0, spec.phase_ramp_rms, (n, 2))
        if spec.phase_offset_rms:
            offset = rng.normal(0.0, spec.phase_offset_rms, n)
        u = np.linspace(0.0, 1.0, n_u, dtype=np.float32)[None, :]
        v = np.linspace(0.0, 1.0, n_v, dtype=np.float32)[:, None]
        for i in range(n):
            out[i] += (ramp[i, 0] * u + ramp[i, 1] * v + offset[i]).astype(np.float32)

    # -- 6. FOV truncation ----------------------------------------------------------
    if spec.truncation_px:
        cut = int(spec.truncation_px)
        if 2 * cut >= n_u:
            raise ValueError(f"truncation_px={cut} removes the whole {n_u} px detector")
        # Symmetric, so it moves no content and the rigid truth survives untouched.
        out = np.ascontiguousarray(out[:, :, cut : n_u - cut])

    # -- 7. detector noise ----------------------------------------------------------
    if spec.noise_rms:
        sigma = spec.noise_rms * float(clean.std())
        out = out + rng.normal(0.0, sigma, out.shape).astype(np.float32)

    truth = GroundTruth(
        dy=dy,
        dx=dx,
        angles_true=angles.copy(),  # replaced by _project_case when angles are perturbed
        angles_reported=angles.copy(),
        magnification=magnification,
        phase_ramp=ramp,
        phase_offset=offset,
        axis_tilt_deg=spec.axis_tilt_deg,
        out_of_plane_tilt_deg=spec.out_of_plane_tilt_deg,
        truncation_px=int(spec.truncation_px),
        deformation_rms_px=float(spec.deformation_px),
        deformation_field=deformation_field,
        noise_rms=float(spec.noise_rms),
    )
    return out.astype(np.float32), truth


def _apply_magnification(stack: np.ndarray, magnification: np.ndarray) -> np.ndarray:
    """Isotropic scale about the frame centre, one factor per projection."""
    from scipy.ndimage import affine_transform  # noqa: PLC0415

    n, n_v, n_u = stack.shape
    centre = np.array([(n_v - 1) / 2.0, (n_u - 1) / 2.0])
    out = np.empty_like(stack)
    for i in range(n):
        scale = 1.0 / float(magnification[i])
        matrix = np.diag([scale, scale])
        offset = centre - matrix @ centre
        out[i] = affine_transform(
            stack[i], matrix, offset=offset, order=3, mode="constant", cval=0.0
        )
    return out


# --------------------------------------------------------------------------------
# Generator (a): fully synthetic
# --------------------------------------------------------------------------------

# Ellipsoids in normalised [-1, 1] coordinates: (value, cx, cy, cz, ax, ay, az).
# Adapted from tktomo.io.phantom._ELLIPSOIDS -- reused rather than reinvented -- with
# two additions. The off-axis blobs matter: a centred, symmetric phantom has a
# near-zero COM sinusoid, so a COM pre-alignment would be trivially satisfied and the
# horizontal direction would never actually be tested.
_ELLIPSOIDS = (
    (0.20, 0.00, 0.00, 0.00, 0.80, 0.80, 0.95),  # outer body, all slices
    (0.30, -0.30, 0.10, -0.55, 0.25, 0.30, 0.30),  # low-z blob
    (-0.20, 0.35, -0.10, 0.00, 0.20, 0.20, 0.35),  # mid, subtractive
    (0.40, 0.00, 0.40, 0.45, 0.30, 0.15, 0.30),  # high-z
    (0.35, -0.20, -0.35, 0.60, 0.18, 0.18, 0.25),  # top small
    (-0.15, 0.10, 0.10, -0.20, 0.12, 0.12, 0.15),  # tiny core
    (0.45, 0.45, 0.35, -0.30, 0.09, 0.09, 0.12),  # off-axis, breaks the COM symmetry
    (0.25, -0.50, -0.15, 0.25, 0.08, 0.14, 0.10),  # off-axis, other side
)


def synthetic_volume(size: int = 64, n_slices: int = 16) -> np.ndarray:
    """A 3D ellipsoid phantom ``(n_slices, size, size)``, numpy only.

    Deliberately the same construction as :func:`tktomo.io.phantom.generate_volume`
    (which the UIs already fall back to) so the benchmark and the app exercise the
    same test article, plus two off-axis blobs -- see :data:`_ELLIPSOIDS`. Masked to
    the inscribed disc, because anything outside it is destroyed by rotation.
    """
    ys, xs = np.mgrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    zs = np.array([0.0]) if n_slices == 1 else np.linspace(-0.8, 0.8, n_slices)
    volume = np.zeros((n_slices, size, size), dtype=np.float64)
    for value, cx, cy, cz, ax, ay, az in _ELLIPSOIDS:
        d = (
            ((xs[None] - cx) / ax) ** 2
            + ((ys[None] - cy) / ay) ** 2
            + ((zs[:, None, None] - cz) / az) ** 2
        )
        volume[d <= 1.0] += value
    np.clip(volume, 0.0, None, out=volume)
    volume *= circular_mask(size, size, radius_fraction=0.98)[None]
    return volume.astype(np.float32)


def synthetic_case(
    *,
    name: str = "synthetic",
    size: int = 64,
    n_slices: int = 12,
    n_angles: int = 60,
    angle_span_deg: float = 180.0,
    spec: PerturbationSpec | None = None,
    margin: int | None = None,
    pixel_size_nm: float = 1.0,
    store_deformation: bool = False,
    order: int = 1,
) -> BenchmarkCase:
    """A self-contained benchmark case: no external data, numpy + scipy only.

    ``margin`` defaults to a zero border wide enough for the spec's largest rigid
    shift (see :meth:`PerturbationSpec.max_rigid_shift`), because the Fourier shift
    used to inject it wraps.
    """
    spec = spec or PerturbationSpec(jitter_dy_rms=2.5, jitter_dx_rms=0.75, seed=0)
    if margin is None:
        margin = int(math.ceil(spec.max_rigid_shift)) + 4

    volume = synthetic_volume(size=size, n_slices=n_slices)
    angles = np.deg2rad(np.linspace(0.0, angle_span_deg, n_angles, endpoint=False))
    return _project_case(
        name=name,
        volume=volume,
        angles=angles,
        spec=spec,
        margin=margin,
        pixel_size_nm=pixel_size_nm,
        store_deformation=store_deformation,
        order=order,
        metadata={"generator": "synthetic", "size": size, "n_slices": n_slices},
    )


# --------------------------------------------------------------------------------
# Generator (b): synthetic-from-real
# --------------------------------------------------------------------------------


def load_volume(
    path: str | Path,
    *,
    dataset: str = "/tomogram/data",
    slices: slice | None = None,
    bin_factor: int = 1,
    pattern: str = "*.tif*",
    dtype: str = "float32",
) -> np.ndarray:
    """Load a reconstructed volume from a user-supplied path.

    Supported layouts, chosen by what ``path`` is:

    * a **directory** of per-slice TIFFs (``pattern``, sorted by filename -- so
      ``cgls_00000.tiff`` style zero-padded names are required for the order to be
      the geometric one). Needs ``tifffile`` or ``imageio``.
    * a **.npy** file holding ``(n_z, n_y, n_x)``.
    * an **.h5 / .hdf5 / .nxs** file, reading ``dataset``. Needs ``h5py``.

    ``slices`` picks a sub-stack of detector rows and ``bin_factor`` mean-pools the
    two in-plane axes -- both matter, because a 1488 x 1816 x 1816 volume is 19 GB in
    float32 and forward-projecting all of it for a benchmark is neither necessary nor
    possible on a login node.

    No path is ever defaulted to measured data. This is the only door through which a
    real volume enters the harness, and the caller opens it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No volume at {path}")

    if path.is_dir():
        volume = _load_tiff_directory(path, pattern=pattern, slices=slices)
    elif path.suffix == ".npy":
        volume = np.load(path, mmap_mode="r")
        volume = np.asarray(volume[slices] if slices else volume)
    elif path.suffix in {".h5", ".hdf5", ".nxs"}:
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Reading an HDF5 volume needs h5py: pip install h5py"
            ) from exc
        with h5py.File(path, "r") as handle:
            if dataset not in handle:
                available = sorted(handle.keys())
                raise KeyError(
                    f"{path} has no dataset {dataset!r}; top-level keys are {available}. "
                    "Pass dataset=... with the right path."
                )
            node = handle[dataset]
            volume = np.asarray(node[slices] if slices else node[...])
    else:
        raise ValueError(
            f"Cannot load a volume from {path.suffix!r}. Give a directory of TIFFs, "
            "a .npy, or an .h5/.hdf5/.nxs with dataset=..."
        )

    volume = np.asarray(volume, dtype=dtype)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D; {path} gave shape {volume.shape}")
    if bin_factor > 1:
        volume = _bin_volume(volume, bin_factor)
    return volume


def _load_tiff_directory(path: Path, *, pattern: str, slices: slice | None) -> np.ndarray:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {path}")
    if slices is not None:
        files = files[slices]
    try:
        from tifffile import imread  # noqa: PLC0415
    except ImportError:
        try:
            from imageio.v3 import imread  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Reading a TIFF slice directory needs tifffile or imageio: "
                "pip install tifffile"
            ) from exc
    return np.stack([np.asarray(imread(f)) for f in files])


def _bin_volume(volume: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool the two in-plane axes. The row axis is left alone (it is the axis)."""
    n_z, n_y, n_x = volume.shape
    n_y -= n_y % factor
    n_x -= n_x % factor
    trimmed = volume[:, :n_y, :n_x]
    return trimmed.reshape(n_z, n_y // factor, factor, n_x // factor, factor).mean(axis=(2, 4))


def load_angles(
    path: str | Path,
    *,
    dataset: str = "exchange/theta",
    degrees: bool = True,
    subsample: int = 1,
) -> np.ndarray:
    """Read the rotation angles from a user-supplied HDF5 file, returned in radians.

    ``exchange/theta`` in degrees is the DXchange layout our own scans are written
    in; NXtomo files usually want ``dataset="/entry/sample/rotation_angle"``.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Reading angles needs h5py: pip install h5py") from exc

    with h5py.File(Path(path), "r") as handle:
        if dataset not in handle:
            raise KeyError(f"{path} has no dataset {dataset!r}")
        angles = np.asarray(handle[dataset][...], dtype=np.float64)
    angles = angles[::subsample]
    return np.deg2rad(angles) if degrees else angles


def volume_case(
    volume: np.ndarray,
    angles: np.ndarray,
    *,
    name: str = "from_volume",
    spec: PerturbationSpec | None = None,
    margin: int | None = None,
    pixel_size_nm: float = 1.0,
    store_deformation: bool = False,
    order: int = 1,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkCase:
    """Forward-project a user-supplied volume, then inject known perturbations.

    Real sample statistics -- the streaks, the noise texture, the phase distribution
    of an actual reconstruction -- with exact ground truth, which is the one thing a
    real dataset can never give you. Note the loop it closes: the volume was itself
    reconstructed from projections that were aligned by *something*, so residual
    misalignment in that reconstruction shows up here as extra blur, not as extra
    truth. The recovered shifts are still exact; the difficulty of the case is
    understated by however good the original alignment was.
    """
    spec = spec or PerturbationSpec()
    if margin is None:
        margin = int(math.ceil(spec.max_rigid_shift)) + 4
    return _project_case(
        name=name,
        volume=np.asarray(volume, dtype=np.float32),
        angles=np.asarray(angles, dtype=np.float64),
        spec=spec,
        margin=margin,
        pixel_size_nm=pixel_size_nm,
        store_deformation=store_deformation,
        order=order,
        metadata={"generator": "volume", **(metadata or {})},
    )


def _project_case(
    *,
    name: str,
    volume: np.ndarray,
    angles: np.ndarray,
    spec: PerturbationSpec,
    margin: int,
    pixel_size_nm: float,
    store_deformation: bool,
    order: int,
    metadata: dict[str, Any],
) -> BenchmarkCase:
    """The shared body of both generators: project, pad, perturb, package."""
    rng = np.random.default_rng(spec.seed + 977)  # a different stream from perturb()'s

    angles_reported = np.asarray(angles, dtype=np.float64)
    angles_true = angles_reported
    if spec.angle_error_rms_deg:
        # The projector uses the TRUE angles; the aligner is told the reported ones.
        angles_true = angles_reported + np.deg2rad(
            rng.normal(0.0, spec.angle_error_rms_deg, angles_reported.size)
        )

    clean = forward_project(
        volume,
        angles_true,
        axis_tilt_deg=spec.axis_tilt_deg,
        out_of_plane_tilt_deg=spec.out_of_plane_tilt_deg,
        order=order,
    )
    if margin > 0:
        clean = np.pad(clean, ((0, 0), (margin, margin), (margin, margin)), mode="constant")

    projections, truth = perturb(
        clean, angles_reported, spec, margin=margin, store_deformation=store_deformation
    )
    truth.angles_true = angles_true
    truth.angles_reported = angles_reported

    # The reference stack the residual metric compares against is the clean one at the
    # same crop as the perturbed one, so the two are the same shape.
    reference = clean
    if spec.truncation_px:
        cut = int(spec.truncation_px)
        reference = np.ascontiguousarray(clean[:, :, cut : clean.shape[2] - cut])

    return BenchmarkCase(
        name=name,
        projections=projections,
        angles=angles_reported,
        truth=truth,
        spec=spec,
        clean=reference,
        volume=volume,
        pixel_size_nm=pixel_size_nm,
        metadata={"margin": margin, "projector_order": order, **metadata},
    )


def cases_from_catalogue(
    base: PerturbationSpec | None = None,
    *,
    names: Sequence[str] | None = None,
    **case_kwargs: Any,
) -> dict[str, BenchmarkCase]:
    """One synthetic case per perturbation, each with that perturbation alone.

    This is the diagnostic sweep: run an aligner over the catalogue and the failures
    say *which* geometry error it cannot handle, which a single all-perturbations-on
    case never can. ``base`` (default: modest jitter) is added to every case so the
    aligner always has something to actually align.
    """
    base = base or PerturbationSpec(jitter_dy_rms=1.5, jitter_dx_rms=0.5, seed=0)
    catalogue: dict[str, dict[str, Any]] = {
        "jitter_only": {},
        "center_offset": {"center_dy": 3.0, "center_dx": 3.0},
        "vertical_drift": {"drift_dy": 6.0},
        "axis_tilt": {"axis_tilt_deg": 0.5},
        "out_of_plane_tilt": {"out_of_plane_tilt_deg": 0.5},
        "magnification_drift": {"magnification_drift": 0.02},
        "angle_error": {"angle_error_rms_deg": 0.3},
        "phase_ramp": {"phase_ramp_rms": 0.5, "phase_offset_rms": 0.5},
        "truncation": {"truncation_px": 6},
        "deformation": {"deformation_px": 1.0},
        "noise": {"noise_rms": 0.05},
    }
    selected = names or list(catalogue)
    out: dict[str, BenchmarkCase] = {}
    for label in selected:
        if label not in catalogue:
            raise KeyError(f"Unknown catalogue entry {label!r}; have {sorted(catalogue)}")
        spec = PerturbationSpec(**{**base.to_dict(), **catalogue[label]})
        out[label] = synthetic_case(name=label, spec=spec, **case_kwargs)
    return out
