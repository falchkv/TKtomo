"""The non-rigid benchmark scenario: where rigid alignment provably cannot win.

Every other case in this harness asks "how accurately does the aligner recover the shift
that was injected?". That question is meaningless here, because the perturbation is not a
shift: the sample itself deforms during the scan, and **no** per-projection rigid
transform satisfies all the projections at once. So this scenario asks a different
question, in three rows:

1. **The rigid floor.** Run the rigid loop to convergence on the deformed data and
   measure the residual it plateaus at. That number is the scenario's headline: it is not
   a failure of a particular implementation, it is the best any rigid model can do, and
   the point of the whole exercise is to establish it before claiming anything beats it.
   The row is quantified three ways -- the plateau value, the gap to the same loop's
   residual on undeformed data (the achievable floor), and the fact that the leftover is
   *localised* rather than spread, which is what says the shortfall is deformation and
   not noise.
2. **The non-rigid answer** on the same data: how much of that gap it closes, how well
   the recovered deformation field matches the known truth (in pixels, gauge-fixed, inside
   the object support), and what happens to the resolution.
3. **The negative control** -- the identical pipeline on the identical phantom with **no
   deformation**. This row is the one that makes the other two mean anything. A non-rigid
   method has enough freedom to improve the fitted residual on *any* data, including data
   with nothing to find, so a scenario that only reported rows 1 and 2 would be a
   demonstration, not a benchmark. The control reports the deformation the method invents
   from nothing and whether the volume got worse.

Run it::

    python -m benchmarks.scenario_nonrigid --size 32 --iterations 4
    python -m benchmarks.scenario_nonrigid --out /somewhere/outside/the/repo.json

WHAT THIS SCENARIO IS HONEST ABOUT
----------------------------------
* **It is a partial inverse crime.** The projector that makes the data is the projector
  that reconstructs it, which flatters every method equally but flatters them all. It is
  mitigated, not removed: the truth is generated with cubic interpolation and the
  reconstruction and warping use linear, so a method cannot score by agreeing with its own
  interpolation kernel. Read the *relative* numbers (rigid floor vs non-rigid, deformed vs
  control), never the absolute residual, which carries the projector's own floor of ~0.13.
* **The rigid loop here is a compact re-implementation**, not
  :class:`~tktomo.ptycho_align.core.engine.AlignmentEngine`. It exists so the scenario
  runs with numpy and scipy alone, on a machine with no tomopy. It uses the same joint
  reprojection scheme (reconstruct, reproject, register, accumulate on the pristine
  stack), its shifts are gauge-fixed the way the engine's are, and it refuses to return a
  result whose residual did not fall. What it is NOT is a benchmark of the engine: it
  establishes the rigid floor, and the floor is a property of the data.
* **A benchmark score is not a validation.** The deformation is judged against a known
  truth here. On real data there is no truth, and the only honest evidence is the held-out
  reprojection residual plus a split-data FSC paired with a residual map.

Imports from ``benchmarks.phantom`` / ``benchmarks.metrics`` / ``benchmarks.runner`` are
all optional and guarded: this file runs standalone with numpy + scipy if none of them are
present, and says which of them it used in the report header.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.ndimage import shift as ndshift

logger = logging.getLogger(__name__)

__all__ = [
    "ScenarioConfig",
    "ScenarioReport",
    "ScipyParallelBackend",
    "StageResult",
    "build_scan",
    "main",
    "rigid_loop",
    "run_scenario",
    "truth_field",
]

# ---------------------------------------------------------------------------------
# Optional pieces of the harness and of tktomo. Every one of these is guarded, and the
# report records which were available, because a number computed by a local fallback and
# a number computed by benchmarks.metrics must never be silently interchangeable.
# ---------------------------------------------------------------------------------

try:
    from benchmarks.metrics import fourier_shell_correlation as _bench_fsc
    from benchmarks.metrics import split_half_indices as _bench_split

    HAVE_BENCH_METRICS = True
except ImportError:  # pragma: no cover - depends on the tree state
    _bench_fsc = _bench_split = None  # type: ignore[assignment]
    HAVE_BENCH_METRICS = False

try:
    from benchmarks.phantom import synthetic_volume as _bench_volume

    HAVE_BENCH_PHANTOM = True
except ImportError:  # pragma: no cover
    _bench_volume = None  # type: ignore[assignment]
    HAVE_BENCH_PHANTOM = False

try:
    from tktomo.ptycho_align.core.deformation import (
        DeformationField,
        DeformationSequence,
        coarse_support_mask,
        invert,
        sequence_rms_difference,
        warp_volume,
    )
    from tktomo.ptycho_align.core.nonrigid import NonRigidAligner, NonRigidConfig

    HAVE_NONRIGID = True
    NONRIGID_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - the core agent's module may be absent
    HAVE_NONRIGID = False
    NONRIGID_IMPORT_ERROR = str(exc)

try:
    from tktomo.ptycho_align.core.nonrigid_gate import (
        GateConfig,
        evaluate_gate,
        format_gate,
    )

    HAVE_GATE = True
except ImportError as exc:  # pragma: no cover
    HAVE_GATE = False
    GATE_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------------
# A scipy-only parallel-beam backend conforming to ReconBackend
# ---------------------------------------------------------------------------------


class ScipyParallelBackend:
    """Parallel-beam projector + FBP, in scipy. Real, slow, and adjoint up to interpolation.

    Volume ``(nz, ny, nx)``, projections ``(n_angles, nz, nx)`` -- TomoPy's layout, so
    :class:`~tktomo.ptycho_align.core.nonrigid.NonRigidAligner` cannot tell it from the
    real backend. Deliberately not a mock: a mock would let a sign error through, and a
    sign error in the projector is the failure this whole scenario would otherwise
    quietly report as "non-rigid alignment does not work".

    ``order`` is the interpolation order. The scenario builds its ground truth at order 3
    and reconstructs at order 1 so that no method can score by agreeing with the kernel
    that made the data.
    """

    name = "scipy_parallel_scenario"

    def __init__(self, order: int = 1) -> None:
        self.order = int(order)

    @staticmethod
    def _shift_for(center: float | None, nx: int) -> float:
        return 0.0 if center is None else float(center) - nx / 2.0

    def reproject(self, volume, angles, *, center=None, **_kwargs) -> np.ndarray:
        volume = np.asarray(volume, dtype=np.float32)
        nz, _ny, nx = volume.shape
        angles = np.asarray(angles, dtype=np.float64)
        out = np.empty((angles.size, nz, nx), dtype=np.float32)
        for i, theta in enumerate(angles):
            out[i] = rotate(
                volume, np.degrees(theta), axes=(1, 2), reshape=False,
                order=self.order, mode="constant", cval=0.0,
            ).sum(axis=1)
        delta = self._shift_for(center, nx)
        if abs(delta) > 1e-9:
            out = ndshift(out, (0, 0, delta), order=1, mode="constant", cval=0.0)
        return out

    def reconstruct(self, projections, angles, *, center=None, algorithm="fbp", **_kwargs):
        if algorithm not in {"fbp", "bp"}:
            raise ValueError(
                f"{self.name} has 'fbp' and 'bp' only; got {algorithm!r}. Anything "
                "iterative would make the scenario's runtime dominated by the "
                "reconstruction rather than by what it is measuring."
            )
        projections = np.asarray(projections, dtype=np.float32)
        n_angles, nz, nx = projections.shape
        delta = self._shift_for(center, nx)
        if abs(delta) > 1e-9:
            projections = ndshift(projections, (0, 0, -delta), order=1, mode="constant")

        if algorithm == "fbp":
            pad = int(2 ** math.ceil(math.log2(max(64, 2 * nx))))
            frequency = np.fft.rfftfreq(pad)
            ramp = 2.0 * frequency * np.sinc(frequency)  # Shepp-Logan
            padded = np.zeros((n_angles, nz, pad), dtype=np.float32)
            padded[:, :, :nx] = projections
            projections = np.fft.irfft(
                np.fft.rfft(padded, axis=2) * ramp, n=pad, axis=2
            )[:, :, :nx]

        volume = np.zeros((nz, nx, nx), dtype=np.float32)
        for i, theta in enumerate(np.asarray(angles, dtype=np.float64)):
            smeared = np.repeat(projections[i][:, None, :], nx, axis=1)
            volume += rotate(
                smeared, -np.degrees(theta), axes=(1, 2), reshape=False,
                order=self.order, mode="constant", cval=0.0,
            )
        volume *= np.pi / (2.0 * n_angles)
        return volume.astype(np.float32)


# ---------------------------------------------------------------------------------
# The phantom, the truth deformation, and the scan
# ---------------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Everything that defines the case and how hard it is. No paths, no external data."""

    size: int = 32  #: detector width and volume side, in voxels
    n_slices: int = 0  #: detector rows; 0 -> size // 2, which keeps the runtime sane
    n_subtomos: int = 4  #: acquisition-time blocks (each a complete sub-tomogram)
    angles_per: int = 30  #: projections per sub-tomogram
    deformation_px: float = 2.0  #: peak amplitude of the injected deformation, voxels
    #: Split between the LOCAL (beam-damage) and GLOBAL (drift/shear) deformation modes.
    #: The decision gate's localisation criterion is most sensitive to this; see
    #: :func:`truth_field`. 0.0 is a purely global deformation, which the gate vetoes.
    local_fraction: float = 0.5
    #: Gaussian detector noise as a fraction of the projection std. NOT zero by default:
    #: with no noise the two FSC half-sets differ only by angular sampling, the curve never
    #: crosses the half-bit threshold, and the resolution column is a row of infinities.
    noise_rms: float = 0.02
    jitter_px: float = 0.0  #: rigid jitter injected on top, px RMS (the rigid loop's job)
    rigid_iterations: int = 6
    nonrigid_iterations: int = 4
    grid_spacing: float = 10.0  #: DVF node spacing in voxels
    seed: int = 0

    @property
    def n_v(self) -> int:
        return self.n_slices or max(8, self.size // 2)

    @property
    def n_angles(self) -> int:
        return self.n_subtomos * self.angles_per

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def phantom(nz: int, nx: int, *, seed: int = 0) -> np.ndarray:
    """An ellipsoid with off-centre inclusions, band-limited.

    Uses ``benchmarks.phantom.synthetic_volume`` when the harness is present so the two
    agree on what "the phantom" means; the local construction is the fallback and is
    equivalent in character (compact support, internal features, smooth edges).
    """
    if HAVE_BENCH_PHANTOM:
        try:
            volume = np.asarray(_bench_volume(size=nx, n_slices=nz), dtype=np.float32)
            if volume.shape == (nz, nx, nx):
                return volume
            logger.warning(
                "benchmarks.phantom.synthetic_volume returned %s, expected %s; using the "
                "local phantom instead.", volume.shape, (nz, nx, nx),
            )
        except Exception as exc:  # noqa: BLE001 - a harness in flux must not break this
            logger.warning("benchmarks.phantom.synthetic_volume failed (%s); using the local one.", exc)

    z, y, x = np.mgrid[0:nz, 0:nx, 0:nx].astype(np.float32)
    cz, cy, cx = (nz - 1) / 2.0, (nx - 1) / 2.0, (nx - 1) / 2.0
    volume = np.zeros((nz, nx, nx), dtype=np.float32)
    body = (
        ((z - cz) / (0.42 * nz)) ** 2
        + ((y - cy) / (0.34 * nx)) ** 2
        + ((x - cx) / (0.30 * nx)) ** 2
    )
    volume[body < 1.0] = 1.0
    for (fz, fy, fx), radius, amplitude in (
        ((0.35, 0.40, 0.42), 0.09, 0.9),
        ((0.62, 0.58, 0.60), 0.07, -0.6),
        ((0.50, 0.62, 0.38), 0.06, 0.8),
        ((0.45, 0.45, 0.60), 0.05, -0.5),
    ):
        inside = ((z - fz * nz) ** 2 + (y - fy * nx) ** 2 + (x - fx * nx) ** 2) < (
            radius * nx
        ) ** 2
        volume[inside] += amplitude
    return gaussian_filter(volume, 0.8).astype(np.float32)


def truth_field(
    t: float, shape: tuple[int, int, int], grid, amplitude: float, local_fraction: float = 0.5
):
    """The known deformation at normalised acquisition time ``t`` in [0, 1].

    THREE spatial modes, and the mix matters more than the amplitude:

    * an axial drift growing monotonically (creep) and an in-plane shear that reverses
      (thermal) -- both **global**;
    * a compact bump displacing one corner of the object, growing with time -- **local**,
      the beam-damage mode Odstrcil et al. actually saw.

    ``local_fraction`` sets the split, and it is the knob the decision gate is most
    sensitive to: the roadmap's localisation criterion asks whether the leftover residual
    is confined to sub-regions, so a purely global deformation (``local_fraction=0``) is
    genuinely non-rigid and will still be VETOED by the gate. That is not a bug in either
    piece; it is the criterion doing exactly what the roadmap specifies, and the scenario
    exists partly to make that visible instead of leaving it in a footnote. Sweep it.

    The global in-plane mode is a pure shear and not a rotation: an in-plane rotation of
    the sample is nearly degenerate with a change in the assigned projection angles, the
    least identifiable mode there is, so building the test article on it would understate
    what the method can do.

    The three time profiles differ, so the field is deliberately NOT separable into "one
    shape scaled by time" -- a method that can only represent that would be caught here.
    """
    gz, gy, gx = grid
    ones = np.ones(grid, dtype=np.float32)
    z = np.linspace(0.0, 1.0, gz, dtype=np.float32)[:, None, None] * ones
    y = np.linspace(-1.0, 1.0, gy, dtype=np.float32)[None, :, None] * ones
    x = np.linspace(-1.0, 1.0, gx, dtype=np.float32)[None, None, :] * ones

    drift = (t - 0.5) * 2.0
    shear = float(np.cos(np.pi * t))
    global_modes = np.stack(
        [drift * np.sin(np.pi * z), shear * 0.7 * x, shear * 0.7 * y]
    ).astype(np.float32)

    # A compact bump at (z, y, x) ~ (0.35, -0.4, +0.45) of the grid, pushed outward.
    blob = np.exp(-(((z - 0.35) / 0.22) ** 2 + ((y + 0.40) / 0.35) ** 2 + ((x - 0.45) / 0.35) ** 2))
    swell = t**1.5  # damage accumulates; it does not come back
    local_modes = (swell * blob * np.stack([0.4 * ones, -0.9 * ones, 1.0 * ones])).astype(
        np.float32
    )

    weight = float(np.clip(local_fraction, 0.0, 1.0))
    vectors = (1.0 - weight) * global_modes + weight * local_modes
    return DeformationField(vectors * amplitude, shape)


def build_scan(config: ScenarioConfig, *, deform: bool) -> dict[str, Any]:
    """A series of interlaced sub-tomograms acquired one after another.

    This is the acquisition geometry the method needs, and the one a P06 series actually
    has: the sample is swept through 0-180 several times over many hours, so a *contiguous
    block of acquisition time* is one whole sub-tomogram -- angularly complete, and a
    single deformation state. Angle order is emphatically not time order here.

    A single sequential 0-180 sweep is NOT offered, because a time-varying deformation is
    not identifiable from one: every acquisition-time block is then an angular wedge, and
    a partial reconstruction of a wedge is elongated in a way optical flow will happily
    call deformation.
    """
    rng = np.random.default_rng(config.seed)
    nz, nx = config.n_v, config.size
    volume = phantom(nz, nx, seed=config.seed)
    truth_backend = ScipyParallelBackend(order=3)  # the data is made at order 3 ...
    grid = DeformationField.grid_for((nz, nx, nx), config.grid_spacing)

    projections, angles, acquisition, times = [], [], [], []
    for k in range(config.n_subtomos):
        t = k / max(1, config.n_subtomos - 1)
        state = (
            warp_volume(
                volume,
                truth_field(
                    t, (nz, nx, nx), grid, config.deformation_px, config.local_fraction
                ),
                order=3,
            )
            if deform
            else volume
        )
        theta = np.linspace(0.0, np.pi, config.angles_per, endpoint=False) + k * np.pi / (
            config.angles_per * config.n_subtomos
        )
        projections.append(truth_backend.reproject(state, theta))
        angles.append(theta)
        acquisition.append(np.arange(config.angles_per) + k * config.angles_per)
        times.append(t)

    stack = np.concatenate(projections).astype(np.float32)
    angles = np.concatenate(angles)
    acquisition = np.concatenate(acquisition)

    truth_dy = np.zeros(stack.shape[0])
    truth_dx = np.zeros(stack.shape[0])
    if config.jitter_px > 0:
        truth_dy = rng.normal(0.0, config.jitter_px, stack.shape[0])
        truth_dx = rng.normal(0.0, config.jitter_px, stack.shape[0])
        stack = np.stack(
            [
                ndshift(stack[i], (truth_dy[i], truth_dx[i]), order=3, mode="nearest")
                for i in range(stack.shape[0])
            ]
        ).astype(np.float32)
    if config.noise_rms > 0:
        stack = (stack + config.noise_rms * float(stack.std()) * rng.standard_normal(stack.shape)).astype(
            np.float32
        )

    return {
        "projections": stack,
        "angles": angles,
        "acquisition_index": acquisition,
        "volume": volume,
        "center": nx / 2.0,
        "grid": grid,
        "shape": (nz, nx, nx),
        "subtomo_times": np.asarray(times),
        "truth_dy": truth_dy,
        "truth_dx": truth_dx,
        "deformed": deform,
    }


def truth_sequence(config: ScenarioConfig, scan: dict[str, Any]):
    """The injected deformation sampled at the acquisition times the aligner will use."""
    span = config.n_angles - 1
    subset_times = np.array(
        [
            float(np.mean(scan["acquisition_index"][k * config.angles_per : (k + 1) * config.angles_per]))
            for k in range(config.n_subtomos)
        ]
    )
    fields = tuple(
        truth_field(
            float(t) / span, scan["shape"], scan["grid"], config.deformation_px,
            config.local_fraction,
        )
        for t in subset_times
    )
    return DeformationSequence(fields, subset_times)


# ---------------------------------------------------------------------------------
# The rigid stage: a compact joint reprojection loop, numpy + scipy only
# ---------------------------------------------------------------------------------


def _lucas_kanade_shift(
    measured: np.ndarray, simulated: np.ndarray, *, iterations: int = 3
) -> tuple[float, float]:
    """The ``(dv, du)`` that maps ``simulated`` onto ``measured``, to sub-pixel accuracy.

    Gradient-based rather than phase correlation, so this file needs no scikit-image.
    Re-linearised around the current estimate each iteration, which is what makes it
    accurate beyond a fraction of a pixel.

    THE SIGN, which is the whole function. ``ndshift(f, d)[p] = f[p - d]``, so content
    moves by ``+d``. Expanding ``ndshift(simulated, d + delta) ~ warped - delta . grad
    warped`` gives ``measured - warped ~ -delta . grad warped``: the least-squares
    coefficient is **minus** the shift increment. Adding it instead of subtracting it
    walks the alignment away from the answer at exactly the rate it should walk towards
    it, and the residual grows smoothly and plausibly -- which is why
    :func:`rigid_loop` refuses to return when the residual has not fallen.
    """
    dv = du = 0.0
    for _ in range(iterations):
        warped = ndshift(simulated, (dv, du), order=1, mode="nearest")
        g_v, g_u = np.gradient(warped)
        residual = (measured - warped).ravel()
        design = np.column_stack([g_v.ravel(), g_u.ravel()])
        gram = design.T @ design
        if not np.all(np.isfinite(gram)) or np.linalg.det(gram) <= 1e-12:
            break
        step = np.linalg.solve(gram, design.T @ residual)
        dv -= float(step[0])
        du -= float(step[1])
    return dv, du


def rigid_loop(
    projections: np.ndarray,
    angles: np.ndarray,
    *,
    center: float,
    backend: ScipyParallelBackend,
    iterations: int = 6,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Joint iterative reprojection alignment: the rigid stage, in 30 lines.

    The three conventions of :mod:`~tktomo.ptycho_align.core.engine` are honoured exactly,
    and each is a bug if it is not:

    1. The registration direction is ``(measured, simulated)`` -- the shift that maps the
       *simulation* onto the *measurement*, subtracted from the cumulative shift.
    2. ``(dv, du)`` is ``(row, column)``.
    3. The cumulative shift is always applied to the **pristine** stack, never to an
       already-shifted one, so no interpolation blur accumulates.

    The gauge is removed from the accumulated shifts every iteration (the mean from ``dv``;
    the constant, ``sin`` and ``cos`` modes from ``du``), because those modes are
    unobservable -- they translate the reconstructed object -- and leaving them in lets the
    solution wander without ever reducing the residual.

    Raises ``RuntimeError`` if the residual does not fall at all over the run: on a
    deformed phantom the loop must still reduce the residual substantially before
    plateauing, and a loop that never improves means a sign or an axis convention is
    wrong, which every downstream number would then quietly inherit.
    """
    projections = np.asarray(projections, dtype=np.float32)
    n_angles = projections.shape[0]
    theta = np.asarray(angles, dtype=np.float64)
    design = np.column_stack([np.ones(n_angles), np.sin(theta), np.cos(theta)])

    dv = np.zeros(n_angles)
    du = np.zeros(n_angles)
    residuals: list[float] = []
    updates: list[float] = []
    aligned = projections.copy()
    simulated = None
    volume = None

    for iteration in range(iterations):
        if progress:
            progress(f"  rigid iteration {iteration + 1}/{iterations}")
        aligned = np.stack(
            [
                ndshift(projections[i], (dv[i], du[i]), order=1, mode="nearest")
                for i in range(n_angles)
            ]
        ).astype(np.float32)
        volume = backend.reconstruct(aligned, theta, center=center, algorithm="fbp")
        simulated = backend.reproject(volume, theta, center=center)

        step_v = np.zeros(n_angles)
        step_u = np.zeros(n_angles)
        for i in range(n_angles):
            sv, su = _lucas_kanade_shift(aligned[i], simulated[i])
            step_v[i], step_u[i] = -sv, -su
        dv += step_v
        du += step_u
        dv -= dv.mean()
        du -= design @ np.linalg.lstsq(design, du, rcond=None)[0]

        residuals.append(_relative_residual(aligned, simulated))
        updates.append(float(np.sqrt(np.mean(step_v**2 + step_u**2))))

    if len(residuals) > 1 and residuals[-1] >= residuals[0]:
        raise RuntimeError(
            f"the rigid loop did not reduce the residual at all ({residuals[0]:.4f} -> "
            f"{residuals[-1]:.4f}). That is a broken convention (registration direction or "
            "row/column order), not a hard dataset, and every number downstream of it "
            "would be wrong in a plausible-looking way."
        )

    return {
        "dv": dv, "du": du, "aligned": aligned, "simulated": simulated, "volume": volume,
        "residuals": residuals, "updates": updates,
    }


def _relative_residual(measured: np.ndarray, simulated: np.ndarray) -> float:
    """``||measured - simulated|| / ||measured||`` over the whole stack."""
    difference = np.ascontiguousarray((measured - simulated).astype(np.float64)).ravel()
    scale = np.ascontiguousarray(measured.astype(np.float64)).ravel()
    denominator = float(np.linalg.norm(scale))
    if denominator <= 0:
        return math.nan
    return float(np.linalg.norm(difference) / denominator)


# ---------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------


def _split_half(n: int) -> tuple[np.ndarray, np.ndarray]:
    if HAVE_BENCH_METRICS:
        return _bench_split(n)
    index = np.arange(n)
    return index[0::2], index[1::2]


def time_blocks(acquisition_index: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    """Contiguous blocks of acquisition time -- one sub-tomogram each, in this geometry."""
    order = np.argsort(acquisition_index, kind="stable")
    return [np.sort(part) for part in np.array_split(order, n_blocks)]


def half_volumes(
    backend: ScipyParallelBackend,
    projections: np.ndarray,
    angles: np.ndarray,
    acquisition_index: np.ndarray,
    *,
    center: float,
    blocks: list[np.ndarray],
    deformation: Callable[[float], Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-data half volumes, with the SAME partition for the rigid and non-rigid rows.

    The split is even/odd *in angle within each acquisition-time block*, not first-half /
    second-half of the scan: interleaving is what makes the two half-sets cover the same
    angular range, and splitting by time would give two limited-angle reconstructions
    whose FSC measures the missing wedge rather than the resolution.

    ``deformation`` is a callable from acquisition time to a
    :class:`~tktomo.ptycho_align.core.deformation.DeformationField`. When it is given,
    each block's volume is carried back into the common frame through the **inverse** of
    its own field before the blocks are averaged, following the aligner's own convention
    (``warp_volume(reference, u_k) ~= partial_k``, so ``warp_volume(partial_k,
    invert(u_k)) ~= reference``). Without that step a split-half FSC of a moving sample
    measures how much the sample moved between blocks, which is precisely the quantity
    the non-rigid stage is supposed to remove -- so the rigid and non-rigid rows must
    differ ONLY by this correction, or the comparison is between two different metrics.
    """
    halves: list[list[np.ndarray]] = [[], []]
    for indices in blocks:
        ordered = indices[np.argsort(angles[indices])]
        field = None
        if deformation is not None:
            field = invert(deformation(float(np.mean(acquisition_index[indices]))))
        for half, selection in enumerate((ordered[0::2], ordered[1::2])):
            volume = backend.reconstruct(
                projections[selection], angles[selection], center=center, algorithm="fbp"
            )
            if field is not None:
                volume = warp_volume(volume, field, order=1)
            halves[half].append(volume)
    return (
        np.mean(np.stack(halves[0]), axis=0),
        np.mean(np.stack(halves[1]), axis=0),
    )


def _fsc_resolution_px(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, str]:
    """Split-data FSC resolution in pixels, and the criterion that produced it.

    Three criteria are tried in order -- half-bit, 0.143, 0.5 -- and the first that the
    curve actually crosses is reported, with its name. That is not criterion shopping: on
    a *noiseless* synthetic the two half-sets differ only by their angular sampling, the
    FSC stays above 0.9 out to Nyquist, and half-bit returns ``inf`` for every row, which
    is a column of nothing rather than a measurement. Reporting which criterion answered
    keeps the number honest, and rows are only comparable when their criteria match --
    the table prints it for exactly that reason. Adding a little ``--noise`` is the better
    fix and makes half-bit meaningful again.

    Uses ``benchmarks.metrics.fourier_shell_correlation`` when available so the number is
    the same one the rest of the harness reports; the fallback is a plain shell average.

    FSC is not evidence on its own. It is exactly invariant to any geometry error applied
    identically to both half-sets, so it will certify a systematically wrong volume; that
    is why every row here also carries a reprojection residual.
    """
    if HAVE_BENCH_METRICS:
        for criterion in ("half-bit", "0.143", "0.5"):
            value = float(_bench_fsc(a, b, criterion=criterion, mask=mask).resolution_px)
            if math.isfinite(value):
                return value, criterion
        return math.inf, "half-bit"

    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    if mask is not None:
        a64, b64 = a64 * mask, b64 * mask
    fa, fb = np.fft.fftn(a64), np.fft.fftn(b64)
    axes = np.meshgrid(*[np.fft.fftfreq(n) for n in a64.shape], indexing="ij")
    radius = np.sqrt(sum(axis**2 for axis in axes))
    n_shells = min(a64.shape) // 2
    edges = np.linspace(0.0, 0.5, n_shells + 1)
    index = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, n_shells - 1)
    cross = np.bincount(index, weights=(fa * np.conj(fb)).real.ravel(), minlength=n_shells)
    power_a = np.bincount(index, weights=(np.abs(fa) ** 2).ravel(), minlength=n_shells)
    power_b = np.bincount(index, weights=(np.abs(fb) ** 2).ravel(), minlength=n_shells)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = cross / np.sqrt(power_a * power_b)
    frequency = 0.5 * (edges[:-1] + edges[1:])
    for threshold, name in ((0.143, "0.143"), (0.5, "0.5")):
        below = np.nonzero(curve < threshold)[0]
        if below.size and frequency[below[0]] > 0:
            return float(1.0 / frequency[below[0]]), name
    return math.inf, "0.143"


def _nrmse(volume: np.ndarray, reference: np.ndarray) -> float:
    """RMS difference from the truth volume, normalised by the truth's own RMS."""
    a = np.asarray(volume, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    scale = float(np.sqrt(np.mean(b**2)))
    if scale <= 0:
        return math.nan
    return float(np.sqrt(np.mean((a - b) ** 2)) / scale)


@dataclass
class StageResult:
    """One row of the scenario table."""

    name: str
    deformed: bool
    residual: float
    residual_holdout: float = math.nan
    #: The aligner's OWN gains, baseline-to-final. Not comparable with the rigid row's
    #: residual: the held-out projections are excluded from the reconstruction that
    #: predicts them, so their residual is higher by construction and only the
    #: before/after comparison within the aligner means anything.
    fitted_gain: float = math.nan
    holdout_gain: float = math.nan
    dvf_rms_px: float = math.nan
    dvf_error_px: float = math.nan
    dvf_error_px_all: float = math.nan
    dvf_correlation: float = math.nan
    dvf_amplitude: float = math.nan
    fsc_px: float = math.nan
    fsc_criterion: str = ""
    volume_nrmse: float = math.nan
    wallclock_s: float = math.nan
    iterations: int = 0
    #: Why the aligner's own guard called this iteration overfitting, if it did. On the
    #: negative control this is the thing that is SUPPOSED to be non-empty.
    overfitting: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in asdict(self).items()}


@dataclass
class ScenarioReport:
    config: ScenarioConfig
    rows: list[StageResult]
    gate: Any | None = None
    control_gate: Any | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    verdicts: list[str] = field(default_factory=list)

    def row(self, name: str) -> StageResult:
        for row in self.rows:
            if row.name == name:
                return row
        raise KeyError(f"no row named {name!r}; have {[r.name for r in self.rows]}")

    def table(self) -> str:
        header = (
            f"{'row':28s} {'def':4s} {'resid':>8s} {'held':>8s} {'DVF err':>8s} "
            f"{'DVF rms':>8s} {'corr':>6s} {'FSC px':>7s} {'crit':>8s} {'NRMSE':>7s} {'s':>6s}"
        )
        lines = [header, "-" * len(header)]
        for row in self.rows:
            def fmt(value: float, spec: str) -> str:
                return f"{value:{spec}}" if math.isfinite(value) else "     -- "
            lines.append(
                f"{row.name:28s} {'yes' if row.deformed else 'no':4s} "
                f"{fmt(row.residual, '8.4f')} {fmt(row.residual_holdout, '8.4f')} "
                f"{fmt(row.dvf_error_px, '8.3f')} {fmt(row.dvf_rms_px, '8.3f')} "
                f"{fmt(row.dvf_correlation, '6.2f')} {fmt(row.fsc_px, '7.2f')} "
                f"{row.fsc_criterion or '--':>8s} "
                f"{fmt(row.volume_nrmse, '7.3f')} {fmt(row.wallclock_s, '6.1f')}"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        return "\n".join(self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "environment": self.environment,
            "rows": [row.to_dict() for row in self.rows],
            "gate": self.gate.to_dict() if self.gate is not None else None,
            "control_gate": self.control_gate.to_dict() if self.control_gate is not None else None,
            "verdicts": self.verdicts,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------------
# The scenario
# ---------------------------------------------------------------------------------


def _dvf_scores(recovered, truth, mask) -> dict[str, float]:
    """Gauge-fixed comparison of two deformation sequences, inside the object support.

    The time-averaged field is unobservable, so both are zero-meaned first (that is what
    :func:`sequence_rms_difference` does). The whole-grid number is reported alongside the
    in-support one and never instead of it: flow in empty air is set by the smoothness
    term alone, so a whole-grid RMS mostly measures how far the regulariser extrapolated.
    """
    error_support = sequence_rms_difference(recovered, truth, mask)
    error_all = sequence_rms_difference(recovered, truth)
    a = recovered.zero_mean().node_array  # (K, 3, gz, gy, gx)
    b = truth.zero_mean().node_array
    if mask is not None:
        a = a[:, :, mask]
        b = b[:, :, mask]
    a_flat, b_flat = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denominator = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    correlation = float(a_flat @ b_flat / denominator) if denominator > 0 else math.nan
    truth_rms = float(np.sqrt(np.mean(b_flat**2)))
    amplitude = (
        float(np.sqrt(np.mean(a_flat**2)) / truth_rms) if truth_rms > 0 else math.nan
    )
    return {
        "dvf_error_px": error_support,
        "dvf_error_px_all": error_all,
        "dvf_correlation": correlation,
        "dvf_amplitude": amplitude,
        "truth_rms_px": truth_rms,
    }


def _run_arm(
    config: ScenarioConfig, *, deform: bool, backend: ScipyParallelBackend,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    """One arm of the scenario: build the scan, align rigidly, gate, align non-rigidly."""
    label = "deformed" if deform else "CONTROL (no deformation)"
    if progress:
        progress(f"[{label}] building the scan")
    scan = build_scan(config, deform=deform)
    rows: list[StageResult] = []

    # -- row 0: no alignment at all, for scale
    started = time.perf_counter()
    volume0 = backend.reconstruct(
        scan["projections"], scan["angles"], center=scan["center"], algorithm="fbp"
    )
    simulated0 = backend.reproject(volume0, scan["angles"], center=scan["center"])
    rows.append(
        StageResult(
            name="no alignment", deformed=deform,
            residual=_relative_residual(scan["projections"], simulated0),
            volume_nrmse=_nrmse(volume0, scan["volume"]),
            wallclock_s=time.perf_counter() - started,
            note="reference point only; with no jitter injected this is already close to the rigid floor",
        )
    )

    # -- row 1: the rigid loop, to its plateau
    started = time.perf_counter()
    rigid = rigid_loop(
        scan["projections"], scan["angles"], center=scan["center"], backend=backend,
        iterations=config.rigid_iterations, progress=progress,
    )
    blocks = time_blocks(scan["acquisition_index"], config.n_subtomos)
    fsc_rigid, fsc_rigid_criterion = _fsc_resolution_px(
        *half_volumes(
            backend, rigid["aligned"], scan["angles"], scan["acquisition_index"],
            center=scan["center"], blocks=blocks,
        )
    )
    rows.append(
        StageResult(
            name="rigid (to plateau)", deformed=deform, residual=rigid["residuals"][-1],
            fsc_px=fsc_rigid, fsc_criterion=fsc_rigid_criterion,
            volume_nrmse=_nrmse(rigid["volume"], scan["volume"]),
            wallclock_s=time.perf_counter() - started,
            note=f"residual history {' -> '.join(f'{r:.4f}' for r in rigid['residuals'])}",
        )
    )

    # -- the gate, on exactly what the rigid stage produced
    verdict = None
    if HAVE_GATE:
        if progress:
            progress(f"[{label}] evaluating the non-rigid decision gate")
        # block=0 lets the gate size its own blocks from the frame; picking one here from
        # the volume side gave 8 blocks on a 32 x 64 frame, where "the hottest 10%" is a
        # single block and the concentration statistic means nothing.
        verdict = evaluate_gate(
            residual_history=rigid["residuals"], shift_history=rigid["updates"],
            measured=rigid["aligned"], simulated=rigid["simulated"],
            angles=scan["angles"], acquisition_index=scan["acquisition_index"],
            config=GateConfig(n_null=24),
        )

    # -- row 2: the non-rigid stage
    nonrigid_row = None
    if HAVE_NONRIGID:
        if progress:
            progress(f"[{label}] non-rigid alignment")
        started = time.perf_counter()
        aligner = NonRigidAligner(
            projections=rigid["aligned"], angles=scan["angles"],
            acquisition_index=scan["acquisition_index"], center=scan["center"],
            config=NonRigidConfig(
                n_subsets=config.n_subtomos, grid_spacing=config.grid_spacing,
                recon_algorithm="fbp", max_angular_gap_deg=30.0,
            ),
            backend=backend,
        )
        results = aligner.run(config.nonrigid_iterations)
        elapsed = time.perf_counter() - started
        if not results:
            nonrigid_row = StageResult(
                name="non-rigid", deformed=deform, residual=math.nan,
                wallclock_s=elapsed, note="the aligner produced no iterations",
            )
        else:
            last = results[-1]
            mask = coarse_support_mask(scan["volume"], scan["grid"])
            scores = _dvf_scores(last.sequence, truth_sequence(config, scan), mask)
            simulated = aligner.simulate()
            try:
                fsc_nonrigid, fsc_nonrigid_criterion = _fsc_resolution_px(
                    *half_volumes(
                        backend, rigid["aligned"], scan["angles"], scan["acquisition_index"],
                        center=scan["center"], blocks=blocks,
                        deformation=aligner.deformation_at,
                    )
                )
            except ValueError as exc:
                # invert() refuses a folding field rather than returning the last iterate.
                logger.warning("deformation-corrected FSC skipped: %s", exc)
                fsc_nonrigid, fsc_nonrigid_criterion = math.nan, ""
            nonrigid_row = StageResult(
                name="non-rigid", deformed=deform, residual=last.residual,
                iterations=len(results), overfitting=last.overfitting or "",
                fitted_gain=last.fitted_gain, holdout_gain=last.holdout_gain,
                residual_holdout=last.holdout_residual, dvf_rms_px=last.dvf_rms_px,
                dvf_error_px=scores["dvf_error_px"], dvf_error_px_all=scores["dvf_error_px_all"],
                dvf_correlation=scores["dvf_correlation"], dvf_amplitude=scores["dvf_amplitude"],
                fsc_px=fsc_nonrigid, fsc_criterion=fsc_nonrigid_criterion,
                volume_nrmse=_nrmse(aligner.reference_volume, scan["volume"]),
                wallclock_s=elapsed,
                note=(
                    f"{len(results)} iteration(s); fitted gain {last.fitted_gain:+.1%}, "
                    f"held-out gain {last.holdout_gain:+.1%}"
                    + (f"; STOPPED: {last.overfitting}" if last.overfitting else "")
                ),
            )
            nonrigid_row.note += f"; truth DVF {scores['truth_rms_px']:.3f} px rms (gauge-fixed)"
            # The residual maps the aligner itself would be judged on, kept for the caller.
            scan["nonrigid_simulated"] = simulated
        rows.append(nonrigid_row)
    else:  # pragma: no cover - only when the core module is absent
        rows.append(
            StageResult(
                name="non-rigid", deformed=deform, residual=math.nan,
                note=f"SKIPPED: {NONRIGID_IMPORT_ERROR}",
            )
        )

    return {"scan": scan, "rigid": rigid, "rows": rows, "gate": verdict}


def run_scenario(
    config: ScenarioConfig | None = None,
    *,
    control: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ScenarioReport:
    """Run the scenario and return the report. ``control=False`` skips the negative control.

    Skipping the control is offered for a quick look and is never the right thing to
    publish: without it the report shows a method improving a residual, which a non-rigid
    method does on any data at all.
    """
    config = config or ScenarioConfig()
    if not HAVE_NONRIGID:
        logger.warning(
            "tktomo.ptycho_align.core.nonrigid is not importable (%s). The rigid rows will "
            "still run; the non-rigid rows will be reported as skipped.", NONRIGID_IMPORT_ERROR
        )
    backend = ScipyParallelBackend(order=1)  # ... and reconstructed at order 1

    rows: list[StageResult] = []
    deformed = _run_arm(config, deform=True, backend=backend, progress=progress)
    rows.extend(deformed["rows"])

    control_arm = None
    if control:
        control_arm = _run_arm(config, deform=False, backend=backend, progress=progress)
        for row in control_arm["rows"]:
            row.name = f"CONTROL {row.name}"
        rows.extend(control_arm["rows"])

    report = ScenarioReport(
        config=config, rows=rows,
        gate=deformed["gate"],
        control_gate=control_arm["gate"] if control_arm else None,
        environment={
            "benchmarks_metrics": HAVE_BENCH_METRICS,
            "benchmarks_phantom": HAVE_BENCH_PHANTOM,
            "nonrigid_core": HAVE_NONRIGID,
            "nonrigid_gate": HAVE_GATE,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    )
    report.verdicts = _verdicts(report, config)
    return report


def _verdicts(report: ScenarioReport, config: ScenarioConfig) -> list[str]:
    """The three sentences the scenario exists to produce. The third is the important one."""
    lines: list[str] = []

    def get(name: str) -> StageResult | None:
        try:
            return report.row(name)
        except KeyError:
            return None

    rigid = get("rigid (to plateau)")
    nonrigid = get("non-rigid")
    control_rigid = get("CONTROL rigid (to plateau)")
    control_nonrigid = get("CONTROL non-rigid")

    if rigid is not None and control_rigid is not None:
        gap = rigid.residual - control_rigid.residual
        lines.append(
            f"1. RIGID FLOOR. On deformed data the rigid loop plateaus at "
            f"{rigid.residual:.4f}; on the identical undeformed phantom the same loop reaches "
            f"{control_rigid.residual:.4f}. The gap of {gap:+.4f} "
            f"({gap / max(control_rigid.residual, 1e-9):+.1%} of the floor) is what no rigid "
            "model can remove, because no per-projection rigid transform satisfies all the "
            "projections at once."
        )
    elif rigid is not None:
        lines.append(f"1. RIGID FLOOR. The rigid loop plateaus at {rigid.residual:.4f}.")

    if nonrigid is not None and rigid is not None and math.isfinite(nonrigid.residual):
        gap = rigid.residual - (control_rigid.residual if control_rigid else 0.0)
        # Only quote "closed X% of the gap" when there IS a gap. Below a few tenths of a
        # percent of the floor the ratio is a division by the difference of two noisy
        # numbers and produces figures like -423491467%, which is not a measurement.
        closed_text = (
            f"closing {(rigid.residual - nonrigid.residual) / gap:.0%} of the gap above. "
            if control_rigid is not None and gap > 0.002 * control_rigid.residual
            else "with no rigid-floor gap to close (the deformation leaves less than 0.2% "
            "of the floor unexplained, so the ratio would be noise over noise). "
        )
        lines.append(
            f"2. NON-RIGID. Residual {rigid.residual:.4f} -> {nonrigid.residual:.4f} fitted, "
            + closed_text
            + "Against the aligner's own baseline the "
            f"gains are {nonrigid.fitted_gain:+.1%} fitted and {nonrigid.holdout_gain:+.1%} "
            "HELD OUT -- the second is the one that decides whether the deformation is real, "
            "and the two must move together. The recovered deformation field matches the "
            "known truth to "
            f"{nonrigid.dvf_error_px:.3f} px rms inside the object support "
            f"({nonrigid.dvf_error_px_all:.3f} px over the whole grid, where the flow is "
            f"unconstrained by data), correlation {nonrigid.dvf_correlation:.2f} at "
            f"{nonrigid.dvf_amplitude:.2f} of the true amplitude."
        )

    if config.deformation_px == 0.0:
        lines.append(
            "NOTE: deformation_px is 0, so the 'deformed' and 'control' arms are the SAME "
            "dataset and rows 1-3 are duplicates. That is the right configuration for asking "
            "whether the gate refuses a scan with nothing in it, and the wrong one for "
            "reading a rigid floor off."
        )
    if control_nonrigid is not None and control_rigid is not None:
        invented = control_nonrigid.dvf_rms_px
        moved = control_nonrigid.residual - control_rigid.residual
        # The control PASSES if the method DECLINES the data: either its own guard fired
        # and `run()` stopped, or it produced no confident improvement. It FAILS only if
        # it reports an unflagged gain on data with nothing in it -- that, and not a small
        # invented field, is the failure mode that would put a fabricated deformation in a
        # paper. A flagged stop is the correct outcome, not a near miss.
        declined = bool(control_nonrigid.overfitting)
        gained = control_nonrigid.holdout_gain
        verdict = "PASS" if declined or (moved <= 1e-3 and not (gained > 0.02)) else "FAIL"
        lines.append(
            f"3. NEGATIVE CONTROL [{verdict}] -- the row that makes the other two mean "
            f"something. With NO deformation in the data the method ran "
            f"{control_nonrigid.iterations} iteration(s), invented {invented:.3f} px rms of "
            f"field, moved the residual by {moved:+.4f} "
            f"({control_rigid.residual:.4f} -> {control_nonrigid.residual:.4f}) and its "
            f"held-out gain was {gained:+.1%}"
            + (
                f", against {nonrigid.holdout_gain:+.1%} for the real case. "
                if nonrigid is not None
                else ". "
            )
            + (
                f"ITS OWN GUARD STOPPED IT: {control_nonrigid.overfitting}"
                if declined
                else "Its guard did not fire."
            )
        )
    elif control_nonrigid is None:
        lines.append(
            "3. NEGATIVE CONTROL: NOT RUN. Rows 1 and 2 alone are a demonstration, not a "
            "benchmark -- a non-rigid model improves the fitted residual on any data at all, "
            "including data with nothing in it. Re-run with control=True before quoting them."
        )

    if report.gate is not None:
        lines.append(f"GATE (deformed data): {report.gate.recommendation.value.upper()} -- "
                     f"{report.gate.headline}")
    if report.control_gate is not None:
        lines.append(f"GATE (control): {report.control_gate.recommendation.value.upper()} -- "
                     f"{report.control_gate.headline}")
    return lines


def _gain(baseline: float, value: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(value):
        return math.nan
    return (baseline - value) / baseline


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The non-rigid benchmark scenario: the rigid floor, the non-rigid "
        "answer, and the negative control.",
    )
    parser.add_argument("--size", type=int, default=32, help="detector width / volume side (px)")
    parser.add_argument("--slices", type=int, default=0, help="detector rows; 0 -> size // 2")
    parser.add_argument("--subtomos", type=int, default=4, help="acquisition-time blocks")
    parser.add_argument("--angles", type=int, default=30, help="projections per sub-tomogram")
    parser.add_argument("--deformation", type=float, default=2.0, help="peak deformation (px)")
    parser.add_argument(
        "--local-fraction", type=float, default=0.5,
        help="0 = purely global deformation (which the gate vetoes, by design), "
             "1 = purely local (beam damage). See truth_field().",
    )
    parser.add_argument("--jitter", type=float, default=0.0, help="rigid jitter injected (px rms)")
    parser.add_argument("--noise", type=float, default=0.02, help="noise, fraction of proj. std")
    parser.add_argument("--rigid-iterations", type=int, default=6)
    parser.add_argument("--nonrigid-iterations", type=int, default=4)
    parser.add_argument("--grid-spacing", type=float, default=10.0, help="DVF node spacing (voxels)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-control", action="store_true",
        help="skip the negative control (never the right thing to publish)",
    )
    parser.add_argument(
        "--out", default=None,
        help="write the report as JSON to this path. No default, and never inside the "
             "repository: results belong beside the data, not in the source tree.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = ScenarioConfig(
        size=args.size, n_slices=args.slices, n_subtomos=args.subtomos,
        angles_per=args.angles, deformation_px=args.deformation, jitter_px=args.jitter,
        local_fraction=args.local_fraction,
        noise_rms=args.noise, rigid_iterations=args.rigid_iterations,
        nonrigid_iterations=args.nonrigid_iterations, grid_spacing=args.grid_spacing,
        seed=args.seed,
    )
    progress = None if args.quiet else (lambda message: print(message, flush=True))
    report = run_scenario(config, control=not args.no_control, progress=progress)

    print()
    print(report.table())
    print()
    print(report.summary())
    if report.gate is not None and HAVE_GATE and not args.quiet:
        print()
        print(format_gate(report.gate))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report.to_json())
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
