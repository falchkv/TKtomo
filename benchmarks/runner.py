"""Drive any aligner through one interface, score it against truth, tabulate.

The harness has to be able to run three very different things -- TKtomo's incumbent
:class:`~tktomo.ptycho_align.core.engine.AlignmentEngine` (Gursoy joint iterative
reprojection), the new Odstrcil vertical-mass + phase-gradient aligner, and the
ported joint-gradient-descent aligner -- and it has to keep working when one of them
does not exist yet. So every aligner is wrapped in an adapter that returns an
:class:`AlignerResult` with a ``status``, and a missing module produces
``status="skipped"`` with a message rather than an exception. A late module never
takes the harness down with it.

Two reference aligners bracket every table and are not optional:

* ``null`` returns zeros. Its score IS the injected misalignment, so any real
  aligner that does not beat it is doing harm.
* ``oracle`` returns the ground truth. Its score must be ~0. If it is not, the
  *scorer* is broken -- wrong gauge, wrong sign, wrong shape -- and no other row in
  the table means anything. Read the oracle row first, always.

Reconstruction inside the loop is done by :class:`NumpyProjectorBackend`, a
numpy+scipy SIRT/FBP projector registered into TKtomo's backend registry under the
name ``benchmark-numpy``. It exists so the benchmark runs with no tomopy, no astra
and no GPU -- not because it is fast. Point the config at ``"tomopy"`` when tomopy
is installed and you want the incumbent measured against its real backend.

**tomopy shim.** ``AlignmentEngine.step`` calls ``tomopy.prep.alignment.shift_images``
and ``blur_edges``; without tomopy the incumbent cannot run at all, and there would
be no baseline to compare the new aligners against. :func:`tomopy_shim` therefore
installs a scipy re-implementation of exactly those two helpers into ``sys.modules``
**for the duration of the run and only when tomopy is genuinely absent**, and every
report says so in ``environment.tomopy_shim``. It is a crutch, it is declared, and
it is removed on exit so it can never leak into another test.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import logging
import math
import platform
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

from benchmarks import metrics
from benchmarks.phantom import (
    BenchmarkCase,
    PerturbationSpec,
    back_project,
    circular_mask,
    forward_project,
    load_angles,
    load_volume,
    synthetic_case,
    volume_case,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AlignerResult",
    "BenchmarkReport",
    "EngineAligner",
    "JirrAligner",
    "JointGdAligner",
    "ModuleAligner",
    "NullAligner",
    "NumpyProjectorBackend",
    "OdstrcilAligner",
    "OracleAligner",
    "comparison_figure",
    "default_aligners",
    "run_benchmark",
    "tomopy_shim",
    "undo_shifts",
]

BACKEND_NAME = "benchmark-numpy"


# --------------------------------------------------------------------------------
# Applying an estimate
# --------------------------------------------------------------------------------


def undo_shifts(stack: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Apply an aligner's estimate to a stack: move content by ``(-sy, -sx)``.

    The same sign as :func:`~tktomo.ptycho_align.core.engine.apply_shifts` (which the
    repo's own test pins: ``apply_shifts(frame, sy=3)`` takes row 10 to row 7), and
    therefore the exact inverse of the injection in :mod:`benchmarks.phantom`. Done
    with a Fourier shift so that a *perfect* estimate reproduces the clean stack to
    machine precision, which is what makes the FSC and residual secondaries
    interpretable: any residual you see is the aligner's, not the interpolator's.
    """
    from scipy.ndimage import fourier_shift  # noqa: PLC0415

    stack = np.asarray(stack, dtype=np.float32)
    sy = np.asarray(sy, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)
    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        spectrum = np.fft.fftn(stack[i])
        out[i] = np.fft.ifftn(fourier_shift(spectrum, (-sy[i], -sx[i]))).real
    return out


def default_center(case: BenchmarkCase) -> float:
    """The true rotation-axis column of the case, ``(width - 1) / 2``.

    Handed to every aligner so none of them is penalised for a centre-finding step
    the benchmark is not measuring. Note this is *not* tomopy's ``width / 2`` default
    -- see the geometry note in :mod:`benchmarks.phantom`.
    """
    return case.center


# --------------------------------------------------------------------------------
# A numpy + scipy reconstruction backend, so the loop runs anywhere
# --------------------------------------------------------------------------------


class NumpyProjectorBackend:
    """SIRT / FBP / BP on ``scipy.ndimage.rotate``. Correct, portable, and slow.

    Registered into TKtomo's backend registry so ``AlignConfig(backend=...)`` can
    select it. Implements the same ``reconstruct`` / ``reproject`` protocol as the
    TomoPy backend, including the keyword arguments the engine passes
    (``algorithm``, ``center``, ``num_iter``, ``init_recon``, ``ncore``); ``ncore`` is
    accepted and ignored, because this is single-threaded by construction.

    Why SIRT is the default and gridrec is absent: with a limited angular range
    gridrec fails outright, and an FBP initialisation converges *slower* in a joint
    alignment loop because its streak artifacts feed straight back into the shift
    estimate. SIRT and MLEM are the robust choices. ``"fbp"`` is offered so that
    claim can be measured rather than asserted.
    """

    name = BACKEND_NAME

    def __init__(self, *, order: int = 1, relaxation: float = 1.0, mask: bool = True) -> None:
        self.order = order
        self.relaxation = relaxation
        self.mask = mask
        self._norm_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}

    # -- the protocol ---------------------------------------------------------------

    def reconstruct(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        *,
        algorithm: str = "sirt",
        center: float | None = None,
        num_iter: int = 2,
        init_recon: np.ndarray | None = None,
        ncore: int | None = None,  # noqa: ARG002 - accepted for protocol compatibility
        filter_name: str = "hann",
        **kwargs: Any,
    ) -> np.ndarray:
        if kwargs:
            logger.debug("%s ignoring unsupported kwargs %s", self.name, sorted(kwargs))
        projections = np.asarray(projections, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float64)

        if algorithm in {"bp", "fbp"}:
            sino = projections
            if algorithm == "fbp":
                sino = _ramp_filter(sino, filter_name)
            volume = back_project(sino, angles, center=center, order=self.order)
            volume *= math.pi / max(angles.size, 1)
            return self._masked(volume)
        if algorithm == "gridrec":
            raise ValueError(
                "The benchmark backend has no gridrec. That is deliberate: gridrec "
                "fails outright on a limited angular range, which is the regime this "
                "harness exists to measure. Use 'sirt' (default), 'fbp' or 'bp'."
            )
        if algorithm != "sirt":
            raise ValueError(
                f"Unknown algorithm {algorithm!r} for {self.name}; have 'sirt', 'fbp', 'bp'."
            )
        return self._sirt(projections, angles, center, num_iter, init_recon)

    def reproject(
        self,
        volume: np.ndarray,
        angles: np.ndarray,
        *,
        center: float | None = None,
        ncore: int | None = None,  # noqa: ARG002 - accepted for protocol compatibility
        **kwargs: Any,
    ) -> np.ndarray:
        if kwargs:
            logger.debug("%s ignoring unsupported kwargs %s", self.name, sorted(kwargs))
        return forward_project(
            np.asarray(volume, dtype=np.float32),
            np.asarray(angles, dtype=np.float64),
            center=center,
            order=self.order,
        )

    # -- internals ------------------------------------------------------------------

    def _masked(self, volume: np.ndarray) -> np.ndarray:
        if not self.mask:
            return volume
        return volume * circular_mask(volume.shape[1], volume.shape[2])[None]

    def _norms(
        self, shape: tuple[int, int, int], angles: np.ndarray, center: float | None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """SIRT's row and column weights ``A(1)``, ``A^T(1)``, and the step scale. Cached.

        Computing the weights properly instead of assuming "path length = width" is
        what keeps the update stable near the edge of the field of view, where a ray
        crosses far fewer voxels than one through the middle.

        **How the weights are floored is load-bearing, and getting it wrong blows the
        reconstruction up silently.** At the rim of the field of view a ray clips the
        inscribed disc and crosses almost no voxels, so ``A(1)`` there is ~1e-3 of its
        value through the middle; a voxel just inside the rim is likewise seen by very
        few rays. Flooring those weights at a small absolute number (1e-6, the obvious
        choice) turns the division into a multiplication by ~1e6 exactly where the data
        carry no information, and the update explodes from the rim inwards. Measured on
        a 92 px benchmark frame -- 64 px of object inside a 14 px zero margin, so the
        rim annulus is wide -- the relative residual went 0.72 -> 1.99 -> 7.54 -> 27.2
        -> 93.7 at 2, 5, 10, 20 and 40 iterations. Every intermediate array stayed
        finite and plausible-looking; the alignment loop on top of it still converged,
        because phase correlation does not care about a global scale. Damping the
        relaxation only slowed it (0.5 still reached 7.0 by iteration 20), and the
        spectral radius measured 0.9989, so neither of the two obvious diagnoses was
        the actual fault.

        The fix is to treat a negligible weight as *no measurement* rather than as a
        tiny one: below ``_WEIGHT_FLOOR`` of the maximum the weight becomes infinite,
        so the update there is exactly zero. That is the honest reading -- no ray, no
        information, no update.

        The third return value is a second, independent guard. Textbook SIRT is
        unconditionally stable because ``M = C A^T R A`` has spectral radius <= 1, and
        that holds only when ``A^T`` is the exact adjoint of ``A``. Here it is not:
        ``A`` is ``scipy.ndimage.rotate`` then a sum, and the transpose of an
        interpolating gather is a scatter, not the reverse gather that
        ``rotate(-theta)`` performs. So rho is *measured* by power iteration and the
        step scaled by ``1 / rho``. With the flooring fixed this is a no-op (rho ~ 1),
        which is exactly what a guard should be.
        """
        key = (shape, angles.tobytes(), center, self.order, self.mask)
        cached = self._norm_cache.get(key)
        if cached is not None:
            return cached

        n_theta, n_z, n_u = shape
        ones_volume = self._masked(np.ones((n_z, n_u, n_u), dtype=np.float32))
        ray = _floor_weights(forward_project(ones_volume, angles, center=center, order=self.order))
        column = _floor_weights(
            back_project(
                np.ones((n_theta, n_z, n_u), dtype=np.float32),
                angles,
                center=center,
                order=self.order,
            )
        )
        rho = self._spectral_radius(shape, angles, center, ray, column)
        norms = (ray, column, rho)
        # One entry is enough: an alignment loop reuses one geometry throughout.
        self._norm_cache = {key: norms}
        return norms

    def _spectral_radius(
        self,
        shape: tuple[int, int, int],
        angles: np.ndarray,
        center: float | None,
        ray: np.ndarray,
        column: np.ndarray,
        *,
        iterations: int = 6,
    ) -> float:
        """Largest eigenvalue of ``C A^T R A`` by power iteration. See :meth:`_norms`."""
        _, n_z, n_u = shape
        vector = self._masked(
            np.random.default_rng(0).standard_normal((n_z, n_u, n_u)).astype(np.float32)
        )
        rho = 1.0
        for _ in range(iterations):
            projected = forward_project(vector, angles, center=center, order=self.order)
            updated = self._masked(
                back_project(projected / ray, angles, center=center, order=self.order) / column
            )
            norm = float(np.linalg.norm(updated))
            reference = float(np.linalg.norm(vector))
            if norm == 0.0 or reference == 0.0:
                return 1.0
            rho = norm / reference
            vector = updated / norm
        return float(rho)

    def _sirt(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        center: float | None,
        num_iter: int,
        init_recon: np.ndarray | None,
    ) -> np.ndarray:
        n_theta, n_z, n_u = projections.shape
        ray_norm, column_norm, rho = self._norms(projections.shape, angles, center)
        step = self.relaxation / max(rho, 1e-3)

        if init_recon is None:
            volume = np.zeros((n_z, n_u, n_u), dtype=np.float32)
        else:
            volume = np.array(init_recon, dtype=np.float32, copy=True)
            if volume.shape != (n_z, n_u, n_u):
                raise ValueError(
                    f"init_recon shape {volume.shape} does not match the "
                    f"({n_z}, {n_u}, {n_u}) volume this slab reconstructs into"
                )

        for _ in range(max(1, int(num_iter))):
            simulated = forward_project(volume, angles, center=center, order=self.order)
            residual = (projections - simulated) / ray_norm
            update = back_project(residual, angles, center=center, order=self.order)
            volume = volume + step * update / column_norm
            volume = self._masked(volume)
        return volume


#: Weights below this fraction of their maximum mean "no ray crossed here", and the
#: SIRT update at those samples is set to exactly zero rather than to a huge number.
#: See :meth:`NumpyProjectorBackend._norms` for what happens without it.
_WEIGHT_FLOOR = 1e-2


def _floor_weights(weights: np.ndarray) -> np.ndarray:
    """Replace negligible SIRT weights with ``inf``, so dividing by them gives zero."""
    peak = float(np.max(weights))
    if peak <= 0.0:
        return np.full_like(weights, np.inf)
    return np.where(weights > _WEIGHT_FLOOR * peak, weights, np.inf).astype(np.float32)


def _ramp_filter(sinogram: np.ndarray, filter_name: str = "hann") -> np.ndarray:
    """Ram-Lak ramp along the detector axis, optionally apodised."""
    n_theta, n_z, n_u = sinogram.shape
    padded = int(2 ** math.ceil(math.log2(max(64, 2 * n_u))))
    frequency = np.fft.rfftfreq(padded)
    ramp = 2.0 * frequency
    if filter_name == "hann":
        ramp = ramp * (0.5 + 0.5 * np.cos(2.0 * np.pi * frequency))
    elif filter_name in {"none", "ramlak", "ram-lak"}:
        pass
    else:
        raise ValueError(f"Unknown filter {filter_name!r}; have 'hann', 'ramlak', 'none'.")

    spectrum = np.fft.rfft(sinogram, n=padded, axis=2)
    filtered = np.fft.irfft(spectrum * ramp[None, None, :], n=padded, axis=2)
    return np.ascontiguousarray(filtered[:, :, :n_u]).astype(np.float32)


def register_numpy_backend(**kwargs: Any) -> NumpyProjectorBackend:
    """Register (or re-register) the numpy backend with TKtomo and return it."""
    from tktomo.recon import register_backend  # noqa: PLC0415

    backend = NumpyProjectorBackend(**kwargs)
    register_backend(backend)
    return backend


# --------------------------------------------------------------------------------
# The tomopy shim
# --------------------------------------------------------------------------------


def tomopy_is_real() -> bool:
    """True when a genuine tomopy (not our shim) is importable."""
    try:
        module = importlib.import_module("tomopy")
    except ImportError:
        return False
    return not getattr(module, "__tktomo_benchmark_shim__", False)


def _shim_shift_images(prj: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """scipy stand-in for ``tomopy.prep.alignment.shift_images``.

    The parameter names are tomopy's, and they are *the wrong way round* -- the first
    array is applied to rows and the second to columns (see the AlignmentEngine module
    docstring, which wraps this as ``apply_shifts(prj, sy, sx)``). Content moves by
    the **negative** of the argument, which is what ``tests/test_ptycho_engine.py``
    pins: ``apply_shifts(frame, sy=3)`` takes row 10 to row 7.

    Interpolation is a 5th-order spline, as in tomopy. Results will differ from real
    tomopy at the sub-0.01 px level (skimage's ``warp`` clips to the input range;
    scipy's ``shift`` does not), which is far below anything this harness resolves,
    but it is a difference and the report declares it.
    """
    from scipy.ndimage import shift as ndi_shift  # noqa: PLC0415

    prj = np.asarray(prj, dtype=np.float32)
    rows = np.atleast_1d(np.asarray(sx, dtype=np.float64))
    columns = np.atleast_1d(np.asarray(sy, dtype=np.float64))
    out = np.empty_like(prj)
    for i in range(prj.shape[0]):
        out[i] = ndi_shift(
            prj[i], (-rows[i], -columns[i]), order=5, mode="constant", cval=0.0
        )
    return out


def _shim_blur_edges(prj: np.ndarray, low: float = 0.0, high: float = 0.8) -> np.ndarray:
    """scipy stand-in for ``tomopy.prep.alignment.blur_edges``.

    A radial cosine-free linear taper: 1 inside ``low * r_max``, falling linearly to 0
    at ``high * r_max``. Applied to *copies* used only for registration -- blurring
    the data itself would destroy it -- which is the engine's own discipline, not
    something this function can enforce.
    """
    prj = np.asarray(prj, dtype=np.float32)
    _, n_v, n_u = prj.shape
    rows, columns = np.mgrid[:n_v, :n_u]
    radius = np.sqrt((rows - n_v / 2.0) ** 2 + (columns - n_u / 2.0) ** 2)
    r_max = float(radius.max())
    r_min, r_out = low * r_max, high * r_max
    mask = np.zeros((n_v, n_u), dtype=np.float32)
    mask[radius < r_min] = 1.0
    zone = (radius >= r_min) & (radius <= r_out)
    mask[zone] = ((r_out - radius[zone]) / max(r_out - r_min, 1e-9)).astype(np.float32)
    return prj * mask[None]


@contextlib.contextmanager
def tomopy_shim(*, enabled: bool = True) -> Iterator[bool]:
    """Temporarily provide ``tomopy.prep.alignment`` from scipy, if tomopy is absent.

    Yields True when the shim was installed. A no-op when a real tomopy is present or
    when ``enabled`` is False. **Always removes what it installed**, including on an
    exception: a leaked stub would make ``pytest.importorskip("tomopy")`` elsewhere in
    the suite succeed and then fail on the first ``tomopy.shepp3d``, turning a clean
    skip into a confusing error.
    """
    if not enabled or tomopy_is_real():
        yield False
        return

    saved = {name: sys.modules.get(name) for name in ("tomopy", "tomopy.prep", "tomopy.prep.alignment")}

    root = types.ModuleType("tomopy")
    root.__tktomo_benchmark_shim__ = True
    root.__version__ = "0.0-benchmark-shim"
    prep = types.ModuleType("tomopy.prep")
    alignment = types.ModuleType("tomopy.prep.alignment")
    alignment.shift_images = _shim_shift_images
    alignment.blur_edges = _shim_blur_edges
    prep.alignment = alignment
    root.prep = prep

    sys.modules["tomopy"] = root
    sys.modules["tomopy.prep"] = prep
    sys.modules["tomopy.prep.alignment"] = alignment
    try:
        yield True
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


# --------------------------------------------------------------------------------
# The aligner interface
# --------------------------------------------------------------------------------


@dataclass
class AlignerResult:
    """What one aligner produced on one case. ``status`` decides how to read it.

    ``"ok"``       -- ``sy``/``sx`` are estimates and will be scored.
    ``"skipped"``  -- the aligner is not available (module missing, optional
                      dependency absent). Not a failure of the method.
    ``"error"``    -- it ran and blew up. That IS a failure of the method, and the
                      message is the evidence.
    """

    name: str
    status: str
    message: str = ""
    sy: np.ndarray | None = None
    sx: np.ndarray | None = None
    iterations: int = 0
    wallclock_s: float = math.nan
    residual_history: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@runtime_checkable
class BenchmarkAligner(Protocol):
    """Anything the runner can drive."""

    name: str

    def run(self, case: BenchmarkCase) -> AlignerResult:  # pragma: no cover - protocol
        ...


class NullAligner:
    """Returns zeros. The floor: its score is the misalignment that was injected."""

    name = "null"

    def run(self, case: BenchmarkCase) -> AlignerResult:
        zeros = np.zeros(case.n_angles)
        return AlignerResult(
            name=self.name,
            status="ok",
            message="no alignment performed (reference floor)",
            sy=zeros,
            sx=zeros.copy(),
            iterations=0,
            wallclock_s=0.0,
        )


class OracleAligner:
    """Returns the ground truth. The ceiling, and the scorer's own self-test.

    Its gauge-removed score must be ~1e-12 px. Anything larger means the scoring path
    is wrong -- a sign flip, a gauge mismatch, an off-by-one in the angle array -- and
    every other row of the table is then meaningless. Read this row first.
    """

    name = "oracle"

    def run(self, case: BenchmarkCase) -> AlignerResult:
        return AlignerResult(
            name=self.name,
            status="ok",
            message="ground truth (upper bound; validates the scorer)",
            sy=case.truth.dy.copy(),
            sx=case.truth.dx.copy(),
            iterations=0,
            wallclock_s=0.0,
        )


def _com_prealign(case: BenchmarkCase, extras: dict[str, Any]) -> tuple[Any, Any]:
    """Centre-of-mass warm start, recording its diagnostics in ``extras``.

    Returns ``(sx0, sy0)``, or ``(None, None)`` if it failed -- a failed pre-alignment
    is a result worth recording (it means the phase offset or the mass sign is wrong),
    not a reason to abandon the run.
    """
    try:
        from tktomo.ptycho_align.core import com_prealign  # noqa: PLC0415

        result = com_prealign(case.projections, case.angles)
    except Exception as exc:  # noqa: BLE001
        extras["com_prealign_error"] = str(exc)
        return None, None
    extras["com_amplitude_px"] = float(result.amplitude)
    extras["com_fit_residual_px"] = float(result.fit_residual)
    return result.sx, result.sy


@dataclass
class EngineAligner:
    """Drives anything with the :class:`AlignmentEngine` surface: construct, ``run(n)``.

    Both series aligners in the repo present that surface --
    :class:`~tktomo.ptycho_align.core.engine.AlignmentEngine` (the incumbent) and
    :class:`~tktomo.ptycho_align.core.odstrcil.OdstrcilEngine`, which is documented as
    a drop-in for it -- so one adapter covers both and the comparison is genuinely
    like-for-like: same case, same config fields, same reconstruction backend, same
    rotation centre, same warm start.

    Notes on how the engines are driven, all of which affect the numbers they score:

    * ``row_chunk`` is set to the full detector height. The engines chunk detector
      rows so a GUI Stop can land mid-iteration; headless, that only makes this
      backend recompute its SIRT weights once per chunk.
    * ``center`` is handed the true rotation axis, so centre-finding is not part of
      what is being measured.
    * ``com_prealign`` is on by default because that is how the app drives it -- and
      because the roadmap's claim that CoM is structurally wrong horizontally is
      something to *measure* (run with ``com_prealign=False`` and compare), not assume.
    * The engine class is imported inside :meth:`run`, so an aligner module that does
      not exist yet yields ``status="skipped"`` instead of an import error at startup.
    """

    name: str
    engine_module: str
    engine_class: str
    iterations: int = 10
    algorithm: str = "sirt"
    inner_iters: int = 2
    mode: str = "joint"
    upsample_factor: int = 20
    blur_edges: bool = True
    com_prealign: bool = True
    backend: str = BACKEND_NAME
    #: Pass the projector object directly (OdstrcilEngine's test/backend seam) instead
    #: of routing through TKtomo's name registry.
    use_projector_kwarg: bool = False
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    allow_tomopy_shim: bool = True

    def run(self, case: BenchmarkCase) -> AlignerResult:
        try:
            from tktomo.ptycho_align.core.engine import AlignConfig  # noqa: PLC0415

            module = importlib.import_module(self.engine_module)
            engine_class = getattr(module, self.engine_class)
        except (ImportError, AttributeError) as exc:
            return AlignerResult(
                self.name,
                "skipped",
                f"{self.engine_module}.{self.engine_class} is not available yet ({exc}). "
                "The harness runs without it; add the module and re-run.",
            )

        backend = register_numpy_backend()
        extras: dict[str, Any] = {
            "engine": f"{self.engine_module}.{self.engine_class}",
            "algorithm": self.algorithm,
            "mode": self.mode,
            "com_prealign": self.com_prealign,
            "backend": "projector-object" if self.use_projector_kwarg else self.backend,
        }
        sx0, sy0 = _com_prealign(case, extras) if self.com_prealign else (None, None)

        config = AlignConfig(
            recon_algorithm=self.algorithm,
            recon_inner_iters=self.inner_iters,
            mode=self.mode,
            backend=self.backend,
            upsample_factor=self.upsample_factor,
            blur_edges=self.blur_edges,
            row_chunk=case.height,  # one chunk; see the class docstring
        )
        kwargs = dict(self.engine_kwargs)
        if self.use_projector_kwarg:
            kwargs["projector"] = backend

        timer = metrics.Timer()
        try:
            with tomopy_shim(enabled=self.allow_tomopy_shim) as shimmed, timer:
                extras["tomopy_shim"] = shimmed
                engine = engine_class(
                    dataset=case.as_projection_data(),
                    config=config,
                    sx0=sx0,
                    sy0=sy0,
                    center=default_center(case),
                    **kwargs,
                )
                history = engine.run(self.iterations)
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            return AlignerResult(
                self.name, "error", f"{type(exc).__name__}: {exc}", extras=extras
            )

        if not history:
            return AlignerResult(
                self.name, "error", "the engine completed zero iterations", extras=extras
            )
        last = history[-1]
        extras["diverging"] = bool(last.diverging)
        extras["runaway"] = last.runaway
        vertical = getattr(engine, "vertical", None)
        if vertical is not None:  # OdstrcilEngine's stage-1 report
            extras["vertical_stage"] = {
                "n_iterations": int(getattr(vertical, "n_iterations", 0)),
                "converged": bool(getattr(vertical, "converged", False)),
                "rms_shift_px": float(getattr(vertical, "rms_shift", math.nan)),
                "truncation_reason": getattr(vertical, "truncation_reason", None),
            }
            extras["vertical_cross_check_px"] = float(
                getattr(engine, "vertical_cross_check", math.nan)
            )
        return AlignerResult(
            name=self.name,
            status="ok",
            message=f"{len(history)} outer iteration(s), {self.algorithm}/{self.mode}",
            sy=np.asarray(last.sy, dtype=np.float64),
            sx=np.asarray(last.sx, dtype=np.float64),
            iterations=len(history),
            wallclock_s=timer.seconds,
            residual_history=[float(r.residual) for r in history],
            extras=extras,
        )


@dataclass
class JointGdAligner:
    """The ported joint gradient-descent aligner, :mod:`tktomo.ptycho_align.core.joint_gd`.

    A different surface from the two engines -- it takes the raw stack rather than a
    ``ProjectionData``, runs a multi-resolution *schedule* rather than N equal outer
    iterations, and the answer must be read out through ``finalize()``, which
    median-centres and rejects MAD outliers. So it gets its own adapter.

    Its production schedule bins by 16/8/4, which is meaningless on a benchmark frame
    of a hundred-odd pixels: bin 16 would leave a 6 px image. :meth:`stages_for` picks
    the coarsest binnings that still leave at least ``min_binned_px`` across the
    detector, so a small case runs the same algorithm on a schedule that makes sense
    for it. Pass ``stages=`` to override.

    ``projector`` defaults to ``"numpy"`` rather than the module's own ``"astra"``:
    the benchmark must run with no GPU.

    ``sign`` negates the returned shifts before they are scored, and exists because
    this is the single most likely thing to be wrong when a whole module is ported
    across a convention boundary. ``ndimage.shift`` moves content by ``+s``;
    ``apply_shifts`` moves it by ``-s``; joint_gd's public surface is documented in the
    latter sense and does the negation in one place (``_shift_stack``). Verified here
    against a phantom whose truth is known, with ``clean`` the unperturbed stack::

        ||aligned_projections(+truth) - clean|| / ||clean||   0.118   <- correct
        ||aligned_projections(-truth) - clean|| / ||clean||   0.868
        ||undo_shifts(+truth)         - clean|| / ||clean||   0.023   <- Fourier, exact

    so ``sign=+1`` is right and is the default. Leave it there. It is a knob rather
    than a constant so that this can be *re-measured* rather than assumed: an earlier
    revision of joint_gd had the negation the other way round, and the benchmark's
    ``sign_check`` caught it as a negative correlation with the truth and a score of
    almost exactly twice the injected RMS. Running the ``joint_gd_negated`` row
    alongside the real one keeps that signature visible in the table.
    """

    name: str = "joint_gd"
    iterations_per_stage: int = 60
    stages: Any = None
    projector: str = "numpy"
    com_prealign: bool = True
    min_binned_px: int = 32
    warmup_iters: int | None = None
    lr_shift: float = 0.5
    shift_cap_px: float = 0.5
    align_vertical: bool = True
    sign: int = 1

    def stages_for(self, case: BenchmarkCase, gd_stage: Any) -> tuple[Any, ...]:
        binnings = [b for b in (4, 2, 1) if case.width // b >= self.min_binned_px] or [1]
        return tuple(
            gd_stage(
                binning=b,
                iterations=self.iterations_per_stage,
                smooth_sigma=1.0 if b > 1 else 0.0,
            )
            for b in binnings
        )

    def run(self, case: BenchmarkCase) -> AlignerResult:
        try:
            from tktomo.ptycho_align.core.joint_gd import (  # noqa: PLC0415
                GDStage,
                JointGDAligner,
                JointGDConfig,
            )
        except ImportError as exc:
            return AlignerResult(
                self.name,
                "skipped",
                f"tktomo.ptycho_align.core.joint_gd is not available yet ({exc}). "
                "The harness runs without it; add the module and re-run.",
            )

        extras: dict[str, Any] = {"com_prealign": self.com_prealign, "projector": self.projector}
        sx0, sy0 = _com_prealign(case, extras) if self.com_prealign else (None, None)
        initial = None
        if sx0 is not None and sy0 is not None:
            initial = np.column_stack([sy0, sx0])

        stages = self.stages or self.stages_for(case, GDStage)
        warmup = (
            self.warmup_iters
            if self.warmup_iters is not None
            else max(1, min(10, self.iterations_per_stage // 4))
        )
        extras["stages"] = [
            {"binning": s.binning, "iterations": s.iterations, "smooth_sigma": s.smooth_sigma}
            for s in stages
        ]
        extras["warmup_iters"] = warmup

        config = JointGDConfig(
            stages=stages,
            projector=self.projector,
            warmup_iters=warmup,
            lr_shift=self.lr_shift,
            shift_cap_px=self.shift_cap_px,
            align_vertical=self.align_vertical,
        )

        timer = metrics.Timer()
        try:
            with timer:
                aligner = JointGDAligner(
                    projections=case.projections,
                    angles=case.angles,
                    config=config,
                    initial_shifts=initial,
                )
                history = aligner.run()
                answer = aligner.finalize()
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            return AlignerResult(
                self.name, "error", f"{type(exc).__name__}: {exc}", extras=extras
            )

        extras["n_outliers"] = int(answer.n_outliers)
        extras["median_offset_px"] = [float(v) for v in answer.median_offset]
        extras["sign"] = int(self.sign)
        return AlignerResult(
            name=self.name,
            status="ok",
            message=f"{len(history)} iteration(s) over {len(stages)} stage(s)"
            + ("" if self.sign == 1 else f", shifts negated (sign={self.sign})"),
            sy=self.sign * np.asarray(answer.shifts[:, 0], dtype=np.float64),
            sx=self.sign * np.asarray(answer.shifts[:, 1], dtype=np.float64),
            iterations=len(history),
            wallclock_s=timer.seconds,
            residual_history=[float(getattr(h, "loss", math.nan)) for h in history],
            extras=extras,
        )


class ModuleAligner:
    """Adapter for an aligner that lives in a module which may not exist yet.

    The two new methods are being written in parallel with this harness, so their
    exact API is not knowable here. This adapter therefore *probes*: it imports
    ``module``, looks for the first entry point in ``entry_points`` that exists, and
    calls it through a short list of plausible signatures, recording in
    ``extras["entry_point"]`` and ``extras["call_form"]`` exactly what worked. An
    ImportError yields ``status="skipped"``; nothing found yields ``status="skipped"``
    with the list of what was tried.

    The **preferred contract**, documented in ``benchmarks/README.md``, is a single
    module-level function::

        def align(projections, angles, *, center=None, iterations=10, **kwargs):
            '''-> object with .sy and .sx (px, engine sign convention)'''

    Anything conforming to that is driven with no probing at all.
    """

    def __init__(
        self,
        name: str,
        module: str,
        *,
        entry_points: Sequence[str] = (),
        iterations: int = 10,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.module = module
        self.entry_points = tuple(entry_points) or (
            "align",
            "align_projections",
            "run_alignment",
            "align_stack",
            "run",
        )
        self.iterations = iterations
        self.kwargs = dict(kwargs or {})

    def run(self, case: BenchmarkCase) -> AlignerResult:
        try:
            module = importlib.import_module(self.module)
        except ImportError as exc:
            return AlignerResult(
                self.name,
                "skipped",
                f"{self.module} is not importable yet ({exc}). The harness runs "
                "without it; add the module and re-run.",
            )

        candidates = [
            (name, getattr(module, name))
            for name in self.entry_points
            if callable(getattr(module, name, None))
        ]
        if not candidates:
            exported = sorted(n for n in vars(module) if not n.startswith("_"))
            return AlignerResult(
                self.name,
                "skipped",
                f"{self.module} exports none of {list(self.entry_points)}; it has "
                f"{exported}. See benchmarks/README.md for the expected contract.",
            )

        entry_name, entry = candidates[0]
        base = {
            "center": default_center(case),
            "iterations": self.iterations,
            "pixel_size_nm": case.pixel_size_nm,
            **self.kwargs,
        }
        attempts: list[tuple[str, Callable[[], Any]]] = [
            ("(projections, angles, **kwargs)", lambda: entry(case.projections, case.angles, **_supported(entry, base))),
            ("(projections, angles)", lambda: entry(case.projections, case.angles)),
            ("(ProjectionData, **kwargs)", lambda: entry(case.as_projection_data(), **_supported(entry, base))),
            ("(ProjectionData)", lambda: entry(case.as_projection_data())),
        ]

        timer = metrics.Timer()
        errors: list[str] = []
        for form, call in attempts:
            try:
                with timer:
                    outcome = call()
            except TypeError as exc:
                errors.append(f"{form}: TypeError: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - it ran and failed; that is a result
                return AlignerResult(
                    self.name,
                    "error",
                    f"{entry_name}{form} raised {type(exc).__name__}: {exc}",
                    extras={"entry_point": entry_name, "call_form": form},
                )

            try:
                sy, sx = _as_shifts(outcome, case.n_angles)
            except Exception as exc:  # noqa: BLE001
                return AlignerResult(
                    self.name,
                    "error",
                    f"{entry_name} returned something the harness cannot read: {exc}",
                    extras={"entry_point": entry_name, "call_form": form,
                            "returned": type(outcome).__name__},
                )
            return AlignerResult(
                name=self.name,
                status="ok",
                message=f"{self.module}.{entry_name} via {form}",
                sy=sy,
                sx=sx,
                iterations=int(getattr(outcome, "iterations", 0) or 0),
                wallclock_s=timer.seconds,
                residual_history=[
                    float(v) for v in (getattr(outcome, "residual_history", None) or [])
                ],
                extras={"entry_point": entry_name, "call_form": form},
            )

        return AlignerResult(
            self.name,
            "skipped",
            f"could not find a working call signature for {self.module}.{entry_name}. "
            + " | ".join(errors),
            extras={"entry_point": entry_name},
        )


def _supported(function: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keyword arguments ``function`` actually accepts.

    Passing an unexpected keyword would be a TypeError indistinguishable from a
    genuine signature mismatch, so filter first and let a real mismatch mean what it
    says. A ``**kwargs`` in the signature means everything is accepted.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):  # builtins, C extensions
        return {}
    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in parameters}


def _as_shifts(outcome: Any, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalise whatever an aligner returned into ``(sy, sx)``.

    Accepts, in order: an object with ``.sy``/``.sx``; an object with ``.shifts``
    shaped ``(n, 2)`` as ``(dy, dx)``; a mapping with ``sy``/``sx`` or ``dy``/``dx``;
    a 2-tuple ``(sy, sx)``; an ``(n, 2)`` array.
    """
    sy = sx = None
    if hasattr(outcome, "sy") and hasattr(outcome, "sx"):
        sy, sx = outcome.sy, outcome.sx
    elif hasattr(outcome, "shifts"):
        shifts = np.asarray(outcome.shifts, dtype=np.float64)
        sy, sx = shifts[:, 0], shifts[:, 1]
    elif isinstance(outcome, dict):
        if "sy" in outcome and "sx" in outcome:
            sy, sx = outcome["sy"], outcome["sx"]
        elif "dy" in outcome and "dx" in outcome:
            sy, sx = outcome["dy"], outcome["dx"]
    elif isinstance(outcome, (tuple, list)) and len(outcome) == 2:
        sy, sx = outcome
    else:
        array = np.asarray(outcome, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 2:
            sy, sx = array[:, 0], array[:, 1]

    if sy is None or sx is None:
        raise TypeError(
            f"cannot read shifts from a {type(outcome).__name__}; expected .sy/.sx, "
            ".shifts, a dict, a 2-tuple or an (n, 2) array"
        )
    sy = np.asarray(sy, dtype=np.float64).ravel()
    sx = np.asarray(sx, dtype=np.float64).ravel()
    if sy.size != n_angles or sx.size != n_angles:
        raise ValueError(
            f"aligner returned {sy.size}/{sx.size} shifts for {n_angles} projections"
        )
    return sy, sx


def JirrAligner(**kwargs: Any) -> EngineAligner:  # noqa: N802 - reads as a class
    """The incumbent: Gursoy et al. 2017 joint iterative reprojection.

    Registers the **phase itself** (not its gradient) of measured against simulated
    with ``skimage.registration.phase_cross_correlation``. This is the baseline the
    roadmap's two new methods have to beat.
    """
    return EngineAligner(
        name=kwargs.pop("name", "jirr"),
        engine_module=kwargs.pop("engine_module", "tktomo.ptycho_align.core.engine"),
        engine_class=kwargs.pop("engine_class", "AlignmentEngine"),
        **kwargs,
    )


def OdstrcilAligner(**kwargs: Any) -> EngineAligner:  # noqa: N802 - reads as a class
    """The roadmap's method: vertical mass distribution, then horizontal on the gradient.

    Driven through the same :class:`EngineAligner` as the incumbent, with the
    projector handed in directly rather than through the name registry, so no part of
    the comparison differs except the algorithm.
    """
    return EngineAligner(
        name=kwargs.pop("name", "odstrcil"),
        engine_module=kwargs.pop("engine_module", "tktomo.ptycho_align.core.odstrcil"),
        engine_class=kwargs.pop("engine_class", "OdstrcilEngine"),
        use_projector_kwarg=kwargs.pop("use_projector_kwarg", True),
        **kwargs,
    )


def default_aligners(*, iterations: int = 10) -> list[Any]:
    """Null, oracle, the incumbent, and the two new methods."""
    return [
        NullAligner(),
        OracleAligner(),
        JirrAligner(iterations=iterations),
        OdstrcilAligner(iterations=iterations),
        JointGdAligner(),
    ]


# --------------------------------------------------------------------------------
# Running and reporting
# --------------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    case: dict[str, Any]
    environment: dict[str, Any]
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"case": self.case, "environment": self.environment, "results": self.rows}

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkReport":
        """Reload a written report, so a figure can be redrawn without re-running.

        The runs are the expensive part and the plotting is the part that gets
        fiddled with, so these are deliberately separable.
        """
        payload = json.loads(Path(path).read_text())
        return cls(
            case=payload["case"], environment=payload["environment"], rows=payload["results"]
        )

    def table(self) -> str:
        """A fixed-width comparison table, primary metric first."""
        header = (
            f"{'aligner':<14}{'status':<9}{'rms dy':>9}{'rms dx':>9}{'max dy':>9}"
            f"{'max dx':>9}{'target':>9}{'iters':>7}{'sec':>9}"
        )
        lines = [header, "-" * len(header)]
        for row in self.rows:
            score = row.get("shift_recovery")
            if not score:
                lines.append(f"{row['name']:<14}{row['status']:<9}  {row.get('message', '')[:60]}")
                continue
            lines.append(
                f"{row['name']:<14}{row['status']:<9}"
                f"{score['rms_dy']:>9.3f}{score['rms_dx']:>9.3f}"
                f"{score['max_dy']:>9.3f}{score['max_dx']:>9.3f}"
                f"{'yes' if score['meets_target'] else 'NO':>9}"
                f"{row['iterations']:>7d}{row['wallclock_s']:>9.1f}"
            )
        lines.append("")
        lines.append(
            "px, after removing the unobservable gauge modes "
            f"(dy: {list(metrics.DY_GAUGE)}, dx: {list(metrics.DX_GAUGE)}). "
            f"target = {self.case.get('target_px', float('nan')):.3f} px "
            f"(1/{round(1 / self.case.get('target_fraction', 1 / 3))} voxel)."
        )
        return "\n".join(lines)


def run_benchmark(
    case: BenchmarkCase,
    aligners: Sequence[Any] | None = None,
    *,
    target_fraction: float = 1.0 / 3.0,
    voxel_nm: float | None = None,
    with_residual: bool = True,
    with_fsc: bool = False,
    fsc_algorithm: str = "sirt",
    fsc_iterations: int = 8,
) -> BenchmarkReport:
    """Run every aligner on ``case``, score it, and return a report.

    ``with_residual`` reprojects each aligner's aligned stack through a single
    reconstruction and records the per-angle residual map -- the diagnostic that
    separates rigid-alignment failure (structured in angle) from genuine non-rigid
    deformation (high but angularly uncorrelated), and the one that, unlike the FSC,
    actually sees a common-mode geometric error.

    ``with_fsc`` additionally reconstructs the odd and even angle subsets separately
    and computes their FSC. It is off by default because it costs two more
    reconstructions per aligner and, on its own, tells you less than it appears to:
    see :func:`benchmarks.metrics.fourier_shell_correlation`.
    """
    aligners = list(aligners if aligners is not None else default_aligners())
    backend = register_numpy_backend()

    rows: list[dict[str, Any]] = []
    for aligner in aligners:
        logger.info("running aligner %r", aligner.name)
        result = aligner.run(case)
        row: dict[str, Any] = {
            "name": result.name,
            "status": result.status,
            "message": result.message,
            "iterations": int(result.iterations),
            "wallclock_s": float(result.wallclock_s) if np.isfinite(result.wallclock_s) else None,
            "residual_history": list(result.residual_history),
            "extras": _jsonable(result.extras),
        }
        if result.ok and result.sy is not None and result.sx is not None:
            try:
                score = metrics.score_shifts(
                    result.sy,
                    result.sx,
                    case.truth.dy,
                    case.truth.dx,
                    case.angles,
                    pixel_size_nm=case.pixel_size_nm,
                    voxel_nm=voxel_nm,
                    target_fraction=target_fraction,
                )
            except ValueError as exc:
                row["status"] = "error"
                row["message"] = f"unscoreable result: {exc}"
                rows.append(row)
                continue
            row["shift_recovery"] = score.to_dict()
            row["sign_check"] = _sign_check(case, result, score, voxel_nm, target_fraction)
            if row["sign_check"]["flipped_is_better"]:
                row["message"] = (
                    f"{row['message']} [SIGN: negating this aligner's shifts scores "
                    f"{row['sign_check']['flipped_rms_dy']:.3f}/"
                    f"{row['sign_check']['flipped_rms_dx']:.3f} px against "
                    f"{score.rms_dy:.3f}/{score.rms_dx:.3f} -- its sign convention is "
                    "the opposite of apply_shifts'. NOT auto-corrected.]"
                )
                logger.warning("%s: %s", result.name, row["message"])
            row["sy"] = np.asarray(result.sy).tolist()
            row["sx"] = np.asarray(result.sx).tolist()
            if result.residual_history:
                row["plateau"] = metrics.residual_plateau(result.residual_history).to_dict()
            if with_residual:
                row["reprojection_residual"] = _residual_row(case, result, backend)
            if with_fsc:
                row["fsc"] = _fsc_row(case, result, backend, fsc_algorithm, fsc_iterations)
        rows.append(row)

    injected_dy, injected_dx = case.truth.rigid_rms
    report = BenchmarkReport(
        case={
            **case.summary(),
            "target_fraction": target_fraction,
            "target_px": target_fraction * ((voxel_nm or case.pixel_size_nm) / case.pixel_size_nm),
            "injected_rms_dy_px": injected_dy,
            "injected_rms_dx_px": injected_dx,
        },
        environment=environment_report(),
        rows=rows,
    )
    return report


def _sign_check(
    case: BenchmarkCase,
    result: AlignerResult,
    score: metrics.ShiftRecovery,
    voxel_nm: float | None,
    target_fraction: float,
) -> dict[str, Any]:
    """Score the negated estimate too, and say so when it is dramatically better.

    A shift convention is a sign, and this repository contains two of them: the
    engine's ``apply_shifts`` moves content by ``-s``, while ``scipy.ndimage.shift``
    moves it by ``+s``. A module documented against one and implemented against the
    other produces a result that is exactly as wrong as doing nothing, and the tell is
    a recovery RMS of about twice the injected RMS.

    The harness reports both numbers and **does not silently correct the sign**. Auto
    -negating would turn a real, shipping-blocking bug into an invisible one; the
    point of a benchmark is to make it visible.
    """
    flipped = metrics.score_shifts(
        -np.asarray(result.sy, dtype=np.float64),
        -np.asarray(result.sx, dtype=np.float64),
        case.truth.dy,
        case.truth.dx,
        case.angles,
        pixel_size_nm=case.pixel_size_nm,
        voxel_nm=voxel_nm,
        target_fraction=target_fraction,
    )
    total = math.hypot(score.rms_dy, score.rms_dx)
    total_flipped = math.hypot(flipped.rms_dy, flipped.rms_dx)
    # The crispest signal, and the one that survives an unconverged run: a correct
    # estimate correlates positively with the truth whatever its magnitude.
    correlations = []
    for estimate, truth in ((result.sy, case.truth.dy), (result.sx, case.truth.dx)):
        estimate = np.asarray(estimate, dtype=np.float64)
        if estimate.std() < 1e-12 or np.asarray(truth).std() < 1e-12:
            correlations.append(math.nan)
        else:
            correlations.append(float(np.corrcoef(estimate, truth)[0, 1]))
    negative = [c for c in correlations if not math.isnan(c) and c < -0.5]
    return {
        "convention": "sy/sx are corrections in the apply_shifts sense (content moves by -s)",
        "flipped_rms_dy": flipped.rms_dy,
        "flipped_rms_dx": flipped.rms_dx,
        "correlation_dy": correlations[0],
        "correlation_dx": correlations[1],
        # Either signal alone is enough. A perfectly sign-flipped estimate scores
        # exactly 2x the injected RMS, so a factor of 2 is the theoretical maximum
        # ratio and an unconverged flipped run shows less -- hence the correlation
        # test alongside it.
        "flipped_is_better": bool(total_flipped * 2.0 < total or len(negative) == 2),
    }


def _residual_row(case: BenchmarkCase, result: AlignerResult, backend: Any) -> dict[str, Any]:
    aligned = undo_shifts(case.projections, result.sy, result.sx)
    volume = backend.reconstruct(
        aligned, case.angles, algorithm="sirt", center=default_center(case), num_iter=20
    )
    simulated = backend.reproject(volume, case.angles, center=default_center(case))
    residual = metrics.reprojection_residual(aligned, simulated)
    out = residual.to_dict()
    if case.clean is not None:
        # How far the aligned stack is from the projections that were actually
        # perturbed -- an error the reprojection residual cannot separate from
        # reconstruction error, but the phantom can.
        out["vs_clean"] = metrics.reprojection_residual(case.clean, aligned).total
    return out


def _fsc_row(
    case: BenchmarkCase, result: AlignerResult, backend: Any, algorithm: str, iterations: int
) -> dict[str, Any]:
    aligned = undo_shifts(case.projections, result.sy, result.sx)
    even, odd = metrics.split_half_indices(case.n_angles)
    halves = []
    for index in (even, odd):
        halves.append(
            backend.reconstruct(
                aligned[index],
                case.angles[index],
                algorithm=algorithm,
                center=default_center(case),
                num_iter=iterations,
            )
        )
    fsc = metrics.fourier_shell_correlation(
        halves[0], halves[1], pixel_size_nm=case.pixel_size_nm
    )
    out = fsc.to_dict()
    out["caveat"] = (
        "FSC is exactly invariant to a geometry error applied identically to both "
        "half-sets. Read it alongside shift_recovery, never instead of it."
    )
    return out


def environment_report() -> dict[str, Any]:
    import scipy  # noqa: PLC0415

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "tomopy": None,
        "tomopy_shim": not tomopy_is_real(),
        "skimage": None,
        "astra": None,
    }
    for name in ("tomopy", "skimage", "astra"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(module, "__tktomo_benchmark_shim__", False):
            continue
        info[name] = getattr(module, "__version__", "unknown")
    return info


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


# --------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------


def comparison_figure(report: BenchmarkReport, path: str | Path) -> Path | None:
    """Four panels: recovery bars, per-angle error, residual history, per-angle residual.

    Returns the written path, or None when matplotlib is not installed -- a missing
    plotting library must never take down a benchmark run.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        logger.warning("matplotlib is not installed; skipping the comparison figure")
        return None

    scored = [r for r in report.rows if r.get("shift_recovery")]
    if not scored:
        logger.warning("no scored results; skipping the comparison figure")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    names = [r["name"] for r in scored]
    target = report.case.get("target_px", float("nan"))

    ax = axes[0, 0]
    positions = np.arange(len(names))
    ax.bar(positions - 0.2, [r["shift_recovery"]["rms_dy"] for r in scored], 0.4, label="dy")
    ax.bar(positions + 0.2, [r["shift_recovery"]["rms_dx"] for r in scored], 0.4, label="dx")
    ax.axhline(target, color="k", ls="--", lw=1, label=f"target {target:.2f} px")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("shift-recovery RMS (px, gauge removed)")
    ax.set_title("PRIMARY: shift recovery vs truth")
    ax.legend(fontsize=8)

    # |error| on a log axis, not the signed error on a linear one: the whole point is
    # that the good aligners are two to three ORDERS below the null, and a linear axis
    # scaled to the null draws every method worth looking at as a flat line on zero.
    ax = axes[0, 1]
    truth_dy = np.asarray(report.case.get("truth_dy", []), dtype=float)
    for row in scored:
        if "sy" not in row or truth_dy.size == 0:
            continue
        error = np.asarray(row["sy"]) - truth_dy
        error = error - error.mean()  # the dy gauge mode
        ax.semilogy(np.abs(error) + 1e-6, lw=1, label=row["name"])
    ax.axhline(target, color="k", ls="--", lw=1)
    ax.set_xlabel("projection index")
    ax.set_ylabel("|dy error| (px, gauge removed)")
    ax.set_title("vertical error per projection")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2)

    # Each aligner's own convergence measure, normalised to its first value. They are
    # NOT the same quantity -- the engines report a reprojection residual, joint-GD a
    # least-squares loss -- so only the shapes are comparable, and the axis says so.
    ax = axes[1, 0]
    plotted = False
    for row in scored:
        history = [v for v in (row.get("residual_history") or []) if np.isfinite(v)]
        if len(history) > 1 and history[0] > 0:
            ax.semilogy(
                range(1, len(history) + 1),
                np.asarray(history) / history[0],
                marker="o",
                ms=2,
                lw=1,
                label=row["name"],
            )
            plotted = True
    ax.set_xlabel("iteration")
    ax.set_ylabel("own metric, relative to iteration 1")
    ax.set_title(
        "convergence shape (residual for the engines, loss for joint-GD)"
        if plotted
        else "convergence (no iterative aligner ran)"
    )
    if plotted:
        ax.legend(fontsize=7)

    ax = axes[1, 1]
    for row in scored:
        residual = row.get("reprojection_residual")
        if residual:
            ax.plot(residual["per_angle"], lw=1, label=row["name"])
    ax.set_xlabel("projection index")
    ax.set_ylabel("relative residual")
    ax.set_title("reprojection residual vs angle (sees what FSC cannot)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)

    figure.suptitle(f"alignment benchmark: {report.case.get('name', '?')}")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def build_case(args: argparse.Namespace) -> BenchmarkCase:
    spec = PerturbationSpec(
        jitter_dy_rms=args.jitter_dy,
        jitter_dx_rms=args.jitter_dx,
        drift_dy=args.drift_dy,
        axis_tilt_deg=args.axis_tilt,
        out_of_plane_tilt_deg=args.out_of_plane_tilt,
        magnification_drift=args.magnification,
        angle_error_rms_deg=args.angle_error,
        phase_ramp_rms=args.phase_ramp,
        phase_offset_rms=args.phase_offset,
        truncation_px=args.truncation,
        deformation_px=args.deformation,
        noise_rms=args.noise,
        seed=args.seed,
    )
    if args.volume is None:
        return synthetic_case(
            name=args.name or "synthetic",
            size=args.size,
            n_slices=args.slices,
            n_angles=args.angles,
            spec=spec,
            pixel_size_nm=args.pixel_size,
        )

    slice_range = None
    if args.slice_range:
        start, stop = (int(v) for v in args.slice_range.split(":"))
        slice_range = slice(start, stop)
    volume = load_volume(
        args.volume, dataset=args.volume_dataset, slices=slice_range, bin_factor=args.bin
    )
    if args.angles_file:
        angles = load_angles(args.angles_file, dataset=args.angles_dataset)
        if args.angles and args.angles < angles.size:
            angles = angles[:: max(1, angles.size // args.angles)][: args.angles]
    else:
        angles = np.deg2rad(np.linspace(0.0, 180.0, args.angles, endpoint=False))
    return volume_case(
        volume,
        angles,
        name=args.name or Path(args.volume).name,
        spec=spec,
        pixel_size_nm=args.pixel_size * args.bin,
        metadata={"volume_path": str(args.volume), "bin_factor": args.bin},
    )


def _run_catalogue(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """One synthetic case per perturbation, so a failure names the geometry error.

    A single all-perturbations-on case tells you an aligner is bad. This tells you
    *which* thing it cannot handle, which is the only version of the answer anyone can
    act on.
    """
    from benchmarks.phantom import cases_from_catalogue  # noqa: PLC0415

    cases = cases_from_catalogue(
        size=args.size, n_slices=args.slices, n_angles=args.angles, pixel_size_nm=args.pixel_size
    )
    out = Path(args.out)
    summary: dict[str, Any] = {}
    for label, case in cases.items():
        selected = _select_aligners(args, parser)
        report = run_benchmark(
            case,
            selected,
            target_fraction=args.target_fraction,
            voxel_nm=args.voxel_nm,
            with_residual=not args.no_residual,
            with_fsc=args.fsc,
        )
        report.to_json(out / f"catalogue_{label}.json")
        summary[label] = {
            row["name"]: (row.get("shift_recovery") or {}).get("rms_dy")
            for row in report.rows
        }
        print(f"\n=== {label} ===")
        print(report.table())
    (out / "catalogue_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'catalogue_summary.json'}")
    return 0


def _select_aligners(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Any]:
    catalogue = {
        "null": NullAligner,
        "oracle": OracleAligner,
        "jirr": lambda: JirrAligner(iterations=args.iterations),
        "odstrcil": lambda: OdstrcilAligner(iterations=args.iterations),
        "joint_gd": lambda: JointGdAligner(
            iterations_per_stage=args.gd_iterations, sign=args.joint_gd_sign
        ),
        "joint_gd_negated": lambda: JointGdAligner(
            name="joint_gd_neg", iterations_per_stage=args.gd_iterations, sign=-1
        ),
    }
    selected = []
    for name in (n.strip() for n in args.aligners.split(",") if n.strip()):
        if name not in catalogue:
            parser.error(f"unknown aligner {name!r}; have {sorted(catalogue)}")
        selected.append(catalogue[name]())
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default="benchmark_results", help="output directory")
    parser.add_argument("--iterations", type=int, default=10, help="outer iterations for the engine-style aligners")
    parser.add_argument("--gd-iterations", type=int, default=60, help="joint_gd iterations per multi-resolution stage")
    parser.add_argument(
        "--joint-gd-sign",
        type=int,
        default=1,
        choices=(1, -1),
        help="negate joint_gd's shifts before scoring; see JointGdAligner.sign",
    )
    parser.add_argument(
        "--aligners", default="null,oracle,jirr,odstrcil,joint_gd,joint_gd_negated"
    )
    parser.add_argument("--fsc", action="store_true", help="also compute the split-half FSC")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--target-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--voxel-nm", type=float, default=None)

    synthetic = parser.add_argument_group("fully synthetic case")
    synthetic.add_argument("--size", type=int, default=64)
    synthetic.add_argument("--slices", type=int, default=12)
    synthetic.add_argument("--angles", type=int, default=60)
    synthetic.add_argument("--pixel-size", type=float, default=1.0, help="nm per pixel")

    real = parser.add_argument_group("synthetic-from-real case (user-supplied volume)")
    real.add_argument("--volume", default=None, help="dir of TIFFs, .npy, or .h5")
    real.add_argument("--volume-dataset", default="/tomogram/data")
    real.add_argument("--angles-file", default=None, help=".h5 holding the scan angles")
    real.add_argument("--angles-dataset", default="exchange/theta")
    real.add_argument("--slice-range", default=None, help="START:STOP detector rows")
    real.add_argument("--bin", type=int, default=1)

    perturb = parser.add_argument_group("perturbations (0 disables)")
    perturb.add_argument("--jitter-dy", type=float, default=2.5)
    perturb.add_argument("--jitter-dx", type=float, default=0.75)
    perturb.add_argument("--drift-dy", type=float, default=0.0)
    perturb.add_argument("--axis-tilt", type=float, default=0.0)
    perturb.add_argument("--out-of-plane-tilt", type=float, default=0.0)
    perturb.add_argument("--magnification", type=float, default=0.0)
    perturb.add_argument("--angle-error", type=float, default=0.0)
    perturb.add_argument("--phase-ramp", type=float, default=0.0)
    perturb.add_argument("--phase-offset", type=float, default=0.0)
    perturb.add_argument("--truncation", type=int, default=0)
    perturb.add_argument("--deformation", type=float, default=0.0)
    perturb.add_argument("--noise", type=float, default=0.0)
    perturb.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--catalogue",
        action="store_true",
        help="run the diagnostic sweep: one case per perturbation, that one alone",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.catalogue:
        return _run_catalogue(args, parser)

    case = build_case(args)
    selected = _select_aligners(args, parser)

    report = run_benchmark(
        case,
        selected,
        target_fraction=args.target_fraction,
        voxel_nm=args.voxel_nm,
        with_residual=not args.no_residual,
        with_fsc=args.fsc,
    )
    # The figure needs the truth alongside the estimates.
    report.case["truth_dy"] = case.truth.dy.tolist()
    report.case["truth_dx"] = case.truth.dx.tolist()

    out = Path(args.out)
    json_path = report.to_json(out / f"{case.name}.json")
    figure_path = comparison_figure(report, out / f"{case.name}.png")

    print(report.table())
    print(f"\nwrote {json_path}")
    if figure_path:
        print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
