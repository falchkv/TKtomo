"""The decoupled two-stage aligner, validated on a synthetic phantom.

Three things are being established:

1. **It recovers known shifts to better than 1/3 of a voxel** -- the roadmap's accuracy
   target -- on a phantom with injected misalignment. See
   :func:`test_recovers_known_shifts_within_a_third_of_a_voxel`.
2. **The gradient domain survives a residual phase ramp and the value domain does not**,
   measured end to end through the whole loop rather than on a single image pair (that
   part lives in ``tests/test_gradient_registration.py``, where the invariance is exact
   to 4e-16 px).
3. **The documented reconstruction failure modes are refused in code**, not left in a
   README: GridRec on a limited angular range, and initialising from a direct
   reconstruction.

Running without tomopy
----------------------
Two substitutions make this file run with numpy + scipy alone, and both are pinned by
their own tests rather than assumed:

* :class:`ReferenceProjector` is an exact-adjoint parallel-beam projector with SIRT and
  FBP. Its forward operator is a sparse matrix built once for one slice (every slice
  shares the same 2-D geometry) and its back-projector is that matrix transposed, so the
  adjoint is exact to floating point. That matters: an earlier version projected by
  rotate-and-sum, whose "adjoint" is only accurate to 3e-4, and SIRT built on it
  oscillated instead of converging -- ``||r||/||b||`` went 0.20, 1.49, 1.05, 1.71 over
  four iterations. An approximate adjoint is not a small approximation.
* :func:`substitute_apply_shifts` replaces the engine's ``apply_shifts``, which wraps
  ``tomopy.prep.alignment.shift_images``. It implements the contract that
  ``tests/test_ptycho_engine.py`` pins on the real one, and
  :func:`test_the_substitute_shift_obeys_the_real_apply_shifts_contract` re-checks it
  here so the stand-in cannot silently drift. When tomopy *is* installed, the real one is
  exercised too.

No beamtime data: the phantom is generated from ``tktomo.io.phantom.generate_volume``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter
from scipy.ndimage import shift as ndi_shift
from scipy.sparse import coo_matrix

from tktomo.io import ProjectionData
from tktomo.io.phantom import generate_volume
from tktomo.ptycho_align.core import (
    AlignConfig,
    AlignmentEngine,
    GradientConfig,
    LimitedAngleError,
    OdstrcilConfig,
    OdstrcilEngine,
    VerticalConfig,
    angular_coverage,
    available_series_aligners,
    check_reconstruction_choice,
    get_series_aligner,
)
from tktomo.ptycho_align.core import odstrcil as odstrcil_module

# Geometry of the test article. Small enough that a full 12-iteration alignment runs in
# about four seconds, large enough that 1/3 of a voxel is a meaningful ask.
N_ANGLES = 48
GRID = 64  # detector width == reconstruction grid
V_MARGIN = 6  # vacuum rows above and below, so the sample is not vertically truncated
CENTER = GRID / 2.0
MAX_SHIFT = 2.0
TARGET_PX = 1.0 / 3.0  # the roadmap's accuracy target: 1/3 of a voxel


# -- a parallel-beam projector with an exact adjoint --------------------------------


class ReferenceProjector:
    """Parallel-beam forward/back projection, SIRT and FBP, in numpy + scipy.

    Satisfies :class:`tktomo.recon.backend.ReconBackend`, so it drops into the engine
    through ``OdstrcilEngine.projector``. Counts its own calls, which is how the tests
    check that stage 1 really does no reconstruction.
    """

    name = "reference"

    def __init__(self) -> None:
        self.reconstruct_calls = 0
        self.reproject_calls = 0
        self._cache: dict[tuple, tuple] = {}

    def _matrix(self, n_grid: int, angles: np.ndarray, center: float):
        """The system matrix for ONE slice; all slices share the same 2-D geometry."""
        key = (n_grid, len(angles), float(angles[0]), float(angles[-1]), float(center))
        if key in self._cache:
            return self._cache[key]

        middle = (n_grid - 1) / 2.0
        detector = np.arange(n_grid) - center  # ray offset from the rotation axis
        along = np.arange(n_grid) - middle  # position along the ray
        u_grid, t_grid = np.meshgrid(detector, along, indexing="ij")
        bin_index = np.broadcast_to(np.arange(n_grid)[:, None], u_grid.shape)

        rows, cols, vals = [], [], []
        for i, theta in enumerate(angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x = middle + u_grid * cos_t - t_grid * sin_t
            y = middle + u_grid * sin_t + t_grid * cos_t
            x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            fx, fy = x - x0, y - y0
            for dy in (0, 1):  # bilinear interpolation weights
                for dx in (0, 1):
                    xi, yi = x0 + dx, y0 + dy
                    weight = (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)
                    ok = (xi >= 0) & (xi < n_grid) & (yi >= 0) & (yi < n_grid) & (weight > 0)
                    rows.append(i * n_grid + bin_index[ok])
                    cols.append((yi * n_grid + xi)[ok])
                    vals.append(weight[ok])

        matrix = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(angles) * n_grid, n_grid * n_grid),
        ).tocsr()

        # SIRT weights. Rays that barely clip the reconstruction circle carry almost no
        # weight, and 1/row then amplifies their residual without bound; zero them.
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        col_sum = np.asarray(matrix.sum(axis=0)).ravel()
        row_w = np.where(row_sum > 0.05 * row_sum.max(), 1.0 / np.maximum(row_sum, 1e-12), 0.0)
        col_w = np.where(col_sum > 0.05 * col_sum.max(), 1.0 / np.maximum(col_sum, 1e-12), 0.0)

        self._cache[key] = (matrix, row_w, col_w)
        return self._cache[key]

    @staticmethod
    def _center(width: int, center: float | None) -> float:
        return (width - 1) / 2.0 if center is None else float(center)

    @staticmethod
    def _to_flat(projections: np.ndarray) -> np.ndarray:
        n_theta, _n_slices, n_u = projections.shape
        return np.moveaxis(projections, 1, 2).reshape(n_theta * n_u, -1)

    @staticmethod
    def _from_flat(flat: np.ndarray, n_theta: int, n_u: int) -> np.ndarray:
        return np.moveaxis(flat.reshape(n_theta, n_u, flat.shape[1]), 2, 1)

    def reconstruct(
        self,
        projections: np.ndarray,
        angles: np.ndarray,
        *,
        center: float | None = None,
        algorithm: str = "sirt",
        num_iter: int = 2,
        init_recon: np.ndarray | None = None,
        ncore: int | None = None,
        **_kwargs,
    ) -> np.ndarray:
        self.reconstruct_calls += 1
        projections = np.asarray(projections, dtype=np.float64)
        n_theta, n_slices, n_u = projections.shape
        matrix, row_w, col_w = self._matrix(n_u, angles, self._center(n_u, center))

        if algorithm in ("gridrec", "fbp"):
            ramp = 2.0 * np.abs(np.fft.rfftfreq(n_u))
            filtered = np.fft.irfft(np.fft.rfft(projections, axis=2) * ramp, n_u, axis=2)
            volume = (matrix.T @ self._to_flat(filtered)).T
            return (volume.reshape(n_slices, n_u, n_u) * (np.pi / n_theta)).astype(np.float32)

        measured = self._to_flat(projections)
        x = (
            np.zeros((n_slices, n_u * n_u))
            if init_recon is None
            else np.array(init_recon, dtype=np.float64).reshape(n_slices, -1)
        )
        for _ in range(int(num_iter)):
            residual = measured - (matrix @ x.T)
            x += (matrix.T @ (residual * row_w[:, None])).T * col_w
        return x.reshape(n_slices, n_u, n_u).astype(np.float32)

    def reproject(
        self, volume: np.ndarray, angles: np.ndarray, *, center: float | None = None, **_kwargs
    ) -> np.ndarray:
        self.reproject_calls += 1
        volume = np.asarray(volume, dtype=np.float64)
        n_slices, _ny, n_grid = volume.shape
        matrix, _row, _col = self._matrix(n_grid, angles, self._center(n_grid, center))
        flat = matrix @ volume.reshape(n_slices, -1).T
        return self._from_flat(flat, len(angles), n_grid).astype(np.float32)


def substitute_apply_shifts(prj: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Stand-in for ``engine.apply_shifts`` (which needs tomopy), same contract.

    ``corrected(v, u) = measured(v + sy, u + sx)``, as pinned by
    ``tests/test_ptycho_engine.py::test_apply_shifts_moves_the_image_by_the_requested_amount``.
    """
    sy = np.asarray(sy, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)
    if not sy.any() and not sx.any():
        return prj
    out = np.empty_like(prj)
    for i in range(prj.shape[0]):
        out[i] = ndi_shift(prj[i], (-sy[i], -sx[i]), order=3, mode="constant")
    return out


HAVE_TOMOPY = importlib.util.find_spec("tomopy") is not None


@pytest.fixture(scope="session", autouse=True)
def shift_backend():
    """Use the real ``apply_shifts`` when tomopy is installed, the substitute otherwise.

    Session-scoped, because the module-scoped fixtures below build engines during their
    own setup and a function-scoped patch would not be in place yet.
    """
    if HAVE_TOMOPY:
        yield "tomopy"
        return
    patch = pytest.MonkeyPatch()
    patch.setattr(odstrcil_module, "apply_shifts", substitute_apply_shifts)
    yield "substitute"
    patch.undo()


@pytest.mark.skipif(not HAVE_TOMOPY, reason="the real apply_shifts needs tomopy")
def test_one_iteration_with_the_real_apply_shifts():
    """When tomopy is present, exercise the genuine shift path at least once."""
    from tktomo.ptycho_align.core.engine import apply_shifts as real_apply_shifts

    dataset, _sx, _sy, projector = misaligned_dataset()
    with pytest.MonkeyPatch().context() as patch:
        patch.setattr(odstrcil_module, "apply_shifts", real_apply_shifts)
        engine = build_engine(dataset, projector)
        engine.run(2)
    assert engine.iteration == 2
    assert np.all(np.isfinite(engine.state.sx))


# -- the test article ----------------------------------------------------------------


def phantom_volume(size: int = 44, n_slices: int = 16) -> np.ndarray:
    """A band-limited 3-D phantom that fits inside the reconstruction circle.

    Smoothed on purpose: a hard-edged phantom is not band-limited, and interpolating one
    -- in the projector, in the shift, anywhere -- introduces displacement-like error of
    its own that would be indistinguishable from an alignment error. Smoothing makes the
    ground truth actually true. ``examples/make_phantom.py`` documents the same trap.
    """
    volume = gaussian_filter(generate_volume(size=size, n_slices=n_slices).astype(np.float64),
                             (0.8, 1.2, 1.2))
    margin = (GRID - size) // 2
    return np.pad(volume, ((0, 0), (margin, margin), (margin, margin)))


def misaligned_dataset(
    *, ramp: float = 0.0, seed: int = 1, max_shift: float = MAX_SHIFT
) -> tuple[ProjectionData, np.ndarray, np.ndarray, ReferenceProjector]:
    """Forward-project the phantom, then corrupt it with known shifts and an optional ramp.

    The shift is injected with ``fourier_shift`` -- an exact translation of a band-limited
    signal, and a *different* implementation from the one the loop corrects with -- so
    recovering the truth proves the loop works rather than that it agrees with itself.
    The vertical vacuum margin is added before the shift so nothing walks out of frame.
    """
    projector = ReferenceProjector()
    angles = np.linspace(0.0, np.pi, N_ANGLES, endpoint=False)
    clean = projector.reproject(phantom_volume(), angles, center=CENTER).astype(np.float64)
    clean = np.pad(clean, ((0, 0), (V_MARGIN, V_MARGIN), (0, 0)))

    rng = np.random.default_rng(seed)
    sx_true = rng.uniform(-max_shift, max_shift, N_ANGLES)
    sy_true = rng.uniform(-max_shift, max_shift, N_ANGLES)
    sx_true -= sx_true.mean()
    sy_true -= sy_true.mean()

    data = np.empty_like(clean)
    for i in range(N_ANGLES):
        data[i] = np.fft.ifftn(fourier_shift(np.fft.fftn(clean[i]), (sy_true[i], sx_true[i]))).real

    if ramp:
        # A horizontal ramp plus a constant, independent per projection: the gauge
        # ptychographic phase retrieval leaves undetermined. Kept horizontal so it does
        # not also corrupt stage 1 -- ramp removal is roadmap step 0 and precedes both
        # stages; what is modelled here is the *residual* after that step.
        n_v, n_u = data.shape[1:]
        u = np.mgrid[0:n_v, 0:n_u][1].astype(np.float64)
        for i in range(N_ANGLES):
            slope, offset = rng.uniform(-ramp, ramp, 2)
            data[i] += slope * u / n_u + offset

    dataset = ProjectionData(data=data.astype(np.float32), angles=angles, metadata={"name": "test"})
    return dataset, sx_true, sy_true, ReferenceProjector()


def observable_error(sx, sy, sx_true, sy_true, angles) -> tuple[float, float, float]:
    """RMS error against truth, ignoring the physically unobservable modes.

    Same construction as ``tests/test_ptycho_engine.py::observable_error`` (repeated
    rather than imported because that module skips itself without tomopy). Translating
    the object in-plane by ``(dx, dy)`` shifts projection ``i`` by
    ``dx*cos(theta) + dy*sin(theta)`` and a constant is absorbed by the rotation centre,
    so the horizontal shifts are only determined up to ``{sin, cos, 1}``; a constant
    vertical shift translates the volume along the axis, so ``{1}`` vertically.

    Returns ``(total, horizontal, vertical)``.
    """
    horizontal_modes = np.column_stack([np.sin(angles), np.cos(angles), np.ones_like(angles)])
    vertical_modes = np.ones((len(angles), 1))

    def observable(vector, modes):
        coefficients, *_ = np.linalg.lstsq(modes, vector, rcond=None)
        return vector - modes @ coefficients

    error_x = observable(sx, horizontal_modes) - observable(sx_true, horizontal_modes)
    error_y = observable(sy, vertical_modes) - observable(sy_true, vertical_modes)
    return (
        float(np.sqrt(np.mean(error_x**2 + error_y**2))),
        float(np.sqrt(np.mean(error_x**2))),
        float(np.sqrt(np.mean(error_y**2))),
    )


def build_engine(dataset, projector, *, gradient=None, config=None, odstrcil=None):
    return OdstrcilEngine(
        dataset=dataset,
        config=config
        or AlignConfig(
            recon_algorithm="sirt", recon_inner_iters=3, mode="joint", row_chunk=1000
        ),
        odstrcil=odstrcil
        or OdstrcilConfig(gradient=gradient or GradientConfig(upsample=50, taper=0.1)),
        center=CENTER,
        projector=projector,
    )


def align(ramp=0.0, domain=None, n_iter=12, seed=1):
    """Run the whole loop and return ``(engine, total, horizontal, vertical)`` error."""
    dataset, sx_true, sy_true, projector = misaligned_dataset(ramp=ramp, seed=seed)
    settings = {"upsample": 50, "taper": 0.1}
    if domain is not None:
        settings["domain"] = domain
    gradient = GradientConfig(**settings)
    engine = build_engine(dataset, projector, gradient=gradient)
    engine.run(n_iter)
    return (engine, *observable_error(engine.state.sx, engine.state.sy, sx_true, sy_true,
                                      dataset.angles))


# -- the substitutions, pinned -------------------------------------------------------


def test_the_substitute_shift_obeys_the_real_apply_shifts_contract():
    """Identical assertions to the ones ``test_ptycho_engine.py`` makes on the real one."""
    frame = np.zeros((32, 32), dtype=np.float32)
    frame[10, 20] = 1.0

    shifted = substitute_apply_shifts(frame[np.newaxis].copy(), np.array([3.0]), np.array([0.0]))
    assert np.unravel_index(np.argmax(shifted[0]), (32, 32)) == (7, 20), "sy must move rows"

    shifted = substitute_apply_shifts(frame[np.newaxis].copy(), np.array([0.0]), np.array([-4.0]))
    assert np.unravel_index(np.argmax(shifted[0]), (32, 32)) == (10, 24), "sx must move columns"

    original = np.random.default_rng(0).random((3, 16, 16)).astype(np.float32)
    reference = original.copy()
    substitute_apply_shifts(original.copy(), np.zeros(3), np.zeros(3))
    np.testing.assert_array_equal(original, reference, "the caller's array was mutated")


def test_the_reference_projector_adjoint_is_exact():
    """SIRT's convergence proof needs the true adjoint, not an approximation.

    The rotate-and-sum projector this replaced was adjoint to only 3e-4 and its SIRT
    oscillated (residual 0.20 -> 1.49 -> 1.05 -> 1.71 over four iterations) rather than
    converging. Anything built on an approximate adjoint should be assumed unstable
    until measured.
    """
    projector = ReferenceProjector()
    angles = np.linspace(0.0, np.pi, 16, endpoint=False)
    matrix, _row, _col = projector._matrix(24, angles, 12.0)

    rng = np.random.default_rng(0)
    x = rng.random((5, 24 * 24))
    y = rng.random((16 * 24, 5))
    forward = float((matrix @ x.T * y).sum())
    backward = float((x * (matrix.T @ y).T).sum())
    assert abs(forward - backward) <= 1e-9 * abs(forward)


def test_the_reference_projector_sirt_converges_monotonically():
    projector = ReferenceProjector()
    angles = np.linspace(0.0, np.pi, N_ANGLES, endpoint=False)
    truth = np.pad(
        projector.reproject(phantom_volume(), angles, center=CENTER).astype(np.float64),
        ((0, 0), (V_MARGIN, V_MARGIN), (0, 0)),
    )
    # Ground truth is read-only. Besides being good hygiene, it is load-bearing here:
    # see test_numpy_temporary_elision_is_not_corrupting_this_environment.
    truth.setflags(write=False)
    norm_truth = float(np.linalg.norm(truth))

    volume, residuals = None, []
    for _ in range(6):
        volume = projector.reconstruct(truth, angles, center=CENTER, num_iter=3, init_recon=volume)
        simulated = projector.reproject(volume, angles, center=CENTER)
        residuals.append(float(np.linalg.norm(np.subtract(truth, simulated)) / norm_truth))

    assert residuals == sorted(residuals, reverse=True), f"SIRT did not converge: {residuals}"
    assert residuals[-1] < 0.06


def test_numpy_temporary_elision_is_not_corrupting_this_environment():
    """A canary for a numpy/CPython interaction that silently destroys input arrays.

    numpy's "temporary elision" optimisation reuses the left operand's buffer for the
    result of ``a - b`` when it believes ``a`` is a temporary, which it decides from the
    reference count. CPython 3.14's ``LOAD_FAST_BORROW`` does not incref, so on
    numpy < 2.3 an ordinary named array passed as ``np.linalg.norm(a - b)`` is judged a
    temporary and ``a`` is overwritten with the difference -- no error, no warning, and
    every subsequent use of ``a`` is wrong. It cost an afternoon here: the SIRT test
    above appeared to diverge because its ground truth was being eaten between
    iterations.

    Where it matters this package writes ``np.subtract(a, b)``, which is not elided, and
    marks ground truth read-only, which numpy refuses to elide into. This test asserts
    that both defences work on whatever interpreter is actually running, and prints the
    versions when they do not.
    """
    a = np.pad(np.ones((8, 4, 16), dtype=np.float64), ((0, 0), (2, 2), (0, 0)))
    b = np.full((8, 8, 16), 0.25, dtype=np.float32)

    checksum = float(a.sum())
    np.linalg.norm(np.subtract(a, b))
    assert float(a.sum()) == checksum, (
        f"np.subtract still elided on numpy {np.__version__} / python "
        f"{sys.version.split()[0]} -- no construct is safe; use an explicit out= buffer"
    )

    a.setflags(write=False)
    np.linalg.norm(a - b)
    assert float(a.sum()) == checksum, "a read-only array was elided into"


# -- the gate ------------------------------------------------------------------------


def test_recovers_known_shifts_within_a_third_of_a_voxel():
    """THE GATE. The roadmap's accuracy target, on injected known shifts.

    Measured with the default configuration (SIRT, joint, ``domain="gradient-both"``,
    12 outer iterations): about 0.167 px horizontally and 0.015 px vertically, against a
    target of 0.333 px. The vertical number is an order of magnitude better than the
    horizontal one, which is the decoupling working as advertised -- stage 1 solves a
    well-posed 1-D problem exactly, while stage 2 is limited by how faithfully a
    half-converged SIRT volume reprojects.
    """
    engine, total, horizontal, vertical = align()

    assert total < TARGET_PX, f"recovery error {total:.4f} px exceeds the {TARGET_PX:.3f} px target"
    assert horizontal < TARGET_PX, f"horizontal {horizontal:.4f} px"
    assert vertical < 0.1, f"vertical {vertical:.4f} px -- stage 1 should be far better than this"
    assert vertical < horizontal, "the decoupled vertical stage should beat the horizontal one"

    # And the loop must actually be converging, not hunting.
    history = engine.history
    assert history[-1].residual < history[0].residual / 5
    assert history[-1].error < history[0].error / 5
    assert not any(step.diverging for step in history)
    assert all(step.runaway is None for step in history)


def test_it_beats_doing_nothing_and_beats_the_vertical_stage_alone():
    """A number is only meaningful against a baseline."""
    dataset, sx_true, sy_true, projector = misaligned_dataset()
    nothing, *_ = observable_error(
        np.zeros(N_ANGLES), np.zeros(N_ANGLES), sx_true, sy_true, dataset.angles
    )

    engine = build_engine(dataset, projector)
    engine.align_vertical_stage()
    vertical_only, *_ = observable_error(
        engine.state.sx, engine.state.sy, sx_true, sy_true, dataset.angles
    )
    engine.run(12)
    both, *_ = observable_error(engine.state.sx, engine.state.sy, sx_true, sy_true, dataset.angles)

    assert vertical_only < nothing
    assert both < vertical_only / 4
    assert both < TARGET_PX


# -- the gradient claim, end to end --------------------------------------------------


RAMP = 4.0  # radians across the frame; the phantom's own line integrals peak near 10


@pytest.fixture(scope="module")
def ramp_sweep():
    """Recovery error for each registration domain, with and without a residual ramp."""
    results = {}
    for domain in ("value", "gradient-x", "gradient-both"):
        for ramp in (0.0, RAMP):
            _engine, _total, horizontal, _vertical = align(ramp=ramp, domain=domain)
            results[domain, ramp] = horizontal
    return results


def test_a_residual_ramp_wrecks_the_value_domain_and_barely_touches_the_gradient(ramp_sweep):
    """The system-level version of the gradient claim.

    Measured horizontal recovery error (px):

        domain           clean   ramp 4 rad   degradation
        value            0.081     0.548         6.8x
        gradient-x       0.256     0.389         1.5x
        gradient-both    0.167     0.263         1.6x

    Read the last column first. The value domain loses most of an order of magnitude; the
    gradient domains move by about half again. Only ``gradient-both`` -- the default --
    is still inside the 1/3-voxel target at this ramp amplitude; ``gradient-x`` misses it
    at 0.389 px, which is part of why it is not the default. On perfectly clean data the
    value domain is in fact the *most* accurate of the three: with no gauge ambiguity
    there is nothing for the derivative to buy and it only costs bandwidth. Ramps are not
    optional in ptychography, so the clean column is the hypothetical one.

    Honest caveat, and it is the reason the gradient columns are not perfectly flat:
    differentiating makes the *registration* immune to the ramp (exactly so -- see
    ``test_gradient_registration.py``, 4e-16 px), but the ramp is still in the data the
    volume is reconstructed from, and a ramp-corrupted volume reprojects to something
    slightly wrong no matter how it is compared. That residue is why ramp removal stays
    step 0 of the pipeline and this is a second line of defence, not a replacement.
    """
    for domain in ("gradient-x", "gradient-both"):
        clean = ramp_sweep[domain, 0.0]
        ramped = ramp_sweep[domain, RAMP]
        assert ramped < 2.0 * clean, f"{domain} degraded {ramped / clean:.1f}x under the ramp"

    # Only the default is claimed to hold the accuracy target through a residual ramp.
    assert ramp_sweep["gradient-both", RAMP] < TARGET_PX, (
        f"the default domain must still meet the {TARGET_PX:.3f} px target with a residual "
        f"ramp; got {ramp_sweep['gradient-both', RAMP]:.4f} px"
    )
    assert ramp_sweep["gradient-x", RAMP] > TARGET_PX, (
        "gradient-x is documented as missing the target at this ramp amplitude, which is "
        "part of the case for gradient-both being the default; if that has changed, the "
        "tables in gradient.py and here need updating"
    )

    value_clean = ramp_sweep["value", 0.0]
    value_ramped = ramp_sweep["value", RAMP]
    assert value_ramped > 4.0 * value_clean, (
        f"expected the value domain to collapse under the ramp; it went "
        f"{value_clean:.4f} -> {value_ramped:.4f} px"
    )
    assert value_ramped > TARGET_PX, "the value domain should miss the target under a ramp"
    assert value_ramped > 1.5 * ramp_sweep["gradient-both", RAMP]


def test_on_perfectly_clean_data_the_value_domain_is_the_more_accurate(ramp_sweep):
    """Stated plainly rather than buried: the gradient is not free.

    Differentiation is a high-pass filter, and it puts the comparison's weight in the
    band where a half-converged SIRT reprojection is least faithful. With no ramp at all
    that is a pure cost. Anyone benchmarking on noiseless, ramp-free simulations will
    find the gradient method looks worse, and they will be right about the simulation and
    wrong about the instrument.
    """
    assert ramp_sweep["value", 0.0] < ramp_sweep["gradient-both", 0.0]
    # ...and keeping both derivatives recovers most of the gap that x-only gives away.
    assert ramp_sweep["gradient-both", 0.0] < ramp_sweep["gradient-x", 0.0]


# -- stage 1 is free -----------------------------------------------------------------


def test_the_vertical_stage_runs_before_and_without_any_reconstruction():
    """The decoupling claim, made checkable by counting backend calls."""
    dataset, _sx, sy_true, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)

    result = engine.align_vertical_stage()

    assert projector.reconstruct_calls == 0
    assert projector.reproject_calls == 0
    assert result.converged
    assert result.truncation_reason is None
    np.testing.assert_allclose(engine.state.sy, result.sy)
    assert float(np.sqrt(np.mean((result.sy - sy_true) ** 2))) < 0.1


def test_step_runs_stage_one_first_then_stops_re_running_it():
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)

    assert engine.vertical is None
    engine.step()
    assert engine.vertical is not None
    first = engine.vertical

    engine.step()
    assert engine.vertical is first, "stage 1 must not re-run by default"

    # ...unless asked, which costs a row-sum and nothing else.
    engine.odstrcil.refine_vertical_each_iteration = True
    engine.step()
    assert engine.vertical is not first


def test_the_horizontal_stage_does_not_move_the_vertical_shifts():
    """Vertical is solved once and then frozen -- the roadmap's order of operations."""
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)
    engine.step()
    after_stage_one = engine.state.sy.copy()

    for _ in range(3):
        result = engine.step()
        np.testing.assert_array_equal(result.dsy, np.zeros(N_ANGLES))
    np.testing.assert_allclose(engine.state.sy, after_stage_one, atol=1e-12)

    # The vertical component of the gradient registration is still computed, as a
    # cross-check: if stage 1 were wrong, this would not be small.
    assert engine.vertical_cross_check < 0.5


def test_the_gradient_vertical_can_be_opted_into():
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(
        dataset,
        projector,
        odstrcil=OdstrcilConfig(
            gradient=GradientConfig(upsample=50, taper=0.1), use_gradient_vertical=True
        ),
    )
    engine.step()
    result = engine.step()
    assert np.any(result.dsy != 0.0)


# -- what it refuses to do -----------------------------------------------------------


def test_gridrec_on_a_limited_angular_range_is_refused():
    """Documented as failing outright, so it is refused in code rather than in a README."""
    dataset, _sx, _sy, projector = misaligned_dataset()
    wedge = ProjectionData(
        data=dataset.data,
        angles=np.linspace(0.0, np.deg2rad(120.0), N_ANGLES),  # a 60-degree missing wedge
        metadata={},
    )

    with pytest.raises(LimitedAngleError, match="gridrec"):
        build_engine(
            wedge,
            projector,
            config=AlignConfig(recon_algorithm="gridrec", row_chunk=1000),
            odstrcil=OdstrcilConfig(direct_recon_policy="refuse"),
        )

    # Still refused under "warn": this combination does not converge slowly, it produces
    # garbage, so a warning would be the wrong response.
    with pytest.raises(LimitedAngleError, match="Fourier space"):
        build_engine(
            wedge,
            projector,
            config=AlignConfig(recon_algorithm="gridrec", row_chunk=1000),
            odstrcil=OdstrcilConfig(direct_recon_policy="warn"),
        )


def test_initialising_from_a_direct_reconstruction_is_refused_by_default():
    dataset, _sx, _sy, projector = misaligned_dataset()

    for algorithm in ("fbp", "gridrec"):
        with pytest.raises(LimitedAngleError, match="direct algorithm"):
            build_engine(
                dataset, projector, config=AlignConfig(recon_algorithm=algorithm, row_chunk=1000)
            )

    # "warn" allows it on a full angular range, loudly, and records why.
    engine = build_engine(
        dataset,
        projector,
        config=AlignConfig(recon_algorithm="fbp", row_chunk=1000),
        odstrcil=OdstrcilConfig(direct_recon_policy="warn"),
    )
    assert engine.recon_warning is not None
    assert "streak" in engine.recon_warning
    assert "WARNING" in engine.describe()
    engine.step()  # and it does run


def test_the_refusal_message_says_what_to_do_instead():
    """A refusal that does not name the alternative just blocks the user."""
    coverage = angular_coverage(np.linspace(0.0, np.pi, 90, endpoint=False))
    assert check_reconstruction_choice("sirt", coverage) is None
    assert check_reconstruction_choice("mlem", coverage) is None

    with pytest.raises(LimitedAngleError) as excinfo:
        check_reconstruction_choice("fbp", coverage)
    assert "Use 'sirt'" in str(excinfo.value)
    assert "direct_recon_policy" in str(excinfo.value)

    # "allow" is the deliberate escape hatch: it returns the reason instead of raising.
    assert check_reconstruction_choice("fbp", coverage, policy="allow") is not None
    with pytest.raises(ValueError, match="policy"):
        check_reconstruction_choice("fbp", coverage, policy="whatever")


def test_emission_algorithms_are_warned_about_on_negative_data(caplog):
    """Inherited from the engine's own guard; the decoupled loop must not lose it."""
    dataset, _sx, _sy, projector = misaligned_dataset()
    negative = ProjectionData(data=-np.abs(dataset.data), angles=dataset.angles, metadata={})

    with caplog.at_level(logging.WARNING):
        build_engine(negative, projector, config=AlignConfig(recon_algorithm="mlem", row_chunk=1000))
    assert any("emission algorithm" in record.message for record in caplog.records)


# -- angular coverage ----------------------------------------------------------------


def test_angular_coverage_reports_the_scan_honestly():
    full = angular_coverage(np.linspace(0.0, np.pi, 90, endpoint=False))
    assert full.n_views == 90
    assert full.span == pytest.approx(178.0)
    assert full.median_step == pytest.approx(2.0)
    assert not full.limited
    assert full.reason is None
    assert "90 views over 178.000 deg" in full.summary()

    wedge = angular_coverage(np.linspace(0.0, np.deg2rad(120.0), 90))
    assert wedge.limited and "angular span" in wedge.reason

    # A full range with a block of views missing is a missing wedge just the same.
    angles = np.concatenate(
        [np.linspace(0.0, np.deg2rad(60.0), 30), np.linspace(np.deg2rad(120.0), np.pi, 30)]
    )
    gapped = angular_coverage(angles)
    assert gapped.span > 170.0
    assert gapped.limited and "missing wedge" in gapped.reason

    assert angular_coverage(np.array([0.0])).limited


# -- it really is an AlignmentEngine -------------------------------------------------


def test_it_is_a_drop_in_alignment_engine():
    """The seam: a host that can drive an AlignmentEngine can drive this one."""
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)

    assert isinstance(engine, AlignmentEngine)
    result = engine.step()
    assert result.iteration == 1
    assert engine.iteration == 1
    assert engine.last_aligned is not None and engine.last_simulated is not None
    assert engine.state.volume is not None
    assert result.volume.shape == (dataset.data.shape[1], GRID, GRID)

    # Inherited run/revert/history, unchanged.
    engine.run(2)
    assert engine.iteration == 3
    sx_at_2 = engine.history[1].sx.copy()
    engine.state.revert_to(2)
    np.testing.assert_allclose(engine.state.sx, sx_at_2)
    engine.step()
    assert engine.iteration == 3


def test_the_pristine_projections_are_never_modified():
    """Convention 3: the cumulative shift is always applied to the original."""
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)
    reference = engine.state.original.copy()
    engine.run(3)
    np.testing.assert_array_equal(engine.state.original, reference)


def test_stepping_matches_running():
    dataset, _sx, _sy, projector = misaligned_dataset()
    stepped = build_engine(dataset, projector)
    for _ in range(3):
        stepped.step()

    dataset2, _sx2, _sy2, projector2 = misaligned_dataset()
    ran = build_engine(dataset2, projector2)
    ran.run(3)

    np.testing.assert_allclose(stepped.state.sx, ran.state.sx)
    np.testing.assert_allclose(stepped.state.sy, ran.state.sy)


def test_the_degenerate_global_modes_are_removed():
    """A constant shift in either axis translates the volume; it is not a misalignment."""
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)
    engine.run(2)
    assert abs(engine.state.sx.mean()) < 1e-9
    assert abs(engine.state.sy.mean()) < 1e-9


def test_progress_and_cancellation_are_inherited():
    from tktomo.ptycho_align.core import Cancelled

    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector, config=AlignConfig(recon_inner_iters=1, ncore=4))

    seen: list[tuple[float, str]] = []
    engine.step(report=lambda fraction, message: seen.append((fraction, message)))
    assert any("stage 1" in message for _f, message in seen)
    assert any("reconstructing" in message for _f, message in seen)
    assert any("reprojecting" in message for _f, message in seen)
    assert [f for f, _ in seen] == sorted(f for f, _ in seen)

    import threading

    cancel = threading.Event()
    cancel.set()
    engine2 = build_engine(dataset, projector)
    with pytest.raises(Cancelled):
        engine2.step(cancel=cancel)
    assert engine2.iteration == 0 and engine2.history == []


def test_config_conditioning_is_inherited():
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(
        dataset,
        projector,
        config=AlignConfig(recon_inner_iters=2, row_chunk=1000, align_horizontal=False),
    )
    result = engine.step()
    np.testing.assert_array_equal(result.dsx, np.zeros(N_ANGLES))


def test_run_vertical_false_keeps_the_shifts_it_was_given():
    dataset, _sx, sy_true, projector = misaligned_dataset()
    engine = OdstrcilEngine(
        dataset=dataset,
        config=AlignConfig(recon_inner_iters=2, row_chunk=1000),
        odstrcil=OdstrcilConfig(run_vertical=False),
        sy0=sy_true,
        center=CENTER,
        projector=projector,
    )
    engine.step()
    np.testing.assert_allclose(engine.state.sy, sy_true - sy_true.mean(), atol=1e-9)


# -- discovery ------------------------------------------------------------------------


def test_the_series_aligner_registry_offers_both_methods():
    """The UI builds its dropdown from a registry, exactly as it does for the other
    interchangeable pieces (aligners, recon backends)."""
    names = available_series_aligners()
    assert "odstrcil-decoupled" in names
    assert "reprojection-joint" in names

    entry = get_series_aligner("odstrcil-decoupled")
    assert entry.factory is OdstrcilEngine
    assert "GRADIENT" in entry.description  # the dropdown tooltip must say what differs
    assert get_series_aligner("reprojection-joint").factory is AlignmentEngine

    with pytest.raises(KeyError, match="Unknown series aligner"):
        get_series_aligner("nope")


def test_the_config_survives_a_json_round_trip():
    """A session file or a socket has no dataclasses, only dicts.

    The nested VerticalConfig/GradientConfig are the trap: ``AlignConfig(**raw)`` would
    leave them as plain dicts and the engine would fail somewhere unrelated much later.
    """
    import json

    config = OdstrcilConfig(
        vertical=VerticalConfig(subpixel="parabolic", upsample=200, row_range=(4, 40)),
        gradient=GradientConfig(domain="gradient-x", sigma=1.0, taper=0.2, upsample=50),
        direct_recon_policy="warn",
        use_gradient_vertical=True,
    )
    restored = OdstrcilConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert isinstance(restored.vertical, VerticalConfig)
    assert isinstance(restored.gradient, GradientConfig)
    assert restored.gradient.domain == "gradient-x"
    assert restored.direct_recon_policy == "warn"

    with pytest.raises(ValueError, match="OdstrcilConfig field"):
        OdstrcilConfig.from_dict({**config.to_dict(), "regularisation": 0.1})
    with pytest.raises(ValueError, match="GradientConfig field"):
        broken = config.to_dict()
        broken["gradient"] = {**broken["gradient"], "whitening": True}
        OdstrcilConfig.from_dict(broken)


def test_describe_summarises_the_setup_for_a_ui():
    dataset, _sx, _sy, projector = misaligned_dataset()
    engine = build_engine(dataset, projector)
    text = engine.describe()
    assert "Odstrcil decoupled aligner" in text
    assert f"{N_ANGLES} views" in text
    assert "sirt" in text and "gradient-both" in text

    ramp_blind = build_engine(
        dataset, projector, gradient=GradientConfig(domain="value", upsample=50)
    )
    assert "NOT ramp-invariant" in ramp_blind.describe()


def test_default_config_is_sirt_and_reacts_to_a_limited_range():
    align_config, odstrcil_config = odstrcil_module.default_odstrcil_config(
        ProjectionData(
            data=np.zeros((90, 2, 2)), angles=np.linspace(0.0, np.pi, 90, endpoint=False)
        )
    )
    assert align_config.recon_algorithm == "sirt"
    assert odstrcil_config.gradient.is_ramp_invariant

    limited, _ = odstrcil_module.default_odstrcil_config(
        ProjectionData(data=np.zeros((90, 2, 2)), angles=np.linspace(0.0, np.deg2rad(90.0), 90))
    )
    assert limited.recon_inner_iters > align_config.recon_inner_iters
