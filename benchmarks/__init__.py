"""Ground-truth benchmarking for tomographic alignment.

Not part of the installed ``tktomo`` package -- this is a harness that lives beside
it, imports it, and measures it. Run it as ``python -m benchmarks.runner`` from the
repository root, or drive it from a notebook::

    from benchmarks import synthetic_case, run_benchmark, PerturbationSpec

    case = synthetic_case(spec=PerturbationSpec(jitter_dy_rms=2.5, jitter_dx_rms=0.75))
    report = run_benchmark(case)
    print(report.table())

The one thing to know before reading any number it produces: the primary metric is
**shift-recovery error against known injected truth**, not FSC. FSC is exactly
invariant to a geometry error applied identically to both half-sets, so it will
happily certify a systematically wrong reconstruction -- see
:func:`benchmarks.metrics.fourier_shell_correlation` and ``benchmarks/README.md``.

No measured data ships with this package. The synthetic-from-real generator takes a
volume *path* from the caller and nothing is defaulted to a beamtime location.
"""

from __future__ import annotations

from benchmarks.metrics import (
    DX_GAUGE,
    DY_GAUGE,
    FscResult,
    Plateau,
    ResidualMap,
    ShiftRecovery,
    Timing,
    fourier_shell_correlation,
    remove_gauge,
    reprojection_residual,
    residual_plateau,
    score_shifts,
    split_half_indices,
)
from benchmarks.phantom import (
    P06_LENS1_SPEC,
    BenchmarkCase,
    GroundTruth,
    PerturbationSpec,
    back_project,
    cases_from_catalogue,
    circular_mask,
    forward_project,
    load_angles,
    load_volume,
    perturb,
    synthetic_case,
    synthetic_volume,
    volume_case,
)
from benchmarks.runner import (
    AlignerResult,
    BenchmarkReport,
    JirrAligner,
    JointGdAligner,
    ModuleAligner,
    NullAligner,
    NumpyProjectorBackend,
    OdstrcilAligner,
    OracleAligner,
    comparison_figure,
    default_aligners,
    run_benchmark,
    tomopy_shim,
    undo_shifts,
)

__all__ = [
    "DX_GAUGE",
    "DY_GAUGE",
    "P06_LENS1_SPEC",
    "AlignerResult",
    "BenchmarkCase",
    "BenchmarkReport",
    "FscResult",
    "GroundTruth",
    "JirrAligner",
    "JointGdAligner",
    "ModuleAligner",
    "NullAligner",
    "NumpyProjectorBackend",
    "OdstrcilAligner",
    "OracleAligner",
    "PerturbationSpec",
    "Plateau",
    "ResidualMap",
    "ShiftRecovery",
    "Timing",
    "back_project",
    "cases_from_catalogue",
    "circular_mask",
    "comparison_figure",
    "default_aligners",
    "forward_project",
    "fourier_shell_correlation",
    "load_angles",
    "load_volume",
    "perturb",
    "remove_gauge",
    "reprojection_residual",
    "residual_plateau",
    "run_benchmark",
    "score_shifts",
    "split_half_indices",
    "synthetic_case",
    "synthetic_volume",
    "tomopy_shim",
    "undo_shifts",
    "volume_case",
]
