"""Manual feature-tracking alignment: Qt-free math and file formats.

This package backs the `tktomo-feature-isolation` and `tktomo-track-model`
UIs. Everything here is importable in a light environment: numpy at module
level, scipy/h5py lazily inside functions, no Qt anywhere.

The model generalizes a two-stage linear bundle adjustment. A feature at
rotating-frame position (a, b) and height y, viewed in parallel beam while
the object rotates about a near-vertical axis, lands on the detector at

    u_ij = s_ij + c(theta_j) + dx_j
    v_ij = y_i + alpha(theta_j) * s_ij + beta(theta_j) * t_ij + dy_j

with s = a*cos + b*sin (across the beam), t = -a*sin + b*cos (along the
beam), c the rotation-axis position, alpha the in-plane axis tilt, beta the
out-of-plane axis tilt, and (dx_j, dy_j) per-view projection displacements.
c, alpha, beta are low-order polynomials in theta (constants by default).
Both stages are linear: solve u for (a, b, c, dx), then s and t are known
numbers and v is linear in (y, alpha, beta, dy).
"""

from tktomo.tracking.coords import CoordinateChain
from tktomo.tracking.model import (
    AxisModel,
    FitResult,
    FreeMask,
    fill_missing_shifts,
    poly_basis,
    residuals,
    solve_model,
)

__all__ = [
    "AxisModel",
    "CoordinateChain",
    "FitResult",
    "FreeMask",
    "fill_missing_shifts",
    "poly_basis",
    "residuals",
    "solve_model",
]
