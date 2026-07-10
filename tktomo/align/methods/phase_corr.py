"""Phase cross-correlation alignment (translation only).

Fast, noise-resilient FFT registration with sub-pixel upsampling via
``skimage.registration.phase_cross_correlation``. This is the default method.
"""

from __future__ import annotations

import numpy as np

from tktomo.align.base import register_aligner
from tktomo.align.transform import Transform, apply_transform


class PhaseCorrelationAligner:
    name = "phase-correlation"

    def __init__(self, upsample_factor: int = 20) -> None:
        self.upsample_factor = upsample_factor

    def estimate(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        initial: Transform | None = None,
    ) -> Transform:
        from skimage.registration import phase_cross_correlation  # noqa: PLC0415

        # Apply the current guess first, then estimate the residual shift so the
        # method refines the UI's existing state rather than restarting.
        base = initial or Transform()
        warped = apply_transform(moving, base) if not base.is_identity() else moving

        shift, _error, _phase = phase_cross_correlation(
            fixed, warped, upsample_factor=self.upsample_factor
        )
        # skimage returns (row, col) = (dy, dx) mapping moving -> fixed.
        dy, dx = float(shift[0]), float(shift[1])
        return Transform(dx=base.dx + dx, dy=base.dy + dy, rotation=base.rotation)


register_aligner(PhaseCorrelationAligner())
