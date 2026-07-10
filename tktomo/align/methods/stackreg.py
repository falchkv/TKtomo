"""StackReg (TurboReg port) alignment: translation or rigid-body.

Wraps :mod:`pystackreg`, imported lazily so the package installs and imports
without it. Registered so it appears in the UI dropdown; a clear error is raised
at use time if pystackreg is missing.
"""

from __future__ import annotations

import numpy as np

from tktomo.align.base import register_aligner
from tktomo.align.transform import Transform


class StackRegAligner:
    def __init__(self, mode: str = "translation") -> None:
        if mode not in ("translation", "rigid"):
            raise ValueError("mode must be 'translation' or 'rigid'")
        self.mode = mode
        self.name = f"stackreg-{mode}"

    def estimate(
        self,
        fixed: np.ndarray,
        moving: np.ndarray,
        initial: Transform | None = None,
    ) -> Transform:
        try:
            from pystackreg import StackReg  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'stackreg-*' methods require pystackreg. "
                "Install with: pip install pystackreg"
            ) from exc

        transformation = (
            StackReg.TRANSLATION if self.mode == "translation" else StackReg.RIGID_BODY
        )
        sr = StackReg(transformation)
        matrix = sr.register(fixed, moving)  # 3x3 homogeneous, moving -> fixed

        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        rotation = 0.0
        if self.mode == "rigid":
            rotation = float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0])))
        return Transform(dx=dx, dy=dy, rotation=rotation)


register_aligner(StackRegAligner("translation"))
register_aligner(StackRegAligner("rigid"))
