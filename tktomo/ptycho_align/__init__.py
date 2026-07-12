"""Interactive reprojection alignment for ptychographic tomography.

Re-implements the Gursoy et al. (2017) joint re-projection alignment loop
(``tomopy.prep.alignment.align_joint``) one outer iteration at a time, so the user
can inspect the tomogram, projections, sinograms and shift estimates *between*
iterations -- which TomoPy's own implementation, whose loop is internal and
uninterruptible, cannot do.

``tktomo.ptycho_align.core`` is headless; ``tktomo.ptycho_align.ui`` is the PySide6
shell over it. Importing this package pulls in neither Qt nor tomopy.
"""

from __future__ import annotations

__all__ = ["core"]
