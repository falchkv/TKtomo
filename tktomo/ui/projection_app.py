"""Projection viewer: scroll between projections in a stack.

Run with::

    python -m tktomo.ui.projection_app [file.h5]
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QMainWindow

from tktomo.io import load_projections
from tktomo.io.phantom import generate_phantom
from tktomo.ui.common import SliceViewer, run_app


class ProjectionWindow(QMainWindow):
    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo — Projections")
        proj = load_projections(path) if path else generate_phantom()
        self.data = proj
        # Scroll axis 0 = projection angle.
        self.viewer = SliceViewer(proj.data, slice_label="Projection")
        self.setCentralWidget(self.viewer)
        self.resize(800, 700)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    return run_app(lambda: ProjectionWindow(path))


if __name__ == "__main__":
    raise SystemExit(main())
