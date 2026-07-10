"""PySide6 + pyqtgraph desktop applications.

Submodules import PySide6 / pyqtgraph at module load, so they are imported only
when a UI is actually launched (e.g. ``python -m tktomo.ui.sinogram_app``), never
by the library layer or the test suite.
"""

__all__: list[str] = []
