"""Small helpers for launching standalone Qt applications."""

from __future__ import annotations

import sys


def run_app(window_factory) -> int:
    """Create a QApplication, show ``window_factory()``, and run the event loop.

    ``window_factory`` is a zero-arg callable returning a QWidget/QMainWindow.
    Reusing an existing QApplication instance if one is already running.
    """
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    app = QApplication.instance() or QApplication(sys.argv)
    window = window_factory()
    window.show()
    return app.exec()
