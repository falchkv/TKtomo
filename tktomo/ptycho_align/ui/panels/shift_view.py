"""Shifts and convergence plots.

Previous iterations are drawn as faded traces behind the current one, so you can
watch the shifts settle (or fail to). The convergence curve is log-y because a
healthy run drops by orders of magnitude and a linear axis hides that entirely.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from tktomo.ptycho_align.core.state import IterationResult

_HISTORY_TRACES = 6  # how many previous iterations to keep on screen


class ShiftView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.sx_plot = pg.PlotWidget(title="Cumulative sx vs angle")
        self.sy_plot = pg.PlotWidget(title="Cumulative sy vs angle")
        self.error_plot = pg.PlotWidget(title="Shift-update RMS vs iteration")
        self.residual_plot = pg.PlotWidget(title="Residual vs iteration")
        self.com_plot = pg.PlotWidget(title="COM sinusoid fit")

        self.error_plot.setLogMode(y=True)

        for plot, x_label, y_label in (
            (self.sx_plot, "angle (deg)", "sx (px)"),
            (self.sy_plot, "angle (deg)", "sy (px)"),
            (self.error_plot, "iteration", "RMS (px)"),
            (self.residual_plot, "iteration", "residual"),
            (self.com_plot, "angle (deg)", "com_u (px)"),
        ):
            plot.setLabel("bottom", x_label)
            plot.setLabel("left", y_label)
            plot.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(plot)

        self._config_markers: list[pg.InfiniteLine] = []

    def update_history(self, history: list[IterationResult], angles: np.ndarray) -> None:
        degrees = np.rad2deg(angles)

        self.sx_plot.clear()
        self.sy_plot.clear()

        # Faded older traces first, so the current iteration draws on top.
        recent = history[-_HISTORY_TRACES:]
        for index, result in enumerate(recent):
            is_current = index == len(recent) - 1
            width = 2 if is_current else 1
            # Older traces fade out; the current one is opaque white.
            alpha = 255 if is_current else int(40 + 120 * index / max(len(recent) - 1, 1))
            colour = (255, 255, 255, alpha) if is_current else (100, 180, 255, alpha)
            pen = pg.mkPen(colour, width=width)
            self.sx_plot.plot(degrees, result.sx, pen=pen)
            self.sy_plot.plot(degrees, result.sy, pen=pen)

        iterations = [r.iteration for r in history]
        # Log-y cannot show a zero, and a perfectly converged update legitimately is
        # zero; floor it at something below any meaningful subpixel shift.
        errors = [max(r.error, 1e-6) for r in history]
        residuals = [r.residual for r in history]

        self.error_plot.clear()
        self.residual_plot.clear()
        if history:
            self.error_plot.plot(
                iterations, errors, pen=pg.mkPen("y", width=2), symbol="o", symbolSize=5
            )
            self.residual_plot.plot(
                iterations, residuals, pen=pg.mkPen("c", width=2), symbol="o", symbolSize=5
            )

        # Mark iterations where the user changed the config, so a kink in the curve
        # can be attributed rather than puzzled over.
        for result in history:
            if result.config_changed:
                for plot in (self.error_plot, self.residual_plot):
                    line = pg.InfiniteLine(
                        pos=result.iteration,
                        angle=90,
                        pen=pg.mkPen("m", style=pg.QtCore.Qt.DotLine),
                        label="config",
                    )
                    plot.addItem(line)

    def update_com(
        self, angles: np.ndarray, com_u: np.ndarray, fitted_u: np.ndarray, residual: float
    ) -> None:
        degrees = np.rad2deg(angles)
        self.com_plot.clear()
        self.com_plot.plot(
            degrees, com_u, pen=None, symbol="o", symbolSize=5, symbolBrush=(255, 80, 80)
        )
        self.com_plot.plot(degrees, fitted_u, pen=pg.mkPen("c", width=2))
        self.com_plot.setTitle(f"COM sinusoid fit (residual {residual:.3f} px)")

    def clear(self) -> None:
        for plot in (self.sx_plot, self.sy_plot, self.error_plot, self.residual_plot):
            plot.clear()
