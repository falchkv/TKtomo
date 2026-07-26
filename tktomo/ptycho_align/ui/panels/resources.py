"""Live RAM and CPU readout.

A run is long (iterations of tens of minutes are normal) and its two failure modes are
invisible from the alignment views: the process creeping towards the memory ceiling
until the kernel kills it -- which looks like the window simply vanishing, with no
traceback -- and the reconstruction quietly running on one core when it was meant to
use eight. Both are obvious the moment they are plotted.

The readings themselves live in :mod:`tktomo.ptycho_align.core.telemetry`, which is
Qt-free: once the compute runs on a cluster this panel has to plot *that* machine, not
the one drawing the window, so it takes a :class:`ResourceSample` from wherever it came
from rather than reading /proc itself.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tktomo.ptycho_align.core.telemetry import ResourceMonitor, ResourceSample

_HISTORY_SECONDS = 300
_INTERVAL_MS = 1000


class ResourceView(QWidget):
    """Plots CPU load, resident set size, and free RAM for the machine doing the work."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        import pyqtgraph as pg  # noqa: PLC0415 - lazy, per the package's layering rule

        self._points = max(1, _HISTORY_SECONDS * 1000 // _INTERVAL_MS)
        self._t: list[float] = []
        self._cpu: list[float] = []
        self._rss: list[float] = []
        self._free: list[float] = []
        self._monitor = ResourceMonitor()
        self._label = "this machine"
        self._started = time.monotonic()

        self.info_label = QLabel("-")

        self.cpu_plot = pg.PlotWidget()
        self.cpu_plot.setLabel("left", "CPU", units="%")
        self.cpu_plot.setYRange(0, 100)
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.2)
        self._cpu_curve = self.cpu_plot.plot(pen=pg.mkPen("#4a9", width=2))

        self.ram_plot = pg.PlotWidget()
        self.ram_plot.setLabel("left", "RAM", units="GB")
        self.ram_plot.setLabel("bottom", "time", units="s")
        self.ram_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ram_plot.addLegend(offset=(-10, 10))
        self._rss_curve = self.ram_plot.plot(pen=pg.mkPen("#d66", width=2), name="process")
        self._free_curve = self.ram_plot.plot(
            pen=pg.mkPen("#89a", width=1, style=pg.QtCore.Qt.DashLine), name="free"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.cpu_plot, 1)
        layout.addWidget(self.ram_plot, 1)

        if not self._monitor.supported:
            self.info_label.setText("Resource monitoring needs /proc (Linux).")
            return

        self._timer = QTimer(self)
        self._timer.setInterval(_INTERVAL_MS)
        self._timer.timeout.connect(self.sample)
        self._timer.start()
        self.sample()

    def follow_remote(self, label: str) -> None:
        """Stop polling locally and wait to be fed by :meth:`show_sample`.

        Once the engine runs elsewhere, the numbers under this window describe the wrong
        machine -- and saying "8 GB free" about a workstation while the cluster node is
        about to be OOM-killed is worse than showing nothing.
        """
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
        self._label = label
        self._reset_series()

    def sample(self) -> None:
        """Take one local reading. Public so a test can drive it without the timer."""
        reading = self._monitor.sample()
        if reading is not None:
            self.show_sample(reading)

    def show_sample(self, reading: ResourceSample) -> None:
        """Plot one reading, wherever it was taken."""
        self._t.append(time.monotonic() - self._started)
        self._cpu.append(reading.cpu_percent)
        self._rss.append(reading.rss_bytes / 1024**3)
        self._free.append(reading.headroom_bytes / 1024**3)
        for series in (self._t, self._cpu, self._rss, self._free):
            del series[: -self._points]

        self._cpu_curve.setData(self._t, self._cpu)
        self._rss_curve.setData(self._t, self._rss)
        self._free_curve.setData(self._t, self._free)

        # Report the cgroup ceiling when there is one: on a cluster node the machine
        # total is the whole box, which is not the number that gets you OOM-killed.
        if reading.cgroup_limit is not None:
            ceiling = f"{reading.cgroup_limit / 1024**3:.1f} GB (cgroup)"
        else:
            ceiling = f"{reading.ram_total / 1024**3:.1f} GB"

        self.info_label.setText(
            f"{self._label}: CPU {reading.cpu_percent:5.1f}% of {reading.cpu_count} cores"
            f"   |   process {reading.rss_bytes / 1024**3:.2f} GB"
            f"   |   {reading.headroom_bytes / 1024**3:.2f} GB free of {ceiling}"
        )

    def _reset_series(self) -> None:
        self._t.clear()
        self._cpu.clear()
        self._rss.clear()
        self._free.clear()
        self._started = time.monotonic()

    def peak_rss_gb(self) -> float:
        return float(np.max(self._rss)) if self._rss else 0.0
