"""Alignment tool: tweak the rigid transform between two images.

Features
--------
* Overlay of a fixed (reference) and moving image, shown as their difference.
* Manual dx / dy / rotation entry boxes, each mirrored by a slider.
* Method dropdown populated from the alignment registry; "Apply auto" runs the
  selected method using the current transform as the initial condition.
* Undo button and Ctrl+Z hotkey (backed by ``TransformHistory``).
* "Send to sinogram" publishes the current transform over the messaging bus.

Run with::

    python -m tktomo.ui.alignment_app
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tktomo.align import (
    Transform,
    TransformHistory,
    apply_transform,
    available_aligners,
    get_aligner,
)
from tktomo.io.phantom import generate_volume
from tktomo.messaging import Message, Publisher, Subscriber
from tktomo.messaging.bus import (
    DEFAULT_REPLY_ADDRESS,
    TOPIC_ALIGNMENT,
    TOPIC_PROJECTION_REPLY,
    TOPIC_PROJECTION_REQUEST,
)
from tktomo.ui.common import run_app

# Slider granularity: integer slider steps map to this fraction of a pixel/degree.
_SHIFT_RANGE = 50.0
_ROT_RANGE = 30.0
_STEPS = 1000


class _LinkedControl:
    """A spin box + slider bound to the same value, with a change callback."""

    def __init__(self, value_range: float, decimals: int, on_change) -> None:
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-value_range, value_range)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(10 ** -decimals)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-_STEPS, _STEPS)
        self._range = value_range
        self._guard = False
        self._on_change = on_change
        self.spin.valueChanged.connect(self._from_spin)
        self.slider.valueChanged.connect(self._from_slider)

    def _from_spin(self, value: float) -> None:
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(int(value / self._range * _STEPS))
        self._guard = False
        self._on_change()

    def _from_slider(self, value: int) -> None:
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(value / _STEPS * self._range)
        self._guard = False
        self._on_change()

    def set_value(self, value: float) -> None:
        self._guard = True
        self.spin.setValue(value)
        self.slider.setValue(int(value / self._range * _STEPS))
        self._guard = False

    @property
    def value(self) -> float:
        return self.spin.value()


class _IntControl:
    """An integer spin box + slider bound to the same value (0..maximum)."""

    def __init__(self) -> None:
        self.spin = QSpinBox()
        self.spin.setRange(0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self._guard = False
        self.spin.valueChanged.connect(self._from_spin)
        self.slider.valueChanged.connect(self._from_slider)

    def _from_spin(self, value: int) -> None:
        if self._guard:
            return
        self._guard = True
        self.slider.setValue(value)
        self._guard = False

    def _from_slider(self, value: int) -> None:
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(value)
        self._guard = False

    def set_maximum(self, maximum: int) -> None:
        maximum = max(0, maximum)
        self._guard = True
        self.spin.setMaximum(maximum)
        self.slider.setMaximum(maximum)
        self._guard = False

    @property
    def value(self) -> int:
        return self.spin.value()


class AlignmentWindow(QMainWindow):
    def __init__(
        self,
        fixed: np.ndarray | None = None,
        moving: np.ndarray | None = None,
        *,
        projection_index: int = 0,
    ) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo — Alignment")

        if fixed is None or moving is None:
            fixed, moving = self._demo_pair()
        self.fixed = np.asarray(fixed, dtype=np.float64)
        self.moving = np.asarray(moving, dtype=np.float64)
        self._initial_projection_index = projection_index

        self.history = TransformHistory()
        self._publisher: Publisher | None = None
        self._reply_subscriber: Subscriber | None = None
        self._range_known = False
        self._updating = False

        self._build_ui()
        self._setup_messaging()
        self._refresh_overlay()

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self.image_item = pg.ImageItem()
        view = pg.GraphicsLayoutWidget()
        plot = view.addPlot()
        plot.setAspectLocked(True)
        plot.addItem(self.image_item)

        self.dx = _LinkedControl(_SHIFT_RANGE, 2, self._on_manual_change)
        self.dy = _LinkedControl(_SHIFT_RANGE, 2, self._on_manual_change)
        self.rot = _LinkedControl(_ROT_RANGE, 3, self._on_manual_change)

        self.projection_index_spin = QSpinBox()
        self.projection_index_spin.setRange(0, 100000)
        self.projection_index_spin.setValue(self._initial_projection_index)
        self.projection_index_spin.setToolTip(
            "Which projection index the sinogram viewer should apply this transform to."
        )

        form = QFormLayout()
        form.addRow("dx (px)", self._pair(self.dx))
        form.addRow("dy (px)", self._pair(self.dy))
        form.addRow("rotation (deg)", self._pair(self.rot))
        form.addRow("projection #", self.projection_index_spin)

        # Fetch projections from the running sinogram viewer.
        self.fetch_index = _IntControl()
        fetch_form = QFormLayout()
        fetch_form.addRow("fetch projection #", self._pair(self.fetch_index))

        fetch_ref_btn = QPushButton("Fetch → reference")
        fetch_ref_btn.setToolTip("Load the selected projection as the fixed reference image.")
        fetch_ref_btn.clicked.connect(lambda: self._fetch("fixed"))
        fetch_mov_btn = QPushButton("Fetch → to-align")
        fetch_mov_btn.setToolTip("Load the selected projection as the moving image to align.")
        fetch_mov_btn.clicked.connect(lambda: self._fetch("moving"))

        fetch_buttons = QHBoxLayout()
        fetch_buttons.addWidget(fetch_ref_btn)
        fetch_buttons.addWidget(fetch_mov_btn)

        self.method_combo = QComboBox()
        self.method_combo.addItems(available_aligners())

        auto_btn = QPushButton("Apply auto")
        auto_btn.clicked.connect(self._apply_auto)
        undo_btn = QPushButton("Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self._undo)
        send_btn = QPushButton("Send to sinogram")
        send_btn.clicked.connect(self._send)

        method_row = QHBoxLayout()
        method_row.addWidget(self.method_combo)
        method_row.addWidget(auto_btn)

        buttons = QHBoxLayout()
        buttons.addWidget(undo_btn)
        buttons.addWidget(send_btn)

        controls = QVBoxLayout()
        controls.addLayout(form)
        controls.addLayout(fetch_form)
        controls.addLayout(fetch_buttons)
        controls.addLayout(method_row)
        controls.addLayout(buttons)
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMaximumWidth(360)

        layout = QHBoxLayout()
        layout.addWidget(view, stretch=1)
        layout.addWidget(controls_widget)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1100, 700)

        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self._undo)

    @staticmethod
    def _pair(control: _LinkedControl) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(control.spin)
        row.addWidget(control.slider, stretch=1)
        return w

    @staticmethod
    def _demo_pair() -> tuple[np.ndarray, np.ndarray]:
        fixed = generate_volume(size=256, n_slices=1)[0]
        moving = apply_transform(fixed, Transform(dx=8.0, dy=-5.0, rotation=2.0))
        return fixed, moving

    # -- transform plumbing ----------------------------------------------
    def _current_transform_from_controls(self) -> Transform:
        return Transform(dx=self.dx.value, dy=self.dy.value, rotation=self.rot.value)

    def _set_controls(self, transform: Transform) -> None:
        self._updating = True
        self.dx.set_value(transform.dx)
        self.dy.set_value(transform.dy)
        self.rot.set_value(transform.rotation)
        self._updating = False

    def _on_manual_change(self) -> None:
        if self._updating:
            return
        self.history.push(self._current_transform_from_controls())
        self._refresh_overlay()

    def _apply_auto(self) -> None:
        try:
            aligner = get_aligner(self.method_combo.currentText())
            result = aligner.estimate(self.fixed, self.moving, self.history.current)
        except Exception as exc:
            QMessageBox.warning(self, "Auto-align failed", str(exc))
            return
        self.history.push(result)
        self._set_controls(result)
        self._refresh_overlay()

    def _undo(self) -> None:
        transform = self.history.undo()
        self._set_controls(transform)
        self._refresh_overlay()

    def _refresh_overlay(self, preferred: np.ndarray | None = None) -> None:
        transform = self.history.current
        warped = apply_transform(self.moving, transform)
        if self.fixed.shape == warped.shape:
            # Difference image: zero when perfectly aligned.
            self.image_item.setImage(self.fixed - warped, autoLevels=True)
        else:
            # Reference and to-align images differ in shape (only one has been
            # fetched so far): show the most recently loaded image instead.
            image = preferred if preferred is not None else warped
            self.image_item.setImage(image, autoLevels=True)

    def _send(self) -> None:
        transform = self.history.current
        params = transform.as_dict()
        params["projection_index"] = self.projection_index_spin.value()
        try:
            self._ensure_publisher().publish(Message(topic=TOPIC_ALIGNMENT, params=params))
        except Exception as exc:  # zmq missing / bind failure
            QMessageBox.warning(self, "Send failed", str(exc))

    # -- fetching projections from the sinogram viewer --------------------
    def _setup_messaging(self) -> None:
        """Bind the request publisher and start polling for projection replies."""
        try:  # Bind early so the sinogram viewer's subscriber connects before use.
            self._ensure_publisher()
        except Exception:  # pragma: no cover - zmq missing / bind failure
            self._publisher = None
        try:
            self._reply_subscriber = Subscriber(
                DEFAULT_REPLY_ADDRESS, topics=[TOPIC_PROJECTION_REPLY]
            )
        except Exception:  # pragma: no cover - zmq missing
            self._reply_subscriber = None
        self._reply_timer = QTimer(self)
        self._reply_timer.timeout.connect(self._poll_replies)
        self._reply_timer.start(150)

    def _ensure_publisher(self) -> Publisher:
        if self._publisher is None:
            self._publisher = Publisher()
        return self._publisher

    def _fetch(self, target: str) -> None:
        """Ask the sinogram viewer for the selected projection into ``target``."""
        try:
            self._ensure_publisher().publish(
                Message(
                    topic=TOPIC_PROJECTION_REQUEST,
                    params={"index": self.fetch_index.value, "target": target},
                )
            )
        except Exception as exc:  # zmq missing / bind failure
            QMessageBox.warning(self, "Fetch failed", str(exc))

    def _poll_replies(self) -> None:
        if self._reply_subscriber is None:
            return
        while (message := self._reply_subscriber.poll(timeout_ms=0)) is not None:
            self._handle_reply(message)
        if not self._range_known:
            # Sinogram viewer may not be up yet; keep asking for the stack size so
            # the selector range populates as soon as it appears.
            try:
                self._ensure_publisher().publish(
                    Message(topic=TOPIC_PROJECTION_REQUEST, params={"target": "query"})
                )
            except Exception:  # pragma: no cover - zmq missing / not up yet
                pass

    def _handle_reply(self, message: Message) -> None:
        n_angles = message.params.get("n_angles")
        if n_angles is not None:
            self.fetch_index.set_maximum(int(n_angles) - 1)
            self._range_known = True
        target = message.params.get("target")
        projection = message.arrays.get("projection")
        if projection is None or target not in ("fixed", "moving"):
            return
        image = np.asarray(projection, dtype=np.float64)
        if target == "fixed":
            self.fixed = image
        else:
            self.moving = image
        self._refresh_overlay(preferred=image)


def main() -> int:
    return run_app(AlignmentWindow)


if __name__ == "__main__":
    raise SystemExit(main())
