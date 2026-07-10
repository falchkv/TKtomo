"""Feature-based alignment tool: align a pair of images by manual marks.

Place marks on either image with the keyboard while hovering the mouse:

* **Number keys 0-9** place (or replace) a *labelled* mark. Marks that share a
  label on the two images form a known correspondence.
* **"a"** places an *unlabelled* mark. Unlabelled marks have no fixed partner;
  their correspondence is recovered by RANSAC.
* **Delete / Backspace** removes the mark nearest the cursor.

Labelled marks are fitted by least squares; unlabelled marks are matched
robustly by RANSAC (toggle + samples slider). The right-hand panel overlays the
fixed image (green) with the warped moving image (magenta) so aligned structure
appears grey/white. See ``docs/feature_alignment.md`` for the method.

Run with::

    python -m tktomo.ui.feature_alignment_app
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tktomo.align import Transform, apply_transform, estimate_feature_transform
from tktomo.ui.common import run_app


def _display_image(item: pg.ImageItem, image: np.ndarray) -> None:
    """Show a ``(row, col)`` image so data x=col, y=row (with an inverted y).

    RGB input must be ``uint8``: pyqtgraph's float-RGB paint path needs explicit
    levels and segfaults without them, so we hand it the canonical 8-bit format.
    """
    if image.ndim == 3:  # RGB(A): (row, col, ch) -> (col, row, ch)
        rgb = np.ascontiguousarray(image.transpose(1, 0, 2))
        item.setImage(rgb, autoLevels=False, levels=(0, 255))
    else:
        item.setImage(np.ascontiguousarray(image.T), autoLevels=True)


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def demo_bird(size: int = 256) -> np.ndarray:
    """A stylized, front-facing eagle with wings spread.

    Drawn with ``skimage.draw`` (a core dependency) so the demo needs no image
    files. It is deliberately feature-rich -- wing-feather tips, tail-feather
    tips, eyes, beak tip and talons all make good marking targets.
    """
    from skimage.draw import disk, ellipse, line, polygon  # noqa: PLC0415

    S = size
    img = np.linspace(0.22, 0.04, S)[:, None] * np.ones((S, S))  # sky gradient

    def poly(rows, cols, val):
        rr, cc = polygon(np.array(rows) * S, np.array(cols) * S, shape=(S, S))
        img[rr, cc] = val

    def blob(r, c, rad, val):
        rr, cc = disk((r * S, c * S), rad * S, shape=(S, S))
        img[rr, cc] = val

    def seg(r0, c0, r1, c1, val):
        rr, cc = line(int(r0 * S), int(c0 * S), int(r1 * S), int(c1 * S))
        ok = (rr >= 0) & (rr < S) & (cc >= 0) & (cc < S)
        img[rr[ok], cc[ok]] = val

    # Wings (spread), feathered outer edge -> many corner features.
    left_rows = [0.46, 0.35, 0.30, 0.34, 0.28, 0.33, 0.27, 0.34, 0.30,
                 0.44, 0.50, 0.52, 0.50]
    left_cols = [0.45, 0.34, 0.30, 0.26, 0.20, 0.17, 0.10, 0.07, 0.02,
                 0.10, 0.22, 0.36, 0.45]
    poly(left_rows, left_cols, 0.42)
    poly(left_rows, [1 - c for c in left_cols], 0.42)  # right wing (mirror)
    for k in range(1, 5):
        f = k / 5.0
        seg(0.48, 0.45 - 0.42 * f, 0.30 + 0.02 * k, 0.30 - 0.28 * f, 0.25)
        seg(0.48, 0.55 + 0.42 * f, 0.30 + 0.02 * k, 0.70 + 0.28 * f, 0.25)

    # Tail: bold fan of feathers, tips are features.
    for off in np.linspace(-0.15, 0.15, 7):
        poly([0.72, 0.93, 0.93], [0.5, 0.5 + off - 0.028, 0.5 + off + 0.028], 0.46)
        seg(0.72, 0.5, 0.93, 0.5 + off, 0.26)

    # Body torso + proud chest patch.
    rr, cc = ellipse(0.57 * S, 0.5 * S, 0.20 * S, 0.11 * S, shape=(S, S))
    img[rr, cc] = 0.52
    rr, cc = ellipse(0.52 * S, 0.5 * S, 0.10 * S, 0.065 * S, shape=(S, S))
    img[rr, cc] = 0.66

    # Short legs + talons (claw tips are features).
    for lc in (0.462, 0.538):
        seg(0.70, lc, 0.745, lc, 0.82)
        for claw in (-0.028, 0.0, 0.028):
            seg(0.745, lc, 0.775, lc + claw, 0.88)

    # Head (white), hooked beak, fierce eyes.
    blob(0.31, 0.5, 0.095, 0.92)
    poly([0.37, 0.37, 0.47, 0.455], [0.475, 0.525, 0.505, 0.49], 0.99)
    seg(0.40, 0.50, 0.46, 0.50, 0.62)
    for ec, brow in ((0.462, (0.255, 0.42, 0.30, 0.458)), (0.538, (0.255, 0.58, 0.30, 0.542))):
        seg(*brow, 0.12)
        blob(0.30, ec, 0.019, 0.05)
        blob(0.293, ec + 0.006, 0.006, 1.0)

    return np.clip(img, 0.0, 1.0)


class _MarkableImage(QWidget):
    """An image on which labelled/unlabelled marks are placed by keyboard."""

    changed = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._marks: list[tuple[str | None, float, float]] = []  # (label, x, y)
        self._mouse: tuple[float, float] | None = None

        self.plot = pg.PlotWidget()
        self.plot.setFocusPolicy(Qt.NoFocus)  # let key events reach this widget
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.scatter = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen("y", width=2), brush=pg.mkBrush(255, 255, 0, 60)
        )
        self.plot.addItem(self.scatter)
        self._labels: list[pg.TextItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        layout.addWidget(self.plot)

        self.setFocusPolicy(Qt.StrongFocus)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # -- data -------------------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        _display_image(self.image_item, image)

    def labelled(self) -> dict[str, tuple[float, float]]:
        return {lab: (x, y) for lab, x, y in self._marks if lab is not None}

    def unlabelled(self) -> list[tuple[float, float]]:
        return [(x, y) for lab, x, y in self._marks if lab is None]

    def clear(self) -> None:
        self._marks.clear()
        self._redraw()
        self.changed.emit()

    # -- interaction ------------------------------------------------------
    def enterEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self.setFocus()
        super().enterEvent(event)

    def _on_mouse_moved(self, scene_pos) -> None:
        vb = self.plot.getPlotItem().vb
        if self.plot.sceneBoundingRect().contains(scene_pos):
            point = vb.mapSceneToView(scene_pos)
            self._mouse = (point.x(), point.y())

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        text = event.text()
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._remove_nearest()
        elif text.isdigit():
            self._add_mark(text)
        elif text in ("a", "A"):
            self._add_mark(None)
        else:
            super().keyPressEvent(event)

    def _add_mark(self, label: str | None) -> None:
        if self._mouse is None:
            return
        x, y = self._mouse
        if label is not None:  # a label is unique per image: replace any existing
            self._marks = [m for m in self._marks if m[0] != label]
        self._marks.append((label, x, y))
        self._redraw()
        self.changed.emit()

    def _remove_nearest(self) -> None:
        if self._mouse is None or not self._marks:
            return
        mx, my = self._mouse
        i = min(
            range(len(self._marks)),
            key=lambda k: (self._marks[k][1] - mx) ** 2 + (self._marks[k][2] - my) ** 2,
        )
        del self._marks[i]
        self._redraw()
        self.changed.emit()

    def _redraw(self) -> None:
        self.scatter.setData(
            x=[m[1] for m in self._marks], y=[m[2] for m in self._marks]
        )
        for text_item in self._labels:
            self.plot.removeItem(text_item)
        self._labels.clear()
        for label, x, y in self._marks:
            text = pg.TextItem(label if label is not None else "•", color="y", anchor=(0.5, 1.2))
            text.setPos(x, y)
            self.plot.addItem(text)
            self._labels.append(text)


class FeatureAlignmentWindow(QMainWindow):
    def __init__(
        self, fixed: np.ndarray | None = None, moving: np.ndarray | None = None
    ) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo — Feature Alignment")
        if fixed is None or moving is None:
            fixed, moving = self._demo_pair()
        self.fixed_img = np.asarray(fixed, dtype=np.float64)
        self.moving_img = np.asarray(moving, dtype=np.float64)
        self.transform = Transform()

        self._build_ui()
        self.fixed_view.set_image(self.fixed_img)
        self.moving_view.set_image(self.moving_img)
        self._align()

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        self.fixed_view = _MarkableImage("Fixed (reference) — mark features")
        self.moving_view = _MarkableImage("Moving (to align) — mark features")
        self.fixed_view.changed.connect(self._align)
        self.moving_view.changed.connect(self._align)

        self.overlay_plot = pg.PlotWidget()
        self.overlay_plot.setAspectLocked(True)
        self.overlay_plot.invertY(True)
        self.overlay_item = pg.ImageItem()
        self.overlay_plot.addItem(self.overlay_item)
        overlay_box = QGroupBox("Overlay — fixed (green) vs aligned moving (magenta)")
        overlay_layout = QVBoxLayout(overlay_box)
        overlay_layout.addWidget(self.overlay_plot)

        panels = QSplitter(Qt.Horizontal)
        panels.addWidget(self.fixed_view)
        panels.addWidget(self.moving_view)
        panels.addWidget(overlay_box)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(panels, stretch=1)
        layout.addWidget(self._build_controls())
        self.setCentralWidget(central)
        self.resize(1400, 720)

    def _build_controls(self) -> QWidget:
        self.ransac_check = QCheckBox("Use RANSAC for unlabelled marks")
        self.ransac_check.setChecked(True)
        self.ransac_check.toggled.connect(self._align)

        self.samples_slider = QSlider(Qt.Horizontal)
        self.samples_slider.setRange(1, 3000)
        self.samples_slider.setValue(500)
        self.samples_slider.setMinimumWidth(180)
        self.samples_value = QLabel("500")
        self.samples_slider.valueChanged.connect(
            lambda v: (self.samples_value.setText(str(v)), self._align())
        )

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 50.0)
        self.threshold_spin.setValue(5.0)
        self.threshold_spin.setSuffix(" px")
        self.threshold_spin.valueChanged.connect(self._align)

        align_btn = QPushButton("Align")
        align_btn.clicked.connect(self._align)
        clear_btn = QPushButton("Clear marks")
        clear_btn.clicked.connect(self._clear_marks)

        self.status = QLabel()
        instructions = QLabel(
            "Hover an image · 0–9 = labelled mark · a = unlabelled mark · Del = remove nearest"
        )
        instructions.setStyleSheet("color: gray;")

        row = QHBoxLayout()
        row.addWidget(self.ransac_check)
        row.addWidget(QLabel("RANSAC samples:"))
        row.addWidget(self.samples_slider)
        row.addWidget(self.samples_value)
        row.addWidget(QLabel("inlier tol:"))
        row.addWidget(self.threshold_spin)
        row.addWidget(align_btn)
        row.addWidget(clear_btn)
        row.addStretch(1)
        row.addWidget(self.status)

        outer = QVBoxLayout()
        outer.addLayout(row)
        outer.addWidget(instructions)
        container = QWidget()
        container.setLayout(outer)
        return container

    @staticmethod
    def _demo_pair() -> tuple[np.ndarray, np.ndarray]:
        fixed = demo_bird(256)
        moving = apply_transform(fixed, Transform(dx=14.0, dy=-9.0, rotation=6.0))
        return fixed, moving

    # -- alignment --------------------------------------------------------
    def _clear_marks(self) -> None:
        self.fixed_view.clear()
        self.moving_view.clear()

    def _matched_labels(self):
        fixed = self.fixed_view.labelled()
        moving = self.moving_view.labelled()
        shared = sorted(set(fixed) & set(moving))
        lab_moving = np.array([moving[k] for k in shared]) if shared else np.empty((0, 2))
        lab_fixed = np.array([fixed[k] for k in shared]) if shared else np.empty((0, 2))
        return lab_moving, lab_fixed, shared

    def _align(self) -> None:
        lab_moving, lab_fixed, shared = self._matched_labels()
        unlab_moving = self.moving_view.unlabelled()
        unlab_fixed = self.fixed_view.unlabelled()

        result = estimate_feature_transform(
            lab_moving,
            lab_fixed,
            unlab_moving,
            unlab_fixed,
            image_shape=self.fixed_img.shape,
            use_ransac=self.ransac_check.isChecked(),
            n_samples=self.samples_slider.value(),
            threshold=self.threshold_spin.value(),
        )
        self.transform = result.transform
        self._refresh_overlay()
        self._update_status(result)

    def _refresh_overlay(self) -> None:
        warped = apply_transform(self.moving_img, self.transform)
        f = _normalize(self.fixed_img)
        m = _normalize(warped)
        rgb = np.zeros((*f.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (m * 255).astype(np.uint8)  # magenta = red + blue
        rgb[..., 1] = (f * 255).astype(np.uint8)  # green
        rgb[..., 2] = (m * 255).astype(np.uint8)
        _display_image(self.overlay_item, rgb)

    def _update_status(self, result) -> None:
        rms = "n/a" if np.isnan(result.rms_error) else f"{result.rms_error:.2f} px"
        parts = [
            f"labelled: {result.n_labelled}",
            f"RMS: {rms}",
            f"dx={self.transform.dx:.2f} dy={self.transform.dy:.2f} rot={self.transform.rotation:.2f}°",
        ]
        if result.used_ransac:
            parts.insert(1, f"RANSAC inliers: {result.n_inliers}/{result.n_putative}")
        self.status.setText("   ".join(parts))


def main() -> int:
    return run_app(FeatureAlignmentWindow)


if __name__ == "__main__":
    raise SystemExit(main())
