"""Sinogram viewer: scroll between detector rows (sinograms) and projections.

Subscribes to the messaging bus so alignment results update the sinogram live.

Run with::

    python -m tktomo.ui.sinogram_app [file.h5]
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QGroupBox, QMainWindow, QSplitter, QVBoxLayout

from tktomo.align import Transform, apply_transform
from tktomo.io import ProjectionData, load_projections
from tktomo.io.phantom import generate_phantom
from tktomo.messaging import Message, Publisher, Subscriber
from tktomo.messaging.bus import (
    DEFAULT_REPLY_ADDRESS,
    TOPIC_ALIGNMENT,
    TOPIC_PROJECTION_REPLY,
    TOPIC_PROJECTION_REQUEST,
)
from tktomo.ui.common import SliceViewer, run_app


class SinogramWindow(QMainWindow):
    def __init__(self, path: str | None = None, *, listen: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("TKtomo — Sinograms")
        proj = load_projections(path) if path else generate_phantom()
        self.data = proj
        # Pristine copy so sent shifts are absolute w.r.t. the unshifted
        # projection rather than accumulating across successive sends.
        self._original_data = np.array(proj.data, copy=True)

        self.viewer = SliceViewer(self._as_sinogram_volume(proj.data), slice_label="Row")
        self.projection_viewer = SliceViewer(proj.data, slice_label="Angle")
        self.setCentralWidget(self._build_central())
        self.resize(1200, 700)

        self._subscriber: Subscriber | None = None
        self._reply_publisher: Publisher | None = None
        if listen:
            self._start_listening()

    def _build_central(self) -> QSplitter:
        """Projection viewer and sinogram viewer side by side, resizable."""
        splitter = QSplitter(Qt.Horizontal)
        for title, viewer in (
            ("Projections", self.projection_viewer),
            ("Sinograms", self.viewer),
        ):
            box = QGroupBox(title)
            layout = QVBoxLayout(box)
            layout.addWidget(viewer)
            splitter.addWidget(box)
        return splitter

    @staticmethod
    def _as_sinogram_volume(data: np.ndarray) -> np.ndarray:
        # data is (n_angles, n_rows, width); scroll axis 0 must be rows.
        return np.moveaxis(data, 1, 0)

    def _start_listening(self) -> None:
        try:
            self._subscriber = Subscriber(
                topics=[TOPIC_ALIGNMENT, TOPIC_PROJECTION_REQUEST]
            )
        except Exception:  # pragma: no cover - zmq missing / bind issues
            return
        try:
            self._reply_publisher = Publisher(DEFAULT_REPLY_ADDRESS, bind=True)
        except Exception:  # pragma: no cover - zmq missing / bind issues
            self._reply_publisher = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_bus)
        self._timer.start(100)

    def _poll_bus(self) -> None:
        if self._subscriber is None:
            return
        # Serve every projection request, but coalesce alignment updates: only the
        # latest transform state needs rendering.
        latest_alignment = None
        while (message := self._subscriber.poll(timeout_ms=0)) is not None:
            if message.topic == TOPIC_PROJECTION_REQUEST:
                self._serve_projection(message)
            else:
                latest_alignment = message
        if latest_alignment is not None:
            self._apply_message(latest_alignment)

    def _serve_projection(self, message) -> None:
        """Reply to the alignment tool with a requested projection (and stack size).

        A ``target`` of ``"query"`` only reports ``n_angles`` (so the alignment
        tool can size its selector) and carries no array.
        """
        if self._reply_publisher is None:
            return
        target = message.params.get("target", "moving")
        n_angles = self.data.data.shape[0]
        params: dict = {"n_angles": n_angles, "target": target}
        arrays: dict = {}
        if target != "query":
            index = int(message.params.get("index", 0))
            index = max(0, min(index, n_angles - 1))
            projection = self.data.projection(index)
            if np.iscomplexobj(projection):
                projection = np.abs(projection)
            params["index"] = index
            arrays["projection"] = np.ascontiguousarray(projection, dtype=np.float64)
        self._reply_publisher.publish(
            Message(topic=TOPIC_PROJECTION_REPLY, params=params, arrays=arrays)
        )

    def _apply_message(self, message) -> None:
        """Update the projection stack from an alignment message and refresh."""
        # A fully-formed sinogram/stack array takes precedence if provided.
        if "sinogram" in message.arrays:
            self.viewer.set_volume(message.arrays["sinogram"])
            return

        index = message.params.get("projection_index")
        if index is None:
            return
        index = int(index)
        if not 0 <= index < self._original_data.shape[0]:
            return  # index refers to a projection this stack does not have
        transform = Transform.from_dict(message.params)
        # Absolute shift: warp the ORIGINAL projection (not the currently shown,
        # possibly already-shifted one) so repeated sends do not accumulate.
        new_data = np.array(self.data.data, copy=True)
        new_data[index] = apply_transform(self._original_data[index], transform)
        self.data = ProjectionData(
            data=new_data, angles=self.data.angles, metadata=self.data.metadata
        )
        self.viewer.set_volume(self._as_sinogram_volume(new_data))
        self.projection_viewer.set_volume(new_data)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    return run_app(lambda: SinogramWindow(path))


if __name__ == "__main__":
    raise SystemExit(main())
