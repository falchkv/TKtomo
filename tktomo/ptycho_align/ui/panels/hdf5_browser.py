"""A dataset browser for HDF5 files that do not follow a known layout.

The convention-driven loader in :mod:`tktomo.io.hdf5_loader` probes NXtomo /
DXchange / blender-sim locations. A ptycho reconstruction pipeline will happily
write its projections to ``/reconstruction/object_phase`` instead, so this dialog
shows the file's actual tree and lets the user point at the array they mean --
including which stored axis is the angle axis, which is the other way real files
differ from the expected layout.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tktomo.ptycho_align.core import Hdf5Entry, list_hdf5_datasets, suggest_hdf5_paths
from tktomo.ptycho_align.ui.panels.crop import ComponentCombo, CropBox

_AXIS_ORDERS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

_ENTRY_ROLE = Qt.UserRole


class Hdf5BrowserDialog(QDialog):
    """Pick the projections dataset, the angles dataset and the axis order.

    ``dialog.selection()`` returns the keyword arguments for
    :func:`tktomo.ptycho_align.core.load_dataset`.

    Pass ``entries`` when the file is not on this machine: listing it is the one part of
    browsing that needs to touch the disk, so it is done wherever the disk is and the
    result handed in. Omitting it reads the local file, as before.
    """

    def __init__(
        self,
        path: str,
        parent: QWidget | None = None,
        *,
        entries: list[Hdf5Entry] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Choose datasets - {path.rsplit('/', 1)[-1]}")
        self.path = path

        self.entries = list(entries) if entries is not None else list_hdf5_datasets(path)
        self._data: Hdf5Entry | None = None
        self._angles: Hdf5Entry | None = None

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Shape", "Type"])
        self.tree.setColumnWidth(0, 260)
        self.tree.setMinimumSize(520, 280)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self._use_as_data())
        self._build_tree()

        self.data_label = QLabel("-")
        self.angle_label = QLabel("-")
        for label in (self.data_label, self.angle_label):
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.data_button = QPushButton("Use selected")
        self.data_button.clicked.connect(self._use_as_data)
        self.angle_button = QPushButton("Use selected")
        self.angle_button.clicked.connect(self._use_as_angles)
        self.angle_clear = QPushButton("None (assume 0-180 deg)")
        self.angle_clear.clicked.connect(self._clear_angles)

        self.axis_combo = QComboBox()
        self.axis_combo.setToolTip(
            "Which stored axis is the rotation axis. The alignment needs "
            "(angle, v, u); a file saved as (v, u, angle) is read with (2, 0, 1)."
        )
        self.axis_combo.currentIndexChanged.connect(self._axis_order_changed)

        self.units_combo = QComboBox()
        self.units_combo.addItems(["Auto-detect", "Degrees", "Radians"])
        self.units_combo.setToolTip("Auto-detect treats a span above 2*pi as degrees.")

        self.component_combo = ComponentCombo()

        self.crop_box = CropBox()
        self.crop_box.changed.connect(self._validate)

        self.status = QLabel()
        self.status.setWordWrap(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Projections:", _row(self.data_label, self.data_button))
        form.addRow("Angles:", _row(self.angle_label, self.angle_button, self.angle_clear))
        form.addRow("Axis order:", self.axis_combo)
        form.addRow("Angle units:", self.units_combo)
        form.addRow("Component:", self.component_combo)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the 3-D projection stack, then its angle array."))
        layout.addWidget(self.tree)
        layout.addLayout(form)
        layout.addWidget(self.crop_box)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

        self._preselect()

    # -- tree ------------------------------------------------------------------------

    def _build_tree(self) -> None:
        groups: dict[str, QTreeWidgetItem] = {}

        def group_item(path: str) -> QTreeWidgetItem | None:
            """Create (or fetch) the tree node for an HDF5 group path."""
            if not path:
                return None
            if path in groups:
                return groups[path]
            parent_path, _, name = path.rpartition("/")
            parent = group_item(parent_path)
            item = QTreeWidgetItem(parent or self.tree, [name, "", ""])
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            item.setExpanded(True)
            groups[path] = item
            return item

        for entry in self.entries:
            parent_path, _, name = entry.path.lstrip("/").rpartition("/")
            parent = group_item(parent_path)
            shape = " x ".join(str(n) for n in entry.shape)
            item = QTreeWidgetItem(parent or self.tree, [name, shape, entry.dtype])
            item.setData(0, _ENTRY_ROLE, entry)
            if not (entry.is_stack or entry.is_angles):
                # Still shown -- seeing the whole file is the point -- but greyed out,
                # because it can be neither a projection stack nor an angle array.
                item.setDisabled(True)

        self.tree.expandAll()

    def _current_entry(self) -> Hdf5Entry | None:
        item = self.tree.currentItem()
        return item.data(0, _ENTRY_ROLE) if item is not None else None

    def _preselect(self) -> None:
        data_path, angle_path = suggest_hdf5_paths(self.entries)
        by_path = {e.path: e for e in self.entries}
        if data_path:
            self._set_data(by_path[data_path])
        if angle_path:
            self._set_angles(by_path[angle_path])
        self._validate()

    # -- selection -------------------------------------------------------------------

    def _selection_changed(self, *_args) -> None:
        entry = self._current_entry()
        self.data_button.setEnabled(entry is not None and entry.ndim == 3)
        self.angle_button.setEnabled(entry is not None and entry.ndim == 1)

    def _use_as_data(self) -> None:
        entry = self._current_entry()
        if entry is not None and entry.ndim == 3:
            self._set_data(entry)
            self._validate()

    def _use_as_angles(self) -> None:
        entry = self._current_entry()
        if entry is not None and entry.ndim == 1:
            self._set_angles(entry)
            self._validate()

    def _clear_angles(self) -> None:
        self._angles = None
        self.angle_label.setText("none - assume a uniform 0-180 deg scan")
        self._validate()

    def _set_data(self, entry: Hdf5Entry) -> None:
        self._data = entry
        self.data_label.setText(
            f"{entry.path}  [{' x '.join(str(n) for n in entry.shape)}, {entry.dtype}]"
        )
        self._rebuild_axis_combo(entry)
        # Only a complex stack has a component to choose; a real one has nothing to pick.
        self.component_combo.setEnabled(entry.is_complex)
        self._reshape_crop_box()

    def _reshape_crop_box(self) -> None:
        if self._data is not None:
            order = self.axis_combo.currentData() or (0, 1, 2)
            self.crop_box.set_full_shape(self._data.stack_shape(order))

    def _set_angles(self, entry: Hdf5Entry) -> None:
        self._angles = entry
        self.angle_label.setText(f"{entry.path}  [{entry.shape[0]}]")

    def _rebuild_axis_combo(self, entry: Hdf5Entry) -> None:
        """Label each permutation with the shape it produces, not with axis numbers.

        "angles=90, v=64, u=64" tells the user which one is right at a glance;
        "(2, 0, 1)" does not.
        """
        previous = self.axis_combo.currentData()
        self.axis_combo.blockSignals(True)
        self.axis_combo.clear()
        for order in _AXIS_ORDERS:
            shape = tuple(entry.shape[axis] for axis in order)
            suffix = "  (as stored)" if order == (0, 1, 2) else ""
            self.axis_combo.addItem(
                f"angles={shape[0]}, v={shape[1]}, u={shape[2]}{suffix}", order
            )
        index = self.axis_combo.findData(previous) if previous else -1
        self.axis_combo.setCurrentIndex(max(index, 0))
        self.axis_combo.blockSignals(False)

    # -- validation ------------------------------------------------------------------

    def _axis_order_changed(self) -> None:
        # The crop is expressed in the post-axis-order frame, so it must be re-based.
        self._reshape_crop_box()
        self._validate()

    def _validate(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.Ok)

        if self._data is None:
            self.status.setText(
                "<b style='color:#d66'>No 3-D dataset selected.</b>"
                if not any(e.is_stack for e in self.entries)
                else "Select a 3-D dataset and press 'Use selected'."
            )
            ok.setEnabled(False)
            return

        order = self.axis_combo.currentData() or (0, 1, 2)
        n_angles = self._data.shape[order[0]]

        if self._angles is not None and self._angles.shape[0] != n_angles:
            self.status.setText(
                f"<b style='color:#d66'>{self._angles.shape[0]} angles but {n_angles} "
                "projections along the chosen angle axis.</b> Either the angle dataset "
                "is the wrong one, or the axis order is."
            )
            ok.setEnabled(False)
            return

        source = self._angles.path if self._angles else "a uniform 0-180 deg scan"
        component = (
            f", {self.component_combo.currentText()} of the complex values"
            if self._data.is_complex
            else ""
        )
        self.status.setText(f"{n_angles} projections, angles from {source}{component}.")
        ok.setEnabled(True)

    # -- result ----------------------------------------------------------------------

    def selection(self) -> dict:
        """Keyword arguments for ``load_dataset(path, **selection())``."""
        if self._data is None:  # pragma: no cover - Ok is disabled until it is set
            raise ValueError("No projection dataset was selected.")
        units = self.units_combo.currentText()
        return {
            "data_path": self._data.path,
            "angle_path": self._angles.path if self._angles else None,
            "axis_order": self.axis_combo.currentData() or (0, 1, 2),
            "angles_in_degrees": {"Degrees": True, "Radians": False}.get(units),
            "component": self.component_combo.currentText(),
            "crop": self.crop_box.crop(),
        }


def _row(*widgets: QWidget) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widgets[0], 1)
    for widget in widgets[1:]:
        layout.addWidget(widget)
    return container


def preview_text(entries: list[Hdf5Entry]) -> str:
    """One-line summary of an HDF5 file's arrays, for the log.

    Takes the listing rather than the path so it can describe a file on another machine
    without a second trip to it.
    """
    stacks = [e for e in entries if e.is_stack]
    return (
        f"{len(entries)} dataset(s), {len(stacks)} of them 3-D "
        f"({', '.join(e.path for e in stacks[:4])}{'...' if len(stacks) > 4 else ''})"
    )


__all__ = ["Hdf5BrowserDialog", "preview_text"]
