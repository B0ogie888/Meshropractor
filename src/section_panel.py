"""Interactive slicer cutaways: clipping affects actor mappers only."""
from copy import deepcopy
import numpy as np
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QHeaderView,
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QPushButton, QSlider, QLabel,
    QColorDialog, QFileDialog, QSizePolicy)

from section_geometry import AXES, default_sections, validate_sections, clipping_plane, projected_range


class SectionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = None
        self.sections = default_sections()
        self.bounds = None
        self.widget = None
        self._syncing = False
        self._selected_row = 0
        self._planes = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Вкл.", "Тип", "Отсечь", "Цвет", "Позиция", "Шаг"])
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(28)
        header.setSectionResizeMode(QHeaderView.Interactive)
        for col, width in enumerate((38, 66, 50, 38, 116, 94)):
            header.resizeSection(col, width)
        header.setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.cellClicked.connect(lambda row, col: self.select_row(row))
        layout.addWidget(self.table)
        self.selection_label = QLabel()
        layout.addWidget(self.selection_label)
        tools = QHBoxLayout()
        for label, callback in (("+", self.add_plane), ("Удалить", self.remove_plane), ("Сброс", self.reset)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            tools.addWidget(button)
        layout.addLayout(tools)
        tools = QHBoxLayout()
        self.manipulate = QPushButton("Указать")
        self.manipulate.setCheckable(True)
        self.manipulate.setToolTip("Показать выбранную плоскость: перетаскивайте её мышью. Для вращения выберите «Произв.».")
        self.manipulate.toggled.connect(self._make_widget)
        tools.addWidget(self.manipulate)
        align = QPushButton("Выровнять")
        align.setToolTip("Направить камеру перпендикулярно выбранному сечению")
        align.clicked.connect(self.align_camera)
        tools.addWidget(align)
        export = QPushButton("Экспорт")
        export.setToolTip("Экспорт контуров выбранного сечения видимых деталей в VTP")
        export.clicked.connect(self.export_contours)
        tools.addWidget(export)
        layout.addLayout(tools)
        movement = QHBoxLayout()
        left, right = QPushButton("◀"), QPushButton("▶")
        left.setMaximumWidth(35)
        right.setMaximumWidth(35)
        left.clicked.connect(lambda: self.move(-1))
        right.clicked.connect(lambda: self.move(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self._slide)
        movement.addWidget(left)
        movement.addWidget(self.slider)
        movement.addWidget(right)
        layout.addLayout(movement)
        hint = QLabel("Позиция и шаг — в мм. «+ / −» — удаляемая сторона.\nДо 6 плоскостей; меняется только отображение.")
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(hint)
        self._rebuild()

    def bind(self, window):
        self.window = window

    def snapshot(self):
        return deepcopy(self.sections)

    def restore(self, sections):
        self.sections = validate_sections(sections)
        self._rebuild()
        self.apply()
        self._make_widget()

    def reset(self):
        self.sections = default_sections()
        if self.bounds is not None:
            center = np.mean(self.bounds, axis=0)
            for section in self.sections:
                section["position"] = float(center @ section["normal"])
        self.manipulate.setChecked(False)
        self._rebuild()
        self._changed()

    def clear(self):
        self.bounds = None
        self.reset()

    def _rebuild(self):
        self._syncing = True
        selected = max(0, min(self._selected_row, len(self.sections) - 1))
        self.table.setRowCount(len(self.sections))
        for row, section in enumerate(self.sections):
            active = QCheckBox()
            active.setChecked(section["active"])
            active.toggled.connect(lambda value, r=row: self._edit(r, "active", value))
            axis = QComboBox()
            axis.addItems([*AXES, "Произв."])
            axis.setCurrentText(section["axis"])
            axis.currentTextChanged.connect(lambda value, r=row: self._edit(r, "axis", value))
            side = QComboBox()
            side.addItems(["+", "−"])
            side.setCurrentText(section["cut"])
            side.currentTextChanged.connect(lambda value, r=row: self._edit(r, "cut", value))
            color = QPushButton()
            color.setStyleSheet(f"background-color: {section['color']};")
            color.clicked.connect(lambda checked=False, r=row: self._color(r))
            controls = [active, axis, side, color]
            for field in ("position", "step"):
                spin = QDoubleSpinBox()
                spin.setDecimals(4)
                spin.setRange(.0001 if field == "step" else -1e9, 1e9)
                spin.setKeyboardTracking(False)
                spin.setMinimumWidth(0)
                spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
                spin.setValue(section[field])
                spin.setSingleStep(section["step"])
                spin.valueChanged.connect(lambda value, r=row, f=field: self._edit(r, f, value))
                controls.append(spin)
            for col, control in enumerate(controls):
                control.setFocusPolicy(Qt.StrongFocus)
                for watched in [control, *control.findChildren(QWidget)]:
                    watched.setProperty("section_row", row)
                    watched.installEventFilter(self)
                self.table.setCellWidget(row, col, control)
        if self.sections:
            self.table.setCurrentCell(selected, 0)
        self._syncing = False
        self.select_row(selected if self.sections else -1)

    def _edit(self, row, field, value):
        if self._syncing:
            return
        self.sections[row][field] = value
        if row == self._selected_row:
            self.selection_label.setText(f"Редактируется сечение {row + 1}: {self.sections[row]['axis']}")
        if field == "axis" and value in AXES:
            self.sections[row]["normal"] = AXES[value].copy()
        if field == "step":
            self.table.cellWidget(row, 4).setSingleStep(value)
        self._changed()
        if field in ("axis", "active"):
            self._make_widget()

    def _changed(self):
        self.apply()
        self._sync_slider()
        self._update_widget()
        if self.window:
            self.window.mark_dirty()

    def eventFilter(self, obj, event):
        row = obj.property("section_row")
        if row is not None:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.select_row(row)
            elif event.type() == QEvent.FocusIn and event.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason):
                self.select_row(row)
            elif event.type() == QEvent.Wheel:
                # Scrolling the panel must never edit another plane under the pointer.
                if row != self._selected_row or not obj.hasFocus():
                    event.ignore()
                    return True
        return super().eventFilter(obj, event)

    def select_row(self, row):
        self._selected_row = row
        if 0 <= row < len(self.sections):
            self.table.setCurrentCell(row, 4)
            self.selection_label.setText(f"Редактируется сечение {row + 1}: {self.sections[row]['axis']}")
        else:
            self.selection_label.setText("Выберите сечение щелчком по ячейке")
        if not self._syncing:
            self._sync_slider()
            self._make_widget()

    def selected(self):
        row = self._selected_row
        return self.sections[row] if 0 <= row < len(self.sections) else None

    def _color(self, row):
        color = QColorDialog.getColor(QColor(self.sections[row]["color"]), self)
        if color.isValid():
            self.sections[row]["color"] = color.name()
            self.table.cellWidget(row, 3).setStyleSheet(f"background-color: {color.name()};")
            self.select_row(row)
            self._changed()
            self._make_widget()

    def add_plane(self):
        if len(self.sections) >= 6:
            return
        section = deepcopy(self.selected() or default_sections()[0])
        section["active"] = True
        self.sections.append(section)
        self._rebuild()
        self.select_row(len(self.sections) - 1)
        self._changed()

    def remove_plane(self):
        if self.selected() is not None:
            del self.sections[self._selected_row]
            self._rebuild()
            self._changed()
            self._make_widget()

    def move(self, direction):
        section = self.selected()
        if section is not None:
            self.table.cellWidget(self._selected_row, 4).setValue(section["position"] + direction * section["step"])

    def _sync_slider(self):
        section = self.selected()
        self.slider.setEnabled(section is not None and self.bounds is not None)
        if section is not None and self.bounds is not None:
            lo, hi = projected_range(self.bounds, section["normal"])
            self.slider.blockSignals(True)
            self.slider.setValue(round(np.clip((section["position"] - lo) / max(hi - lo, 1e-10), 0, 1) * 1000))
            self.slider.blockSignals(False)

    def _slide(self, value):
        section = self.selected()
        if section is not None and self.bounds is not None:
            lo, hi = projected_range(self.bounds, section["normal"])
            self.table.cellWidget(self._selected_row, 4).setValue(lo + (hi - lo) * value / 1000)

    def _plotter(self):
        return self.window.ui.slicer_plotter if self.window else None

    def visible_parts(self):
        plotter = self._plotter()
        if plotter is None:
            return []
        return [part for part in self.window.slicer_parts
                if any(plotter.actors.get(part["actor_name"] + suffix) is not None
                       and plotter.actors[part["actor_name"] + suffix].GetVisibility() for suffix in ("", "__bbox"))]

    def refresh(self):
        parts = self.visible_parts()
        if parts:
            bounds = np.array([part["mesh"].bounds for part in parts])
            first = self.bounds is None
            self.bounds = np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])
            if first and not any(s["active"] or s["position"] != 0 for s in self.sections):
                center = self.bounds.mean(axis=0)
                for section in self.sections:
                    section["position"] = float(center @ section["normal"])
                self._rebuild()
        else:
            self.bounds = None
        self.apply()
        self._sync_slider()
        self._make_widget()

    def apply(self):
        plotter = self._plotter()
        if plotter is None:
            return
        self._planes = [clipping_plane(section) for section in self.sections if section["active"]]
        for part in self.window.slicer_parts:
            for suffix in ("", "__bbox"):
                actor = plotter.actors.get(part["actor_name"] + suffix)
                if actor is not None:
                    mapper = actor.GetMapper()
                    mapper.RemoveAllClippingPlanes()
                    for plane in self._planes:
                        mapper.AddClippingPlane(plane)
        if hasattr(self.window, 'workspace_tools'):
            from part_supports import sync_actors
            sync_actors(self.window)
            self.window.workspace_tools.refresh_overlays()
        plotter.render()

    def _origin(self, section):
        normal = np.asarray(section["normal"])
        center = self.bounds.mean(axis=0)
        return center + normal * (section["position"] - center @ normal)

    def _make_widget(self, *args):
        plotter = self._plotter()
        if self.widget is not None:
            self.widget.Off()
            if plotter is not None:
                plotter.clear_plane_widgets()
            self.widget = None
        section = self.selected()
        if plotter is None or self.bounds is None or section is None or not self.manipulate.isChecked() or not self.isEnabled():
            if plotter is not None:
                plotter.render()
            return
        lo, hi = self.bounds
        padding = max(float(np.max(hi - lo)) * .05, .01)
        bounds = np.column_stack((lo - padding, hi + padding)).ravel()
        self.widget = plotter.add_plane_widget(self._drag, normal=section["normal"], origin=self._origin(section),
            bounds=bounds, color=section["color"], test_callback=False,
            normal_rotation=section["axis"] == "Произв.", outline_translation=False,
            interaction_event="always")
        plotter.render()

    def _update_widget(self):
        section = self.selected()
        if self.widget is not None and section is not None and self.bounds is not None:
            self.widget.SetNormal(*section["normal"])
            self.widget.SetOrigin(*self._origin(section))
            self._plotter().render()

    def _drag(self, normal, origin):
        section = self.selected()
        if section is None:
            return
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)
        section["normal"] = normal.tolist()
        section["position"] = float(normal @ origin)
        spin = self.table.cellWidget(self._selected_row, 4)
        spin.blockSignals(True)
        spin.setValue(section["position"])
        spin.blockSignals(False)
        self.apply()
        self._sync_slider()
        self.window.mark_dirty()

    def align_camera(self):
        section, plotter = self.selected(), self._plotter()
        if section is None or plotter is None or self.bounds is None:
            return
        normal = np.asarray(section["normal"])
        origin = self._origin(section)
        distance = max(np.linalg.norm(self.bounds[1] - self.bounds[0]) * 2, 1)
        up = np.array([0., 0., 1.]) if abs(normal[2]) < .9 else np.array([0., 1., 0.])
        plotter.camera_position = [origin + normal * distance * (1 if section["cut"] == "+" else -1), origin, up]
        plotter.render()

    def export_contours(self):
        section = self.selected()
        if section is None or not self.visible_parts():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Контуры сечения", "section.vtp", "VTK PolyData (*.vtp)")
        if not path:
            return
        try:
            import pyvista as pv
            contours = [part["mesh_pv"].slice(normal=section["normal"], origin=np.asarray(section["normal"]) * section["position"])
                        for part in self.visible_parts()]
            contours = [contour for contour in contours if contour.n_points]
            if not contours:
                self.window.log("Выбранная плоскость не пересекает видимые детали.")
                return
            pv.merge(contours).save(path if path.lower().endswith(".vtp") else path + ".vtp")
            self.window.log("Контуры сечения сохранены.")
        except Exception as exc:
            self.window.log(f"Ошибка экспорта сечения: {exc}")
