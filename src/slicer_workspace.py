"""Viewport selection, overlays and contextual actions, separate from mesh mutations."""
import numpy as np
from PySide6.QtCore import QObject, QEvent, Qt, QSize, QTimer, QRect
from PySide6.QtWidgets import QWidget, QHBoxLayout, QToolButton, QButtonGroup, QComboBox, QDoubleSpinBox, QLabel, QCheckBox, QRubberBand, QPushButton
from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkHardwareSelector
from vtkmodules.vtkCommonDataModel import vtkSelectionNode
from vtkmodules.util.numpy_support import vtk_to_numpy
from workspace_icons import workspace_icon, SUPPORT_NAMES
from surface_selection import SurfaceTopology
from radial_menu import RadialMenu


class SlicerWorkspace(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.plotter = None
        self.mode = 'part'
        self.selection = {}
        self.topologies = {}
        self.overlays = {}
        self.preview = {}
        self.left_down = False
        self.right_start = None
        self.rubber = None
        self.menu = None
        self.manual = None
        self.cube = None
        self.toolbar = QWidget()
        self.toolbar.setStyleSheet('QLabel {color: #ddd;} QToolButton {border: 1px solid transparent; padding: 2px;} QToolButton:hover {background: #444;}')
        layout = QHBoxLayout(self.toolbar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons = {}
        names = [('part', 'Выбор деталей / навигация'), ('triangle', 'Выбор треугольника'),
                 ('plane', 'Выбор связной плоскости'), ('smooth', 'Выбор плавной поверхности'),
                 ('component', 'Выбор связной оболочки'), ('brush', 'Кисть по поверхности'),
                 ('rectangle', 'Прямоугольник: видимые треугольники')]
        for mode, title in names:
            button = QToolButton()
            button.setIcon(workspace_icon(mode))
            button.setIconSize(QSize(24, 24))
            button.setToolTip(title)
            button.setAccessibleName(title)
            button.setCheckable(True)
            button.setStyleSheet('QToolButton:checked {background: #506d38; border: 1px solid #b4db91;}')
            self.group.addButton(button)
            layout.addWidget(button)
            button.clicked.connect(lambda checked=False, value=mode: self.set_mode(value))
            self.buttons[mode] = button
        self.buttons['part'].setChecked(True)
        self.operation = QComboBox()
        self.operation.addItems(['Заменить', 'Добавить', 'Вычесть'])
        self.operation.setToolTip('Shift: добавить; Ctrl: вычесть. Без модификатора — выбранная операция.')
        layout.addWidget(self.operation)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(.01, 10000)
        self.radius.setValue(3)
        self.radius.setSuffix(' мм')
        self.radius.setPrefix('Кисть: ')
        self.radius.setToolTip('Радиус кисти в единицах модели (мм)')
        self.radius.setMaximumWidth(140)
        layout.addWidget(self.radius)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(.1, 90)
        self.angle.setValue(5)
        self.angle.setSuffix('°')
        self.angle.setToolTip('Максимальный угол между соседними треугольниками при выборе плавной поверхности и кистью')
        self.angle.setMaximumWidth(76)
        layout.addWidget(self.angle)
        self.clear_button = QToolButton()
        self.clear_button.setIcon(workspace_icon('clear'))
        self.clear_button.setToolTip('Снять выделение поверхностей (Esc)')
        self.clear_button.clicked.connect(self.clear_selection)
        layout.addWidget(self.clear_button)
        self.count = QLabel('Грани: 0')
        layout.addWidget(self.count)
        layout.addStretch()
        self.cancel_button = QPushButton('Отменить расчёт')
        self.cancel_button.clicked.connect(window.cancel_current_job)
        window.statusBar().addPermanentWidget(self.cancel_button)
        self.cancel_button.hide()
        window.ui.surface_toolbar = self.toolbar
        window.ui.workspace_tools = self
        window.ui._slicer_center_layout.insertWidget(0, self.toolbar)
        from support_tools import SupportTools
        self.supports = SupportTools(window, self)
        self.measurements = window.ui.measurement_panel
        self.measurements.bind(window, self)
        for index, name in enumerate(SUPPORT_NAMES):
            button = window.ui.ribbon_btns[name]
            button.setEnabled(True)
            button.setToolTip(name)
            button.clicked.connect(lambda checked=False, action=index: self.supports.open(action))
        self.set_mode('part')

    def attach(self, plotter):
        if self.plotter is plotter: return
        self.plotter = plotter
        if isinstance(plotter, QWidget):
            plotter.installEventFilter(self)
            self.rubber = QRubberBand(QRubberBand.Rectangle, plotter)
            from orientation_cube import OrientationCube
            self.cube = OrientationCube(plotter)
            plotter.hide_axes()

    def busy(self):
        return bool(self.window._job or getattr(self.window, '_transform_session', None))

    def set_mode(self, mode):
        if hasattr(self, 'measurements'): self.measurements.stop()
        self.mode = mode
        self.buttons[mode].setChecked(True)
        self.radius.setEnabled(mode == 'brush')
        self.angle.setEnabled(mode in ('brush', 'smooth'))
        self.operation.setEnabled(mode != 'part')
        self.manual = None
        if mode != 'part':
            self.window.ui.section_panel.manipulate.setChecked(False)
        if isinstance(self.plotter, QWidget): self.plotter.setCursor(Qt.ArrowCursor if mode == 'part' else Qt.CrossCursor)
        self.window.ui.status_label.setText('Готово' if mode == 'part' else 'Поверхности выбранных деталей: Shift — добавить, Ctrl — вычесть, Alt + мышь — навигация, Esc — сброс.')

    def clear_selection(self):
        self.selection.clear()
        self.draw_selection()

    def clear(self):
        if hasattr(self, 'measurements'): self.measurements.clear()
        self.manual = None
        self.left_down = False
        if self.rubber: self.rubber.hide()
        if self.menu: self.menu.close()
        for mapping in (self.overlays, self.preview):
            if self.plotter:
                for actor, _ in mapping.values(): self.plotter.remove_actor(actor)
            mapping.clear()
        self.selection.clear()
        self.topologies.clear()
        self.count.setText('Грани: 0')

    def invalidate(self, row):
        self.measurements.clear()
        self.topologies.pop(row, None)
        self.selection.pop(row, None)
        if row in self.preview:
            actor, _ = self.preview.pop(row)
            self.plotter.remove_actor(actor)
        self.draw_selection()

    def visible_rows(self, selected_only=True):
        rows = self.window.selected_slicer_rows() if selected_only else range(len(self.window.slicer_parts))
        return [r for r in rows if self.plotter.actors[self.window.slicer_parts[r]['actor_name']].GetVisibility()]

    def picker(self, pos, selected_only=True, rows=None):
        picker = vtkCellPicker()
        picker.SetTolerance(.001)
        picker.PickFromListOn()
        rows = self.visible_rows(selected_only) if rows is None else [r for r in rows if r in self.visible_rows(False)]
        for row in rows: picker.AddPickList(self.plotter.actors[self.window.slicer_parts[row]['actor_name']])
        ratio = self.plotter.devicePixelRatioF()
        width, height = self.plotter.render_window.GetSize()
        if not picker.Pick(pos.x() * ratio, height - 1 - pos.y() * ratio, 0, self.plotter.renderer): return None
        actor = picker.GetActor()
        row = next((r for r in rows if self.plotter.actors[self.window.slicer_parts[r]['actor_name']] == actor), None)
        if row is None or picker.GetCellId() < 0: return None
        return row, picker.GetCellId(), np.asarray(picker.GetPickPosition())

    def edit_selection(self, row, ids, operation):
        if operation == 'replace': self.selection.clear()
        selected = self.selection.setdefault(row, set())
        if operation == 'subtract': selected.difference_update(ids)
        else: selected.update(ids)
        if not selected: self.selection.pop(row, None)
        self.draw_selection()

    def gesture_operation(self, event):
        if event.modifiers() & Qt.ControlModifier: return 'subtract'
        if event.modifiers() & Qt.ShiftModifier: return 'add'
        return ('replace', 'add', 'subtract')[self.operation.currentIndex()]

    def select_at(self, position, operation):
        hit = self.picker(position)
        if hit is None:
            if operation == 'replace': self.clear_selection()
            return
        row, face, point = hit
        mesh = self.window.slicer_parts[row]['mesh']
        topology = self.topologies.get(row)
        if topology is None or topology.mesh is not mesh:
            topology = self.topologies[row] = SurfaceTopology(mesh)
        ids = topology.select(face, self.mode, point, self.angle.value(), self.radius.value())
        self.edit_selection(row, ids, operation)

    def draw_selection(self):
        if self.plotter is None: self.plotter = self.window.ui.slicer_plotter
        if self.plotter:
            for actor, _ in self.overlays.values(): self.plotter.remove_actor(actor)
            self.overlays.clear()
            for row, ids in self.selection.items():
                self.add_overlay(row, ids, self.overlays, '#ffb326', 'surface_selection')
            self.refresh_overlays()
            self.plotter.render()
        self.count.setText(f'Грани: {sum(map(len, self.selection.values()))}')
        if hasattr(self, 'supports') and self.supports.panel: self.supports.panel.selection_changed()

    def add_overlay(self, row, ids, mapping, color, prefix):
        if not len(ids): return
        data = self.window.slicer_parts[row]['mesh_pv'].extract_cells(sorted(ids)).extract_surface(algorithm='dataset_surface')
        actor = self.plotter.add_mesh(data, name=f'{prefix}_{row}', color=color, opacity=1, pickable=False, lighting=False)
        actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
        actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
        mapping[row] = (actor, self.window.slicer_parts[row]['mesh'])

    def refresh_overlays(self):
        if not self.plotter: return
        if self.cube:
            self.plotter.hide_axes()
            self.cube.camera_changed()
        for mapping in (self.overlays, self.preview):
            for row, (actor, mesh) in list(mapping.items()):
                if row >= len(self.window.slicer_parts) or self.window.slicer_parts[row]['mesh'] is not mesh:
                    self.plotter.remove_actor(actor)
                    mapping.pop(row)
                    continue
                source = self.plotter.actors.get(self.window.slicer_parts[row]['actor_name'])
                actor.SetVisibility(bool(source and source.GetVisibility()))
                mapper = actor.GetMapper()
                mapper.RemoveAllClippingPlanes()
                if source:
                    planes = source.GetMapper().GetClippingPlanes()
                    if planes:
                        for i in range(planes.GetNumberOfItems()): mapper.AddClippingPlane(planes.GetItem(i))

    def select_rectangle(self, rect, operation):
        selector = vtkHardwareSelector()
        selector.SetRenderer(self.plotter.renderer)
        selector.SetFieldAssociation(1)  # vtkDataObject.FIELD_ASSOCIATION_CELLS
        ratio = self.plotter.devicePixelRatioF()
        width, height = self.plotter.render_window.GetSize()
        clamp_x = lambda x: max(0, min(width - 1, int(x * ratio)))
        clamp_y = lambda y: max(0, min(height - 1, int(height - 1 - y * ratio)))
        selector.SetArea(clamp_x(rect.left()), clamp_y(rect.bottom()), clamp_x(rect.right()), clamp_y(rect.top()))
        sources = {r: self.plotter.actors[self.window.slicer_parts[r]['actor_name']] for r in self.visible_rows()}
        if operation == 'replace': self.selection.clear()
        result = selector.Select()
        if result:
            for i in range(result.GetNumberOfNodes()):
                node = result.GetNode(i)
                actor = node.GetProperties().Get(vtkSelectionNode.PROP())
                row = next((r for r, source in sources.items() if source == actor), None)
                if row is not None:
                    ids = set(map(int, vtk_to_numpy(node.GetSelectionList())))
                    current = self.selection.setdefault(row, set())
                    current.difference_update(ids) if operation == 'subtract' else current.update(ids)
        self.draw_selection()

    def popup(self, position):
        if self.busy() or not self.window.selected_slicer_rows() or self.manual: return
        if self.menu: self.menu.deleteLater()
        self.menu = RadialMenu(self.window)
        self.menu.popup(position)

    def eventFilter(self, obj, event):
        if obj is not self.plotter or self.busy(): return False
        kind = event.type()
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.manual = None
            self.set_mode('part')
            self.clear_selection()
            self.supports.clear_preview()
            return True
        if kind == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self.right_start = event.position().toPoint()
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
            if self.right_start is not None and (event.position().toPoint() - self.right_start).manhattanLength() < 5:
                position = event.globalPosition().toPoint()
                QTimer.singleShot(0, lambda: self.popup(position))
            self.right_start = None
        if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.left_start = event.position().toPoint()
            if self.measurements.active and not event.modifiers() & Qt.AltModifier:
                self._tool_click = True
                self.measurements.pick(self.left_start)
                return True
            if self.manual:
                self._tool_click = True
                self.supports.manual_at(self.left_start)
                return True
            if self.mode != 'part' and not event.modifiers() & Qt.AltModifier:
                self.left_down = True
                self.stroke_operation = self.gesture_operation(event)
                if self.mode == 'rectangle':
                    self.rubber.setGeometry(QRect(self.left_start, self.left_start))
                    self.rubber.show()
                else:
                    self.select_at(self.left_start, self.stroke_operation)
                    if self.stroke_operation == 'replace': self.stroke_operation = 'add'
                return True
        if kind == QEvent.MouseMove and self.left_down:
            if self.mode == 'brush': self.select_at(event.position().toPoint(), self.stroke_operation)
            if self.mode == 'rectangle': self.rubber.setGeometry(QRect(self.left_start, event.position().toPoint()).normalized())
            return True
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if getattr(self, '_tool_click', False):
                self._tool_click = False
                return True
            if self.left_down:
                self.left_down = False
                if self.mode == 'rectangle':
                    rect = self.rubber.geometry()
                    self.rubber.hide()
                    self.select_rectangle(rect, self.stroke_operation)
                return True
            if self.mode == 'part' and hasattr(self, 'left_start') and (event.position().toPoint() - self.left_start).manhattanLength() < 4:
                hit = self.picker(event.position().toPoint(), selected_only=False)
                if hit:
                    row = hit[0]
                    additive = event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)
                    for r in range(len(self.window.slicer_parts)):
                        check = self.window.ui.tbl_parts.cellWidget(r, 1).findChild(QCheckBox)
                        if r == row: check.setChecked(not check.isChecked() if additive else True)
                        elif not additive: check.setChecked(False)
                    self.window.ui.tbl_parts.selectRow(row)
        return False
