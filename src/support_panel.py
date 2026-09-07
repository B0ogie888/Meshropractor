"""Part-owned support regions and editable profiles in the slicer's left pane."""
import json
from copy import deepcopy
import numpy as np
import trimesh
from PySide6.QtCore import Qt, QPointF, QSignalBlocker, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QComboBox, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QDoubleSpinBox, QCheckBox, QAbstractItemView, QInputDialog
from background_tasks import FunctionWorker
from support_geometry import generate_supports, validate
from part_supports import make_group, remove_actors

KINDS = ['Отсутствует', 'Блок', 'Линии', 'Точечные', 'Сеть', 'Контур', 'Конусы', 'Ветвящиеся']


class SupportPlan(QWidget):
    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.setMinimumHeight(140)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#16191c'))
        panel = self.panel
        row = panel.row()
        if row is None: return
        mesh = panel.window.slicer_parts[row]['mesh']
        ids = panel.faces()
        if not ids:
            painter.setPen(QColor('#b0bac4'))
            painter.drawText(self.rect(), Qt.AlignCenter, 'Выделите поверхность детали')
            return
        triangles = mesh.triangles[ids, :, :2]
        low, high = triangles.reshape(-1, 2).min(axis=0), triangles.reshape(-1, 2).max(axis=0)
        size = np.maximum(high - low, 1e-3)
        scale = min((self.width()-20)/size[0], (self.height()-20)/size[1])
        def point(p): return QPointF(self.width()/2+(p[0]-(low[0]+high[0])/2)*scale, self.height()/2-(p[1]-(low[1]+high[1])/2)*scale)
        painter.setPen(QPen(QColor('#d9b758'), .6))
        painter.setBrush(QColor('#3d4136'))
        stride = max(1, len(triangles)//2500)
        for triangle in triangles[::stride]: painter.drawPolygon(QPolygonF([point(p) for p in triangle]))
        group = panel.current_group()
        if group is not None and len(group['faces']):
            vertices = np.asarray(group['vertices'])[:, :2]
            painter.setPen(QPen(QColor('#91d781'), 1.2))
            for p in vertices[::max(1, len(vertices)//2000)]: painter.drawPoint(point(p))
        painter.setPen(QColor('#e1e5e8'))
        painter.drawText(8, 17, 'План XY · ' + panel.kind.currentText())


class SupportPanel(QWidget):
    def __init__(self, tools):
        super().__init__()
        self.tools, self.window, self.workspace = tools, tools.window, tools.workspace
        self.loading = False
        self.previous_sizes = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        header = QHBoxLayout()
        header.addWidget(QLabel('Поддержки · поверхности детали'))
        back = QPushButton('К деталям')
        back.clicked.connect(self.finish)
        header.addWidget(back)
        layout.addLayout(header)
        self.part = QComboBox()
        self.part.currentIndexChanged.connect(self.part_changed)
        layout.addWidget(self.part)
        tabs = QTabWidget()
        listing = QWidget()
        listing_layout = QVBoxLayout(listing)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['ID', 'Тип', 'Грани', 'Площадь, мм²', 'Z макс.', 'Контакты'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(115)
        self.table.setMaximumHeight(190)
        self.table.setStyleSheet('QTableWidget::item:selected {background: #34566d; color: white;}')
        for column, width in enumerate((38, 90, 48, 100, 72, 70)): self.table.setColumnWidth(column, width)
        self.table.itemSelectionChanged.connect(self.group_changed)
        listing_layout.addWidget(self.table)
        row_buttons = QHBoxLayout()
        for text, callback in [('Добавить область', self.add_region), ('Выбрать грани', self.select_faces), ('Удалить', self.delete_region)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            row_buttons.addWidget(button)
        listing_layout.addLayout(row_buttons)
        tabs.addTab(listing, 'Список поддержек')
        self.surface_info = QLabel()
        self.surface_info.setWordWrap(True)
        tabs.addTab(self.surface_info, 'Поверхность')
        self.part_info = QLabel()
        self.part_info.setWordWrap(True)
        tabs.addTab(self.part_info, 'Деталь')
        layout.addWidget(tabs)
        self.settings_tabs = QTabWidget()
        types = QWidget()
        type_layout = QVBoxLayout(types)
        self.kind = QComboBox()
        self.kind.addItems(KINDS)
        self.kind.setCurrentText('Блок')
        type_layout.addWidget(self.kind)
        self.plan = SupportPlan(self)
        type_layout.addWidget(self.plan)
        self.kind.currentTextChanged.connect(lambda _: self.plan.update())
        self.settings_tabs.addTab(types, 'Тип и 2D план')
        general = QWidget()
        form = QFormLayout(general)
        self.fields = {}
        for key, label, low, high in [('spacing','Шаг, мм',.2,100), ('diameter','Диаметр стойки, мм',.1,30),
                                     ('tip_diameter','Контакт / стенка, мм',.05,30), ('tip_height','Высота сужения, мм',.1,30),
                                     ('foot_diameter','Диаметр основания, мм',.1,50), ('foot_height','Высота основания, мм',.1,30),
                                     ('angle','Угол нависания, °',.1,89.9), ('base_z','Z платформы, мм',-100000,100000)]:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(3)
            spin.setValue(self.tools.params[key])
            self.fields[key] = spin
            form.addRow(label, spin)
        self.only_platform = QCheckBox('Только до платформы, пропускать препятствия')
        form.addRow(self.only_platform)
        self.settings_tabs.addTab(general, 'Параметры')
        layout.addWidget(self.settings_tabs)
        self.show_support = QCheckBox('Показывать выбранную группу поддержек')
        self.show_support.setChecked(True)
        self.show_support.toggled.connect(self.toggle_visible)
        layout.addWidget(self.show_support)
        actions = QHBoxLayout()
        rebuild = QPushButton('Перестроить 2D и 3D')
        rebuild.clicked.connect(self.rebuild)
        manual = QPushButton('Указать точку вручную')
        manual.clicked.connect(self.start_points)
        actions.addWidget(rebuild)
        actions.addWidget(manual)
        layout.addLayout(actions)
        profile_row = QHBoxLayout()
        self.profile = QComboBox()
        self.profiles = {'Стандарт': dict(tools.params)}
        try:
            saved = json.loads(self.window.settings.value('support_profiles_v1', '{}'))
            for name, values in saved.items():
                validate(values)
                self.profiles[name] = values
        except (ValueError, TypeError, KeyError, AttributeError): pass
        self.profile.addItems(self.profiles)
        self.profile.activated.connect(self.load_profile)
        save = QPushButton('Сохранить профиль…')
        save.clicked.connect(self.save_profile)
        profile_row.addWidget(self.profile)
        profile_row.addWidget(save)
        layout.addLayout(profile_row)
        self.status = QLabel('Выделите грани инструментами над сценой, добавьте область и перестройте поддержки.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.hide()

    def row(self):
        row = self.part.currentData()
        return row if row is not None and row < len(self.window.slicer_parts) else None

    def current_group(self):
        row, index = self.row(), self.table.currentRow()
        if row is None: return None
        groups = self.window.slicer_parts[row].get('supports', [])
        return groups[index] if 0 <= index < len(groups) else None

    def faces(self):
        row = self.row()
        if row is None: return []
        selected = sorted(self.workspace.selection.get(row, set()))
        group = self.current_group()
        return selected if selected else group.get('surface_faces', []) if group else []

    def parameters(self):
        params = {key: spin.value() for key, spin in self.fields.items()}
        params['only_platform'] = self.only_platform.isChecked()
        validate(params)
        return params

    def open(self):
        if not self.window.slicer_parts:
            self.window.log('Сначала загрузите деталь.')
            return
        self.show()
        for group in self.window.ui.slicer_normal_groups: group.hide()
        if self.previous_sizes is None: self.previous_sizes = self.window.ui.slicer_splitter.sizes()
        self.window.ui.slicer_splitter.setSizes([620, max(400, self.window.width()-620), 0])
        self.refresh()
        self.part_changed()
        self.workspace.set_mode('plane')
        QTimer.singleShot(0, lambda: self.window.ui.slicer_left_scroll.ensureWidgetVisible(self, 0, 0))

    def finish(self):
        self.workspace.set_mode('part')
        self.hide()
        for group in self.window.ui.slicer_normal_groups: group.show()
        if self.previous_sizes: self.window.ui.slicer_splitter.setSizes(self.previous_sizes)
        self.previous_sizes = None

    def refresh(self):
        row = self.row()
        if row is None:
            selected = self.window.selected_slicer_rows()
            row = selected[0] if selected else 0
        index = self.table.currentRow()
        self.loading = True
        blocker = QSignalBlocker(self.part)
        self.part.clear()
        for r, part in enumerate(self.window.slicer_parts): self.part.addItem(part['filename'], r)
        self.part.setCurrentIndex(min(row, self.part.count()-1))
        del blocker
        self.table.setRowCount(0)
        row = self.row()
        if row is not None:
            part = self.window.slicer_parts[row]
            for i, group in enumerate(part.get('supports', [])):
                self.table.insertRow(i)
                ids = group['surface_faces']
                area = part['mesh'].area_faces[ids].sum() if ids else 0
                z = np.asarray(group['vertices'])[:, 2].max() if len(group['vertices']) else 0
                for column, text in enumerate((str(i+1), group['kind'], str(len(ids)), f'{area:.3f}', f'{z:.3f}', str(group['contacts']))):
                    self.table.setItem(i, column, QTableWidgetItem(text))
            self.part_info.setText(f"{part['filename']}\nТреугольники: {len(part['mesh'].faces)}\nПлощадь: {part['mesh'].area:.3f} мм²\nГрупп поддержек: {len(part.get('supports', []))}")
        if self.table.rowCount(): self.table.selectRow(max(0, min(index, self.table.rowCount()-1)))
        self.loading = False
        self.group_changed(False)

    def part_changed(self):
        if self.loading: return
        row = self.row()
        if row is None: return
        self.workspace.clear_selection()
        for r in range(len(self.window.slicer_parts)):
            self.window.ui.tbl_parts.cellWidget(r, 1).findChild(QCheckBox).setChecked(r == row)
        self.refresh()

    def group_changed(self, select_surface=True):
        if self.loading: return
        group = self.current_group()
        if group:
            self.kind.setCurrentText(group['kind'])
            self.load_values(group['params'])
            blocker = QSignalBlocker(self.show_support)
            self.show_support.setChecked(group.get('visible', True))
            del blocker
            if select_surface: self.workspace.edit_selection(self.row(), set(group['surface_faces']), 'replace')
        self.selection_changed()

    def selection_changed(self):
        row = self.row()
        if row is not None:
            ids = self.faces()
            mesh = self.window.slicer_parts[row]['mesh']
            self.surface_info.setText(f'Выделено граней: {len(ids)}\nПлощадь: {mesh.area_faces[ids].sum() if ids else 0:.4f} мм²')
        self.plan.update()

    def add_region(self):
        if self.window._busy(): return
        row = self.row()
        ids = sorted(self.workspace.selection.get(row, set()))
        if row is None or not ids:
            self.status.setText('Сначала выделите поверхность инструментами над сценой.')
            return
        try: params = self.parameters()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.window.flush_history()
        empty = trimesh.Trimesh(np.empty((0,3)), np.empty((0,3), dtype=np.int64), process=False)
        group = make_group(empty, ids, self.kind.currentText(), params)
        self.window.slicer_parts[row]['supports'].append(group)
        self.window.mark_dirty()
        self.window.flush_history('Добавить область поддержек')
        self.refresh()
        self.table.selectRow(self.table.rowCount()-1)
        self.status.setText('Область добавлена. Выберите тип и нажмите «Перестроить 2D и 3D».')

    def select_faces(self):
        group = self.current_group()
        if group:
            self.workspace.set_mode('plane')
            self.workspace.edit_selection(self.row(), set(group['surface_faces']), 'replace')

    def delete_region(self):
        if self.window._busy(): return
        group = self.current_group()
        if group:
            self.window.flush_history()
            remove_actors(self.window, [group])
            self.window.slicer_parts[self.row()]['supports'].pop(self.table.currentRow())
            self.window.mark_dirty()
            self.window.flush_history('Удалить поддержку')
            self.window.refresh_scene_visibility()
            self.refresh()

    def toggle_visible(self, checked):
        group = self.current_group()
        if group and not self.loading:
            group['visible'] = checked
            self.window.mark_dirty()
            self.loading = True
            try: self.window.refresh_scene_visibility()
            finally: self.loading = False

    def rebuild(self):
        if self.window._busy(): return
        row, group = self.row(), self.current_group()
        if row is None or not self.faces():
            self.status.setText('Выберите область поверхности.')
            return
        try: params = self.parameters()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        ids, kind = list(self.faces()), self.kind.currentText()
        replace_id = group['id'] if group else None
        records = self.tools.records()
        def generate():
            if kind == 'Отсутствует':
                mesh = trimesh.Trimesh(np.empty((0,3)), np.empty((0,3), dtype=np.int64), process=False)
                results = [dict(mesh=mesh, row=row, contacts=0, surface_faces=ids, kind=kind, params=params)]
            else: results = generate_supports(records, {row: ids}, params, kind=kind)
            for result in results: result['replace_id'] = replace_id
            return results
        self.tools.params = params
        self.window.start_job(FunctionWorker(generate), self.tools.append_results)

    def start_points(self):
        if self.window._busy() or self.row() is None: return
        try: self.tools.params = self.parameters()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.workspace.set_mode('part')
        self.window.ui.section_panel.manipulate.setChecked(False)
        self.workspace.manual = dict(rows=[self.row()], records=self.tools.records(), worlds={})
        self.workspace.plotter.setFocus()
        self.workspace.plotter.setCursor(Qt.CrossCursor)
        self.status.setText('Щёлкайте по нижней поверхности для установки стоек. Esc — закончить.')

    def load_values(self, values):
        for key, value in values.items():
            if key in self.fields: self.fields[key].setValue(value)
        self.only_platform.setChecked(values.get('only_platform', False))

    def load_profile(self, index):
        self.load_values(self.profiles[self.profile.itemText(index)])

    def save_profile(self):
        try: values = self.parameters()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        name, accepted = QInputDialog.getText(self, 'Профиль поддержек', 'Название:')
        if accepted and name.strip():
            self.profiles[name.strip()] = deepcopy(values)
            self.window.settings.setValue('support_profiles_v1', json.dumps(self.profiles, ensure_ascii=False))
            self.profile.clear()
            self.profile.addItems(self.profiles)
            self.profile.setCurrentText(name.strip())
