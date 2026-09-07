"""Detailed modeless move, rotate, scale and mirror panels."""
import json
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QDoubleSpinBox, QCheckBox, QLabel, QPushButton, QRadioButton, QButtonGroup,
    QComboBox, QListWidget, QInputDialog, QDialogButtonBox, QScrollArea, QGroupBox, QSizePolicy)
from transform_math import anchor, selection_bounds, unit, plane_from_points


class TransformDialog(QDialog):
    changed = Signal()
    apply_requested = Signal(bool)
    pick_requested = Signal(str, int)

    def __init__(self, operation, meshes, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.meshes = meshes
        self.bounds = selection_bounds(meshes)
        self.initial_bounds = self.bounds.copy()
        self.initial_mesh_bounds = [m.bounds.copy() for m in meshes]
        self._sync = False
        self.absolute = False
        self.setWindowTitle({'Перемещать': 'Перемещение деталей', 'Вращать': 'Вращать',
            'Масштабировать': 'Масштабировать детали', 'Отзеркалить': 'Отзеркалить детали'}[operation])
        self.setWindowModality(Qt.NonModal)
        self.setStyleSheet('QRadioButton::indicator {width: 12px; height: 12px; border: 1px solid #999; border-radius: 7px; background: #252525;} QRadioButton::indicator:checked {border: 3px solid #62b4e8; background: #dceeff;}')
        self.setMinimumWidth(455 if operation != 'Масштабировать' else 560)
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        self.layout_body = QVBoxLayout(body)
        self.layout_body.setSpacing(8)
        self.layout_body.setAlignment(Qt.AlignTop)
        scroll.setWidget(body)
        root.addWidget(scroll)
        self.preview = QCheckBox('Предпросмотр')
        self.preview.setChecked(True)
        self.copy = QCheckBox('Создать копию')
        self.keep_z = QCheckBox('Сохранять исходное положение по Z')
        self.keep_z.setToolTip('Сохранить нижнюю точку каждой детали по Z после преобразования')
        self.individual = QCheckBox('Индивидуальный центр деталей')
        self.center = self.spins(self.bounds.mean(axis=0))
        self.line_a = self.spins(self.bounds.mean(axis=0))
        self.line_b = self.spins(self.bounds.mean(axis=0) + [1, 0, 0])
        self.along_line = QCheckBox('Перемещать вдоль линии' if operation == 'Перемещать' else 'Вращать вокруг линии')
        self.line_angle = self.spin(0, suffix=' °')
        if operation == 'Перемещать': self.build_move()
        elif operation == 'Вращать': self.build_rotate()
        elif operation == 'Масштабировать': self.build_scale()
        else: self.build_mirror()
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('color: #edbc6b;')
        root.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.apply_button = QPushButton('Применить')
        self.apply_button.clicked.connect(lambda: self.apply_requested.emit(False))
        buttons.addWidget(self.apply_button)
        self.yes_button = QPushButton('Да')
        self.yes_button.clicked.connect(lambda: self.apply_requested.emit(True))
        if operation in ('Перемещать', 'Вращать'): buttons.addWidget(self.yes_button)
        close = QPushButton('Закрыть')
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        root.addLayout(buttons)
        # Connect after construction to avoid partial parameter sets during initial layout.
        for control in self.findChildren(QDoubleSpinBox): control.valueChanged.connect(self.notify)
        for control in self.findChildren(QCheckBox): control.toggled.connect(self.notify)
        for control in self.findChildren(QRadioButton): control.toggled.connect(self.notify)
        for control in self.findChildren(QComboBox): control.currentIndexChanged.connect(self.notify)
        height = {'Перемещать': 690, 'Вращать': 690, 'Масштабировать': 860, 'Отзеркалить': 650}[operation]
        available = self.screen().availableGeometry().height() if self.screen() else 950
        self.resize(self.minimumWidth(), min(height, available - 80))

    def spin(self, value=0, minimum=-1e9, suffix=' мм', decimals=4):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(minimum, 1e9)
        spin.setValue(float(value))
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(85)
        return spin

    def spins(self, values, **kwargs):
        return [self.spin(value, **kwargs) for value in values]

    @staticmethod
    def numbers(spins): return np.array([spin.value() for spin in spins])

    @staticmethod
    def set_values(spins, values):
        for spin, value in zip(spins, values):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def notify(self, *args):
        if not self._sync: self.changed.emit()

    def group(self, title):
        box = QGroupBox(title)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        box.setCheckable(True)
        box.setChecked(True)
        layout = QVBoxLayout(box)
        content = QWidget()
        layout.addWidget(content)
        inner = QVBoxLayout(content)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setAlignment(Qt.AlignTop)
        box.toggled.connect(content.setVisible)
        self.layout_body.addWidget(box)
        return inner

    def coordinates(self, layout, columns, labels):
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        for col, label in enumerate(labels): grid.addWidget(QLabel(label), 0, col + 1)
        for axis in range(3):
            grid.addWidget(QLabel('XYZ'[axis]), axis + 1, 0)
            for col, controls in enumerate(columns): grid.addWidget(controls[axis], axis + 1, col + 1)
        layout.addLayout(grid)

    def options(self):
        row = QHBoxLayout()
        row.addWidget(self.copy)
        row.addWidget(self.preview)
        self.layout_body.addLayout(row)

    def snapping(self, default, suffix):
        row = QHBoxLayout()
        self.snap = QCheckBox('Активировать привязку')
        self.snap.setChecked(True)
        self.snap_step = self.spin(default, minimum=.0001, suffix=suffix)
        row.addWidget(self.snap)
        row.addStretch()
        row.addWidget(QLabel('Шаг:'))
        row.addWidget(self.snap_step)
        self.layout_body.addLayout(row)
        self.snap.setToolTip('Шаг применяется при перетаскивании 3D-манипулятора. Числа можно вводить точно.')
        self.snap.toggled.connect(self.snap_step.setEnabled)

    def line_controls(self):
        self.layout_body.addWidget(self.along_line)
        self.line_box = QWidget()
        layout = QVBoxLayout(self.line_box)
        self.coordinates(layout, [self.line_a, self.line_b], ['Начало линии', 'Конец линии'])
        pick = QPushButton('Указать линию двумя точками')
        pick.clicked.connect(lambda: self.pick_requested.emit('line', 2))
        layout.addWidget(pick)
        if self.operation == 'Вращать':
            row = QHBoxLayout()
            row.addWidget(QLabel('Угол вокруг линии:'))
            row.addWidget(self.line_angle)
            layout.addLayout(row)
            self.along_line.toggled.connect(lambda enabled: [s.setEnabled(not enabled) for s in self.values])
        self.line_box.hide()
        self.along_line.toggled.connect(self.line_box.setVisible)
        self.layout_body.addWidget(self.line_box)

    def build_move(self):
        self.values = self.spins([0, 0, 0])
        self.target = self.spins(self.bounds.mean(axis=0))
        self.coordinates(self.layout_body, [self.target, self.values], ['Результирующие координаты', 'Относительное перемещение'])
        self.snapping(5, ' мм')
        self.line_controls()
        self.options()
        layout = self.group('Начало смещения')
        self.individual = QRadioButton('Индивидуально для каждой детали')
        common = QRadioButton('Общее для выбранных деталей')
        common.setChecked(True)
        origin_group = QButtonGroup(self)
        origin_group.addButton(common)
        origin_group.addButton(self.individual)
        layout.addWidget(common)
        layout.addWidget(self.individual)
        self.anchor_groups = []
        self.anchor_custom = self.spins(self.bounds.mean(axis=0))
        grid = QGridLayout()
        for col, text in enumerate(['', 'Мин.', 'Центр', 'Макс.', 'Задать', 'Координата']): grid.addWidget(QLabel(text), 0, col)
        for axis in range(3):
            grid.addWidget(QLabel('XYZ'[axis]), axis + 1, 0)
            group = QButtonGroup(self)
            for mode in range(4):
                button = QRadioButton()
                group.addButton(button, mode)
                grid.addWidget(button, axis + 1, mode + 1)
                if mode == 1: button.setChecked(True)
            self.anchor_custom[axis].setEnabled(False)
            group.idClicked.connect(lambda mode, a=axis: self.anchor_custom[a].setEnabled(mode == 3))
            group.idClicked.connect(self.move_origin_changed)
            self.anchor_groups.append(group)
            grid.addWidget(self.anchor_custom[axis], axis + 1, 5)
        layout.addLayout(grid)
        pick = QPushButton('Указать точку')
        pick.clicked.connect(lambda: self.pick_requested.emit('move_origin', 1))
        layout.addWidget(pick)
        row = QHBoxLayout()
        reset = QPushButton('К исходной позиции')
        reset.clicked.connect(lambda: self.reset_move(False))
        reset_z = QPushButton('К исходной Z позиции')
        reset_z.clicked.connect(lambda: self.reset_move(True))
        row.addWidget(reset); row.addWidget(reset_z)
        self.layout_body.addLayout(row)
        for control in self.values: control.valueChanged.connect(lambda _: self.sync_move(False))
        for control in self.target: control.valueChanged.connect(lambda _: self.sync_move(True))
        for control in self.anchor_custom: control.valueChanged.connect(self.move_origin_changed)
        self.individual.toggled.connect(self.move_origin_changed)
        self.along_line.toggled.connect(self.move_origin_changed)
        for spin in self.line_a + self.line_b: spin.valueChanged.connect(self.move_origin_changed)

    def move_anchor(self, bounds=None):
        if bounds is None: bounds = self.meshes[0].bounds if self.individual.isChecked() else self.bounds
        return anchor(bounds, [g.checkedId() for g in self.anchor_groups], self.numbers(self.anchor_custom))

    def sync_move(self, absolute):
        if self._sync: return
        self.absolute = absolute
        delta = self.numbers(self.target) - self.move_anchor() if absolute else self.numbers(self.values)
        if self.along_line.isChecked():
            try:
                direction = unit(self.numbers(self.line_b) - self.numbers(self.line_a))
                delta = direction * np.dot(delta, direction)
            except ValueError:
                pass  # The preview reports the invalid line and disables Apply.
        self.set_values(self.values, delta)
        self.set_values(self.target, self.move_anchor() + delta)

    def move_origin_changed(self, *args):
        self.sync_move(self.absolute)
        self.notify()

    def reset_move(self, z_only):
        origin_bounds = self.initial_mesh_bounds[0] if self.individual.isChecked() else self.initial_bounds
        target = self.move_anchor(origin_bounds)
        if z_only: target[:2] = self.numbers(self.target)[:2]
        self.set_values(self.target, target)
        self.sync_move(True)
        self.notify()

    def build_rotate(self):
        self.values = self.spins([0, 0, 0], suffix=' °')
        self.coordinates(self.layout_body, [self.values], ['Углы вращения'])
        self.snapping(45, ' °')
        self.line_controls()
        self.layout_body.addWidget(self.keep_z)
        self.options()
        layout = self.group('Центр вращения')
        self.center_mode = QButtonGroup(self)
        for index, text in enumerate(['Центр выбранного', 'Индивидуальный центр деталей', 'Заданный центр вращения']):
            button = QRadioButton(text)
            self.center_mode.addButton(button, index)
            layout.addWidget(button)
            if index == 0: button.setChecked(True)
        self.coordinates(layout, [self.center], ['Координаты центра'])
        row = QHBoxLayout()
        pick = QPushButton('Указать точку')
        pick.clicked.connect(lambda: self.pick_requested.emit('center', 1))
        default = QPushButton('Центр по умолчанию')
        default.clicked.connect(self.default_center)
        row.addWidget(pick); row.addWidget(default)
        layout.addLayout(row)
        self.center_mode.idClicked.connect(self.center_mode_changed)
        self.center_mode_changed(0)

    def center_mode_changed(self, index):
        for spin in self.center: spin.setEnabled(index == 2)
        self.individual.setChecked(index == 1)
        if index != 2: self.set_values(self.center, self.bounds.mean(axis=0))
        self.notify()

    def default_center(self):
        self.center_mode.button(0).setChecked(True)
        self.center_mode_changed(0)

    def reference_size(self):
        return self.meshes[0].extents if self.individual.isChecked() else self.bounds[1] - self.bounds[0]

    def build_scale(self):
        self.individual.setChecked(True)
        size = self.reference_size()
        self.values = self.spins([1, 1, 1], minimum=.00001, suffix='', decimals=5)
        self.final_size = self.spins(size, minimum=.00001)
        self.difference = self.spins([0, 0, 0])
        self.original_size = self.spins(size)
        for spin in self.original_size: spin.setReadOnly(True); spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.coordinates(self.layout_body, [self.values, self.final_size, self.difference, self.original_size],
                         ['Фактор', 'Конечный размер', 'Разница', 'Исходный размер'])
        self.uniform = QCheckBox('Равномерное масштабирование')
        self.layout_body.addWidget(self.uniform)
        self.scale_hint = QLabel('')
        self.scale_hint.setWordWrap(True)
        self.layout_body.addWidget(self.scale_hint)
        self.options()
        fit = self.group('Подогнать измерение')
        self.fit = QCheckBox('Включить режим подгона измерений')
        fit.addWidget(self.fit)
        self.measured = self.spin(0, minimum=0)
        self.measured.setReadOnly(True)
        self.fit_target = self.spin(1, minimum=.00001)
        self.fit_difference = self.spin(0)
        row = QGridLayout()
        for col, (text, spin) in enumerate([('Конечный размер', self.fit_target), ('Разница', self.fit_difference), ('Исходный размер', self.measured)]):
            row.addWidget(QLabel(text), 0, col); row.addWidget(spin, 1, col)
        fit.addLayout(row)
        self.pick_measure = QPushButton('Указать две точки измерения')
        self.pick_measure.clicked.connect(lambda: self.pick_requested.emit('measure', 2))
        fit.addWidget(self.pick_measure)
        self.fit.toggled.connect(self.fit_mode_changed)
        self.fit_target.valueChanged.connect(lambda _: self.fit_changed(False))
        self.fit_difference.valueChanged.connect(lambda _: self.fit_changed(True))
        self.fit_mode_changed(False)
        library = self.group('Библиотека факторов масштабирования')
        row = QHBoxLayout()
        self.presets = QListWidget()
        self.presets.setFixedHeight(80)
        row.addWidget(self.presets)
        buttons = QVBoxLayout()
        self.library_buttons = {}
        for label, action in [('Новый', 'new'), ('Редактировать', 'edit'), ('Удалить', 'delete')]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, a=action: self.edit_preset(a))
            self.library_buttons[action] = button
            buttons.addWidget(button)
        row.addLayout(buttons)
        library.addLayout(row)
        defaults = [{'name': 'inch → mm', 'factors': [25.4] * 3}, {'name': 'mm → inch', 'factors': [1 / 25.4] * 3}]
        try:
            self.library = json.loads(self.parent().settings.value('scale_presets_v1', json.dumps(defaults)))
            if not isinstance(self.library, list) or any(not isinstance(p['name'], str) or len(p['factors']) != 3 or not np.isfinite(p['factors']).all() or min(p['factors']) <= 0 for p in self.library): raise ValueError()
        except (ValueError, TypeError, KeyError): self.library = defaults
        self.reload_presets()
        self.presets.currentRowChanged.connect(self.use_preset)
        self.presets.currentRowChanged.connect(self.update_preset_buttons)
        self.update_preset_buttons()
        layout = self.group('Центр масштабирования')
        self.coordinates(layout, [self.center], ['Координаты центра'])
        self.individual.setText('Масштабировать каждую деталь относительно её центра')
        layout.addWidget(self.individual)
        layout.addWidget(self.keep_z)
        self.individual.toggled.connect(self.scale_reference_changed)
        for axis in range(3):
            self.values[axis].valueChanged.connect(lambda _, a=axis: self.sync_scale(a, 'factor'))
            self.final_size[axis].valueChanged.connect(lambda _, a=axis: self.sync_scale(a, 'final'))
            self.difference[axis].valueChanged.connect(lambda _, a=axis: self.sync_scale(a, 'difference'))
        self.uniform.toggled.connect(lambda enabled: self.sync_scale(0, 'factor') if enabled else None)
        self.scale_reference_changed()

    def scale_reference_changed(self, *args):
        for spin in self.center: spin.setEnabled(not self.individual.isChecked())
        size = self.reference_size()
        self.set_values(self.original_size, size)
        for index, length in enumerate(size):
            for spins in (self.values, self.final_size, self.difference): spins[index].setEnabled(bool(length > 1e-10 and not self.fit.isChecked()))
        self.sync_scale(0, 'factor')
        self.scale_hint.setText('Размеры первой выбранной детали; факторы применяются ко всем.' if self.individual.isChecked() else 'Размеры общего габарита выбранных деталей.')

    def sync_scale(self, axis, source):
        if self._sync: return
        self._sync = True
        size = self.reference_size()
        factors = self.numbers(self.values)
        if size[axis] > 1e-10:
            if source == 'final': factors[axis] = self.final_size[axis].value() / size[axis]
            elif source == 'difference': factors[axis] = max(.00001, (size[axis] + self.difference[axis].value()) / size[axis])
        if self.uniform.isChecked(): factors[:] = factors[axis]
        self.set_values(self.values, factors)
        self.set_values(self.final_size, size * factors)
        self.set_values(self.difference, size * (factors - 1))
        self._sync = False

    def fit_mode_changed(self, enabled):
        for control in (self.fit_target, self.fit_difference, self.pick_measure): control.setEnabled(enabled)
        for controls in (self.values, self.final_size, self.difference):
            for index, spin in enumerate(controls): spin.setEnabled(bool(not enabled and self.reference_size()[index] > 1e-10))
        if enabled: self.fit_changed(False)

    def fit_changed(self, difference):
        measured = self.measured.value()
        if difference: self.set_values([self.fit_target], [max(.00001, measured + self.fit_difference.value())])
        else: self.set_values([self.fit_difference], [self.fit_target.value() - measured])
        if self.fit.isChecked() and measured > 1e-10:
            self.set_values(self.values, [self.fit_target.value() / measured] * 3)
            self.sync_scale(0, 'factor')

    def reload_presets(self):
        self.presets.blockSignals(True)
        self.presets.clear()
        self.presets.addItems([p['name'] for p in self.library])
        self.presets.blockSignals(False)
        self.update_preset_buttons()

    def update_preset_buttons(self, *args):
        selected = 0 <= self.presets.currentRow() < len(self.library)
        self.library_buttons['edit'].setEnabled(selected)
        self.library_buttons['delete'].setEnabled(selected)

    def use_preset(self, index):
        if 0 <= index < len(self.library):
            factors = self.library[index]['factors']
            self.fit.setChecked(False)
            self.uniform.setChecked(bool(np.allclose(factors, factors[0])))
            self.set_values(self.values, factors)
            self.sync_scale(0, 'factor')
            self.notify()

    def edit_preset(self, action):
        index = self.presets.currentRow()
        if action != 'new' and index < 0: return
        if action == 'delete': del self.library[index]
        else:
            preset = {'name': '', 'factors': self.numbers(self.values).tolist()} if action == 'new' else self.library[index]
            name, accepted = QInputDialog.getText(self, 'Фактор масштаба', 'Название:', text=preset['name'])
            if not accepted or not name.strip(): return
            dialog = QDialog(self)
            dialog.setWindowTitle(name.strip())
            layout = QVBoxLayout(dialog)
            spins = self.spins(preset['factors'], minimum=.00001, suffix='', decimals=5)
            self.coordinates(layout, [spins], ['Факторы масштаба'])
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if not dialog.exec(): return
            entry = {'name': name.strip(), 'factors': self.numbers(spins).tolist()}
            if action == 'new': self.library.append(entry)
            else: self.library[index] = entry
        self.parent().settings.setValue('scale_presets_v1', json.dumps(self.library, ensure_ascii=False))
        self.reload_presets()

    def build_mirror(self):
        self.plane_group = QButtonGroup(self)
        for index, text in enumerate(['YZ-плоскость', 'XZ-плоскость', 'XY-плоскость', 'Указать']):
            button = QRadioButton(text)
            self.plane_group.addButton(button, index)
            self.layout_body.addWidget(button)
            if index == 0: button.setChecked(True)
        row = QHBoxLayout()
        self.plane_method = QComboBox()
        self.plane_method.addItems(['3 точки', 'Точка и нормаль'])
        self.pick_plane = QPushButton('Указать')
        self.pick_plane.clicked.connect(lambda: self.pick_requested.emit('plane' if self.plane_method.currentIndex() == 0 else 'plane_origin', 3 if self.plane_method.currentIndex() == 0 else 1))
        row.addWidget(self.plane_method); row.addWidget(self.pick_plane)
        self.layout_body.addLayout(row)
        self.normal = self.spins([1, 0, 0], suffix='')
        normal_layout = self.group('Нормаль произвольной плоскости')
        self.coordinates(normal_layout, [self.normal], ['Направление'])
        self.options()
        layout = self.group('Позиция плоскости')
        self.set_values(self.center, [0, 0, 0])
        self.coordinates(layout, [self.center], ['Координаты'])
        self.individual.setText('Центр детали')
        layout.addWidget(self.individual)
        self.plane_group.idClicked.connect(self.mirror_mode_changed)
        self.individual.toggled.connect(lambda enabled: [spin.setEnabled(not enabled) for spin in self.center])
        self.mirror_mode_changed(0)

    def mirror_mode_changed(self, index):
        for control in [self.plane_method, self.pick_plane, *self.normal]: control.setEnabled(index == 3)
        if index < 3: self.set_values(self.normal, np.eye(3)[index])
        self.notify()

    def accept_points(self, purpose, points):
        if purpose == 'line':
            unit(np.asarray(points[1]) - points[0])
            self.set_values(self.line_a, points[0]); self.set_values(self.line_b, points[1])
            self.along_line.setChecked(True)
        elif purpose == 'move_origin':
            for group, spin in zip(self.anchor_groups, self.anchor_custom): group.button(3).setChecked(True); spin.setEnabled(True)
            self.set_values(self.anchor_custom, points[0]); self.move_origin_changed()
        elif purpose == 'center':
            self.center_mode.button(2).setChecked(True)
            self.center_mode_changed(2)
            self.set_values(self.center, points[0])
        elif purpose == 'measure':
            distance = np.linalg.norm(np.asarray(points[1]) - points[0])
            if distance < 1e-10: raise ValueError('Точки измерения должны различаться.')
            self.set_values([self.measured], [distance]); self.fit_changed(False)
        elif purpose in ('plane', 'plane_origin'):
            origin, normal = plane_from_points(points) if purpose == 'plane' else (points[0], self.numbers(self.normal))
            self.set_values(self.center, origin); self.set_values(self.normal, normal)
            self.individual.setChecked(False)
            self.plane_group.button(3).setChecked(True)
            self.mirror_mode_changed(3)
        self.notify()

    def parameters(self):
        params = dict(advanced=True, create_copy=self.copy.isChecked(), individual=self.individual.isChecked(),
                      center=self.numbers(self.center), keep_z=self.keep_z.isChecked())
        if self.operation == 'Перемещать':
            params.update(values=self.numbers(self.values), target=self.numbers(self.target), absolute=self.absolute,
                anchor_modes=[g.checkedId() for g in self.anchor_groups], anchor_custom=self.numbers(self.anchor_custom),
                along_line=self.along_line.isChecked(), line_a=self.numbers(self.line_a), line_b=self.numbers(self.line_b))
        elif self.operation == 'Вращать':
            params.update(values=self.numbers(self.values), along_line=self.along_line.isChecked(),
                line_a=self.numbers(self.line_a), line_b=self.numbers(self.line_b), line_angle=self.line_angle.value())
        elif self.operation == 'Масштабировать':
            if self.fit.isChecked() and self.measured.value() <= 1e-10: raise ValueError('Укажите две точки исходного измерения.')
            params['values'] = self.numbers(self.values)
        else: params['normal'] = self.numbers(self.normal)
        return params

    def rebase(self, meshes):
        self._sync = True
        self.meshes = meshes
        self.bounds = selection_bounds(meshes)
        if self.operation == 'Перемещать':
            self.absolute = False
            self.set_values(self.values, [0, 0, 0]); self.set_values(self.target, self.move_anchor())
        elif self.operation == 'Вращать':
            self.set_values(self.values, [0, 0, 0]); self.set_values([self.line_angle], [0])
            if self.center_mode.checkedId() != 2: self.set_values(self.center, self.bounds.mean(axis=0))
        elif self.operation == 'Масштабировать':
            self.fit.setChecked(False)
            self.set_values([self.measured, self.fit_difference, self.fit_target], [0, 0, 1])
            self.set_values(self.values, [1, 1, 1])
            self._sync = False; self.scale_reference_changed(); self._sync = True
        self._sync = False
