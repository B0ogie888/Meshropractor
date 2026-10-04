"""Reviewable parameters for mesh repair and direct surface editing."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QLineEdit, QLabel, QPushButton,
    QPlainTextEdit, QWidget)
from display_settings import DIALOG_STYLE
from repair_ribbon import REPAIR_COMMANDS


# key, caption, default, minimum, maximum, suffix; integer defaults use integer fields.
FIELDS = {
    'auto': [('passes', 'Проходов', 3, 1, 10, ''),
             ('tolerance_mm', 'Допуск изменения формы', .05, .001, 10., ' мм'),
             ('target_faces', 'Цель оптимизации (0 — без неё)', 400000, 0, 5000000, '')],
    'wrap': [('voxel_size_mm', 'Размер ячейки оболочки', 1., .001, 1000., ' мм')],
    'stitch': [('tolerance_mm', 'Расстояние сшивания', .01, .000001, 10., ' мм')],
    'holes': [('max_diameter_mm', 'Максимальный диаметр отверстия', 10., .001, 10000., ' мм')],
    'noise': [('min_faces', 'Меньше треугольников', 10, 0, 1000000, ''),
              ('min_volume_mm3', 'Объём меньше', 0., 0., 1e9, ' мм³')],
    'remove_small': [('min_faces', 'Меньше треугольников', 10, 0, 1000000, ''),
                     ('min_volume_mm3', 'Объём меньше', 0., 0., 1e9, ' мм³')],
    'slivers': [('min_angle_deg', 'Минимальный угол треугольника', 5., .01, 59., ' °')],
    'overlaps': [('tolerance_mm', 'Допуск совпадения плоскостей', .000001, .00000001, .1, ' мм')],
    'decimate': [('target_ratio', 'Сохранить долю треугольников', .5, .01, 1., '')],
    'smooth': [('iterations', 'Итераций', 10, 1, 100, ''), ('relaxation', 'Сила сглаживания', .1, .001, .5, '')],
    'clean_smooth': [('iterations', 'Итераций', 10, 1, 100, ''), ('relaxation', 'Сила сглаживания', .1, .001, .5, '')],
    'subdivide': [('iterations', 'Уровней подразделения', 1, 1, 3, '')],
    'remesh': [('target_edge_mm', 'Целевая длина ребра', 1., .001, 10000., ' мм'),
               ('iterations', 'Итераций переразбивки', 2, 1, 5, '')],
}
MANUAL = {'fill_hole', 'bridge', 'add_triangle', 'delete_faces', 'move_vertices', 'drag_vertices'}
NOTES = {
    'auto': 'Полное лечение закрывает отверстия и перестраивает дефекты. Расчёт может занять несколько минут. '
            'Перед применением проверьте отчёт и изменение формы. Превышение допуска блокирует применение.',
    'wrap': 'Создаёт приближённую внешнюю оболочку по объёмной сетке. Мелкие элементы и полости могут исчезнуть. '
            'Размер ячейки задаёт детализацию; проверьте результат перед применением.',
    'normals': 'Согласование ориентации соседних треугольников и внешних нормалей замкнутых оболочек.',
    'stitch': 'Объединяет близкие вершины в пределах допуска. Большой допуск может изменить мелкие элементы.',
    'holes': 'Закрывает простые плоские контуры, включая вогнутые, в пределах заданного размера. '
             'Конструктивные отверстия тоже могут закрыться. Для одного отверстия используйте «Режим заливки дыр».',
    'noise': 'Удаляет малые несвязанные фрагменты внутри выбранных деталей. Самый крупный фрагмент сохраняется.',
    'remove_small': 'Удаляет только выбранные детали, подходящие хотя бы под один ненулевой порог. '
                    'Для открытых сеток учитывается только число треугольников.',
    'unify': 'Булево объединение выбранных замкнутых деталей одной платформы. Внутренние пересечения удаляются.',
    'split': 'Каждая связная оболочка становится отдельной деталью. Поддержки переносятся по привязанным поверхностям. '
             'Области, охватывающие несколько оболочек, нужно разделить заранее. За один раз — до 200 фрагментов.',
    'slivers': 'Находит и подсвечивает треугольники с малым углом. Геометрия не удаляется.',
    'overlaps': 'Находит и подсвечивает площадные нахлёсты и поперечные пересечения треугольников. Геометрия не изменяется.',
    'fill_hole': 'Щёлкните по вершине границы отверстия. Будет закрыт только связанный с ней простой контур.',
    'bridge': 'Выберите четыре вершины: две на первом открытом ребре, затем две на втором. Создаются два треугольника перемычки.',
    'add_triangle': 'Выберите три существующие вершины в порядке обхода будущего треугольника.',
    'delete_faces': 'Удаляет выделенные треугольники. Можно выбирать их щелчками в сцене; повторный щелчок снимает выбор.',
    'move_vertices': 'Выберите вершины щелчками или возьмите вершины выделенных поверхностей. Затем задайте смещение.',
    'drag_vertices': 'Включите выбор в сцене и потяните вершину левой кнопкой. Перемещение идёт в плоскости экрана; '
                     'после отпускания подготовится результат для проверки.',
    'clip': 'Разрезает геометрию выбранных деталей плоскостью. Сторона задаёт сохраняемую часть. '
            'Это изменение сетки, в отличие от сечений отображения.',
    'remesh': 'Приближает плотность сетки к заданной длине ребра разбиением, сокращением и сглаживанием. '
              'Точная длина рёбер и равносторонность не гарантируются; форма может измениться.',
}


class RepairDialog(QDialog):
    changed = Signal()
    prepare_requested = Signal()
    apply_requested = Signal()
    pick_requested = Signal(bool)
    selection_requested = Signal()
    selection_changed = Signal()

    def __init__(self, operation, count, center, parent, *, caption=None):
        super().__init__(parent)
        self.operation = operation
        self.running = False
        self.setWindowTitle(caption or REPAIR_COMMANDS[operation])
        self.setStyleSheet(DIALOG_STYLE + '''
            QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #262626; color: #e0e0e0; border: 1px solid #666; padding: 4px;
            }
            QPushButton:disabled { color: #777; background: #303030; border-color: #494949; }
        ''')
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        title = QLabel(f'Выбрано деталей: {count}. Изменения применяются одним шагом отмены.')
        title.setWordWrap(True); layout.addWidget(title)
        self.note = QLabel(NOTES.get(operation, 'Подготовьте результат, проверьте предпросмотр и нажмите «Применить».'))
        self.note.setWordWrap(True); layout.addWidget(self.note)
        self.parameters_widget = QWidget()
        self.form = QFormLayout(self.parameters_widget)
        self.form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.parameters_widget)
        self.fields = {}
        for key, caption, default, minimum, maximum, suffix in FIELDS.get(operation, []):
            spin = QSpinBox() if isinstance(default, int) else QDoubleSpinBox()
            if isinstance(spin, QDoubleSpinBox): spin.setDecimals(8 if key == 'tolerance_mm' else 4)
            spin.setRange(minimum, maximum); spin.setValue(default); spin.setSuffix(suffix)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda *_: self.changed.emit())
            self.fields[key] = spin; self.form.addRow(caption + ':', spin)
        self.flags = {}
        for key, caption, checked in (
            [('flip', 'Развернуть нормали вместо согласования', False)] if operation == 'normals' else
            [('preserve_boundary', 'Сохранять открытые границы', True)] if operation in ('smooth', 'clean_smooth') else
            [('preserve_topology', 'Сохранять топологию', True)] if operation == 'decimate' else []):
            check = QCheckBox(caption); check.setChecked(checked); check.toggled.connect(lambda *_: self.changed.emit())
            self.flags[key] = check; self.form.addRow(check)
        self.only_faces = QCheckBox('Только выделенные треугольники')
        self.only_faces.toggled.connect(lambda *_: self.changed.emit())
        if operation in ('normals', 'smooth', 'slivers', 'overlaps'): self.form.addRow(self.only_faces)
        else: self.only_faces.hide()
        self.ids = QLineEdit()
        self._selected_ids_cache = None
        self.ids.setMaxLength(2147483647)
        self.ids.setPlaceholderText('Выберите в сцене или введите номера через запятую')
        self.ids.textChanged.connect(self._ids_edited)
        self.pick_button = QPushButton('Выбирать в сцене'); self.pick_button.setCheckable(True)
        self.pick_button.toggled.connect(self.pick_requested.emit)
        self.from_selection = QPushButton('Из выделенных поверхностей')
        self.from_selection.clicked.connect(self.selection_requested.emit)
        if operation in MANUAL:
            self.form.addRow('Треугольники (с 0):' if operation == 'delete_faces' else 'Вершины (с 0):', self.ids)
            row = QHBoxLayout(); row.addWidget(self.pick_button); row.addWidget(self.from_selection)
            clear = QPushButton('Очистить'); clear.clicked.connect(lambda: self.set_selected_ids([])); row.addWidget(clear)
            self.form.addRow(row)
        else:
            self.ids.hide(); self.pick_button.hide(); self.from_selection.hide()
        self.vectors = {}
        if operation in ('move_vertices', 'drag_vertices'):
            self.add_vector('delta', 'Смещение, мм', [0, 0, 0])
            self.absolute = QCheckBox('Координаты одной вершины вместо смещения')
            self.absolute.toggled.connect(lambda *_: self.changed.emit()); self.form.addRow(self.absolute)
        if operation == 'clip':
            self.add_vector('point', 'Точка плоскости, мм', center)
            self.add_vector('normal', 'Нормаль плоскости', [0, 0, 1])
            self.side = QComboBox(); self.side.addItems(['В сторону нормали', 'Против нормали'])
            self.side.currentIndexChanged.connect(lambda *_: self.changed.emit()); self.form.addRow('Сохранить:', self.side)
            self.cap = QCheckBox('Закрыть поверхность разреза'); self.cap.setChecked(True)
            self.cap.toggled.connect(lambda *_: self.changed.emit()); self.form.addRow(self.cap)
        self.preview = QCheckBox('Показывать подготовленный результат'); self.preview.setChecked(True)
        layout.addWidget(self.preview)
        support_note = QLabel('После изменения сетки проверьте поддержки. Их геометрия сохраняется; '
                             'привязки к граням сохраняются или перенумеровываются, где это возможно, иначе сбрасываются.')
        support_note.setWordWrap(True); layout.addWidget(support_note); self.support_note = support_note
        self.status = QLabel('Alt + мышь — навигация. Исходная геометрия сохраняется до применения.')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True); self.report.setMaximumHeight(170)
        layout.addWidget(self.report)
        row = QHBoxLayout(); layout.addLayout(row)
        self.prepare = QPushButton('Подготовить результат')
        self.prepare.clicked.connect(self.prepare_requested.emit); row.addWidget(self.prepare)
        self.cancel = QPushButton('Остановить'); self.cancel.setEnabled(False); row.addWidget(self.cancel)
        self.apply = QPushButton('Применить'); self.apply.setEnabled(False)
        self.apply.clicked.connect(self.apply_requested.emit); row.addWidget(self.apply)
        self.close_button = QPushButton('Закрыть'); self.close_button.clicked.connect(self.reject); row.addWidget(self.close_button)
        if operation in ('slivers', 'overlaps'):
            self.prepare.setText('Найти и подсветить'); self.apply.hide(); self.preview.hide()

    def add_vector(self, key, label, values):
        row = QHBoxLayout(); fields = []
        for axis, value in zip('XYZ', values):
            spin = QDoubleSpinBox(); spin.setDecimals(5); spin.setRange(-1e9, 1e9)
            spin.setPrefix(axis + ': '); spin.setValue(float(value)); spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda *_: self.changed.emit()); row.addWidget(spin); fields.append(spin)
        self.vectors[key] = fields; self.form.addRow(label + ':', row)

    def selected_ids(self):
        if self._selected_ids_cache is not None:
            return list(self._selected_ids_cache)
        text = self.ids.text().replace(';', ',').replace(' ', ',')
        try: return list(dict.fromkeys(int(value) for value in text.split(',') if value))
        except ValueError: raise ValueError('Номера должны быть целыми числами, разделёнными запятыми.')

    def _ids_edited(self, *_):
        self._selected_ids_cache = None
        self.ids.setReadOnly(False)
        self.ids.setToolTip('')
        self.changed.emit()
        self.selection_changed.emit()

    def set_selected_ids(self, values):
        """Keep large surface selections intact without filling a line edit with megabytes."""
        values = list(dict.fromkeys(map(int, values)))
        large = len(values) > 1000
        self._selected_ids_cache = tuple(values) if large else None
        was_blocked = self.ids.blockSignals(True)
        self.ids.setReadOnly(large)
        self.ids.setText(f'Выбрано: {len(values):,}. Изменить выбор — в сцене; сбросить — «Очистить».' if large
                         else ', '.join(map(str, values)))
        self.ids.setToolTip('Все выбранные номера сохранены; сводная подпись не ограничивает операцию.' if large else '')
        self.ids.blockSignals(was_blocked)
        self.changed.emit()
        self.selection_changed.emit()

    def parameters(self):
        result = {key: spin.value() for key, spin in self.fields.items()}
        result.update({key: check.isChecked() for key, check in self.flags.items()})
        result.update({key: [spin.value() for spin in values] for key, values in self.vectors.items()})
        if self.operation in ('move_vertices', 'drag_vertices') and self.absolute.isChecked():
            result['absolute'] = result.pop('delta')
        if self.operation == 'clip': result.update(side='positive' if self.side.currentIndex() == 0 else 'negative', cap=self.cap.isChecked())
        return result

    def set_running(self, running):
        self.running = running
        self.parameters_widget.setEnabled(not running)
        self.prepare.setEnabled(not running); self.cancel.setEnabled(running)
        self.close_button.setEnabled(not running)
        if running: self.apply.setEnabled(False)

    def reject(self):
        if self.running:
            self.cancel.click()
            self.status.setText('Запрошена отмена. Окно можно закрыть после завершения расчёта.')
            return
        super().reject()
