"""Modeless parameters for placement, orientation and packing previews."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QWidget,
    QLabel, QComboBox, QCheckBox, QDoubleSpinBox, QPushButton, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)

from display_settings import DIALOG_STYLE
from placement_ribbon import PLACEMENT_COMMANDS


PLATFORM_OPERATIONS = {'auto_arrange', 'fit_platform', 'pack_3d'}
ORIENTATION_OPERATIONS = {'optimize_orientation', 'compare_orientations', 'minimize_bbox'}
NOTES = {
    'free_move': 'Потяните выбранную деталь левой кнопкой мыши: всё выделение перемещается вместе. '
                 'Alt + мышь — вращение сцены. Можно задать точное смещение по XYZ.',
    'top_bottom': 'Укажите треугольник детали. Его нормаль будет направлена вверх или вниз; '
                  'выбранная поверхность станет горизонтальной.',
    'auto_arrange': 'Размещает выбранные детали на платформе с зазором. Учитываются поддержки, '
                    'невыбранные детали этой платформы и запретные зоны, даже если они скрыты.',
    'optimize_orientation': 'Сравнивает набор ориентаций по выбранному критерию. Площадь нависаний — '
                            'геометрическая оценка, а не расчёт объёма поддержек или качества печати.',
    'compare_orientations': 'Подготовьте варианты и выбирайте строки таблицы для просмотра. '
                            'Применяется только выбранный вариант одной детали.',
    'minimize_bbox': 'Ищет ориентацию с меньшим объёмом ограничивающего параллелепипеда. '
                     'Форма и размеры самой детали сохраняются.',
    'fit_platform': 'Подбирает положение выбранной группы в габаритах платформы. '
                    'Взаимное расположение деталей сохраняется; масштаб не меняется.',
    'sort_by_shape': 'Переносит ориентацию образца на геометрически похожие детали. '
                     'Центры деталей сохраняются, размеры не меняются. Непохожие остаются на месте. '
                     'Соответствие проверяется по выборкам поверхности; симметрии могут давать несколько вариантов.',
    'pack_3d': 'Размещает детали по всему объёму камеры с зазорами. Проверка использует габариты, '
               'поэтому плотная укладка сложных форм не гарантируется. Размещение над платформой '
               'само по себе не создаёт опор — проверьте пригодность для вашей технологии печати.',
}


class PlacementDialog(QDialog):
    changed = Signal()
    prepare_requested = Signal()
    apply_requested = Signal()
    pick_requested = Signal(bool)
    variant_requested = Signal(int)

    def __init__(self, operation, names, platforms, parent=None):
        super().__init__(parent)
        self.operation, self.running = operation, False
        self.setWindowTitle(PLACEMENT_COMMANDS[operation])
        self.setMinimumWidth(760 if operation == 'compare_orientations' else 620)
        self.setStyleSheet(DIALOG_STYLE + '''
            QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background: #262626; color: #e0e0e0; border: 1px solid #666; padding: 4px;
            }
            QPushButton:disabled {color: #777; background: #303030; border-color: #494949;}
            QHeaderView::section {background: #383838; color: #e0e0e0; padding: 5px;}
            QTableWidget::item:selected {background: #36556b; color: white;}
        ''')
        root = QVBoxLayout(self)
        count = QLabel(f'Выбрано деталей: {len(names)}. Применение — один шаг отмены.')
        root.addWidget(count)
        note = QLabel(NOTES[operation]); note.setWordWrap(True); root.addWidget(note)
        self.parameters_widget = QWidget()
        self.form = QFormLayout(self.parameters_widget); self.form.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.parameters_widget)
        self.platform = QComboBox()
        for platform in platforms:
            dims = ' × '.join(f'{v:g}' for v in platform['dim'])
            self.platform.addItem(f"{platform['name']} ({dims} мм)", platform)
        if operation in PLATFORM_OPERATIONS:
            self.form.addRow('Платформа:', self.platform)
        else: self.platform.hide()
        self.fields = {}
        if operation in PLATFORM_OPERATIONS:
            if operation != 'fit_platform': self.number('gap_mm', 'Зазор между деталями', 2, 0, 1000)
            self.number('margin_mm', 'Отступ от краёв платформы', 2, 0, 1000)
            self.number('clearance_mm', 'Нижний уровень над платформой', 0, 0, 10000)
        self.rotation = QCheckBox('Подбирать ориентацию всей группы' if operation == 'fit_platform' else
                                  'Разрешить повороты по всем осям на 90°' if operation == 'pack_3d' else
                                  'Разрешить повороты вокруг Z на 90°')
        self.rotation.setChecked(True)
        if operation in PLATFORM_OPERATIONS: self.form.addRow(self.rotation)
        else: self.rotation.hide()
        self.objective = QComboBox()
        for title, key in [('Минимальная высота', 'height'), ('Минимальная площадь габаритов XY', 'footprint'),
                           ('Минимальная площадь нависаний', 'support'), ('Минимальный габаритный объём', 'bbox')]:
            self.objective.addItem(title, key)
        if operation in {'optimize_orientation', 'compare_orientations'}:
            self.form.addRow('Критерий:', self.objective)
            self.number('overhang_angle_deg', 'Порог нависания от горизонтали', 45, 1, 89, ' °')
        else: self.objective.hide()
        self.reference = QComboBox(); self.reference.addItems(names)
        if operation == 'sort_by_shape':
            self.form.addRow('Деталь-образец:', self.reference)
            self.number('tolerance_mm', 'Допуск сходства поверхностей', .1, .00001, 100)
        else: self.reference.hide()
        self.drop = QCheckBox('Поставить нижнюю точку каждой детали на Z = 0')
        self.drop.setChecked(operation != 'sort_by_shape')
        if operation in ORIENTATION_OPERATIONS | {'sort_by_shape', 'top_bottom'}: self.form.addRow(self.drop)
        else: self.drop.hide()
        self.pick = QPushButton('Выбирать поверхность в сцене'); self.pick.setCheckable(True)
        self.surface = QLabel('Поверхность ещё не выбрана')
        self.side = QComboBox(); self.side.addItems(['Нижняя поверхность: нормаль вниз (−Z)', 'Верхняя поверхность: нормаль вверх (+Z)'])
        if operation == 'top_bottom':
            self.form.addRow(self.side); self.form.addRow(self.pick); self.form.addRow(self.surface)
        else: self.side.hide(); self.pick.hide(); self.surface.hide()
        self.plane = QComboBox(); self.plane.addItems(['Плоскость экрана', 'XY', 'XZ', 'YZ'])
        self.delta = []
        if operation == 'free_move':
            self.form.addRow('Плоскость перетаскивания:', self.plane)
            self.number('snap_mm', 'Шаг привязки (0 — свободно)', 0, 0, 1000)
            row = QHBoxLayout()
            for axis in 'XYZ':
                spin = QDoubleSpinBox(); spin.setDecimals(5); spin.setRange(-1e9, 1e9)
                spin.setPrefix(axis + ': '); spin.setKeyboardTracking(False)
                self.delta.append(spin); row.addWidget(spin)
            self.form.addRow('Смещение, мм:', row)
        else: self.plane.hide()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Вариант', 'Высота, мм', 'Габариты XY, мм²', 'Нависания, мм²', 'Габариты, мм³'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide(); self.table.setMaximumHeight(220)
        if operation == 'compare_orientations': root.addWidget(self.table)
        else: self.table.hide()
        self.preview = QCheckBox('Предпросмотр'); self.preview.setChecked(True); root.addWidget(self.preview)
        hint = QLabel('Поддержки перемещаются вместе с деталью. После поворота проверьте их опору на платформу.')
        hint.setWordWrap(True); root.addWidget(hint)
        self.status = QLabel('Исходные детали сохраняются до нажатия «Применить».')
        self.status.setWordWrap(True); root.addWidget(self.status)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True); self.report.setMaximumHeight(115)
        root.addWidget(self.report)
        row = QHBoxLayout(); root.addLayout(row)
        self.prepare = QPushButton('Подготовить результат'); row.addWidget(self.prepare)
        self.cancel = QPushButton('Остановить'); self.cancel.setEnabled(False); row.addWidget(self.cancel)
        self.apply = QPushButton('Применить'); self.apply.setEnabled(False); row.addWidget(self.apply)
        self.close_button = QPushButton('Закрыть'); row.addWidget(self.close_button)
        if operation == 'free_move': self.prepare.hide(); self.cancel.hide(); self.report.hide()
        for control in self.parameters_widget.findChildren(QDoubleSpinBox): control.valueChanged.connect(lambda *_: self.changed.emit())
        for control in self.parameters_widget.findChildren(QCheckBox): control.toggled.connect(lambda *_: self.changed.emit())
        for control in self.parameters_widget.findChildren(QComboBox): control.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.prepare.clicked.connect(self.prepare_requested.emit)
        self.apply.clicked.connect(self.apply_requested.emit)
        self.pick.toggled.connect(self.pick_requested.emit)
        self.close_button.clicked.connect(self.reject)
        self.table.currentCellChanged.connect(lambda row, *_: self.variant_requested.emit(row))

    def number(self, key, title, value, minimum, maximum, suffix=' мм'):
        control = QDoubleSpinBox(); control.setDecimals(5); control.setRange(minimum, maximum)
        control.setValue(value); control.setSuffix(suffix); control.setKeyboardTracking(False)
        self.fields[key] = control; self.form.addRow(title + ':', control)

    def parameters(self):
        result = {key: control.value() for key, control in self.fields.items()}
        result.update(platform=self.platform.currentData(), allow_rotation=self.rotation.isChecked(),
                      objective=self.objective.currentData(), reference_index=self.reference.currentIndex(),
                      drop=self.drop.isChecked(), side=self.side.currentIndex(), plane=self.plane.currentIndex(),
                      delta=[spin.value() for spin in self.delta])
        return result

    def set_variants(self, variants):
        self.table.blockSignals(True)
        self.table.setRowCount(len(variants))
        for row, variant in enumerate(variants):
            values = [variant.get('label', f'Вариант {row + 1}')]
            values += [f"{variant.get(key, 0):.3f}" for key in ('height_mm', 'footprint_mm2', 'overhang_area_mm2', 'bbox_volume_mm3')]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value); item.setToolTip(value)
                self.table.setItem(row, column, item)
        self.table.selectRow(0); self.table.blockSignals(False)

    def set_running(self, running):
        self.running = running
        self.parameters_widget.setEnabled(not running); self.table.setEnabled(not running)
        self.prepare.setEnabled(not running); self.cancel.setEnabled(running); self.close_button.setEnabled(not running)
        if running: self.apply.setEnabled(False)

    def reject(self):
        if self.running:
            self.cancel.click(); self.status.setText('Запрошена отмена; ожидается завершение текущего шага.')
        else: super().reject()
