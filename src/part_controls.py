"""Visible, selection-aware part controls backed by the existing scene model."""
from PySide6.QtCore import QTimer, QSignalBlocker, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QColorDialog, QSlider, QSizePolicy)

from ribbon_layout import asset_icon


class PartControls(QWidget):
    MODES = [('Затенение', 'shaded'), ('Треугольники', 'triangles'),
             ('Затенение и каркас', 'shaded_wire'), ('Каркас', 'wireframe'),
             ('Габариты', 'bbox'), ('Прозрачный', 'transparent'), ('Без затенения', 'flat')]

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setObjectName('PartControls')
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 12, 0, 4); layout.setSpacing(8)
        layout.setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        title = QLabel('СВОЙСТВА ВЫБРАННОГО'); title.setObjectName('PartsEyebrow')
        self.summary = QLabel('Не выбрано'); self.summary.setObjectName('SelectionCount')
        heading = QHBoxLayout(); heading.addWidget(title); heading.addStretch(); heading.addWidget(self.summary)
        layout.addLayout(heading)
        self.caption = QLabel('Нажмите на деталь в списке или сцене'); self.caption.setObjectName('PartsHint')
        self.caption.setWordWrap(True); layout.addWidget(self.caption)
        actions = QHBoxLayout()
        self.hide_button = QPushButton('Скрыть')
        self.show_button = QPushButton('Показать')
        self.color_button = QPushButton('Цвет…')
        for b in (self.hide_button, self.show_button, self.color_button): actions.addWidget(b)
        self.color_button.setIcon(asset_icon('display', 'triangle_colors'))
        self.hide_button.setToolTip('Скрыть выбранные детали; выбор сохраняется')
        self.show_button.setToolTip('Показать выбранные детали')
        self.color_button.setToolTip('Цвет выбранных деталей; текстуры управляются во вкладке «Текстуры»')
        layout.addLayout(actions)
        form = QFormLayout(); form.setVerticalSpacing(10); form.setHorizontalSpacing(12)
        self.shading = QComboBox(); self.shading.setAccessibleName('Затенение выбранных деталей')
        for label, value in self.MODES: self.shading.addItem(label, value)
        self.opacity = QSpinBox(); self.opacity.setRange(0, 100); self.opacity.setSuffix(' %')
        self.opacity.setAccessibleName('Прозрачность выбранных деталей')
        self.opacity.setToolTip('0% — непрозрачная; 100% — полностью прозрачная')
        self.opacity.setKeyboardTracking(False)
        self.opacity.setFixedWidth(78)
        self.opacity_slider = QSlider(Qt.Horizontal); self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setTracking(False); self.opacity_slider.setAccessibleName('Прозрачность выбранных деталей')
        opacity_row = QHBoxLayout(); opacity_row.setSpacing(12)
        opacity_row.addWidget(self.opacity_slider, 1); opacity_row.addWidget(self.opacity)
        self.opacity_slider.valueChanged.connect(self.opacity.setValue)
        self.opacity.valueChanged.connect(self.opacity_slider.setValue)
        form.addRow('Затенение', self.shading); form.addRow('Прозрачность', opacity_row)
        layout.addLayout(form)
        self.extra_columns = QCheckBox('Табличный вид · все параметры')
        self.extra_columns.setToolTip('Полная таблица с индивидуальными режимами затенения и прозрачности')
        layout.addWidget(self.extra_columns)
        self.extra_columns.toggled.connect(self.show_columns)
        self.hide_button.clicked.connect(lambda: self.set_visibility(False))
        self.show_button.clicked.connect(lambda: self.set_visibility(True))
        self.color_button.clicked.connect(self.choose_color)
        self.shading.activated.connect(self.set_shading)
        self.opacity.valueChanged.connect(self.set_transparency)
        for table in getattr(self, 'tables', [window.ui.tbl_parts]):
            model = table.model()
            for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset, model.dataChanged):
                signal.connect(self.schedule_refresh)
        window.ui.scene_tabs.currentChanged.connect(self.schedule_refresh)
        self.show_columns(False)
        self.refresh()

    def schedule_refresh(self, *args):
        if not self.timer.isActive(): self.timer.start(0)

    def rows(self):
        table = self.window.ui.tbl_parts
        return [r for r in range(min(table.rowCount(), len(self.window.slicer_parts)))
                if not table.isRowHidden(r) and table.cellWidget(r, 1) is not None
                and table.cellWidget(r, 1).findChild(QCheckBox).isChecked()]

    def show_columns(self, visible):
        for column in (0, 3, 4): self.window.ui.tbl_parts.setColumnHidden(column, not visible)
        browser = getattr(self.window.ui, 'parts_browser', None)
        if browser is not None: browser.set_table_mode(visible)

    def refresh(self):
        browser = getattr(self.window.ui, 'parts_browser', None)
        if browser is not None: browser.refresh()
        rows = self.rows()
        for widget in (self.hide_button, self.show_button, self.color_button, self.shading, self.opacity, self.opacity_slider):
            widget.setEnabled(bool(rows) and self.window._job is None)
        if not rows:
            self.summary.setText('Не выбрано')
            self.caption.setText('Нажмите на деталь в списке или сцене')
            return
        table = self.window.ui.tbl_parts
        parts = [self.window.slicer_parts[r] for r in rows]
        visible = sum(table.cellWidget(r, 2).findChild(QCheckBox).isChecked() for r in rows)
        bodies = sum('cad_native' in p['mesh'].metadata for p in parts)
        supports = sum(len(p.get('supports', [])) for p in parts)
        self.summary.setText(f'Выбрано: {len(rows):02d}')
        self.caption.setText(f'Видно {visible}   /   CAD {bodies}   /   Групп поддержек {supports}')
        modes = {p.get('last_visible_mode', 'shaded_wire') for p in parts}
        values = {int(table.item(r, 4).text().rstrip('%')) for r in rows if table.item(r, 4)}
        with QSignalBlocker(self.shading), QSignalBlocker(self.opacity), QSignalBlocker(self.opacity_slider):
            self.shading.setCurrentIndex(self.shading.findData(next(iter(modes))) if len(modes) == 1 else -1)
            self.shading.setPlaceholderText('Разные значения')
            self.opacity.setSpecialValueText('Разные' if len(values) != 1 else '')
            self.opacity.setValue(next(iter(values)) if len(values) == 1 else 0)
            self.opacity_slider.setValue(self.opacity.value())

    def edit(self, label, change):
        if self.window._busy(): return
        rows = self.rows()
        if not rows: return
        self.window.flush_history()
        for row in rows: change(row)
        self.window.mark_dirty(); self.window.flush_history(label)
        self.schedule_refresh()

    def set_visibility(self, visible):
        self.edit('Видимость выбранных деталей', lambda row:
                  self.window.ui.tbl_parts.cellWidget(row, 2).findChild(QCheckBox).setChecked(visible))

    def set_shading(self, index):
        mode = self.shading.itemData(index)
        if mode: self.edit('Затенение выбранных деталей', lambda r: self.window._apply_part_display_mode(r, mode))

    def set_transparency(self, value):
        self.edit('Прозрачность выбранных деталей', lambda r: self.window._apply_part_transparency(r, value))

    def choose_color(self):
        rows = self.rows()
        if not rows or self.window._busy(): return
        from pyvista import Color
        color = QColor(Color(self.window._style_for(self.window.ui.tbl_parts, rows[0])['color']).hex_rgb)
        color = QColorDialog.getColor(color, self.window, 'Цвет выбранных деталей')
        if color.isValid(): self.set_color(color)

    def set_color(self, color):
        def change(row):
            part = self.window.slicer_parts[row]
            actor = self.window.ui.slicer_plotter.actors[part['actor_name']]
            actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())
            button = self.window.ui.tbl_parts.cellWidget(row, 5).findChild(QPushButton)
            button.setStyleSheet(f'background-color: {color.name()}; border: 1px solid #555;')
        self.edit('Цвет выбранных деталей', change)
        if self.window.ui.slicer_plotter: self.window.ui.slicer_plotter.render()
