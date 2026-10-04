"""Compact pre-deformation model lists; original tables remain the source of truth."""
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QStackedWidget, QLabel, QPushButton,
                               QCheckBox, QSizePolicy, QColorDialog)
from PySide6.QtGui import QColor
from parts_view import PartListModel, PartListView
from part_controls import PartControls


class PredefListView(PartListView):
    def select_entry(self, index, modifiers=Qt.NoModifier, toggle=False):
        browser = self.browser; window = browser.window
        if window._busy() or not 0 <= index < len(self.model().entries): return
        if modifiers & Qt.ShiftModifier and self.anchor is not None:
            lo, hi = sorted((min(self.anchor, len(self.model().entries)-1), index))
            targets = {e['key'] for e in self.model().entries[lo:hi+1]}; operation = 'add'
        else:
            targets = {self.model().entries[index]['key']}
            operation = 'toggle' if toggle or modifiers & Qt.ControlModifier else 'replace'
            self.anchor = index
        browser.select(targets, operation)
        row = self.model().entries[index]['row']
        if browser.table is window.ui.tbl_heat: window.select_heatmap(row, 6)
        elif browser.table is window.ui.tbl_res: window.select_result(row, 6)


class PredefBrowser(QWidget):
    def __init__(self, window, table, empty_text):
        super().__init__()
        self.window = window; self.table = table
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)
        self.model = PartListModel(self); self.list = PredefListView(self); self.list.setModel(self.model)
        self.empty = QLabel(empty_text); self.empty.setObjectName('PartsHint'); self.empty.setWordWrap(True)
        self.stack = QStackedWidget(); self.stack.addWidget(self.list); self.stack.addWidget(self.empty)
        table.setMinimumHeight(100); self.stack.addWidget(table); layout.addWidget(self.stack)
        self.refresh()

    def select(self, keys, operation):
        if self.window._busy(): return
        for table in (self.window.ui.tbl_cad, self.window.ui.tbl_scan, self.window.ui.tbl_heat, self.window.ui.tbl_res):
            for row in range(table.rowCount()):
                key = table.item(row, 0).data(Qt.UserRole)
                check = table.cellWidget(row, 1).findChild(QCheckBox)
                if operation == 'replace': check.setChecked(key in keys)
                elif key in keys: check.setChecked(not check.isChecked() if operation == 'toggle' else operation == 'add')
        self.window.mark_dirty()

    def select_all(self): self.select({e['key'] for e in self.model.entries}, 'add')
    def select_none(self): self.select({e['key'] for e in self.model.entries}, 'subtract')

    def set_table_mode(self, visible):
        self.stack.setCurrentWidget(self.table if visible else self.list); self.refresh()

    def refresh(self):
        from pyvista import Color
        entries = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None or self.table.cellWidget(row, 5) is None or self.table.item(row, 6) is None: continue
            key = item.data(Qt.UserRole); record = self.window.scene_models.get(key)
            if record is None: continue
            style = self.window._style_for(self.table, row, key)
            mesh = record['mesh']
            kind = 'BREP' if 'cad_native' in mesh.metadata else 'СЕТКА'
            entries.append(dict(row=row, key=key, name=self.table.item(row, 6).text(),
                detail=kind + f'  /  {len(mesh.faces):,} треуг.'.replace(',', ' '),
                selected=style['is_selected'], visible=style['is_visible'], color=Color(style['color']).hex_rgb))
        self.model.replace(entries)
        table_mode = self.stack.currentWidget() is self.table
        if not table_mode: self.stack.setCurrentWidget(self.list if entries else self.empty)
        self.stack.setFixedHeight(170 if table_mode else min(196, len(entries)*64+4) if entries else 36)
        self.list.setEnabled(self.window._job is None)


class PredefControls(PartControls):
    def __init__(self, window, browsers):
        self.browsers = browsers
        self.tables = [b.table for b in browsers]
        super().__init__(window)

    def rows(self):
        selected = []
        for table in self.tables:
            for row in range(table.rowCount()):
                item = table.item(row, 0); cell = table.cellWidget(row, 1)
                if item and cell and table.cellWidget(row, 5) and cell.findChild(QCheckBox).isChecked():
                    key = item.data(Qt.UserRole)
                    if key in self.window.scene_models: selected.append((table, row, key))
        return selected

    def refresh(self):
        for browser in self.browsers: browser.refresh()
        targets = self.rows()
        for widget in (self.hide_button, self.show_button, self.color_button, self.shading, self.opacity, self.opacity_slider):
            widget.setEnabled(bool(targets) and self.window._job is None)
        self.summary.setText(f'Выбрано: {len(targets):02d}' if targets else 'Не выбрано')
        if not targets:
            self.caption.setText('Выберите CAD, скан, карту или результат'); return
        styles = [self.window._style_for(t, r, k) for t, r, k in targets]
        self.caption.setText(f"Видно {sum(s['is_visible'] for s in styles)}  /  CAD, сканы и результаты")
        modes = {s.get('last_visible_mode', 'triangles') for s in styles}
        values = {s.get('transparency', 0) for s in styles}
        with QSignalBlocker(self.shading), QSignalBlocker(self.opacity), QSignalBlocker(self.opacity_slider):
            self.shading.setCurrentIndex(self.shading.findData(next(iter(modes))) if len(modes)==1 else -1)
            self.shading.setPlaceholderText('Разные значения')
            self.opacity.setSpecialValueText('Разные' if len(values)!=1 else '')
            self.opacity.setValue(next(iter(values)) if len(values)==1 else 0)
            self.opacity_slider.setValue(self.opacity.value())

    def show_columns(self, visible):
        for browser in self.browsers: browser.set_table_mode(visible)

    def set_visibility(self, visible):
        self.edit('Видимость моделей', lambda target: target[0].cellWidget(target[1], 2).findChild(QCheckBox).setChecked(visible))

    def set_shading(self, index):
        mode = self.shading.itemData(index)
        if mode: self.edit('Затенение моделей', lambda target: self.window._apply_def_display_mode(*target, mode))

    def set_transparency(self, value):
        self.edit('Прозрачность моделей', lambda target: self.window._apply_def_transparency(*target, value))

    def choose_color(self):
        targets = self.rows()
        if not targets or self.window._busy(): return
        color = QColor(self.window.ui.mesh_colors.get(targets[0][2], '#d3d3d3'))
        chosen = QColorDialog.getColor(color, self.window, 'Цвет выбранных моделей')
        if chosen.isValid(): self.set_color(chosen)

    def set_color(self, color):
        def change(target):
            table, row, key = target
            self.window.ui.mesh_colors[key] = color.name()
            actor = self.window.actors.get(key)
            if actor: actor.GetProperty().SetColor(color.redF(), color.greenF(), color.blueF())
            table.cellWidget(row, 5).findChild(QPushButton).setStyleSheet(f'background: {color.name()}; border: 1px solid #777;')
        self.edit('Цвет моделей', change)
        if self.window.ui.plotter: self.window.ui.plotter.render()
