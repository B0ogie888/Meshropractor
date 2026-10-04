"""Selection inspector backed by committed models and existing tools."""
from collections import OrderedDict
from pathlib import Path
import weakref

import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGridLayout, QLabel, QPushButton, QToolButton, QSizePolicy)

from cad_state import cad_status
from ribbon_layout import asset_icon


class PartInspector(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setObjectName('PartInspector')
        self._geometry = OrderedDict()
        self.timer = QTimer(self); self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 18, 18, 18); layout.setSpacing(16)
        heading = QHBoxLayout()
        self.caption = QLabel('КОНТЕКСТ / ДЕТАЛЬ'); self.caption.setObjectName('PartsEyebrow')
        heading.addWidget(self.caption); heading.addStretch()
        self.close_button = QToolButton(); self.close_button.setText('›')
        self.close_button.setAccessibleName('Скрыть свойства деталей')
        self.close_button.setToolTip('Скрыть свойства деталей')
        self.close_button.clicked.connect(lambda: window.ui.set_inspector_visible(False))
        heading.addWidget(self.close_button); layout.addLayout(heading)
        self.title = QLabel(); self.title.setObjectName('InspectorTitle'); self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.PlainText)
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.title)
        self.empty = QLabel('Выберите деталь в списке или на сцене.\nCtrl — выбрать несколько деталей.')
        self.empty.setObjectName('PartsHint'); self.empty.setWordWrap(True); layout.addWidget(self.empty)
        self.details = QWidget(); form = QFormLayout(self.details)
        form.setContentsMargins(0, 0, 0, 0); form.setHorizontalSpacing(14); form.setVerticalSpacing(12)
        self.values = {}
        for key, label in (('type', 'Тип'), ('geometry', 'Геометрия'), ('units', 'Единицы'),
                           ('platform', 'Платформа'), ('triangles', 'Треугольники'), ('vertices', 'Вершины'),
                           ('bodies', 'Тела CAD'), ('faces', 'Поверхности CAD'), ('supports', 'Группы поддержек')):
            name = QLabel(label); name.setObjectName('PartsHint')
            value = QLabel('—'); value.setWordWrap(True); value.setTextFormat(Qt.PlainText)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            form.addRow(name, value); self.values[key] = value
        self.values['units'].setToolTip('Рабочие координаты в миллиметрах. STEP приводится к мм при импорте. '
            'STL не хранит единицы: его координаты интерпретируются как мм; для пересчёта используйте масштабирование.')
        layout.addWidget(self.details)
        self.dimensions = QWidget(); dims = QVBoxLayout(self.dimensions); dims.setContentsMargins(0, 0, 0, 0)
        self.dimensions_title = QLabel('ГАБАРИТЫ / XYZ'); self.dimensions_title.setObjectName('PartsEyebrow')
        dims.addWidget(self.dimensions_title)
        axes = QHBoxLayout(); self.axis_values = []
        for axis in 'XYZ':
            cell = QWidget(); col = QVBoxLayout(cell); col.setContentsMargins(8, 10, 8, 10)
            cell.setObjectName('InspectorDimension')
            name = QLabel(axis); name.setObjectName('PartsHint'); col.addWidget(name)
            value = QLabel('—'); value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setWordWrap(True); value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            col.addWidget(value); self.axis_values.append(value); axes.addWidget(cell, 1)
        dims.addLayout(axes)
        hint = QLabel('мм · по текущей сетке, без поддержек')
        hint.setObjectName('PartsHint'); hint.setWordWrap(True); dims.addWidget(hint); layout.addWidget(self.dimensions)
        self.actions = QWidget(); grid = QGridLayout(self.actions); grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(8)
        self.buttons = {}
        for i, (key, title, icon) in enumerate((('Перемещать', 'Переместить', 'move'),
                ('Вращать', 'Вращать', 'rotate'), ('Масштабировать', 'Масштаб', 'scale'),
                ('Отзеркалить', 'Отзеркалить', 'mirror'))):
            button = QPushButton(title); button.setIcon(asset_icon('placement', icon)); button.setIconSize(QSize(20, 20))
            button.setMinimumHeight(36); button.clicked.connect(lambda checked=False, operation=key: self.run(operation))
            grid.addWidget(button, i//2, i%2); self.buttons[key] = button
        layout.addWidget(self.actions)
        self.cad_button = QPushButton('CAD: тела и поверхности'); self.cad_button.setIcon(asset_icon('tools', 'cad'))
        self.cad_button.clicked.connect(lambda: window.cad_tools.open()); layout.addWidget(self.cad_button)
        self.save_button = QPushButton('Сохранить выбранные…')
        self.save_button.setIcon(asset_icon('main', 'Сохранить выбранные детали как'))
        self.save_button.clicked.connect(lambda: window.save_selected_slicer_parts()); layout.addWidget(self.save_button)
        layout.addStretch()
        self.import_button = QPushButton('+  Добавить модель'); self.import_button.setProperty('primary', True)
        self.import_button.setMinimumHeight(38)
        self.import_button.clicked.connect(lambda: window.import_slicer_part()); layout.addWidget(self.import_button)
        self.setStyleSheet('''#PartInspector {background: #fafbf8; border: 0;}
            #InspectorTitle {font-size: 18px; font-weight: 600; color: #252b2b;}
            #PartsEyebrow {font-family: "Consolas", "DejaVu Sans Mono"; font-size: 11px; color: #69796d;}
            #PartsHint {color: #69766d; font-size: 11px;}
            #InspectorDimension {background: #f0f3ed; border: 0; border-radius: 5px;}
            QPushButton {padding: 8px 6px; border-radius: 4px;}
            QToolButton {background: transparent; border: 0; padding: 0 5px; font-size: 20px;}''')
        model = window.ui.tbl_parts.model()
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset, model.dataChanged):
            signal.connect(self.schedule_refresh)
        window.ui.scene_tabs.currentChanged.connect(self.schedule_refresh)
        self.refresh()

    def schedule_refresh(self, *args):
        if not self.timer.isActive(): self.timer.start(0)

    def invalidate(self, row):
        if 0 <= row < len(self.window.slicer_parts):
            self._geometry.pop(id(self.window.slicer_parts[row]['mesh']), None)
        self.schedule_refresh()

    def geometry_info(self, mesh):
        # CAD validation hashes the proxy and BREP: cache it per immutable mesh
        # snapshot, not on every selection, visibility change or camera frame.
        key = id(mesh); cached = self._geometry.get(key)
        if cached is not None and cached[0]() is mesh:
            self._geometry.move_to_end(key); return cached[1]
        state = cad_status(mesh)
        cad = mesh.metadata.get('cad_native', {}) if state == 'native' else {}
        info = dict(status=state, bodies=len(cad.get('bodies', [])), faces=len(cad.get('face_info', [])))
        self._geometry[key] = (weakref.ref(mesh), info)
        while len(self._geometry) > 64: self._geometry.popitem(last=False)
        return info

    def refresh(self):
        window = self.window
        # Table updates can arrive while project restoration is building rows.
        rows = window.ui.part_controls.rows()
        parts = [window.slicer_parts[r] for r in rows]
        selected = bool(parts)
        busy = window._job is not None or any(getattr(window, name, None) is not None
            for name in ('_transform_session', '_placement_session', '_repair_session','_duplicate_session'))
        for widget in (self.details, self.dimensions, self.actions, self.save_button): widget.setVisible(selected)
        self.empty.setVisible(not selected)
        self.import_button.setEnabled(not busy)
        self.save_button.setEnabled(selected and not busy)
        for button in self.buttons.values(): button.setEnabled(selected and not busy)
        self.cad_button.hide()
        if not parts:
            self.title.setText('Ничего не выбрано'); self.caption.setText('КОНТЕКСТ / ДЕТАЛЬ')
            for value in self.values.values(): value.setText('—')
            for value in self.axis_values: value.setText('—')
            return
        self.title.setText(parts[0]['filename'] if len(parts)==1 else f'Выбрано деталей: {len(parts)}')
        self.caption.setText('КОНТЕКСТ / ДЕТАЛЬ' if len(parts)==1 else 'КОНТЕКСТ / ГРУППА')
        self.title.setToolTip('\n'.join(p['filename'] for p in parts[:20]))
        meshes = [p['mesh'] for p in parts]
        info = [self.geometry_info(m) for m in meshes]
        native = sum(i['status']=='native' for i in info)
        modified = sum(i['status']=='modified' for i in info)
        mesh_count = len(parts)-native
        if native == len(parts): kind = 'CAD / BREP'
        elif native: kind = f'CAD: {native} · сетки: {mesh_count}'
        else: kind = 'STL / сетка' if all(Path(p['filename']).suffix.lower()=='.stl' for p in parts) else 'Сетка'
        geometry = 'BREP + триангуляция' if native==len(parts) else 'Треугольная сетка' if not native else 'CAD и треугольные сетки'
        if modified: geometry += '\nСвязь с CAD утрачена: ' + str(modified)
        platforms = {p.get('platform') or 'Модельная сцена' for p in parts}
        values = dict(type=kind, geometry=geometry, units='мм', platform=next(iter(platforms)) if len(platforms)==1 else 'Несколько платформ',
            triangles=sum(len(m.faces) for m in meshes), vertices=sum(len(m.vertices) for m in meshes),
            bodies=sum(i['bodies'] for i in info) if native else '—', faces=sum(i['faces'] for i in info) if native else '—',
            supports=sum(len(p.get('supports', [])) for p in parts))
        for name, value in values.items(): self.values[name].setText(f'{value:,}'.replace(',', ' ') if isinstance(value, int) else value)
        bounds = np.asarray([m.bounds for m in meshes])
        extents = bounds[:, 1].max(axis=0)-bounds[:, 0].min(axis=0)
        self.dimensions_title.setText('ГАБАРИТЫ / XYZ' if len(parts)==1 else 'ОБЩИЙ ГАБАРИТ / XYZ')
        for label, value in zip(self.axis_values, extents): label.setText(f'{value:.3f}')
        self.cad_button.setVisible(bool(native or modified)); self.cad_button.setEnabled(not busy)

    def run(self, operation):
        self.window.run_slicer_tool(operation)
        self.refresh()
