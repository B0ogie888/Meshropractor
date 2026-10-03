"""Transactional texture ribbon operations, with project history and surface picking."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QDialog, QDialogButtonBox, QFileDialog,
    QListWidget, QVBoxLayout)
from PySide6.QtGui import QColor
from display_settings import DIALOG_STYLE
from texture_geometry import (KEY, add_layer, appearance, bake_colors, change_layer,
    paint, read_image, split_colors)
from texture_ribbon import COMMANDS
from texture_dialog import TextureDialog, IMAGE_FILTER


class TextureTools(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window; self.active = None; self.clipboard = None
        self.buttons = window.ui.texture_buttons
        for op, button in self.buttons.items(): button.clicked.connect(lambda checked=False, operation=op: self.trigger(operation))
        for op in ('texture', 'triangle_colors'):
            window.ui.display_buttons[op].clicked.connect(self.sync_buttons)

    def sync_buttons(self, *args):
        visible = bool(self.window.display_tools.state['texture'])
        for op in ('show', 'colors'): self.buttons[op].setChecked(visible)

    def show(self, visible=True):
        tools = self.window.display_tools
        tools.state['texture'] = bool(visible); tools.buttons['texture'].setChecked(bool(visible))
        if visible:
            tools.state['triangle_colors'] = False; tools.buttons['triangle_colors'].setChecked(False)
        tools.on_scene_changed(); self.sync_buttons()

    def notify(self, text): self.window.display_tools.notify(text)

    def base(self, row):
        from pyvista import Color
        color = self.window._style_for(self.window.ui.tbl_parts, row)['color']
        return tuple(round(c * 255) for c in Color(color).float_rgb) + (255,)

    def textures(self):
        result = []
        for row, part in enumerate(self.window.slicer_parts):
            value = appearance(part['mesh'])
            if value is None: continue
            for layer in value['layers']: result.append((row, layer))
        return result

    def choose(self, force=False):
        entries = self.textures()
        if not entries: raise ValueError('Текстур пока нет. Нажмите «Новая текстура» или «Деталь-в-текстуру».')
        if not force:
            active = next(((r, layer) for r, layer in entries if (r, layer['id']) == self.active), None)
            rows = self.window.selected_slicer_rows()
            if active is not None and (not rows or active[0] in rows): return active
            selected = [(r, layer) for r, layer in entries if r in rows]
            if len(selected) == 1: self.active = (selected[0][0], selected[0][1]['id']); return selected[0]
        dialog = QDialog(self.window); dialog.setWindowTitle('Текстуры загруженных деталей'); dialog.setStyleSheet(DIALOG_STYLE); dialog.resize(580, 350)
        layout = QVBoxLayout(dialog); items = QListWidget(); layout.addWidget(items)
        for row, layer in entries:
            count = int(layer['mask'].sum()); visible = 'видна' if layer.get('visible', True) else 'скрыта'
            items.addItem(f"{self.window.slicer_parts[row]['filename']} — {layer['name']} ({count} треугольников, {visible})")
        items.setCurrentRow(next((i for i, (r, layer) in enumerate(entries) if (r, layer['id']) == self.active), 0))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); items.itemDoubleClicked.connect(lambda item: dialog.accept())
        accepted = dialog.exec(); index = items.currentRow(); dialog.deleteLater()
        if not accepted: return None
        row, layer = entries[index]; self.active = (row, layer['id'])
        return row, layer

    def selected(self, surfaces=False):
        rows = self.window.selected_slicer_rows()
        selection = self.window.workspace_tools.selection
        if surfaces:
            rows = [r for r in rows if selection.get(r)]
            if not rows:
                self.window.workspace_tools.set_mode('cad_face' if any('cad_native' in p['mesh'].metadata for p in self.window.slicer_parts) else 'triangle')
                raise ValueError('Выделите поверхности в сцене и нажмите команду ещё раз. Доступны CAD-поверхность, плоскость, кисть и рамка.')
        if not rows: raise ValueError('Выберите детали в сцене или в списке.')
        return rows

    def apply(self, meshes, label, show=True):
        window = self.window; window.flush_history(); before = window.capture_project()
        try:
            for row, mesh in meshes.items(): window.slicer_parts[row]['mesh'] = mesh
            window.display_tools.datasets.clear()
            if show: self.show(True)
            else: window.display_tools.on_scene_changed()
            window.mark_dirty(); window.flush_history(label)
            self.notify(label + ': применено. Ctrl+Z — отменить.')
        except Exception:
            window.restore_project(before); raise

    def trigger(self, operation):
        if self.window._busy(): self.sync_buttons(); return
        try:
            if operation in ('show', 'colors'):
                self.window.flush_history()
                self.show(self.buttons[operation].isChecked())
                self.window.mark_dirty(); self.window.flush_history(COMMANDS[operation]); return
            if operation == 'select':
                chosen = self.choose(True)
                if chosen is None: return
                row, layer = chosen
                part = self.window.slicer_parts[row]
                defaults = [p for p in self.window.platforms if p.get('is_default')]
                index = next((i + 1 for i, p in enumerate(defaults) if p['name'] == part.get('platform')), 0)
                self.window.ui.scene_tabs.setCurrentIndex(index)
                for r in range(len(self.window.slicer_parts)):
                    self.window.ui.tbl_parts.cellWidget(r, 1).findChild(QCheckBox).setChecked(r == row)
                self.window.ui.tbl_parts.cellWidget(row, 2).findChild(QCheckBox).setChecked(True)
                self.window.workspace_tools.set_mode('triangle')
                self.window.workspace_tools.edit_selection(row, np.flatnonzero(layer['mask']).tolist(), 'replace')
                return
            if operation in ('new', 'paste', 'bake', 'paint_part', 'paint_faces', 'split'):
                return self.on_parts(operation)
            chosen = self.choose()
            if chosen is None: return
            row, layer = chosen; mesh = self.window.slicer_parts[row]['mesh']
            if operation == 'copy':
                self.clipboard = deepcopy(layer); self.notify('Текстура скопирована. Выберите детали или поверхности и нажмите «Вставить текстуру».'); return
            params = None; image = None; name = None; path = layer.get('path', '')
            if operation == 'edit':
                dialog = TextureDialog(layer, self.window)
                accepted = dialog.exec()
                if not accepted: dialog.deleteLater(); return
                image, name, path = dialog.image, dialog.name.text().strip() or 'Текстура', dialog.path
                candidate = dialog.parameters()
                if candidate['projection'] not in ('Атлас цветов', 'Исходная UV'): params = candidate
                dialog.deleteLater()
            elif operation == 'update':
                if not path:
                    if layer['params']['projection'] == 'Атлас цветов':
                        updated, identifier = bake_colors(mesh, self.base(row)); self.active = (row, identifier)
                        self.apply({row: updated}, COMMANDS[operation]); return
                    path, _ = QFileDialog.getOpenFileName(self.window, 'Источник изображения текстуры', '', IMAGE_FILTER)
                    if not path: return
                image = read_image(path)
            elif operation == 'clear':
                if not self.window.workspace_tools.selection.get(row): raise ValueError('Выделите треугольники на детали выбранной текстуры.')
            updated = change_layer(mesh, layer['id'], 'edit' if operation == 'update' else operation,
                ids=self.window.workspace_tools.selection.get(row), image=image, params=params, name=name)
            if operation in ('edit', 'update'):
                next(item for item in updated.metadata[KEY]['layers'] if item['id'] == layer['id'])['path'] = path
            self.apply({row: updated}, COMMANDS[operation], show=operation != 'invert')
            if operation == 'delete': self.active = None
        except Exception as exc: self.notify(f'{COMMANDS[operation]}: {exc}')

    def on_parts(self, operation):
        rows = self.selected(surfaces=operation == 'paint_faces')
        meshes = {}; selection = self.window.workspace_tools.selection
        if operation in ('paint_part', 'paint_faces'):
            color = QColorDialog.getColor(QColor(*self.base(rows[0])[:3]), self.window, COMMANDS[operation])
            if not color.isValid(): return
            rgba = [color.red(), color.green(), color.blue(), 255]
            for row in rows: meshes[row] = paint(self.window.slicer_parts[row]['mesh'], rgba, selection.get(row) if operation == 'paint_faces' else None, self.base(row))
        elif operation == 'bake':
            for row in rows:
                meshes[row], identifier = bake_colors(self.window.slicer_parts[row]['mesh'], self.base(row)); self.active = (row, identifier)
        elif operation in ('new', 'paste'):
            if operation == 'new':
                path, _ = QFileDialog.getOpenFileName(self.window, 'Новая текстура', '', IMAGE_FILTER)
                if not path: return
                layer = dict(image=read_image(path), path=path, name=Path(path).stem, params={'projection': 'По граням'})
                dialog = TextureDialog(layer, self.window)
                if not dialog.exec(): dialog.deleteLater(); return
                layer.update(image=dialog.image, path=dialog.path, name=dialog.name.text().strip() or 'Текстура', params=dialog.parameters()); dialog.deleteLater()
            else:
                if self.clipboard is None: raise ValueError('Сначала скопируйте текстуру.')
                layer = self.clipboard
            params = layer['params']
            if params.get('projection') in ('Исходная UV', 'Атлас цветов'): params = {'projection': 'По граням'}
            for row in rows:
                meshes[row], identifier = add_layer(self.window.slicer_parts[row]['mesh'], layer['image'], params,
                    ids=selection.get(row), name=layer['name'], path=layer.get('path', ''), base=self.base(row)); self.active = (row, identifier)
        elif operation == 'split':
            window = self.window; window.flush_history(); before = window.capture_project(); state = deepcopy(before)
            new_parts = []
            for row, part in enumerate(state.parts):
                if row not in rows: new_parts.append(part); continue
                regions = split_colors(part['mesh'], part.get('supports', []), self.base(row))
                for index, region in enumerate(regions, 1):
                    record = dict(part, mesh=region['mesh'], supports=region['supports'],
                        filename=f"{Path(part['filename']).stem} — цвет {index}.stl")
                    record['style'] = dict(part.get('style', {}), color=[c / 255 for c in region['color'][:3]])
                    new_parts.append(record)
            state.parts = new_parts
            cameras = deepcopy(window.ui.slicer_plotter.camera_position)
            try:
                window.restore_project(state); window.ui.slicer_plotter.camera_position = cameras
                self.show(True); window.mark_dirty(); window.flush_history(COMMANDS[operation]); self.active = None
            except Exception: window.restore_project(before); raise
            self.notify('Деталь разделена по цветам. Результат — отдельные сетки; срезы могут быть открыты. Поддержки сохранены у владельцев.'); return
        self.apply(meshes, COMMANDS[operation])
