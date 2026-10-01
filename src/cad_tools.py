"""CAD bodies share the STL workspace while retaining their exact BREP source."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton

from background_tasks import FunctionWorker
from cad_state import cad_status, require_native, strip_native


def load_native_parts(path, linear, angle, *, split=True, progress=None, cancelled=None):
    from cad_import import load_step
    from cad_geometry import split_bodies
    mesh = load_step(path, linear, angle, native=True, progress=progress, cancelled=cancelled)
    if split and len(require_native(mesh)['bodies']) > 200:
        raise ValueError('В STEP более 200 тел. Снимите «Загрузить тела отдельными деталями» и импортируйте одной CAD-деталью.')
    return split_bodies(mesh, progress=progress, cancelled=cancelled) if split else [mesh]


def prepare_cad_changes(records, operation, precision=None, *, progress=None, cancelled=None):
    from cad_geometry import retessellate, split_bodies
    from cad_supports import rebind_supports
    progress = progress or (lambda value: None)
    cancelled = cancelled or (lambda: False)
    result = []
    for record in records:
        if cancelled(): raise InterruptedError('Операция CAD отменена.')
        progress(record['name'] + ': ' + ('триангуляция CAD…' if operation == 'quality' else 'разделение тел…'))
        source = record['mesh']
        if operation == 'quality':
            mesh = retessellate(source, *precision, progress=progress, cancelled=cancelled)
            groups = rebind_supports(record.get('supports', []), source, mesh)
            result.append(dict(target=record, meshes=[mesh], supports=groups))
        else:
            if record.get('supports'):
                raise ValueError('Разделите CAD на тела до создания поддержек. Существующие поддержки сохранены.')
            result.append(dict(target=record, meshes=split_bodies(source, progress=progress, cancelled=cancelled), supports=[]))
    return result


class CADTools(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.dialog = None
        self.last_surface = None
        window.ui.ribbon_btns['CAD / STEP'].clicked.connect(self.open)
        window.ui.btn_cad_tools.clicked.connect(self.open)
        window.ui.stack.currentChanged.connect(self.refresh)

    def targets(self):
        window = self.window
        if window.ui.stack.currentWidget() is window.ui.page_slicer:
            return [dict(scope='part', key=row, mesh=window.slicer_parts[row]['mesh'],
                         name=window.slicer_parts[row]['filename'], supports=window.slicer_parts[row].get('supports', []))
                    for row in window.selected_slicer_rows()]
        return [dict(scope='model', key=key, mesh=record['mesh'], name=record['name'], supports=[])
                for key, record in window.scene_models.items() if record['kind'] == 'CAD']

    def message(self, text):
        self.window.log(text)
        if self.dialog is not None: self.dialog.set_info(text)

    def open(self):
        if self.window._busy(): return
        from cad_dialog import CADToolsDialog
        if self.dialog is None:
            self.dialog = CADToolsDialog(self.window)
            self.dialog.retessellate_requested.connect(self.quality)
            self.dialog.split_requested.connect(self.split)
            self.dialog.export_requested.connect(self.export)
            self.dialog.convert_requested.connect(self.convert)
            self.properties_button = QPushButton('Точные свойства CAD')
            self.properties_button.clicked.connect(self.properties)
            self.dialog.layout().insertWidget(3, self.properties_button)
        self.refresh()
        self.dialog.show()
        self.dialog.raise_()

    def refresh(self):
        if self.dialog is None: return
        targets = self.targets()
        statuses = [cad_status(record['mesh']) for record in targets]
        native = bool(targets) and all(state == 'native' for state in statuses)
        split = native and all(record['scope'] == 'part' and not record['supports'] for record in targets)
        split = split and any(len(require_native(record['mesh'])['bodies']) > 1 for record in targets)
        self.dialog.update_selection(f'Выбрано деталей: {len(targets)}. CAD (BREP): {statuses.count("native")}.', native, split)
        self.dialog.convert.setEnabled(any(state != 'mesh' for state in statuses))
        self.properties_button.setEnabled(native)
        lines = []
        for record, state in zip(targets, statuses):
            mesh = record['mesh']
            lines.append(record['name'])
            if state == 'native':
                cad = require_native(mesh)
                lines += [f'  CAD-тел: {len(cad["bodies"])}; поверхностей: {len(cad["face_info"])}; треугольников: {len(mesh.faces)}.',
                          '  BREP сохранён. Поддержки и расчёты используют сетку заданной точности.']
            elif state == 'modified':
                lines.append('  Сетка изменена: исходный BREP больше не соответствует детали. Экспорт STEP недоступен; Ctrl+Z восстанавливает CAD.')
            else:
                lines.append('  Треугольная сетка. Для BREP загрузите исходный STEP в режиме CAD.')
        self.dialog.set_info('\n'.join(lines) if lines else 'Выделите детали в таблице или загрузите номинальную CAD-модель в предеформацию.')

    def surface_info(self, row, triangle):
        mesh = self.window.slicer_parts[row]['mesh']
        cad = require_native(mesh)
        face_id = int(cad['face_ids'][triangle])
        self.last_surface = (mesh, face_id)
        info = cad['face_info'][face_id]
        text = f'CAD-поверхность {face_id + 1}: {info.get("type", "поверхность")}. Выделена целиком; доступны поддержки по выбранной поверхности.'
        self.window.ui.status_label.setText(text)
        if self.dialog is not None and self.dialog.isVisible():
            self.refresh()
            self.dialog.details.appendPlainText('\n' + text)

    def _native_targets(self):
        targets = self.targets()
        if not targets: raise ValueError('Сначала выберите CAD-детали.')
        for record in targets: require_native(record['mesh'])
        return targets

    def properties(self):
        if self.window._busy(): return
        try:
            targets = self._native_targets()
            selected = self.last_surface
            def calculate(*, progress, cancelled):
                from cad_geometry import native_info
                return [(record, native_info(record['mesh'], progress=progress, cancelled=cancelled)) for record in targets]
            def ready(results):
                lines = []
                for record, info in results:
                    volume = info['volume_mm3']
                    lines += [record['name'], f'  CAD-тел: {info["body_count"]}; поверхностей: {info["face_count"]}.',
                              f'  Площадь: {info["area_mm2"]:.6g} мм².',
                              f'  Объём: {volume:.6g} мм³.' if volume is not None else '  Объём не определён: незамкнутые CAD-поверхности.',
                              '  Геометрия BREP корректна.' if info['is_valid'] else '  Проверка BREP обнаружила ошибки геометрии.']
                    if selected is not None and selected[0] is record['mesh']:
                        face = info['face_info'][selected[1]]
                        lines.append(f'  Выбранная поверхность {selected[1] + 1}: {face.get("type", "поверхность")}')
                        for key, label, unit in (('area_mm2', 'Площадь', 'мм²'), ('radius_mm', 'Радиус', 'мм')):
                            if face.get(key) is not None: lines.append(f'    {label}: {face[key]:.6g} {unit}')
                self.dialog.set_info('\n'.join(lines))
            self._start(FunctionWorker(calculate, with_progress=True), ready)
        except (ValueError, RuntimeError) as exc: self.message(str(exc))

    def _start(self, worker, callback):
        if self.dialog is not None:
            worker.progress.connect(self.dialog.set_info)
            worker.error.connect(self.dialog.set_info)
            # Keep the result/error text visible; eligibility is refreshed on next selection.
        return self.window.start_job(worker, callback)

    def quality(self):
        if self.window._busy(): return
        try:
            targets = self._native_targets()
            from import_dialog import StepImportDialog
            dialog = StepImportDialog(self.window, allow_split=False)
            dialog.setWindowTitle('Изменить качество сетки CAD')
            dialog.native.setChecked(True); dialog.native.setEnabled(False)
            source = targets[0]['mesh'].metadata
            dialog.linear.setValue(source.get('linear_deflection_mm', .05))
            dialog.angle.setValue(np.degrees(source.get('angular_deflection_rad', .25)))
            accepted = dialog.exec()
            precision = dialog.values()
            dialog.deleteLater()
            if not accepted: return
            self._start(FunctionWorker(prepare_cad_changes, targets, 'quality', precision, with_progress=True),
                        lambda items: self.apply_changes(items, 'Изменить качество CAD'))
        except (ValueError, RuntimeError) as exc: self.message(str(exc))

    def split(self):
        if self.window._busy(): return
        try:
            targets = self._native_targets()
            if any(record['scope'] != 'part' for record in targets):
                raise ValueError('Разделение CAD на отдельные детали доступно в слайсере.')
            self._start(FunctionWorker(prepare_cad_changes, targets, 'split', with_progress=True),
                        lambda items: self.apply_changes(items, 'Разделить CAD на тела'))
        except (ValueError, RuntimeError) as exc: self.message(str(exc))

    def convert(self):
        if self.window._busy(): return
        targets = [record for record in self.targets() if cad_status(record['mesh']) != 'mesh']
        if not targets: return
        if QMessageBox.question(self.window, 'Преобразовать CAD в сетку',
                'Удалить BREP у выбранных деталей и оставить текущую сетку для редактирования? '
                'После этого сохранение в STEP недоступно. Действие можно отменить.',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes: return
        items = []
        for record in targets:
            groups = deepcopy(record['supports'])
            for group in groups: group.pop('cad_binding', None)
            items.append(dict(target=record, meshes=[strip_native(record['mesh'])], supports=groups))
        self.apply_changes(items, 'Преобразовать CAD в сетку')

    def export(self, path=None, records=None):
        if self.window._busy(): return
        try:
            targets = self._native_targets() if records is None else records
            if not targets: raise ValueError('Нет выбранных CAD-деталей.')
            for record in targets: require_native(record['mesh'])
            if any(record.get('supports') for record in targets):
                if QMessageBox.question(self.window, 'Экспорт CAD в STEP',
                    'STEP сохранит точные CAD-тела. Треугольные поддержки остаются в проекте; '
                    'для детали вместе с поддержками используйте STL. Сохранить CAD-тела?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) != QMessageBox.Yes: return
            if path is None:
                path, _ = QFileDialog.getSaveFileName(self.window, 'Сохранить CAD-тела', 'CAD_parts.step', 'STEP (*.step *.stp)')
            if not path: return
            if Path(path).suffix.lower() not in ('.step', '.stp'): path += '.step'
            from cad_geometry import export_step
            self._start(FunctionWorker(export_step, [record['mesh'] for record in targets], path, with_progress=True),
                        lambda report: self.message(f'CAD-тела сохранены в STEP: {path}'))
        except (ValueError, RuntimeError) as exc: self.message(str(exc))

    def append_import(self, meshes, path, platform):
        window = self.window
        before = window.capture_project()
        try:
            window._slicer_batch = True
            for index, mesh in enumerate(meshes):
                name = Path(path).name if len(meshes) == 1 else f'{Path(path).stem} — {mesh.metadata.get("cad_body_name", f"тело {index + 1}")}.step'
                style = dict(last_visible_mode='shaded', color=mesh.metadata.get('cad_color') or '#d3d3d3')
                window._append_slicer_part(mesh, name, platform, style)
                window._apply_part_display_mode(window.ui.tbl_parts.rowCount() - 1, 'shaded')
            window.update_info_combobox()
            window.refresh_scene_visibility()
            window.update_parts_table_filter(window.ui.scene_tabs.currentIndex())
            window.mark_dirty()
            window.log(f'Загружено CAD-деталей: {len(meshes)}. BREP сохранён; для поверхностей используйте кнопку CAD над сценой.')
        except Exception:
            window.restore_project(before)
            raise
        finally: window._slicer_batch = False

    def apply_changes(self, items, label):
        window = self.window
        for item in items:
            record = item['target']
            current = window.slicer_parts[record['key']] if record['scope'] == 'part' else window.scene_models[record['key']]
            if current['mesh'] is not record['mesh']:
                raise ValueError('Деталь изменилась во время расчёта CAD. Повторите операцию.')
        window.flush_history()
        before = window.capture_project()
        state = deepcopy(before)
        cameras = [deepcopy(plotter.camera_position) if plotter is not None else None
                   for plotter in (window.ui.plotter, window.ui.slicer_plotter)]
        tab = window.ui.scene_tabs.currentIndex()
        by_row = {item['target']['key']: item for item in items if item['target']['scope'] == 'part'}
        parts = []
        for row, record in enumerate(state.parts):
            item = by_row.get(row)
            if item is None: parts.append(record); continue
            for index, mesh in enumerate(item['meshes']):
                replacement = dict(record, mesh=mesh, supports=deepcopy(item['supports']))
                if len(item['meshes']) > 1:
                    replacement['filename'] = f'{Path(record["filename"]).stem} — {mesh.metadata.get("cad_body_name", f"тело {index + 1}")}.step'
                parts.append(replacement)
        state.parts = parts
        for item in items:
            if item['target']['scope'] != 'model': continue
            record = next(record for record in state.models if record['key'] == item['target']['key'])
            record['mesh'] = item['meshes'][0]
            # Existing comparison results were computed on a different proxy.
            state.models = [record for record in state.models if record['kind'] != 'Heatmap']
            state.active_heatmap = None
            state.callouts = []
        try:
            window.restore_project(state)
        except Exception:
            window.restore_project(before)
            raise
        finally:
            window.ui.scene_tabs.setCurrentIndex(min(tab, window.ui.scene_tabs.count() - 1))
            for plotter, camera in zip((window.ui.plotter, window.ui.slicer_plotter), cameras):
                if plotter is not None and camera is not None: plotter.camera_position = camera; plotter.render()
        window.mark_dirty()
        window.flush_history(label)
        for item in items:
            for group in item['supports']:
                notice = group.get('cad_binding', {}).get('notice')
                if notice: window.log('[i] ' + notice)
        self.refresh()
        window.log(label + ': готово. Ctrl+Z — отменить.')
