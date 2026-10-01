"""Placement commands, asynchronous matrix previews and direct scene dragging."""
from copy import deepcopy
import numpy as np
import trimesh
from PySide6.QtCore import QObject, QEvent, Qt, QTimer
from PySide6.QtWidgets import QCheckBox

from background_tasks import FunctionWorker
from placement_dialog import PlacementDialog, PLATFORM_OPERATIONS
from placement_ribbon import PLACEMENT_COMMANDS


BASIC = {'move': 'Перемещать', 'rotate': 'Вращать', 'scale': 'Масштабировать', 'mirror': 'Отзеркалить'}


def floor_matrix(matrix, mesh):
    result = np.asarray(matrix).copy()
    result[2, 3] -= np.min(mesh.vertices[mesh.referenced_vertices] @ result[2, :3] + result[2, 3])
    return result


def calculate_placement(records, operation, parameters, all_parts=(), surface=None,
                        progress=lambda message: None, cancelled=lambda: False):
    from part_supports import combined_mesh
    from placement_geometry import (orientation_candidates, optimize_orientation,
                                    minimum_oriented_bounds, fit_platform, pack_meshes)
    if cancelled(): raise InterruptedError('Размещение отменено.')
    matrices = [np.eye(4) for _ in records]
    reports, variants = [], []
    meshes = []
    for record in records:
        if cancelled(): raise InterruptedError('Размещение отменено.')
        meshes.append(combined_mesh(record))
    if operation in PLATFORM_OPERATIONS:
        platform = parameters.get('platform')
        if not platform: raise ValueError('Выберите платформу для размещения.')
        rows = {record['row'] for record in records}
        obstacles = []
        for record in all_parts:
            if cancelled(): raise InterruptedError('Размещение отменено.')
            if record['row'] not in rows and record.get('platform') == platform['name']:
                obstacles.append(combined_mesh(record))
        kwargs = {key: parameters[key] for key in ('margin_mm', 'clearance_mm', 'allow_rotation')}
        kwargs.update(obstacles=obstacles, progress=progress, cancelled=cancelled)
        if operation == 'fit_platform': matrices, report = fit_platform(meshes, platform, **kwargs)
        else:
            matrices, report = pack_meshes(meshes, platform, dimensions=3 if operation == 'pack_3d' else 2,
                                          gap_mm=parameters['gap_mm'], **kwargs)
        reports.append(report)
    elif operation == 'sort_by_shape':
        from placement_shapes import transfer_orientations
        matrices, report = transfer_orientations([record['mesh'] for record in records],
            reference_index=parameters['reference_index'], tolerance_mm=parameters['tolerance_mm'],
            progress=progress, cancelled=cancelled)
        reports.append(report)
        if parameters['drop']:
            matched = {entry['index'] for entry in report['matches'] if entry['matched']}
            matrices = [floor_matrix(matrix, mesh) if i in matched else matrix
                        for i, (matrix, mesh) in enumerate(zip(matrices, meshes))]
    elif operation == 'top_bottom':
        if surface is None: raise ValueError('Сначала выберите поверхность в сцене.')
        normal = records[0]['mesh'].face_normals[surface]
        if np.linalg.norm(normal) < 1e-12: raise ValueError('У вырожденного треугольника нет направления. Выберите другую поверхность.')
        matrix = trimesh.geometry.align_vectors(normal, [0, 0, -1 if parameters['side'] == 0 else 1])
        center = meshes[0].bounds.mean(axis=0)
        matrix[:3, 3] = center - matrix[:3, :3] @ center
        matrices = [floor_matrix(matrix, meshes[0]) if parameters['drop'] else matrix]
        reports.append(dict(warnings=['Проверьте положение детали и поддержек относительно платформы.']))
    else:
        for index, (record, mesh) in enumerate(zip(records, meshes)):
            if cancelled(): raise InterruptedError('Размещение отменено.')
            progress(f"{index + 1}/{len(records)}: {record['filename']}")
            if operation == 'minimize_bbox':
                matrix, report = minimum_oriented_bounds(mesh, progress=progress, cancelled=cancelled)
            else:
                kwargs = dict(objective=parameters['objective'], overhang_angle_deg=parameters['overhang_angle_deg'],
                              progress=progress, cancelled=cancelled)
                if operation == 'compare_orientations':
                    variants = orientation_candidates(mesh, **kwargs)
                    if parameters['drop']:
                        for variant in variants: variant['matrix'] = floor_matrix(variant['matrix'], mesh)
                    matrix, report = variants[0]['matrix'], variants[0]
                else: matrix, report = optimize_orientation(mesh, **kwargs)
            if parameters['drop']: matrix = floor_matrix(matrix, mesh)
            matrices[index] = matrix
            reports.append(dict(report, name=record['filename']))
    if cancelled(): raise InterruptedError('Размещение отменено.')
    if len(matrices) != len(records) or any(np.asarray(m).shape != (4, 4) or not np.isfinite(m).all() for m in matrices):
        raise ValueError('Расчёт вернул некорректное преобразование.')
    return dict(matrices=dict(zip((record['row'] for record in records), matrices)), reports=reports,
                variants=variants, platform=parameters.get('platform') if operation in PLATFORM_OPERATIONS else None)


class PlacementTools(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        for operation, button in window.ui.placement_buttons.items():
            if operation not in BASIC:
                button.clicked.connect(lambda checked=False, op=operation: self.open(op))

    def open(self, operation):
        window = self.window
        if window._busy(): return
        rows = window.selected_slicer_rows()
        if not rows:
            window.log('Отметьте детали в таблице для изменения расположения.')
            window.ui.status_label.setText('Сначала выберите детали.')
            return
        if operation in {'compare_orientations', 'top_bottom'} and len(rows) != 1:
            window.log('Для этой команды выберите одну деталь.'); return
        if operation == 'sort_by_shape' and not 2 <= len(rows) <= 30:
            window.log('Для переноса ориентации выберите образец и похожие детали: всего от 2 до 30.'); return
        if len(rows) > 200:
            window.log('За одну операцию размещайте не более 200 деталей.'); return
        window.workspace_tools.measurements.stop()
        window.workspace_tools.manual = None
        window.workspace_tools.clear_selection()
        window.flush_history()
        PlacementSession(window, operation, rows)


class PlacementSession(QObject):
    def __init__(self, window, operation, rows):
        super().__init__(window)
        self.window, self.operation, self.rows = window, operation, list(rows)
        self.plotter = window.ui.slicer_plotter
        self.records = [dict(row=row, **{key: window.slicer_parts[row].get(key) for key in
                                        ('mesh', 'filename', 'platform', 'supports')}) for row in rows]
        for record in self.records: record['supports'] = record['supports'] or []
        platforms = [deepcopy(p) for p in window.platforms if p.get('is_default', False)]
        self.dialog = PlacementDialog(operation, [r['filename'] for r in self.records], platforms, window)
        index = window.ui.scene_tabs.currentIndex() - 1
        if 0 <= index < len(platforms): self.dialog.platform.setCurrentIndex(index)
        self.result, self.surface, self.drag = None, None, None
        self.platform_preview_name = None
        self.closed, self.picking, self._consumed_press = False, operation == 'free_move', False
        controls = [window.ui.magics_ribbon, window.ui.toolbar, window.ui.tbl_parts, window.ui.scene_tabs,
                    window.ui.section_panel, window.ui.action_save, window.ui.action_undo, window.ui.action_redo,
                    window.ui.btn_back_to_start, window.ui.surface_toolbar, window.ui.measurement_panel, window.ui.cb_plat]
        if window.workspace_tools.supports.panel: controls.append(window.workspace_tools.supports.panel)
        self.controls = [(control, control.isEnabled()) for control in controls]
        for control, _ in self.controls: control.setEnabled(False)
        window._placement_session = self
        window.ui.section_panel._make_widget()
        window.update_history_actions()
        self.dialog.changed.connect(self.changed)
        self.dialog.prepare_requested.connect(self.prepare)
        self.dialog.apply_requested.connect(self.apply)
        self.dialog.pick_requested.connect(self.set_picking)
        self.dialog.variant_requested.connect(self.choose_variant)
        self.dialog.preview.toggled.connect(self.preview)
        self.dialog.cancel.clicked.connect(window.cancel_current_job)
        self.dialog.finished.connect(self.close)
        if hasattr(self.plotter, 'installEventFilter'): self.plotter.installEventFilter(self)
        self.dialog.show()
        position = window.mapToGlobal(window.rect().topLeft())
        self.dialog.move(position.x() + 25, position.y() + 110)
        if operation == 'free_move': self.changed()

    def identity(self): return {row: np.eye(4) for row in self.rows}

    def sources_current(self):
        if any(record['row'] >= len(self.window.slicer_parts) or
               self.window.slicer_parts[record['row']]['mesh'] is not record['mesh'] for record in self.records):
            raise ValueError('Модель изменилась. Закройте инструмент и повторите операцию.')

    def changed(self, *_):
        if self.closed: return
        self.result = None
        self.dialog.apply.setEnabled(False)
        self.dialog.table.setRowCount(0)
        if self.operation == 'free_move':
            matrix = np.eye(4)
            matrix[:3, 3] = self.dialog.parameters()['delta']
            self.result = dict(matrices={row: matrix.copy() for row in self.rows}, reports=[], variants=[], platform=None)
            self.dialog.apply.setEnabled(bool(np.any(matrix[:3, 3])))
            self.dialog.status.setText('Потяните деталь мышью или задайте смещение. Alt + мышь — вращение сцены.')
        else: self.dialog.status.setText('Подготовьте результат с выбранными параметрами.')
        self.preview()

    def preview(self, *_):
        if not self.closed:
            matrices = self.result['matrices'] if self.result and self.dialog.preview.isChecked() and not self.picking else self.identity()
            # Free dragging owns the gesture but must still show its preview.
            if self.operation == 'free_move' and self.result and self.dialog.preview.isChecked(): matrices = self.result['matrices']
            target = self.result.get('platform') if self.result and self.dialog.preview.isChecked() else None
            if target and self.platform_preview_name != target['name']:
                self.platform_preview_name = target['name']
                self.window.draw_platform(target)
                self.show_target_parts(target['name'])
            elif target is None and self.platform_preview_name is not None:
                self.platform_preview_name = None
                self.window.refresh_scene_visibility()
            self.window.preview_transforms(matrices)

    def show_target_parts(self, platform):
        # The tab itself still belongs to the original scene until Apply. Show
        # the actual target scene for review without changing any assignments.
        for row, part in enumerate(self.window.slicer_parts):
            visible = self.window.ui.tbl_parts.cellWidget(row, self.window.COL_VISIBLE).findChild(QCheckBox).isChecked()
            show = row in self.rows or (part.get('platform') == platform and visible)
            if show:
                self.window._apply_part_display_mode(row, part.get('last_visible_mode', 'shaded_wire'), sync_visible_checkbox=False)
            else:
                for suffix in ('', '__bbox'):
                    actor = self.plotter.actors.get(part['actor_name'] + suffix)
                    if actor is not None: actor.SetVisibility(False)
        from part_supports import sync_actors
        sync_actors(self.window)

    def prepare(self):
        if self.closed or self.dialog.running: return
        try:
            self.sources_current()
            params = self.dialog.parameters()
            if self.operation == 'top_bottom' and self.surface is None: raise ValueError('Сначала выберите поверхность в сцене.')
            self.set_picking(False)
            self.changed()
            parts = [dict(row=row, mesh=part['mesh'], supports=part.get('supports', []), platform=part.get('platform'))
                     for row, part in enumerate(self.window.slicer_parts)]
            worker = FunctionWorker(calculate_placement, self.records, self.operation, params, parts, self.surface, with_progress=True)
            worker.progress.connect(self.dialog.status.setText)
            worker.error.connect(self.dialog.report.setPlainText)
            worker.finished.connect(lambda: QTimer.singleShot(0, self.finished_calculation))
            self.window._applying_placement = True
            try: accepted = self.window.start_job(worker, lambda result: setattr(self.window, '_job_next', lambda: self.prepared(result)))
            finally: self.window._applying_placement = False
            if accepted: self.dialog.set_running(True)
        except Exception as exc: self.dialog.status.setText(str(exc))

    def finished_calculation(self):
        if not self.closed:
            self.dialog.set_running(False)
            if self.result is None: self.dialog.status.setText('Расчёт остановлен или завершён с ошибкой. Исходные детали сохранены.')

    def prepared(self, result):
        if self.closed: return
        self.result = result
        lines = []
        for report in result['reports']:
            if report.get('name'): lines.append(report['name'])
            for key, caption, unit in [('height_mm', 'Высота', 'мм'), ('footprint_mm2', 'Габаритное основание', 'мм²'),
                                      ('overhang_area_mm2', 'Площадь нависаний', 'мм²'), ('bbox_volume_mm3', 'Габаритный объём', 'мм³')]:
                if key in report: lines.append(f"{caption}: {report[key]:.3f} {unit}")
            for entry in report.get('matches', []):
                name = self.records[entry['index']]['filename']
                lines.append(f"{name}: {'ориентация перенесена' if entry['matched'] else 'без изменения'} — {entry.get('reason', '')}")
            lines.extend(report.get('warnings', []))
        self.dialog.report.setPlainText('\n'.join(lines) or f'Подготовлено деталей: {len(self.rows)}. Проверьте расположение в сцене.')
        if result['variants']: self.dialog.set_variants(result['variants'])
        self.dialog.apply.setEnabled(True)
        self.dialog.status.setText('Проверьте предпросмотр. Геометрия будет изменена только после «Применить».')
        self.preview()

    def choose_variant(self, row):
        if not self.result or not 0 <= row < len(self.result['variants']): return
        variant = self.result['variants'][row]
        self.result['matrices'][self.rows[0]] = variant['matrix']
        self.dialog.status.setText('Предпросмотр: ' + variant.get('label', f'Вариант {row + 1}'))
        self.dialog.report.setPlainText('\n'.join(
            f'{caption}: {variant[key]:.3f} {unit}' for key, caption, unit in (
                ('height_mm', 'Высота', 'мм'), ('footprint_mm2', 'Габаритное основание', 'мм²'),
                ('overhang_area_mm2', 'Площадь нависаний', 'мм²'), ('bbox_volume_mm3', 'Габаритный объём', 'мм³'))
            if key in variant))
        self.preview()

    def apply(self):
        if not self.result or self.dialog.running or self.drag: return
        try:
            self.sources_current()
            window = self.window
            window.preview_transforms(self.identity())
            window.flush_history()
            before = window.capture_project()
            prepared = []
            from part_supports import transformed, remove_actors
            target = self.result['platform']['name'] if self.result.get('platform') else None
            for record in self.records:
                row, matrix = record['row'], self.result['matrices'][record['row']]
                if np.array_equal(matrix, np.eye(4)) and (target is None or record['platform'] == target): continue
                mesh = record['mesh'].copy()
                from cad_state import apply_cad_transform
                apply_cad_transform(mesh, matrix)
                prepared.append((row, mesh, transformed(record['supports'], matrix)))
            camera = deepcopy(self.plotter.camera_position) if hasattr(self.plotter, 'camera_position') else None
            was_restoring = window._restoring
            window._restoring = True
            try:
                for row, mesh, supports in prepared:
                    part = window.slicer_parts[row]
                    remove_actors(window, part.get('supports', []))
                    part['supports'] = supports
                    if target is not None: part['platform'] = target
                    window.replace_slicer_mesh(row, mesh)
                if target is not None:
                    platforms = [p for p in window.platforms if p.get('is_default', False)]
                    index = next(i + 1 for i, p in enumerate(platforms) if p['name'] == target)
                    window.ui.scene_tabs.setCurrentIndex(index)
                window.refresh_scene_visibility(); window.update_info_combobox()
                window.update_parts_table_filter(window.ui.scene_tabs.currentIndex())
            except Exception:
                window.restore_project(before)
                raise
            finally:
                window._restoring = was_restoring
                if camera is not None: self.plotter.camera_position = camera
                self.plotter.render()
            if prepared:
                window.mark_dirty(); window.flush_history(PLACEMENT_COMMANDS[self.operation])
                window.log(PLACEMENT_COMMANDS[self.operation] + ': применено. Ctrl+Z — отменить.')
            self.dialog.accept()
        except Exception as exc: self.dialog.status.setText(str(exc))

    def set_picking(self, enabled):
        self.picking = bool(enabled)
        self.dialog.pick.blockSignals(True); self.dialog.pick.setChecked(self.picking); self.dialog.pick.blockSignals(False)
        if hasattr(self.plotter, 'setCursor'): self.plotter.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        if enabled:
            self.changed()
            self.dialog.status.setText('Укажите поверхность левой кнопкой. Alt + мышь — вращение; Esc — завершить выбор.')
        else: self.preview()

    def ray(self, position):
        renderer = self.plotter.renderer
        ratio = self.plotter.devicePixelRatioF()
        points = []
        for depth in (0., 1.):
            renderer.SetDisplayPoint(position.x() * ratio, self.plotter.render_window.GetSize()[1] - 1 - position.y() * ratio, depth)
            renderer.DisplayToWorld(); value = np.asarray(renderer.GetWorldPoint())
            if abs(value[3]) < 1e-15: raise ValueError('Не удалось определить направление курсора.')
            points.append(value[:3] / value[3])
        return points[0], points[1] - points[0]

    def drag_point(self, position, origin, normal):
        start, direction = self.ray(position)
        denominator = float(direction @ normal)
        if abs(denominator) <= np.linalg.norm(direction) * 1e-9:
            raise ValueError('Выбранная плоскость видна с ребра. Поверните сцену или выберите плоскость экрана.')
        return start + direction * ((origin - start) @ normal / denominator)

    def eventFilter(self, obj, event):
        if obj is not self.plotter or self.closed or self.dialog.running: return False
        kind = event.type()
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            if self.drag:
                for spin, value in zip(self.dialog.delta, self.drag['delta']):
                    spin.blockSignals(True); spin.setValue(float(value)); spin.blockSignals(False)
                self.changed()
            self.drag = None
            if self.operation == 'top_bottom': self.set_picking(False)
            return True
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._consumed_press:
            self._consumed_press = False; self.drag = None; return True
        if kind == QEvent.MouseMove and self.drag:
            try:
                point = self.drag_point(event.position(), self.drag['origin'], self.drag['normal'])
                increment = point - self.drag['start']
                step = self.dialog.fields['snap_mm'].value()
                if step > 0:
                    basis = self.drag['basis']
                    increment = (np.round(basis @ increment / step) * step) @ basis
                delta = self.drag['delta'] + increment
                for spin, value in zip(self.dialog.delta, delta):
                    spin.blockSignals(True); spin.setValue(float(value)); spin.blockSignals(False)
                self.changed()
            except ValueError as exc: self.dialog.status.setText(str(exc))
            return True
        if kind != QEvent.MouseButtonPress or event.button() != Qt.LeftButton or event.modifiers() & Qt.AltModifier: return False
        if self.operation != 'free_move' and not self.picking: return False
        cube = self.window.workspace_tools.cube
        if cube and cube.face_at(event.position() - cube.geometry().topLeft()) is not None: return False
        hit = self.window.workspace_tools.picker(event.position().toPoint(), rows=self.rows)
        if hit is None: return False
        self._consumed_press = True
        row, face, point = hit
        if self.operation == 'top_bottom':
            self.surface = face
            self.dialog.surface.setText(f'Выбран треугольник №{face}')
            self.set_picking(False)
            self.prepare()
        elif self.operation == 'free_move':
            try:
                plane = self.dialog.plane.currentIndex()
                normal = np.asarray(self.plotter.camera.GetDirectionOfProjection()) if plane == 0 else np.eye(3)[{1: 2, 2: 1, 3: 0}[plane]]
                if plane:
                    basis = np.eye(3)[{1: [0, 1], 2: [0, 2], 3: [1, 2]}[plane]]
                else:
                    horizontal = np.cross(normal, np.asarray(self.plotter.camera.GetViewUp()))
                    length = np.linalg.norm(horizontal)
                    if length < 1e-12: raise ValueError('Не удалось определить оси экрана; выберите координатную плоскость.')
                    horizontal /= length
                    basis = np.array([horizontal, np.cross(horizontal, normal)])
                self.drag = dict(origin=point, normal=normal, start=self.drag_point(event.position(), point, normal),
                                 delta=np.array([spin.value() for spin in self.dialog.delta]), basis=basis)
            except ValueError as exc: self.dialog.status.setText(str(exc))
        return True

    def close(self, *_):
        if self.closed: return
        self.closed = True; self.drag = None
        self.window.preview_transforms(self.identity())
        if self.platform_preview_name is not None:
            self.platform_preview_name = None
            self.window.refresh_scene_visibility()
        if hasattr(self.plotter, 'removeEventFilter'): self.plotter.removeEventFilter(self)
        if hasattr(self.plotter, 'setCursor'): self.plotter.setCursor(Qt.ArrowCursor)
        self.window._placement_session = None
        for control, enabled in self.controls: control.setEnabled(enabled)
        self.window.ui.section_panel._make_widget(); self.window.update_history_actions()
        self.dialog.deleteLater(); self.deleteLater()
