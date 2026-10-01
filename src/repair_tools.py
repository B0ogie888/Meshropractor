"""Slicer repair transactions, asynchronous previews and direct vertex picking."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject, QEvent, Qt, QPointF, QTimer

from background_tasks import FunctionWorker
from repair_dialog import RepairDialog, MANUAL
from repair_ribbon import REPAIR_COMMANDS


def split_support_groups(groups, face_maps, face_count):
    """Reparent supports using exact source-face IDs, never proximity guesses."""
    result = [[] for _ in face_maps]
    if not groups:
        return result
    owner = np.full(face_count, -1, dtype=np.int32)
    local = np.full(face_count, -1, dtype=np.int64)
    for index, ids in enumerate(face_maps):
        owner[ids] = index
        local[ids] = np.arange(len(ids))
    for group in groups:
        ids = np.asarray(group.get('surface_faces', []), dtype=np.int64)
        if not len(ids):
            raise ValueError('Нельзя определить фрагмент поддержки без привязки к поверхности. '
                             'Сначала заново задайте её область или удалите эту поддержку.')
        components = np.unique(owner[ids])
        if len(components) != 1 or components[0] < 0:
            raise ValueError('Область одной поддержки охватывает несколько фрагментов. '
                             'Перед разделением задайте отдельные области поддержек для каждого фрагмента.')
        copy = deepcopy(group)
        copy['surface_faces'] = local[ids].tolist()
        result[int(components[0])].append(copy)
    return result


def calculate_repairs(records, operation, parameters, selection=None, vertex_ids=None,
                      progress=lambda message: None, cancelled=lambda: False):
    """Pure preparation: no GUI objects and no mutation of source project meshes."""
    from repair_operations import repair_mesh
    from repair_manual_geometry import edit_mesh, split_components, combine_meshes
    selection = selection or {}
    items = []
    split_count = 0
    if operation == 'unify':
        if cancelled(): raise InterruptedError('Объединение отменено')
        progress('Булево объединение выбранных деталей…')
        mesh = combine_meshes([record['mesh'] for record in records])
        if cancelled(): raise InterruptedError('Объединение отменено')
        return dict(mode=operation, items=[dict(row=records[0]['row'], meshes=[mesh],
                    report=dict(changed=True, before={'faces': sum(len(x['mesh'].faces) for x in records)},
                                after={'faces': len(mesh.faces)}))])
    for record in records:
        if cancelled(): raise InterruptedError('Исправление отменено')
        progress(record['name'] + ': ' + REPAIR_COMMANDS[operation])
        row, source = record['row'], record['mesh']
        selected = selection.get(row)
        extra = {}
        if operation == 'split':
            meshes, face_maps = split_components(source, return_face_maps=True, max_components=200 - split_count)
            split_count += len(meshes)
            if len(meshes) > 1:
                extra['supports'] = split_support_groups(record.get('supports', []), face_maps, len(source.faces))
            report = dict(changed=len(meshes) > 1, fragments=len(meshes))
        elif operation == 'remove_small':
            small = (parameters.get('min_faces', 0) > 0 and len(source.faces) < parameters['min_faces'])
            if parameters.get('min_volume_mm3', 0) > 0 and source.is_watertight and source.is_winding_consistent:
                small |= abs(source.volume) < parameters['min_volume_mm3']
            meshes, report = ([] if small else [source]), dict(changed=bool(small), removed=bool(small))
        elif operation == 'auto':
            from mesh_full_repair import prepare_full_repair
            mesh, report = prepare_full_repair(source, 'Part', progress=progress, cancelled=cancelled, **parameters)
            meshes = [mesh]
        elif operation in MANUAL or operation == 'clip':
            mesh, report = edit_mesh(source, 'move_vertices' if operation == 'drag_vertices' else operation,
                                    parameters, face_ids=selected, vertex_ids=vertex_ids,
                                    progress=progress, cancelled=cancelled)
            meshes = [mesh]
        else:
            mesh, report = repair_mesh(source, operation, parameters, face_ids=selected,
                                      progress=progress, cancelled=cancelled)
            meshes = [mesh]
        if operation in ('normals', 'smooth', 'holes', 'move_vertices', 'drag_vertices', 'fill_hole', 'bridge', 'add_triangle'):
            report['face_indices_preserved'] = True
        if operation == 'delete_faces':
            face_map = np.full(len(source.faces), -1, dtype=np.int64)
            keep = np.ones(len(source.faces), dtype=bool)
            keep[selected] = False
            face_map[keep] = np.arange(np.count_nonzero(keep))
            report['face_map'] = face_map
        items.append(dict(row=row, meshes=meshes, report=report, **extra))
    if cancelled(): raise InterruptedError('Исправление отменено')
    return dict(mode=operation, items=items)


class RepairTools(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        for operation, button in window.ui.repair_buttons.items():
            button.clicked.connect(lambda checked=False, op=operation: self.open(op))

    def open(self, operation):
        window = self.window
        if window._busy(): return
        if operation == 'wizard':
            window.open_repair_wizard()
            return
        rows = window.selected_slicer_rows()
        if not rows:
            window.log('Отметьте детали в таблице для исправления.')
            window.ui.status_label.setText('Выберите деталь перед исправлением.')
            return
        if operation in MANUAL and len(rows) != 1:
            window.log('Для ручного исправления выберите одну деталь.')
            return
        if operation == 'unify':
            if len(rows) < 2:
                window.log('Для объединения выберите не менее двух деталей.')
                return
            if len({window.slicer_parts[row].get('platform') for row in rows}) != 1:
                window.log('Объединяемые детали должны находиться на одной платформе.')
                return
        RepairSession(window, operation, rows)


class RepairSession(QObject):
    def __init__(self, window, operation, rows):
        super().__init__(window)
        self.window, self.operation, self.rows = window, operation, list(rows)
        self.plotter = window.ui.slicer_plotter
        self.records = [dict(row=row, mesh=window.slicer_parts[row]['mesh'], name=window.slicer_parts[row]['filename'],
                             supports=window.slicer_parts[row].get('supports', [])) for row in rows]
        self.selection = {row: sorted(window.workspace_tools.selection.get(row, ())) for row in rows}
        center = np.mean([record['mesh'].bounds.mean(axis=0) for record in self.records], axis=0)
        self.dialog = RepairDialog(operation, len(rows), center, window)
        self.result = None
        self.closed = False
        self.preview_actors = []
        self.hidden_actors = []
        self.marker = None
        self.picking = False
        self.drag = None
        self._consumed_press = False
        controls = [window.ui.magics_ribbon, window.ui.toolbar, window.ui.tbl_parts, window.ui.scene_tabs,
                    window.ui.section_panel, window.ui.action_save, window.ui.action_undo, window.ui.action_redo,
                    window.ui.btn_back_to_start, window.ui.surface_toolbar, window.ui.measurement_panel]
        if window.workspace_tools.supports.panel: controls.append(window.workspace_tools.supports.panel)
        self.controls = [(control, control.isEnabled()) for control in controls]
        for control, _ in self.controls: control.setEnabled(False)
        window.workspace_tools.measurements.stop()
        window.workspace_tools.manual = None
        window._repair_session = self
        window.ui.section_panel._make_widget()
        window.flush_history()
        if operation == 'delete_faces': self.dialog.set_selected_ids(self.selection[rows[0]])
        self.dialog.changed.connect(self.invalidate)
        self.dialog.selection_changed.connect(self.selection_updated)
        self.dialog.prepare_requested.connect(self.prepare)
        self.dialog.apply_requested.connect(self.apply)
        self.dialog.pick_requested.connect(self.set_picking)
        self.dialog.selection_requested.connect(self.from_selection)
        self.dialog.preview.toggled.connect(self.preview)
        self.dialog.cancel.clicked.connect(window.cancel_current_job)
        self.dialog.finished.connect(self.close)
        self.plotter.installEventFilter(self) if hasattr(self.plotter, 'installEventFilter') else None
        self.dialog.show()
        position = window.mapToGlobal(window.rect().topLeft())
        self.dialog.move(position.x() + 25, position.y() + 110)
        window.update_history_actions()

    def sources_current(self):
        if any(r['row'] >= len(self.window.slicer_parts) or self.window.slicer_parts[r['row']]['mesh'] is not r['mesh'] for r in self.records):
            raise ValueError('Модель изменилась: закройте инструмент и повторите операцию.')

    def invalidate(self, *_):
        if self.closed: return
        self.clear_preview()
        self.result = None
        self.dialog.apply.setEnabled(False)
        self.dialog.status.setText('Параметры изменены. Подготовьте новый результат.')

    def prepare(self):
        if self.closed or self.dialog.running: return
        try:
            self.sources_current()
            parameters = self.dialog.parameters()
            selection = self.selection if self.dialog.only_faces.isChecked() else {}
            if self.dialog.only_faces.isChecked() and not all(selection.values()):
                raise ValueError('Сначала выделите поверхности каждой выбранной детали.')
            vertices = None
            if self.operation in MANUAL:
                ids = self.dialog.selected_ids()
                if not ids: raise ValueError('Выберите треугольники или вершины в сцене.')
                if self.operation == 'delete_faces': selection = {self.rows[0]: ids}
                else: vertices = ids
            if self.operation == 'remove_small' and parameters['min_faces'] == 0 and parameters['min_volume_mm3'] == 0:
                raise ValueError('Задайте хотя бы один ненулевой порог.')
            self.set_picking(False)
            self.invalidate()
            self.clear_markers()
            worker = FunctionWorker(calculate_repairs, self.records, self.operation, parameters, selection, vertices, with_progress=True)
            worker.progress.connect(self.dialog.status.setText)
            worker.error.connect(self.dialog.report.setPlainText)
            worker.finished.connect(lambda: QTimer.singleShot(0, self.finished_calculation))
            self.window._applying_repair = True
            try:
                accepted = self.window.start_job(worker, lambda value: setattr(self.window, '_job_next', lambda: self.prepared(value)))
            finally: self.window._applying_repair = False
            if accepted: self.dialog.set_running(True)
        except Exception as exc:
            self.dialog.status.setText(str(exc))

    def finished_calculation(self):
        if not self.closed:
            self.dialog.set_running(False)
            if self.result is None: self.dialog.status.setText('Расчёт завершён или отменён. Результат ещё не применён.')

    def prepared(self, result):
        if self.closed: return
        self.result = result
        lines = []
        for item in result['items']:
            report = item['report']
            lines.append(self.window.slicer_parts[item['row']]['filename'])
            if 'before' in report and 'after' in report:
                lines.append(f"Треугольников: {report['before'].get('faces', '—')} → {report['after'].get('faces', '—')}")
            if 'fragments' in report: lines.append(f"Фрагментов: {report['fragments']}")
            if 'removed' in report: lines.append('Будет удалена' if report['removed'] else 'Сохраняется')
            if 'shape' in report: lines.append(f"Изменение формы, максимум: {report['shape']['max_mm']:.6f} мм")
            if 'defects' in report:
                lines.append('Остаются дефекты: результат не считается полностью исправным.' if report['defects']
                             else 'По полной проверке дефектов не найдено.')
            if 'selected_faces' in report: lines.append(f"Найдено треугольников: {len(report['selected_faces'])}")
            lines.extend(report.get('warnings', []))
            if report.get('changed', True) and self.operation not in ('slivers', 'overlaps'):
                from cad_state import cad_status
                source = self.window.slicer_parts[item['row']]['mesh']
                if cad_status(source) == 'native' and any(cad_status(mesh) != 'native' for mesh in item['meshes']):
                    lines.append('Результат редактируется как сетка: BREP больше не соответствует изменённой геометрии. '
                                 'Экспорт STEP станет недоступен; Ctrl+Z восстановит CAD.')
            if not report.get('acceptable', True): lines.append('Допуск формы превышен: применение заблокировано.')
        self.dialog.report.setPlainText('\n'.join(lines))
        if self.operation in ('slivers', 'overlaps'):
            work = self.window.workspace_tools
            work.selection = {item['row']: set(map(int, item['report']['selected_faces'])) for item in result['items']
                              if len(item['report']['selected_faces'])}
            work.draw_selection()
            self.selection = {row: sorted(work.selection.get(row, ())) for row in self.rows}
            self.dialog.status.setText('Найденные треугольники подсвечены. После закрытия можно исправить выделение вручную.')
        else:
            changed = any(item['report'].get('changed', True) for item in result['items'])
            allowed = all(item['report'].get('acceptable', True) for item in result['items'])
            self.dialog.apply.setEnabled(changed and allowed)
            self.dialog.status.setText('Проверьте предпросмотр и отчёт перед применением.' if changed else 'Изменения не требуются.')
            self.preview()

    def clear_preview(self):
        for actor in self.preview_actors: self.plotter.remove_actor(actor, render=False)
        self.preview_actors.clear()
        for actor, visibility in self.hidden_actors: actor.SetVisibility(visibility)
        self.hidden_actors.clear()
        if hasattr(self.plotter, 'render'): self.plotter.render()

    def preview(self, *_):
        self.clear_preview()
        if not self.result or not self.dialog.preview.isChecked() or self.operation in ('slivers', 'overlaps'): return
        for row in self.rows:
            name = self.window.slicer_parts[row]['actor_name']
            for key in (name, name + '__bbox', f'surface_selection_{row}'):
                actor = self.plotter.actors.get(key)
                if actor is not None:
                    self.hidden_actors.append((actor, actor.GetVisibility())); actor.SetVisibility(False)
        for index, item in enumerate(self.result['items']):
            for child, mesh in enumerate(item['meshes']):
                actor = self.plotter.add_mesh(self.window.trimesh_to_pyvista(mesh), color='#75d6b0',
                    name=f'repair_preview_{index}_{child}', pickable=False, render=False)
                source = self.plotter.actors[self.window.slicer_parts[item['row']]['actor_name']]
                planes = source.GetMapper().GetClippingPlanes()
                if planes:
                    for i in range(planes.GetNumberOfItems()): actor.GetMapper().AddClippingPlane(planes.GetItem(i))
                self.preview_actors.append(actor)
        self.plotter.render()

    def apply(self):
        if not self.result or self.dialog.running: return
        if not all(item['report'].get('acceptable', True) for item in self.result['items']): return
        try:
            self.sources_current()
            self.clear_preview()
            self.clear_markers()
            window = self.window
            window.flush_history()
            before = window.capture_project()
            cameras = [deepcopy(p.camera_position) if p is not None and hasattr(p, 'camera_position') else None
                       for p in (window.ui.plotter, window.ui.slicer_plotter)]
            previous_restoring = window._restoring
            window._restoring = True
            try:
                if self.operation in ('unify', 'split', 'remove_small'):
                    # Existing meshes are immutable snapshots; copying every CAD/scan
                    # here would stall the GUI and duplicate unrelated large models.
                    state = deepcopy(before, {id(record['mesh']): record['mesh'] for record in before.models + before.parts})
                    by_row = {item['row']: item for item in self.result['items']}
                    parts = []
                    for row, record in enumerate(state.parts):
                        if row not in self.rows:
                            parts.append(record); continue
                        if self.operation == 'unify' and row != self.rows[0]: continue
                        item = by_row.get(row)
                        if not item['report'].get('changed', True):
                            parts.append(record)
                            continue
                        for index, mesh in enumerate(item['meshes']):
                            replacement = deepcopy(record, {id(record['mesh']): mesh})
                            if self.operation == 'unify':
                                replacement['filename'] = 'Объединение.stl'
                                replacement['supports'] = [deepcopy(group) for r in self.rows for group in before.parts[r].get('supports', [])]
                            elif self.operation == 'split' and len(item['meshes']) > 1:
                                replacement['filename'] = f"{Path(record['filename']).stem} — фрагмент {index+1}.stl"
                                replacement['supports'] = deepcopy(item['supports'][index])
                            if self.operation != 'split':
                                for group in replacement.get('supports', []): group['surface_faces'] = []
                            parts.append(replacement)
                    state.parts = parts
                    window.restore_project(state)
                else:
                    for item in self.result['items']:
                        if not item['report'].get('changed', True): continue
                        window.replace_slicer_mesh(item['row'], item['meshes'][0])
                        for group in window.slicer_parts[item['row']].get('supports', []):
                            mapping = item['report'].get('face_map')
                            if mapping is not None:
                                indices = mapping[np.asarray(group['surface_faces'], dtype=np.int64)]
                                group['surface_faces'] = indices[indices >= 0].tolist()
                            elif not item['report'].get('face_indices_preserved', False):
                                group['surface_faces'] = []
                    window.update_info_combobox(); window.refresh_scene_visibility(); window.ui.section_panel.apply()
            except Exception:
                window.restore_project(before)
                raise
            finally:
                window._restoring = previous_restoring
                for plotter, camera in zip((window.ui.plotter, window.ui.slicer_plotter), cameras):
                    if plotter is not None and camera is not None: plotter.camera_position = camera; plotter.render()
            window.mark_dirty(); window.flush_history(REPAIR_COMMANDS[self.operation])
            window.log(REPAIR_COMMANDS[self.operation] + ': применено. Ctrl+Z — отменить.')
            self.dialog.accept()
        except Exception as exc:
            self.dialog.status.setText(str(exc))

    def from_selection(self):
        faces = self.selection[self.rows[0]]
        ids = faces if self.operation == 'delete_faces' else np.unique(self.records[0]['mesh'].faces[faces]).tolist()
        self.dialog.set_selected_ids(ids)

    def selection_updated(self):
        self.show_markers()
        if self.operation == 'delete_faces':
            try: ids = self.dialog.selected_ids()
            except ValueError: return
            if ids and (min(ids) < 0 or max(ids) >= len(self.records[0]['mesh'].faces)): return
            self.window.workspace_tools.edit_selection(self.rows[0], ids, 'replace')

    def set_picking(self, enabled):
        self.picking = bool(enabled)
        self.dialog.pick_button.blockSignals(True); self.dialog.pick_button.setChecked(self.picking); self.dialog.pick_button.blockSignals(False)
        self.plotter.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor) if hasattr(self.plotter, 'setCursor') else None
        if enabled:
            self.invalidate()
            self.dialog.status.setText('Левая кнопка — выбор; Alt + мышь — вращение; Esc — завершить выбор.')

    def clear_markers(self):
        if self.marker is not None: self.plotter.remove_actor(self.marker); self.marker = None

    def show_markers(self):
        self.clear_markers()
        try: ids = self.dialog.selected_ids()
        except ValueError: return
        if not ids: return
        mesh = self.records[0]['mesh']
        # Markers are only visual hints. The complete selection is used by the operation.
        marker_ids = np.asarray(ids)[np.linspace(0, len(ids) - 1, min(len(ids), 5000), dtype=int)]
        if self.operation == 'delete_faces':
            if min(ids) < 0 or max(ids) >= len(mesh.faces): return
            points = mesh.vertices[mesh.faces[marker_ids]].mean(axis=1)
        else:
            if min(ids) < 0 or max(ids) >= len(mesh.vertices): return
            points = mesh.vertices[marker_ids]
        if hasattr(self.plotter, 'add_points'):
            self.marker = self.plotter.add_points(points, color='#ffc34d', point_size=11, render_points_as_spheres=True,
                                                   name='repair_points', pickable=False)
        self.plotter.render()

    def eventFilter(self, obj, event):
        if obj is not self.plotter or self.closed: return False
        kind = event.type()
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.restore_drag(); self.set_picking(False); return True
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self._consumed_press:
            self._consumed_press = False
            if self.drag:
                delta = self.drag['last'] - self.drag['source']
                self.restore_drag()
                self.dialog.absolute.setChecked(False)
                for spin, value in zip(self.dialog.vectors['delta'], delta): spin.setValue(float(value))
                self.prepare()
            return True
        if self.drag and kind == QEvent.MouseMove:
            renderer = self.plotter.renderer
            ratio = self.plotter.devicePixelRatioF()
            renderer.SetDisplayPoint(event.position().x()*ratio, self.plotter.render_window.GetSize()[1]-1-event.position().y()*ratio, self.drag['depth'])
            renderer.DisplayToWorld(); value = np.array(renderer.GetWorldPoint())
            if abs(value[3]) > 1e-12:
                point = value[:3]/value[3]
                self.drag['last'] = point
                self.drag['preview'].points[self.drag['vertex']] = point
                self.drag['preview'].GetPoints().Modified()
                self.plotter.render()
            return True
        if not self.picking or self.dialog.running or event.type() != QEvent.MouseButtonPress: return False
        if event.button() != Qt.LeftButton or event.modifiers() & Qt.AltModifier: return False
        cube = self.window.workspace_tools.cube
        if cube and cube.face_at(event.position() - cube.geometry().topLeft()) is not None: return False
        self._consumed_press = True
        hit = self.window.workspace_tools.picker(event.position().toPoint(), rows=self.rows)
        if hit is None: return True
        row, face, point = hit
        mesh = self.records[0]['mesh']
        vertex = int(mesh.faces[face][np.argmin(np.linalg.norm(mesh.vertices[mesh.faces[face]]-point, axis=1))])
        try: ids = self.dialog.selected_ids()
        except ValueError: ids = []
        chosen = face if self.operation == 'delete_faces' else vertex
        if self.operation in ('fill_hole', 'drag_vertices'): ids = [chosen]
        elif chosen in ids: ids.remove(chosen)
        else: ids.append(chosen)
        self.dialog.set_selected_ids(ids)
        if self.operation == 'drag_vertices':
            actor = self.plotter.actors[self.window.slicer_parts[row]['actor_name']]
            original = actor.GetMapper().GetInput()
            preview = original.copy(deep=True)
            actor.GetMapper().SetInputData(preview)
            source = mesh.vertices[vertex].copy()
            renderer = self.plotter.renderer
            renderer.SetWorldPoint(*source, 1.); renderer.WorldToDisplay()
            self.drag = dict(actor=actor, original=original, preview=preview, vertex=vertex,
                             source=source, last=source, depth=renderer.GetDisplayPoint()[2])
        return True

    def restore_drag(self):
        if self.drag:
            self.drag['actor'].GetMapper().SetInputData(self.drag['original'])
            self.drag = None; self.plotter.render()

    def close(self, *_):
        if self.closed: return
        self.closed = True
        self.restore_drag(); self.clear_markers(); self.clear_preview()
        if hasattr(self.plotter, 'removeEventFilter'): self.plotter.removeEventFilter(self)
        if hasattr(self.plotter, 'setCursor'): self.plotter.setCursor(Qt.ArrowCursor)
        self.window._repair_session = None
        for control, enabled in self.controls: control.setEnabled(enabled)
        self.window.ui.section_panel._make_widget()
        self.window.update_history_actions()
        self.dialog.deleteLater()
        self.deleteLater()
