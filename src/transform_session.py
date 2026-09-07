"""Owns a modeless transform panel, scene handles and unapplied preview."""
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QTimer
from transform_dialog import TransformDialog
from transform_math import unit, selection_bounds


class TransformSession:
    def __init__(self, window, operation, rows):
        self.window, self.operation, self.rows = window, operation, list(rows)
        self.plotter = window.ui.slicer_plotter
        self.dialog = TransformDialog(operation, [window.slicer_parts[r]['mesh'] for r in rows], window)
        self.timer = QTimer(self.dialog)
        self.timer.setSingleShot(True)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.update_preview)
        self.dialog.changed.connect(self.timer.start)
        self.dialog.apply_requested.connect(self.apply)
        self.dialog.pick_requested.connect(self.start_pick)
        self.dialog.finished.connect(self.close)
        self.gizmo = None
        self.proxy = None
        self.ghosts = []
        self.plane_actor = None
        self.picking = None
        self.pickability = {}
        self.closed = False
        self.gizmo_pending = False
        controls = [window.ui.magics_ribbon, window.ui.toolbar, window.ui.tbl_parts, window.ui.scene_tabs,
                    window.ui.section_panel, window.ui.action_save, window.ui.action_undo, window.ui.action_redo,
                    window.ui.btn_back_to_start]
        if hasattr(window.ui, 'surface_toolbar'): controls.append(window.ui.surface_toolbar)
        controls.append(window.ui.measurement_panel)
        if window.workspace_tools.supports.panel: controls.append(window.workspace_tools.supports.panel)
        self.controls = [(control, control.isEnabled()) for control in controls]
        for control, _ in self.controls: control.setEnabled(False)
        window.ui.section_panel._make_widget()
        window._transform_session = self
        self.dialog.show()
        # Keep the model visible beside the panel.
        pos = window.mapToGlobal(window.rect().topLeft())
        self.dialog.move(pos.x() + 25, pos.y() + 110)
        self.update_preview()

    def identity(self): return {row: np.eye(4) for row in self.rows}

    def clear_preview(self):
        self.window.preview_transforms(self.identity())
        for actor in self.ghosts: self.plotter.remove_actor(actor)
        self.ghosts = []

    def update_preview(self):
        if self.closed or self.picking: return
        try:
            params = self.dialog.parameters()
            matrices = self.window.matrices_for(self.operation, params, self.rows)
            self.dialog.status.setText('')
            self.dialog.apply_button.setEnabled(True)
            self.dialog.yes_button.setEnabled(True)
            if self.dialog.preview.isChecked():
                if params['create_copy']:
                    self.window.preview_transforms(self.identity())
                    if not self.ghosts:
                        for row in self.rows:
                            part = self.window.slicer_parts[row]
                            from part_supports import combined_mesh
                            ghost = self.plotter.add_mesh(self.window.trimesh_to_pyvista(combined_mesh(part)), color='#74bfea', opacity=.55, name=f'transform_preview_{row}')
                            for plane in self.window.ui.section_panel._planes: ghost.mapper.AddClippingPlane(plane)
                            ghost.pickable = False
                            self.ghosts.append(ghost)
                    for ghost, matrix in zip(self.ghosts, matrices.values()): ghost.user_matrix = matrix
                else:
                    for actor in self.ghosts: self.plotter.remove_actor(actor)
                    self.ghosts = []
                    self.window.preview_transforms(matrices)
            else: self.clear_preview()
            self.update_handles(params, matrices)
            self.plotter.render()
        except (ValueError, FloatingPointError) as exc:
            self.clear_preview()
            self.dialog.status.setText(str(exc))
            self.dialog.apply_button.setEnabled(False)
            self.dialog.yes_button.setEnabled(False)

    def clear_handles(self):
        if self.gizmo is not None:
            self.gizmo.remove()
            self.gizmo = None
        if self.proxy is not None:
            self.plotter.remove_actor(self.proxy)
            self.proxy = None
        if self.plane_actor is not None:
            self.plotter.remove_actor(self.plane_actor)
            self.plane_actor = None

    def update_handles(self, params, matrices):
        if self.operation in ('Перемещать', 'Вращать') and hasattr(self.plotter, 'add_affine_transform_widget'):
            bounds = selection_bounds([self.window.slicer_parts[r]['mesh'] for r in self.rows])
            center = self.dialog.move_anchor() if self.operation == 'Перемещать' else (
                np.asarray(params['line_a']) if params['along_line'] else bounds.mean(axis=0) if params['individual'] else params['center'])
            if self.gizmo is None:
                spans = np.maximum(bounds[1] - bounds[0], .01)
                self.proxy = self.plotter.add_mesh(pv.Cube(center=bounds.mean(axis=0), x_length=spans[0], y_length=spans[1], z_length=spans[2]), opacity=0, name='transform_handle_proxy')
                self.proxy.pickable = False
                self.gizmo = self.plotter.add_affine_transform_widget(self.proxy, origin=tuple(center), scale=.65,
                    axes_colors=('#ef5350', '#72c64b', '#448aff'), interact_callback=self.queue_gizmo,
                    release_callback=self.gizmo_released)
                # PyVista 0.47 pins these actor lists; hide the irrelevant half of its affine handle.
                for actor in self.gizmo._circles if self.operation == 'Перемещать' else self.gizmo._arrows:
                    actor.visibility = False
                    actor.pickable = False
            matrix = next(iter(matrices.values()))
            if not self.gizmo._pressing_down:
                axes = np.eye(3)
                if params['along_line']:
                    direction = unit(np.asarray(params['line_b']) - params['line_a'])
                    reference = [0, 0, 1] if abs(direction[2]) < .9 else [0, 1, 0]
                    first = unit(np.cross(reference, direction))
                    axes = np.array([first, np.cross(direction, first), direction])
                self.gizmo.axes = axes
                for index in range(3):
                    enabled = not params['along_line'] or index == 2
                    for actor, allowed in ((self.gizmo._arrows[index], self.operation == 'Перемещать'),
                                           (self.gizmo._circles[index], self.operation == 'Вращать')):
                        actor.visibility = enabled and allowed
                        actor.pickable = enabled and allowed
                self.gizmo.origin = center + matrix[:3, 3] if self.operation == 'Перемещать' else center
                self.gizmo._cached_matrix = matrix.copy()
            self.proxy.user_matrix = matrix
        elif self.operation == 'Отзеркалить':
            if self.plane_actor is not None: self.plotter.remove_actor(self.plane_actor)
            length = max(np.linalg.norm(self.dialog.bounds[1] - self.dialog.bounds[0]), .1)
            origin = self.dialog.bounds.mean(axis=0) if params['individual'] else params['center']
            self.plane_actor = self.plotter.add_mesh(pv.Plane(center=origin, direction=unit(params['normal']),
                i_size=length, j_size=length), color='#78bff0', opacity=.18, name='transform_mirror_plane')
            self.plane_actor.pickable = False

    def queue_gizmo(self, matrix):
        # PyVista calls its interaction callback before assigning the new matrix.
        if not self.gizmo_pending:
            self.gizmo_pending = True
            QTimer.singleShot(0, self.read_gizmo)

    def read_gizmo(self):
        self.gizmo_pending = False
        if not self.closed and not self.picking and self.proxy is not None:
            self.gizmo_changed(self.proxy.user_matrix)

    def gizmo_changed(self, matrix):
        if self.closed: return
        dialog = self.dialog
        matrix = np.asarray(matrix)
        if self.operation == 'Перемещать':
            values = matrix[:3, 3].copy()
            if dialog.along_line.isChecked():
                direction = unit(dialog.numbers(dialog.line_b) - dialog.numbers(dialog.line_a))
                distance = float(values @ direction)
                if dialog.snap.isChecked(): distance = round(distance / dialog.snap_step.value()) * dialog.snap_step.value()
                values = direction * distance
            elif dialog.snap.isChecked(): values = np.round(values / dialog.snap_step.value()) * dialog.snap_step.value()
            dialog.set_values(dialog.values, values)
            dialog.sync_move(False)
        elif self.operation == 'Вращать':
            rotation = Rotation.from_matrix(matrix[:3, :3])
            if dialog.along_line.isChecked():
                direction = unit(dialog.numbers(dialog.line_b) - dialog.numbers(dialog.line_a))
                angle = np.rad2deg(rotation.as_rotvec() @ direction)
                if dialog.snap.isChecked(): angle = round(angle / dialog.snap_step.value()) * dialog.snap_step.value()
                dialog.set_values([dialog.line_angle], [angle])
            else:
                angles = rotation.as_euler('xyz', degrees=True)
                if dialog.snap.isChecked(): angles = np.round(angles / dialog.snap_step.value()) * dialog.snap_step.value()
                dialog.set_values(dialog.values, angles)
        self.update_preview()

    def gizmo_released(self, matrix):
        self.gizmo_changed(matrix)
        # Recreate between drags so the affine widget's cached matrix matches snapped fields.
        QTimer.singleShot(0, self.rebuild_handles)

    def rebuild_handles(self):
        if not self.closed and not self.picking:
            self.clear_handles()
            self.update_preview()

    def start_pick(self, purpose, count):
        self.timer.stop()
        self.stop_pick()
        self.clear_handles()
        self.clear_preview()
        self.picking = (purpose, count, [])
        self.dialog.status.setText(f'Щёлкните левой кнопкой по исходной поверхности: точек нужно {count}. Esc в окне — закрыть инструмент.')
        self.pickability = {key: actor.pickable for key, actor in self.plotter.actors.items() if hasattr(actor, 'pickable')}
        selected = {self.window.slicer_parts[row]['actor_name'] for row in self.rows}
        for key, actor in self.plotter.actors.items():
            if hasattr(actor, 'pickable'): actor.pickable = key in selected
        self.plotter.enable_surface_point_picking(callback=self.picked, left_clicking=True, show_message=False,
            show_point=True, color='#ffc107', point_size=12, pickable_window=False)

    def picked(self, point):
        if not self.picking: return
        purpose, count, points = self.picking
        points.append(np.asarray(point, dtype=float).copy())
        if len(points) < count:
            self.dialog.status.setText(f'Выбрано {len(points)} из {count} точек.')
            return
        self.stop_pick()
        try:
            self.dialog.accept_points(purpose, points)
            self.update_preview()
        except ValueError as exc:
            self.dialog.status.setText(str(exc) + ' Повторите выбор точек.')

    def stop_pick(self):
        if self.picking:
            self.plotter.disable_picking()
            self.plotter.remove_actor('_picked_point')
            for key, pickable in self.pickability.items():
                actor = self.plotter.actors.get(key)
                if actor is not None: actor.pickable = pickable
            self.picking = None
            self.pickability = {}

    def apply(self, close_after=False):
        if self.picking:
            self.dialog.status.setText('Завершите выбор точек перед применением.')
            return
        self.timer.stop()
        try:
            params = self.dialog.parameters()
            self.window.matrices_for(self.operation, params, self.rows)
            self.clear_handles()
            self.clear_preview()
            before_count = len(self.window.slicer_parts)
            self.window._applying_transform = True
            try: self.window.apply_slicer_tool(self.operation, params, self.rows)
            finally: self.window._applying_transform = False
            if params['create_copy']: self.rows = list(range(before_count, len(self.window.slicer_parts)))
            self.dialog.rebase([self.window.slicer_parts[row]['mesh'] for row in self.rows])
            if self.operation == 'Отзеркалить': self.dialog.preview.setChecked(False)
            if close_after: self.dialog.accept()
            else: self.update_preview()
        except Exception as exc:
            self.dialog.status.setText(str(exc))

    def close(self, *args):
        if self.closed: return
        self.closed = True
        self.timer.stop()
        self.stop_pick()
        self.clear_handles()
        self.clear_preview()
        self.window._transform_session = None
        for control, enabled in self.controls: control.setEnabled(enabled)
        self.window.ui.section_panel._make_widget()
        self.window.update_history_actions()
        self.dialog.deleteLater()
