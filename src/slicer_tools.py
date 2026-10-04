"""Slicer primitives, copies and reversible world-coordinate transforms."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtWidgets import QCheckBox


TOOL_NAMES = ("Создать", "Дублировать", "Пакетное дублирование", "Перемещать", "Вращать", "Масштабировать", "Отзеркалить")


def transform_matrix(operation, values, center):
    matrix = np.eye(4)
    values, center = np.asarray(values, dtype=float), np.asarray(center, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(center).all():
        raise ValueError("Координаты должны быть конечными числами.")
    if operation == "Перемещать":
        matrix[:3, 3] = values
        return matrix
    if operation == "Вращать":
        matrix[:3, :3] = Rotation.from_euler('xyz', values, degrees=True).as_matrix()
    elif operation == "Масштабировать":
        if np.any(values <= 0): raise ValueError("Масштаб должен быть больше нуля.")
        matrix[:3, :3] = np.diag(values)
    elif operation == "Отзеркалить":
        normal = values / np.linalg.norm(values)
        if not np.isfinite(normal).all(): raise ValueError("Выберите ось отражения.")
        matrix[:3, :3] -= 2 * np.outer(normal, normal)
    else:
        raise ValueError("Неизвестное преобразование.")
    matrix[:3, 3] = center - matrix[:3, :3] @ center
    return matrix


def create_primitive(kind, dimensions, center, mesh_settings=None):
    from primitive_geometry import build_primitive
    if isinstance(dimensions,dict): return build_primitive(kind,dimensions,center,mesh_settings)
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.shape!=(3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("Размеры должны быть больше нуля.")
    if kind=='Параллелепипед': values = dict(x=dimensions[0],y=dimensions[1],h=dimensions[2])
    elif kind=='Цилиндр': values = dict(r=dimensions[0]/2,h=dimensions[2])
    elif kind=='Сфера': values = dict(r=dimensions[0]/2)
    else: raise ValueError("Неизвестный примитив.")
    return build_primitive(kind,values,center,mesh_settings or dict(mode='segments',segments=96))


class SlicerToolsMixin:
    def init_slicer_tools(self):
        for name in TOOL_NAMES:
            self.ui.ribbon_btns[name].clicked.connect(lambda checked=False, operation=name: self.run_slicer_tool(operation))
        for name, button in self.ui.position_buttons.items():
            button.clicked.connect(lambda checked=False, operation=name: self.run_slicer_tool(operation))

    def selected_slicer_rows(self):
        from PySide6.QtWidgets import QCheckBox
        return [row for row in range(len(self.slicer_parts)) if not self.ui.tbl_parts.isRowHidden(row)
                and self.ui.tbl_parts.cellWidget(row, 1).findChild(QCheckBox).isChecked()]

    def matrices_for(self, operation, params, rows):
        if params.get('advanced'):
            from transform_math import matrices
            return dict(zip(rows, matrices([self.slicer_parts[row]['mesh'] for row in rows], operation, params)))
        bounds = np.array([self.slicer_parts[row]['mesh'].bounds for row in rows])
        center = (bounds[:, 0].min(axis=0) + bounds[:, 1].max(axis=0)) / 2
        return {row: transform_matrix(operation, params['values'],
                    self.slicer_parts[row]['mesh'].bounds.mean(axis=0) if params['pivot'] == 1 else
                    np.zeros(3) if params['pivot'] == 2 else center) for row in rows}

    def preview_transforms(self, matrices):
        from vtkmodules.vtkCommonMath import vtkMatrix4x4
        for row, matrix in matrices.items():
            vtk_matrix = vtkMatrix4x4()
            vtk_matrix.DeepCopy(np.asarray(matrix).ravel())
            for suffix in ('', '__bbox'):
                actor = self.ui.slicer_plotter.actors.get(self.slicer_parts[row]['actor_name'] + suffix)
                if actor is not None: actor.SetUserMatrix(vtk_matrix)
            for group in self.slicer_parts[row].get('supports', []):
                actor = self.ui.slicer_plotter.actors.get('part_support_' + group['id'])
                if actor is not None: actor.SetUserMatrix(vtk_matrix)
        self.ui.slicer_plotter.reset_camera_clipping_range()
        if hasattr(self, 'display_tools'): self.display_tools.on_scene_changed()
        self.ui.slicer_plotter.render()

    def run_slicer_tool(self, operation):
        if self._busy(): return
        if hasattr(self, 'workspace_tools'):
            self.workspace_tools.set_mode('part')
            self.workspace_tools.clear()
        rows = self.selected_slicer_rows()
        if operation != "Создать" and not rows:
            self.log("Отметьте детали в столбце «Выбранные» текущей сцены.")
            return
        self.flush_history()
        if operation in TOOL_NAMES[3:]:
            from transform_session import TransformSession
            TransformSession(self, operation, rows)
            return
        if operation in TOOL_NAMES[1:3]:
            from duplicate_session import DuplicateSession
            DuplicateSession(self,operation,rows)
            return
        from primitive_dialog import PrimitiveDialog
        dialog = PrimitiveDialog(self)
        if dialog.exec():
            try: self.apply_slicer_tool(operation, dialog.parameters(), rows)
            except Exception as exc: self.log(f"Не удалось выполнить «{operation}»: {exc}")
        dialog.deleteLater()

    def replace_slicer_mesh(self, row, mesh):
        if hasattr(self, 'workspace_tools'): self.workspace_tools.invalidate(row)
        part = self.slicer_parts[row]
        actor = self.ui.slicer_plotter.actors[part['actor_name']]
        part['mesh'] = mesh
        part['mesh_pv'] = self.trimesh_to_pyvista(mesh)
        actor.mapper.dataset = part['mesh_pv']
        actor.SetUserMatrix(None)
        bbox = self.ui.slicer_plotter.actors.get(part['actor_name'] + '__bbox')
        if bbox is not None:
            bbox.mapper.dataset = part['mesh_pv'].outline()
            bbox.SetUserMatrix(None)

    def apply_slicer_tool(self, operation, params, rows=None):
        if self._busy(): return
        rows = self.selected_slicer_rows() if rows is None else rows
        if operation != "Создать" and not rows: raise ValueError("Нет выбранных деталей.")
        self.flush_history()
        before = self.capture_project()
        prepared = []
        if operation == "Создать":
            mesh = create_primitive(params['kind'], params.get('primitive',params.get('values')), params['center'],params.get('mesh_settings'))
            index = self.ui.scene_tabs.currentIndex()
            platforms = [p for p in self.platforms if p['is_default']]
            platform = platforms[index - 1]['name'] if 0 < index <= len(platforms) else None
            prepared.append((mesh, f"{params['kind']}.stl", platform, {}))
        elif operation in TOOL_NAMES[1:3]:
            from duplicate_layout import duplicate_plan
            from part_supports import combined_mesh
            offsets = duplicate_plan([combined_mesh(self.slicer_parts[row]) for row in rows],params,operation)['offsets']
            for row in rows:
                source = self.slicer_parts[row]
                style = self._style_for(self.ui.tbl_parts, row)
                for index, offset in enumerate(offsets, 1):
                    mesh = source['mesh'].copy()
                    from part_supports import transformed
                    matrix = np.eye(4)
                    matrix[:3, 3] = offset
                    from cad_state import apply_cad_transform
                    apply_cad_transform(mesh, matrix)
                    children = transformed(source.get('supports', []), matrix)
                    from uuid import uuid4
                    for child in children: child['id'] = str(uuid4())
                    prepared.append((mesh, f"{Path(source['filename']).stem}_копия_{index}.stl", source.get('platform'), dict(style, _supports=children)))
        else:
            for row, matrix in self.matrices_for(operation, params, rows).items():
                mesh = self.slicer_parts[row]['mesh'].copy()
                from cad_state import apply_cad_transform
                apply_cad_transform(mesh, matrix)
                from part_supports import transformed
                children = transformed(self.slicer_parts[row].get('supports', []), matrix)
                if params.get('create_copy'):
                    part = self.slicer_parts[row]
                    from uuid import uuid4
                    for child in children: child['id'] = str(uuid4())
                    style = dict(self._style_for(self.ui.tbl_parts, row), _supports=children)
                    prepared.append((mesh, f"{Path(part['filename']).stem}_копия.stl", part.get('platform'), style))
                else:
                    prepared.append((row, mesh, children))
        try:
            if operation in TOOL_NAMES[:3] or params.get("create_copy"):
                if operation != "Создать":
                    for row in rows:
                        self.ui.tbl_parts.cellWidget(row, 1).findChild(QCheckBox).setChecked(False)
                self._slicer_batch = True
                self.ui.tbl_parts.setUpdatesEnabled(False)
                try:
                    used = {part['filename'].casefold() for part in self.slicer_parts}
                    for mesh, name, platform, style in prepared:
                        base, suffix, counter = Path(name).stem, Path(name).suffix, 2
                        while name.casefold() in used:
                            name = f"{base}_{counter}{suffix}"
                            counter += 1
                        used.add(name.casefold())
                        self._append_slicer_part(mesh, name, platform, style)
                finally:
                    self._slicer_batch = False
                    self.ui.tbl_parts.setUpdatesEnabled(True)
            else:
                from part_supports import remove_actors
                for row, mesh, children in prepared:
                    remove_actors(self, self.slicer_parts[row].get('supports', []))
                    self.slicer_parts[row]['supports'] = children
                    self.replace_slicer_mesh(row, mesh)
            self.refresh_scene_visibility()
            self.update_info_combobox()
            self.update_parts_table_filter(self.ui.scene_tabs.currentIndex())
            self.mark_dirty()
            self.flush_history(operation)
        except Exception:
            self.restore_project(before)
            raise
