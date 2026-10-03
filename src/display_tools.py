"""Scene display tools; annotations and display datasets never modify project meshes."""
from pathlib import Path
import json
import math
import os

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QRect, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QLabel, QMenu, QPlainTextEdit, QToolButton, QVBoxLayout)
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkRenderingCore import vtkBillboardTextActor3D

from display_ribbon import DEFAULTS, DISPLAY_COMMANDS, TOGGLES, STATISTICS
from display_settings import DIALOG_STYLE
from part_supports import combined_mesh, support_mesh


VIEWS = {'Изометрия': (None, 1), 'Спереди (−Y)': (1, -1), 'Сзади (+Y)': (1, 1),
         'Сверху (+Z)': (2, 1), 'Снизу (−Z)': (2, -1), 'Слева (−X)': (0, -1), 'Справа (+X)': (0, 1)}
PREFIX = 'display_overlay_'


def scene_font():
    """VTK's embedded Latin font omits Cyrillic; use an installed Unicode font."""
    candidates = [Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/segoeui.ttf',
                  Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts/arial.ttf',
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    return next((str(path) for path in candidates if path.is_file()), None)


def solid_volume(mesh):
    """Return reliable solid volume in mm³; an open sheet has no implied volume."""
    if not mesh.is_watertight or not mesh.is_winding_consistent: return None
    value = abs(float(mesh.volume))
    return value if np.isfinite(value) and value > 1e-12 else None


def scene_statistics(parts):
    """Component sums, including supports separately, without treating overlaps as union."""
    rows, total, unknown = [], 0., 0
    for part in parts:
        volume = solid_volume(part['mesh'])
        support_values = [solid_volume(support_mesh(group)) for group in part.get('supports', []) if len(group.get('faces', []))]
        support_total = sum(value for value in support_values if value is not None)
        missing = int(volume is None) + sum(value is None for value in support_values)
        total += (volume or 0) + support_total; unknown += missing
        rows.append(dict(name=part.get('filename', 'Деталь'), volume=volume, supports=support_total, unknown=missing))
    return dict(rows=rows, total_mm3=total, unknown=unknown)


def material_estimate(volume_mm3, density_g_cm3, price_per_kg):
    mass_g = float(volume_mm3) / 1000 * float(density_g_cm3)
    return mass_g, mass_g / 1000 * float(price_per_kg)


def _bounds_overlap(first, second):
    return bool(np.all(np.minimum(first[1], second[1]) - np.maximum(first[0], second[0]) > 1e-7))


def platform_bounds(platform):
    dimensions = np.asarray(platform.get('dim', []), dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('У платформы некорректные габариты.')
    return np.array([[-dimensions[0] / 2, -dimensions[1] / 2, 0],
                     [dimensions[0] / 2, dimensions[1] / 2, dimensions[2]]])


def geometric_checks(parts, platform=None):
    """Conservative broad-phase checks, explicitly not a build simulation."""
    warnings, flagged, outside = [], set(), set()
    bounds = [combined_mesh(part).bounds for part in parts]
    build = platform_bounds(platform) if platform else None
    for index, (part, bounds_i) in enumerate(zip(parts, bounds)):
        name = part.get('filename', f'Деталь {index + 1}')
        if solid_volume(part['mesh']) is None:
            warnings.append(f'{name}: открытая, несогласованная или вырожденная сетка; объём не определён.'); flagged.add(index)
        if build is not None and (np.any(bounds_i[0] < build[0] - 1e-6) or np.any(bounds_i[1] > build[1] + 1e-6)):
            warnings.append(f'{name}: деталь или её поддержки выходят за габариты платформы.')
            flagged.add(index); outside.add(index)
        if platform and platform.get('use_zones'):
            for zone_index, zone in enumerate(platform.get('zones', [])):
                r = float(zone.get('r', 5)); x, y = float(zone.get('x', 0)), float(zone.get('y', 0))
                z0, z1 = (0., float(platform['dim'][2])) if zone.get('full_h') else sorted((float(zone.get('zmin', 0)), float(zone.get('zmax', 0))))
                if abs(z1 - z0) < .001: z0 -= .0005; z1 += .0005
                box = np.array([[x - r, y - r, z0], [x + r, y + r, z1]])
                if _bounds_overlap(bounds_i, box):
                    warnings.append(f'{name}: габарит пересекает габарит запретной зоны №{zone_index + 1}; проверьте точную форму.')
                    flagged.add(index)
    if len(parts) <= 300:
        for first in range(len(parts)):
            for second in range(first + 1, len(parts)):
                if _bounds_overlap(bounds[first], bounds[second]):
                    warnings.append(f"Возможное пересечение габаритов: {parts[first].get('filename', first + 1)} и {parts[second].get('filename', second + 1)}.")
                    flagged.update((first, second))
                    if len(warnings) >= 500: break
            if len(warnings) >= 500:
                warnings.append('Показаны первые 500 замечаний. Проверяйте меньшие группы деталей.'); break
    else: warnings.append('Попарные габариты не проверены: для этой проверки оставьте не более 300 деталей.')
    return dict(warnings=warnings, flagged=flagged, outside=outside)


class DisplayTools(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window, self.plotter = window, None
        self.state = dict(DEFAULTS)
        self.overlays, self.owners, self.datasets, self._base_styles = {}, {}, {}, {}
        self._signature, self._updating, self.rebuild_count = None, False, 0
        self._report_dialog = None
        self._density, self._price = 1., 0.
        self._cube_state = None
        self._font_file = scene_font()
        self.buttons = window.ui.display_buttons
        menu = QMenu(self.buttons['view'])
        for title, (axis, sign) in VIEWS.items():
            menu.addAction(title).triggered.connect(lambda checked=False, a=axis, s=sign: self.orient(a, s))
        self.buttons['view'].setMenu(menu)
        self.buttons['view'].setPopupMode(QToolButton.MenuButtonPopup)
        menu = QMenu(self.buttons['material_cost'])
        menu.addAction('Плотность и цена материала…').triggered.connect(lambda: self.show_report('material_cost'))
        self.buttons['material_cost'].setMenu(menu)
        self.buttons['material_cost'].setPopupMode(QToolButton.MenuButtonPopup)
        from display_statistics import LiveStatistics
        self.statistics = LiveStatistics(self)
        for operation, button in self.buttons.items():
            button.clicked.connect(lambda checked=False, op=operation: self.trigger(op))
        self.attach(getattr(window.ui, 'slicer_plotter', None))

    def attach(self, plotter):
        if plotter is self.plotter: return
        self.plotter = plotter
        self.overlays.clear(); self.owners.clear(); self.datasets.clear(); self._base_styles.clear()
        self._signature, self._cube_state = None, None
        if plotter is not None: self.on_scene_changed()

    def notify(self, message):
        self.window.log(message)
        self.window.ui.status_label.setText(message)

    def active_platform(self):
        placement = getattr(self.window, '_placement_session', None)
        name = getattr(placement, 'platform_preview_name', None)
        if name:
            return next((platform for platform in self.window.platforms if platform['name'] == name), None)
        active = [platform for platform in self.window.platforms if platform.get('is_default')]
        index = self.window.ui.scene_tabs.currentIndex() - 1
        return active[index] if 0 <= index < len(active) else None

    def visible_rows(self):
        result, platform = [], self.active_platform()
        placement = getattr(self.window, '_placement_session', None)
        preview_rows = set(placement.rows) if placement and getattr(placement, 'platform_preview_name', None) else set()
        for row, part in enumerate(self.window.slicer_parts):
            widget = self.window.ui.tbl_parts.cellWidget(row, self.window.COL_VISIBLE)
            checked = widget.findChild(QCheckBox).isChecked() if widget else False
            if row in preview_rows or (checked and (platform is None or part.get('platform') == platform['name'])):
                result.append(row)
        return result

    def trigger(self, operation):
        self.attach(getattr(self.window.ui, 'slicer_plotter', None))
        if self.plotter is None:
            self.notify('Откройте рабочую сцену слайсера.'); return
        try:
            if operation == 'view': self.orient(); return
            if operation in STATISTICS:
                self.state[operation] = self.buttons[operation].isChecked()
                self.statistics.update()
                self.plotter.render()
                return
            if operation in TOGGLES:
                self.state[operation] = self.buttons[operation].isChecked()
                if operation in {'texture', 'triangle_colors'} and self.state[operation]:
                    other = 'triangle_colors' if operation == 'texture' else 'texture'
                    self.state[other] = False; self.buttons[other].setChecked(False)
                if operation == 'texture' and self.state[operation]:
                    available = [row for row in self.visible_rows() if self._visual(self.window.slicer_parts[row]['mesh']) is not None]
                    if not available:
                        self.state[operation] = False; self.buttons[operation].setChecked(False)
                        self.notify('У видимых моделей нет сохранённых цветов или UV-текстуры. Импортируйте модель с цветами граней/вершин либо UV и изображением текстуры.'); return
                self.on_scene_changed()
                if operation in {'outside', 'build_risk'} and self.state[operation]:
                    if operation == 'outside' and self.active_platform() is None:
                        self.notify('Выберите вкладку платформы: на модельной сцене нет активной камеры построения.')
                    elif operation == 'build_risk': self.show_report(operation)
                return
            if operation == 'export_png': self.export_png()
            elif operation == 'clipboard': self.copy_image()
            elif operation == 'print': self.print_image()
        except Exception as exc:
            self.notify(f'{DISPLAY_COMMANDS[operation]}: {exc}')

    def orient(self, axis=None, sign=1):
        if self.plotter is None: return
        cube = getattr(getattr(self.window, 'workspace_tools', None), 'cube', None)
        if cube is not None:
            cube.orient(axis, sign); return
        camera = getattr(self.plotter, 'camera', None)
        if camera is None: return
        focus = np.asarray(camera.focal_point)
        distance = max(float(np.linalg.norm(np.asarray(camera.position) - focus)), 1.)
        direction = np.ones(3) / np.sqrt(3) if axis is None else np.eye(3)[axis] * sign
        camera.position = focus + distance * direction
        camera.up = (0, 1, 0) if axis == 2 else (0, 0, 1)
        self.plotter.reset_camera_clipping_range(); self.plotter.render()

    @staticmethod
    def _visual(mesh):
        from texture_geometry import appearance
        if appearance(mesh) is not None: return 'appearance', None
        visual = mesh.visual
        kind = getattr(visual, 'kind', None)
        if kind in {'face', 'vertex'} and visual.defined:
            return kind, np.asarray(visual.face_colors if kind == 'face' else visual.vertex_colors)
        if kind == 'texture':
            uv = getattr(visual, 'uv', None)
            material = getattr(visual, 'material', None)
            image = getattr(material, 'image', None)
            if image is None: image = getattr(material, 'baseColorTexture', None)
            if uv is not None and len(uv) == len(mesh.vertices) and image is not None:
                return 'texture', (np.asarray(uv), np.asarray(image))
        return None

    def _display_dataset(self, row, part):
        source, mesh = part['mesh_pv'], part['mesh']
        key = (id(source), id(mesh), self.state['smooth'], self.state['texture'], self.state['triangle_colors'])
        record = self.datasets.get(row)
        if record and record['key'] == key: return record
        data, texture, colors = source, None, None
        if any((self.state['smooth'], self.state['texture'], self.state['triangle_colors'])):
            data = source.copy(deep=True)
            if self.state['smooth']:
                normals = vtkPolyDataNormals(); normals.SetInputData(data)
                normals.SplittingOff(); normals.ConsistencyOff(); normals.AutoOrientNormalsOff()
                normals.ComputePointNormalsOn(); normals.ComputeCellNormalsOff(); normals.Update()
                data = pv.wrap(normals.GetOutput()).copy(deep=True)
            if self.state['triangle_colors']:
                palette = np.array([[94, 163, 210], [220, 136, 100], [160, 194, 102],
                                    [172, 139, 198], [220, 195, 110], [105, 188, 175]], dtype=np.uint8)
                data.cell_data['_display_colors'] = palette[np.arange(data.n_cells) % len(palette)]
                colors = 'face'
            elif self.state['texture']:
                visual = self._visual(mesh)
                if visual is not None:
                    kind, values = visual
                    if kind == 'appearance':
                        from texture_render import display_data
                        data, texture, colors = display_data(mesh, data)
                    elif kind == 'texture':
                        data.active_texture_coordinates = values[0]
                        texture = pv.Texture(values[1])
                    else:
                        (data.cell_data if kind == 'face' else data.point_data)['_display_colors'] = values
                        colors = kind
        record = dict(key=key, data=data, texture=texture, colors=colors)
        self.datasets[row] = record
        return record

    def _apply_styles(self, visible):
        active_rows = set(range(len(self.window.slicer_parts)))
        for row in set(self.datasets) - active_rows: self.datasets.pop(row, None)
        for row, part in enumerate(self.window.slicer_parts):
            actor = self.plotter.actors.get(part['actor_name'])
            if actor is None: continue
            key = (row, id(actor), id(part['mesh_pv']))
            saved = self._base_styles.get(key)
            if saved is None:
                saved = dict(scalar=actor.mapper.GetScalarVisibility(), mode=actor.mapper.GetScalarMode(),
                             color=actor.mapper.GetColorMode(), name=actor.mapper.GetArrayName())
                self._base_styles[key] = saved
            record = self._display_dataset(row, part)
            actor.mapper.dataset = record['data']
            # A visibility toggle must also clear an atlas retained by a previous
            # render/restore; it is never part of the untextured actor style.
            actor.SetTexture(record['texture'] if self.state['texture'] else None)
            if record['colors']:
                actor.mapper.SetScalarVisibility(True); actor.mapper.SetColorModeToDirectScalars()
                if record['colors'] == 'face': actor.mapper.SetScalarModeToUseCellFieldData()
                else: actor.mapper.SetScalarModeToUsePointFieldData()
                actor.mapper.SelectColorArray('_display_colors')
            else:
                actor.mapper.SetScalarVisibility(saved['scalar']); actor.mapper.SetScalarMode(saved['mode'])
                actor.mapper.SetColorMode(saved['color'])
                if saved['name']: actor.mapper.SelectColorArray(saved['name'])
            if self.state['smooth']: actor.prop.SetInterpolationToPhong()
            elif part.get('last_visible_mode') == 'triangles': actor.prop.SetInterpolationToFlat()
            else: actor.prop.SetInterpolationToGouraud()
            actor.SetVisibility(row in visible and not self.state['simplified'] and part.get('last_visible_mode') != 'bbox')
            bbox = self.plotter.actors.get(part['actor_name'] + '__bbox')
            if bbox is not None: bbox.SetVisibility(row in visible and not self.state['simplified'] and part.get('last_visible_mode') == 'bbox')
        live = {(row, id(self.plotter.actors.get(part['actor_name'])), id(part['mesh_pv'])) for row, part in enumerate(self.window.slicer_parts)}
        self._base_styles = {key: value for key, value in self._base_styles.items() if key in live}
        from part_supports import sync_actors
        sync_actors(self.window)

    def on_scene_changed(self):
        plotter = getattr(self.window.ui, 'slicer_plotter', None)
        if self._updating or plotter is None: return
        if plotter is not self.plotter: self.attach(plotter); return
        self._updating = True
        try:
            rows, platform = self.visible_rows(), self.active_platform()
            self._apply_styles(set(rows))
            for name, actor in self.plotter.actors.items():
                if name.startswith('plat_zone_'): actor.SetVisibility(bool(platform) and self.state['zones'])
            cube = getattr(getattr(self.window, 'workspace_tools', None), 'cube', None)
            cube_state = (id(cube), self.state['coordinates'])
            if cube is not None and cube_state != self._cube_state:
                cube.setVisible(self.state['coordinates']); self._cube_state = cube_state
            parts = self.window.slicer_parts
            signature = (tuple((row, id(parts[row]['mesh']), id(parts[row]['mesh_pv']), parts[row].get('filename'),
                                parts[row].get('source_path'), parts[row].get('path'),
                                tuple((group['id'], id(group['vertices']), id(group['faces'])) for group in parts[row].get('supports', []))) for row in rows),
                         tuple(sorted((key, value) for key, value in self.state.items() if key not in STATISTICS)), json.dumps(platform, sort_keys=True, default=str))
            missing = any(name not in self.plotter.actors for name in self.overlays)
            if signature != self._signature or missing:
                self.clear_overlays(); self._build_overlays(rows, platform)
                self._signature = signature; self.rebuild_count += 1
            self._sync_overlays()
        finally:
            self._updating = False
        self.plotter.render()
        self.statistics.request()
        workspace = getattr(self.window, 'workspace_tools', None)
        if hasattr(workspace, 'refresh_part_selection'): workspace.refresh_part_selection()

    def clear_overlays(self):
        for name in self.overlays:
            self.plotter.remove_actor(name, render=False)
        self.overlays.clear(); self.owners.clear()

    def _mesh_actor(self, name, data, *, row=None, **kwargs):
        name = PREFIX + name
        actor = self.plotter.add_mesh(data, name=name, pickable=False, reset_camera=False, render=False, **kwargs)
        actor.SetPickable(False)
        self.overlays[name], self.owners[name] = actor, row
        return actor

    def _label(self, name, text, point, *, row=None, color=(.10, .20, .28), size=11):
        actor = vtkBillboardTextActor3D(); actor.SetInput(str(text)); actor.SetPosition(*np.asarray(point, dtype=float))
        actor.SetPickable(False)
        prop = actor.GetTextProperty(); prop.SetFontSize(size); prop.SetColor(*color)
        if self._font_file:
            from vtkmodules.vtkCommonCore import VTK_FONT_FILE
            prop.SetFontFamily(VTK_FONT_FILE)
            prop.SetFontFile(self._font_file)
        prop.SetBackgroundColor(.96, .97, .98); prop.SetBackgroundOpacity(.82)
        name = PREFIX + name
        if hasattr(self.plotter, 'add_actor'):
            self.plotter.add_actor(actor, name=name, pickable=False, reset_camera=False, render=False)
        else:
            # Test adapters store real VTK actors without constructing a renderer.
            self.plotter.actors[name] = actor
        self.overlays[name], self.owners[name] = actor, row
        return actor

    def _sync_overlays(self):
        for name, actor in self.overlays.items():
            row = self.owners[name]
            if row is None: continue
            source = self.plotter.actors.get(self.window.slicer_parts[row]['actor_name'])
            actor.SetUserMatrix(source.GetUserMatrix() if source is not None else None)
            if hasattr(actor, 'GetMapper'):
                mapper = actor.GetMapper()
                if mapper is not None and hasattr(mapper, 'RemoveAllClippingPlanes'):
                    mapper.RemoveAllClippingPlanes()
                    for plane in getattr(self.window.ui.section_panel, '_planes', []): mapper.AddClippingPlane(plane)

    def _scene_bounds(self, rows, platform):
        if platform: return platform_bounds(platform)
        if rows:
            bounds = np.array([self.window.slicer_parts[row]['mesh'].bounds for row in rows])
            return np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])
        return np.array([[-50., -50., 0.], [50., 50., 100.]])

    @staticmethod
    def _nice_step(length):
        raw = max(float(length) / 10, .00001)
        base = 10 ** math.floor(math.log10(raw))
        return next(value * base for value in (1, 2, 5, 10) if value * base >= raw)

    def _build_overlays(self, rows, platform):
        bounds = self._scene_bounds(rows, platform)
        if self.state['grid'] or self.state['ruler']:
            step = self._nice_step(max(bounds[1, :2] - bounds[0, :2]))
            xs = np.arange(math.ceil(bounds[0, 0] / step), math.floor(bounds[1, 0] / step) + 1) * step
            ys = np.arange(math.ceil(bounds[0, 1] / step), math.floor(bounds[1, 1] / step) + 1) * step
            if self.state['grid']:
                segments = [[[x, bounds[0, 1], .02], [x, bounds[1, 1], .02]] for x in xs]
                segments += [[[bounds[0, 0], y, .02], [bounds[1, 0], y, .02]] for y in ys]
                if segments: self._mesh_actor('grid', pv.line_segments_from_points(np.reshape(segments, (-1, 3))), color='#7996a2', opacity=.55, line_width=1)
                self._label('grid_step', f'Сетка: {step:g} мм', [bounds[0, 0], bounds[1, 1], .02])
            if self.state['ruler']:
                points = [[bounds[0, 0], bounds[0, 1], .04], [bounds[1, 0], bounds[0, 1], .04],
                          [bounds[0, 0], bounds[0, 1], .04], [bounds[0, 0], bounds[1, 1], .04]]
                tick = step * .08
                for i, x in enumerate(xs):
                    points += [[x, bounds[0, 1] - tick, .04], [x, bounds[0, 1] + tick, .04]]
                    self._label(f'ruler_x_{i}', f'X {x:g}', [x, bounds[0, 1] - tick * 2, .04], size=9)
                for i, y in enumerate(ys):
                    points += [[bounds[0, 0] - tick, y, .04], [bounds[0, 0] + tick, y, .04]]
                    self._label(f'ruler_y_{i}', f'Y {y:g}', [bounds[0, 0] - tick * 2, y, .04], size=9)
                self._mesh_actor('rulers', pv.line_segments_from_points(points), color='#345367', line_width=2)
        if self.state['origin']:
            length = max(float(np.max(bounds[1] - bounds[0])) * .15, .01)
            for axis, color in enumerate(('#e35b54', '#78b83e', '#458ed6')):
                endpoint = np.eye(3)[axis] * length
                self._mesh_actor('axis_' + str(axis), pv.Line(np.zeros(3), endpoint), color=color, line_width=3)
                self._label('axis_text_' + str(axis), 'XYZ'[axis], endpoint)
        if self.state['bbox'] and rows:
            all_bounds = self._scene_bounds(rows, None)
            self._mesh_actor('overall_bbox', pv.Box(np.column_stack(all_bounds).ravel()).outline(), color='#b47f26', line_width=2)
        checks = geometric_checks([self.window.slicer_parts[row] for row in rows], platform) if self.state['outside'] or self.state['build_risk'] else None
        for index, row in enumerate(rows):
            part = self.window.slicer_parts[row]; mesh = part['mesh']; lo, hi = mesh.bounds
            center = mesh.bounds.mean(axis=0)
            if self.state['simplified']:
                self._mesh_actor(f'simple_{row}', part['mesh_pv'].outline(), row=row, color='#678899', line_width=2)
            if self.state['dimensions']:
                extents = hi - lo
                self._label(f'dimensions_{row}', ' × '.join(f'{value:.3f}' for value in extents) + ' мм (XYZ)',
                            [hi[0], lo[1], lo[2]], row=row)
            if self.state['center_mass']:
                valid = solid_volume(mesh) is not None
                point = mesh.center_mass if valid else center
                radius = max(float(np.max(hi - lo)) * .01, .001)
                self._mesh_actor(f'center_{row}', pv.Sphere(radius=radius, center=point, theta_resolution=8, phi_resolution=8), row=row, color='#74a73d')
                self._label(f'center_label_{row}', 'Центр масс' if valid else 'Центр габаритов (открытая сетка)', point, row=row, size=10)
            labels = []
            if self.state['part_number']: labels.append(f'№{row + 1}')
            if self.state['part_name']: labels.append(str(part.get('filename', 'Деталь')))
            if self.state['part_path']:
                path = part.get('source_path') or part.get('path') or mesh.metadata.get('source_path') or mesh.metadata.get('file_path')
                filename = str(part.get('filename', ''))
                if not path and Path(filename).is_absolute(): path = filename
                labels.append(str(path) if path else 'Исходный путь не сохранён')
            if labels: self._label(f'part_label_{row}', '\n'.join(labels), [center[0], center[1], hi[2]], row=row)
            if self.state['overhang'] and not self.state['simplified']:
                ids = np.flatnonzero((mesh.face_normals[:, 2] < -math.cos(math.radians(45))) & (mesh.triangles_center[:, 2] > .01))
                if len(ids):
                    data = part['mesh_pv'].extract_cells(ids).extract_surface(algorithm='dataset_surface')
                    actor = self._mesh_actor(f'overhang_{row}', data, row=row, color='#ef973c', opacity=.85)
                    actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
                    actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -2)
            if checks and (index in checks['outside'] and self.state['outside'] or index in checks['flagged'] and self.state['build_risk']):
                combined = combined_mesh(part)
                box = pv.Box(np.column_stack(combined.bounds).ravel()).outline()
                self._mesh_actor(f'warning_{row}', box, row=row, color='#df584b', line_width=3)

    def _stat_parts(self):
        rows = self.window.selected_slicer_rows()
        return [self.window.slicer_parts[row] for row in rows]

    def _report_text(self, operation):
        if operation == 'build_risk':
            parts = [self.window.slicer_parts[row] for row in self.visible_rows()]
            result = geometric_checks(parts, self.active_platform())
            header = ('Геометрические проверки видимых деталей и их поддержек.\n'
                      'Пересечение габаритов означает только возможный конфликт.\n'
                      'Это не симуляция печати, прочности, температур или деформаций.\n\n')
            if self.active_platform() is None: header += 'Платформа не выбрана: её границы и запретные зоны не проверены.\n\n'
            return header + ('\n'.join(result['warnings']) if result['warnings'] else 'Перечисленные геометрические проверки замечаний не выявили.')
        statistics = scene_statistics(self._stat_parts())
        lines = ['Выбранные детали и их поддержки. При пустом выборе — ноль. Единицы: мм³.']
        for row in statistics['rows']:
            volume = f"{row['volume']:.3f}" if row['volume'] is not None else 'не определён (незамкнутая сетка)'
            lines.append(f"{row['name']}: {volume}; поддержки: {row['supports']:.3f}")
        lines += ['', f"Сумма определённых объёмов: {statistics['total_mm3']:.3f} мм³.",
                  'Объёмы тел и поддержек складываются; перекрытия не вычитаются.']
        if statistics['unknown']: lines.append(f"Неопределённых объёмов: {statistics['unknown']}. Сумма неполная.")
        if operation == 'material_cost':
            mass, cost = material_estimate(statistics['total_mm3'], self._density, self._price)
            lines += ['', f'Плотность: {self._density:g} г/см³; цена: {self._price:g} за кг.',
                      f'Расчётная масса: {mass:.3f} г. Стоимость материала: {cost:.2f}.',
                      'Оценка для сплошного однородного материала. Заполнение, потери и работа оборудования не учтены.']
        if operation == 'packing_density':
            platform = self.active_platform()
            if platform is None: lines.append('\nВыберите вкладку платформы для расчёта доли объёма камеры.')
            else:
                volume = float(np.prod(platform_bounds(platform)[1] - platform_bounds(platform)[0]))
                fraction = statistics['total_mm3'] / volume * 100
                lines += ['', f"Камера: {platform['name']}, {volume:.3f} мм³.",
                          f'Суммарный объём / объём камеры: {fraction:.3f} %.',
                          'Показатель не проверяет размещение внутри камеры и не оценивает плотность порошка.']
        return '\n'.join(lines)

    def show_report(self, operation):
        if self._report_dialog is not None:
            self._report_dialog.close(); self._report_dialog.deleteLater()
        dialog = QDialog(self.window); dialog.setWindowTitle(DISPLAY_COMMANDS[operation])
        dialog.setStyleSheet(DIALOG_STYLE + 'QPlainTextEdit, QDoubleSpinBox {background:#262626;color:#e0e0e0;padding:6px;}')
        dialog.resize(700, 480); dialog.setModal(False)
        layout = QVBoxLayout(dialog)
        report = QPlainTextEdit(); report.setReadOnly(True)
        if operation == 'material_cost':
            form = QFormLayout()
            density = QDoubleSpinBox(); density.setDecimals(5); density.setRange(.00001, 100); density.setValue(self._density); density.setSuffix(' г/см³')
            price = QDoubleSpinBox(); price.setDecimals(2); price.setRange(0, 1e9); price.setValue(self._price)
            form.addRow('Плотность материала:', density); form.addRow('Цена за кг (ваша валюта):', price)
            layout.addLayout(form)
            def update():
                self._density, self._price = density.value(), price.value()
                report.setPlainText(self._report_text(operation))
                self.statistics.request()
            density.valueChanged.connect(lambda *_: update()); price.valueChanged.connect(lambda *_: update())
            dialog.density, dialog.price = density, price
        report.setPlainText(self._report_text(operation)); layout.addWidget(report)
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.button(QDialogButtonBox.Close).setText('Закрыть')
        buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        dialog.report = report; self._report_dialog = dialog; dialog.show()

    def capture_image(self):
        if not hasattr(self.plotter, 'screenshot'): raise ValueError('Снимок доступен после открытия 3D-сцены.')
        data = np.ascontiguousarray(self.plotter.screenshot(return_img=True, transparent_background=False))
        if data.ndim != 3 or data.shape[2] not in (3, 4): raise ValueError('Не удалось получить изображение сцены.')
        data = data.astype(np.uint8, copy=False)
        image_format = QImage.Format_RGB888 if data.shape[2] == 3 else QImage.Format_RGBA8888
        return QImage(data.data, data.shape[1], data.shape[0], data.strides[0], image_format).copy()

    def export_png(self):
        filename, _ = QFileDialog.getSaveFileName(self.window, 'Сохранить изображение сцены', 'scene.png', 'PNG (*.png)')
        if not filename: return
        if not filename.lower().endswith('.png'): filename += '.png'
        if not self.capture_image().save(filename, 'PNG'): raise ValueError('Не удалось записать PNG. Проверьте путь и доступ к папке.')
        self.notify('Изображение сохранено: ' + filename)

    def copy_image(self):
        QApplication.clipboard().setImage(self.capture_image())
        self.notify('Изображение сцены скопировано в буфер обмена.')

    def print_image(self):
        image = self.capture_image()
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self.window); dialog.setWindowTitle('Печать изображения сцены')
        if dialog.exec() != QDialog.Accepted: return
        painter = QPainter(printer)
        if not painter.isActive(): raise ValueError('Принтер недоступен.')
        try:
            viewport = painter.viewport()
            size = image.size(); size.scale(viewport.size(), Qt.KeepAspectRatio)
            target = QRect(viewport.x() + (viewport.width() - size.width()) // 2,
                           viewport.y() + (viewport.height() - size.height()) // 2, size.width(), size.height())
            painter.drawImage(target, image)
        finally: painter.end()
