"""Display tools with real Qt controls/VTK datasets, independent of OpenGL drivers."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import trimesh
from PySide6.QtWidgets import QCheckBox, QDialog, QLabel, QMainWindow, QTabWidget, QTableWidget, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_desktop import APP
from test_repair_tools import RepairPlotter
from display_ribbon import DEFAULTS, DISPLAY_COMMANDS, GROUPS, TOGGLES, create_display_ribbon
from display_tools import DisplayTools, PREFIX, VIEWS, geometric_checks, material_estimate, scene_statistics
from part_supports import make_group


class DisplayPlotter(RepairPlotter):
    def __init__(self):
        super().__init__()
        self.camera = pv.Camera()
        self.camera.position = (10, -20, 30)
        self.camera.focal_point = (1, 2, 3)
        self.camera.up = (0, 0, 1)
        self.render_count = 0
        self.image = np.full((5, 8, 3), [30, 90, 150], dtype=np.uint8)

    def render(self): self.render_count += 1

    def add_actor(self, actor, name=None, **kwargs):
        self.actors[name] = actor
        return actor, None

    def screenshot(self, **kwargs): return self.image.copy()


class DisplayWindow(QMainWindow):
    COL_VISIBLE = 0

    def __init__(self, meshes=None):
        super().__init__()
        self.ribbon, buttons = create_display_ribbon()
        self.setCentralWidget(self.ribbon)
        plotter = DisplayPlotter()
        scene_tabs = QTabWidget(self)
        for name in ('Модель', 'Build A', 'Build B'): scene_tabs.addTab(QWidget(), name)
        table = QTableWidget(0, 1, self)
        self.ui = SimpleNamespace(display_buttons=buttons, slicer_plotter=plotter, scene_tabs=scene_tabs,
                                  status_label=QLabel(self), tbl_parts=table, section_panel=SimpleNamespace(_planes=[]))
        self.messages, self.selected, self.slicer_parts = [], [], []
        self.platforms = [dict(name='Build A', dim=[20, 20, 20], is_default=True, use_zones=True,
                              zones=[dict(x=0, y=0, r=1, full_h=True, shape=0)]),
                          dict(name='Build B', dim=[40, 40, 40], is_default=True)]
        self.workspace_tools = SimpleNamespace(cube=None)
        for row, mesh in enumerate(meshes if meshes is not None else [trimesh.creation.box(extents=[2, 4, 6])]):
            data = pv.PolyData(mesh.vertices.copy(), np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel())
            name = f'slicer_part_{row}'
            actor = plotter.add_mesh(data, name=name, color='#476b91', opacity=.65)
            actor.mapper.SetScalarVisibility(False)
            support = trimesh.creation.box(extents=[.2, .2, .5]); support.apply_translation(mesh.bounds.mean(axis=0))
            self.slicer_parts.append(dict(mesh=mesh, mesh_pv=data, actor_name=name, filename=f'part_{row}.stl',
                                         platform='Build A' if row == 0 else 'Build B', supports=[make_group(support, [0])],
                                         last_visible_mode='transparent'))
            table.insertRow(row); cell = QWidget(); box = QCheckBox(cell); box.setChecked(True); table.setCellWidget(row, 0, cell)
        plotter.add_mesh(pv.Cube(), name='plat_zone_0')
        self.tools = DisplayTools(self)

    def log(self, text): self.messages.append(text)
    def selected_slicer_rows(self): return list(self.selected)

    def trimesh_to_pyvista(self, mesh):
        return pv.PolyData(mesh.vertices.copy(), np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel())


class DisplayToolsTests(unittest.TestCase):
    def window(self, meshes=None):
        window = DisplayWindow(meshes)
        def cleanup():
            if window.tools._report_dialog: window.tools._report_dialog.close()
            window.close(); window.deleteLater(); APP.processEvents()
        self.addCleanup(cleanup)
        return window

    def toggle(self, window, operation, enabled=True):
        button = window.ui.display_buttons[operation]
        if button.isChecked() != enabled: button.click()

    def test_all_25_commands_icons_groups_view_menu_and_real_action_connections(self):
        window = self.window()
        self.assertEqual(len(DISPLAY_COMMANDS), 25)
        self.assertEqual(list(window.ui.display_buttons), list(DISPLAY_COMMANDS))
        self.assertEqual([label.text() for label in window.ribbon.findChildren(QLabel)], [name for name, _ in GROUPS])
        calls = []
        with patch.object(window.tools, 'trigger', lambda operation: calls.append(operation)):
            for operation, button in window.ui.display_buttons.items():
                self.assertTrue(button.isEnabled(), operation)
                self.assertFalse(button.icon().isNull(), operation)
                self.assertEqual(button.isCheckable(), operation in TOGGLES)
                if operation in DEFAULTS: self.assertEqual(button.isChecked(), DEFAULTS[operation])
                button.click()
        self.assertEqual(calls, list(DISPLAY_COMMANDS))
        self.assertEqual([action.text() for action in window.ui.display_buttons['view'].menu().actions()], list(VIEWS))

    def test_smoothing_changes_only_display_normals_face_and_vertex_ids_remain_identical(self):
        window = self.window([trimesh.creation.icosphere(subdivisions=2)])
        part = window.slicer_parts[0]; actor = window.ui.slicer_plotter.actors[part['actor_name']]
        source, points, faces = part['mesh_pv'], part['mesh_pv'].points.copy(), part['mesh_pv'].faces.copy()
        original_normals = set(source.point_data.keys())
        self.toggle(window, 'smooth')
        shown = actor.mapper.dataset
        self.assertIsNot(shown, source)
        self.assertEqual(shown.n_points, source.n_points)
        np.testing.assert_array_equal(shown.points, points)
        np.testing.assert_array_equal(shown.faces, faces)
        self.assertIn('Normals', shown.point_data)
        self.assertEqual(set(source.point_data.keys()), original_normals)
        self.assertEqual(actor.prop.GetInterpolationAsString(), 'Phong')
        self.toggle(window, 'smooth', False)
        self.assertIs(actor.mapper.dataset, source)
        self.assertAlmostEqual(actor.prop.opacity, .65)
        np.testing.assert_array_equal(part['mesh'].vertices, points)

    def test_annotations_are_non_pickable_cached_and_removed_without_touching_sources(self):
        window = self.window()
        part = window.slicer_parts[0]; plotter = window.ui.slicer_plotter
        source, faces, vertices = part['mesh'], part['mesh'].faces.copy(), part['mesh'].vertices.copy()
        for operation in ('grid', 'ruler', 'dimensions', 'center_mass', 'bbox', 'origin', 'part_number', 'part_name', 'part_path'):
            self.toggle(window, operation)
        overlays = dict(window.tools.overlays)
        self.assertGreater(len(overlays), 10)
        self.assertTrue(all(not actor.GetPickable() for actor in overlays.values()))
        if window.tools._font_file:
            self.assertEqual(overlays[PREFIX + 'part_label_0'].GetTextProperty().GetFontFile(), window.tools._font_file)
        self.assertIn(PREFIX + 'overall_bbox', overlays)
        self.assertIn('Исходный путь не сохранён', overlays[PREFIX + 'part_label_0'].GetInput())
        count = window.tools.rebuild_count
        window.tools.on_scene_changed(); window.tools.on_scene_changed()
        self.assertEqual(window.tools.rebuild_count, count)
        self.assertTrue(all(window.tools.overlays[name] is actor for name, actor in overlays.items()))
        matrix = np.eye(4); matrix[:3, 3] = [3, 4, 5]
        plotter.actors[part['actor_name']].user_matrix = matrix
        window.tools.on_scene_changed()
        self.assertEqual(window.tools.rebuild_count, count)
        shifted = window.tools.overlays[PREFIX + 'part_label_0'].GetUserMatrix()
        self.assertEqual([shifted.GetElement(i, 3) for i in range(3)], [3, 4, 5])
        for operation in ('grid', 'ruler', 'dimensions', 'center_mass', 'bbox', 'origin', 'part_number', 'part_name', 'part_path'):
            self.toggle(window, operation, False)
        self.assertEqual(window.tools.overlays, {})
        self.assertIs(part['mesh'], source)
        np.testing.assert_array_equal(source.faces, faces); np.testing.assert_array_equal(source.vertices, vertices)

    def test_platform_grid_has_exact_millimetres_from_origin_and_clips_partial_edge_cells(self):
        window = self.window()
        window.platforms[0]['dim'] = [220, 220, 280]
        window.ui.scene_tabs.setCurrentIndex(1)
        self.toggle(window, 'grid')
        def segments(name):
            data = window.tools.overlays[PREFIX + name].mapper.dataset
            return data.points[data.lines.reshape(-1, 3)[:, 1:]]
        major, minor = segments('grid_major'), segments('grid_minor')
        self.assertEqual(len(major), 46)  # 23 lines per direction, -110 ... 0 ... +110.
        self.assertEqual(len(minor), 396)
        for axis in (0, 1):
            main_lines = major[major[:, 0, axis] == major[:, 1, axis], 0, axis]
            fine_lines = minor[minor[:, 0, axis] == minor[:, 1, axis], 0, axis]
            np.testing.assert_array_equal(main_lines, np.arange(-110, 111, 10))
            np.testing.assert_array_equal(np.sort(np.r_[main_lines, fine_lines]), np.arange(-110, 111))
            self.assertEqual(np.count_nonzero(main_lines > 0), 11)
            self.assertEqual(np.count_nonzero(main_lines < 0), 11)
        self.assertLess(window.tools.overlays[PREFIX+'grid_minor'].prop.opacity,
                        window.tools.overlays[PREFIX+'grid_major'].prop.opacity)
        window.platforms[1]['dim'] = [225, 143, 80]
        window.ui.scene_tabs.setCurrentIndex(2); window.tools.on_scene_changed()
        for name in ('grid_minor', 'grid_major'):
            lines = segments(name)
            np.testing.assert_allclose(lines.min(axis=(0, 1))[:2], [-112.5, -71.5])
            np.testing.assert_allclose(lines.max(axis=(0, 1))[:2], [112.5, 71.5])
            for axis in (0, 1):
                values = lines[lines[:, 0, axis] == lines[:, 1, axis], 0, axis]
                np.testing.assert_array_equal(values % 1, 0)
                if name == 'grid_major': np.testing.assert_array_equal(values % 10, 0)
        self.toggle(window, 'grid', False)
        self.assertFalse(any(name.startswith(PREFIX+'grid') for name in window.tools.overlays))

    def test_grid_depth_bias_tracks_plate_replacement_and_is_removed_on_toggle_off(self):
        from vtkmodules.vtkRenderingCore import vtkMapper
        window = self.window()
        plotter = window.ui.slicer_plotter
        global_mode = vtkMapper.GetResolveCoincidentTopology()
        plate = plotter.add_mesh(pv.Plane(), name='plat_base')
        self.toggle(window, 'grid')
        self.assertEqual(plate.GetShaderProperty().GetNumberOfShaderReplacements(), 1)
        cached = window.tools.overlays[PREFIX+'grid_major']
        window.tools.on_scene_changed()
        self.assertIs(window.tools.overlays[PREFIX+'grid_major'], cached)
        # Redrawing a platform with the same dimensions creates a different actor.
        replacement = plotter.add_mesh(pv.Plane(), name='plat_base')
        window.tools.on_scene_changed()
        self.assertEqual(plate.GetShaderProperty().GetNumberOfShaderReplacements(), 0)
        self.assertEqual(replacement.GetShaderProperty().GetNumberOfShaderReplacements(), 1)
        self.toggle(window, 'grid', False)
        self.assertEqual(replacement.GetShaderProperty().GetNumberOfShaderReplacements(), 0)
        self.assertEqual(vtkMapper.GetResolveCoincidentTopology(), global_mode)

    def test_simplified_view_restores_visibility_supports_and_respects_platform_filter(self):
        window = self.window([trimesh.creation.box(), trimesh.creation.box()])
        plotter = window.ui.slicer_plotter
        window.ui.scene_tabs.setCurrentIndex(1); window.tools.on_scene_changed()
        first, second = window.slicer_parts
        self.assertTrue(plotter.actors[first['actor_name']].GetVisibility())
        self.assertFalse(plotter.actors[second['actor_name']].GetVisibility())
        support = plotter.actors['part_support_' + first['supports'][0]['id']]
        self.toggle(window, 'simplified')
        self.assertFalse(plotter.actors[first['actor_name']].GetVisibility())
        self.assertFalse(support.GetVisibility())
        self.assertIn(PREFIX + 'simple_0', window.tools.overlays)
        self.assertNotIn(PREFIX + 'simple_1', window.tools.overlays)
        self.toggle(window, 'simplified', False)
        self.assertTrue(plotter.actors[first['actor_name']].GetVisibility())
        self.assertTrue(support.GetVisibility())
        self.assertFalse(plotter.actors[second['actor_name']].GetVisibility())
        self.assertFalse(window.tools.overlays)

    def test_texture_colors_and_triangle_colors_restore_original_mapper_and_no_data_explains(self):
        window = self.window()
        part, plotter = window.slicer_parts[0], window.ui.slicer_plotter
        actor = plotter.actors[part['actor_name']]
        self.toggle(window, 'texture')
        self.assertFalse(window.ui.display_buttons['texture'].isChecked())
        self.assertIn('UV-текстуры', window.messages[-1])
        part['mesh'].visual.face_colors = np.tile([80, 120, 220, 255], (len(part['mesh'].faces), 1))
        self.toggle(window, 'texture')
        self.assertTrue(actor.mapper.GetScalarVisibility())
        np.testing.assert_array_equal(actor.mapper.dataset.cell_data['_display_colors'], part['mesh'].visual.face_colors)
        self.assertNotIn('_display_colors', part['mesh_pv'].cell_data)
        self.toggle(window, 'triangle_colors')
        self.assertFalse(window.ui.display_buttons['texture'].isChecked())
        self.assertGreater(len(np.unique(actor.mapper.dataset.cell_data['_display_colors'], axis=0)), 1)
        self.toggle(window, 'triangle_colors', False)
        self.assertFalse(actor.mapper.GetScalarVisibility())
        self.assertIs(actor.mapper.dataset, part['mesh_pv'])

    def test_camera_views_and_cube_visibility_do_not_rebuild_meshes(self):
        window = self.window()
        plotter = window.ui.slicer_plotter
        focus = np.asarray(plotter.camera.focal_point)
        distance = np.linalg.norm(np.asarray(plotter.camera.position) - focus)
        for action, (axis, sign) in zip(window.ui.display_buttons['view'].menu().actions(), VIEWS.values()):
            action.trigger()
            expected = np.ones(3) / np.sqrt(3) if axis is None else np.eye(3)[axis] * sign
            np.testing.assert_allclose((np.asarray(plotter.camera.position) - focus) / distance, expected)
        calls = []
        window.workspace_tools.cube = SimpleNamespace(setVisible=lambda visible: calls.append(visible), orient=lambda *args: calls.append(args))
        window.tools.on_scene_changed()
        self.toggle(window, 'coordinates', False)
        self.toggle(window, 'coordinates')
        window.tools.orient(0, -1)
        self.assertEqual(calls, [True, False, True, (0, -1)])

    def test_platform_zone_toggle_survives_redraw_and_no_zone_in_model_scene(self):
        window = self.window()
        plotter = window.ui.slicer_plotter
        self.assertFalse(plotter.actors['plat_zone_0'].GetVisibility())
        window.ui.scene_tabs.setCurrentIndex(1); window.tools.on_scene_changed()
        self.assertTrue(plotter.actors['plat_zone_0'].GetVisibility())
        self.toggle(window, 'zones', False)
        plotter.add_mesh(pv.Cube(), name='plat_zone_0')
        window.tools.on_scene_changed()
        self.assertFalse(plotter.actors['plat_zone_0'].GetVisibility())
        self.toggle(window, 'zones')
        self.assertTrue(plotter.actors['plat_zone_0'].GetVisibility())

    def test_overhang_and_outside_highlight_real_faces_and_checks_are_conservative(self):
        mesh = trimesh.creation.box(extents=[2, 2, 2]); mesh.apply_translation([12, 0, 4])
        window = self.window([mesh]); window.ui.scene_tabs.setCurrentIndex(1)
        self.toggle(window, 'overhang')
        self.assertIn(PREFIX + 'overhang_0', window.tools.overlays, window.messages)
        actor = window.tools.overlays[PREFIX + 'overhang_0']
        self.assertEqual(actor.mapper.dataset.n_cells, 2)
        self.toggle(window, 'outside')
        self.assertIn(PREFIX + 'warning_0', window.tools.overlays)
        self.toggle(window, 'build_risk')
        report = window.tools._report_dialog.report.toPlainText()
        self.assertIn('не симуляция', report)
        self.assertIn('выходят за габариты', report)
        first = trimesh.creation.box(); second = first.copy(); second.apply_translation([.3, .3, .3])
        checks = geometric_checks([{'mesh': first, 'filename': 'first'}, {'mesh': second, 'filename': 'second'}])
        self.assertEqual(checks['flagged'], {0, 1})
        self.assertTrue(any('Возможное пересечение габаритов' in warning for warning in checks['warnings']))

    def test_statistics_handle_open_meshes_mass_units_and_live_cost_fields(self):
        closed = trimesh.creation.box(extents=[10, 10, 10])
        opened = closed.copy(); opened.update_faces(np.arange(11))
        result = scene_statistics([{'mesh': closed, 'filename': 'closed'}, {'mesh': opened, 'filename': 'open'}])
        self.assertEqual(result['total_mm3'], 1000)
        self.assertEqual(result['unknown'], 1)
        self.assertIsNone(result['rows'][1]['volume'])
        self.assertEqual(material_estimate(1_000_000, 2, 50), (2000, 100))
        window = self.window([closed, opened]); window.selected = [0]
        window.ui.display_buttons['material_cost'].click()
        window.tools.show_report('material_cost')
        report = window.tools._report_dialog
        report.density.setValue(2); report.price.setValue(50)
        self.assertIn('2.000', report.report.toPlainText())
        self.assertIn('0.10', report.report.toPlainText())
        self.assertNotIn('part_1.stl', report.report.toPlainText())
        window.ui.scene_tabs.setCurrentIndex(1)
        window.ui.display_buttons['packing_density'].click()
        self.assertAlmostEqual(window.tools.statistics.data['usage_percent'], 12.50025)
        window.ui.display_buttons['volume'].click()
        self.assertAlmostEqual(window.tools.statistics.data['parts_mm3'], 1000)

    def test_image_export_clipboard_and_print_cancel_use_current_scene(self):
        window = self.window()
        image = window.tools.capture_image()
        self.assertEqual((image.width(), image.height()), (8, 5))
        self.assertEqual(image.pixelColor(0, 0).getRgb()[:3], (30, 90, 150))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'сцена.png')
            with patch('display_tools.QFileDialog.getSaveFileName', lambda *args: (path, 'PNG')):
                window.ui.display_buttons['export_png'].click()
            self.assertTrue(Path(path).is_file())
            window.ui.display_buttons['clipboard'].click()
            self.assertEqual(APP.clipboard().image().pixelColor(0, 0).getRgb()[:3], (30, 90, 150))
        calls = []
        with patch('display_tools.QPrintDialog.exec', lambda self: calls.append(self.windowTitle()) or QDialog.Rejected):
            window.ui.display_buttons['print'].click()
        self.assertEqual(calls, ['Печать изображения сцены'])
        self.assertFalse(any('не удалось' in message.lower() for message in window.messages))


if __name__ == '__main__':
    unittest.main()
