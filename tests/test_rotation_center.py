from qt_test_cleanup import delete_widget
"""Camera pivots follow viewport double-clicks and never table selection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import trimesh
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_desktop import APP
from test_repair_tools import RepairPlotter
from main_window import MainWindow
from project_store import ProjectState


class PivotPlotter(RepairPlotter):
    def __init__(self):
        super().__init__()
        self.camera = pv.Camera()
        self.camera.position = (80., -120., 100.)
        self.camera.focal_point = (0., 0., 0.)
        self.camera.up = (0., 0., 1.)
        self.framed = False

    def add_mesh(self, mesh, **kwargs):
        actor = super().add_mesh(mesh, **kwargs)
        if not self.framed:
            self.camera.focal_point = actor.center
            self.framed = True
        return actor

    def reset_camera(self):
        self.camera.focal_point = (0., 0., 100.)

    def add_actor(self, actor, name=None, **kwargs):
        self.actors[name] = actor
        return actor, None


class RotationCenterTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.window.ui._ensure_slicer_plotter = lambda: setattr(
            self.window.ui, 'slicer_plotter', self.window.ui.slicer_plotter or PivotPlotter())
        self.window.ui._ensure_def_plotter = lambda: setattr(
            self.window.ui, 'plotter', self.window.ui.plotter or RepairPlotter())
        self.window.restore_project(ProjectState(page='slicer'))
        self.plotter = self.window.ui.slicer_plotter
        self.workspace = self.window.workspace_tools
        self.workspace.plotter = self.plotter
        self.mesh = trimesh.creation.box(); self.mesh.apply_translation([30, 40, 15])
        self.window._append_slicer_part(self.mesh, 'деталь.stl')
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.window.dirty = False; self.window.close(); delete_widget(self.window); APP.processEvents()

    def event(self, kind=QEvent.MouseButtonDblClick, modifiers=Qt.NoModifier):
        return QMouseEvent(kind, QPointF(100, 100), QPointF(100, 100),
                           Qt.LeftButton, Qt.LeftButton if kind != QEvent.MouseButtonRelease else Qt.NoButton,
                           modifiers)

    def test_initial_import_and_table_clicks_keep_plate_center(self):
        np.testing.assert_allclose(self.plotter.camera.focal_point, [0, 0, 0])
        before = np.array(self.plotter.camera.position)
        self.window.ui.tbl_parts.cellClicked.emit(0, 0)
        self.window.ui.tbl_parts.cellDoubleClicked.emit(0, 0)
        np.testing.assert_allclose(self.plotter.camera.focal_point, [0, 0, 0])
        np.testing.assert_allclose(self.plotter.camera.position, before)

    def test_double_click_part_uses_actor_world_center_and_retains_view(self):
        actor = self.plotter.actors[self.window.slicer_parts[0]['actor_name']]
        actor.SetPosition(10, -5, 2)
        camera = self.plotter.camera
        direction = np.asarray(camera.position) - np.asarray(camera.focal_point)
        up = camera.up; zoom = camera.parallel_scale
        with patch.object(self.workspace, 'picker', return_value=(0, 0, self.mesh.centroid)) as picker:
            self.assertTrue(self.workspace.eventFilter(self.plotter, self.event()))
            picker.assert_called_once_with(self.event().position().toPoint(), selected_only=False)
        np.testing.assert_allclose(camera.focal_point, [40, 35, 17])
        np.testing.assert_allclose(np.asarray(camera.position) - np.asarray(camera.focal_point), direction)
        np.testing.assert_allclose(camera.up, up)
        self.assertEqual(camera.parallel_scale, zoom)
        self.assertTrue(self.workspace.eventFilter(self.plotter, self.event(QEvent.MouseButtonRelease)))

    def test_empty_double_click_returns_to_plate_without_resetting_view(self):
        self.window.set_slicer_rotation_center([30, 40, 15])
        before = np.asarray(self.plotter.camera.position) - np.asarray(self.plotter.camera.focal_point)
        with patch.object(self.workspace, 'picker', return_value=None):
            self.workspace.eventFilter(self.plotter, self.event())
        np.testing.assert_allclose(self.plotter.camera.focal_point, [0, 0, 0])
        np.testing.assert_allclose(self.plotter.camera.position, before)

    def test_single_scene_click_and_later_import_do_not_move_chosen_pivot(self):
        self.window.set_slicer_rotation_center([30, 40, 15])
        with patch.object(self.workspace, 'picker', return_value=(0, 0, self.mesh.centroid)):
            self.workspace.eventFilter(self.plotter, self.event(QEvent.MouseButtonPress))
            self.workspace.eventFilter(self.plotter, self.event(QEvent.MouseButtonRelease))
        self.window._append_slicer_part(trimesh.creation.box(), 'вторая.stl')
        np.testing.assert_allclose(self.plotter.camera.focal_point, [30, 40, 15])

    def test_platform_switch_and_new_project_reset_default_pivot(self):
        self.window.platforms = [dict(name='Плита', dim=[50, 50, 60], is_default=True)]
        self.window.update_platform_ui(draw=False)
        self.window.set_slicer_rotation_center([30, 40, 15])
        self.window.ui.scene_tabs.setCurrentIndex(1)
        np.testing.assert_allclose(self.plotter.camera.focal_point, [0, 0, 0])
        self.window.set_slicer_rotation_center([30, 40, 15])
        self.window.clear_project_data()
        np.testing.assert_allclose(self.plotter.camera.focal_point, [0, 0, 0])

    def test_surface_and_measurement_modes_require_alt_for_pivot(self):
        self.workspace.mode = 'triangle'
        with patch.object(self.workspace, 'picker', return_value=(0, 0, self.mesh.centroid)) as picker:
            self.assertFalse(self.workspace.eventFilter(self.plotter, self.event()))
            picker.assert_not_called()
            self.workspace.left_down = True
            self.assertTrue(self.workspace.eventFilter(self.plotter, self.event(modifiers=Qt.AltModifier)))
            self.assertFalse(self.workspace.left_down)
        np.testing.assert_allclose(self.plotter.camera.focal_point, self.mesh.centroid)
        self.workspace.mode = 'part'; self.workspace.measurements.active = True
        with patch.object(self.workspace, 'focus_at') as focus:
            self.assertFalse(self.workspace.eventFilter(self.plotter, self.event()))
            focus.assert_not_called()

    def test_shortcuts_describe_both_double_clicks_and_table_behavior(self):
        with patch.object(self.window.display_settings, '_show_text') as show:
            self.window.display_settings.show_shortcuts()
        title, html = show.call_args.args
        self.assertEqual(title, 'Горячие клавиши и управление')
        self.assertIn('по детали в слайсере', html)
        self.assertIn('по пустому месту', html)
        self.assertIn('в центр плиты построения', html)
        self.assertIn('Выбор детали в списке центр не меняет', html)
