"""Camera gestures keep geometry/pivot stable and never turn a drag into a menu."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock
import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QPointF, QRect, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scene_navigation import SceneNavigation, SceneOutline

APP = QApplication.instance() or QApplication([])


class SceneNavigationTests(unittest.TestCase):
    def setUp(self):
        self.plotter = QWidget()
        self.plotter.resize(900, 600)
        self.plotter.renderer = vtkRenderer()
        self.plotter.render_window = vtkRenderWindow()
        self.plotter.render_window.SetSize(900, 600)
        self.plotter.render_window.AddRenderer(self.plotter.renderer)
        self.plotter.camera = pv.Camera()
        self.plotter.camera.position = (3., 4., 20.)
        self.plotter.camera.focal_point = (3., 4., 5.)
        self.plotter.camera.up = (0., 1., 0.)
        self.plotter.iren = SimpleNamespace(interactor=vtkRenderWindowInteractor())
        self.plotter.render = Mock()
        self.plotter.reset_camera_clipping_range = Mock()
        self.click = Mock()
        self.navigation = SceneNavigation(self.plotter, self.click)
        self.events = []
        for name in ('StartInteractionEvent', 'EndInteractionEvent'):
            self.plotter.iren.interactor.AddObserver(name, lambda obj, event: self.events.append(event))

    def tearDown(self):
        self.navigation.dispose()
        self.plotter.close()

    def mouse(self, kind, point, buttons=Qt.RightButton):
        button = Qt.NoButton if kind == QEvent.MouseMove else Qt.RightButton
        point = QPointF(*point)
        event = QMouseEvent(kind, point, point, button, buttons, Qt.NoModifier)
        return self.navigation.eventFilter(self.plotter, event)

    def test_inside_orbits_without_alt_and_keeps_distance_and_pivot(self):
        camera = self.plotter.camera
        position, focus = np.array(camera.position), np.array(camera.focal_point)
        self.mouse(QEvent.MouseButtonPress, (450, 300))
        self.assertTrue(self.navigation.guide.actor.GetVisibility())
        self.mouse(QEvent.MouseMove, (510, 345))
        self.mouse(QEvent.MouseButtonRelease, (510, 345), Qt.NoButton)
        self.assertFalse(np.allclose(camera.position, position))
        np.testing.assert_allclose(camera.focal_point, focus)
        self.assertAlmostEqual(np.linalg.norm(np.array(camera.position) - focus), np.linalg.norm(position - focus))
        self.assertEqual(self.events, ['StartInteractionEvent', 'EndInteractionEvent'])
        self.assertFalse(self.navigation.guide.actor.GetVisibility())
        APP.processEvents()
        self.click.assert_not_called()

    def test_outside_rolls_clockwise_and_counterclockwise_without_moving_camera(self):
        camera = self.plotter.camera
        camera.position = (3., 4., 20.)
        position, focus = camera.position, camera.focal_point
        self.mouse(QEvent.MouseButtonPress, (710, 300))
        self.mouse(QEvent.MouseMove, (450, 560))
        np.testing.assert_allclose(camera.up, [-1, 0, 0], atol=1e-12)
        self.assertEqual(camera.position, position)
        self.assertEqual(camera.focal_point, focus)
        self.mouse(QEvent.MouseMove, (710, 300))
        np.testing.assert_allclose(camera.up, [0, 1, 0], atol=1e-12)
        self.mouse(QEvent.MouseButtonRelease, (710, 300), Qt.NoButton)
        APP.processEvents()
        self.click.assert_not_called()

    def test_mode_latches_at_press_when_crossing_circle(self):
        self.mouse(QEvent.MouseButtonPress, (450, 300))
        self.mouse(QEvent.MouseMove, (800, 300))
        self.assertEqual(self.navigation.mode, 'orbit')
        self.navigation.cancel()
        self.mouse(QEvent.MouseButtonPress, (800, 300))
        self.mouse(QEvent.MouseMove, (460, 300))
        self.assertEqual(self.navigation.mode, 'roll')

    def test_short_click_opens_menu_after_release_without_moving_camera(self):
        position = self.plotter.camera.position
        self.mouse(QEvent.MouseButtonPress, (450, 300))
        self.mouse(QEvent.MouseMove, (451, 300))
        self.mouse(QEvent.MouseButtonRelease, (451, 300), Qt.NoButton)
        APP.processEvents()
        self.click.assert_called_once()
        self.assertEqual(self.plotter.camera.position, position)
        self.assertEqual(self.events, [])

    def test_escape_and_lost_button_end_interaction_and_hide_guide(self):
        for reason in ('button', 'focus', 'escape'):
            self.mouse(QEvent.MouseButtonPress, (450, 300))
            self.mouse(QEvent.MouseMove, (470, 310))
            if reason == 'button':
                self.mouse(QEvent.MouseMove, (471, 310), Qt.NoButton)
            elif reason == 'focus':
                self.navigation.eventFilter(self.plotter, QEvent(QEvent.FocusOut))
            else:
                self.assertTrue(self.navigation.eventFilter(
                    self.plotter, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)))
            self.assertIsNone(self.navigation.start)
            self.assertFalse(self.navigation.guide.actor.GetVisibility())
        self.assertEqual(self.events.count('StartInteractionEvent'), 3)
        self.assertEqual(self.events.count('EndInteractionEvent'), 3)

    def test_guides_have_lines_without_polygons_and_reattach_after_clear(self):
        outline = SceneOutline(self.plotter)
        outline.setGeometry(QRect(10, 20, 90, 60))
        self.assertEqual(outline.geometry(), QRect(10, 20, 90, 60))
        self.assertEqual(outline.data.GetNumberOfPolys(), 0)
        self.assertEqual(outline.data.GetNumberOfLines(), 1)
        outline.show()
        self.plotter.renderer.RemoveAllViewProps()
        outline.show()
        self.assertTrue(self.plotter.renderer.HasViewProp(outline.actor))
        outline.dispose()
        self.mouse(QEvent.MouseButtonPress, (450, 300))
        self.assertEqual(self.navigation.guide.data.GetNumberOfPolys(), 0)
        self.assertEqual(self.navigation.guide.data.GetNumberOfLines(), 64)


if __name__ == '__main__': unittest.main()
