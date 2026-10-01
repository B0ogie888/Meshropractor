import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkActor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from orientation_cube import OrientationCube, ORIGIN

APP = QApplication.instance() or QApplication([])


class CubeTests(unittest.TestCase):
    def setUp(self):
        self.plotter = QWidget()
        self.plotter.resize(800, 600)
        self.plotter.renderer = vtkRenderer()
        self.plotter.render_window = vtkRenderWindow()
        self.plotter.render_window.AddRenderer(self.plotter.renderer)
        self.plotter.camera = pv.Camera()
        self.plotter.camera.position = (5., 5., 5.)
        self.plotter.camera.focal_point = (0., 0., 0.)
        self.plotter.camera.up = (0., 0., 1.)
        self.plotter.renderer.SetActiveCamera(self.plotter.camera)
        self.plotter.actors = {}
        self.plotter.render = Mock(side_effect=lambda: self.plotter.renderer.InvokeEvent('StartEvent'))
        self.plotter.reset_camera_clipping_range = Mock()
        self.cube = OrientationCube(self.plotter)

    def tearDown(self):
        self.cube.dispose()
        self.plotter.close()

    def test_camera_sync_uses_same_frame_without_requesting_another_render(self):
        for _ in range(20):
            self.plotter.camera.Azimuth(13.)
            self.plotter.renderer.InvokeEvent('StartEvent')
            source = self.plotter.camera.GetViewTransformMatrix()
            result = self.cube.renderer.GetActiveCamera().GetViewTransformMatrix()
            for row in range(3):
                np.testing.assert_allclose([source.GetElement(row, j) for j in range(3)],
                                           [result.GetElement(row, j) for j in range(3)], atol=1e-12)
            for axis, actor in enumerate(self.cube.axis_actors):
                points = actor.GetMapper().GetInput().GetPoints()
                np.testing.assert_array_equal(points.GetPoint(0), ORIGIN)
                endpoint = ORIGIN.copy()
                endpoint[axis] = 1
                np.testing.assert_array_equal(points.GetPoint(1), endpoint)
        self.plotter.render.assert_not_called()
        self.assertFalse(isinstance(self.cube, QWidget))
        self.assertTrue(self.cube.renderer.GetPreserveColorBuffer())

    def test_faces_click_and_empty_corner_passes_through(self):
        self.cube.orient = Mock()
        offset = self.cube.geometry().topLeft()
        def event(kind, point, button=Qt.LeftButton, buttons=Qt.LeftButton):
            return QMouseEvent(kind, QPointF(point), QPointF(point), button, buttons, Qt.NoModifier)
        outside = QPointF(offset) + QPointF(1., 1.)
        self.assertFalse(self.cube.eventFilter(self.plotter, event(QEvent.MouseButtonPress, outside)))
        self.cube.orient.assert_not_called()
        polygon, axis, sign = self.cube.faces[-1]
        inside = QPointF(offset) + polygon.boundingRect().center()
        self.assertTrue(self.cube.eventFilter(self.plotter, event(QEvent.MouseButtonPress, inside)))
        self.cube.orient.assert_called_once_with(axis, sign)
        self.assertTrue(self.cube.eventFilter(self.plotter, event(QEvent.MouseButtonRelease, outside, buttons=Qt.NoButton)))
        self.assertTrue(self.cube.eventFilter(self.plotter, event(QEvent.MouseButtonDblClick, inside)))
        self.cube.orient.assert_called_with(None, sign)

    def test_layer_survives_scene_clear_and_is_excluded_from_selection(self):
        self.plotter.renderer.RemoveAllViewProps()
        self.plotter.renderer.SetUseFXAA(True)
        self.plotter.renderer.InvokeEvent('StartEvent')
        self.assertTrue(self.cube.renderer.GetUseFXAA())
        self.assertTrue(self.plotter.render_window.GetRenderers().IsItemPresent(self.cube.renderer))
        self.assertGreater(self.cube.renderer.GetViewProps().GetNumberOfItems(), 0)
        self.plotter.renderer.GetSelector = Mock(return_value=object())
        self.plotter.renderer.InvokeEvent('StartEvent')
        self.assertFalse(self.cube.renderer.GetDraw())
        self.plotter.renderer.GetSelector.return_value = None
        self.plotter.renderer.InvokeEvent('StartEvent')
        self.assertTrue(self.cube.renderer.GetDraw())
        self.cube.setVisible(False)
        self.plotter.renderer.InvokeEvent('StartEvent')
        self.assertFalse(self.cube.renderer.GetDraw())
        self.cube.setVisible(True)
        self.assertTrue(self.cube.renderer.GetDraw())

    def test_platform_transparency_and_dispose_after_camera_replacement(self):
        actor = vtkActor()
        self.plotter.actors['plat_base'] = actor
        self.cube.orient(2, -1)
        self.assertAlmostEqual(actor.GetProperty().GetOpacity(), .12)
        self.cube.orient(2, 1)
        self.assertEqual(actor.GetProperty().GetOpacity(), 1.)
        observed = self.plotter.camera
        self.plotter.camera = pv.Camera()
        self.cube.dispose()
        self.assertFalse(observed.HasObserver('ModifiedEvent'))
        self.assertFalse(self.plotter.render_window.GetRenderers().IsItemPresent(self.cube.renderer))
        self.cube.dispose()
