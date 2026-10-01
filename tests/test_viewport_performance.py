import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication
from vtkmodules.vtkRenderingCore import vtkProperty, vtkRenderWindowInteractor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from viewport_performance import ViewportPerformance
from surface_selection import SurfaceTopology
APP = QApplication.instance() or QApplication([])


class PerformanceTests(unittest.TestCase):
    def test_quality_switch_resets_previous_aa_and_edges_can_stay_visible(self):
        plotter = QObject()
        plotter.iren = SimpleNamespace(interactor=vtkRenderWindowInteractor())
        plotter.render = Mock()
        plotter.disable_anti_aliasing = Mock()
        plotter.enable_anti_aliasing = Mock()
        prop = vtkProperty()
        prop.SetEdgeVisibility(True)
        actor = Mock()
        actor.GetProperty.return_value = prop
        actor.GetVisibility.return_value = True
        actor.GetMapper.return_value.GetInput.return_value.GetNumberOfCells.return_value = 200000
        plotter.actors = {'part': actor}
        perf = ViewportPerformance(plotter)
        try:
            perf.configure(anti_aliasing='fxaa')
            perf.configure(anti_aliasing='msaa', samples=8)
            self.assertEqual(plotter.disable_anti_aliasing.call_count, 2)
            plotter.enable_anti_aliasing.assert_called_with('msaa', multi_samples=8)
            perf.begin()
            self.assertFalse(prop.GetEdgeVisibility())
            perf.configure(anti_aliasing='msaa', samples=8, interactive_edges=False)
            self.assertEqual(plotter.disable_anti_aliasing.call_count, 2)
            self.assertTrue(prop.GetEdgeVisibility())
            perf.begin()
            self.assertTrue(prop.GetEdgeVisibility())
            perf.configure(anti_aliasing='none')
            self.assertEqual(plotter.enable_anti_aliasing.call_count, 2)
        finally:
            perf.dispose()

    def test_large_mesh_edges_restore_after_interaction(self):
        plotter = QObject()
        plotter.iren = SimpleNamespace(interactor=vtkRenderWindowInteractor())
        plotter.render = Mock()
        def actor(cells):
            prop = vtkProperty(); prop.SetEdgeVisibility(True)
            item = Mock()
            item.GetProperty.return_value = prop
            item.GetVisibility.return_value = True
            item.GetMapper.return_value.GetInput.return_value.GetNumberOfCells.return_value = cells
            return item, prop
        large, prop = actor(200000); small, small_prop = actor(12)
        plotter.actors = dict(large=large, small=small)
        perf = ViewportPerformance(plotter)
        plotter.iren.interactor.InvokeEvent('StartInteractionEvent')
        self.assertFalse(prop.GetEdgeVisibility())
        self.assertTrue(small_prop.GetEdgeVisibility())
        plotter.iren.interactor.InvokeEvent('EndInteractionEvent')
        self.assertTrue(prop.GetEdgeVisibility())
        plotter.render.assert_called_once()
        perf.dispose()

    def test_triangle_selection_does_not_construct_adjacency(self):
        class TrianglesOnly:
            faces = range(1000000)
            @property
            def face_adjacency(self):
                raise AssertionError('Slow topology build for a single triangle')
        self.assertEqual(SurfaceTopology(TrianglesOnly()).select(51, 'triangle'), {51})
