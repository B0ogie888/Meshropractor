"""Selected-part statistics, transformed previews and mouse selection gestures."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import trimesh
from PySide6.QtCore import QEvent, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QCheckBox
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonDataModel import vtkSelection, vtkSelectionNode

import test_display_tools
import test_rotation_center
from test_desktop import APP


class LiveStatisticsTests(unittest.TestCase):
    def window(self, meshes=None):
        window = test_display_tools.DisplayWindow(meshes)
        def cleanup():
            if window.tools._report_dialog is not None: window.tools._report_dialog.close()
            window.close(); window.deleteLater(); APP.processEvents()
        self.addCleanup(cleanup)
        return window

    def test_empty_selection_is_zero_and_all_three_commands_render_nonpickable_hud(self):
        window = self.window(); window.ui.scene_tabs.setCurrentIndex(1)
        for key in ('volume', 'material_cost', 'packing_density'): window.ui.display_buttons[key].click()
        stats = window.tools.statistics
        self.assertEqual(stats.data['count'], 0)
        self.assertEqual(stats.data['total_mm3'], 0)
        self.assertEqual(stats.data['usage_percent'], 0)
        self.assertEqual(stats.data['packing_percent'], 0)
        self.assertTrue(all(not actor.GetPickable() for actor in stats.actors.values()))
        self.assertIn('Высота сборки', dict(stats.lines))
        self.assertIn('Объём деталей', dict(stats.lines))
        self.assertIn('Стоимость материала', dict(stats.lines))
        for key in ('volume', 'material_cost', 'packing_density'): window.ui.display_buttons[key].click()
        self.assertTrue(all(not actor.GetVisibility() for actor in stats.actors.values()))

    def test_live_selection_supports_preview_mass_cost_and_density_formulas(self):
        mesh = trimesh.creation.box(extents=[10, 10, 10]); mesh.apply_translation([0, 0, 5])
        window = self.window([mesh]); window.selected = [0]; window.ui.scene_tabs.setCurrentIndex(1)
        window.tools._density, window.tools._price = 2., 50.
        window.ui.display_buttons['packing_density'].click()
        data = window.tools.statistics.data
        self.assertAlmostEqual(data['total_mm3'], 1000.02)
        self.assertAlmostEqual(data['height_mm'], 10)
        self.assertAlmostEqual(data['usage_percent'], 1000.02 / 8000 * 100)
        self.assertAlmostEqual(data['packing_percent'], 1000.02 / 4000 * 100)
        self.assertAlmostEqual(data['mass_g'], 2.00004)
        self.assertAlmostEqual(data['cost'], .100002)
        matrix = vtkMatrix4x4(); matrix.Identity(); matrix.SetElement(2, 3, 10)
        window.ui.slicer_plotter.actors['slicer_part_0'].SetUserMatrix(matrix)
        window.tools.on_scene_changed(); window.tools.statistics.update()
        self.assertAlmostEqual(window.tools.statistics.data['height_mm'], 20)
        self.assertAlmostEqual(window.tools.statistics.data['packing_percent'], 1000.02 / 8000 * 100)
        matrix.SetElement(0, 0, 2); matrix.SetElement(1, 1, 2); matrix.SetElement(2, 2, 2)
        window.tools.statistics.update()
        self.assertAlmostEqual(window.tools.statistics.data['total_mm3'], 1000.02 * 8)
        window.selected = []; window.tools.statistics.request()
        window.tools.statistics.update()
        self.assertEqual(window.tools.statistics.data['total_mm3'], 0)

    def test_unknown_volumes_are_marked_and_platform_absence_is_explicit(self):
        mesh = trimesh.creation.box(); mesh.update_faces(np.arange(11))
        window = self.window([mesh]); window.selected = [0]
        window.ui.display_buttons['volume'].click(); window.ui.display_buttons['packing_density'].click()
        stats = window.tools.statistics
        self.assertEqual(stats.data['unknown'], 1)
        self.assertIsNone(stats.data['usage_percent'])
        self.assertEqual(dict(stats.lines)['Использование объёма платформы'], '—')
        self.assertIn('≥', dict(stats.lines)['Суммарный объём'])

    def test_statistics_list_layout_and_material_menu(self):
        window = self.window(); window.ribbon.resize(2400, 85); window.ribbon.show(); APP.processEvents()
        buttons = [window.ui.display_buttons[key] for key in ('volume', 'material_cost', 'packing_density')]
        self.assertEqual(len({button.x() for button in buttons}), 1)
        self.assertLess(buttons[0].y(), buttons[1].y()); self.assertLess(buttons[1].y(), buttons[2].y())
        buttons[1].menu().actions()[0].trigger()
        self.assertTrue(window.tools._report_dialog.isVisible())


class PartSelectionTests(unittest.TestCase):
    setUp = test_rotation_center.RotationCenterTests.setUp
    cleanup = test_rotation_center.RotationCenterTests.cleanup

    def add_second(self):
        mesh = trimesh.creation.box(); mesh.apply_translation([-20, 0, 5])
        self.window._append_slicer_part(mesh, 'second.stl')

    def mouse(self, kind, modifiers=Qt.NoModifier, pos=(100, 100)):
        button = Qt.NoButton if kind == QEvent.MouseMove else Qt.LeftButton
        buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        return QMouseEvent(kind, QPointF(*pos), QPointF(*pos), button, buttons, modifiers)

    def click(self, hit, modifiers=Qt.NoModifier):
        with patch.object(self.workspace, 'picker', return_value=hit):
            self.workspace.eventFilter(self.plotter, self.mouse(QEvent.MouseButtonPress, modifiers))
            self.workspace.eventFilter(self.plotter, self.mouse(QEvent.MouseButtonRelease, modifiers))

    def test_click_shift_ctrl_and_empty_selection_sync_checkboxes_and_hud(self):
        self.add_second()
        self.window.ui.display_buttons['volume'].click()
        self.click((1, 0, np.zeros(3)))
        self.assertEqual(self.window.selected_slicer_rows(), [1])
        self.click((0, 0, np.zeros(3)), Qt.ShiftModifier)
        self.assertEqual(self.window.selected_slicer_rows(), [0, 1])
        self.click((1, 0, np.zeros(3)), Qt.ControlModifier)
        self.assertEqual(self.window.selected_slicer_rows(), [0])
        before = self.plotter.camera.position
        self.click(None)
        self.assertEqual(self.window.selected_slicer_rows(), [])
        self.window.display_tools.statistics.update()
        self.assertEqual(self.window.display_tools.statistics.data['count'], 0)
        np.testing.assert_allclose(self.plotter.camera.position, before)

    def test_drag_empty_space_selects_parts_but_alt_keeps_camera_navigation(self):
        self.add_second()
        with patch.object(self.workspace, 'picker', return_value=None), \
             patch.object(self.workspace, 'select_parts_rectangle') as box:
            self.workspace.eventFilter(self.plotter, self.mouse(QEvent.MouseButtonPress, pos=(10, 10)))
            self.workspace.eventFilter(self.plotter, self.mouse(QEvent.MouseMove, pos=(200, 100)))
            self.workspace.eventFilter(self.plotter, self.mouse(QEvent.MouseButtonRelease, pos=(200, 100)))
            box.assert_called_once_with(QRect(10, 10, 191, 91), 'replace')
            self.assertFalse(self.workspace.eventFilter(self.plotter, self.mouse(QEvent.MouseButtonPress, Qt.AltModifier)))
        self.assertFalse(self.workspace.part_box_candidate)

    def test_hardware_rectangle_uses_visible_actor_identity_and_updates_whole_parts(self):
        self.add_second()
        result = vtkSelection(); node = vtkSelectionNode()
        node.GetProperties().Set(vtkSelectionNode.PROP(), self.plotter.actors['slicer_part_1'])
        result.AddNode(node)
        self.plotter.renderer = SimpleNamespace()
        self.plotter.render_window = SimpleNamespace(GetSize=lambda: (1000, 800))
        self.plotter.devicePixelRatioF = lambda: 2.
        with patch('slicer_workspace.vtkHardwareSelector') as selector:
            selector.return_value.Select.return_value = result
            self.workspace.select_parts_rectangle(QRect(10, 20, 100, 50), 'replace')
            selector.return_value.SetArea.assert_called_once_with(20, 661, 218, 759)
        self.assertEqual(self.window.selected_slicer_rows(), [1])

    def test_selection_outlines_follow_previews_and_are_nonpickable(self):
        self.workspace.refresh_part_selection()
        outline = self.workspace.part_outlines[0][0]
        self.assertFalse(outline.GetPickable())
        matrix = vtkMatrix4x4(); matrix.Identity(); matrix.SetElement(0, 3, 12)
        self.plotter.actors['slicer_part_0'].SetUserMatrix(matrix)
        self.workspace.refresh_part_selection()
        self.assertEqual(outline.GetUserMatrix().GetElement(0, 3), 12)
        self.workspace.select_parts([])
        self.assertFalse(self.workspace.part_outlines)
