import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
import trimesh
import pyvista as pv
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from Meshropractor import MainWindow
from background_tasks import FunctionWorker
from project_store import ProjectState, load_project

APP = QApplication.instance() or QApplication([])


class TestPlotter:
    """Real VTK datasets/actors without requiring an OpenGL window in CI."""
    def __init__(self):
        self.actors = {}
        self.scalar_bars = {}
    def add_mesh(self, mesh, name=None, **kwargs):
        actor = pv.Actor(mapper=pv.DataSetMapper(dataset=mesh))
        if 'color' in kwargs: actor.prop.color = kwargs['color']
        self.actors[name or str(id(actor))] = actor
        return actor
    def remove_actor(self, actor):
        key = actor if isinstance(actor, str) else next((k for k, v in self.actors.items() if v is actor), None)
        self.actors.pop(key, None)
    def clear(self): self.actors.clear(); self.scalar_bars.clear()
    def add_axes(self): pass
    def reset_camera(self): pass
    def reset_camera_clipping_range(self): pass
    def render(self): pass
    def close(self): pass
    def disable_picking(self): pass


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.window.ui._ensure_def_plotter = lambda: setattr(self.window.ui, 'plotter', self.window.ui.plotter or TestPlotter())
        self.window.ui._ensure_slicer_plotter = lambda: setattr(self.window.ui, 'slicer_plotter', self.window.ui.slicer_plotter or TestPlotter())
        self.window.add_to_recent = lambda path: None
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        self.window.dirty = False
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()

    def wait_for_job(self):
        deadline = time.monotonic() + 10
        while self.window._job is not None and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(0.005)
        self.assertIsNone(self.window._job, "worker did not finish")

    def test_worker_error_restores_controls_on_gui_thread(self):
        window = self.window
        def fail(): raise ValueError("expected test failure")
        window.start_job(FunctionWorker(fail), lambda _: self.fail("success callback after error"))
        self.wait_for_job()
        self.assertTrue(window.ui.btn_run_icp.isEnabled())
        self.assertEqual(window.ui.comp_stack.currentIndex(), 0)

    def test_worker_result_uses_gui_thread(self):
        received = []
        self.window.start_job(FunctionWorker(lambda: 12), lambda value: received.append((value, QThread.currentThread() == APP.thread())))
        self.wait_for_job()
        self.assertEqual(received, [(12, True)])

    def test_cancel_ignores_late_result_and_restores_controls(self):
        received = []
        self.window.start_job(FunctionWorker(lambda: 12), received.append)
        self.window.cancel_current_job()
        self.wait_for_job()
        self.assertEqual(received, [])
        self.assertTrue(self.window.ui.btn_run_def.isEnabled())

    def test_corrupt_project_keeps_existing_models(self):
        self.window.cad_mesh = trimesh.creation.box()
        previous = self.window.cad_mesh
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.mrp"
            path.write_bytes(b"broken archive")
            self.window.load_mrp_file(str(path))
            self.wait_for_job()
        self.assertIs(self.window.cad_mesh, previous)

    def test_cancel_new_project_dialog_does_not_clear(self):
        self.window.cad_mesh = trimesh.creation.box()
        previous = self.window.cad_mesh
        with patch("UI_Meshropractor.DialogNewProject.exec", return_value=0):
            self.window.action_new_project()
        self.assertIs(self.window.cad_mesh, previous)

    def test_restore_all_results_and_active_map(self):
        cube = trimesh.creation.box()
        state = ProjectState(models=[
            dict(key="CAD_0", kind="CAD", name="CAD", mesh=cube, style={}),
            dict(key="Result_0", kind="Result", name="one", mesh=cube.copy(), style={}),
            dict(key="Def_1", kind="Def", name="two", mesh=cube.copy(), style={}),
            dict(key="Heatmap_0", kind="Heatmap", name="map", mesh=cube.copy(), deviations=np.ones(8), style={}),
        ], active_result="Def_1", active_heatmap="Heatmap_0")
        self.window.restore_project(state)
        actual = self.window.capture_project()
        self.assertEqual(len(actual.models), 4)
        self.assertEqual(actual.active_result, "Def_1")
        np.testing.assert_array_equal(self.window.pv_heatmap["Deviation"], np.ones(8))
        self.window.clear_project_data()
        self.assertEqual(self.window.scene_models, {})
        self.assertEqual(self.window.actors, {})

    def test_async_save_roundtrip_with_markers_parts_and_settings(self):
        cube = trimesh.creation.box()
        self.window.restore_project(ProjectState(models=[dict(key="CAD_0", kind="CAD", name="CAD", mesh=cube, style={})],
            parts=[dict(filename="part.stl", mesh=cube.copy(), style={"last_visible_mode": "transparent", "transparency": 25})],
            cad_pts=[np.array([0.1, 0.2, 0.3])], settings={"samples": 0, "points": 3000, "linked": False, "factor_z": 2.0}))
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "project.mrp")
            self.window.project_path = path
            self.window.mark_dirty()
            self.window.save_project()
            self.wait_for_job()
            self.assertFalse(self.window.dirty)
            state = load_project(path)
            self.assertEqual(state.cad_pts, [[0.1, 0.2, 0.3]])
            self.assertEqual(state.parts[0]['style']['transparency'], 25)
            self.window.load_mrp_file(path)
            self.wait_for_job()
            self.assertTrue(self.window.ui.sb_points_wrapper.isEnabled())
            self.assertEqual(self.window.ui.sb_points.value(), 3000)
            actor = self.window.ui.slicer_plotter.actors['slicer_part_0']
            self.assertAlmostEqual(actor.GetProperty().GetOpacity(), 0.75)

    def test_active_map_controls_dataset_and_picking(self):
        cube = trimesh.creation.box()
        self.window.restore_project(ProjectState(models=[
            dict(key=f"Heatmap_{i}", kind="Heatmap", name=f"Map {i}", mesh=cube.copy(), deviations=np.full(8, i), style={})
            for i in range(2)], active_heatmap="Heatmap_1"))
        self.window.select_heatmap(0, 6)
        np.testing.assert_array_equal(self.window.pv_heatmap['Deviation'], np.zeros(8))
        self.window.ui.chk_callouts.blockSignals(True)
        self.window.ui.chk_callouts.setChecked(True)
        self.window.ui.chk_callouts.blockSignals(False)
        self.window._configure_heatmap_picking()
        self.assertTrue(self.window.actors['Heatmap_0'].pickable)
        self.assertFalse(self.window.actors['Heatmap_1'].pickable)

    def test_implemented_ribbon_commands_are_enabled(self):
        self.assertTrue(self.window.ui.ribbon_btns['Импорт детали'].isEnabled())
        self.assertFalse(self.window.ui.ribbon_btns['Создание срезов Concept Laser'].isEnabled())

    def test_alignment_thread_restores_controls_after_invalid_markers(self):
        self.window.cad_mesh = trimesh.creation.box()
        self.window.scan_mesh = trimesh.creation.box()
        self.window.cad_pts = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        self.window.scan_pts = list(self.window.cad_pts)
        self.window.run_icp()
        self.wait_for_job()
        self.assertTrue(self.window.ui.btn_run_icp.isEnabled())
        self.assertIn('прямой', self.window.ui.log_view.toPlainText())

    def test_platform_removal_does_not_reassign_parts_to_neighbor(self):
        self.window.platforms = [dict(id='a', name='A', dim=[10, 10, 10], is_default=True),
                                 dict(id='b', name='B', dim=[10, 10, 10], is_default=True)]
        self.window.slicer_parts = [dict(platform='A'), dict(platform='B')]
        class Dialog:
            def get_data(self):
                return [dict(id='b', name='B renamed', dim=[10, 10, 10], is_default=True)]
        with patch.object(self.window, 'update_platform_ui'), patch.object(self.window.settings, 'setValue'):
            self.window.apply_platform_settings(Dialog())
        self.assertEqual([p['platform'] for p in self.window.slicer_parts], [None, 'B renamed'])

    def test_sections_clip_display_and_survive_project_roundtrip(self):
        cube = trimesh.creation.box(extents=[10, 20, 30])
        self.window.restore_project(ProjectState(parts=[dict(mesh=cube, filename="cube.step")], page="slicer"))
        panel = self.window.ui.section_panel
        panel.table.cellWidget(0, 0).setChecked(True)
        panel.table.cellWidget(0, 4).setValue(3)
        panel.table.cellWidget(0, 5).setValue(.5)
        panel.move(1)
        self.assertEqual(panel.sections[0]["position"], 3.5)
        panel.add_plane()
        panel.table.cellWidget(3, 2).setCurrentText("−")
        panel.table.cellWidget(3, 4).setValue(2)
        actor = self.window.ui.slicer_plotter.actors["slicer_part_0"]
        self.assertEqual(actor.mapper.GetNumberOfClippingPlanes(), 2)
        planes = actor.mapper.GetClippingPlanes()
        self.assertTrue(all(planes.GetItem(i).EvaluateFunction([0, 0, 3]) >= 0 for i in range(2)))
        self.assertTrue(any(planes.GetItem(i).EvaluateFunction([0, 0, 5]) < 0 for i in range(2)))
        self.assertTrue(any(planes.GetItem(i).EvaluateFunction([0, 0, 0]) < 0 for i in range(2)))
        np.testing.assert_array_equal(self.window.slicer_parts[0]["mesh"].vertices, cube.vertices)
        self.window._apply_part_display_mode(0, "bbox")
        bbox = self.window.ui.slicer_plotter.actors["slicer_part_0__bbox"]
        self.assertEqual(bbox.mapper.GetNumberOfClippingPlanes(), 2)
        with tempfile.TemporaryDirectory() as folder:
            self.window.project_path = str(Path(folder) / "sections.mrp")
            self.window.save_project()
            self.wait_for_job()
            restored = load_project(self.window.project_path)
            self.assertEqual(restored.sections, panel.snapshot())
            self.window.restore_project(restored)
            self.assertEqual(self.window.ui.slicer_plotter.actors["slicer_part_0"].mapper.GetNumberOfClippingPlanes(), 2)
        self.window.clear_project_data()
        self.assertFalse(any(s["active"] for s in panel.sections))

    def test_section_arbitrary_plane_slider_export_and_import_with_active_plane(self):
        cube = trimesh.creation.box(extents=[10, 10, 10])
        self.window.restore_project(ProjectState(parts=[dict(mesh=cube, filename="cube.stl")], page="slicer"))
        panel = self.window.ui.section_panel
        panel.table.cellWidget(0, 0).setChecked(True)
        panel.table.cellWidget(0, 1).setCurrentText("Произв.")
        panel._drag([1, 1, 0], [0, 0, 0])
        plane = self.window.ui.slicer_plotter.actors["slicer_part_0"].mapper.GetClippingPlanes().GetItem(0)
        self.assertLess(plane.EvaluateFunction([1, 1, 0]), 0)
        self.assertGreater(plane.EvaluateFunction([-1, -1, 0]), 0)
        panel.slider.setValue(750)
        self.assertAlmostEqual(panel.sections[0]["position"], np.sqrt(50) / 2, places=3)
        self.window._append_slicer_part(cube.copy(), "new.step")
        self.assertEqual(self.window.ui.slicer_plotter.actors["slicer_part_1"].mapper.GetNumberOfClippingPlanes(), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "contour.vtp")
            with patch("section_panel.QFileDialog.getSaveFileName", return_value=(path, "")):
                panel.export_contours()
            self.assertGreater(pv.read(path).n_lines, 0)
        panel.reset()
        self.assertEqual(self.window.ui.slicer_plotter.actors["slicer_part_1"].mapper.GetNumberOfClippingPlanes(), 0)

    def test_step_import_in_both_pages_and_cancel_precision_dialog(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "part.step")
            writer = STEPControl_Writer()
            writer.Transfer(BRepPrimAPI_MakeBox(4, 5, 6).Shape(), STEPControl_AsIs)
            writer.Write(path)
            with patch("project_controller.QFileDialog.getOpenFileName", return_value=(path, "")), patch("import_dialog.StepImportDialog.exec", return_value=1), patch("import_dialog.StepImportDialog.values", return_value=(.05, .25)):
                self.window.load_cad()
                self.wait_for_job()
                self.window.import_slicer_part()
                self.wait_for_job()
            np.testing.assert_allclose(self.window.cad_mesh.extents, [4, 5, 6])
            np.testing.assert_allclose(self.window.slicer_parts[0]["mesh"].extents, [4, 5, 6])
            with patch("project_controller.QFileDialog.getOpenFileName", return_value=(path, "")), patch("import_dialog.StepImportDialog.exec", return_value=0):
                self.window.import_slicer_part()
            self.assertEqual(len(self.window.slicer_parts), 1)

    def test_section_selection_requires_click_and_ignores_hover_and_wheel(self):
        from PySide6.QtCore import QEvent, QPoint, QPointF
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtTest import QTest
        self.window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(), filename="box.stl")], page="slicer"))
        panel = self.window.ui.section_panel
        panel.select_row(0)
        first = panel.sections[0]['position']
        second = panel.table.cellWidget(1, 4)
        APP.sendEvent(second, QEvent(QEvent.Enter))
        wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        APP.sendEvent(second, wheel)
        self.assertEqual(panel._selected_row, 0)
        self.assertEqual(panel.sections[1]['position'], 0)
        QTest.mouseClick(second.lineEdit(), Qt.LeftButton)
        self.assertEqual(panel._selected_row, 1)
        panel.slider.setValue(800)
        self.assertEqual(panel.sections[0]['position'], first)
        self.assertAlmostEqual(panel.sections[1]['position'], .3)
        self.assertIn('2', panel.selection_label.text())

    def test_tools_transform_undo_redo_and_branch(self):
        cube = trimesh.creation.box(extents=[4, 6, 8])
        self.window.restore_project(ProjectState(parts=[dict(mesh=cube, filename='box.step', style={'color': '#aabbcc', 'transparency': 25})], page='slicer'))
        self.window.dirty = False
        self.window.reset_history()
        params = dict(values=[5, -2, 3], pivot=0)
        self.window.apply_slicer_tool('Перемещать', params, [0])
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].bounds.mean(axis=0), [5, -2, 3])
        self.assertTrue(self.window.ui.action_undo.isEnabled())
        self.window.undo_action()
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].vertices, cube.vertices)
        self.assertFalse(self.window.dirty)
        self.assertTrue(self.window.ui.action_redo.isEnabled())
        self.window.redo_action()
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].bounds.mean(axis=0), [5, -2, 3])
        self.window.undo_action()
        self.window.apply_slicer_tool('Масштабировать', dict(values=[2, 1, .5], pivot=0), [0])
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].extents, [8, 6, 4])
        self.assertFalse(self.window.ui.action_redo.isEnabled())
        self.assertAlmostEqual(self.window.ui.slicer_plotter.actors['slicer_part_0'].prop.opacity, .75)

    def test_create_duplicate_array_mirror_and_selection(self):
        from slicer_tools import TOOL_NAMES
        self.window.apply_slicer_tool('Создать', dict(kind='Цилиндр', values=[10, 10, 20], center=[0, 0, 10]), [])
        original = self.window.slicer_parts[0]['mesh'].copy()
        self.window.apply_slicer_tool('Дублировать', dict(counts=[2], values=[20, 0, 0]), [0])
        self.assertEqual(len(self.window.slicer_parts), 3)
        self.assertEqual(self.window.selected_slicer_rows(), [1, 2])
        self.window.undo_action()
        self.assertEqual(len(self.window.slicer_parts), 1)
        self.window.apply_slicer_tool('Пакетное дублирование', dict(counts=[2, 2, 1], values=[20, 30, 40]), [0])
        self.assertEqual(len(self.window.slicer_parts), 4)
        centers = [part['mesh'].bounds.mean(axis=0).tolist() for part in self.window.slicer_parts]
        self.assertIn([20, 30, 10], centers)
        self.window.apply_slicer_tool('Отзеркалить', dict(values=[0, 0, 1], pivot=2), [0])
        self.assertAlmostEqual(self.window.slicer_parts[0]['mesh'].bounds.mean(axis=0)[2], -10)
        self.assertGreater(self.window.slicer_parts[0]['mesh'].volume, 0)
        self.window.undo_action()
        np.testing.assert_array_equal(self.window.slicer_parts[0]['mesh'].vertices, original.vertices)
        self.assertTrue(all(self.window.ui.ribbon_btns[name].isEnabled() for name in TOOL_NAMES))

    def test_transform_preview_cancel_does_not_change_mesh_or_history(self):
        cube = trimesh.creation.box()
        self.window.restore_project(ProjectState(parts=[dict(mesh=cube, filename='box.stl')], page='slicer'))
        self.window.dirty = False
        self.window.reset_history()
        key = self.window.history.key
        self.window.run_slicer_tool('Перемещать')
        session = self.window._transform_session
        session.dialog.values[0].setValue(12)
        session.update_preview()
        self.assertAlmostEqual(self.window.ui.slicer_plotter.actors['slicer_part_0'].GetMatrix().GetElement(0, 3), 12)
        session.dialog.reject()
        np.testing.assert_array_equal(self.window.slicer_parts[0]['mesh'].vertices, cube.vertices)
        self.assertEqual(self.window.ui.slicer_plotter.actors['slicer_part_0'].GetMatrix().GetElement(0, 3), 0)
        self.assertEqual(self.window.history.key, key)
        self.assertFalse(self.window.dirty)

    def test_history_savepoint_sections_settings_and_busy_job(self):
        self.window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(), filename='box.stl')], page='slicer'))
        self.window.dirty = False
        self.window.reset_history()
        panel = self.window.ui.section_panel
        panel.table.cellWidget(0, 0).setChecked(True)
        panel.table.cellWidget(0, 4).setValue(.2)
        self.window.ui.sb_align_tolerance.setValue(.15)
        self.window.flush_history()
        with tempfile.TemporaryDirectory() as folder:
            self.window.project_path = str(Path(folder) / 'saved.mrp')
            self.window.save_project()
            self.assertFalse(self.window.ui.action_undo.isEnabled())
            self.wait_for_job()
            self.assertFalse(self.window.dirty)
            self.window.undo_action()
            self.assertFalse(panel.sections[0]['active'])
            self.assertEqual(self.window.ui.sb_align_tolerance.value(), 0)
            self.assertTrue(self.window.dirty)
            self.window.redo_action()
            self.assertTrue(panel.sections[0]['active'])
            self.assertEqual(self.window.ui.sb_align_tolerance.value(), .15)
            self.assertFalse(self.window.dirty)
            self.window.load_mrp_file(self.window.project_path)
            self.wait_for_job()
            self.assertFalse(self.window.ui.action_undo.isEnabled())

    def test_step_angle_dialog_converts_degrees_and_passes_to_import(self):
        from import_dialog import StepImportDialog
        dialog = StepImportDialog(self.window)
        dialog.angle.setValue(12)
        dialog.linear.setValue(.03)
        self.assertAlmostEqual(dialog.values()[1], np.deg2rad(12))
        self.assertEqual(dialog.values()[0], .03)
        with patch('import_dialog.StepImportDialog.exec', return_value=1), patch('import_dialog.StepImportDialog.values', return_value=(.03, np.deg2rad(12))), patch('project_controller.QFileDialog.getOpenFileName', return_value=('part.step', '')), patch('project_controller.load_mesh', return_value=trimesh.creation.box()) as loader:
            self.window.load_cad()
            self.wait_for_job()
            loader.assert_called_once_with('part.step', .03, np.deg2rad(12))

    def test_undo_alignment_restores_scan_markers_and_quality(self):
        cube = trimesh.creation.box()
        scan = cube.copy()
        scan.apply_translation([2, 0, 0])
        self.window.restore_project(ProjectState(models=[dict(key='CAD_0', kind='CAD', name='CAD', mesh=cube),
            dict(key='Scan_0', kind='Scan', name='Scan', mesh=scan)], cad_pts=[[0, 0, 0]], scan_pts=[[2, 0, 0]]))
        self.window.dirty = False
        self.window.reset_history()
        aligned = cube.copy()
        aligned.metadata['alignment'] = dict(rmse=.001, coverage=.99, tolerance=.02, p95=.001)
        self.window.on_icp_done((aligned, .001))
        self.window.flush_history('Совмещение')
        self.window.undo_action()
        np.testing.assert_array_equal(self.window.scan_mesh.vertices, scan.vertices)
        np.testing.assert_allclose(self.window.scan_pts, [[2, 0, 0]])
        self.assertNotIn('99', self.window.ui.lbl_align_quality.text())
        self.window.redo_action()
        np.testing.assert_array_equal(self.window.scan_mesh.vertices, cube.vertices)
        self.assertEqual(len(self.window.scan_pts), 0)
        self.assertIn('99', self.window.ui.lbl_align_quality.text())

    def test_detailed_move_apply_copy_reset_and_close(self):
        self.window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(extents=[2, 4, 6]), filename='box.stl')], page='slicer'))
        self.window.dirty = False
        self.window.reset_history()
        self.window.run_slicer_tool('Перемещать')
        session = self.window._transform_session
        dialog = session.dialog
        dialog.target[0].setValue(12)
        np.testing.assert_allclose(dialog.numbers(dialog.values), [12, 0, 0])
        self.assertFalse(self.window.ui.action_undo.isEnabled())
        dialog.copy.setChecked(True)
        session.update_preview()
        self.assertEqual(len(session.ghosts), 1)
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].bounds.mean(axis=0), [0, 0, 0])
        session.apply(False)
        self.assertEqual(len(self.window.slicer_parts), 2)
        np.testing.assert_allclose(self.window.slicer_parts[1]['mesh'].bounds.mean(axis=0), [12, 0, 0])
        dialog.copy.setChecked(False)
        dialog.reset_move(False)
        np.testing.assert_allclose(dialog.numbers(dialog.values), [-12, 0, 0])
        session.apply(False)
        np.testing.assert_allclose(self.window.slicer_parts[1]['mesh'].bounds.mean(axis=0), [0, 0, 0])
        dialog.reject()
        self.window.undo_action()
        np.testing.assert_allclose(self.window.slicer_parts[1]['mesh'].bounds.mean(axis=0), [12, 0, 0])
        self.window.undo_action()
        self.assertEqual(len(self.window.slicer_parts), 1)

    def test_scale_size_difference_measure_and_presets(self):
        self.window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(extents=[10, 20, 30]), filename='box.stl')], page='slicer'))
        self.window.run_slicer_tool('Масштабировать')
        session = self.window._transform_session
        dialog = session.dialog
        dialog.final_size[0].setValue(15)
        self.assertEqual(dialog.values[0].value(), 1.5)
        self.assertEqual(dialog.difference[0].value(), 5)
        dialog.difference[1].setValue(-10)
        self.assertEqual(dialog.values[1].value(), .5)
        dialog.uniform.setChecked(True)
        dialog.values[2].setValue(2)
        np.testing.assert_allclose(dialog.numbers(dialog.final_size), [20, 40, 60])
        dialog.fit.setChecked(True)
        dialog.fit_target.setValue(10)
        dialog.accept_points('measure', [[0, 0, 0], [0, 0, 5]])
        np.testing.assert_allclose(dialog.parameters()['values'], [2, 2, 2])
        dialog.fit_difference.setValue(10)
        np.testing.assert_allclose(dialog.parameters()['values'], [3, 3, 3])
        dialog.use_preset(0)
        np.testing.assert_allclose(dialog.parameters()['values'], [25.4] * 3)
        dialog.keep_z.setChecked(True)
        session.apply(False)
        self.assertAlmostEqual(self.window.slicer_parts[0]['mesh'].bounds[0, 2], -15)
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].extents, [254, 508, 762])
        dialog.reject()

    def test_rotation_line_and_custom_mirror_dialogs(self):
        cube = trimesh.creation.box(extents=[2, 4, 6])
        self.window.restore_project(ProjectState(parts=[dict(mesh=cube, filename='box.stl')], page='slicer'))
        self.window.run_slicer_tool('Вращать')
        session = self.window._transform_session
        dialog = session.dialog
        dialog.accept_points('line', [[0, 0, 0], [0, 0, 1]])
        dialog.line_angle.setValue(90)
        session.apply(True)
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].extents, [4, 2, 6])
        self.window.run_slicer_tool('Отзеркалить')
        session = self.window._transform_session
        dialog = session.dialog
        dialog.accept_points('plane', [[1, 0, 0], [1, 1, 0], [1, 0, 1]])
        dialog.copy.setChecked(True)
        session.apply(True)
        self.assertEqual(len(self.window.slicer_parts), 2)
        np.testing.assert_allclose(self.window.slicer_parts[1]['mesh'].bounds.mean(axis=0), [2, 0, 0])
        self.assertGreater(self.window.slicer_parts[1]['mesh'].volume, 0)

    def test_preset_crud_and_home_icons(self):
        from transform_dialog import TransformDialog
        dialog = TransformDialog('Масштабировать', [trimesh.creation.box()], self.window)
        with patch.object(self.window.settings, 'setValue') as saved:
            with patch('transform_dialog.QInputDialog.getText', return_value=('Custom', True)), patch('transform_dialog.QDialog.exec', return_value=1):
                dialog.edit_preset('new')
                self.assertEqual(dialog.library[-1]['name'], 'Custom')
                dialog.presets.setCurrentRow(len(dialog.library) - 1)
                dialog.edit_preset('edit')
                dialog.presets.setCurrentRow(len(dialog.library) - 1)
                dialog.edit_preset('delete')
            self.assertEqual(saved.call_count, 3)
        for name in ['Новый проект', 'Загрузить проект', 'Сохранить проект', 'Импорт детали', 'Выгрузить деталь']:
            self.assertFalse(self.window.ui.ribbon_btns[name].icon().isNull())
        dialog.close()


    def test_unload_only_selected_preserves_style_sections_and_reindexed_controls(self):
        from PySide6.QtWidgets import QCheckBox
        window = self.window
        parts = [dict(mesh=trimesh.creation.box(), filename=f'{i}.stl',
                      style=dict(is_selected=i == 1, color='#804020', transparency=25)) for i in range(3)]
        window.restore_project(ProjectState(parts=parts, page='slicer'))
        window.ui.section_panel.table.cellWidget(0, 0).setChecked(True)
        sections = window.ui.section_panel.snapshot()
        window.reset_history()
        window.unload_slicer_part()
        self.assertEqual([p['filename'] for p in window.slicer_parts], ['0.stl', '2.stl'])
        self.assertEqual(window.ui.section_panel.snapshot(), sections)
        self.assertEqual(window._style_for(window.ui.tbl_parts, 1)['transparency'], 25)
        self.assertEqual(window.slicer_parts[1]['actor_name'], 'slicer_part_1')
        window.ui.tbl_parts.cellWidget(1, 2).findChild(QCheckBox).setChecked(False)
        self.assertFalse(window.ui.slicer_plotter.actors['slicer_part_1'].GetVisibility())
        window.undo_action()  # undo visibility
        window.undo_action()  # undo unload
        self.assertEqual([p['filename'] for p in window.slicer_parts], ['0.stl', '1.stl', '2.stl'])
        window.redo_action()
        self.assertEqual(len(window.slicer_parts), 2)

    def test_unload_and_export_ignore_checked_rows_outside_current_scene(self):
        window = self.window
        window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(), filename=f'{i}.stl') for i in range(2)]))
        window.ui.tbl_parts.setRowHidden(1, True)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'selected.stl')
            with patch('Meshropractor.QFileDialog.getSaveFileName', return_value=(path, '')):
                window.save_selected_slicer_parts()
            self.assertEqual(len(trimesh.load(path).faces), 12)
        window.unload_slicer_part()
        self.assertEqual([p['filename'] for p in window.slicer_parts], ['1.stl'])

    def test_surface_overlay_selection_modes_and_sections(self):
        window = self.window
        window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(), filename='part.stl')]))
        work = window.workspace_tools
        work.edit_selection(0, {0, 1}, 'replace')
        work.edit_selection(0, {2}, 'add')
        work.edit_selection(0, {1}, 'subtract')
        self.assertEqual(work.selection, {0: {0, 2}})
        actor = work.overlays[0][0]
        self.assertEqual(actor.mapper.dataset.n_cells, 2)
        window.ui.section_panel.table.cellWidget(0, 0).setChecked(True)
        self.assertEqual(actor.GetMapper().GetNumberOfClippingPlanes(), 1)
        window.replace_slicer_mesh(0, trimesh.creation.icosphere(subdivisions=1))
        self.assertFalse(work.selection)
        self.assertFalse(work.overlays)

    def test_support_preview_generation_history_and_project_roundtrip(self):
        from support_geometry import generate_supports
        from support_tools import DEFAULTS
        window = self.window
        cube = trimesh.creation.box([6, 6, 2])
        cube.apply_translation([0, 0, 11])
        window.restore_project(ProjectState(parts=[dict(mesh=cube, filename='part.stl')]))
        work = window.workspace_tools
        work.plotter = window.ui.slicer_plotter
        work.supports.preview_regions({0: None})
        self.assertEqual(work.preview[0][0].mapper.dataset.n_cells, 2)
        work.supports.clear_preview()
        window.reset_history()
        records = [dict(row=0, mesh=cube, filename='part.stl', platform=None)]
        window.start_job(FunctionWorker(generate_supports, records, {0: None}, DEFAULTS), work.supports.append_results)
        self.wait_for_job()
        self.assertEqual(len(window.slicer_parts), 1)
        self.assertEqual(len(window.slicer_parts[0]['supports']), 1)
        window.undo_action()
        self.assertEqual(len(window.slicer_parts), 1)
        self.assertFalse(window.slicer_parts[0]['supports'])
        window.redo_action()
        self.assertEqual(len(window.slicer_parts[0]['supports']), 1)
        with tempfile.TemporaryDirectory() as folder:
            from project_store import save_project
            path = str(Path(folder) / 'supports.mrp')
            save_project(path, window.capture_project())
            from part_supports import support_mesh
            self.assertTrue(support_mesh(load_project(path).parts[0]['supports'][0]).is_watertight)

    def test_radial_actions_and_support_platform_icons(self):
        from radial_menu import RadialMenu
        from workspace_icons import SUPPORT_NAMES
        from PySide6.QtCore import QPoint
        window = self.window
        window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(), filename='part.stl')]))
        with patch.object(window, 'run_slicer_tool') as run:
            menu = RadialMenu(window)
            menu.popup(QPoint(1, 1))
            self.assertGreaterEqual(menu.x(), 0)
            menu.buttons[0].click()
            APP.processEvents()
            run.assert_called_once_with('Перемещать')
            menu.deleteLater()
        for name in SUPPORT_NAMES + ['Управление платформами']:
            self.assertTrue(window.ui.ribbon_btns[name].isEnabled())
            self.assertFalse(window.ui.ribbon_btns[name].icon().isNull())


    def test_manual_panel_owns_regions_rebuilds_and_deletes_with_undo(self):
        from support_geometry import overhang_faces
        window = self.window
        mesh = trimesh.creation.box([6, 6, 2])
        mesh.apply_translation([0, 0, 11])
        window.restore_project(ProjectState(parts=[dict(mesh=mesh, filename='body.stl')], page='slicer'))
        window.workspace_tools.supports.open(3)
        panel = window.workspace_tools.supports.panel
        self.assertFalse(panel.isHidden())
        self.assertTrue(all(group.isHidden() for group in window.ui.slicer_normal_groups))
        window.workspace_tools.edit_selection(0, set(overhang_faces(mesh)), 'replace')
        panel.add_region()
        self.assertEqual(len(window.slicer_parts[0]['supports']), 1)
        self.assertEqual(panel.table.rowCount(), 1)
        with tempfile.TemporaryDirectory() as folder:
            from project_store import save_project
            path = str(Path(folder) / 'empty-region.mrp')
            save_project(path, window.capture_project())
            restored = load_project(path)
            self.assertEqual(restored.parts[0]['supports'][0]['surface_faces'], list(map(int, overhang_faces(mesh))))
            self.assertEqual(restored.parts[0]['supports'][0]['faces'].shape, (0,3))
        panel.kind.setCurrentText('Блок')
        panel.rebuild()
        self.wait_for_job()
        self.assertEqual(len(window.slicer_parts), 1)
        group = window.slicer_parts[0]['supports'][0]
        self.assertTrue(len(group['faces']) > 0)
        self.assertEqual(group['kind'], 'Блок')
        panel.delete_region()
        self.assertFalse(window.slicer_parts[0]['supports'])
        window.undo_action()
        self.assertEqual(len(window.slicer_parts[0]['supports']), 1)
        panel.finish()
        self.assertTrue(panel.isHidden())

    def test_supports_follow_transforms_copies_and_export(self):
        from part_supports import make_group, support_mesh
        from support_geometry import column
        from support_tools import DEFAULTS
        window = self.window
        child = make_group(column([0,0,0], [0,0,10], DEFAULTS), [0])
        body = trimesh.creation.box()
        body.apply_translation([0,0,10])
        window.restore_project(ProjectState(parts=[dict(mesh=body, filename='body.stl', supports=[child])]))
        window.apply_slicer_tool('Перемещать', dict(advanced=True, values=[5,0,0], absolute=False,
                                                   along_line=False, individual=False, anchor_modes=[1,1,1], anchor_custom=[0,0,0]), [0])
        np.testing.assert_allclose(support_mesh(window.slicer_parts[0]['supports'][0]).bounds.mean(axis=0)[0], 5)
        window.apply_slicer_tool('Дублировать', dict(counts=[1,1,1], values=[3,0,0]), [0])
        copied = window.slicer_parts[1]['supports'][0]
        self.assertNotEqual(copied['id'], window.slicer_parts[0]['supports'][0]['id'])
        self.assertAlmostEqual(support_mesh(copied).bounds.mean(axis=0)[0], 8)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'attached.stl')
            with patch('Meshropractor.QFileDialog.getSaveFileName', return_value=(path, '')): window.save_selected_slicer_parts()
            self.assertGreater(len(trimesh.load(path).faces), len(body.faces))

    def test_measurement_panel_numeric_results_and_cleanup(self):
        window = self.window
        window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box([10,10,10]), filename='box.stl')]))
        work = window.workspace_tools
        work.plotter = window.ui.slicer_plotter
        panel = work.measurements
        panel.active = (0, 0)
        panel.hits = [(0,0,np.array([0.,0,0])), (0,1,np.array([3.,4,0]))]
        panel.calculate()
        self.assertIn('5.0000', panel.results.item(0).text())
        panel.active = (1, 0)
        panel.hits = [(0,0,np.array(p, dtype=float)) for p in ([2,0,0], [0,2,0], [-2,0,0])]
        panel.calculate()
        self.assertIn('R=2.0000', panel.results.item(1).text())
        panel.active = (0, 2)
        panel.hits = [(0,0,np.array([10.,0,0])), (0,1,np.array([5.,1,0]))]
        panel.calculate()
        self.assertIn('5.0000', panel.results.item(2).text())
        panel.active = (0, 1)
        panel.hits = [(0,0,np.array([10.,0,0])), (0,0,np.array([-5.,0,0]))]
        panel.calculate()
        self.assertIn('15.0000', panel.results.item(3).text())
        panel.set_hidden(True)
        self.assertTrue(all(not actor.GetVisibility() for name, actor in work.plotter.actors.items() if name.startswith('measurement_')))
        window.replace_slicer_mesh(0, trimesh.creation.box())
        self.assertEqual(panel.results.count(), 0)


if __name__ == "__main__": unittest.main()
