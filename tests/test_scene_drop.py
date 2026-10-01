"""Real Qt drop events and asynchronous imports into the captured scene."""
import time
from pathlib import Path
import unittest
from unittest.mock import patch

import trimesh
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QDialog

import test_cad_tools
from test_desktop import APP
from cad_state import cad_status
from scene_drop import SceneDropController


class SceneDropTests(unittest.TestCase):
    setUpClass = classmethod(test_cad_tools.CADToolsTests.setUpClass.__func__)
    tearDownClass = classmethod(test_cad_tools.CADToolsTests.tearDownClass.__func__)
    setUp = test_cad_tools.CADToolsTests.setUp
    close_window = test_cad_tools.CADToolsTests.close_window
    wait_job = test_cad_tools.CADToolsTests.wait_job

    def stl(self, name='деталь.STL'):
        path = Path(self.folder.name) / name
        trimesh.creation.box(extents=[4, 5, 6]).export(path, file_type='stl')
        return path

    def drop(self, paths, scope='slicer'):
        ui = self.window.ui
        ui.stack.setCurrentWidget(ui.page_slicer if scope == 'slicer' else ui.page_predef)
        target = ui._slicer_center_container if scope == 'slicer' else ui._def_center_container
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
        enter = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(target, enter)
        event = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(target, event)
        return enter, event

    def drain(self):
        deadline = time.monotonic() + 45
        while (self.window._job is not None or self.window.drop_imports.pending) and time.monotonic() < deadline:
            APP.processEvents(); time.sleep(.005)
        self.assertIsNone(self.window._job, self.messages)
        self.assertFalse(self.window.drop_imports.pending, self.messages)
        for _ in range(3): APP.processEvents()

    def test_real_drop_multiple_stl_captures_platform_and_uses_workers_without_file_dialog(self):
        paths = [self.stl(), self.stl('вторая.stl')]
        self.window.platforms = [dict(name='Платформа A', dim=[50, 50, 60], is_default=True)]
        self.window.update_platform_ui(draw=False)
        self.window.ui.scene_tabs.setCurrentIndex(1)
        with patch('mesh_repair.request_repair', return_value=False), \
             patch('project_controller.QFileDialog.getOpenFileName', side_effect=AssertionError('Unexpected file dialog')):
            enter, drop = self.drop(paths)
            self.assertTrue(enter.isAccepted()); self.assertTrue(drop.isAccepted())
            self.window.ui.scene_tabs.setCurrentIndex(0)
            self.drain()
        self.assertEqual(len(self.window.slicer_parts), 2, self.messages)
        self.assertTrue(all(p['platform'] == 'Платформа A' for p in self.window.slicer_parts))
        self.assertEqual([p['filename'] for p in self.window.slicer_parts], [p.name for p in paths])
        self.window.undo_action(); self.assertEqual(len(self.window.slicer_parts), 1)

    def test_step_drop_choices_keep_brep_or_load_stl_proxy(self):
        with patch('import_dialog.StepImportDialog.exec', return_value=1), \
             patch('mesh_repair.request_repair') as healing:
            self.drop([self.assembly_path]); self.drain()
        healing.assert_not_called()
        self.assertEqual(len(self.window.slicer_parts), 2)
        self.assertTrue(all(cad_status(p['mesh']) == 'native' for p in self.window.slicer_parts))
        def mesh_choice(dialog):
            dialog.mesh.click()
            dialog.mesh.click()  # Clicking an active mode must not deselect it.
            self.assertTrue(dialog.mesh.isChecked())
            self.assertFalse(dialog.native.isChecked())
            self.assertFalse(dialog.split_bodies.isEnabled())
            return 1
        with patch('import_dialog.StepImportDialog.exec', new=mesh_choice), \
             patch('mesh_repair.request_repair', return_value=False):
            self.drop([self.box_path]); self.drain()
        self.assertEqual(cad_status(self.window.slicer_parts[-1]['mesh']), 'mesh')

    def test_predeformation_drop_adds_scans_and_native_cad_on_captured_page(self):
        with patch('scene_drop.QInputDialog.getItem', return_value=('Фактическая модель (скан)', True)), \
             patch('mesh_repair.request_repair', return_value=False):
            self.drop([self.stl('scan1.stl'), self.stl('scan2.stl')], 'predef')
            self.window.ui.stack.setCurrentWidget(self.window.ui.page_slicer)
            self.drain()
        self.assertEqual(self.window.ui.tbl_scan.rowCount(), 2)
        self.assertEqual(len(self.window.slicer_parts), 0)
        with patch('import_dialog.StepImportDialog.exec', return_value=1), \
             patch('scene_drop.QInputDialog.getItem', side_effect=AssertionError('STEP must be CAD')):
            self.drop([self.assembly_path], 'predef'); self.drain()
        self.assertEqual(self.window.ui.tbl_cad.rowCount(), 1)
        self.assertEqual(cad_status(self.window.cad_mesh), 'native')
        self.assertEqual(len(self.window.scene_models), 3)

    def test_unsupported_remote_missing_and_duplicate_paths(self):
        path = self.stl()
        other = Path(self.folder.name) / 'text.txt'; other.write_text('test')
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path)), QUrl.fromLocalFile(str(path)),
                      QUrl.fromLocalFile(str(other)), QUrl('https://example.com/file.stl'),
                      QUrl.fromLocalFile(str(other.parent / 'missing.stl')), QUrl.fromLocalFile(str(other.parent))])
        self.assertEqual(SceneDropController.paths(mime), [str(path)])
        enter, drop = self.drop([other])
        self.assertFalse(enter.isAccepted()); self.assertFalse(drop.isAccepted())
        self.assertFalse(self.window.drop_imports.pending)

    def test_cancel_role_or_step_dialog_keeps_scene_unchanged(self):
        with patch('scene_drop.QInputDialog.getItem', return_value=('', False)):
            self.drop([self.stl()], 'predef'); self.drain()
        with patch('import_dialog.StepImportDialog.exec', return_value=0):
            self.drop([self.box_path]); self.drain()
        self.assertFalse(self.window.slicer_parts)
        self.assertFalse(self.window.scene_models)

    def test_busy_drop_rejected_and_cancelling_worker_stops_queue(self):
        paths = [self.stl(), self.stl('next.stl')]
        with patch('mesh_repair.request_repair', return_value=False):
            self.drop(paths)
            self.window.drop_imports._advance()
            self.assertIsNotNone(self.window._job)
            enter, drop = self.drop([self.box_path])
            self.assertFalse(enter.isAccepted()); self.assertFalse(drop.isAccepted())
            self.window.cancel_current_job()
            self.drain()
        self.assertFalse(any(p['filename'] == 'next.stl' for p in self.window.slicer_parts))

    def test_new_project_generation_and_modal_dialog_pause_pending_queue(self):
        self.drop([self.box_path])
        modal = QDialog(self.window); modal.setModal(True); modal.show()
        try:
            self.window.drop_imports._advance()
            self.assertFalse(self.window.slicer_parts)
            self.assertTrue(self.window.drop_imports.pending)
        finally:
            modal.close(); modal.deleteLater()
        self.window._generation += 1
        self.window.drop_imports._advance()
        self.assertFalse(self.window.drop_imports.pending)

    def test_failed_file_does_not_block_next_import(self):
        bad = Path(self.folder.name) / 'broken.stl'; bad.write_text('invalid')
        with patch('mesh_repair.request_repair', return_value=False):
            self.drop([bad, self.stl()]); self.drain()
        self.assertEqual(len(self.window.slicer_parts), 1)
        self.assertTrue(any('[!]' in line for line in self.messages), self.messages)

    def test_predeformation_healing_replaces_only_new_drop_and_preserves_undo(self):
        source = trimesh.creation.box(extents=[4, 5, 6])
        repaired = source.copy(); repaired.apply_translation([0, 0, 2])
        report = dict(defects=True, changed=True, before=dict(boundary=4), after=dict(boundary=0))
        def repair(*args, **kwargs):
            return source, repaired, report
        with patch('scene_drop.QInputDialog.getItem', return_value=('Фактическая модель (скан)', True)), \
             patch('mesh_repair.request_repair', return_value={}), \
             patch('mesh_repair.load_with_repair', new=repair), \
             patch('mesh_repair.choose_repair', return_value='apply'):
            self.drop([self.stl()], 'predef'); self.drain()
        self.assertEqual(self.window.ui.tbl_scan.rowCount(), 1)
        self.assertEqual(len(self.window.scene_models), 1)
        self.assertAlmostEqual(self.window.scan_mesh.bounds[0, 2], -1)
        self.window.undo_action()
        self.assertEqual(self.window.ui.tbl_scan.rowCount(), 1)
        self.assertAlmostEqual(self.window.scan_mesh.bounds[0, 2], -3)
