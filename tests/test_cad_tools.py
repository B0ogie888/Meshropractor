"""CAD import, support ownership, placement and undo through the desktop controller."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtWidgets import QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_desktop import APP
from test_repair_tools import RepairPlotter
import test_cad_import
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Pnt
from Meshropractor import MainWindow
from cad_import import load_step
from cad_state import cad_status, require_native, cad_face_triangles
from cad_tools import prepare_cad_changes
from project_store import ProjectState, load_project, save_project
from support_geometry import generate_supports
from support_tools import DEFAULTS


class CADToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.box_path = Path(cls.folder.name) / 'CAD_деталь.step'
        cls.cylinder_path = Path(cls.folder.name) / 'цилиндр.step'
        cls.assembly_path = Path(cls.folder.name) / 'сборка.step'
        case = test_cad_import.CADImportTests()
        case.write_step(BRepPrimAPI_MakeBox(gp_Pnt(-3, -3, 10), 6, 6, 2).Shape(), cls.box_path)
        case.write_step(BRepPrimAPI_MakeCylinder(3, 5).Shape(), cls.cylinder_path)
        compound = TopoDS_Compound(); builder = BRep_Builder(); builder.MakeCompound(compound)
        builder.Add(compound, BRepPrimAPI_MakeBox(5, 6, 7).Shape())
        builder.Add(compound, BRepPrimAPI_MakeBox(gp_Pnt(20, 0, 0), 4, 4, 4).Shape())
        case.write_step(compound, cls.assembly_path)

    @classmethod
    def tearDownClass(cls): cls.folder.cleanup()

    def setUp(self):
        self.window = MainWindow()
        self.window.ui._ensure_def_plotter = lambda: setattr(self.window.ui, 'plotter', self.window.ui.plotter or RepairPlotter())
        self.window.ui._ensure_slicer_plotter = lambda: setattr(self.window.ui, 'slicer_plotter', self.window.ui.slicer_plotter or RepairPlotter())
        self.window.add_to_recent = lambda path: None
        self.messages = []; self.window.log = self.messages.append
        self.addCleanup(self.close_window)

    def close_window(self):
        if self.window._job is not None:
            self.window.cancel_current_job(); self.wait_job()
        if self.window.cad_tools.dialog is not None: self.window.cad_tools.dialog.close()
        self.window.dirty = False; self.window.close(); self.window.deleteLater(); APP.processEvents()

    def wait_job(self):
        deadline = time.monotonic() + 45
        while self.window._job is not None and time.monotonic() < deadline:
            APP.processEvents(); time.sleep(.005)
        self.assertIsNone(self.window._job, self.messages)
        for _ in range(3): APP.processEvents()

    def load(self, path=None):
        mesh = load_step(path or self.box_path)
        self.window.restore_project(ProjectState(parts=[dict(mesh=mesh, filename='CAD.step')], page='slicer'))
        self.window.dirty = False; self.window.reset_history()
        self.window.workspace_tools.plotter = self.window.ui.slicer_plotter
        return self.window.slicer_parts[0]['mesh']

    def test_native_import_skips_mesh_healing_and_splits_assembly_in_one_undo(self):
        self.window.ui.stack.setCurrentWidget(self.window.ui.page_slicer)
        with patch('project_controller.QFileDialog.getOpenFileName', return_value=(str(self.assembly_path), '')), \
             patch('import_dialog.StepImportDialog.exec', return_value=1), \
             patch('mesh_repair.request_repair') as healing:
            self.window.import_slicer_part(); self.wait_job()
        healing.assert_not_called()
        self.assertEqual(len(self.window.slicer_parts), 2, self.messages)
        self.assertTrue(all(cad_status(p['mesh']) == 'native' for p in self.window.slicer_parts))
        np.testing.assert_allclose(self.window.slicer_parts[1]['mesh'].bounds[0], [20, 0, 0], atol=1e-7)
        self.window.undo_action(); self.assertEqual(len(self.window.slicer_parts), 0)
        self.window.redo_action(); self.assertEqual(len(self.window.slicer_parts), 2)

    def test_cad_face_selection_supports_remesh_and_project_roundtrip(self):
        mesh = self.load()
        triangle = int(np.flatnonzero(mesh.face_normals[:, 2] < -.99)[0])
        workspace = self.window.workspace_tools
        workspace.set_mode('cad_face')
        with patch.object(workspace, 'picker', return_value=(0, triangle, mesh.triangles_center[triangle])):
            workspace.select_at(None, 'replace')
        expected = set(map(int, cad_face_triangles(mesh, triangle)))
        self.assertEqual(workspace.selection[0], expected)
        result = generate_supports(workspace.supports.records(), {0: sorted(expected)}, DEFAULTS)
        self.assertTrue(result)
        workspace.supports.append_results(result)
        self.assertEqual(len(self.window.slicer_parts), 1)
        group = deepcopy(self.window.slicer_parts[0]['supports'][0])
        self.assertIn('cad_binding', group)
        targets = self.window.cad_tools.targets()
        self.window.cad_tools.apply_changes(prepare_cad_changes(targets, 'quality', (.01, .1)), 'Качество CAD')
        part = self.window.slicer_parts[0]
        self.assertEqual(cad_status(part['mesh']), 'native')
        self.assertEqual(part['supports'][0]['id'], group['id'])
        np.testing.assert_array_equal(part['supports'][0]['vertices'], group['vertices'])
        self.assertTrue(part['supports'][0]['surface_faces'])
        archive = Path(self.folder.name) / 'supports.mrp'
        save_project(archive, self.window.capture_project())
        loaded = load_project(archive).parts[0]
        self.assertEqual(cad_status(loaded['mesh']), 'native')
        from project_history import digest
        self.assertEqual(digest(loaded['supports'][0]['cad_binding']), digest(part['supports'][0]['cad_binding']))

    def test_transform_duplicate_mirror_and_undo_keep_brep_exportable(self):
        original = self.load()
        self.window.apply_slicer_tool('Масштабировать', dict(values=[2, 1, .5], pivot=0), [0])
        self.assertEqual(cad_status(self.window.slicer_parts[0]['mesh']), 'native')
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].extents, original.extents * [2, 1, .5])
        self.window.apply_slicer_tool('Дублировать', dict(counts=[1], values=[20, 0, 0]), [0])
        self.assertTrue(all(cad_status(p['mesh']) == 'native' for p in self.window.slicer_parts))
        self.window.apply_slicer_tool('Отзеркалить', dict(values=[1, 0, 0], pivot=2), [1])
        self.assertEqual(cad_status(self.window.slicer_parts[1]['mesh']), 'native')
        self.window.undo_action(); self.window.redo_action()
        self.assertEqual(cad_status(self.window.slicer_parts[1]['mesh']), 'native')

    def test_quality_command_runs_worker_and_increases_curved_mesh_resolution(self):
        source = self.load(self.cylinder_path)
        with patch('import_dialog.StepImportDialog.exec', return_value=1), \
             patch('import_dialog.StepImportDialog.values', return_value=(.001, .02)):
            self.window.cad_tools.quality(); self.wait_job()
        result = self.window.slicer_parts[0]['mesh']
        self.assertGreater(len(result.faces), len(source.faces), self.messages)
        self.assertEqual(cad_status(result), 'native')
        self.window.undo_action()
        np.testing.assert_array_equal(self.window.slicer_parts[0]['mesh'].vertices, source.vertices)

    def test_selected_step_export_and_modified_mesh_refusal(self):
        source = self.load()
        path = Path(self.folder.name) / 'выбранные.step'
        with patch('Meshropractor.QFileDialog.getSaveFileName', return_value=(str(path), 'STEP — CAD-тела (*.step *.stp)')):
            self.window.save_selected_slicer_parts(); self.wait_job()
        self.assertTrue(path.exists(), self.messages)
        np.testing.assert_allclose(load_step(path).bounds, source.bounds, atol=1e-6)
        previous = path.read_bytes()
        self.window.slicer_parts[0]['mesh'].vertices[0] += [1, 0, 0]
        self.window.cad_tools.export(str(path))
        self.assertIsNone(self.window._job)
        self.assertEqual(path.read_bytes(), previous)

    def test_convert_keeps_supports_and_undo_restores_native(self):
        mesh = self.load()
        self.window.workspace_tools.supports.append_results(generate_supports(
            [dict(row=0, mesh=mesh, filename='CAD.step', platform=None)], {0: None}, DEFAULTS))
        groups = deepcopy(self.window.slicer_parts[0]['supports'])
        with patch('cad_tools.QMessageBox.question', return_value=QMessageBox.Yes): self.window.cad_tools.convert()
        part = self.window.slicer_parts[0]
        self.assertEqual(cad_status(part['mesh']), 'mesh')
        self.assertNotIn('cad_binding', part['supports'][0])
        np.testing.assert_array_equal(part['supports'][0]['vertices'], groups[0]['vertices'])
        self.window.undo_action()
        self.assertEqual(cad_status(self.window.slicer_parts[0]['mesh']), 'native')
        self.assertIn('cad_binding', self.window.slicer_parts[0]['supports'][0])

    def test_predeformation_import_retains_combined_cad_and_can_change_quality(self):
        with patch('project_controller.QFileDialog.getOpenFileName', return_value=(str(self.assembly_path), '')), \
             patch('import_dialog.StepImportDialog.exec', return_value=1), \
             patch('mesh_repair.request_repair') as healing:
            self.window.load_cad(); self.wait_job()
        healing.assert_not_called()
        self.assertEqual(cad_status(self.window.cad_mesh), 'native')
        self.assertEqual(len(require_native(self.window.cad_mesh)['bodies']), 2)
        from geometry_analysis import compute_heatmap
        cad = self.window.cad_mesh
        scan = trimesh.Trimesh(vertices=np.asarray(cad.vertices) + [0, 0, .02], faces=cad.faces[::3], process=False)
        self.assertFalse(scan.is_watertight)
        deviations = compute_heatmap(cad, scan)
        self.assertEqual(len(deviations), len(scan.vertices))
        self.assertTrue(np.isfinite(deviations).all())
        self.assertEqual(cad_status(cad), 'native')
        self.window.ui.stack.setCurrentWidget(self.window.ui.page_predef)
        targets = self.window.cad_tools.targets()
        self.assertEqual(len(targets), 1)
        self.window.cad_tools.apply_changes(prepare_cad_changes(targets, 'quality', (.1, .2)), 'CAD')
        self.assertEqual(cad_status(self.window.cad_mesh), 'native')


if __name__ == '__main__': unittest.main()
