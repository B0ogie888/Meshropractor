"""A partial CAD binding must not rebuild from its remaining complete faces."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from part_supports import make_group
from support_panel import SupportPanel
from support_tools import DEFAULTS

APP = QApplication.instance() or QApplication([])


class CADSupportPanelTests(unittest.TestCase):
    def setUp(self):
        mesh = trimesh.creation.box()
        group = make_group(trimesh.creation.box(extents=[.2, .2, .5]), [0, 1], params=DEFAULTS)
        group['cad_binding'] = dict(requires_reselect=True, notice='Выберите частичную область заново.')
        self.jobs = []
        self.window = SimpleNamespace(slicer_parts=[dict(mesh=mesh, filename='CAD.step', supports=[group])],
            settings=SimpleNamespace(value=lambda key, fallback: fallback), selected_slicer_rows=lambda: [0],
            _busy=lambda: False, start_job=lambda worker, callback: self.jobs.append((worker, callback)))
        self.workspace = SimpleNamespace(selection={0: {0, 1}}, mode='part')
        def edit_selection(row, ids, operation):
            self.workspace.selection.clear()
            if ids: self.workspace.selection[row] = set(ids)
        self.workspace.edit_selection = edit_selection
        self.workspace.set_mode = lambda value: setattr(self.workspace, 'mode', value)
        self.tools = SimpleNamespace(window=self.window, workspace=self.workspace, params=dict(DEFAULTS),
            records=lambda: [dict(mesh=mesh, row=0, filename='CAD.step')], append_results=lambda results: None)
        self.panel = SupportPanel(self.tools)
        self.panel.refresh()
        self.addCleanup(self.cleanup_panel)

    def cleanup_panel(self):
        self.panel.close(); self.panel.deleteLater(); APP.processEvents()
        self.jobs.clear()

    def test_pending_group_clears_stale_selection_and_never_uses_complete_face_fallback(self):
        self.assertEqual(self.workspace.selection, {})
        self.assertEqual(self.panel.faces(), [])
        self.panel.rebuild()
        self.assertEqual(self.jobs, [])
        self.assertIn('заново', self.panel.status.text())
        self.assertEqual(self.panel.current_group()['surface_faces'], [0, 1])
        self.panel.group_changed()
        self.assertEqual(self.workspace.selection, {})
        self.panel.select_faces()
        self.assertEqual(self.workspace.selection, {})
        self.assertEqual(self.workspace.mode, 'plane')

    def test_explicit_new_selection_rebuilds_only_that_region_and_preserves_source_until_apply(self):
        original = deepcopy(self.panel.current_group())
        self.workspace.edit_selection(0, {3, 4}, 'replace')
        self.panel.selection_changed()
        self.panel.refresh()  # Parameter/visibility refresh must retain new user input.
        self.assertEqual(self.panel.faces(), [3, 4])
        self.panel.kind.setCurrentText('Отсутствует')
        self.panel.rebuild()
        self.assertEqual(len(self.jobs), 1)
        worker, _ = self.jobs[0]
        result, = worker.function()
        self.assertEqual(result['surface_faces'], [3, 4])
        self.assertEqual(result['replace_id'], original['id'])
        self.assertEqual(self.panel.current_group()['surface_faces'], original['surface_faces'])
        np.testing.assert_array_equal(self.panel.current_group()['vertices'], original['vertices'])

    def test_new_proxy_invalidates_fresh_but_now_stale_selection_again(self):
        self.workspace.edit_selection(0, {3}, 'replace')
        self.window.slicer_parts[0]['mesh'] = self.window.slicer_parts[0]['mesh'].copy()
        self.panel.refresh()
        self.assertEqual(self.panel.faces(), [])
        self.panel.rebuild()
        self.assertEqual(self.jobs, [])

    def test_ordinary_and_complete_cad_groups_retain_existing_selection_behavior(self):
        group = self.panel.current_group()
        group['cad_binding']['requires_reselect'] = False
        self.panel.group_changed()
        self.assertEqual(self.workspace.selection[0], {0, 1})
        self.workspace.selection.clear()
        self.assertEqual(self.panel.faces(), [0, 1])
        group.pop('cad_binding')
        self.panel.select_faces()
        self.assertEqual(self.workspace.selection[0], {0, 1})

    def test_choose_faces_uses_native_cad_face_and_stl_plane_modes(self):
        from cad_state import attach_native
        mesh = self.window.slicer_parts[0]['mesh']
        attach_native(mesh, dict(version=1, brep=np.frombuffer(b'test', dtype=np.uint8), matrix=np.eye(4),
            face_ids=np.zeros(len(mesh.faces), dtype=np.int64), body_ids=np.full(len(mesh.faces), -1, dtype=np.int64),
            face_info=[{'body_id': -1}], bodies=[]))
        self.panel.select_faces()
        self.assertEqual(self.workspace.mode, 'cad_face')
        mesh.vertices[0] += 1
        self.panel.select_faces()
        self.assertEqual(self.workspace.mode, 'plane')


if __name__ == '__main__':
    unittest.main()
