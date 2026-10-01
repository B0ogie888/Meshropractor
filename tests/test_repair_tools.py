"""Qt repair transactions with real mesh actors but no OpenGL window."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from PySide6.QtWidgets import QCheckBox, QPushButton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_desktop import APP, TestPlotter
from Meshropractor import MainWindow
from part_supports import make_group
from project_store import ProjectState
from repair_dialog import MANUAL
from repair_ribbon import REPAIR_COMMANDS


class RepairPlotter(TestPlotter):
    def __init__(self):
        super().__init__()
        self.camera_position = [(9., 8., 7.), (1., 2., 3.), (0., 0., 1.)]
        self.marker_counts = []

    def add_mesh(self, mesh, name=None, **kwargs):
        actor = super().add_mesh(mesh, name=name, **kwargs)
        if 'opacity' in kwargs: actor.prop.opacity = kwargs['opacity']
        if 'pickable' in kwargs: actor.SetPickable(kwargs['pickable'])
        return actor

    def add_points(self, points, **kwargs):
        import pyvista as pv
        self.marker_counts.append(len(points))
        return self.add_mesh(pv.PolyData(np.asarray(points)), **kwargs)

    def remove_actor(self, actor, **kwargs):
        super().remove_actor(actor)


class RepairToolsTests(unittest.TestCase):
    def setUp(self):
        self.window = MainWindow()
        self.window.ui._ensure_def_plotter = lambda: setattr(self.window.ui, 'plotter', self.window.ui.plotter or RepairPlotter())
        self.window.ui._ensure_slicer_plotter = lambda: setattr(self.window.ui, 'slicer_plotter', self.window.ui.slicer_plotter or RepairPlotter())
        self.window.add_to_recent = lambda path: None
        self.messages = []
        self.window.log = self.messages.append
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        if self.window._job is not None:
            self.window.cancel_current_job()
            self.wait_for_job()
        session = getattr(self.window, '_repair_session', None)
        if session is not None:
            session.dialog.reject()
        self.window.dirty = False
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()

    def wait_for_job(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            APP.processEvents()
            session = getattr(self.window, '_repair_session', None)
            if self.window._job is None and (session is None or not session.dialog.running):
                for _ in range(3): APP.processEvents()
                return
            time.sleep(.005)
        self.fail('Repair worker did not finish')

    def load(self, meshes=None, *, selected=(0,), supports=False):
        meshes = meshes or [trimesh.creation.box(), trimesh.creation.box()]
        parts = []
        for row, mesh in enumerate(meshes):
            groups = [make_group(trimesh.creation.box(extents=(.1, .1, .2)), [0], contacts=row + 1)] if supports else []
            parts.append(dict(mesh=mesh, filename=f'part_{row}.stl', platform='Build A' if row in selected else 'Build B', supports=groups,
                              style=dict(is_selected=row in selected, is_visible=True, color='#476b91',
                                         last_visible_mode='transparent' if row == 0 else 'shaded', transparency=35 if row == 0 else 0)))
        self.window.restore_project(ProjectState(parts=parts, platforms=[
            dict(name='Build A', dim=[50, 50, 50], is_default=True),
            dict(name='Build B', dim=[80, 80, 80], is_default=False)], page='slicer'))
        self.window.dirty = False
        self.window.reset_history()
        return self.window.ui.slicer_plotter

    def select_rows(self, rows):
        for row in range(len(self.window.slicer_parts)):
            box = self.window.ui.tbl_parts.cellWidget(row, 1).findChild(QCheckBox)
            box.setChecked(row in rows)
        self.window.flush_history()

    def session(self, operation):
        self.window.repair_tools.open(operation)
        result = getattr(self.window, '_repair_session', None)
        self.assertIsNotNone(result, self.messages)
        self.assertEqual(result.operation, operation)
        return result

    def prepared(self, operation):
        session = self.session(operation)
        session.prepare()
        self.wait_for_job()
        self.assertIsNotNone(session.result, session.dialog.report.toPlainText())
        return session

    def assert_group_equal(self, actual, expected, *, faces=None):
        for key in ('id', 'kind', 'contacts', 'visible', 'params'):
            self.assertEqual(actual[key], expected[key])
        np.testing.assert_array_equal(actual['vertices'], expected['vertices'])
        np.testing.assert_array_equal(actual['faces'], expected['faces'])
        self.assertEqual(actual['surface_faces'], expected['surface_faces'] if faces is None else faces)

    def test_all_repair_buttons_have_icons_and_dispatch_the_correct_command(self):
        self.load(selected=(0, 1))
        wizard_calls, session_calls = [], []
        with patch.object(self.window, 'open_repair_wizard', lambda: wizard_calls.append(True)), \
             patch('repair_tools.RepairSession', lambda window, op, rows: session_calls.append((op, list(rows)))):
            for operation, button in self.window.ui.repair_buttons.items():
                self.assertTrue(button.isEnabled(), operation)
                self.assertFalse(button.icon().isNull(), operation)
                self.select_rows((0,) if operation in MANUAL else (0, 1))
                button.click()
        self.assertEqual(wizard_calls, [True])
        self.assertEqual([op for op, _ in session_calls], [op for op in REPAIR_COMMANDS if op != 'wizard'])
        for operation, rows in session_calls:
            self.assertEqual(rows, [0] if operation in MANUAL else [0, 1])

    def test_prepare_preview_cancel_preserves_source_style_camera_and_controls(self):
        plotter = self.load()
        source = self.window.slicer_parts[0]['mesh']
        faces, vertices = source.faces.copy(), source.vertices.copy()
        actor = plotter.actors['slicer_part_0']
        original = (actor.GetVisibility(), actor.prop.opacity, actor.prop.color, deepcopy(plotter.camera_position))
        original_history = len(self.window.history.entries)
        session = self.session('normals')
        session.dialog.flags['flip'].setChecked(True)
        session.dialog.prepare.click()
        self.assertIsNotNone(self.window._job)
        self.wait_for_job()
        self.assertIsNotNone(session.result, session.dialog.report.toPlainText())
        self.assertTrue(session.preview_actors)
        self.assertFalse(actor.GetVisibility())
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        np.testing.assert_array_equal(source.faces, faces)
        np.testing.assert_array_equal(source.vertices, vertices)
        session.dialog.flags['flip'].setChecked(False)
        self.assertIsNone(session.result)
        self.assertFalse(session.dialog.apply.isEnabled())
        self.assertTrue(actor.GetVisibility())
        session.dialog.reject()
        self.assertEqual((actor.GetVisibility(), actor.prop.opacity, actor.prop.color, plotter.camera_position), original)
        self.assertEqual(len(self.window.history.entries), original_history)
        self.assertTrue(self.window.ui.magics_ribbon.isEnabled())
        self.assertTrue(all(button.isEnabled() for button in self.window.ui.repair_buttons.values()))
        self.assertTrue(self.window.ui.ribbon_btns['Импорт детали'].isEnabled())
        self.assertFalse(self.window.ui.ribbon_btns['Сохранить все в папку'].isEnabled())

    def test_every_nonwizard_command_opens_its_actual_parameter_session(self):
        self.load(selected=(0, 1))
        for operation in REPAIR_COMMANDS:
            if operation == 'wizard': continue
            with self.subTest(operation=operation):
                self.select_rows((0,) if operation in MANUAL else (0, 1))
                session = self.session(operation)
                self.assertEqual(session.dialog.windowTitle(), REPAIR_COMMANDS[operation])
                self.assertEqual(len(session.records), 1 if operation in MANUAL else 2)
                self.assertTrue(session.dialog.prepare.isEnabled())
                self.assertFalse(session.dialog.apply.isEnabled())
                session.dialog.reject()
                self.assertIsNone(self.window._repair_session)
                APP.processEvents()

    def test_apply_selected_parts_is_one_undo_redo_step(self):
        self.load([trimesh.creation.box() for _ in range(3)], selected=(0, 2), supports=True)
        original = self.window.capture_project()
        untouched = self.window.slicer_parts[1]['mesh']
        history = self.window.history.index
        session = self.session('normals')
        session.dialog.flags['flip'].setChecked(True)
        session.prepare(); self.wait_for_job()
        self.assertTrue(session.dialog.apply.isEnabled())
        session.dialog.apply.click()
        self.assertIsNone(self.window._repair_session)
        self.assertEqual(self.window.history.index, history + 1)
        self.assertIs(self.window.slicer_parts[1]['mesh'], untouched)
        for row in (0, 2):
            np.testing.assert_array_equal(self.window.slicer_parts[row]['mesh'].faces, original.parts[row]['mesh'].faces[:, ::-1])
            self.assert_group_equal(self.window.slicer_parts[row]['supports'][0], original.parts[row]['supports'][0])
        self.window.undo_action()
        for row in range(3):
            np.testing.assert_array_equal(self.window.slicer_parts[row]['mesh'].faces, original.parts[row]['mesh'].faces)
            self.assert_group_equal(self.window.slicer_parts[row]['supports'][0], original.parts[row]['supports'][0])
        self.window.redo_action()
        for row in (0, 2):
            np.testing.assert_array_equal(self.window.slicer_parts[row]['mesh'].faces, original.parts[row]['mesh'].faces[:, ::-1])

    def test_split_preserves_metadata_support_geometry_and_unselected_part(self):
        right = trimesh.creation.box(); right.apply_translation([3, 0, 0])
        disconnected = trimesh.util.concatenate([trimesh.creation.box(), right])
        self.load([disconnected, trimesh.creation.box()], supports=True)
        before = self.window.capture_project()
        session = self.prepared('split'); session.apply()
        self.assertIsNone(self.window._repair_session)
        after = self.window.capture_project()
        self.assertEqual(len(after.parts), 3)
        for part in after.parts[:2]:
            self.assertEqual(part['platform'], before.parts[0]['platform'])
            self.assertEqual(part['style'], before.parts[0]['style'])
            self.assertEqual(len(part['mesh'].faces), 12)
        self.assert_group_equal(after.parts[0]['supports'][0], before.parts[0]['supports'][0])
        self.assertEqual(after.parts[1]['supports'], [])
        self.assertEqual(after.parts[2]['filename'], before.parts[1]['filename'])
        self.assertEqual(after.parts[2]['platform'], 'Build B')
        self.assertEqual(after.parts[2]['style'], before.parts[1]['style'])
        self.assert_group_equal(after.parts[2]['supports'][0], before.parts[1]['supports'][0])
        np.testing.assert_array_equal(after.parts[2]['mesh'].vertices, before.parts[1]['mesh'].vertices)

    def test_split_attaches_support_to_its_fragment_and_undo_restores_original_ids(self):
        right = trimesh.creation.box(); right.apply_translation([3, 0, 0])
        disconnected = trimesh.util.concatenate([trimesh.creation.box(), right])
        self.load([disconnected], supports=True)
        support_mesh = trimesh.creation.box(extents=[.1, .1, .2]); support_mesh.apply_translation([3, 0, 0])
        second = make_group(support_mesh, [12, 15], contacts=2)
        self.window.slicer_parts[0]['supports'].append(second)
        self.window.reset_history()
        original = self.window.capture_project()
        session = self.prepared('split'); session.apply()
        self.assertIsNone(self.window._repair_session)
        after = self.window.capture_project()
        self.assertEqual(len(after.parts), 2)
        right_part = next(part for part in after.parts if part['mesh'].centroid[0] > 2)
        left_part = next(part for part in after.parts if part['mesh'].centroid[0] < 1)
        self.assertEqual(len(right_part['supports']), 1)
        self.assert_group_equal(right_part['supports'][0], second, faces=[0, 3])
        self.assertEqual(left_part['supports'][0]['id'], original.parts[0]['supports'][0]['id'])
        np.testing.assert_array_equal(right_part['mesh'].triangles[right_part['supports'][0]['surface_faces']],
                                      original.parts[0]['mesh'].triangles[second['surface_faces']])
        self.window.undo_action()
        self.assertEqual(len(self.window.slicer_parts), 1)
        self.assert_group_equal(self.window.slicer_parts[0]['supports'][1], second)
        self.window.redo_action()
        restored_right = next(part for part in self.window.slicer_parts if part['mesh'].centroid[0] > 2)
        self.assert_group_equal(restored_right['supports'][0], second, faces=[0, 3])

    def test_delete_faces_remaps_surviving_support_references_and_undo(self):
        self.load(supports=True)
        self.window.slicer_parts[0]['supports'][0]['surface_faces'] = [0, 2, 8]
        self.window.reset_history()
        original = self.window.capture_project()
        session = self.session('delete_faces')
        session.dialog.set_selected_ids([0, 3])
        session.prepare(); self.wait_for_job()
        self.assertIsNotNone(session.result, session.dialog.report.toPlainText())
        session.apply()
        self.assertIsNone(self.window._repair_session)
        part = self.window.slicer_parts[0]
        self.assertEqual(len(part['mesh'].faces), 10)
        self.assert_group_equal(part['supports'][0], original.parts[0]['supports'][0], faces=[1, 6])
        np.testing.assert_array_equal(part['mesh'].triangles[[1, 6]], original.parts[0]['mesh'].triangles[[2, 8]])
        self.assert_group_equal(self.window.slicer_parts[1]['supports'][0], original.parts[1]['supports'][0])
        self.window.undo_action()
        self.assert_group_equal(self.window.slicer_parts[0]['supports'][0], original.parts[0]['supports'][0])
        self.window.redo_action()
        self.assert_group_equal(self.window.slicer_parts[0]['supports'][0], original.parts[0]['supports'][0], faces=[1, 6])

    def test_split_rejects_unassigned_or_cross_fragment_support_before_preview(self):
        for references, message in (([], 'без привязки'), ([0, 12], 'несколько фрагментов')):
            with self.subTest(references=references):
                right = trimesh.creation.box(); right.apply_translation([3, 0, 0])
                self.load([trimesh.util.concatenate([trimesh.creation.box(), right])], supports=True)
                self.window.slicer_parts[0]['supports'][0]['surface_faces'] = references
                self.window.reset_history()
                source = self.window.slicer_parts[0]['mesh']
                before = self.window.capture_project()
                history = self.window.history.index
                session = self.session('split')
                with self.assertLogs('root', level='ERROR'):
                    session.prepare(); self.wait_for_job()
                self.assertIsNone(session.result)
                self.assertEqual(session.preview_actors, [])
                self.assertFalse(session.dialog.apply.isEnabled())
                self.assertIn(message, session.dialog.report.toPlainText())
                self.assertIs(self.window.slicer_parts[0]['mesh'], source)
                np.testing.assert_array_equal(source.faces, before.parts[0]['mesh'].faces)
                self.assert_group_equal(self.window.slicer_parts[0]['supports'][0], before.parts[0]['supports'][0])
                self.assertEqual(self.window.history.index, history)
                session.dialog.reject()

    def test_unify_keeps_supports_and_unselected_part(self):
        other = trimesh.creation.box(); other.apply_translation([.5, 0, 0])
        self.load([trimesh.creation.box(), other, trimesh.creation.box()], selected=(0, 1), supports=True)
        before = self.window.capture_project()
        session = self.prepared('unify'); session.apply()
        self.assertIsNone(self.window._repair_session)
        after = self.window.capture_project()
        self.assertEqual(len(after.parts), 2)
        self.assertAlmostEqual(after.parts[0]['mesh'].volume, 1.5, places=6)
        self.assertEqual(after.parts[0]['platform'], 'Build A')
        self.assertEqual(after.parts[0]['style'], before.parts[0]['style'])
        for row in (0, 1): self.assert_group_equal(after.parts[0]['supports'][row], before.parts[row]['supports'][0], faces=[])
        self.assertEqual(after.parts[1]['filename'], before.parts[2]['filename'])
        self.assertEqual(after.parts[1]['platform'], 'Build B')
        self.assert_group_equal(after.parts[1]['supports'][0], before.parts[2]['supports'][0])

    def test_remove_small_deletes_selected_only_and_keeps_retained_support_bindings(self):
        large = trimesh.creation.icosphere(subdivisions=1)
        self.load([trimesh.creation.box(), large, trimesh.creation.box()], selected=(0, 1), supports=True)
        before = self.window.capture_project()
        session = self.session('remove_small')
        session.dialog.fields['min_faces'].setValue(13)
        session.prepare(); self.wait_for_job(); session.apply()
        self.assertIsNone(self.window._repair_session)
        after = self.window.capture_project()
        self.assertEqual([part['filename'] for part in after.parts], ['part_1.stl', 'part_2.stl'])
        for row, previous in enumerate((1, 2)):
            self.assertEqual(after.parts[row]['platform'], before.parts[previous]['platform'])
            self.assertEqual(after.parts[row]['style'], before.parts[previous]['style'])
            self.assert_group_equal(after.parts[row]['supports'][0], before.parts[previous]['supports'][0])

    def test_stale_source_rejects_apply_without_overwriting_new_mesh(self):
        self.load()
        session = self.session('normals')
        session.dialog.flags['flip'].setChecked(True)
        session.prepare(); self.wait_for_job()
        replacement = trimesh.creation.box(extents=[2, 2, 2])
        self.window.slicer_parts[0]['mesh'] = replacement
        history = self.window.history.index
        session.apply()
        self.assertIn('Модель изменилась', session.dialog.status.text())
        self.assertIs(self.window.slicer_parts[0]['mesh'], replacement)
        self.assertEqual(self.window.history.index, history)

    def test_diagnostics_retain_face_ids_for_manual_selection_after_close(self):
        self.load()
        source = self.window.slicer_parts[0]['mesh']
        history = self.window.history.index
        def diagnostics(*args, **kwargs):
            return dict(mode='slivers', items=[dict(row=0, meshes=[source], report=dict(changed=False, selected_faces=[1, 4]))])
        with patch('repair_tools.calculate_repairs', diagnostics):
            session = self.prepared('slivers')
        self.assertEqual(self.window.workspace_tools.selection, {0: {1, 4}})
        self.assertFalse(session.dialog.apply.isEnabled())
        session.dialog.reject()
        self.assertEqual(self.window.workspace_tools.selection, {0: {1, 4}})
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        self.assertEqual(self.window.history.index, history)
        manual = self.session('delete_faces')
        self.assertEqual(manual.dialog.selected_ids(), [1, 4])

    def test_cancelled_async_calculation_never_applies(self):
        self.load()
        source = self.window.slicer_parts[0]['mesh']
        history = self.window.history.index
        def cancellable(*args, progress, cancelled):
            while not cancelled(): time.sleep(.005)
            raise InterruptedError('test cancellation')
        with patch('repair_tools.calculate_repairs', cancellable):
            session = self.session('normals')
            session.prepare()
            self.assertIsNotNone(self.window._job)
            session.dialog.cancel.click()
            self.wait_for_job()
        self.assertIsNone(session.result)
        self.assertFalse(session.dialog.apply.isEnabled())
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        self.assertEqual(self.window.history.index, history)
        session.dialog.reject()
        self.assertTrue(self.window.ui.repair_buttons['normals'].isEnabled())

    def test_manual_triangle_and_vertex_input_validation_then_valid_prepare(self):
        mesh = trimesh.creation.box()
        missing = mesh.faces[-1].copy(); mesh.update_faces(np.arange(11))
        self.load([mesh])
        source = self.window.slicer_parts[0]['mesh']
        session = self.session('add_triangle')
        session.dialog.ids.setText('0, 1')
        with self.assertLogs('root', level='ERROR'):
            session.prepare(); self.wait_for_job()
        self.assertIsNone(session.result)
        self.assertIn('ровно 3', session.dialog.report.toPlainText())
        session.dialog.ids.setText(','.join(map(str, missing)))
        session.prepare(); self.wait_for_job()
        self.assertIsNotNone(session.result, session.dialog.report.toPlainText())
        self.assertEqual(len(source.faces), 11)
        session.apply()
        self.assertTrue(self.window.slicer_parts[0]['mesh'].is_watertight)
        moving = self.session('move_vertices')
        moving.dialog.ids.setText('9999')
        moving.dialog.vectors['delta'][0].setValue(.05)
        with self.assertLogs('root', level='ERROR'):
            moving.prepare(); self.wait_for_job()
        self.assertIsNone(moving.result)
        self.assertIn('существующие индексы', moving.dialog.report.toPlainText())
        moving.dialog.ids.setText('0')
        moving.prepare(); self.wait_for_job()
        self.assertIsNotNone(moving.result, moving.dialog.report.toPlainText())
        np.testing.assert_allclose(moving.result['items'][0]['meshes'][0].vertices[0],
                                   self.window.slicer_parts[0]['mesh'].vertices[0] + [.05, 0, 0])

    def test_large_surface_selection_preserves_all_ids_but_limits_markers(self):
        mesh = trimesh.creation.icosphere(subdivisions=5)
        plotter = self.load([mesh])
        source = self.window.slicer_parts[0]['mesh']
        original_vertices = source.vertices.copy()
        self.window.workspace_tools.selection = {0: set(range(len(source.faces)))}
        session = self.session('move_vertices')
        session.dialog.from_selection.click()
        expected = list(range(len(source.vertices)))
        self.assertGreater(len(', '.join(map(str, expected))), 32767)
        self.assertEqual(session.dialog.selected_ids(), expected)
        self.assertTrue(session.dialog.ids.isReadOnly())
        self.assertLess(len(session.dialog.ids.text()), 200)
        self.assertEqual(plotter.marker_counts[-1], 5000)
        self.assertTrue(all(count <= 5000 for count in plotter.marker_counts))
        session.dialog.vectors['delta'][0].setValue(.01)
        session.dialog.prepare.click(); self.wait_for_job()
        self.assertIsNotNone(session.result, session.dialog.report.toPlainText())
        np.testing.assert_allclose(session.result['items'][0]['meshes'][0].vertices,
                                   original_vertices + [.01, 0, 0])
        np.testing.assert_array_equal(source.vertices, original_vertices)
        # Editing or clearing the visible field must never reuse a previous
        # large selection cached behind its compact display text.
        session.dialog.ids.setText('3, 7')
        self.assertEqual(session.dialog.selected_ids(), [3, 7])
        self.assertFalse(session.dialog.ids.isReadOnly())
        self.assertIsNone(session.result)
        self.assertEqual(plotter.marker_counts[-1], 2)
        session.dialog.set_selected_ids(expected + [expected[-1]])
        self.assertEqual(session.dialog.selected_ids(), expected)
        clear = next(button for button in session.dialog.findChildren(QPushButton) if button.text() == 'Очистить')
        clear.click()
        self.assertEqual(session.dialog.selected_ids(), [])
        self.assertFalse(session.dialog.ids.isReadOnly())
        self.assertIsNone(session.marker)
        session.dialog.reject()
        # The delete tool initializes its IDs from the selected faces without
        # truncating at QLineEdit's former 32767-character default.
        selected_faces = list(range(12000))
        self.window.workspace_tools.selection = {0: set(selected_faces)}
        deletion = self.session('delete_faces')
        self.assertEqual(deletion.dialog.selected_ids(), selected_faces)
        self.assertTrue(deletion.dialog.ids.isReadOnly())


if __name__ == '__main__':
    unittest.main()
