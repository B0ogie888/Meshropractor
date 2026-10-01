"""Placement dialogs and reversible transactions using real VTK actors without GL."""
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
from PySide6.QtWidgets import QCheckBox, QLabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from test_desktop import APP
from test_repair_tools import RepairPlotter
from Meshropractor import MainWindow
from part_supports import make_group, combined_mesh
from placement_ribbon import PLACEMENT_COMMANDS
from placement_tools import BASIC
from project_store import ProjectState


def actor_matrix(actor):
    matrix = actor.GetMatrix()
    return np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])


class PlacementToolsTests(unittest.TestCase):
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
        for name in ('_placement_session', '_transform_session'):
            session = getattr(self.window, name, None)
            if session is not None: session.dialog.reject()
        self.window.dirty = False
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()

    def wait_for_job(self):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            APP.processEvents()
            session = getattr(self.window, '_placement_session', None)
            if self.window._job is None and (session is None or not session.dialog.running):
                for _ in range(3): APP.processEvents()
                return
            time.sleep(.005)
        self.fail('Placement worker did not finish')

    def load(self, meshes=None, *, selected=(0,), supports=False):
        if meshes is None: meshes = [trimesh.creation.box() for _ in range(3)]
        parts = []
        for row, mesh in enumerate(meshes):
            groups = []
            if supports:
                support = trimesh.creation.box(extents=(.1, .1, .3))
                support.apply_translation(mesh.bounds.mean(axis=0) + [0, 0, -.5])
                groups = [make_group(support, [0, 1], contacts=row + 1)]
            parts.append(dict(mesh=mesh, filename=f'part_{row}.stl', platform='Build A' if row in selected else 'Build B', supports=groups,
                              style=dict(is_selected=row in selected, is_visible=True, color='#476b91',
                                         last_visible_mode='transparent' if row == 0 else 'shaded', transparency=35 if row == 0 else 0)))
        self.window.restore_project(ProjectState(parts=parts, platforms=[
            dict(name='Build A', dim=[50, 50, 50], is_default=True),
            dict(name='Build B', dim=[80, 80, 80], is_default=True)], page='slicer'))
        self.window.dirty = False
        self.window.reset_history()
        return self.window.ui.slicer_plotter

    def select_rows(self, rows):
        for row in range(len(self.window.slicer_parts)):
            self.window.ui.tbl_parts.cellWidget(row, 1).findChild(QCheckBox).setChecked(row in rows)
        self.window.flush_history()

    def session(self, operation):
        self.window.placement_tools.open(operation)
        result = getattr(self.window, '_placement_session', None)
        self.assertIsNotNone(result, self.messages)
        self.assertEqual(result.operation, operation)
        return result

    def prepare(self, session):
        session.dialog.prepare.click()
        self.assertIsNotNone(self.window._job)
        self.wait_for_job()
        self.assertIsNotNone(session.result, session.dialog.report.toPlainText())
        self.assertTrue(session.dialog.apply.isEnabled())

    def assert_support_transform(self, actual, original, matrix):
        for key in ('id', 'kind', 'contacts', 'visible', 'params', 'surface_faces'):
            self.assertEqual(actual[key], original[key])
        np.testing.assert_array_equal(actual['faces'], original['faces'])
        np.testing.assert_allclose(actual['vertices'], trimesh.transform_points(original['vertices'], matrix), atol=1e-8)

    def test_ribbon_two_groups_all_thirteen_icons_and_each_button_dispatches_once(self):
        self.load(selected=(0, 2))
        placement_calls, basic_calls = [], []
        with patch('placement_tools.PlacementSession', lambda window, op, rows: placement_calls.append((op, list(rows)))), \
             patch('transform_session.TransformSession', lambda window, op, rows: basic_calls.append((op, list(rows)))):
            for operation, button in self.window.ui.placement_buttons.items():
                with self.subTest(operation=operation):
                    self.assertTrue(button.isEnabled())
                    self.assertFalse(button.icon().isNull())
                    self.select_rows((0,) if operation in {'top_bottom', 'compare_orientations'} else (0, 2))
                    button.click()
            for operation, title in BASIC.items():
                self.assertIsNot(self.window.ui.ribbon_btns[title], self.window.ui.placement_buttons[operation])
                self.window.ui.ribbon_btns[title].click()
        self.assertEqual(list(self.window.ui.placement_buttons), list(PLACEMENT_COMMANDS))
        self.assertEqual([op for op, _ in placement_calls], [op for op in PLACEMENT_COMMANDS if op not in BASIC])
        self.assertEqual([op for op, _ in basic_calls], list(BASIC.values()) * 2)
        for op, rows in placement_calls:
            self.assertEqual(rows, [0] if op in {'top_bottom', 'compare_orientations'} else [0, 2])
        ribbon = self.window.ui.magics_ribbon
        panel = next(ribbon.widget(i) for i in range(ribbon.count()) if ribbon.tabText(i) == 'РАСПОЛОЖЕНИЕ')
        self.assertEqual([label.text() for label in panel.findChildren(QLabel)], ['Базовый', 'Автоматический'])

    def test_every_extended_command_opens_a_real_dialog_and_basic_uses_transform_session(self):
        self.load(selected=(0, 2))
        for operation in PLACEMENT_COMMANDS:
            with self.subTest(operation=operation):
                self.select_rows((0,) if operation in {'top_bottom', 'compare_orientations'} else (0, 2))
                self.window.ui.placement_buttons[operation].click()
                session = getattr(self.window, '_transform_session' if operation in BASIC else '_placement_session', None)
                self.assertIsNotNone(session, self.messages)
                try:
                    self.assertEqual(session.operation, BASIC.get(operation, operation))
                    if operation not in BASIC:
                        self.assertEqual(session.dialog.windowTitle(), PLACEMENT_COMMANDS[operation])
                finally:
                    session.dialog.reject()
                APP.processEvents()
                self.assertTrue(self.window.ui.magics_ribbon.isEnabled())

    def test_free_numeric_preview_cancel_restores_matrix_style_camera_source_and_controls(self):
        plotter = self.load(supports=True)
        source = self.window.slicer_parts[0]['mesh']
        original_vertices = source.vertices.copy()
        original_support = deepcopy(self.window.slicer_parts[0]['supports'][0])
        actor = plotter.actors['slicer_part_0']
        style = (actor.GetVisibility(), actor.prop.opacity, actor.prop.color)
        camera, history = deepcopy(plotter.camera_position), self.window.history.index
        session = self.session('free_move')
        for spin, value in zip(session.dialog.delta, [3, -4, 2]): spin.setValue(value)
        expected = np.eye(4); expected[:3, 3] = [3, -4, 2]
        self.assertTrue(session.dialog.apply.isEnabled())
        np.testing.assert_array_equal(actor_matrix(actor), expected)
        support_actor = plotter.actors['part_support_' + original_support['id']]
        np.testing.assert_array_equal(actor_matrix(support_actor), expected)
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        np.testing.assert_array_equal(source.vertices, original_vertices)
        self.assert_support_transform(self.window.slicer_parts[0]['supports'][0], original_support, np.eye(4))
        session.dialog.preview.setChecked(False)
        np.testing.assert_array_equal(actor_matrix(actor), np.eye(4))
        session.dialog.preview.setChecked(True)
        np.testing.assert_array_equal(actor_matrix(actor), expected)
        session.dialog.reject()
        np.testing.assert_array_equal(actor_matrix(actor), np.eye(4))
        np.testing.assert_array_equal(actor_matrix(support_actor), np.eye(4))
        self.assertEqual((actor.GetVisibility(), actor.prop.opacity, actor.prop.color), style)
        self.assertEqual(plotter.camera_position, camera)
        self.assertEqual(self.window.history.index, history)
        self.assertTrue(self.window.ui.magics_ribbon.isEnabled())
        self.assertTrue(self.window.ui.cb_plat.isEnabled())
        self.assertFalse(self.window.ui.ribbon_btns['Сохранить все в папку'].isEnabled())

    def test_free_apply_selected_batch_transforms_supports_one_undo_redo_step(self):
        plotter = self.load(selected=(0, 2), supports=True)
        before = self.window.capture_project()
        sentinel = self.window.slicer_parts[1]['mesh']
        camera, history = deepcopy(plotter.camera_position), self.window.history.index
        session = self.session('free_move')
        for spin, value in zip(session.dialog.delta, [7, -3, 2]): spin.setValue(value)
        session.dialog.apply.click()
        self.assertIsNone(self.window._placement_session)
        self.assertEqual(self.window.history.index, history + 1)
        self.assertIs(self.window.slicer_parts[1]['mesh'], sentinel)
        matrix = np.eye(4); matrix[:3, 3] = [7, -3, 2]
        for row in (0, 2):
            part = self.window.slicer_parts[row]
            np.testing.assert_allclose(part['mesh'].vertices, before.parts[row]['mesh'].vertices + [7, -3, 2])
            np.testing.assert_array_equal(part['mesh'].faces, before.parts[row]['mesh'].faces)
            self.assert_support_transform(part['supports'][0], before.parts[row]['supports'][0], matrix)
        self.assertEqual(plotter.camera_position, camera)
        self.window.undo_action()
        for row in range(3):
            np.testing.assert_array_equal(self.window.slicer_parts[row]['mesh'].vertices, before.parts[row]['mesh'].vertices)
            self.assert_support_transform(self.window.slicer_parts[row]['supports'][0], before.parts[row]['supports'][0], np.eye(4))
        self.window.redo_action()
        for row in (0, 2):
            np.testing.assert_allclose(self.window.slicer_parts[row]['mesh'].vertices, before.parts[row]['mesh'].vertices + [7, -3, 2])
            self.assert_support_transform(self.window.slicer_parts[row]['supports'][0], before.parts[row]['supports'][0], matrix)

    def test_async_arrange_assigns_target_platform_without_mutating_preview_or_sentinel(self):
        plotter = self.load(selected=(0, 2), supports=True)
        before = self.window.capture_project()
        sources = [part['mesh'] for part in self.window.slicer_parts]
        history = self.window.history.index
        session = self.session('auto_arrange')
        session.dialog.platform.setCurrentIndex(1)
        session.dialog.rotation.setChecked(False)
        self.prepare(session)
        matrices = deepcopy(session.result['matrices'])
        self.assertEqual(session.result['platform']['name'], 'Build B')
        self.assertEqual(session.platform_preview_name, 'Build B')
        for row in range(3):
            self.assertIs(self.window.slicer_parts[row]['mesh'], sources[row])
            np.testing.assert_array_equal(sources[row].vertices, before.parts[row]['mesh'].vertices)
        np.testing.assert_array_equal(actor_matrix(plotter.actors['slicer_part_1']), np.eye(4))
        session.dialog.apply.click()
        self.assertIsNone(self.window._placement_session)
        self.assertEqual(self.window.history.index, history + 1)
        self.assertEqual(self.window.ui.scene_tabs.currentIndex(), 2)
        self.assertIs(self.window.slicer_parts[1]['mesh'], sources[1])
        for row in (0, 2):
            part = self.window.slicer_parts[row]
            self.assertEqual(part['platform'], 'Build B')
            np.testing.assert_allclose(part['mesh'].vertices, trimesh.transform_points(before.parts[row]['mesh'].vertices, matrices[row]))
            self.assert_support_transform(part['supports'][0], before.parts[row]['supports'][0], matrices[row])
            bounds = combined_mesh(part).bounds
            self.assertGreaterEqual(bounds[0, 2], -1e-8)
            self.assertTrue(np.all(bounds[0, :2] >= -38 - 1e-8))
            self.assertTrue(np.all(bounds[1, :2] <= 38 + 1e-8))
        self.window.undo_action()
        self.assertEqual(self.window.slicer_parts[0]['platform'], 'Build A')
        self.assertEqual(self.window.slicer_parts[2]['platform'], 'Build A')
        self.window.redo_action()
        self.assertEqual(self.window.slicer_parts[0]['platform'], 'Build B')

    def test_fit_preserves_group_and_parameter_changes_invalidate_preview(self):
        right = trimesh.creation.box(); right.apply_translation([4, 2, 7])
        plotter = self.load([trimesh.creation.box(), right], selected=(0, 1), supports=True)
        originals = [part['mesh'] for part in self.window.slicer_parts]
        session = self.session('fit_platform')
        session.dialog.rotation.setChecked(False)
        self.prepare(session)
        matrix = session.result['matrices'][0]
        np.testing.assert_array_equal(matrix, session.result['matrices'][1])
        np.testing.assert_array_equal(actor_matrix(plotter.actors['slicer_part_0']), matrix)
        session.dialog.fields['margin_mm'].setValue(3)
        self.assertIsNone(session.result)
        self.assertFalse(session.dialog.apply.isEnabled())
        for row in (0, 1):
            self.assertIs(self.window.slicer_parts[row]['mesh'], originals[row])
            np.testing.assert_array_equal(actor_matrix(plotter.actors[f'slicer_part_{row}']), np.eye(4))
        self.prepare(session)
        matrix = session.result['matrices'][0].copy()
        session.apply()
        for row in (0, 1):
            np.testing.assert_allclose(self.window.slicer_parts[row]['mesh'].vertices,
                                       trimesh.transform_points(originals[row].vertices, matrix))

    def test_target_platform_preview_shows_its_visible_obstacles_and_restores_source_scene(self):
        plotter = self.load([trimesh.creation.box() for _ in range(4)], supports=True)
        self.window.slicer_parts[2]['platform'] = 'Build A'
        self.window.ui.tbl_parts.cellWidget(3, self.window.COL_VISIBLE).findChild(QCheckBox).setChecked(False)
        self.window.ui.scene_tabs.setCurrentIndex(1)
        self.window.refresh_scene_visibility()
        self.window.reset_history()
        originals = [part['mesh'] for part in self.window.slicer_parts]
        before = self.window.capture_project()
        history, entries = self.window.history.index, len(self.window.history.entries)

        def visibility():
            part_values, support_values = [], []
            for part in self.window.slicer_parts:
                part_values.append(bool(plotter.actors[part['actor_name']].GetVisibility()))
                support_values.append(bool(plotter.actors['part_support_' + part['supports'][0]['id']].GetVisibility()))
            return part_values, support_values

        def assert_unchanged_project():
            after = self.window.capture_project()
            self.assertEqual(self.window.ui.scene_tabs.currentIndex(), 1)
            self.assertEqual(self.window.history.index, history)
            self.assertEqual(len(self.window.history.entries), entries)
            for row, part in enumerate(after.parts):
                self.assertIs(self.window.slicer_parts[row]['mesh'], originals[row])
                self.assertEqual(part['platform'], before.parts[row]['platform'])
                self.assertEqual(part['style'], before.parts[row]['style'])
                np.testing.assert_array_equal(part['mesh'].vertices, before.parts[row]['mesh'].vertices)
                self.assert_support_transform(part['supports'][0], before.parts[row]['supports'][0], np.eye(4))

        source_visible = [True, False, True, False]
        preview_visible = [True, True, False, False]
        self.assertEqual(visibility(), (source_visible, source_visible))
        session = self.session('auto_arrange')
        session.dialog.platform.setCurrentIndex(1)
        session.dialog.rotation.setChecked(False)
        self.prepare(session)
        self.assertEqual(visibility(), (preview_visible, preview_visible))
        assert_unchanged_project()
        session.dialog.preview.setChecked(False)
        self.assertEqual(visibility(), (source_visible, source_visible))
        assert_unchanged_project()
        session.dialog.preview.setChecked(True)
        self.assertEqual(visibility(), (preview_visible, preview_visible))
        session.dialog.reject()
        self.assertEqual(visibility(), (source_visible, source_visible))
        assert_unchanged_project()

    def test_top_bottom_requires_surface_then_uses_selected_triangle_normal(self):
        self.load()
        source = self.window.slicer_parts[0]['mesh']
        session = self.session('top_bottom')
        session.prepare()
        self.assertIsNone(self.window._job)
        self.assertIn('выберите поверхность', session.dialog.status.text())
        face = int(np.flatnonzero(source.face_normals[:, 0] > .9)[0])
        session.surface = face
        self.prepare(session)
        matrix = session.result['matrices'][0]
        np.testing.assert_allclose(matrix[:3, :3] @ source.face_normals[face], [0, 0, -1], atol=1e-8)
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        session.apply()
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].face_normals[face], [0, 0, -1], atol=1e-8)
        self.assertAlmostEqual(self.window.slicer_parts[0]['mesh'].bounds[0, 2], 0)

    def test_compare_table_selection_previews_and_applies_chosen_variant(self):
        plotter = self.load([trimesh.creation.box(extents=(1, 2, 4))])
        source = self.window.slicer_parts[0]['mesh']
        session = self.session('compare_orientations')
        self.prepare(session)
        variants = session.result['variants']
        self.assertGreater(len(variants), 1)
        self.assertEqual(session.dialog.table.rowCount(), len(variants))
        chosen = len(variants) - 1
        session.dialog.table.selectRow(chosen)
        matrix = variants[chosen]['matrix'].copy()
        np.testing.assert_allclose(session.result['matrices'][0], matrix)
        np.testing.assert_allclose(actor_matrix(plotter.actors['slicer_part_0']), matrix)
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        session.apply()
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].vertices, trimesh.transform_points(source.vertices, matrix))

    def test_shape_sorter_selected_reference_is_not_global_row_and_unmatched_stays_unchanged(self):
        self.load(selected=(0, 2), supports=True)
        sources = [part['mesh'] for part in self.window.slicer_parts]
        original = self.window.capture_project()
        calls = []
        matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        def transfer(meshes, reference_index, tolerance_mm, **kwargs):
            calls.append((list(meshes), reference_index, tolerance_mm))
            return [matrix.copy(), np.eye(4)], {'matches': [
                {'index': 0, 'matched': True, 'reason': 'Совпадает'},
                {'index': 1, 'matched': False, 'reason': 'Образец'}]}
        with patch('placement_shapes.transfer_orientations', transfer):
            session = self.session('sort_by_shape')
            self.assertEqual(session.dialog.reference.count(), 2)
            session.dialog.reference.setCurrentIndex(1)
            session.dialog.fields['tolerance_mm'].setValue(.025)
            self.prepare(session)
        self.assertEqual(calls[0][1:], (1, .025))
        self.assertIs(calls[0][0][0], sources[0])
        self.assertIs(calls[0][0][1], sources[2])
        self.assertIn('part_2.stl', session.dialog.report.toPlainText())
        session.apply()
        np.testing.assert_allclose(self.window.slicer_parts[0]['mesh'].vertices, trimesh.transform_points(sources[0].vertices, matrix))
        self.assertIs(self.window.slicer_parts[1]['mesh'], sources[1])
        self.assertIs(self.window.slicer_parts[2]['mesh'], sources[2])
        self.assert_support_transform(self.window.slicer_parts[0]['supports'][0], original.parts[0]['supports'][0], matrix)

    def test_cancellation_restores_own_enabled_controls_without_apply(self):
        self.load()
        source, history = self.window.slicer_parts[0]['mesh'], self.window.history.index
        def cancellable(*args, progress, cancelled):
            while not cancelled(): time.sleep(.005)
            raise InterruptedError('test cancellation')
        with patch('placement_tools.calculate_placement', cancellable):
            session = self.session('auto_arrange')
            session.prepare()
            self.assertTrue(session.dialog.running)
            session.dialog.reject()
            self.assertIs(self.window._placement_session, session)
            self.wait_for_job()
        self.assertIsNone(session.result)
        self.assertFalse(session.dialog.apply.isEnabled())
        self.assertFalse(self.window.ui.magics_ribbon.isEnabled())
        self.assertIs(self.window.slicer_parts[0]['mesh'], source)
        self.assertEqual(self.window.history.index, history)
        session.dialog.reject()
        self.assertTrue(self.window.ui.magics_ribbon.isEnabled())
        self.assertTrue(self.window.ui.ribbon_btns['Импорт детали'].isEnabled())
        self.assertTrue(all(button.isEnabled() for button in self.window.ui.placement_buttons.values()))
        self.assertFalse(self.window.ui.ribbon_btns['Сохранить все в папку'].isEnabled())

    def test_stale_source_rejects_apply_and_close_preserves_new_mesh(self):
        self.load()
        history = self.window.history.index
        session = self.session('free_move')
        session.dialog.delta[0].setValue(3)
        replacement = trimesh.creation.icosphere(subdivisions=1)
        self.window.replace_slicer_mesh(0, replacement)
        session.apply()
        self.assertIs(self.window._placement_session, session)
        self.assertIn('Модель изменилась', session.dialog.status.text())
        self.assertIs(self.window.slicer_parts[0]['mesh'], replacement)
        self.assertEqual(self.window.history.index, history)
        session.dialog.reject()
        self.assertIs(self.window.slicer_parts[0]['mesh'], replacement)


if __name__ == '__main__':
    unittest.main()
