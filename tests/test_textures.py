"""Texture provenance, rendering indices, archive persistence and colour regions."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pyvista as pv
import trimesh
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from texture_geometry import (KEY, add_layer, appearance, bake_colors, change_layer, effective_colors,
    fresh, paint, project_uv, read_image, split_colors, validate_appearance)
from texture_render import display_data
from project_store import ProjectState, load_project, save_project
from project_history import ProjectHistory
from part_supports import make_group, validate_groups
from cad_state import apply_cad_transform, cad_status
import test_cad_state


class TextureTests(unittest.TestCase):
    def image(self, color=(210, 80, 40)):
        return np.tile(np.uint8(color), (8, 12, 1))

    def source(self, mesh):
        return pv.PolyData(mesh.vertices, np.column_stack([np.full(len(mesh.faces), 3), mesh.faces]).ravel())

    def test_layers_partial_clear_visibility_and_paint_keep_cad_native(self):
        mesh = test_cad_state.CADStateTests().native(); original = mesh.copy()
        textured, identifier = add_layer(mesh, self.image(), ids=[0, 1])
        self.assertEqual(cad_status(textured), 'native')
        self.assertNotIn(KEY, mesh.metadata)
        np.testing.assert_array_equal(textured.vertices, original.vertices)
        np.testing.assert_array_equal(textured.faces, original.faces)
        np.testing.assert_array_equal(effective_colors(textured)[:2, :3], np.tile([210, 80, 40], (2, 1)))
        changed = change_layer(textured, identifier, 'clear', ids=[0])
        self.assertEqual(fresh(changed)['layers'][0]['mask'].sum(), 1)
        hidden = change_layer(changed, identifier, 'invert')
        self.assertFalse(fresh(hidden)['layers'][0]['visible'])
        self.assertTrue(fresh(changed)['layers'][0]['visible'])
        painted = paint(changed, [10, 30, 200, 255], ids=[1])
        np.testing.assert_array_equal(effective_colors(painted)[1], [10, 30, 200, 255])
        self.assertEqual(cad_status(painted), 'native')
        self.assertEqual(len(fresh(change_layer(painted, identifier, 'delete'))['layers']), 0)

    def test_all_projections_are_finite_for_flat_faces_and_repeat_validation(self):
        mesh = trimesh.creation.box()
        for mode in ('По граням', 'XY', 'XZ', 'YZ', 'Цилиндр', 'Сфера'):
            uv = project_uv(mesh, dict(projection=mode, repeat_u=2, offset_v=-.2, angle=45), np.ones(12, dtype=bool))
            self.assertEqual(uv.shape, (12, 3, 2)); self.assertTrue(np.isfinite(uv).all())
        with self.assertRaises(ValueError): project_uv(mesh, {'repeat_u': 20}, np.ones(12, dtype=bool))
        with self.assertRaises(ValueError): add_layer(mesh, self.image(), ids=[12])

    def test_display_proxy_keeps_face_order_normals_and_sources_immutable(self):
        mesh = trimesh.creation.box()
        mesh, one = add_layer(mesh, self.image(), ids=[0, 1])
        mesh, two = add_layer(mesh, self.image((20, 180, 90)), {'projection': 'Сфера', 'repeat_u': 2}, ids=[2, 3])
        source = self.source(mesh).compute_normals(split_vertices=False, consistent_normals=False)
        data, texture, colors = display_data(mesh, source)
        self.assertEqual(data.n_cells, len(mesh.faces)); self.assertEqual(data.n_points, len(mesh.faces) * 3)
        np.testing.assert_array_equal(data.points.reshape(-1, 3, 3), mesh.triangles)
        self.assertEqual(colors, 'face'); self.assertIsNotNone(texture)
        self.assertIn('Normals', data.point_data)
        self.assertNotIn('_display_colors', source.cell_data)
        self.assertTrue(((data.active_texture_coordinates >= 0) & (data.active_texture_coordinates <= 1)).all())
        hidden = change_layer(change_layer(mesh, one, 'invert'), two, 'invert')
        _, texture, _ = display_data(hidden, self.source(hidden))
        self.assertIsNone(texture)

    def test_mirror_keeps_uv_with_its_corner_and_topology_edits_drop_obsolete_mapping(self):
        mesh, identifier = add_layer(trimesh.creation.box(), self.image(), {'projection': 'XY'})
        value = fresh(mesh); expected = value['layers'][0]['uv'].copy(); old_faces = mesh.faces.copy()
        mirrored = mesh.copy(); apply_cad_transform(mirrored, np.diag([-1., 1., 1., 1.]))
        actual = fresh(mirrored)['layers'][0]['uv']
        for face in range(len(mesh.faces)):
            for corner in range(3):
                old = np.flatnonzero(old_faces[face] == mirrored.faces[face, corner])[0]
                np.testing.assert_array_equal(actual[face, corner], expected[face, old])
        edited = mesh.copy(); edited.update_faces(np.arange(11))
        self.assertIsNone(appearance(edited))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'edited.mrp'; save_project(path, ProjectState(parts=[dict(mesh=edited, filename='edited')]))
            self.assertNotIn(KEY, load_project(path).parts[0]['mesh'].metadata)

    def test_project_embeds_images_uv_colors_history_undo_redo_and_source_can_disappear(self):
        mesh = test_cad_state.CADStateTests().native(); supports = [make_group(trimesh.creation.box(), [0, 1])]
        history = ProjectHistory(); history.reset(ProjectState(parts=[dict(mesh=mesh, filename='cad', supports=supports)]))
        painted = paint(mesh, [10, 20, 30, 255], [3])
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / 'цвет.png'; Image.fromarray(self.image()).save(image_path)
            textured, identifier = add_layer(painted, read_image(image_path), ids=[0, 1], path=image_path)
            state = ProjectState(parts=[dict(mesh=textured, filename='CAD.step', supports=supports)])
            history.push(state, 'Текстура'); saved = Path(folder) / 'test.mrp'; save_project(saved, state)
            image_path.unlink(); loaded = load_project(saved).parts[0]
            self.assertEqual(cad_status(loaded['mesh']), 'native'); validate_groups(loaded['supports'], 12)
            np.testing.assert_array_equal(fresh(loaded['mesh'])['layers'][0]['image'], self.image())
            np.testing.assert_array_equal(effective_colors(loaded['mesh'])[3], [10, 20, 30, 255])
            self.assertNotIn(KEY, history.entries[0][1].parts[0]['mesh'].metadata)
            self.assertEqual(history.entries[1][1].parts[0]['mesh'].metadata[KEY]['layers'][0]['id'], identifier)

    def test_bake_preserves_colors_and_split_remaps_support_owner_without_duplicates(self):
        mesh = paint(trimesh.creation.box(), [200, 0, 0, 255])
        mesh = paint(mesh, [0, 0, 200, 255], [0, 1])
        baked, identifier = bake_colors(mesh)
        np.testing.assert_array_equal(effective_colors(baked), effective_colors(mesh))
        group = make_group(trimesh.creation.box(), [0, 1])
        regions = split_colors(baked, [group]); self.assertEqual(len(regions), 2)
        self.assertEqual(sum(len(r['mesh'].faces) for r in regions), 12)
        self.assertEqual(sum(len(r['supports']) for r in regions), 1)
        owner = next(r for r in regions if r['supports']); validate_groups(owner['supports'], len(owner['mesh'].faces))
        self.assertEqual(owner['supports'][0]['id'], group['id'])
        np.testing.assert_array_equal(owner['mesh'].triangles, mesh.triangles[owner['source_faces']])
        with self.assertRaisesRegex(ValueError, 'разным цветам'):
            split_colors(mesh, [make_group(trimesh.creation.box(), [0, 4])])

    def test_corrupt_arrays_are_rejected(self):
        mesh, identifier = add_layer(trimesh.creation.box(), self.image())
        for field, data in [('image', np.zeros((4, 4), dtype=np.uint8)), ('uv', np.full((12, 3, 2), np.nan)), ('mask', np.zeros(11, dtype=bool))]:
            value = deepcopy(mesh.metadata[KEY]); value['layers'][0][field] = data
            with self.assertRaises(ValueError): validate_appearance(value, 12)

    def test_transparent_image_empty_layer_edit_and_invalid_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'alpha.png'
            pixels = np.zeros((4, 4, 4), dtype=np.uint8); pixels[0, 0] = [100, 20, 60, 255]
            Image.fromarray(pixels).save(path); image = read_image(path)
            np.testing.assert_array_equal(image[1, 1], [255, 255, 255])
            np.testing.assert_array_equal(image[0, 0], [100, 20, 60])
        mesh, identifier = add_layer(trimesh.creation.box(), self.image())
        cleared = change_layer(mesh, identifier, 'clear')
        edited = change_layer(cleared, identifier, 'edit', params={'projection': 'Сфера'})
        self.assertFalse(fresh(edited)['layers'][0]['mask'].any())
        with self.assertRaises(ValueError): add_layer(mesh, self.image(), ids=[.5])

    def test_ribbon_has_all_15_commands_original_icons_and_no_placeholders(self):
        from test_desktop import APP
        from texture_ribbon import COMMANDS, GROUPS, create_texture_ribbon
        from PySide6.QtWidgets import QLabel
        ribbon, buttons = create_texture_ribbon()
        self.assertEqual(len(buttons), 15); self.assertEqual(list(buttons), list(COMMANDS))
        self.assertEqual([label.text() for label in ribbon.findChildren(QLabel)], [name for name, _ in GROUPS])
        for op, button in buttons.items(): self.assertTrue(button.isEnabled()); self.assertFalse(button.icon().isNull())
        ribbon.deleteLater(); APP.processEvents()


if __name__ == '__main__': unittest.main()
