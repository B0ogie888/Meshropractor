from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repair_manual_geometry import edit_mesh, split_components, combine_meshes


class ManualRepairTests(unittest.TestCase):
    def setUp(self):
        self.box = trimesh.creation.box()
        self.box.metadata = {'nested': {'label': 'original'}}
        self.vertices = self.box.vertices.copy()
        self.faces = self.box.faces.copy()
        self.metadata = deepcopy(self.box.metadata)

    def tearDown(self):
        np.testing.assert_array_equal(self.box.vertices, self.vertices)
        np.testing.assert_array_equal(self.box.faces, self.faces)
        self.assertEqual(self.box.metadata, self.metadata)

    def open_top(self):
        result = self.box.copy()
        result.update_faces(result.face_normals[:, 2] < .9)
        return result

    def test_delete_selected_only_and_keep_vertex_ids(self):
        edited, report = edit_mesh(self.box, 'delete_faces', {}, face_ids=[1, 4])
        np.testing.assert_array_equal(edited.vertices, self.box.vertices)
        np.testing.assert_array_equal(edited.faces, np.delete(self.box.faces, [1, 4], axis=0))
        self.assertEqual(report['deleted_faces'], 2)
        self.assertTrue(report['vertex_ids_preserved'])
        edited.metadata['nested']['label'] = 'changed'

    def test_invalid_ids_and_delete_all_are_rejected(self):
        for ids in (None, [], [0, 0], [-1], [12], [1.2], [True], list(range(12))):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                edit_mesh(self.box, 'delete_faces', {}, face_ids=ids)

    def test_add_triangle_closes_exact_missing_face(self):
        opened = self.box.copy()
        opened.update_faces(np.arange(len(opened.faces) - 1))
        edited, report = edit_mesh(opened, 'add_triangle', {}, vertex_ids=self.box.faces[-1])
        self.assertTrue(edited.is_watertight)
        self.assertTrue(edited.is_winding_consistent)
        self.assertEqual(report['added_faces'], 1)
        self.assertEqual(len(opened.faces), 11)

    def test_add_triangle_rejects_duplicate_degenerate_and_wrong_winding(self):
        for ids in (self.box.faces[0], self.box.faces[0][::-1], [0, 1, 6]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                edit_mesh(self.box, 'add_triangle', {}, vertex_ids=ids)
        opened = self.box.copy()
        opened.update_faces(np.arange(11))
        with self.assertRaisesRegex(ValueError, 'нормал'):
            edit_mesh(opened, 'add_triangle', {}, vertex_ids=self.box.faces[-1][::-1])
        flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
                               faces=[[0, 1, 2]], process=False)
        with self.assertRaisesRegex(ValueError, 'вырожден'):
            edit_mesh(flat, 'add_triangle', {}, vertex_ids=[0, 1, 3])

    def test_bridge_two_opposite_boundary_edges_closes_top(self):
        opened = self.open_top()
        counts = np.bincount(opened.edges_unique_inverse)
        edges = opened.edges[counts[opened.edges_unique_inverse] == 1]
        first = edges[0]
        second = next(edge for edge in edges if not set(first) & set(edge))
        result, report = edit_mesh(opened, 'bridge', {}, vertex_ids=[*first[::-1], *second])
        self.assertEqual(report['added_faces'], 2)
        self.assertTrue(result.is_volume)
        self.assertAlmostEqual(result.volume, 1)
        self.assertEqual(len(opened.faces), 10)
        with self.assertRaisesRegex(ValueError, 'границ'):
            edit_mesh(self.box, 'bridge', {}, vertex_ids=[*first, *second])

    def test_move_vertices_delta_absolute_and_noop(self):
        result, report = edit_mesh(self.box, 'move_vertices', {'delta': [.1, 0, 0]}, vertex_ids=[0, 1])
        np.testing.assert_allclose(result.vertices[:2], self.vertices[:2] + [.1, 0, 0])
        np.testing.assert_array_equal(result.vertices[2:], self.vertices[2:])
        self.assertEqual(report['moved_vertices'], 2)
        absolute = self.box.vertices[0] + [.1, .1, .1]
        result, _ = edit_mesh(self.box, 'move_vertices', {'absolute': absolute}, vertex_ids=[0])
        np.testing.assert_allclose(result.vertices[0], absolute)
        _, info = edit_mesh(self.box, 'move_vertices', {'delta': [0, 0, 0]}, vertex_ids=[0])
        self.assertFalse(info['changed'])

    def test_move_rejects_ambiguous_nonfinite_and_collapsed_faces(self):
        for parameters, ids in (({}, [0]), ({'delta': [0, 0, 0], 'absolute': [1, 1, 1]}, [0]),
                                ({'absolute': [0, 0, 0]}, [0, 1]), ({'delta': [np.nan, 0, 0]}, [0]),
                                ({'absolute': self.box.vertices[1]}, [0])):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                edit_mesh(self.box, 'move_vertices', parameters, vertex_ids=ids)

    def test_fill_hole_by_vertex_and_by_incident_face(self):
        opened = self.open_top()
        top_vertex = int(np.flatnonzero(opened.vertices[:, 2] > 0)[0])
        result, report = edit_mesh(opened, 'fill_hole', {}, vertex_ids=[top_vertex], face_ids=[])
        self.assertTrue(result.is_volume)
        self.assertEqual(report['added_faces'], 2)
        incident = int(np.flatnonzero((opened.faces == top_vertex).any(axis=1))[0])
        result, _ = edit_mesh(opened, 'fill_hole', {}, face_ids=[incident], vertex_ids=[])
        self.assertTrue(result.is_volume)

    def test_fill_one_hole_keeps_other_open_and_rejects_multiple_selection(self):
        tube = trimesh.creation.cylinder(radius=1, height=2, sections=8)
        tube.update_faces(np.abs(tube.face_normals[:, 2]) < .5)
        referenced = np.unique(tube.faces)
        top = int(referenced[np.argmax(tube.vertices[referenced, 2])])
        bottom = int(referenced[np.argmin(tube.vertices[referenced, 2])])
        result, report = edit_mesh(tube, 'fill_hole', {}, vertex_ids=[top])
        self.assertFalse(result.is_watertight)
        self.assertTrue(result.is_winding_consistent)
        self.assertEqual(report['before']['boundary'], 16)
        self.assertEqual(report['after']['boundary'], 8)
        with self.assertRaisesRegex(ValueError, 'несколько'):
            edit_mesh(tube, 'fill_hole', {}, vertex_ids=[top, bottom])

    def test_fill_concave_contour(self):
        ring = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
        vertices = np.vstack((np.column_stack((ring, np.zeros(6))),
                              np.column_stack((ring, np.ones(6)))))
        faces = []
        for a in range(6):
            b = (a + 1) % 6
            faces.extend([[a, b, b + 6], [a, b + 6, a + 6]])
        wall = trimesh.Trimesh(vertices, faces, process=False)
        result, report = edit_mesh(wall, 'fill_hole', {}, vertex_ids=[6])
        self.assertEqual(report['added_faces'], 4)
        self.assertEqual(report['after']['boundary'], 6)
        self.assertAlmostEqual(result.area - wall.area, 3.)

    def test_fill_rejects_self_crossing_and_branched_boundaries(self):
        ring = np.array([[0, 0], [2, 2], [0, 2], [2, 0]])
        vertices = np.vstack((np.column_stack((ring, np.zeros(4))),
                              np.column_stack((ring, np.ones(4)))))
        faces = []
        for a in range(4):
            b = (a + 1) % 4
            faces.extend([[a, b, b + 4], [a, b + 4, a + 4]])
        wall = trimesh.Trimesh(vertices, faces, process=False)
        with self.assertRaises(ValueError):
            edit_mesh(wall, 'fill_hole', {}, vertex_ids=[4])
        branching = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                     [-1, 0, 0], [0, -1, 0]], [[0, 1, 2], [0, 3, 4]], process=False)
        with self.assertRaisesRegex(ValueError, 'разветвлён'):
            edit_mesh(branching, 'fill_hole', {}, vertex_ids=[0])

    def test_clip_both_sides_caps_and_open_cut(self):
        for side, expected in (('positive', [0, .5]), ('negative', [-.5, 0])):
            with self.subTest(side=side):
                parameters = dict(point=[0, 0, 0], normal=[2, 0, 0], side=side, cap=True)
                result, info = edit_mesh(self.box, 'clip', parameters)
                self.assertTrue(result.is_volume)
                self.assertAlmostEqual(result.volume, .5)
                np.testing.assert_allclose(result.bounds[:, 0], expected)
                self.assertFalse(info['vertex_ids_preserved'])
                parameters['cap'] = False
                result, _ = edit_mesh(self.box, 'clip', parameters)
                self.assertFalse(result.is_watertight)

    def test_clip_cap_preserves_inner_void(self):
        outer = trimesh.creation.box(extents=[4, 4, 4])
        inner = trimesh.creation.box(extents=[2, 2, 2])
        inner.invert()
        hollow = trimesh.util.concatenate([outer, inner])
        result, _ = edit_mesh(hollow, 'clip', dict(point=[0, 0, 0], normal=[0, 0, 1], cap=True))
        self.assertTrue(result.is_volume)
        self.assertAlmostEqual(result.volume, 28.)
        self.assertFalse(result.contains([[0, 0, .5]])[0])

    def test_clip_noop_delete_all_invalid_plane_and_large_coordinates(self):
        _, info = edit_mesh(self.box, 'clip', dict(point=[-2, 0, 0], normal=[1, 0, 0]))
        self.assertFalse(info['changed'])
        for parameters in (dict(point=[2, 0, 0], normal=[1, 0, 0]),
                           dict(point=[0, 0, 0], normal=[0, 0, 0]),
                           dict(point=[0, 0, 0], normal=[1, 0, 0], side='both')):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                edit_mesh(self.box, 'clip', parameters)
        shifted = self.box.copy()
        shifted.apply_translation([1e8, 1e8, 1e8])
        result, _ = edit_mesh(shifted, 'clip', dict(point=[1e8]*3, normal=[1, 0, 0], cap=True))
        self.assertTrue(result.is_watertight)
        local = result.copy()
        local.apply_translation([-1e8]*3)
        self.assertAlmostEqual(local.volume, .5)

    def test_clip_never_mutates_parameter_arrays_and_normal_can_be_rescaled(self):
        normal = np.array([2., 0, 0])
        point = np.array([0., 0, 0])
        result, _ = edit_mesh(self.box, 'clip', dict(point=point, normal=normal, cap=True))
        np.testing.assert_array_equal(normal, [2., 0, 0])
        np.testing.assert_array_equal(point, [0., 0, 0])
        large, _ = edit_mesh(self.box, 'clip', dict(point=point, normal=[1e300, 0, 0], cap=True))
        self.assertAlmostEqual(large.volume, result.volume)

    def test_cancellation_and_unknown_operation_leave_source_unchanged(self):
        with self.assertRaises(InterruptedError):
            edit_mesh(self.box, 'delete_faces', {}, face_ids=[0], cancelled=lambda: True)
        messages = []
        with self.assertRaises(InterruptedError):
            edit_mesh(self.box, 'delete_faces', {}, face_ids=[0], progress=messages.append,
                      cancelled=lambda: len(messages) > 1)
        with self.assertRaises(ValueError):
            edit_mesh(self.box, 'unknown', {})

    def test_split_retains_open_and_single_triangle_components_without_repair(self):
        open_box = self.open_top()
        triangle = trimesh.Trimesh([[3, 0, 0], [4, 0, 0], [3, 1, 0]], [[0, 1, 2]], process=False)
        combined = combine_meshes([open_box, triangle], boolean=False)
        parts = split_components(combined)
        self.assertEqual(sorted(len(part.faces) for part in parts), [1, 10])
        self.assertTrue(all(not part.is_watertight for part in parts))
        self.assertEqual(combined.metadata['combination'], 'group')

    def test_split_face_maps_preserve_exact_source_ids_for_coincident_components(self):
        triangle = trimesh.Trimesh([[3, 0, 0], [4, 0, 0], [3, 1, 0]], [[0, 1, 2]], process=False)
        source = trimesh.util.concatenate([self.box, self.box.copy(), triangle])
        # Interleave two geometrically identical but topologically separate shells.
        order = np.ravel(np.column_stack((np.arange(12), np.arange(12, 24))))
        order = np.insert(order, 7, 24)
        source.faces = source.faces[order]
        source.metadata = {'nested': {'label': 'source'}}
        vertices, faces = source.vertices.copy(), source.faces.copy()
        parts, maps = split_components(source, return_face_maps=True)
        expected = {tuple(np.flatnonzero(order < 12)),
                    tuple(np.flatnonzero((order >= 12) & (order < 24))),
                    tuple(np.flatnonzero(order == 24))}
        self.assertEqual({tuple(face_map) for face_map in maps}, expected)
        self.assertEqual(sorted(len(part.faces) for part in parts), [1, 12, 12])
        np.testing.assert_array_equal(np.sort(np.concatenate(maps)), np.arange(len(faces)))
        for part, face_map in zip(parts, maps):
            self.assertEqual(face_map.dtype, np.dtype(np.int64))
            np.testing.assert_array_equal(part.triangles, vertices[faces[face_map]])
            # Submesh vertex IDs may change; face ordering and winding do not.
            np.testing.assert_array_equal(part.face_normals, source.face_normals[face_map])
        default_parts = split_components(source)
        self.assertIsInstance(default_parts, list)
        for default_part, part in zip(default_parts, parts):
            np.testing.assert_array_equal(default_part.vertices, part.vertices)
            np.testing.assert_array_equal(default_part.faces, part.faces)
        parts[0].vertices[0] += 10
        parts[0].metadata['nested']['label'] = 'edited'
        maps[0][0] = -1
        np.testing.assert_array_equal(source.vertices, vertices)
        np.testing.assert_array_equal(source.faces, faces)
        self.assertEqual(source.metadata, {'nested': {'label': 'source'}})
        self.assertTrue(all(part.metadata['nested']['label'] == 'source' for part in parts[1:]))

    def test_split_face_maps_include_disconnected_single_triangles(self):
        source = trimesh.Trimesh(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
            [[3, 4, 5], [0, 1, 2]], process=False)
        parts, maps = split_components(source, return_face_maps=True)
        self.assertEqual({tuple(face_map) for face_map in maps}, {(0,), (1,)})
        self.assertEqual(len(parts), 2)
        for part, face_map in zip(parts, maps):
            self.assertEqual(len(part.faces), 1)
            np.testing.assert_array_equal(part.triangles, source.triangles[face_map])

    def test_split_fragment_limit_precedes_submesh_allocation_and_preserves_source(self):
        vertices = np.tile([[0., 0, 0], [1, 0, 0], [0, 1, 0]], (1001, 1))
        vertices[:, 0] += np.repeat(np.arange(1001) * 3, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        source = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
        source.metadata = {'nested': {'label': 'unchanged'}}
        with patch.object(trimesh.Trimesh, 'submesh', side_effect=AssertionError('Allocated submeshes')):
            with self.assertRaisesRegex(ValueError, 'Слишком много фрагментов: 1001.*удалите шум'):
                split_components(source, return_face_maps=True, max_components=200)
        np.testing.assert_array_equal(source.vertices, vertices)
        np.testing.assert_array_equal(source.faces, faces)
        self.assertEqual(source.metadata, {'nested': {'label': 'unchanged'}})

    def test_boolean_union_coplanar_overlapping_disjoint_and_large_offset(self):
        for shift, volume in (([.5, 0, 0], 1.5), ([2, 0, 0], 2.), ([0, 0, 0], 1.)):
            with self.subTest(shift=shift):
                other = self.box.copy()
                other.apply_translation(shift)
                original = other.vertices.copy()
                result = combine_meshes([self.box, other])
                self.assertTrue(result.is_volume)
                self.assertAlmostEqual(result.volume, volume, places=6)
                self.assertEqual(result.metadata['combination'], 'boolean_union')
                np.testing.assert_array_equal(other.vertices, original)
        a, b = self.box.copy(), self.box.copy()
        a.apply_translation([1e8]*3)
        b.apply_translation([1e8 + .5, 1e8, 1e8])
        result = combine_meshes([a, b])
        result.apply_translation([-1e8]*3)
        self.assertAlmostEqual(result.volume, 1.5, places=6)

    def test_union_never_falls_back_to_grouping_or_accepts_open_mesh(self):
        with self.assertRaisesRegex(ValueError, 'замкнут'):
            combine_meshes([self.box, self.open_top()])
        with patch('trimesh.boolean.engines_available', set()), self.assertRaisesRegex(ValueError, 'manifold3d'):
            combine_meshes([self.box, self.box])
        with self.assertRaises(ValueError):
            combine_meshes([self.box])
        grouped = combine_meshes([self.box, self.box], boolean=False)
        self.assertEqual(len(grouped.faces), 24)
        self.assertAlmostEqual(grouped.volume, 2.)


if __name__ == '__main__':
    unittest.main()
