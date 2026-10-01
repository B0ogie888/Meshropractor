import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repair_operations import repair_mesh


class RepairOperationsTests(unittest.TestCase):
    def setUp(self):
        self.box = trimesh.creation.box(extents=[4, 4, 4])
        self.box.metadata = {'source': 'CAD', 'nested': {'values': [1, 2, 3]}}

    def assert_source(self, source, vertices, faces):
        np.testing.assert_array_equal(source.vertices, vertices)
        np.testing.assert_array_equal(source.faces, faces)

    def test_normals_fix_and_partial_flip_preserve_source_and_metadata(self):
        mesh = self.box.copy()
        mesh.invert()
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        result, report = repair_mesh(mesh, 'normals', {})
        self.assertTrue(result.is_winding_consistent)
        self.assertGreater(result.volume, 0)
        self.assertTrue(report['changed'])
        result.metadata['nested']['values'].append(4)
        self.assertEqual(mesh.metadata['nested']['values'], [1, 2, 3])
        self.assert_source(mesh, vertices, faces)
        result, _ = repair_mesh(self.box, 'normals', {'flip': True}, face_ids=[0])
        np.testing.assert_array_equal(result.faces[0], self.box.faces[0, ::-1])
        np.testing.assert_array_equal(result.faces[1:], self.box.faces[1:])

    def test_normals_preserve_hollow_shells_and_capped_cut_volume(self):
        from repair_manual_geometry import edit_mesh
        inner = trimesh.creation.box(extents=[2, 2, 2])
        inner.invert()
        hollow = trimesh.util.concatenate([self.box, inner])
        for kind in ('correct', 'inverted', 'one_face'):
            source = hollow.copy()
            if kind == 'inverted': source.invert()
            if kind == 'one_face': source.faces[13] = source.faces[13, ::-1]
            vertices, faces = source.vertices.copy(), source.faces.copy()
            with self.subTest(kind=kind):
                result, report = repair_mesh(source, 'normals', {})
                self.assertTrue(result.is_winding_consistent)
                self.assertEqual(report['closed_shells'], 2)
                self.assertEqual(report['cavity_shells'], 1)
                self.assertAlmostEqual(result.volume, 56.)
                self.assertEqual(report['changed'], kind != 'correct')
                np.testing.assert_array_equal(result.vertices, vertices)
                np.testing.assert_array_equal(np.sort(result.faces, axis=1), np.sort(faces, axis=1))
                self.assert_source(source, vertices, faces)
                cut, _ = edit_mesh(result, 'clip', dict(point=[0, 0, 0], normal=[0, 0, 1], cap=True))
                self.assertTrue(cut.is_watertight)
                self.assertAlmostEqual(cut.volume, 28.)

    def test_normals_orient_disconnected_solids_and_three_nested_shells(self):
        shells = [trimesh.creation.box(extents=[size] * 3) for size in (6, 4, 2)]
        detached = trimesh.creation.box(extents=[3] * 3)
        detached.apply_translation([12, 0, 0])
        shells.append(detached)
        for mesh in shells: mesh.invert()
        source = trimesh.util.concatenate(shells)
        result, report = repair_mesh(source, 'normals', {})
        self.assertEqual(report['closed_shells'], 4)
        self.assertEqual(report['cavity_shells'], 1)
        self.assertAlmostEqual(result.volume, 216. - 64. + 8. + 27.)
        self.assertTrue(result.is_winding_consistent)

    def test_normals_bounds_containment_is_not_sufficient_for_cavity(self):
        annulus = trimesh.creation.annulus(r_min=2., r_max=3., height=4., sections=24)
        center = trimesh.creation.box()
        center.invert()
        source = trimesh.util.concatenate([annulus, center])
        result, report = repair_mesh(source, 'normals', {})
        self.assertEqual(report['cavity_shells'], 0)
        self.assertAlmostEqual(result.volume, annulus.volume + 1.)

    def test_normals_reject_touching_and_intersecting_closed_shells(self):
        for offset in ([1, 0, 0], [4, 0, 0], [4, 4, 4], [0, 0, 0]):
            other = self.box.copy()
            other.apply_translation(offset)
            source = trimesh.util.concatenate([self.box, other])
            vertices, faces = source.vertices.copy(), source.faces.copy()
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'пересекаются|касаются'):
                repair_mesh(source, 'normals', {})
            self.assert_source(source, vertices, faces)

    def test_normals_partial_repair_keeps_other_faces_and_handles_large_offset(self):
        inner = trimesh.creation.box(extents=[2, 2, 2])
        inner.invert()
        source = trimesh.util.concatenate([self.box, inner])
        source.apply_translation([1e8] * 3)
        source.faces[13] = source.faces[13, ::-1]
        result, _ = repair_mesh(source, 'normals', {}, face_ids=[13])
        untouched = np.arange(len(source.faces)) != 13
        np.testing.assert_array_equal(result.faces[untouched], source.faces[untouched])
        self.assertTrue(result.is_winding_consistent)
        result.apply_translation([-1e8] * 3)
        self.assertAlmostEqual(result.volume, 56.)

    def test_normals_shell_budget_cancellation_and_open_patch_warning(self):
        inner = trimesh.creation.box()
        source = trimesh.util.concatenate([self.box, inner])
        with patch('repair_operations.MAX_SHELL_PAIRS', 0), self.assertRaisesRegex(ValueError, 'Слишком много'):
            repair_mesh(source, 'normals', {})
        messages = []
        with self.assertRaises(InterruptedError):
            repair_mesh(source, 'normals', {}, progress=messages.append,
                        cancelled=lambda: any('вложенности' in message for message in messages))
        patch_mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
        _, report = repair_mesh(patch_mesh, 'normals', {})
        self.assertEqual(report['closed_shells'], 0)
        self.assertTrue(any('Открытые' in warning for warning in report['warnings']))

    def test_stitch_uses_distance_tolerance_and_noop_preserves_face_ids(self):
        result, report = repair_mesh(self.box, 'stitch', {'tolerance_mm': .001})
        self.assertFalse(report['changed'])
        np.testing.assert_array_equal(result.faces, self.box.faces)
        # An imported STL stores each triangle separately; offset one copy of
        # a shared vertex by less than the requested physical tolerance.
        vertices = self.box.triangles.reshape(-1, 3).copy()
        vertices[0, 0] += .002
        soup = trimesh.Trimesh(vertices, np.arange(len(vertices)).reshape(-1, 3), process=False)
        result, report = repair_mesh(soup, 'stitch', {'tolerance_mm': .005})
        self.assertTrue(result.is_watertight)
        self.assertEqual(len(result.vertices), 8)
        self.assertEqual(len(soup.vertices), 36)

    def test_duplicates_are_detected_by_coordinates_even_with_separate_indices(self):
        original = self.box.triangles[0]
        vertices = np.vstack((self.box.vertices, original[::-1]))
        faces = np.vstack((self.box.faces, [8, 9, 10]))
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        result, report = repair_mesh(mesh, 'duplicates', {})
        self.assertEqual(report['removed_faces'], 1)
        self.assertEqual(len(result.faces), len(self.box.faces))
        self.assertTrue(result.is_watertight)
        self.assertEqual(len(mesh.faces), 13)

    def test_holes_respects_diameter_and_can_close_planar_boundary(self):
        mesh = self.box.copy()
        # Remove both triangles belonging to the same planar face.
        normal = mesh.face_normals[0]
        mesh.update_faces(mesh.face_normals @ normal < .99)
        result, small = repair_mesh(mesh, 'holes', {'max_diameter_mm': 1})
        self.assertEqual(small['holes_filled'], 0)
        self.assertFalse(result.is_watertight)
        result, report = repair_mesh(mesh, 'holes', {'max_diameter_mm': 6})
        self.assertEqual(report['holes_filled'], 1)
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(abs(result.volume), abs(self.box.volume))
        self.assertFalse(mesh.is_watertight)

    def test_noise_protects_largest_geometry_not_densest_fragment(self):
        tiny = trimesh.creation.icosphere(subdivisions=2, radius=.05)
        tiny.apply_translation([20, 0, 0])
        mesh = trimesh.util.concatenate([self.box, tiny])
        result, report = repair_mesh(mesh, 'noise', {'min_faces': 0, 'min_volume_mm3': 1, 'keep_largest': True})
        self.assertEqual(report['removed_components'], 1)
        self.assertEqual(len(result.faces), 12)
        np.testing.assert_allclose(result.bounds, self.box.bounds)
        with self.assertRaises(ValueError):
            repair_mesh(self.box, 'noise', {'min_faces': 100, 'keep_largest': False})

    def test_holes_fill_concave_boundaries_without_covering_the_notch(self):
        ring = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
        vertices = np.vstack((np.column_stack((ring, np.zeros(6))),
                              np.column_stack((ring, np.ones(6)))))
        faces = []
        for first in range(6):
            second = (first + 1) % 6
            faces.extend([[first, second, second + 6], [first, second + 6, first + 6]])
        wall = trimesh.Trimesh(vertices, faces, process=False)
        result, report = repair_mesh(wall, 'holes', {'max_diameter_mm': 3})
        self.assertEqual(report['holes_filled'], 2)
        self.assertTrue(result.is_watertight)
        self.assertTrue(result.is_winding_consistent)
        self.assertAlmostEqual(result.volume, 3.)
        self.assertAlmostEqual(result.area - wall.area, 6.)
        self.assert_source(wall, vertices, np.asarray(faces))

    def test_holes_skip_ambiguous_boundary_without_blocking_valid_hole(self):
        opened = self.box.copy()
        opened.update_faces(opened.face_normals @ opened.face_normals[0] < .99)
        branching = trimesh.Trimesh([[10, 0, 0], [11, 0, 0], [10, 1, 0],
                                     [9, 0, 0], [10, -1, 0]], [[0, 1, 2], [0, 3, 4]], process=False)
        source = trimesh.util.concatenate([opened, branching])
        result, report = repair_mesh(source, 'holes', {'max_diameter_mm': 6})
        self.assertEqual(report['holes_filled'], 1)
        self.assertEqual(report['holes_skipped']['ambiguous'], 1)
        self.assertEqual(len(result.faces), len(source.faces) + 2)
        self.assertTrue(report['warnings'])

    def test_noise_does_not_use_false_volume_for_open_components(self):
        patch_mesh = trimesh.Trimesh([[10, 0, 0], [11, 0, 0], [10, 1, 0]], [[0, 1, 2]], process=False)
        source = trimesh.util.concatenate([self.box, patch_mesh])
        result, report = repair_mesh(source, 'noise', {'min_faces': 0, 'min_volume_mm3': 1})
        self.assertEqual(len(result.faces), 13)
        self.assertTrue(report['warnings'])

    def test_smoothing_pins_selection_border_and_changes_only_interior(self):
        # A square patch with one raised interior vertex.
        source = trimesh.Trimesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]],
                                [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], process=False)
        result, report = repair_mesh(source, 'smooth', {'iterations': 10, 'relaxation': .2})
        np.testing.assert_array_equal(result.vertices[:4], source.vertices[:4])
        self.assertNotEqual(result.vertices[4, 2], source.vertices[4, 2])
        self.assertTrue(report['changed'])
        partial, report = repair_mesh(source, 'smooth', {}, face_ids=[0, 1])
        # Every selected vertex touches the fixed boundary or an unselected face.
        np.testing.assert_array_equal(partial.vertices, source.vertices)
        self.assertFalse(report['changed'])

    def test_clean_smooth_removes_duplicate_and_degenerate_faces(self):
        source = trimesh.Trimesh(self.box.vertices.copy(), np.vstack((self.box.faces, self.box.faces[0], [0, 0, 1])), process=False)
        result, _ = repair_mesh(source, 'clean_smooth', {'iterations': 2})
        self.assertTrue(result.nondegenerate_faces().all())
        self.assertTrue(result.unique_faces().all())
        self.assertTrue(result.is_watertight)

    def test_decimation_and_subdivision_retain_closed_topology(self):
        source = trimesh.creation.icosphere(subdivisions=2)
        vertices, faces = source.vertices.copy(), source.faces.copy()
        result, report = repair_mesh(source, 'decimate', {'target_ratio': .5})
        self.assertLessEqual(len(result.faces), len(source.faces) * .6)
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(report['actual_ratio'], len(result.faces) / len(source.faces))
        subdivided, _ = repair_mesh(source, 'subdivide', {'iterations': 1})
        self.assertEqual(len(subdivided.faces), len(source.faces) * 4)
        self.assertTrue(subdivided.is_watertight)
        np.testing.assert_allclose(subdivided.bounds, source.bounds)
        self.assert_source(source, vertices, faces)

    def test_remesh_is_bounded_and_reports_approximation(self):
        result, report = repair_mesh(self.box, 'remesh', {'target_edge_mm': 1.5, 'iterations': 2})
        self.assertGreater(len(result.faces), len(self.box.faces))
        self.assertTrue(result.is_watertight)
        self.assertTrue(np.isfinite(result.vertices).all())
        self.assertTrue(report['warnings'])
        with self.assertRaises(ValueError):
            repair_mesh(self.box, 'remesh', {'target_edge_mm': 1e-200})

    def test_voxel_wrap_closes_small_open_model_and_rejects_excessive_grid(self):
        source = self.box.copy()
        source.update_faces(np.arange(len(source.faces)) != 0)
        result, report = repair_mesh(source, 'wrap', {'voxel_size_mm': .5})
        self.assertTrue(result.is_watertight)
        self.assertTrue(result.is_winding_consistent)
        self.assertGreater(result.volume, 0)
        self.assertLess(report['voxel_count'], 4_000_000)
        self.assertTrue(report['warnings'])
        with patch('repair_operations._native', side_effect=AssertionError('Grid budget must fail before native sampling')):
            with self.assertRaises(ValueError):
                repair_mesh(self.box, 'wrap', {'voxel_size_mm': .00001})

    def test_diagnostic_slivers_and_overlaps_keep_original_face_ids(self):
        source = trimesh.Trimesh([[0, 0, 0], [2, 0, 0], [0, .001, 0],
                                 [0, 0, 1], [1, 0, 1], [0, 1, 1]], [[0, 1, 2], [3, 4, 5]], process=False)
        result, report = repair_mesh(source, 'slivers', {'min_angle_deg': 5})
        np.testing.assert_array_equal(report['selected_faces'], [0])
        self.assertFalse(report['changed'])
        np.testing.assert_array_equal(result.faces, source.faces)
        overlap = trimesh.Trimesh([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                                  [.1, .1, 0], [1.1, .1, 0], [.1, 1.1, 0]], [[0, 1, 2], [3, 4, 5]], process=False)
        result, report = repair_mesh(overlap, 'overlaps', {}, face_ids=[1])
        np.testing.assert_array_equal(report['selected_faces'], [1])
        np.testing.assert_array_equal(report['overlap_faces'], [0, 1])
        self.assertFalse(report['changed'])
        # Ordinary shared edges in a valid closed model aren't intersections.
        _, report = repair_mesh(self.box, 'overlaps', {})
        self.assertEqual(len(report['selected_faces']), 0)

    def test_parameters_selection_and_cancellation_are_validated(self):
        for operation, params in [('smooth', {'relaxation': np.nan}), ('subdivide', {'iterations': 1.5}),
                                  ('decimate', {'target_ratio': 0}), ('wrap', {'voxel_size_mm': -1}),
                                  ('normals', {'flip': 'yes'}), ('stitch', {'tolerance_mm': complex(1, 2)})]:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                repair_mesh(self.box, operation, params)
        for selected in ([], [-1], [len(self.box.faces)], [1.5]):
            with self.assertRaises(ValueError):
                repair_mesh(self.box, 'smooth', {}, face_ids=selected)
        with self.assertRaises(ValueError):
            repair_mesh(self.box, 'subdivide', {}, face_ids=[0])
        with self.assertRaises(InterruptedError):
            repair_mesh(self.box, 'normals', {}, cancelled=lambda: True)
        cancelled = [False]
        def progress(message):
            if message.startswith('Сглаживание: 1/'):
                cancelled[0] = True
        vertices, faces = self.box.vertices.copy(), self.box.faces.copy()
        with self.assertRaises(InterruptedError):
            repair_mesh(self.box, 'smooth', {}, progress=progress, cancelled=lambda: cancelled[0])
        self.assert_source(self.box, vertices, faces)

    def test_empty_nonfinite_mesh_and_growth_budgets(self):
        with self.assertRaises(ValueError):
            repair_mesh(trimesh.Trimesh(), 'normals', {})
        source = self.box.copy()
        source.vertices[0, 0] = np.nan
        with self.assertRaises(ValueError):
            repair_mesh(source, 'smooth', {})
        with patch('repair_operations.MAX_OUTPUT_FACES', 20), patch('trimesh.remesh.subdivide', side_effect=AssertionError('Must reject before allocating')):
            with self.assertRaises(ValueError):
                repair_mesh(self.box, 'subdivide', {'iterations': 1})
        with patch('repair_operations.MAX_QUERY_CANDIDATES', 1):
            with self.assertRaises(ValueError):
                repair_mesh(self.box, 'overlaps', {})


if __name__ == '__main__':
    unittest.main()
