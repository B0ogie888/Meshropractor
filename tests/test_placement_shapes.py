from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from placement_shapes import transfer_orientations


class PlacementShapeTests(unittest.TestCase):
    @staticmethod
    def asymmetric():
        return trimesh.util.concatenate([
            trimesh.creation.box(extents=[18, 11, 7]),
            trimesh.creation.box(extents=[4, 5, 9],
                transform=trimesh.transformations.translation_matrix([5, 2, 6]))])

    def assert_preserved(self, meshes, snapshots):
        for mesh, (vertices, faces, metadata) in zip(meshes, snapshots):
            np.testing.assert_array_equal(mesh.vertices, vertices)
            np.testing.assert_array_equal(mesh.faces, faces)
            np.testing.assert_equal(mesh.metadata, metadata)

    @staticmethod
    def snapshots(meshes):
        return [(mesh.vertices.copy(), mesh.faces.copy(), deepcopy(mesh.metadata)) for mesh in meshes]

    def test_identical_translated_parts_and_reference_stay_in_place(self):
        first = self.asymmetric()
        second = first.copy()
        second.apply_translation([120, -5, 20])
        matrices, report = transfer_orientations([first, second], tolerance_mm=.001)
        for matrix in matrices:
            np.testing.assert_allclose(matrix, np.eye(4), atol=1e-12)
        self.assertTrue(all(match['matched'] for match in report['matches']))
        self.assertEqual(report['method'], 'sampled_bidirectional_surface')

    def test_asymmetric_rotation_transfers_reference_orientation_and_preserves_bbox_center(self):
        reference = self.asymmetric()
        reference.metadata = {'nested': {'name': 'sample'}}
        reference.apply_transform(trimesh.transformations.euler_matrix(.15, -.3, .4))
        source = reference.copy()
        transform = trimesh.transformations.euler_matrix(1.1, -.7, 2.)
        transform[:3, 3] = [45, -17, 80]
        source.apply_transform(transform)
        models = [reference, source]
        before = self.snapshots(models)
        old_center = source.bounds.mean(axis=0)
        matrices, report = transfer_orientations(models, tolerance_mm=.002)
        self.assertTrue(report['matches'][1]['matched'])
        np.testing.assert_allclose(matrices[1][:3, :3], transform[:3, :3].T, atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(matrices[1][:3, :3]), 1.)
        result = source.copy()
        result.apply_transform(matrices[1])
        np.testing.assert_allclose(result.bounds.mean(axis=0), old_center, atol=1e-10)
        np.testing.assert_allclose(result.vertices - old_center,
                                   reference.vertices - reference.bounds.mean(axis=0), atol=1e-9)
        self.assert_preserved(models, before)

    def test_different_tessellation_uses_deterministic_triangle_registration(self):
        reference = self.asymmetric()
        source = reference.subdivide()
        transform = trimesh.transformations.euler_matrix(.8, -.9, 1.7)
        transform[:3, 3] = [31, -22, 17]
        source.apply_transform(transform)
        original_center = source.bounds.mean(axis=0)
        matrices, report = transfer_orientations([source, reference], reference_index=1, tolerance_mm=.005)
        self.assertTrue(report['matches'][0]['matched'], report)
        result = source.copy()
        result.apply_transform(matrices[0])
        np.testing.assert_allclose(result.vertices - original_center,
            reference.subdivide().vertices - reference.bounds.mean(axis=0), atol=.003)
        np.testing.assert_array_equal(matrices[1], np.eye(4))
        repeated, _ = transfer_orientations([source, reference], reference_index=1, tolerance_mm=.005)
        np.testing.assert_allclose(matrices, repeated, atol=1e-12)

    def test_different_size_is_not_scaled_or_moved(self):
        reference = trimesh.creation.box(extents=[10, 7, 4])
        source = reference.copy()
        source.apply_scale(1.3)
        source.apply_translation([20, 30, 40])
        matrices, report = transfer_orientations([reference, source], tolerance_mm=.01)
        np.testing.assert_array_equal(matrices[1], np.eye(4))
        self.assertFalse(report['matches'][1]['matched'])
        self.assertTrue(report['warnings'])

    def test_partial_surface_is_rejected_by_reverse_check(self):
        reference = trimesh.creation.box()
        source = reference.copy()
        source.update_faces(np.arange(len(source.faces)) != 0)
        matrices, report = transfer_orientations([reference, source], tolerance_mm=.02)
        match = report['matches'][1]
        self.assertFalse(match['matched'])
        self.assertIn('reverse', match)
        self.assertLess(match['forward']['max_mm'], 1e-5)
        self.assertGreater(match['reverse']['max_mm'], .02)
        np.testing.assert_array_equal(matrices[1], np.eye(4))

    def test_same_area_different_shape_rejected(self):
        reference = trimesh.creation.box()
        source = trimesh.creation.icosphere(subdivisions=0)
        source.apply_scale(np.sqrt(reference.area / source.area))
        matrices, report = transfer_orientations([reference, source], tolerance_mm=.01)
        self.assertAlmostEqual(source.area, reference.area)
        self.assertFalse(report['matches'][1]['matched'])
        self.assertIn('reverse', report['matches'][1])
        np.testing.assert_array_equal(matrices[1], np.eye(4))

    def test_large_coordinates_are_centered_for_distance_queries(self):
        reference = self.asymmetric()
        source = reference.copy()
        source.apply_transform(trimesh.transformations.euler_matrix(.4, -.7, 1.9))
        reference.apply_translation([1e8, -1e8, 1e8])
        source.apply_translation([-1e8, 1e8, 1e8])
        center = source.bounds.mean(axis=0)
        matrices, report = transfer_orientations([reference, source], tolerance_mm=.002)
        self.assertTrue(report['matches'][1]['matched'])
        result = source.copy()
        result.apply_transform(matrices[1])
        np.testing.assert_allclose(result.bounds.mean(axis=0), center, atol=1e-6)

    def test_cancellation_leaves_meshes_and_metadata_unchanged(self):
        reference, source = self.asymmetric(), self.asymmetric()
        source.apply_transform(trimesh.transformations.euler_matrix(.6, .8, -.2))
        models = [reference, source]
        before = self.snapshots(models)
        with self.assertRaises(InterruptedError):
            transfer_orientations(models, cancelled=lambda: True)
        messages = []
        with self.assertRaises(InterruptedError):
            transfer_orientations(models, progress=messages.append, cancelled=lambda: bool(messages))
        self.assert_preserved(models, before)

    def test_unreferenced_vertices_do_not_change_surface_match_or_orientation(self):
        box = trimesh.creation.box(extents=[2, 3, 4])
        loose = trimesh.Trimesh(np.vstack([box.vertices, [100, 100, 100]]), box.faces, process=False)
        snapshots = self.snapshots([box, loose])
        matrices, report = transfer_orientations([box, loose], tolerance_mm=.001)
        self.assertTrue(report['matches'][1]['matched'], report)
        np.testing.assert_allclose(matrices[1], np.eye(4), atol=1e-12)
        self.assert_preserved([box, loose], snapshots)

        reference = self.asymmetric()
        source = reference.copy()
        transform = trimesh.transformations.euler_matrix(.8, -.9, 1.3)
        transform[:3, 3] = [17, -20, 6]
        source.apply_transform(transform)
        center = source.bounds.mean(axis=0)
        # Loose vertices precede used ones in one model and follow them in the
        # other, exercising exact face correspondence after private compaction.
        reference = trimesh.Trimesh(np.vstack([[200, 300, 400], reference.vertices]), reference.faces + 1, process=False)
        source = trimesh.Trimesh(np.vstack([source.vertices, [-200, -300, -400]]), source.faces, process=False)
        reference.metadata = {'nested': {'label': 'reference'}}
        source.metadata = {'nested': {'label': 'source'}}
        snapshots = self.snapshots([reference, source])
        matrices, report = transfer_orientations([reference, source], tolerance_mm=.002)
        self.assertTrue(report['matches'][1]['matched'], report)
        np.testing.assert_allclose(matrices[1][:3, :3], transform[:3, :3].T, atol=1e-9)
        result = source.copy(); result.apply_transform(matrices[1])
        np.testing.assert_allclose(result.bounds.mean(axis=0), center, atol=1e-9)
        self.assert_preserved([reference, source], snapshots)

    def test_total_geometry_budget_rejected_before_copies_or_expensive_validation(self):
        mesh = self.asymmetric()
        size = mesh.vertices.nbytes + mesh.faces.nbytes
        before = self.snapshots([mesh])
        with patch('placement_shapes.MAX_INPUT_GEOMETRY_BYTES', 2 * size - 1):
            with patch('placement_shapes.validate_mesh', side_effect=AssertionError('Validation allocated geometry')):
                with patch.object(trimesh.Trimesh, 'copy', side_effect=AssertionError('Copied geometry')):
                    with self.assertRaisesRegex(ValueError, '512 МиБ'):
                        transfer_orientations([mesh, mesh])
        self.assert_preserved([mesh], before)

    def test_invalid_input_and_part_limit(self):
        mesh = self.asymmetric()
        for kwargs in ({'tolerance_mm': 0}, {'tolerance_mm': np.nan}, {'tolerance_mm': True},
                       {'reference_index': -1}, {'reference_index': 2}, {'reference_index': 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                transfer_orientations([mesh, mesh], **kwargs)
        with self.assertRaisesRegex(ValueError, '30'):
            transfer_orientations([mesh] * 31)
        with self.assertRaises(ValueError):
            transfer_orientations([])


if __name__ == '__main__':
    unittest.main()
