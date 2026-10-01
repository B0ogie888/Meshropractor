import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from placement_geometry import (orientation_candidates, optimize_orientation, minimum_oriented_bounds,
                                fit_platform, pack_meshes)


def box(size, center=(0, 0, 0)):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


def moved_bounds(mesh, matrix):
    result = mesh.copy()
    result.apply_transform(matrix)
    return result.bounds


class PlacementGeometryTests(unittest.TestCase):
    def assert_rigid(self, matrix):
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1], atol=1e-10)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.)

    def assert_separated(self, a, b, gap=0.):
        self.assertTrue(np.any(a[1] + gap <= b[0] + 1e-7) or np.any(b[1] + gap <= a[0] + 1e-7),
                        f'AABBs overlap with required gap {gap}: {a}, {b}')

    def assert_layout(self, meshes, matrices, report, gap=0., obstacles=()):
        bounds = []
        for index, (mesh, matrix) in enumerate(zip(meshes, matrices)):
            self.assert_rigid(matrix)
            actual = moved_bounds(mesh, matrix)
            np.testing.assert_allclose(actual, report['bounds'][index], atol=1e-7)
            self.assertTrue(np.all(actual[0] >= report['platform_bounds'][0] - 1e-7))
            self.assertTrue(np.all(actual[1] <= report['platform_bounds'][1] + 1e-7))
            for other in bounds:
                self.assert_separated(actual, other, gap)
            for obstacle in obstacles:
                self.assert_separated(actual, obstacle.bounds, gap)
            bounds.append(actual)
        return bounds

    def test_oriented_bounds_recovers_rotated_box_without_scaling(self):
        source = box([2, 4, 10])
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [17, 32, 21], degrees=True).as_matrix()
        transform[:3, 3] = [1234, -50, 37]
        source.apply_transform(transform)
        before = source.vertices.copy()
        matrix, report = minimum_oriented_bounds(source)
        self.assert_rigid(matrix)
        bounds = moved_bounds(source, matrix)
        np.testing.assert_allclose(np.sort(bounds[1] - bounds[0]), [2, 4, 10], atol=1e-7)
        np.testing.assert_allclose(bounds.mean(axis=0), source.bounds.mean(axis=0), atol=1e-7)
        np.testing.assert_array_equal(source.vertices, before)
        self.assertAlmostEqual(report['bbox_volume_mm3'], 80., places=6)
        self.assertTrue(report['warnings'])

    def test_orientation_objectives_have_metrics_and_different_best_height(self):
        source = box([2, 4, 10])
        height_matrix, height = optimize_orientation(source, objective='height')
        footprint_matrix, footprint = optimize_orientation(source, objective='footprint')
        self.assertAlmostEqual(height['height_mm'], 2.)
        self.assertAlmostEqual(footprint['footprint_mm2'], 8.)
        self.assertAlmostEqual(footprint['height_mm'], 10.)
        for matrix in (height_matrix, footprint_matrix): self.assert_rigid(matrix)
        candidates = orientation_candidates(source, objective='support', max_candidates=8)
        self.assertLessEqual(len(candidates), 8)
        self.assertEqual([item['score'] for item in candidates], sorted(item['score'] for item in candidates))
        for item in candidates:
            for key in ('height_mm', 'footprint_mm2', 'overhang_area_mm2', 'bbox_volume_mm3', 'label'):
                self.assertIn(key, item)
            self.assertGreaterEqual(item['overhang_area_mm2'], 0.)

    def test_fit_group_keeps_relative_positions_and_clearance(self):
        meshes = [box([2, 3, 4], [100, 20, 7]), box([2, 3, 4], [105, 20, 7])]
        source = [mesh.vertices.copy() for mesh in meshes]
        matrices, report = fit_platform(meshes, {'dim': [20, 20, 20]}, margin_mm=2, clearance_mm=3)
        np.testing.assert_allclose(matrices[0], matrices[1])
        bounds = self.assert_layout(meshes, matrices, report)
        self.assertAlmostEqual(bounds[1].mean(axis=0)[0] - bounds[0].mean(axis=0)[0], 5.)
        self.assertAlmostEqual(min(bound[0, 2] for bound in bounds), 3.)
        for mesh, vertices in zip(meshes, source): np.testing.assert_array_equal(mesh.vertices, vertices)

    def test_fit_can_turn_a_rotated_long_part_upright_without_scaling(self):
        source = box([12, 2, 2])
        source.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1]))
        platform = {'dim': [4, 4, 14]}
        with self.assertRaisesRegex(ValueError, 'Не удалось найти размещение'):
            fit_platform([source], platform, margin_mm=.5, allow_rotation=False)
        matrices, report = fit_platform([source], platform, margin_mm=.5, allow_rotation=True)
        self.assert_layout([source], matrices, report)
        self.assertAlmostEqual(report['height_mm'], 12.)

    def test_fit_avoids_obstacles_while_preferring_center(self):
        source = box([4, 4, 4])
        obstacle = box([6, 6, 6], [0, 0, 3])
        matrices, report = fit_platform([source], {'dim': [20, 20, 20]}, obstacles=[obstacle])
        self.assert_layout([source], matrices, report, obstacles=[obstacle])
        self.assertAlmostEqual(report['bounds'][0, 0, 2], 0.)

    def test_pack_2d_four_boxes_with_gap_and_edge_margin(self):
        meshes = [box([8, 8, 8], [100 + 20 * i, 40, -30]) for i in range(4)]
        matrices, report = pack_meshes(meshes, {'dim': [20, 20, 20]}, gap_mm=2, margin_mm=1)
        bounds = self.assert_layout(meshes, matrices, report, gap=2)
        self.assertTrue(all(abs(item[0, 2]) < 1e-7 for item in bounds))

    def test_pack_returns_matrices_in_original_order_after_size_sorting(self):
        meshes = [box([3, 3, 3]), box([8, 6, 4]), box([2, 2, 7])]
        matrices, report = pack_meshes(meshes, {'dim': [30, 20, 20]}, gap_mm=2, margin_mm=1)
        self.assert_layout(meshes, matrices, report, gap=2)
        np.testing.assert_allclose(np.sort(report['bounds'][0, 1] - report['bounds'][0, 0]), [3, 3, 3])
        np.testing.assert_allclose(np.sort(report['bounds'][1, 1] - report['bounds'][1, 0]), [4, 6, 8])

    def test_pack_rotation_checkbox_is_respected(self):
        source = box([12, 7, 2])
        platform = {'dim': [10, 14, 5]}
        with self.assertRaisesRegex(ValueError, 'Не удалось найти размещение'):
            pack_meshes([source], platform, margin_mm=1, allow_rotation=False)
        matrices, report = pack_meshes([source], platform, margin_mm=1, allow_rotation=True)
        self.assert_layout([source], matrices, report)
        self.assertAlmostEqual(report['bounds'][0, 1, 2] - report['bounds'][0, 0, 2], 2.)

    def test_pack_3d_stacks_boxes_only_when_requested(self):
        meshes = [box([8, 8, 8]) for _ in range(3)]
        platform = {'dim': [10, 10, 28]}
        with self.assertRaisesRegex(ValueError, 'Не удалось найти размещение'):
            pack_meshes(meshes, platform, dimensions=2, gap_mm=2, margin_mm=1)
        matrices, report = pack_meshes(meshes, platform, dimensions=3, gap_mm=2, margin_mm=1)
        bounds = self.assert_layout(meshes, matrices, report, gap=2)
        self.assertEqual(sorted(round(item[0, 2]) for item in bounds), [0, 10, 20])
        self.assertTrue(any('над платформой' in text for text in report['warnings']))

    def test_pack_avoids_unselected_obstacles_and_conservative_cylinder_zone(self):
        obstacle = box([8, 8, 8], [-6, -6, 4])
        platform = {'dim': [24, 24, 20], 'use_zones': True,
                    'zones': [{'x': 6, 'y': 6, 'r': 3, 'shape': 0, 'full_h': True}]}
        meshes = [box([5, 5, 4]) for _ in range(3)]
        matrices, report = pack_meshes(meshes, platform, gap_mm=1, margin_mm=1, obstacles=[obstacle])
        bounds = self.assert_layout(meshes, matrices, report, gap=1, obstacles=[obstacle])
        zone = np.array([[3, 3, 0], [9, 9, 20]])
        for placed in bounds: self.assert_separated(placed, zone, 1)

    def test_zone_vertical_interval_and_disabled_zones_are_respected(self):
        source = box([4, 4, 4])
        zone = dict(x=0, y=0, r=100, zmin=10, zmax=20, shape=1)
        platform = dict(dim=[10, 10, 20], use_zones=True, zones=[zone])
        matrices, report = pack_meshes([source], platform, margin_mm=1)
        self.assert_layout([source], matrices, report)
        platform['zones'][0]['full_h'] = True
        with self.assertRaisesRegex(ValueError, 'Не удалось найти размещение'):
            pack_meshes([source], platform, margin_mm=1)
        platform['use_zones'] = False
        pack_meshes([source], platform, margin_mm=1)

    def test_3d_can_place_above_height_limited_zone_with_gap(self):
        source = box([4, 4, 4])
        platform = dict(dim=[10, 10, 12], use_zones=True,
                        zones=[dict(x=0, y=0, r=100, zmin=0, zmax=4, shape=1)])
        matrices, report = pack_meshes([source], platform, dimensions=3, gap_mm=1, margin_mm=1)
        self.assert_layout([source], matrices, report)
        self.assertAlmostEqual(report['bounds'][0, 0, 2], 5.)

    def test_combined_support_envelope_is_not_ignored(self):
        part, support = box([4, 4, 4], [0, 0, 6]), box([10, 10, 2], [0, 0, 1])
        source = trimesh.util.concatenate([part, support])
        with self.assertRaisesRegex(ValueError, 'Не удалось найти размещение'):
            pack_meshes([source], {'dim': [8, 8, 20]}, margin_mm=0)
        matrices, report = pack_meshes([source], {'dim': [14, 14, 20]}, margin_mm=1)
        self.assert_layout([source], matrices, report)

    def test_source_vertices_faces_metadata_and_platform_are_unchanged(self):
        source = box([3, 4, 5], [50, -20, 13])
        source.metadata = {'nested': {'values': [1, 2, 3]}}
        vertices, faces = source.vertices.copy(), source.faces.copy()
        platform = {'dim': [20, 20, 20], 'use_zones': False, 'zones': []}
        obstacle = box([2, 2, 2], [-7, -7, 1])
        obstacle_vertices = obstacle.vertices.copy()
        pack_meshes([source], platform, margin_mm=1, obstacles=[obstacle])
        np.testing.assert_array_equal(source.vertices, vertices)
        np.testing.assert_array_equal(source.faces, faces)
        np.testing.assert_array_equal(obstacle.vertices, obstacle_vertices)
        self.assertEqual(source.metadata, {'nested': {'values': [1, 2, 3]}})
        self.assertEqual(platform, {'dim': [20, 20, 20], 'use_zones': False, 'zones': []})

    def test_varied_3d_parts_remain_separated_from_each_other_and_obstacles(self):
        random = np.random.default_rng(2026)
        meshes = [box(random.uniform(2, 7, 3), random.uniform(-100, 100, 3)) for _ in range(25)]
        obstacles = [box([10, 10, 4], [-9, -9, 2]), box([4, 4, 10], [8, 8, 8])]
        matrices, report = pack_meshes(meshes, {'dim': [40, 40, 35]}, dimensions=3,
                                       gap_mm=1.5, margin_mm=2, clearance_mm=1, obstacles=obstacles)
        self.assert_layout(meshes, matrices, report, gap=1.5, obstacles=obstacles)
        self.assertIn('overhang_area_mm2', report)

    def test_cancellation_invalid_inputs_and_infeasible_keep_sources(self):
        source = box([2, 2, 2])
        before = source.vertices.copy()
        for operation in (lambda: pack_meshes([source], {'dim': [20, 20, 20]}, cancelled=lambda: True),
                          lambda: optimize_orientation(source, cancelled=lambda: True),
                          lambda: minimum_oriented_bounds(source, cancelled=lambda: True),
                          lambda: fit_platform([source], {'dim': [20, 20, 20]}, cancelled=lambda: True)):
            with self.assertRaises(InterruptedError): operation()
        messages = []
        with self.assertRaises(InterruptedError):
            pack_meshes([source], {'dim': [20, 20, 20]}, progress=messages.append,
                        cancelled=lambda: any('Поиск размещения' in m for m in messages))
        for kwargs in ({'gap_mm': -1}, {'margin_mm': float('nan')}, {'dimensions': 4},
                       {'allow_rotation': 'yes'}, {'clearance_mm': 30}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pack_meshes([source], {'dim': [20, 20, 20]}, **kwargs)
        with self.assertRaises(ValueError): pack_meshes([], {'dim': [20, 20, 20]})
        with self.assertRaises(ValueError): pack_meshes([source], {'dim': [0, 20, 20]})
        invalid = source.copy(); invalid.vertices[0, 0] = np.nan
        with self.assertRaises(ValueError): pack_meshes([invalid], {'dim': [20, 20, 20]})
        with patch('placement_geometry.MAX_SEARCH_STEPS', 0), self.assertRaisesRegex(ValueError, 'число попыток'):
            pack_meshes([source], {'dim': [20, 20, 20]})
        with self.assertRaisesRegex(ValueError, 'Не удалось найти размещение'):
            pack_meshes([source], {'dim': [1, 1, 1]}, margin_mm=0)
        np.testing.assert_array_equal(source.vertices, before)


if __name__ == '__main__':
    unittest.main()
