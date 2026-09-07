import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from slicer_tools import transform_matrix, create_primitive
from project_history import ProjectHistory, digest
from project_store import ProjectState


class SlicerToolGeometryTests(unittest.TestCase):
    def test_detailed_absolute_anchors_line_rotation_and_preserve_z(self):
        from transform_math import matrices
        cube = create_primitive('Параллелепипед', [2, 4, 6], [10, 20, 30])
        other = cube.copy(); other.apply_translation([20, 0, 0])
        params = dict(individual=True, absolute=True, target=[0, 0, 0], values=[0, 0, 0],
            anchor_modes=[0, 1, 2], anchor_custom=[0, 0, 0], along_line=False)
        for mesh, matrix in zip([cube, other], matrices([cube, other], 'Перемещать', params)):
            moved = mesh.copy(); moved.apply_transform(matrix)
            np.testing.assert_allclose([moved.bounds[0, 0], moved.bounds[:, 1].mean(), moved.bounds[1, 2]], [0, 0, 0])
        params.update(absolute=False, values=[4, 3, 8], along_line=True, line_a=[0, 0, 0], line_b=[1, 1, 0])
        np.testing.assert_allclose(matrices([cube], 'Перемещать', params)[0][:3, 3], [3.5, 3.5, 0])
        params = dict(individual=False, along_line=True, line_a=[10, 20, 30], line_b=[10, 20, 31], line_angle=90, keep_z=False)
        rotated = cube.copy(); rotated.apply_transform(matrices([cube], 'Вращать', params)[0])
        np.testing.assert_allclose(rotated.extents, [4, 2, 6])
        np.testing.assert_allclose(rotated.bounds.mean(axis=0), [10, 20, 30])
        params = dict(values=[45, 0, 0], along_line=False, center=[0, 0, 0], individual=False, keep_z=True)
        rotated = cube.copy(); rotated.apply_transform(matrices([cube], 'Вращать', params)[0])
        self.assertAlmostEqual(rotated.bounds[0, 2], cube.bounds[0, 2])

    def test_three_point_mirror_and_invalid_line(self):
        from transform_math import matrices, plane_from_points
        origin, normal = plane_from_points([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        cube = create_primitive('Параллелепипед', [1, 1, 1], [3, 0, 0])
        reflected = cube.copy()
        reflected.apply_transform(matrices([cube], 'Отзеркалить', dict(center=origin, normal=normal))[0])
        np.testing.assert_allclose(reflected.bounds.mean(axis=0), [1, -2, 0], atol=1e-8)
        self.assertGreater(reflected.volume, 0)
        with self.assertRaises(ValueError): plane_from_points([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    def test_rotation_about_offset_center_and_mirror_winding(self):
        center = np.array([11, 22, 33])
        matrix = transform_matrix('Вращать', [0, 0, 90], center)
        np.testing.assert_allclose(trimesh.transform_points([center + [2, 0, 0]], matrix), [center + [0, 2, 0]])
        mesh = create_primitive('Параллелепипед', [2, 4, 8], center)
        original_volume = mesh.volume
        mesh.apply_transform(transform_matrix('Отзеркалить', [1, 0, 0], np.zeros(3)))
        self.assertAlmostEqual(mesh.volume, original_volume)
        self.assertTrue(mesh.is_winding_consistent)
        np.testing.assert_allclose(mesh.bounds.mean(axis=0), [-11, 22, 33])

    def test_primitives_dimensions_and_invalid_scale(self):
        for kind in ('Параллелепипед', 'Цилиндр', 'Сфера'):
            mesh = create_primitive(kind, [10, 10, 10], [0, 0, 5])
            self.assertTrue(mesh.is_watertight)
            np.testing.assert_allclose(mesh.extents, [10, 10, 10], atol=.01)
        with self.assertRaises(ValueError):
            transform_matrix('Масштабировать', [0, 1, 1], [0, 0, 0])

    def test_history_shares_unchanged_meshes_and_preserves_saved_key(self):
        history = ProjectHistory(max_steps=3)
        initial = ProjectState(parts=[dict(mesh=trimesh.creation.icosphere(subdivisions=2), filename='sphere.stl')])
        history.reset(initial)
        initial_key = history.key
        for index in range(7):
            changed = deepcopy(initial)
            changed.sections[0]['position'] = index + 1
            history.push(changed)
        self.assertEqual(len(history.entries), 4)
        self.assertEqual(history.saved, initial_key)
        meshes = [entry[1].parts[0]['mesh'] for entry in history.entries]
        self.assertEqual(len(set(map(id, meshes))), 1)
        before = meshes[0].vertices.copy()
        independent = deepcopy(history.entries[-1][1])
        independent.parts[0]['mesh'].apply_translation([5, 0, 0])
        np.testing.assert_array_equal(meshes[0].vertices, before)
        self.assertNotEqual(digest({'a': {'b': 1}, 'c': 2}), digest({'a': {'b': 1, 'c': 2}}))
