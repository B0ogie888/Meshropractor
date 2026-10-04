"""Ray constraints and rotation about arbitrary axes, independent of camera UI."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transform_gizmo import ray_plane, axis_drag_normal, signed_angle, rotate_about


class HandleMathTests(unittest.TestCase):
    def test_plane_drag_stays_on_plane_and_parallel_ray_has_no_solution(self):
        origin = np.array([1., 2., 3.])
        for normal in np.eye(3):
            point = ray_plane(np.array([4., 5., 9.]), np.array([-.2, -.3, -1.]), origin, normal)
            self.assertAlmostEqual(np.dot(point - origin, normal), 0)
        self.assertIsNone(ray_plane(origin, np.array([1., 0, 0]), origin, np.array([0., 0, 1.])))

    def test_axis_drag_plane_contains_axis_and_faces_camera(self):
        normal = axis_drag_normal(np.array([1., 0, 0]), np.array([1., 2., 3.]) / np.sqrt(14))
        self.assertAlmostEqual(normal[0], 0)
        self.assertAlmostEqual(np.linalg.norm(normal), 1)
        self.assertIsNone(axis_drag_normal(np.array([1., 0, 0]), np.array([1., 0, 0])))

    def test_screen_rotation_keeps_center_and_uses_view_axis(self):
        center = np.array([17., -11., 9.])
        axis = np.array([1., 2., 3.]) / np.sqrt(14)
        matrix = rotate_about(center, axis, np.pi / 3)
        np.testing.assert_allclose(matrix @ np.r_[center, 1], np.r_[center, 1])
        np.testing.assert_allclose(matrix[:3, :3] @ axis, axis)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(signed_angle([1, 0, 0], [0, 1, 0], [0, 0, 1]), np.pi / 2)
        self.assertAlmostEqual(signed_angle([1, 0, 0], [0, -1, 0], [0, 0, 1]), -np.pi / 2)
