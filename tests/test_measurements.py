import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from measurement_geometry import distance, project_plane, circle, angle, normal_angle


class MeasurementTests(unittest.TestCase):
    def test_circle_in_oblique_plane(self):
        u = np.array([1., 1., 0.]) / np.sqrt(2)
        v = np.array([0., 0., 1.])
        center = np.array([100., -200., 300.])
        points = [center + 7*u, center + 7*v, center - 7*u]
        result, radius, normal = circle(points)
        np.testing.assert_allclose(result, center)
        self.assertAlmostEqual(radius, 7)
        self.assertAlmostEqual(abs(normal @ np.cross(u,v)), 1)

    def test_distance_plane_and_angles(self):
        self.assertEqual(distance([0,0,0], [3,4,0])[0], 5)
        point, value = project_plane([2,3,4], [0,0,1], [0,0,2])
        np.testing.assert_allclose(point, [2,3,1])
        self.assertEqual(value, 3)
        self.assertEqual(angle([[1,0,0], [0,0,0], [0,1,0]]), 90)
        self.assertEqual(normal_angle([0,0,1], [0,0,-1]), 0)

    def test_degenerate_points_fail(self):
        for points in ([[0,0,0], [1,0,0], [2,0,0]], [[0,0,0]]*3):
            with self.assertRaises(ValueError): circle(points)
        with self.assertRaises(ValueError): angle([[0,0,0]]*3)
