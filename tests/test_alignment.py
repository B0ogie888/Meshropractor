from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alignment import align_surfaces


class AlignmentTests(unittest.TestCase):
    def test_dense_scan_aligns_to_sparse_cad_triangle_interiors(self):
        cad = trimesh.creation.box(extents=[17, 11, 7])
        scan = cad.subdivide().subdivide().subdivide()
        rotation = Rotation.from_euler("xyz", [5, -3, 4], degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [1, -.5, .2]
        scan.apply_transform(transform)
        original = scan.vertices.copy()
        aligned, quality = align_surfaces(cad, scan, settings=dict(search_time=0, samples=6000))
        self.assertLess(quality["rmse"], 1e-4)
        self.assertGreater(quality["coverage"], .999)
        np.testing.assert_allclose(aligned.vertices, cad.subdivide().subdivide().subdivide().vertices, atol=1e-4)
        np.testing.assert_array_equal(scan.vertices, original)

    def test_large_rotation_asymmetric_model(self):
        cad = trimesh.util.concatenate([trimesh.creation.box(extents=[18, 11, 7]),
            trimesh.creation.box(extents=[4, 5, 9], transform=trimesh.transformations.translation_matrix([5, 2, 6]))])
        scan = cad.subdivide()
        transform = trimesh.transformations.euler_matrix(1.1, -.7, 2.0)
        transform[:3, 3] = [45, -17, 80]
        scan.apply_transform(transform)
        aligned, quality = align_surfaces(cad, scan, settings=dict(search_time=1, samples=7000))
        self.assertGreater(quality["coverage"], .995)
        np.testing.assert_allclose(aligned.vertices, cad.subdivide().vertices, atol=.002)

    def test_markers_partial_scan_outliers_and_large_coordinates(self):
        cad = trimesh.creation.box(extents=[17, 11, 7])
        scan = cad.submesh([np.arange(8)], append=True).subdivide().subdivide()
        original = scan.vertices.copy()
        scan = trimesh.util.concatenate([scan, trimesh.creation.box(extents=[.5, .5, .5], transform=trimesh.transformations.translation_matrix([30, 30, 30]))])
        transform = trimesh.transformations.euler_matrix(.1, -.15, .05)
        transform[:3, 3] = [4, -2, 1]
        markers = np.array([[-8.5, -5.5, -3.5], [8.5, -5.5, -3.5], [8.5, 5.5, 3.5]])
        scan_markers = trimesh.transform_points(markers, transform)
        scan.apply_transform(transform)
        offset = np.array([1e6, -2e6, 3e6])
        cad.apply_translation(offset)
        scan.apply_translation(offset)
        aligned, quality = align_surfaces(cad, scan, markers + offset, scan_markers + offset,
            dict(search_time=0, samples=6000, min_fitness=.9))
        self.assertGreater(quality["coverage"], .99)
        np.testing.assert_allclose(aligned.vertices[:len(original)], original + offset, atol=.001)

    def test_mismatch_rejected_and_cancelled(self):
        cad = trimesh.creation.box(extents=[10, 7, 4])
        scan = trimesh.creation.icosphere(radius=20)
        with self.assertRaisesRegex(ValueError, "не принято"):
            align_surfaces(cad, scan, settings=dict(search_time=0, samples=1000, min_fitness=.9))
        with self.assertRaises(InterruptedError):
            align_surfaces(cad, cad.copy(), cancelled=lambda: True)

    def test_markers_without_icp_report_actual_surface_error(self):
        cad = trimesh.creation.box()
        scan = cad.copy()
        scan.apply_translation([4, 2, 1])
        markers = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, .5]])
        aligned, quality = align_surfaces(cad, scan, markers, markers + [4, 2, 1], dict(do_icp=False, samples=1000))
        self.assertLess(quality["rmse"], 1e-6)
        np.testing.assert_allclose(aligned.vertices, cad.vertices, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
