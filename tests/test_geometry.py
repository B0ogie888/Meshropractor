from pathlib import Path
import sys
import unittest

import numpy as np
import pyvista as pv
import trimesh
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geometry_analysis import sample_surface, compute_heatmap, validate_deformation
from ml_deformation import NativeDeformationService


def as_pv(mesh):
    return pv.PolyData(mesh.vertices, np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)))


class GeometryTests(unittest.TestCase):
    def test_sampling_count_reproducibility_and_normals(self):
        cube = as_pv(trimesh.creation.box())
        a, b = sample_surface(cube, 1000), sample_surface(cube, 1000)
        self.assertEqual(a.n_points, 1000)
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_allclose(np.linalg.norm(a.point_normals, axis=1), 1)
        self.assertEqual(sample_surface(cube, 0).n_points, cube.n_points)

    def test_heatmap_sign_for_expanded_and_shrunken_sphere(self):
        sphere = trimesh.creation.icosphere(subdivisions=2)
        expanded = sphere.copy(); expanded.apply_scale(1.1)
        shrunken = sphere.copy(); shrunken.apply_scale(0.9)
        self.assertTrue((compute_heatmap(sphere, expanded) > 0).all())
        self.assertTrue((compute_heatmap(sphere, shrunken) < 0).all())

    def test_heatmap_open_cad_uses_surface_normal_sign(self):
        mesh = trimesh.Trimesh(vertices=[[0,0,0], [2,0,0], [0,2,0]], faces=[[0,1,2]], process=False)
        scan = trimesh.Trimesh(vertices=[[.2,.2,.1], [.2,.2,-.2], [.3,.2,0]], faces=[[0,1,2]], process=False)
        values, info = compute_heatmap(mesh, scan, return_info=True)
        np.testing.assert_allclose(values, [.1, -.2, 0], atol=1e-6)
        self.assertIn('нормалью', info)

    def test_heatmap_welds_cad_seams_and_accepts_open_scan(self):
        solid = trimesh.creation.box()
        vertices = solid.triangles.reshape(-1, 3).copy()
        faces = np.arange(len(vertices)).reshape(-1, 3)
        cad = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Duplicate export facets must not turn the boundary non-manifold.
        cad.faces = np.vstack([cad.faces, cad.faces[:1]])
        scan = trimesh.Trimesh(vertices=[[.6, 0, 0], [.4, 0, 0], [.5, .1, 0]],
                               faces=[[0, 1, 2]], process=False)
        before_faces = cad.faces.copy()
        self.assertFalse(cad.is_watertight)
        self.assertFalse(scan.is_watertight)
        np.testing.assert_allclose(compute_heatmap(cad, scan), [.1, -.1, 0], atol=1e-6)
        np.testing.assert_array_equal(cad.vertices, vertices)
        np.testing.assert_array_equal(cad.faces, before_faces)

    def test_raycast_measures_known_offset(self):
        source = sample_surface(as_pv(trimesh.creation.box(extents=[10, 10, 10])), 500)
        target = as_pv(trimesh.creation.box(extents=[12, 12, 12]))
        pts, deviations = NativeDeformationService()._compute_deviations_o3d(source, target, 2)
        self.assertEqual(len(pts), 500)
        np.testing.assert_allclose(np.linalg.norm(deviations, axis=1), 1, atol=1e-5)

    def test_rejects_low_coverage_and_honors_cancellation(self):
        service = NativeDeformationService()
        cad = as_pv(trimesh.creation.icosphere(subdivisions=1))
        scan = as_pv(trimesh.creation.icosphere(subdivisions=1, radius=3))
        with self.assertRaises(ValueError):
            service.create_deformed_model(cad, scan, max_dev=0.1, sample_count=100)
        self.assertIsNone(service.create_deformed_model(cad, scan, cancel_callback=lambda: True))

    def test_mesh_validation_detects_collapse(self):
        cad = as_pv(trimesh.creation.box())
        collapsed = cad.copy(); collapsed.points[:, 2] = 0
        with self.assertRaises(ValueError): validate_deformation(cad, collapsed)
        validate_deformation(cad, cad.copy())

    def test_neural_compensation_contracts_expanded_sphere_and_preserves_topology(self):
        service = NativeDeformationService()
        service.device = torch.device('cpu')
        cad = as_pv(trimesh.creation.icosphere(subdivisions=1, radius=10))
        scan = cad.copy(); scan.points *= 1.01
        result = service.create_deformed_model(cad, scan, max_dev=0.5, sample_count=500,
                                              deformation_type=0, epochs=160, factor=[1, 1, 2])
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.faces, cad.faces)
        np.testing.assert_allclose(result['Deformation_Vectors'], result.points - cad.points, atol=1e-6)
        self.assertLess(np.mean(np.einsum('ij,ij->i', result.points - cad.points, cad.points)), 0)
        self.assertLess(service.last_quality['validation_rmse_mm'], 0.05)
        self.assertEqual(service.last_quality['sample_count'], 500)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp_medium_and_hard_networks(self):
        cad = as_pv(trimesh.creation.icosphere(subdivisions=1, radius=10))
        scan = cad.copy(); scan.points *= 1.01
        for kind in (1, 2):
            with self.subTest(network=kind):
                service = NativeDeformationService()
                result = service.create_deformed_model(cad, scan, max_dev=0.5, sample_count=500,
                                                      deformation_type=kind, epochs=50)
                self.assertIsNotNone(result)
                self.assertTrue(np.isfinite(result.points).all())
                self.assertLess(np.mean(np.einsum('ij,ij->i', result.points - cad.points, cad.points)), 0)


if __name__ == "__main__": unittest.main()
