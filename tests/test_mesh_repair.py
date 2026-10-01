import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mesh_repair import prepare_repair


class RepairTests(unittest.TestCase):
    def test_later_pass_repairs_small_polygon_boundary(self):
        source = trimesh.creation.cylinder(radius=.02, height=.02, sections=8)
        source.update_faces(source.face_normals[:, 2] < .9)
        first, one = prepare_repair(source, passes=1)
        repaired, report = prepare_repair(source, passes=3)
        self.assertFalse(first.is_watertight)
        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertEqual(report['passes'][1]['holes'], 1)
        self.assertEqual(len(report['passes']), 2)  # early stop when closed

    def test_repair_cancels_between_stages(self):
        stages = []
        with self.assertRaises(InterruptedError):
            prepare_repair(trimesh.creation.box(), progress=stages.append, cancelled=lambda: len(stages) > 0)
        self.assertEqual(len(stages), 1)
    def test_small_hole_repaired_but_scan_boundary_preserved(self):
        source = trimesh.creation.box(extents=[.02]*3)
        source.update_faces(np.arange(11))
        original = source.faces.copy()
        repaired, report = prepare_repair(source)
        self.assertTrue(repaired.is_watertight)
        self.assertTrue(repaired.is_winding_consistent)
        self.assertEqual(report['holes'], 1)
        scan, info = prepare_repair(source, 'Scan')
        self.assertFalse(scan.is_watertight)
        self.assertEqual(info['holes'], 0)
        np.testing.assert_array_equal(source.faces, original)

    def test_large_designed_opening_is_not_filled(self):
        source = trimesh.creation.box()
        source.update_faces(np.arange(10))
        repaired, report = prepare_repair(source)
        self.assertFalse(repaired.is_watertight)
        self.assertEqual(report['holes'], 0)

    def test_clean_model_does_not_prompt_or_reindex(self):
        source = trimesh.creation.box()
        repaired, report = prepare_repair(source)
        self.assertFalse(report['changed'])
        self.assertFalse(report['defects'])
        np.testing.assert_array_equal(source.faces, repaired.faces)
        np.testing.assert_array_equal(source.vertices, repaired.vertices)

    def test_duplicate_faces_and_seams_cleaned(self):
        solid = trimesh.creation.box()
        source = trimesh.Trimesh(vertices=solid.triangles.reshape(-1,3),
                                 faces=np.arange(36).reshape(-1,3), process=False)
        source.faces = np.vstack([source.faces, source.faces[:1], [0,0,1]])
        repaired, report = prepare_repair(source)
        self.assertTrue(report['changed'])
        self.assertTrue(repaired.is_watertight)
        self.assertEqual(len(repaired.faces), 12)
