import unittest
from unittest.mock import patch
import numpy as np
import trimesh
from mesh_diagnostics import diagnose_mesh
from mesh_full_repair import prepare_full_repair, run_engine


class FullRepairTests(unittest.TestCase):
    def test_full_repair_orients_cavity_inward_after_engine(self):
        from repair_manual_geometry import edit_mesh
        outer = trimesh.creation.box(extents=[4, 4, 4])
        inner = trimesh.creation.box(extents=[2, 2, 2])
        inner.invert()
        source = trimesh.util.concatenate([outer, inner])
        source.metadata = {'nested': {'value': 1}}
        source.faces[13] = source.faces[13, ::-1]
        vertices, faces = source.vertices.copy(), source.faces.copy()
        self.assertFalse(diagnose_mesh(source)['clean'])
        with patch('mesh_full_repair.run_engine', return_value=(source.copy(), [{'pass': 1}])) as engine:
            repaired, report = prepare_full_repair(source, tolerance_mm=.001)
        engine.assert_called_once()
        np.testing.assert_array_equal(source.vertices, vertices)
        np.testing.assert_array_equal(source.faces, faces)
        self.assertTrue(report['after']['clean'])
        self.assertTrue(report['acceptable'])
        self.assertEqual(report['cavity_shells'], 1)
        self.assertEqual(report['warnings'], [])
        self.assertAlmostEqual(repaired.volume, 56.)
        cut, _ = edit_mesh(repaired, 'clip', dict(point=[0, 0, 0], normal=[0, 0, 1], cap=True))
        self.assertAlmostEqual(cut.volume, 28.)
        repaired.metadata['nested']['value'] = 2
        self.assertEqual(source.metadata['nested']['value'], 1)
        self.assertEqual(repaired.metadata['repair']['orientation']['cavity_shells'], 1)

    def test_full_repair_preserves_orientation_warnings_and_cancellation(self):
        source = trimesh.creation.box()
        source.update_faces(np.arange(11))
        with patch('mesh_full_repair.run_engine', return_value=(source.copy(), [])):
            repaired, report = prepare_full_repair(source)
        self.assertFalse(report['after']['clean'])
        self.assertTrue(any('Открытые' in message for message in report['warnings']))
        self.assertEqual(repaired.metadata['repair']['orientation']['warnings'], report['warnings'])
        original = source.faces.copy()
        messages = []
        with patch('mesh_full_repair.run_engine', return_value=(source.copy(), [])), self.assertRaises(InterruptedError):
            prepare_full_repair(source, progress=messages.append,
                                cancelled=lambda: any('Согласование ориентации' in message for message in messages))
        np.testing.assert_array_equal(source.faces, original)

    def test_open_box_repaired_and_rechecked(self):
        source = trimesh.creation.box()
        source.update_faces(np.arange(11))
        original = source.faces.copy()
        repaired, report = prepare_full_repair(source, tolerance_mm=1)
        np.testing.assert_array_equal(source.faces, original)
        self.assertTrue(report['after']['clean'])
        self.assertTrue(repaired.is_watertight)
        self.assertGreater(report['shape']['max_mm'], .01)

    def test_already_clean_mesh_unchanged(self):
        source = trimesh.creation.box()
        repaired, report = prepare_full_repair(source)
        self.assertFalse(report['changed'])
        np.testing.assert_array_equal(source.faces, repaired.faces)

    def test_full_diagnostics_catches_inverted_and_overlapping(self):
        mesh = trimesh.creation.box(); mesh.invert()
        self.assertTrue(diagnose_mesh(mesh)['negative_volume'])
        self.assertFalse(diagnose_mesh(mesh)['clean'])
        other = mesh.copy(); other.apply_translation([.2, .2, 0])
        report = diagnose_mesh(mesh+other)
        self.assertGreater(report['overlaps'], 0)
        self.assertGreater(report['intersections'], 0)

    def test_engine_can_be_cancelled_without_changing_input(self):
        source = trimesh.creation.box()
        with self.assertRaises(InterruptedError):
            run_engine(source, 3, lambda _: None, lambda: True)
        self.assertTrue(source.is_watertight)

    def test_quick_diagnostics_never_claims_full_success(self):
        report = diagnose_mesh(trimesh.creation.box(), full=False)
        self.assertIsNone(report['overlaps']); self.assertFalse(report['clean'])

    def test_vertex_only_connection_is_not_manifold(self):
        a=trimesh.creation.box(); b=a.copy(); b.apply_translation([1,1,1])
        joined=a+b; joined.merge_vertices()
        report=diagnose_mesh(joined)
        self.assertEqual(report['nonmanifold'], 0)
        self.assertEqual(report['nonmanifold_vertices'], 1)
        self.assertFalse(report['clean'])

    def test_shape_tolerance_blocks_changed_surface(self):
        source=trimesh.creation.box(); source.update_faces(np.arange(11))
        _, report=prepare_full_repair(source, tolerance_mm=.001)
        self.assertTrue(report['after']['clean'])
        self.assertFalse(report['acceptable'])

    def test_fragments_are_reported(self):
        first = trimesh.creation.box(); second=first.copy(); second.apply_translation([3,0,0])
        report = diagnose_mesh(first+second)
        self.assertEqual(report['components'], 2)
