from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cad_state import (CAD_KEY, mesh_digest, cad_status, require_native, attach_native,
                       apply_cad_transform, strip_native, cad_face_triangles)
from project_history import ProjectHistory
from project_store import ProjectState, save_project, load_project


class CADStateTests(unittest.TestCase):
    def payload(self, mesh):
        # This layer verifies provenance, not the BRep format; no OCCT is needed.
        count = len(mesh.faces)
        return dict(version=1, brep=np.frombuffer(b'BRep fixture; parsing belongs to cad_geometry', dtype=np.uint8),
                    matrix=np.eye(4), face_ids=np.arange(count, dtype=np.int64) // 2,
                    body_ids=np.zeros(count, dtype=np.int64), bodies=[{'name': 'Body'}],
                    face_info=[{'body_id': 0, 'type': 'plane'} for _ in range((count + 1) // 2)])

    def native(self):
        mesh = trimesh.creation.box(extents=[2, 3, 4])
        return attach_native(mesh, self.payload(mesh))

    def test_attach_copies_payload_and_annotations_do_not_change_validity(self):
        mesh = trimesh.creation.box()
        payload = self.payload(mesh)
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        self.assertIs(attach_native(mesh, payload), mesh)
        self.assertEqual(cad_status(mesh), 'native')
        payload['matrix'][0, 3] = 20
        payload['face_ids'][0] = 3
        payload['bodies'][0]['name'] = 'Changed externally'
        mesh.metadata['alignment'] = {'rmse': .01}
        mesh.metadata['note'] = 'User annotation'
        self.assertEqual(cad_status(mesh), 'native')
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.faces, faces)
        self.assertEqual(require_native(mesh)['proxy_digest'], mesh_digest(mesh))

    def test_copy_and_strip_do_not_change_original(self):
        mesh = self.native()
        copy = mesh.copy()
        self.assertEqual(cad_status(copy), 'native')
        copy.metadata[CAD_KEY]['brep'][0] ^= 1
        self.assertEqual(cad_status(copy), 'modified')
        self.assertEqual(cad_status(mesh), 'native')
        stripped = strip_native(mesh)
        self.assertEqual(cad_status(stripped), 'mesh')
        stripped.vertices[0] += 1
        self.assertEqual(cad_status(mesh), 'native')
        with self.assertRaisesRegex(ValueError, 'нет исходной CAD'):
            require_native(stripped)

    def test_affine_composition_reflection_and_anisotropic_scaling(self):
        mesh = self.native()
        ordinary = mesh.copy()
        original_face_ids = require_native(mesh)['face_ids'].copy()
        transforms = [trimesh.transformations.translation_matrix([4, -2, 6]),
                      trimesh.transformations.euler_matrix(.3, -.7, .1),
                      np.diag([1.2, .7, 2., 1.]), np.diag([-1., 1., 1., 1.])]
        combined = np.eye(4)
        for transform in transforms:
            self.assertIs(apply_cad_transform(mesh, transform), mesh)
            ordinary.apply_transform(transform)
            combined = transform @ combined
            self.assertEqual(cad_status(mesh), 'native')
            np.testing.assert_allclose(require_native(mesh)['matrix'], combined)
            np.testing.assert_allclose(mesh.vertices, ordinary.vertices)
            np.testing.assert_array_equal(mesh.faces, ordinary.faces)
            np.testing.assert_array_equal(require_native(mesh)['face_ids'], original_face_ids)

    def test_edited_vertices_and_faces_invalidate_and_transforms_cannot_revive(self):
        mesh = self.native()
        mesh.vertices += [3, 0, 0]
        self.assertEqual(cad_status(mesh), 'modified')
        apply_cad_transform(mesh, trimesh.transformations.translation_matrix([-3, 0, 0]))
        # Coordinates match the original again, but the invalid binding stays invalid.
        self.assertEqual(require_native(self.native())['proxy_digest'], mesh_digest(mesh))
        self.assertEqual(cad_status(mesh), 'modified')
        with self.assertRaisesRegex(ValueError, 'утрачена'):
            require_native(mesh)
        mesh = self.native()
        mesh.faces[0] = mesh.faces[0, ::-1]
        self.assertEqual(cad_status(mesh), 'modified')
        apply_cad_transform(mesh, np.eye(4))
        self.assertEqual(cad_status(mesh), 'modified')
        mesh = self.native()
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
        self.assertEqual(cad_status(mesh), 'modified')

    def test_semantically_valid_payload_tampering_is_detected(self):
        changes = [lambda payload: payload['matrix'].__setitem__((0, 3), 2.),
                   lambda payload: payload['face_ids'].__setitem__(0, 2),
                   lambda payload: payload['brep'].__setitem__(0, payload['brep'][0] ^ 1),
                   lambda payload: payload['face_info'][0].update(type='cylinder')]
        for change in changes:
            with self.subTest(change=change):
                mesh = self.native()
                change(mesh.metadata[CAD_KEY])
                self.assertEqual(cad_status(mesh), 'modified')
                with self.assertRaisesRegex(ValueError, 'контрольная сумма'):
                    require_native(mesh)

    def test_invalid_schema_and_mappings_are_rejected_without_replacing_payload(self):
        changes = [lambda p: p.update(version=2), lambda p: p.update(version=True),
                   lambda p: p.update(brep=np.array([], np.uint8)),
                   lambda p: p.update(face_ids=p['face_ids'].astype(np.int32)),
                   lambda p: p.update(body_ids=p['body_ids'][:-1]),
                   lambda p: p['face_ids'].__setitem__(0, len(p['face_info'])),
                   lambda p: p['body_ids'].__setitem__(0, 1),
                   lambda p: p['face_info'][0].update(body_id=-1),
                   lambda p: p.update(face_info=[])]
        for change in changes:
            with self.subTest(change=change):
                mesh = self.native()
                payload = deepcopy(mesh.metadata[CAD_KEY])
                change(payload)
                with self.assertRaises(ValueError):
                    attach_native(mesh, payload)
                self.assertEqual(cad_status(mesh), 'native')
                mesh.metadata[CAD_KEY] = payload
                self.assertEqual(cad_status(mesh), 'modified')

    def test_open_surface_body_sentinel_and_cad_face_selection(self):
        mesh = trimesh.creation.box()
        payload = self.payload(mesh)
        payload['bodies'] = []
        payload['body_ids'][:] = -1
        for item in payload['face_info']: item['body_id'] = -1
        attach_native(mesh, payload)
        np.testing.assert_array_equal(cad_face_triangles(mesh, 3), [2, 3])
        for triangle in (-1, len(mesh.faces), 1.5, True):
            with self.subTest(triangle=triangle), self.assertRaises(ValueError):
                cad_face_triangles(mesh, triangle)

    def test_invalid_affine_is_atomic_and_ordinary_mesh_is_supported(self):
        matrices = [np.eye(3), np.full((4, 4), np.nan), np.diag([0., 1., 1., 1.])]
        projective = np.eye(4); projective[3, 0] = .01; matrices.append(projective)
        overflow = np.diag([1e308, 1e308, 1e308, 1.]); matrices.append(overflow)
        mesh = self.native()
        before = mesh_digest(mesh)
        for matrix in matrices:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                apply_cad_transform(mesh, matrix)
            self.assertEqual(mesh_digest(mesh), before)
            self.assertEqual(cad_status(mesh), 'native')
        plain = trimesh.creation.box()
        apply_cad_transform(plain, trimesh.transformations.translation_matrix([2, 3, 4]))
        self.assertEqual(cad_status(plain), 'mesh')
        np.testing.assert_allclose(plain.bounds.mean(axis=0), [2, 3, 4])

    def test_project_npz_roundtrip_and_history_undo_redo_preserve_cad(self):
        mesh = self.native()
        history = ProjectHistory()
        history.reset(ProjectState(parts=[dict(mesh=mesh.copy(), filename='cad.step')]))
        transform = np.diag([-1.5, .75, 2., 1.]); transform[:3, 3] = [3, 4, 5]
        apply_cad_transform(mesh, transform)
        history.push(ProjectState(parts=[dict(mesh=mesh.copy(), filename='cad.step')]), 'Transform')
        self.assertEqual(len(history.entries), 2)
        for index, expected in ((0, np.eye(4)), (1, transform), (0, np.eye(4)), (1, transform)):
            restored = deepcopy(history.entries[index][1]).parts[0]['mesh']
            self.assertEqual(cad_status(restored), 'native')
            np.testing.assert_allclose(require_native(restored)['matrix'], expected)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cad.mrp'
            save_project(path, ProjectState(parts=[dict(mesh=mesh, filename='cad.step')]))
            restored = load_project(path).parts[0]['mesh']
            self.assertEqual(cad_status(restored), 'native')
            np.testing.assert_equal(restored.metadata[CAD_KEY], mesh.metadata[CAD_KEY])
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read('project.json'))
                payload = manifest['parts'][0]['info']['metadata'][CAD_KEY]
                self.assertIn('__array__', payload['brep'])
                self.assertIn('__array__', payload['face_ids'])


if __name__ == '__main__':
    unittest.main()
