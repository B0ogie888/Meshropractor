from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cad_state import attach_native, apply_cad_transform, cad_status, require_native
from cad_supports import bind_support_surface, rebind_supports
from part_supports import make_group, transformed
from project_store import ProjectState, load_project, save_project


class CADSupportTests(unittest.TestCase):
    def native(self):
        mesh = trimesh.creation.box(extents=[4, 6, 8])
        _, ids = np.unique(mesh.face_normals, axis=0, return_inverse=True)
        return attach_native(mesh, dict(version=1, brep=np.frombuffer(b'test BRep', dtype=np.uint8),
            matrix=np.eye(4), face_ids=ids.astype(np.int64), body_ids=np.zeros(len(mesh.faces), dtype=np.int64),
            bodies=[{'name': 'Box'}], face_info=[{'body_id': 0} for _ in range(6)]))

    def refined(self, mesh):
        count = len(mesh.vertices)
        faces = [[a, b, count + index] for index, (a, b, c) in enumerate(mesh.faces)]
        faces += [[b, c, count + index] for index, (a, b, c) in enumerate(mesh.faces)]
        faces += [[c, a, count + index] for index, (a, b, c) in enumerate(mesh.faces)]
        result = trimesh.Trimesh(np.vstack((mesh.vertices, mesh.triangles_center)), faces, process=False)
        payload = deepcopy(require_native(mesh))
        payload['face_ids'] = np.tile(payload['face_ids'], 3)
        payload['body_ids'] = np.tile(payload['body_ids'], 3)
        return attach_native(result, payload)

    def group(self, mesh, selected):
        group = make_group(trimesh.creation.box(extents=[.4, .4, 2]), selected, contacts=3)
        return bind_support_surface(group, mesh)

    def test_whole_faces_transfer_exactly_with_independent_support_geometry(self):
        source = self.native()
        selected = np.flatnonzero(require_native(source)['face_ids'] == 0)
        group = self.group(source, selected)
        newmesh = self.refined(source)
        result, = rebind_supports([group], source, newmesh)
        self.assertEqual(result['id'], group['id'])
        self.assertEqual(result['contacts'], 3)
        self.assertEqual(result['surface_faces'], np.flatnonzero(require_native(newmesh)['face_ids'] == 0).tolist())
        self.assertFalse(result['cad_binding']['requires_reselect'])
        np.testing.assert_array_equal(result['vertices'], group['vertices'])
        np.testing.assert_array_equal(result['faces'], group['faces'])
        result['vertices'][0] += 100
        self.assertFalse(np.array_equal(result['vertices'], group['vertices']))
        self.assertEqual(group['surface_faces'], selected.tolist())
        self.assertEqual(cad_status(source), 'native')
        self.assertEqual(cad_status(newmesh), 'native')
        # Supports from older projects gain a binding using their original IDs.
        legacy = deepcopy(group)
        legacy.pop('cad_binding')
        self.assertEqual(rebind_supports([legacy], source, newmesh)[0]['surface_faces'],
                         np.flatnonzero(require_native(newmesh)['face_ids'] == 0).tolist())

    def test_partial_faces_never_expand_and_second_remesh_keeps_reference(self):
        source = self.native()
        owners = require_native(source)['face_ids']
        selected = np.r_[np.flatnonzero(owners == 0), np.flatnonzero(owners == 1)[:1]]
        group = self.group(source, selected)
        newmesh = self.refined(source)
        updated, = rebind_supports([group], source, newmesh)
        self.assertEqual(updated['surface_faces'], np.flatnonzero(require_native(newmesh)['face_ids'] == 0).tolist())
        binding = updated['cad_binding']
        self.assertTrue(binding['requires_reselect'])
        self.assertIn('заново', binding['notice'])
        np.testing.assert_array_equal(binding['cad_face_ids'], [0, 1])
        np.testing.assert_array_equal(binding['partial_face_ids'], [1])
        np.testing.assert_allclose(binding['partial_triangle_centers'], source.triangles_center[selected[-1:]])
        second, = rebind_supports([updated], newmesh, self.refined(newmesh))
        self.assertTrue(second['cad_binding']['requires_reselect'])
        np.testing.assert_array_equal(second['cad_binding']['partial_triangle_centers'], binding['partial_triangle_centers'])
        np.testing.assert_array_equal(updated['vertices'], group['vertices'])
        self.assertFalse(group['cad_binding']['requires_reselect'])

    def test_identical_tessellation_preserves_partial_triangle_ids(self):
        source = self.native()
        group = self.group(source, [0])
        updated, = rebind_supports([group], source, source.copy())
        self.assertEqual(updated['surface_faces'], [0])
        self.assertFalse(updated['cad_binding']['requires_reselect'])

    def test_local_partial_references_survive_affine_part_transform(self):
        source = self.native()
        group = self.group(source, [0])
        original_centers = group['cad_binding']['partial_triangle_centers'].copy()
        matrix = np.diag([-2., 3., .5, 1.])
        matrix[:3, 3] = [4, 5, 6]
        apply_cad_transform(source, matrix)
        moved, = transformed([group], matrix)
        updated, = rebind_supports([moved], source, self.refined(source))
        np.testing.assert_array_equal(updated['cad_binding']['partial_triangle_centers'], original_centers)
        np.testing.assert_allclose(updated['vertices'], moved['vertices'])
        rebound = self.group(source, [0])
        np.testing.assert_allclose(rebound['cad_binding']['partial_triangle_centers'], original_centers)

    def test_reject_unrelated_shape_placement_invalid_mesh_and_corrupt_binding(self):
        source = self.native()
        group = self.group(source, [0])
        different = self.refined(source)
        payload = deepcopy(require_native(different))
        payload['brep'][0] ^= 1
        attach_native(different, payload)
        with self.assertRaisesRegex(ValueError, 'той же'):
            rebind_supports([group], source, different)
        moved = self.refined(source)
        matrix = np.eye(4); matrix[2, 3] = 1
        apply_cad_transform(moved, matrix)
        with self.assertRaisesRegex(ValueError, 'том же положении'):
            rebind_supports([group], source, moved)
        edited = source.copy(); edited.vertices[0] += .1
        with self.assertRaises(ValueError):
            rebind_supports([group], edited, self.refined(source))
        invalid = deepcopy(group)
        invalid['cad_binding']['full_face_ids'] = invalid['cad_binding']['partial_face_ids']
        invalid['cad_binding']['partial_face_ids'] = np.empty(0, dtype=np.int64)
        with self.assertRaises(ValueError):
            rebind_supports([invalid], source, self.refined(source))
        invalid = deepcopy(group); invalid['surface_faces'] = [-1]
        with self.assertRaises(ValueError):
            bind_support_surface(invalid, source)

    def test_stl_and_edited_mesh_supports_keep_the_triangle_workflow(self):
        mesh = trimesh.creation.box()
        group = make_group(trimesh.creation.box(), [0], contacts=1)
        group['cad_binding'] = {'obsolete': True}
        self.assertIs(bind_support_surface(group, mesh), group)
        self.assertNotIn('cad_binding', group)
        self.assertEqual(group['surface_faces'], [0])
        edited = self.native(); edited.vertices[0] += 1
        bind_support_surface(group, edited)
        self.assertNotIn('cad_binding', group)

    def test_real_cad_bottom_support_generation_remesh_and_project_roundtrip(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
        from cad_import import load_step
        from cad_geometry import retessellate
        from support_geometry import generate_supports, overhang_faces
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cylinder.step'
            writer = STEPControl_Writer()
            self.assertEqual(writer.Transfer(BRepPrimAPI_MakeCylinder(4, 3).Shape(), STEPControl_AsIs), IFSelect_RetDone)
            self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
            source = load_step(path, .5, .8, native=True)
            matrix = np.eye(4); matrix[2, 3] = 5
            apply_cad_transform(source, matrix)
            params = dict(angle=45., spacing=3., diameter=.6, tip_diameter=.3, tip_height=.5,
                          foot_diameter=1., foot_height=.4, base_z=0., only_platform=True)
            records = [dict(mesh=source, row=0, filename=path.name, platform=0)]
            generated, = generate_supports(records, {0: overhang_faces(source)}, params)
            group = make_group(generated['mesh'], generated['surface_faces'], contacts=generated['contacts'], params=params)
            bind_support_surface(group, source)
            self.assertGreater(group['contacts'], 0)
            self.assertGreater(len(group['cad_binding']['full_face_ids']), 0)
            self.assertEqual(len(group['cad_binding']['partial_face_ids']), 0)
            newmesh = retessellate(source, .02, .12)
            self.assertGreater(len(newmesh.faces), len(source.faces))
            groups = rebind_supports([group], source, newmesh)
            self.assertFalse(groups[0]['cad_binding']['requires_reselect'])
            np.testing.assert_array_equal(groups[0]['surface_faces'], overhang_faces(newmesh))
            np.testing.assert_array_equal(groups[0]['vertices'], group['vertices'])
            project = Path(folder) / 'supports.mrp'
            save_project(project, ProjectState(parts=[dict(mesh=newmesh, filename=path.name, supports=groups)]))
            restored = load_project(project)
            self.assertEqual(len(restored.parts), 1)
            self.assertEqual(len(restored.parts[0]['supports']), 1)
            self.assertEqual(cad_status(restored.parts[0]['mesh']), 'native')
            rebound = rebind_supports(restored.parts[0]['supports'], restored.parts[0]['mesh'],
                                     retessellate(restored.parts[0]['mesh'], .01, .1))
            self.assertEqual(rebound[0]['id'], group['id'])
            self.assertEqual(rebound[0]['contacts'], group['contacts'])
            self.assertNotIn('brep', rebound[0]['cad_binding'])


if __name__ == '__main__':
    unittest.main()
