"""Native STEP bodies survive proxy changes, transforms and project storage."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cad_import import load_step
from cad_geometry import retessellate, split_bodies, export_step, native_info
from cad_state import require_native, apply_cad_transform, cad_status, cad_face_triangles
from project_store import ProjectState, save_project, load_project
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Trsf, gp_Vec, gp_Pln, gp_Pnt, gp_Dir
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone


class CADNativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def imported(self, shape=None, name='деталь.step', **parameters):
        shape = shape if shape is not None else BRepPrimAPI_MakeBox(2, 3, 4).Shape()
        path = self.folder / name
        writer = STEPControl_Writer()
        self.assertEqual(writer.Transfer(shape, STEPControl_AsIs), IFSelect_RetDone)
        self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
        return load_step(path, **parameters)

    def assembly(self, sheet=False):
        compound, builder = TopoDS_Compound(), BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, BRepPrimAPI_MakeBox(2, 3, 4).Shape())
        transform = gp_Trsf(); transform.SetTranslation(gp_Vec(20, 0, 0))
        cylinder = BRepPrimAPI_MakeCylinder(2, 5).Shape().Moved(TopLoc_Location(transform))
        builder.Add(compound, cylinder)
        if sheet:
            plane = gp_Pln(gp_Pnt(40, 0, 2), gp_Dir(0, 0, 1))
            builder.Add(compound, BRepBuilderAPI_MakeFace(plane, 0, 2, 0, 3).Face())
        return compound

    def test_native_body_face_mapping_exact_properties_and_legacy_mode(self):
        mesh = self.imported()
        payload = require_native(mesh)
        self.assertEqual(cad_status(mesh), 'native')
        self.assertEqual(payload['brep'].dtype, np.uint8)
        self.assertEqual(payload['face_ids'].dtype, np.int64)
        self.assertEqual(set(payload['face_ids']), set(range(6)))
        self.assertEqual(set(payload['body_ids']), {0})
        for face_id in range(6):
            selected = cad_face_triangles(mesh, int(np.flatnonzero(payload['face_ids'] == face_id)[0]))
            self.assertEqual(len(selected), 2)
        info = native_info(mesh)
        self.assertTrue(info['is_valid'])
        self.assertEqual(info['solid_body_count'], 1)
        self.assertAlmostEqual(info['volume_mm3'], 24.)
        self.assertAlmostEqual(info['area_mm2'], 52.)
        self.assertTrue(all(face['type'] == 'plane' for face in info['face_info']))
        self.assertEqual(mesh.metadata['source_path'], str((self.folder / 'деталь.step').resolve()))
        legacy = load_step(self.folder / 'деталь.step', native=False)
        self.assertEqual(cad_status(legacy), 'mesh')
        self.assertTrue(legacy.is_watertight)
        np.testing.assert_allclose(legacy.bounds, mesh.bounds)

    def test_retessellation_preserves_native_face_ids_matrix_and_source(self):
        mesh = self.imported(BRepPrimAPI_MakeCylinder(5, 10).Shape(), linear_deflection=.5, angular_deflection=.8)
        matrix = trimesh.transformations.rotation_matrix(.3, [1, 1, 0])
        matrix[:3, 3] = [12, -6, 20]
        apply_cad_transform(mesh, matrix)
        original_vertices, original_faces = mesh.vertices.copy(), mesh.faces.copy()
        old = require_native(mesh)
        refined = retessellate(mesh, .01, .1)
        new = require_native(refined)
        self.assertGreater(len(refined.faces), len(mesh.faces))
        np.testing.assert_array_equal(new['brep'], old['brep'])
        np.testing.assert_array_equal(new['matrix'], old['matrix'])
        self.assertEqual(new['face_info'], old['face_info'])
        self.assertEqual(set(new['face_ids']), set(old['face_ids']))
        self.assertTrue(refined.is_watertight)
        np.testing.assert_array_equal(mesh.vertices, original_vertices)
        np.testing.assert_array_equal(mesh.faces, original_faces)
        self.assertAlmostEqual(native_info(refined)['volume_mm3'], np.pi * 25 * 10, places=7)

    def test_splitting_keeps_locations_and_open_surfaces(self):
        mesh = self.imported(self.assembly(sheet=True))
        info = native_info(mesh)
        self.assertEqual(info['body_count'], 3)
        self.assertEqual(info['solid_body_count'], 2)
        self.assertIsNone(info['volume_mm3'])
        self.assertAlmostEqual(info['solid_volume_mm3'], 24 + np.pi * 4 * 5, places=7)
        translation = np.eye(4); translation[:3, 3] = [3, 7, 11]
        apply_cad_transform(mesh, translation)
        parts = split_bodies(mesh)
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(sum(native_info(part)['area_mm2'] for part in parts), info['area_mm2'], places=7)
        self.assertEqual([native_info(part)['solid_body_count'] for part in parts], [1, 1, 0])
        for part in parts:
            payload = require_native(part)
            self.assertEqual(len(payload['bodies']), 1)
            self.assertEqual(set(payload['body_ids']), {0})
            np.testing.assert_array_equal(payload['matrix'], translation)
            self.assertTrue(part.metadata['cad_body_name'])
        np.testing.assert_allclose(parts[0].bounds, [[3, 7, 11], [5, 10, 15]], atol=1e-7)
        np.testing.assert_allclose(parts[2].bounds, [[43, 7, 13], [45, 10, 13]], atol=1e-7)
        path = self.folder / 'surfaces_export.step'
        export_step(parts, path)
        self.assertEqual(native_info(load_step(path))['body_count'], 3)

    def test_affine_export_roundtrip_nonuniform_shear_mirror_and_atomic_replace(self):
        source = self.imported()
        for name, linear in [('scale', np.diag([2., 3., 4.])),
                             ('shear', np.array([[1., .4, 0], [0, 1., 0], [0, 0, 1.]])),
                             ('mirror', np.diag([-1., 1., 1.]))]:
            with self.subTest(name=name):
                mesh = source.copy()
                matrix = np.eye(4); matrix[:3, :3] = linear; matrix[:3, 3] = [8, -3, 6]
                apply_cad_transform(mesh, matrix)
                info = native_info(mesh)
                expected = 24 * abs(np.linalg.det(linear))
                self.assertTrue(info['is_valid'])
                self.assertAlmostEqual(info['volume_mm3'], expected, places=7)
                path = self.folder / f'{name}_export.step'
                report = export_step([mesh], path)
                self.assertEqual(report['body_count'], 1)
                restored = load_step(path)
                np.testing.assert_allclose(restored.bounds, mesh.bounds, atol=1e-6)
                self.assertAlmostEqual(native_info(restored)['volume_mm3'], expected, places=6)
                self.assertTrue(restored.is_watertight)
                self.assertGreater(restored.volume, 0)
        self.assertAlmostEqual(native_info(source)['volume_mm3'], 24.)

    def test_analytic_radii_and_world_linear_tolerance_after_scaling(self):
        mesh = self.imported(BRepPrimAPI_MakeCylinder(2, 4).Shape())
        info = native_info(mesh)
        self.assertEqual([face['radius_mm'] for face in info['face_info'] if face['type'] == 'cylinder'], [2.])
        self.assertTrue(any(face['circular_edges'] for face in info['face_info']))
        moved = mesh.copy(); rigid = trimesh.transformations.rotation_matrix(.4, [1, 1, 1]); rigid[:3, 3] = [10, -7, 12]
        apply_cad_transform(moved, rigid)
        np.testing.assert_allclose([face['radius_mm'] for face in native_info(moved)['face_info'] if face['type'] == 'cylinder'], [2.])
        uniform = mesh.copy(); matrix = np.diag([3., 3., 3., 1.]); apply_cad_transform(uniform, matrix)
        radii = [face['radius_mm'] for face in native_info(uniform)['face_info'] if face['type'] == 'cylinder']
        np.testing.assert_allclose(radii, [6.])
        anisotropic = mesh.copy(); matrix = np.diag([2., 3., 4., 1.]); apply_cad_transform(anisotropic, matrix)
        refined = retessellate(anisotropic, .04, .1)
        self.assertAlmostEqual(refined.metadata['linear_deflection_mm'], .04)
        self.assertAlmostEqual(refined.metadata['native_linear_deflection_mm'], .01)
        self.assertEqual(refined.metadata['angular_deflection_space'], 'local_cad')
        for face in native_info(anisotropic)['face_info']:
            if face['type'] == 'bsplinesurface': self.assertNotIn('radius_mm', face)

    def test_modified_proxy_refuses_native_actions_and_project_preserves_binding(self):
        mesh = self.imported(self.assembly())
        project = self.folder / 'native.mrp'
        save_project(project, ProjectState(parts=[dict(mesh=mesh, filename='native.step')]))
        restored = load_project(project).parts[0]['mesh']
        self.assertEqual(cad_status(restored), 'native')
        np.testing.assert_array_equal(require_native(restored)['brep'], require_native(mesh)['brep'])
        restored.vertices[0] += [.1, 0, 0]
        destination = self.folder / 'must_not_change.step'
        destination.write_bytes(b'original')
        for operation in (lambda: retessellate(restored, .1, .2), lambda: split_bodies(restored),
                          lambda: native_info(restored), lambda: export_step([restored], destination)):
            with self.assertRaises(ValueError): operation()
        self.assertEqual(destination.read_bytes(), b'original')

    def test_cancellation_budgets_and_export_failure_preserve_destination(self):
        mesh = self.imported()
        destination = self.folder / 'atomic.step'
        destination.write_bytes(b'keep existing STEP')
        for operation in (lambda: load_step(self.folder / 'деталь.step', cancelled=lambda: True),
                          lambda: retessellate(mesh, .1, .2, cancelled=lambda: True),
                          lambda: split_bodies(mesh, cancelled=lambda: True),
                          lambda: native_info(mesh, cancelled=lambda: True),
                          lambda: export_step([mesh], destination, cancelled=lambda: True)):
            with self.assertRaises(InterruptedError): operation()
        messages = []
        with self.assertRaises(InterruptedError):
            export_step([mesh], destination, progress=messages.append,
                        cancelled=lambda: any('Запись STEP' in message for message in messages))
        self.assertEqual(destination.read_bytes(), b'keep existing STEP')
        self.assertFalse(list(self.folder.glob('.atomic.step.*')))
        with patch('cad_geometry.MAX_TRIANGLES', 1), self.assertRaisesRegex(ValueError, 'плотная'):
            retessellate(mesh, .1, .2)
        with patch('cad_geometry.MAX_INPUT_BYTES', 1), self.assertRaisesRegex(ValueError, '256'):
            load_step(self.folder / 'деталь.step')
        for linear, angular in ((1e-12, .2), (.1, 1e-9), (-1, .2), (.1, np.nan)):
            with self.assertRaises(ValueError): retessellate(mesh, linear, angular)
        self.assertEqual(cad_status(mesh), 'native')

    def test_two_named_colored_bodies_survive_step_roundtrip(self):
        from OCP.TDocStd import TDocStd_Document
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen
        from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCP.STEPCAFControl import STEPCAFControl_Writer
        document = TDocStd_Document(TCollection_ExtendedString('BinXCAF'))
        shapes = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
        colors = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
        names = ['Красная деталь', 'Blue cylinder']
        for name, shape, color in zip(names, [BRepPrimAPI_MakeBox(2, 3, 4).Shape(), BRepPrimAPI_MakeCylinder(1, 5).Shape()],
                                     [[1., 0., 0.], [0., 0., 1.]]):
            label = shapes.AddShape(shape, False)
            TDataStd_Name.Set_s(label, TCollection_ExtendedString(name, True))
            colors.SetColor(label, Quantity_Color(*color, Quantity_TOC_RGB), XCAFDoc_ColorGen)
        path = self.folder / 'named.step'
        writer = STEPCAFControl_Writer(); writer.SetNameMode(True); writer.SetColorMode(True)
        self.assertTrue(writer.Transfer(document, STEPControl_AsIs))
        self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
        mesh = load_step(path)
        payload = require_native(mesh)
        self.assertEqual([body['name'] for body in payload['bodies']], names)
        np.testing.assert_allclose([body['color'] for body in payload['bodies']], [[1, 0, 0], [0, 0, 1]])
        exported = self.folder / 'named_export.step'
        export_step([mesh], exported)
        self.assertEqual([body['name'] for body in require_native(load_step(exported))['bodies']], names)


if __name__ == '__main__':
    unittest.main()
