from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_store import load_mesh, ProjectState, save_project, load_project
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Trsf, gp_Vec


class CADImportTests(unittest.TestCase):
    def write_step(self, shape, path, unit="MM"):
        writer = STEPControl_Writer()
        Interface_Static.SetCVal_s("write.step.unit", unit)
        try:
            self.assertEqual(writer.Transfer(shape, STEPControl_AsIs), IFSelect_RetDone)
            self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
        finally:
            Interface_Static.SetCVal_s("write.step.unit", "MM")

    def test_step_units_unicode_path_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            for unit in ("MM", "INCH"):
                path = Path(folder) / f"деталь_{unit}.STEP"
                self.write_step(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), path, unit)
                mesh = load_mesh(path)
                np.testing.assert_allclose(mesh.extents, [10, 20, 30], atol=1e-6)
                self.assertTrue(mesh.is_watertight)
                self.assertGreater(mesh.volume, 0)
                self.assertEqual(mesh.metadata["units"], "mm")
                archive = Path(folder) / "test.mrp"
                save_project(archive, ProjectState(parts=[dict(mesh=mesh, filename=path.name)]))
                restored = load_project(archive).parts[0]["mesh"]
                self.assertEqual(restored.metadata["source_format"], "STEP")
                np.testing.assert_allclose(restored.vertices, mesh.vertices)

    def test_compound_locations_and_curved_tessellation(self):
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, BRepPrimAPI_MakeBox(10, 10, 10).Shape())
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(40, 0, 0))
        cylinder = BRepPrimAPI_MakeCylinder(5, 20).Shape().Moved(TopLoc_Location(transform))
        builder.Add(compound, cylinder)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "assembly.stp"
            self.write_step(compound, path)
            coarse = load_mesh(path, .5)
            fine = load_mesh(path, .001)
            np.testing.assert_allclose(fine.bounds, [[0, -5, 0], [45, 10, 20]], atol=.02)
            self.assertEqual(fine.metadata["cad_body_count"], 2)
            self.assertGreater(len(fine.faces), len(coarse.faces))
            self.assertTrue(fine.is_watertight)
            self.assertAlmostEqual(fine.volume, 1000 + np.pi * 25 * 20, delta=2)

    def test_broken_file_and_invalid_precision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.step"
            path.write_text("invalid STEP")
            with self.assertRaises(ValueError):
                load_mesh(path)
            with self.assertRaises(ValueError):
                load_mesh(path, -1)

    def test_angular_tolerance_changes_curved_surface_density(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'angle.step'
            self.write_step(BRepPrimAPI_MakeCylinder(10, 20).Shape(), path)
            coarse = load_mesh(path, 1, np.deg2rad(45))
            fine = load_mesh(path, 1, np.deg2rad(5))
            self.assertGreater(len(fine.faces), len(coarse.faces))
            self.assertAlmostEqual(fine.metadata['angular_deflection_rad'], np.deg2rad(5))
            self.assertTrue(fine.is_watertight)


if __name__ == "__main__":
    unittest.main()
