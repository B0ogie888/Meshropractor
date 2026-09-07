import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_store import ProjectState, load_project, save_project


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.mrp"

    def state(self):
        mesh = trimesh.creation.box()
        mesh.metadata = {"vectors": np.full((8, 3), 0.1), "settings": {"factor": [1, 1, 2]}}
        models = [dict(key="CAD_0", kind="CAD", name="CAD.stl", mesh=mesh, style={}),
                  dict(key="Result_0", kind="Result", name="Результат 1", mesh=mesh.copy(), style={"transparency": 40}),
                  dict(key="Def_1", kind="Def", name="Результат 2", mesh=mesh.copy(), style={}),
                  dict(key="Heatmap_0", kind="Heatmap", name="Карта", mesh=mesh.copy(), deviations=np.arange(8.), style={})]
        return ProjectState(models=models, active_result="Def_1", active_heatmap="Heatmap_0", settings={"points": 5000})

    def test_roundtrip_all_results_arrays_and_parameters(self):
        state = self.state()
        save_project(self.path, state)
        actual = load_project(self.path)
        self.assertEqual([r["name"] for r in actual.models], [r["name"] for r in state.models])
        self.assertEqual(actual.active_result, "Def_1")
        self.assertEqual(actual.settings["points"], 5000)
        for expected, found in zip(state.models, actual.models):
            np.testing.assert_array_equal(found["mesh"].vertices, expected["mesh"].vertices)
            np.testing.assert_array_equal(found["mesh"].faces, expected["mesh"].faces)
            np.testing.assert_array_equal(found["mesh"].metadata["vectors"], expected["mesh"].metadata["vectors"])
        np.testing.assert_array_equal(actual.models[-1]["deviations"], np.arange(8.))

    def test_failed_replace_preserves_existing_archive_and_cleans_temp(self):
        save_project(self.path, self.state())
        original = self.path.read_bytes()
        with patch("project_store.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError): save_project(self.path, self.state())
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.path])

    def test_bad_metadata_cannot_truncate_previous_save(self):
        save_project(self.path, self.state())
        original = self.path.read_bytes()
        invalid = self.state()
        invalid.settings["bad"] = float("nan")
        with self.assertRaises(ValueError): save_project(self.path, invalid)
        self.assertEqual(self.path.read_bytes(), original)

    def test_legacy_parts_are_loaded_in_numeric_order(self):
        with zipfile.ZipFile(self.path, "w") as z:
            z.writestr("project.json", json.dumps({"version": "1.4", "slicer_name_2": "two", "slicer_name_10": "ten"}))
            for i in (10, 2): z.writestr(f"meshes/slicer_part_{i}.stl", trimesh.creation.box().export(file_type="stl"))
        self.assertEqual([p["filename"] for p in load_project(self.path).parts], ["two", "ten"])

    def test_unknown_version_rejected(self):
        with zipfile.ZipFile(self.path, "w") as z:
            z.writestr("project.json", '{"version":"99.0"}')
        with self.assertRaisesRegex(ValueError, "не поддерживается"): load_project(self.path)


if __name__ == "__main__": unittest.main()
