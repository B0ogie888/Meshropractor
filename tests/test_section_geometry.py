from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from section_geometry import default_sections, validate_sections, clipping_plane, projected_range


class SectionGeometryTests(unittest.TestCase):
    def test_axes_sides_and_offset(self):
        for section in default_sections():
            section["position"] = 3
            normal = np.array(section["normal"])
            for side in ("+", "−"):
                section["cut"] = side
                plane = clipping_plane(section)
                self.assertAlmostEqual(plane.EvaluateFunction(normal * 3), 0)
                self.assertEqual(np.sign(plane.EvaluateFunction(normal * 4)), -1 if side == "+" else 1)
        np.testing.assert_allclose(projected_range([[-1, -2, -3], [1, 2, 3]], [0, 0, 1]), [-3, 3])

    def test_reject_invalid_planes_before_loading_project(self):
        for key, value in (("normal", [0, 0, 0]), ("step", 0), ("position", float("nan")), ("color", "invalid"), ("axis", "bad")):
            section = default_sections()[0]
            section[key] = value
            with self.assertRaises(ValueError):
                validate_sections([section])
        with self.assertRaises(ValueError):
            validate_sections(default_sections() * 3)
