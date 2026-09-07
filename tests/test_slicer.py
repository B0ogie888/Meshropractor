from pathlib import Path
import sys
import tempfile
import unittest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from CL_Slicer import slice_stl_to_cls


class SlicerTests(unittest.TestCase):
    def test_invalid_layer_and_cancel_preserve_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'part.stl', Path(folder) / 'job.cls'
            trimesh.creation.box().export(source)
            target.write_bytes(b'previous job')
            with self.assertRaises(ValueError): slice_stl_to_cls(source, None, target, layer_height=0)
            self.assertFalse(slice_stl_to_cls(source, None, target, cancel_callback=lambda: True))
            self.assertEqual(target.read_bytes(), b'previous job')
            self.assertEqual(len(list(Path(folder).iterdir())), 2)

    def test_support_bounds_and_partial_layer_are_included(self):
        with tempfile.TemporaryDirectory() as folder:
            source, support, target = [Path(folder) / p for p in ('part.stl', 'support.stl', 'job.cls')]
            part = trimesh.creation.box(); part.apply_translation([0, 0, 1]); part.export(source)
            trimesh.creation.box().export(support)
            progress = []
            self.assertTrue(slice_stl_to_cls(source, support, target, layer_height=0.3, progress_callback=progress.append))
            self.assertEqual(target.read_bytes().count(b'NEW_LAYER'), 7)
            self.assertEqual(progress[-1], 100)
            self.assertTrue(all(0 <= p <= 100 for p in progress))
