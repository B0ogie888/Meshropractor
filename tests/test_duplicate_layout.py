from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from duplicate_layout import duplicate_plan


class DuplicateLayoutTests(unittest.TestCase):
    def test_gap_uses_selected_group_bounds_and_preserves_sources(self):
        first=trimesh.creation.box([10,4,6]); second=trimesh.creation.box([2,2,2])
        second.apply_translation([10,0,0]); before=[m.vertices.copy() for m in (first,second)]
        plan=duplicate_plan([first,second],dict(matrix_layout=True,counts=[2,2,1],gaps=[5,3,1]))
        np.testing.assert_allclose(plan['steps'],[21,7,7])
        self.assertEqual({tuple(p) for p in plan['offsets']},{(0,7,0),(21,0,0),(21,7,0)})
        self.assertEqual((plan['new_parts'],plan['total_parts']),(6,8))
        for mesh,vertices in zip((first,second),before): np.testing.assert_array_equal(mesh.vertices,vertices)

    def test_legacy_linear_and_batch_offsets_remain_compatible(self):
        source=trimesh.creation.box()
        linear=duplicate_plan([source],dict(counts=[2],values=[3,-4,5]))
        np.testing.assert_array_equal(linear['offsets'],[[3,-4,5],[6,-8,10]])
        array=duplicate_plan([source],dict(counts=[2,2,1],values=[3,4,0]),'Пакетное дублирование')
        self.assertEqual({tuple(p) for p in array['offsets']},{(0,4,0),(3,0,0),(3,4,0)})

    def test_empty_cell_produces_no_new_copies(self):
        plan=duplicate_plan([trimesh.creation.box()],dict(matrix_layout=True,counts=[1,1,1],gaps=[0,0,0]))
        self.assertEqual(plan['offsets'].shape,(0,3)); self.assertEqual(plan['new_parts'],0)

    def test_invalid_and_excessive_layouts_are_rejected_before_allocating(self):
        source=trimesh.creation.box()
        for fields in (dict(counts=[0,1,1]),dict(counts=[1.5,1,1]),dict(counts=[1001,2,1]),
                       dict(counts=[2,1]),dict(gaps=[-1,0,0]),dict(gaps=[np.inf,0,0])):
            params=dict(matrix_layout=True,counts=[2,1,1],gaps=[5,5,1]); params.update(fields)
            with self.subTest(fields=fields),self.assertRaises(ValueError): duplicate_plan([source],params)
        class Large:
            vertices=np.empty(1); faces=np.empty(1)
        class Bytes:
            nbytes=300*1024**2
        large=Large(); large.vertices=large.faces=Bytes()
        with self.assertRaisesRegex(ValueError,'512'):
            duplicate_plan([large],dict(matrix_layout=True,counts=[2,1,1],gaps=[5,5,1]))
