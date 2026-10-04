from pathlib import Path
import sys
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from analysis_geometry import build_estimate, cavities, collisions, point_thickness, slice_distribution, wall_samples
from analysis_tools import DEFAULTS, plain


class AnalysisTests(unittest.TestCase):
    def records(self,*meshes):return [dict(mesh=m,filename=str(i),supports=[]) for i,m in enumerate(meshes)]

    def test_true_collision_containment_touch_and_cancellation(self):
        a=trimesh.creation.box([2,2,2]);b=a.copy();b.apply_translation([1,0,0])
        r=collisions(self.records(a,b));self.assertAlmostEqual(r[0]['volume'],4.)
        b.apply_translation([1,0,0]);self.assertEqual(collisions(self.records(a,b))[0]['volume'],0.)
        inner=trimesh.creation.box([1,1,1]);self.assertAlmostEqual(collisions(self.records(a,inner))[0]['volume'],1.)
        with self.assertRaises(InterruptedError):collisions(self.records(a,b),cancelled=lambda:True)
        b.update_faces(np.arange(11));self.assertIsNone(collisions(self.records(a,b))[0]['volume'])

    def test_sampled_extraction_finds_later_obstruction(self):
        a=trimesh.creation.box([2,2,2]);b=a.copy();b.apply_translation([0,0,6])
        self.assertEqual(collisions(self.records(a,b))[0]['volume'],0.)
        r=collisions(self.records(a,b),direction=[0,0,1],travel=8,steps=16)
        self.assertGreater(r[0]['volume'],0);self.assertGreater(r[0]['offset'],0)

    def test_wall_thickness_box_is_exact_and_open_surface_is_not_certified(self):
        a=trimesh.creation.box([2,3,4]);r=wall_samples(a,count=1200,threshold=2.5)
        self.assertTrue(r['reliable']);self.assertEqual(r['hits'],1200)
        self.assertAlmostEqual(r['minimum'],2.,places=4);self.assertAlmostEqual(r['maximum'],4.,places=4)
        face=int(np.flatnonzero(a.face_normals[:,0]>.9)[0]);point=a.triangles_center[face]
        end,value=point_thickness(a,point,a.face_normals[face]);self.assertAlmostEqual(value,2.,places=4)
        np.testing.assert_allclose(end[0],-1.,atol=1e-4)
        sheet=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
        self.assertFalse(wall_samples(sheet,count=10)['reliable'])
        self.assertEqual(wall_samples(sheet,count=10)['hits'],0)
        with self.assertRaises(ValueError):point_thickness(sheet,[.2,.2,0],[0,0,1])

    def test_inner_shell_diagnostics_preserve_hollow_volume(self):
        outer=trimesh.creation.box([4,4,4]);inner=trimesh.creation.box([2,2,2]);inner.invert()
        hollow=trimesh.util.concatenate([outer,inner]);r=cavities(hollow)
        self.assertEqual(r['inward_shells'],1);self.assertTrue(r['reliable'])
        self.assertAlmostEqual(sum(s['signed_volume'] for s in r['shells']),56.)

    def test_slices_have_correct_area_and_hollow_holes(self):
        outer=trimesh.creation.box([4,4,4]);outer.apply_translation([0,0,2])
        r=slice_distribution(self.records(outer),layer_height=.5,samples=3)
        np.testing.assert_allclose(r['area_mm2'],16.);self.assertEqual(r['layers'],8)
        inner=trimesh.creation.box([2,2,4]);inner.apply_translation([0,0,2]);inner.invert()
        r=slice_distribution(self.records(trimesh.util.concatenate([outer,inner])),layer_height=.5,samples=3)
        np.testing.assert_allclose(r['area_mm2'],12.)

    def test_units_time_cost_and_unknown_volume(self):
        a=trimesh.creation.box([10,10,10]);a.apply_translation([0,0,5])
        params=dict(DEFAULTS,layer_height=1.,volume_rate=10.,layer_seconds=2.,setup_minutes=1.,density=2.,material_price=100.,hour_price=36.,fixed_cost=3.)
        r=build_estimate(self.records(a),params)
        self.assertEqual(r['layers'],10);self.assertEqual(r['seconds'],180.);self.assertEqual(r['mass_g'],2.)
        self.assertAlmostEqual(r['total_cost'],5.);self.assertAlmostEqual(r['material_cost'],.2)
        self.assertEqual(plain({'a':np.array([np.nan,np.inf,1.])}),{'a':[None,None,1.]})
        with self.assertRaises(ValueError):build_estimate(self.records(a),dict(params,volume_rate=0.))


if __name__=='__main__':unittest.main()
