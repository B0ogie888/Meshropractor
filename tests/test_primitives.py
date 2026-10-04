"""Physical dimensions, cavity volumes, geometric tolerance and mesh budgets."""
from pathlib import Path
import sys
import unittest
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from primitive_geometry import KINDS,FIELDS,build_primitive,tessellation


class PrimitiveTests(unittest.TestCase):
    def values(self,kind): return {key:value for key,_,value in FIELDS[kind]}

    def test_all_shapes_are_closed_and_centered(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                mesh = build_primitive(kind,self.values(kind),[12,-7,21])
                self.assertTrue(mesh.is_volume)
                np.testing.assert_allclose(mesh.bounds.mean(0),[12,-7,21],atol=.01)
                self.assertEqual(mesh.metadata['primitive']['kind'],kind)

    def test_pipe_cone_pyramid_and_prism_volumes(self):
        expected = dict(Труба=math.pi*(25-6.25)*10,
            Конус=math.pi*10/3*(25+12.5+6.25),Пирамида=1000/3,Призма=6/2*25*math.sin(math.pi/3)*10)
        for kind,volume in expected.items():
            mesh = build_primitive(kind,self.values(kind),[0,0,0])
            self.assertAlmostEqual(mesh.volume,volume,delta=volume*.01)
        values = dict(r=5,top=0,h=10)
        cone = build_primitive('Конус',values,[0,0,0])
        self.assertTrue(cone.is_volume)
        self.assertAlmostEqual(cone.volume,math.pi*25*10/3,delta=3)

    def test_tolerance_limits_sphere_torus_and_cylinder_surface_error(self):
        tolerance = .02
        for kind in ('Цилиндр','Сфера','Тор'):
            p = self.values(kind); mesh = build_primitive(kind,p,[0,0,0],dict(tolerance=tolerance))
            points = mesh.triangles_center
            if kind=='Сфера': error = p['r']-np.linalg.norm(points,axis=1)
            elif kind=='Тор': error = p['tube']-np.linalg.norm(np.c_[np.linalg.norm(points[:,:2],axis=1)-p['r'],points[:,2]],axis=1)
            else:
                points = points[np.abs(mesh.face_normals[:,2])<.1]
                error = p['r']-np.linalg.norm(points[:,:2],axis=1)
            self.assertLessEqual(error.max(),tolerance+1e-8,kind)
            finer = build_primitive(kind,p,[0,0,0],dict(tolerance=tolerance/4))
            self.assertGreater(len(finer.faces),len(mesh.faces))

    def test_fillet_and_chamfer_keep_outer_dimensions(self):
        for kind in KINDS[8:]:
            p = self.values(kind); p.update(x=12,y=16,h=20)
            mesh = build_primitive(kind,p,[0,0,0])
            np.testing.assert_allclose(mesh.extents,[12,16,20],atol=1e-7)
            self.assertLess(mesh.volume,12*16*20)
        kind = KINDS[8]; p = self.values(kind); p['fillet']=6
        with self.assertRaises(ValueError): build_primitive(kind,p,[0,0,0])

    def test_invalid_inner_radius_and_excessive_resolution_do_not_silently_change_settings(self):
        p = self.values('Труба'); p['inner']=p['r']
        with self.assertRaises(ValueError): build_primitive('Труба',p,[0,0,0])
        p = self.values('Тор'); p['tube']=p['r']
        with self.assertRaises(ValueError): build_primitive('Тор',p,[0,0,0])
        with self.assertRaises(ValueError): tessellation('Сфера',self.values('Сфера'),dict(tolerance=1e-12))
        with self.assertRaises(ValueError): tessellation('Тор',self.values('Тор'),dict(mode='segments',segments=1024))
        with self.assertRaises(ValueError): build_primitive('Цилиндр',self.values('Цилиндр'),[float('nan'),0,0])


if __name__=='__main__': unittest.main()
