import unittest
import numpy as np
import trimesh
from mesh_intersections import find_intersections, triangle_contacts


class TriangleContactsTests(unittest.TestCase):
    def test_coplanar_overlaps_and_shared_edge(self):
        a=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]*4)
        b=np.array([[[.1,.1,0],[1.1,.1,0],[.1,1.1,0]],
                    [[1.,0,0],[1,1,0],[0,1,0]],
                    [[0.,0,0],[.5,0,0],[0,.5,0]],
                    [[0.,0,.01],[1,0,.01],[0,1,.01]]])
        ov,ix=triangle_contacts(a,b)
        self.assertEqual(ov.tolist(),[True,False,True,False])
        self.assertFalse(ix.any())

    def test_crossing_and_vertex_contact(self):
        a=np.array([[[0.,0,0],[2,0,0],[0,2,0]]]*2)
        b=np.array([[[.5,.5,-1],[.5,.5,1],[1.5,.5,0]],
                    [[0.,0,0],[-1,0,1],[0,-1,1]]])
        ov,ix=triangle_contacts(a,b)
        self.assertFalse(ov.any()); self.assertEqual(ix.tolist(),[True,False])

    def test_closed_solid_has_no_intersections(self):
        for mesh in (trimesh.creation.box(),trimesh.creation.icosphere(subdivisions=2)):
            report=find_intersections(mesh)
            self.assertEqual((report['overlaps'],report['intersections']),(0,0))

    def test_duplicates_and_disconnected_intersecting_solids(self):
        a=trimesh.creation.box()
        b=a.copy(); b.apply_translation([.5,.5,.5])
        self.assertGreater(find_intersections(a+b)['intersections'],0)
        duplicate=trimesh.Trimesh(a.vertices,np.vstack((a.faces,a.faces[0])),process=False)
        self.assertEqual(find_intersections(duplicate)['overlaps'],2)

    def test_cancel(self):
        with self.assertRaises(InterruptedError): find_intersections(trimesh.creation.box(),cancelled=lambda:True)
