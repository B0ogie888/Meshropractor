"""Solid volumes, cavities, holes, arrays and bounded implicit edits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import trimesh
from model_tool_geometry import calculate


class ModelToolTests(unittest.TestCase):
    def setUp(self):
        self.box=trimesh.creation.box([10,10,10]);self.box.apply_translation([0,0,5])
    def run_op(self,op,params=None,meshes=None,selection=None):
        records=[dict(row=i,mesh=m,name=str(i)) for i,m in enumerate(meshes or [self.box])]
        return calculate(records,op,params or {},selection=selection)
    def mesh(self,op,params=None):return self.run_op(op,params)['items'][0]['meshes'][0]
    def test_boolean_volumes_and_merge_differ(self):
        b=self.box.copy();b.apply_translation([5,0,0])
        for op,volume in [('union',1500),('intersection',500),('difference',500),('remove_volume',500),('merge',2000)]:
            result=self.run_op(op,meshes=[self.box,b]);self.assertEqual(result['mode'],'merge')
            self.assertAlmostEqual(result['items'][0]['meshes'][0].volume,volume,delta=.001)
        self.assertEqual(self.box.volume,1000)
    def test_cut_keeps_both_capped_halves(self):
        meshes=self.run_op('cut',dict(point=[0,0,5],normal=[0,0,1]))['items'][0]['meshes']
        self.assertEqual(len(meshes),2)
        for mesh in meshes:self.assertTrue(mesh.is_volume);self.assertAlmostEqual(mesh.volume,500)
    def test_hollow_core_and_mould_are_closed(self):
        hollow=self.mesh('hollow',dict(step=.5,wall=1.5));self.assertTrue(hollow.is_volume)
        self.assertAlmostEqual(hollow.volume,1000-7**3,delta=45)
        pieces=self.run_op('shell_core',dict(step=.5,wall=1.5))['items'][0]['meshes']
        self.assertEqual(len(pieces),2);self.assertAlmostEqual(sum(m.volume for m in pieces),1000,delta=40)
        mould=self.mesh('formfit',dict(step=.5,wall=1.5,gap=.5));self.assertTrue(mould.is_volume);self.assertGreater(mould.bounds[1,0],5.5)
    def test_rounding_offsets_and_fixture(self):
        for op in ('round','round_offset'):
            mesh=self.mesh(op,dict(step=.5,radius=1.5,offset=1.))
            self.assertTrue(mesh.is_volume)
        offset=self.mesh('offset',dict(offset=.5));self.assertGreater(offset.volume,1000)
        fixture=self.run_op('rapidfit',dict(wall=2.,gap=.2));self.assertEqual(fixture['mode'],'add');self.assertTrue(fixture['items'][0]['meshes'][0].is_volume)
    def test_all_structures_have_real_voids_and_solid_boundaries(self):
        for op in ('honeycomb','lattice','slice_lattice','tetra','tetra_slices'):
            with self.subTest(op=op):
                mesh=self.mesh(op,dict(step=.5,wall=0.,cell=5.,thickness=1.5))
                self.assertTrue(mesh.is_volume);self.assertGreater(mesh.volume,0);self.assertLess(mesh.volume,900)
    def test_perforations_and_surface_extrusion(self):
        perforated=self.mesh('perforate',dict(radius=1.,cell=5.,axis=2))
        self.assertTrue(perforated.is_volume);self.assertLess(perforated.volume,1000)
        ids=np.flatnonzero(self.box.face_normals[:,2]>.9).tolist()
        for distance,volume in [(2.,1200),(-2.,800)]:
            mesh=self.run_op('extrude',dict(distance=distance),selection={0:ids})['items'][0]['meshes'][0]
            self.assertTrue(mesh.is_volume);self.assertAlmostEqual(mesh.volume,volume,delta=.001)
        result=self.run_op('surface_array',dict(distance=2.,count=3),selection={0:ids})
        self.assertEqual(result['mode'],'add');self.assertEqual(len(result['items'][0]['meshes']),3)
        self.assertEqual(result['items'][0]['meshes'][2].bounds[0,2],16.)
    def test_text_holes_and_strut(self):
        text=self.run_op('label',dict(text_contours=[[[0,0],[4,0],[4,4],[0,4],[0,0]],[[1,1],[3,1],[3,3],[1,3],[1,1]]],height=2.,position=[0,0,0],standalone=True))
        mesh=text['items'][0]['meshes'][0];self.assertTrue(mesh.is_volume);self.assertAlmostEqual(mesh.volume,24.)
        strut=self.mesh('struts',dict(radius=1.,start=[0,0,0],end=[3,4,0]))
        self.assertTrue(strut.is_volume);self.assertAlmostEqual(strut.volume,np.pi*5,delta=.1)
    def test_limits_open_models_and_cancellation_leave_sources_untouched(self):
        source_vertices=self.box.vertices.copy()
        with self.assertRaises(ValueError):self.mesh('hollow',dict(step=.001,wall=1.))
        with self.assertRaises(ValueError):self.mesh('hollow',dict(step=1.,wall=.5))
        opened=self.box.copy();opened.update_faces(np.arange(11))
        with self.assertRaises(ValueError):self.run_op('hollow',dict(step=.5,wall=1.5),meshes=[opened])
        with self.assertRaises(InterruptedError):calculate([dict(row=0,mesh=self.box,name='a')],'hollow',{},cancelled=lambda:True)
        with self.assertRaises(ValueError):calculate([dict(row=0,mesh=self.box,name='a',supports=[{'id':'existing'}])],'offset',dict(offset=.5))
        self.assertEqual(self.run_op('formfit',dict(step=.5,wall=1.5,gap=.5))['mode'],'add')
        np.testing.assert_array_equal(self.box.vertices,source_vertices)


if __name__=='__main__':unittest.main()
