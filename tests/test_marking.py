"""Physical relief, attachment transforms and project round trips."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from marking_geometry import KEY, build_relief, merge_relief, store_plans, plans, mask_contours, silhouette
from cad_state import apply_cad_transform
from project_store import ProjectState, save_project, load_project


class MarkingGeometryTests(unittest.TestCase):
    def setUp(self):
        self.source = trimesh.creation.box([20,20,10])
        self.frame = np.eye(4); self.frame[2,3] = 5
        self.params = dict(frame=self.frame.tolist(), width=10.,area_height=10.,depth=.4,
            resolution=1.,text_size=4.,fit=True,project=True,
            contours=[[[0,0],[4,0],[4,4],[0,4]],[[1,1],[3,1],[3,3],[1,3]]])

    def test_relief_holes_and_emboss_engrave_volumes(self):
        original = self.source.vertices.copy()
        for engrave,expected in ((False,4000+12*.4),(True,4000-12*.4)):
            params = dict(self.params,engrave=engrave)
            mesh = build_relief(self.source,params)
            self.assertTrue(mesh.is_volume)
            merged = merge_relief(self.source,mesh,params)
            self.assertTrue(merged.is_volume)
            self.assertAlmostEqual(merged.volume,expected,delta=.015)
        np.testing.assert_array_equal(self.source.vertices,original)

    def test_projection_on_rotated_and_curved_surface(self):
        transform = trimesh.transformations.rotation_matrix(.7,[1,1,0])
        source = self.source.copy(); source.apply_transform(transform)
        params = dict(self.params,frame=(transform@self.frame).tolist())
        relief = build_relief(source,params)
        self.assertTrue(relief.is_volume)
        local = relief.copy(); local.apply_transform(np.linalg.inv(transform))
        self.assertAlmostEqual(local.bounds[1,2],5.4,places=4)
        sphere = trimesh.creation.icosphere(subdivisions=3,radius=10)
        frame = np.eye(4); frame[2,3] = 10
        relief = build_relief(sphere,dict(self.params,frame=frame.tolist()))
        self.assertTrue(relief.is_volume)
        self.assertGreater(np.ptp(relief.vertices[:,2]),.4)

    def test_through_hole_and_missed_surface_are_explicit(self):
        params = dict(self.params,through=True)
        relief = build_relief(self.source,params)
        merged = merge_relief(self.source,relief,params)
        self.assertAlmostEqual(merged.volume,4000-12*10,delta=.02)
        frame = self.frame.copy(); frame[0,3] = 30
        with self.assertRaises(ValueError): build_relief(self.source,dict(self.params,frame=frame.tolist()))
        with self.assertRaises(InterruptedError): build_relief(self.source,self.params,cancelled=lambda:True)

    def test_plans_survive_project_and_follow_part_transforms(self):
        item = dict(id='a',name='Area',frame=self.frame.tolist(),params=self.params)
        store_plans(self.source,[item])
        transform = np.eye(4); transform[:3,3] = [12,3,7]
        moved = self.source.copy(); apply_cad_transform(moved,transform)
        np.testing.assert_allclose(plans(moved)[0]['frame'],transform@self.frame)
        np.testing.assert_allclose(plans(self.source)[0]['frame'],self.frame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'marking.mrp'
            save_project(path,ProjectState(parts=[dict(mesh=moved,filename='body.stl')]))
            loaded = load_project(path).parts[0]['mesh']
            self.assertEqual(plans(loaded)[0]['id'],'a')
        moved.vertices[0] += 1
        with self.assertRaises(ValueError): plans(moved)

    def test_bitmap_holes_and_datamatrix_quiet_zone(self):
        mask = np.ones((12,12),bool); mask[3:9,3:9] = False
        params = dict(self.params,content='image',contours=mask_contours(mask))
        self.assertEqual(len(silhouette(params).interiors),1)
        from pystrich.datamatrix import DataMatrixData,DataMatrixEncoder
        pixels = np.asarray(DataMatrixEncoder(DataMatrixData('ABC-123',auto_encoding=True)).get_pilimage(cellsize=1).convert('L'))<128
        self.assertFalse(pixels[:2].any()); self.assertFalse(pixels[:,:2].any())
        params = dict(self.params,content='datamatrix',contours=mask_contours(pixels))
        shape = silhouette(params); cell = 10/pixels.shape[1]
        self.assertGreaterEqual(shape.bounds[0],-5+2*cell-1e-8)
        self.assertTrue(build_relief(self.source,params).is_volume)
        with self.assertRaises(ValueError): silhouette(dict(params,shift_x=3))
        with self.assertRaises(ValueError): silhouette(dict(params,circular=True))

    def test_invalid_saved_region_is_rejected_before_opening_editor(self):
        store_plans(self.source,[dict(name='Area',frame=self.frame.tolist(),params=self.params)])
        self.source.metadata[KEY]['items'][0]['params']['width']='broken'
        with self.assertRaises(ValueError): plans(self.source)


if __name__=='__main__': unittest.main()
