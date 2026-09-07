import sys
from pathlib import Path
import unittest
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from surface_selection import SurfaceTopology
from support_geometry import column, generate_supports, overhang_faces, RayWorld, tree_or_columns
from support_tools import DEFAULTS


class WorkspaceGeometryTests(unittest.TestCase):
    def test_plane_and_component_do_not_cross_disconnected_shells(self):
        cube = trimesh.creation.box()
        second = cube.copy()
        second.apply_translation([3, 0, 0])
        mesh = trimesh.util.concatenate([cube, second])
        topology = SurfaceTopology(mesh)
        self.assertEqual(len(topology.select(0, 'triangle')), 1)
        self.assertEqual(len(topology.select(0, 'plane')), 2)
        self.assertEqual(len(topology.select(0, 'smooth', angle=5)), 2)
        self.assertEqual(topology.select(0, 'component'), set(range(12)))

    def test_brush_selects_triangle_intersections_without_crossing_sharp_corner(self):
        mesh = trimesh.creation.box()
        seed = 0
        selected = SurfaceTopology(mesh).select(seed, 'brush', mesh.triangles_center[seed], angle=10, radius=3)
        self.assertEqual(len(selected), 2)
        self.assertTrue(np.all(mesh.face_normals[list(selected)] == mesh.face_normals[seed]))

    def test_column_closed_outward_with_contact_overlap(self):
        mesh = column([2, 3, 0], [2, 3, 10], DEFAULTS)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)
        self.assertLess(mesh.bounds[0, 2], 0)
        self.assertGreater(mesh.bounds[1, 2], 10)

    def records(self):
        cube = trimesh.creation.box([6, 6, 2])
        cube.apply_translation([0, 0, 11])
        return [dict(row=0, mesh=cube, filename='part.stl', platform=None)]

    def test_grid_is_deterministic_and_faces_limit_contacts(self):
        records = self.records()
        result = generate_supports(records, {0: None}, DEFAULTS)
        self.assertEqual(result[0]['contacts'], 4)
        self.assertTrue(result[0]['mesh'].is_watertight)
        other = generate_supports(records, {0: None}, DEFAULTS)
        np.testing.assert_array_equal(result[0]['mesh'].vertices, other[0]['mesh'].vertices)
        self.assertEqual(generate_supports(records, {0: []}, DEFAULTS), [])
        upper = np.flatnonzero(records[0]['mesh'].face_normals[:, 2] > .9)
        self.assertEqual(generate_supports(records, {0: upper}, DEFAULTS), [])

    def test_rays_land_on_obstacle_or_skip_platform_only(self):
        records = self.records()
        blocker = trimesh.creation.box([8, 8, 2])
        blocker.apply_translation([0, 0, 4])
        records.append(dict(row=1, mesh=blocker, filename='blocker.stl', platform=None))
        world = RayWorld(records)
        np.testing.assert_allclose(world.bottom([1, 1, 10], 0), [1, 1, 5], atol=1e-5)
        self.assertIsNone(world.bottom([1, 1, 10], 0, True))
        self.assertEqual(generate_supports(records, {0: None}, dict(DEFAULTS, only_platform=True)), [])
        result = generate_supports(records, {0: None}, DEFAULTS)[0]['mesh']
        self.assertGreater(result.bounds[0, 2], 4.9)

    def test_tree_merges_clear_contacts_and_respects_branch_blockers(self):
        records = self.records()
        world = RayWorld(records)
        pairs = [(np.array([x, y, 0.]), np.array([x, y, 10.])) for x, y in ((1, 1), (2, 1), (1, 2))]
        pieces = tree_or_columns(pairs, world, DEFAULTS)
        self.assertEqual(len(pieces), 7)  # one trunk, three branches and three tapered tips
        self.assertTrue(all(piece.is_watertight for piece in pieces))
        blocker = trimesh.creation.box([6, 6, 1])
        blocker.apply_translation([0, 0, 5])
        records.append(dict(row=1, mesh=blocker))
        blocked = tree_or_columns(pairs, RayWorld(records), DEFAULTS)
        self.assertEqual(len(blocked), 3)

    def test_separate_platforms_do_not_block_each_other(self):
        records = self.records()
        other = records[0]['mesh'].copy()
        other.apply_translation([0, 0, -6])
        records.append(dict(row=1, mesh=other, filename='other.stl', platform='other'))
        result = generate_supports(records, {0: None}, dict(DEFAULTS, only_platform=True))
        self.assertEqual(result[0]['contacts'], 4)

    def test_density_limit_rejects_unbounded_grid(self):
        records = self.records()
        with self.assertRaisesRegex(ValueError, '250'):
            generate_supports(records, {0: None}, dict(DEFAULTS, spacing=.001))

    def test_support_patterns_have_distinct_closed_geometry(self):
        records = self.records()
        counts = []
        for kind in ('Точечные', 'Блок', 'Линии', 'Сеть', 'Контур', 'Конусы'):
            result = generate_supports(records, {0: None}, DEFAULTS, kind=kind)[0]
            self.assertTrue(result['mesh'].is_watertight, kind)
            self.assertGreater(result['mesh'].volume, 0, kind)
            self.assertEqual(result['row'], 0)
            counts.append(len(result['mesh'].faces))
        self.assertEqual(len(set(counts)), len(counts))

    def test_block_handles_stacked_contacts_at_the_same_xy(self):
        records = self.records()
        upper = records[0]['mesh'].copy()
        upper.apply_translation([0,0,7])
        records[0]['mesh'] = trimesh.util.concatenate([records[0]['mesh'], upper])
        mesh = generate_supports(records, {0: None}, DEFAULTS, kind='Блок')[0]['mesh']
        self.assertTrue(np.isfinite(mesh.vertices).all())
        self.assertTrue(mesh.is_watertight)


if __name__ == '__main__': unittest.main()
