"""Deterministic, CPU ray-cast support generation in millimetres, with bounded work."""
import numpy as np
import trimesh
import math
from PySide6.QtCore import QThread


def cancelled():
    if QThread.currentThread().isInterruptionRequested():
        raise InterruptedError('Генерация поддержек отменена.')


def overhang_faces(mesh, angle=45.):
    return np.flatnonzero(mesh.face_normals[:, 2] < -np.cos(np.deg2rad(angle)))


def validate(params):
    positive = ('spacing', 'diameter', 'tip_diameter', 'tip_height', 'foot_diameter', 'foot_height')
    if any(not np.isfinite(params[k]) or params[k] <= 0 for k in positive):
        raise ValueError('Размеры поддержек должны быть положительными конечными числами.')
    if not 0 < params['angle'] < 90 or not np.isfinite(params['base_z']):
        raise ValueError('Проверьте угол нависания и высоту платформы.')
    if params['tip_diameter'] > params['diameter']:
        raise ValueError('Диаметр контакта не должен превышать диаметр стойки.')
    if params['foot_diameter'] < params['diameter']:
        raise ValueError('Диаметр основания не должен быть меньше диаметра стойки.')


class RayWorld:
    def __init__(self, records):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.ids = {}
        for record in records:
            mesh = record['mesh']
            gid = self.scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                                           o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
            self.ids[record['row']] = gid
        self.top = max(r['mesh'].bounds[1, 2] for r in records) + 1

    def cast(self, origins, directions, all_hits=False):
        rays = np.column_stack((np.atleast_2d(origins), np.broadcast_to(directions, np.atleast_2d(origins).shape))).astype(np.float32)
        tensor = self.o3d.core.Tensor(rays)
        result = self.scene.list_intersections(tensor) if all_hits else self.scene.cast_rays(tensor)
        return {key: value.numpy() for key, value in result.items()}

    def bottom(self, point, base_z, only_platform=False):
        epsilon = max(1e-4, abs(point[2]) * 2e-7)
        origin = np.asarray(point).copy()
        origin[2] -= epsilon
        result = self.cast(origin, [0, 0, -1])
        distance = result['t_hit'][0]
        z = origin[2] - distance if np.isfinite(distance) else base_z
        if z > base_z + epsilon and only_platform: return None
        bottom = max(float(z), base_z)
        if point[2] - bottom < .05: return None
        return np.array([point[0], point[1], bottom])

    def clear_segment(self, start, end, radius):
        delta = np.asarray(end) - start
        length = np.linalg.norm(delta)
        if length < 1e-5: return False
        direction = delta / length
        reference = [0, 0, 1] if abs(direction[2]) < .9 else [1, 0, 0]
        side = np.cross(direction, reference)
        side /= np.linalg.norm(side)
        other = np.cross(direction, side)
        offsets = np.array([np.zeros(3), side * radius, -side * radius, other * radius, -other * radius])
        hits = self.cast(np.asarray(start) + offsets + direction * 1e-4, direction)['t_hit']
        return bool(np.all(hits >= length - 2e-4))


def contact_grid(record, faces, world, params):
    mesh = record['mesh']
    if len(faces) == 0: return np.empty((0, 3))
    vertices = mesh.triangles[faces].reshape(-1, 3)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    spacing = params['spacing']
    counts = np.maximum(1, np.ceil((high[:2] - low[:2]) / spacing).astype(int))
    if math.prod(map(int, counts)) > 250000:
        raise ValueError('Слишком плотная сетка поддержек. Увеличьте шаг (не более 250 000 лучей на деталь).')
    xy = [low[i] + (np.arange(counts[i]) + .5) * (high[i] - low[i]) / counts[i] for i in range(2)]
    x, y = np.meshgrid(*xy)
    origins = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, world.top)))
    allowed = np.zeros(len(mesh.faces), dtype=bool)
    allowed[faces] = True
    contacts = []
    covered = set()
    for offset in range(0, len(origins), 8192):
        cancelled()
        batch = origins[offset:offset + 8192]
        hits = world.cast(batch, [0, 0, -1], all_hits=True)
        valid = hits['geometry_ids'] == world.ids[record['row']]
        indices = np.flatnonzero(valid)
        indices = indices[allowed[hits['primitive_ids'][indices]]]
        points = batch[hits['ray_ids'][indices]].copy()
        points[:, 2] -= hits['t_hit'][indices]
        contacts.extend(points)
        covered.update(map(int, hits['primitive_ids'][indices]))
        if len(contacts) > 5000: raise ValueError('Более 5000 контактов на деталь. Увеличьте шаг.')
    # A narrow disconnected island can lie between grid rays: give each missed island one contact.
    from surface_selection import SurfaceTopology
    topology = SurfaceTopology(mesh)
    remaining = set(map(int, faces))
    while remaining:
        seed = min(remaining)
        group, pending = {seed}, [seed]
        remaining.remove(seed)
        while pending:
            for neighbor in topology.neighbors[pending.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    pending.append(neighbor)
        if not group.intersection(covered):
            face = max(group, key=lambda i: mesh.area_faces[i])
            contacts.append(mesh.triangles_center[face])
        if len(contacts) > 5000: raise ValueError('Более 5000 контактов на деталь. Увеличьте шаг.')
    return np.unique(np.round(np.asarray(contacts).reshape(-1, 3), 6), axis=0)


def column(bottom, top, params, tapered=False):
    """One watertight surface with foot, shaft and tapered tip, including contact overlap."""
    bottom, top = np.asarray(bottom), np.asarray(top)
    height = float(top[2] - bottom[2])
    if height <= .01: raise ValueError('Недостаточно места для поддержки.')
    tip = min(params['tip_height'], height * .35)
    foot = min(params['foot_height'], height * .2)
    overlap = min(.05, height * .02)
    levels = [(-overlap, params['foot_diameter'] / 2), (foot, params['diameter'] / 2),
              (height - tip, params['diameter'] / 2), (height + overlap, params['tip_diameter'] / 2)]
    if tapered: levels = [levels[0], levels[-1]]
    count = 16
    angles = np.arange(count) * 2 * np.pi / count
    points = np.array([[bottom[0] + radius * np.cos(a), bottom[1] + radius * np.sin(a), bottom[2] + z]
                       for z, radius in levels for a in angles])
    faces = []
    for level in range(len(levels) - 1):
        for i in range(count):
            a, b = level * count + i, level * count + (i + 1) % count
            faces.extend([[a, b, b + count], [a, b + count, a + count]])
    for i in range(1, count - 1):
        faces.extend([[0, i + 1, i], [(len(levels) - 1) * count, (len(levels) - 1) * count + i, (len(levels) - 1) * count + i + 1]])
    return trimesh.Trimesh(points, faces, process=False)


def tree_or_columns(pairs, world, params):
    """Merge nearby platform contacts when all branches have a clear path; otherwise columns."""
    groups = {}
    pieces = []
    for bottom, top in pairs:
        if abs(bottom[2] - params['base_z']) > 1e-3:
            pieces.append(column(bottom, top, params))
            continue
        cell = tuple(np.floor(top[:2] / (params['spacing'] * 3)).astype(int))
        groups.setdefault(cell, []).append((bottom, top))
    for group in groups.values():
        cancelled()
        tops = np.array([top for _, top in group])
        xy = tops[:, :2].mean(axis=0)
        reach = np.linalg.norm(tops[:, :2] - xy, axis=1).max()
        junction = np.r_[xy, tops[:, 2].min() - params['tip_height'] - max(reach, params['tip_height'])]
        base = np.r_[xy, params['base_z']]
        radius = params['diameter'] / 2
        # End the collision test before the tapered contact enters the supported surface.
        ends = [top - np.array([0, 0, params['tip_height']]) for top in tops]
        clear = len(group) > 1 and junction[2] > base[2] + params['foot_height'] * 2
        clear = clear and world.clear_segment(base + [0, 0, .001], junction, radius)
        clear = clear and all(world.clear_segment(junction, end, radius) for end in ends)
        if not clear:
            pieces.extend(column(bottom, top, params) for bottom, top in group)
            continue
        trunk = dict(params, tip_diameter=params['diameter'])
        pieces.append(column(base, junction, trunk))
        for top, end in zip(tops, ends):
            pieces.append(trimesh.creation.cylinder(radius=radius, segment=[junction, end + [0, 0, .02]], sections=16))
            pieces.append(column(end, top, dict(params, foot_diameter=params['diameter'])))
    return pieces


def generate_supports(records, targets, params, branching=False, kind=None):
    validate(params)
    results = []
    worlds = {}
    for record in records:
        if record['row'] not in targets: continue
        cancelled()
        platform = record.get('platform')
        if platform not in worlds:
            worlds[platform] = RayWorld([r for r in records if r.get('platform') == platform])
        world = worlds[platform]
        faces = overhang_faces(record['mesh'], params['angle'])
        if targets[record['row']] is not None: faces = np.intersect1d(faces, targets[record['row']])
        kind = kind or ('Ветвящиеся' if branching else 'Точечные')
        contacts = contact_grid(record, faces, world, params)
        if kind == 'Контур' and len(faces):
            selected = record['mesh'].faces[faces]
            edges = np.sort(np.vstack((selected[:, [0,1]], selected[:, [1,2]], selected[:, [2,0]])), axis=1)
            edges, counts = np.unique(edges, axis=0, return_counts=True)
            points = []
            for edge in edges[counts == 1]:
                a, b = record['mesh'].vertices[edge]
                steps = max(1, int(np.ceil(np.linalg.norm(b - a) / params['spacing'])))
                if len(points) + steps > 5000: raise ValueError('Слишком плотный контур: увеличьте шаг.')
                center = record['mesh'].triangles_center[faces].mean(axis=0)
                points.extend((a + (b-a) * t) * .9999 + center * .0001 for t in np.linspace(0, 1, steps, endpoint=False))
            contacts = np.asarray(points)
        pairs = []
        for point in contacts:
            if point[2] <= params['base_z'] + .05: continue
            bottom = world.bottom(point, params['base_z'], params['only_platform'])
            if bottom is not None: pairs.append((bottom, point))
        if not pairs: continue
        if branching or kind == 'Ветвящиеся':
            pieces = tree_or_columns(pairs, world, params)
        else:
            pieces = []
            for i, (bottom, top) in enumerate(pairs):
                if i % 64 == 0: cancelled()
                if kind == 'Конусы':
                    pieces.append(column(bottom, top, params, tapered=True))
                else: pieces.append(column(bottom, top, params))
            if kind in ('Блок', 'Линии', 'Сеть'):
                from scipy.spatial import cKDTree
                tops = np.asarray([pair[1] for pair in pairs])
                neighbors = cKDTree(tops[:, :2]).query_pairs(params['spacing'] * 1.05)
                for a, b in sorted(neighbors):
                    cancelled()
                    start, end = pairs[a], pairs[b]
                    dx, dy = abs(start[1][0] - end[1][0]), abs(start[1][1] - end[1][1])
                    if np.hypot(dx, dy) < 1e-6: continue
                    if min(dx, dy) > params['spacing'] * .1: continue
                    if kind == 'Линии' and dx < dy: continue
                    low_a, low_b = start[0] + [0,0,params['foot_height']], end[0] + [0,0,params['foot_height']]
                    high_a, high_b = start[1] - [0,0,params['tip_height']], end[1] - [0,0,params['tip_height']]
                    if min(high_a[2]-low_a[2], high_b[2]-low_b[2]) <= 0: continue
                    if not world.clear_segment(high_a, high_b, params['tip_diameter'] / 2): continue
                    if kind == 'Сеть':
                        for p, q in ((low_a, high_b), (low_b, high_a)):
                            if world.clear_segment(p, q, params['tip_diameter']/2):
                                pieces.append(trimesh.creation.cylinder(radius=params['tip_diameter']/2, segment=[p,q], sections=12))
                    else:
                        vector = high_b - high_a
                        side = np.cross(vector, [0,0,1])
                        side /= np.linalg.norm(side)
                        side *= params['tip_diameter'] / 2
                        corners = np.array([low_a, low_b, high_b, high_a])
                        vertices = np.vstack((corners - side, corners + side))
                        wall_faces = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
                        wall = trimesh.Trimesh(vertices, wall_faces, process=False)
                        wall.fix_normals()
                        pieces.append(wall)
        results.append(dict(mesh=trimesh.util.concatenate(pieces), filename=record['filename'],
                            platform=platform, contacts=len(pairs), row=record['row'], surface_faces=faces.tolist(), kind=kind, params=dict(params)))
    return results
