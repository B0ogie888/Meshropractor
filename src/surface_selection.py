"""Triangle selection independent of rendering; connectivity prevents selection through walls."""
import numpy as np
from trimesh.triangles import closest_point


class SurfaceTopology:
    def __init__(self, mesh):
        self.mesh = mesh
        self.neighbors = [[] for _ in mesh.faces]
        for a, b in mesh.face_adjacency:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)

    def select(self, seed, mode, point=None, angle=5., radius=3.):
        if seed < 0 or seed >= len(self.mesh.faces): return set()
        if mode == 'triangle': return {int(seed)}
        normals = self.mesh.face_normals
        threshold = np.cos(np.deg2rad(angle))
        allowed = np.ones(len(normals), dtype=bool)
        if mode == 'plane':
            distance = np.abs((self.mesh.triangles_center - self.mesh.triangles_center[seed]) @ normals[seed])
            allowed = (normals @ normals[seed] >= threshold) & (distance <= max(self.mesh.scale * 1e-6, 1e-7))
        elif mode == 'brush':
            point = np.asarray(point)
            triangles = self.mesh.triangles
            candidates = np.flatnonzero(np.all(triangles.min(axis=1) <= point + radius, axis=1)
                                        & np.all(triangles.max(axis=1) >= point - radius, axis=1))
            allowed[:] = False
            if len(candidates):
                nearest = closest_point(triangles[candidates], np.tile(point, (len(candidates), 1)))
                allowed[candidates] = np.linalg.norm(nearest - point, axis=1) <= radius
        found, pending = {int(seed)}, [int(seed)]
        while pending:
            current = pending.pop()
            for other in self.neighbors[current]:
                if other in found or not allowed[other]: continue
                if mode in ('smooth', 'brush') and normals[current] @ normals[other] < threshold: continue
                found.add(int(other))
                pending.append(int(other))
        return found
