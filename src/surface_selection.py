"""Triangle selection independent of rendering; connectivity prevents selection through walls."""
import numpy as np
from trimesh.triangles import closest_point


class SurfaceTopology:
    def __init__(self, mesh):
        self.mesh = mesh
        self._neighbors = None
        self._triangle_bounds = None

    @property
    def neighbors(self):
        if self._neighbors is None:
            # CSR uses compact integer arrays, avoiding millions of Python lists.
            from scipy.sparse import csr_matrix
            pairs = self.mesh.face_adjacency
            row = np.concatenate((pairs[:, 0], pairs[:, 1]))
            col = np.concatenate((pairs[:, 1], pairs[:, 0]))
            self._neighbors = csr_matrix((np.ones(len(row), dtype=np.uint8), (row, col)),
                                         shape=(len(self.mesh.faces), len(self.mesh.faces)))
        return self._neighbors

    def neighbor_ids(self, face):
        graph = self.neighbors
        return graph.indices[graph.indptr[face]:graph.indptr[face+1]]

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
            if self._triangle_bounds is None:
                self._triangle_bounds = triangles.min(axis=1), triangles.max(axis=1)
            low, high = self._triangle_bounds
            candidates = np.flatnonzero(np.all(low <= point + radius, axis=1)
                                        & np.all(high >= point - radius, axis=1))
            allowed[:] = False
            if len(candidates):
                nearest = closest_point(triangles[candidates], np.tile(point, (len(candidates), 1)))
                allowed[candidates] = np.linalg.norm(nearest - point, axis=1) <= radius
        found, pending = {int(seed)}, [int(seed)]
        neighbors = self.neighbors
        while pending:
            current = pending.pop()
            for other in neighbors.indices[neighbors.indptr[current]:neighbors.indptr[current+1]]:
                if other in found or not allowed[other]: continue
                if mode in ('smooth', 'brush') and normals[current] @ normals[other] < threshold: continue
                found.add(int(other))
                pending.append(int(other))
        return found
