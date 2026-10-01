"""Repair-wizard diagnostics for the original, unmodified indexed mesh."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from mesh_repair import inspect_mesh
from mesh_intersections import find_intersections


def diagnose_mesh(mesh, full=True, progress=lambda message: None, cancelled=lambda: False):
    if cancelled(): raise InterruptedError('Диагностика отменена')
    progress('Диагностика рёбер, нормалей и фрагментов…')
    report = inspect_mesh(mesh)
    edges, inverse = mesh.edges_unique, mesh.edges_unique_inverse
    counts = np.bincount(inverse)
    boundary = edges[counts == 1]
    if len(boundary):
        vertices, reduced = np.unique(boundary, return_inverse=True)
        pairs = reduced.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
        n, labels = connected_components(graph, directed=False)
        degree = np.bincount(reduced.ravel(), minlength=len(vertices))
        bad = np.unique(labels[degree != 2])
        report['contours'] = int(n-len(bad))
        report['branched_boundaries'] = int(len(bad))
    else:
        report.update(contours=0, branched_boundaries=0)
    # Vertex connectivity includes faces incident to nonmanifold edges too.
    graph = coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(len(mesh.vertices), len(mesh.vertices))).tocsr()
    _, labels = connected_components(graph, directed=False)
    face_labels = labels[mesh.faces[:, 0]]
    _, face_counts = np.unique(face_labels, return_counts=True)
    report['components'] = len(face_counts)
    report['noise_components'] = int(np.count_nonzero(face_counts < max(4, len(mesh.faces)*1e-5)))
    # Isolated vertices aren't fragments of a surface.
    report['unreferenced'] = int(len(mesh.vertices)-len(np.unique(mesh.faces)))
    if cancelled(): raise InterruptedError('Диагностика отменена')
    progress('Проверка связности окрестностей вершин…')
    import open3d as o3d
    native = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                      o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))
    report['nonmanifold_vertices'] = len(native.get_non_manifold_vertices())
    del native
    report['negative_volume'] = bool(report['closed'] and report['winding'] and mesh.volume <= 0)
    report['full'] = bool(full)
    report['overlaps'] = report['intersections'] = None
    if full:
        report.update(find_intersections(mesh, progress, cancelled))
    report['clean'] = all(report[k] == 0 for k in ('boundary', 'nonmanifold', 'nonmanifold_vertices', 'duplicates', 'degenerate',
                                                   'overlaps', 'intersections', 'unreferenced')) and report['winding'] and not report['negative_volume']
    return report


def serializable_report(report):
    return {key: value for key, value in report.items() if not isinstance(value, np.ndarray)}
