"""Transfer an example's orientation using deterministic surface registration.

Similarity is checked on area samples and mesh vertices in both directions.
It is a geometric approximation, not proof that two complete solids are equal.
No scaling, reflection, support geometry or mutation of input meshes is used.
"""
from itertools import permutations, product

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from alignment import rigid_fit, surface_points
from project_store import validate_mesh


MAX_INPUT_GEOMETRY_BYTES = 512 * 1024 ** 2


def transfer_orientations(meshes, reference_index=0, tolerance_mm=.1,
                          progress=lambda message: None, cancelled=lambda: False):
    """Return rigid matrices in input order and a sampled-similarity report.

    The reference and unmatched parts receive identity. For a matched part the
    final bounding-box center stays at its original position, while its shape
    takes the reference orientation. ``report['matches']`` has one dict per part
    with ``index``, ``matched``, ``reason`` and, after registration, bidirectional
    distance metrics. The routine supports at most 30 parts and 512 MiB of input
    vertex/face arrays, reference included. Unreferenced vertices are ignored.
    """
    def check():
        if cancelled():
            raise InterruptedError('Перенос ориентации отменён.')

    check()
    meshes = list(meshes)
    if not meshes or len(meshes) > 30:
        raise ValueError('Выберите от 1 до 30 деталей, включая образец.')
    if (isinstance(reference_index, (bool, np.bool_))
            or not isinstance(reference_index, (int, np.integer))
            or not 0 <= reference_index < len(meshes)):
        raise ValueError('Индекс образца вне списка деталей.')
    if isinstance(tolerance_mm, (bool, np.bool_)):
        raise ValueError('Допуск должен быть положительным конечным числом в мм.')
    try:
        tolerance = float(tolerance_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError('Допуск должен быть положительным конечным числом в мм.') from exc
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Допуск должен быть положительным конечным числом в мм.')
    geometry_bytes = 0
    for mesh in meshes:
        check()
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError('Модель должна содержать вершины и треугольники.')
        geometry_bytes += mesh.vertices.nbytes + mesh.faces.nbytes
    # Even validation can allocate triangle/area caches. Check the complete input
    # first, before validation, copies, surface samples or native raycasting trees.
    if geometry_bytes > MAX_INPUT_GEOMETRY_BYTES:
        raise ValueError('Слишком большой объём геометрии для сравнения форм '
                         '(максимум 512 МиБ). Выберите меньше деталей или упростите сетки.')
    surfaces = []
    for mesh in meshes:
        check()
        validate_mesh(mesh)
        # Deleting faces deliberately preserves original vertex IDs. Such loose
        # vertices are not part of the surface and must not affect matching,
        # dimensions or the final placement center. Compact only private copies.
        surface = mesh.copy()
        surface.remove_unreferenced_vertices()
        surfaces.append(surface)
    meshes = surfaces

    reference = meshes[reference_index]
    center = reference.bounds.mean(axis=0)
    scale = float(np.linalg.norm(reference.extents))
    epsilon = scale * 2e-7
    limit = tolerance + epsilon
    sample_count = 6000

    def scene_for(mesh, origin):
        check()
        scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        scene.add_triangles(o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(mesh.vertices - origin, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int64))))
        return scene

    def nearest(scene, points):
        check()
        answer = scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(points, dtype=np.float32)), nthreads=2)
        positions = answer['points'].numpy().astype(float)
        normals = answer['primitive_normals'].numpy().astype(float)
        return positions, normals, np.linalg.norm(points - positions, axis=1)

    def vertex_sample(mesh, origin):
        ids = np.linspace(0, len(mesh.vertices) - 1, min(20000, len(mesh.vertices)), dtype=int)
        return mesh.vertices[ids] - origin

    target_scene = scene_for(reference, center)
    target_points = surface_points(reference, sample_count, seed=42) - center
    target_vertices = vertex_sample(reference, center)
    target_axes = np.linalg.eigh(np.cov(target_points.T))[1]
    matrices = [np.eye(4) for _ in meshes]
    report = dict(reference_index=int(reference_index), tolerance_mm=tolerance,
                  samples_per_direction=sample_count, method='sampled_bidirectional_surface',
                  matches=[], warnings=[])

    def distances_report(scene, area_points, vertices):
        distances = nearest(scene, area_points)[2]
        vertex_distances = nearest(scene, vertices)[2]
        return dict(coverage=float(np.mean(distances <= limit)),
                    rmse_mm=float(np.sqrt(np.mean(distances ** 2))),
                    p95_mm=float(np.percentile(distances, 95)),
                    max_mm=float(max(distances.max(), vertex_distances.max())),
                    vertex_samples=len(vertices))

    def refine(transform, points, radius, iterations, point_to_point=False):
        transform = transform.copy()
        for _ in range(iterations):
            check()
            p = points @ transform[:3, :3].T + transform[:3, 3]
            q, normals, distance = nearest(target_scene, p)
            keep = distance <= radius
            if np.count_nonzero(keep) < 12:
                break
            p, q, normals = p[keep], q[keep], normals[keep]
            if point_to_point:
                delta = rigid_fit(p, q)
            else:
                origin = p.mean(axis=0)
                a = np.column_stack((np.cross((p - origin) / scale, normals), normals))
                b = np.einsum('ij,ij->i', q - p, normals)
                step = np.linalg.lstsq(a, b, rcond=1e-7)[0]
                angle = np.linalg.norm(step[:3]) / scale
                if angle > .2:
                    step *= .2 / angle
                delta = np.eye(4)
                delta[:3, :3] = Rotation.from_rotvec(step[:3] / scale).as_matrix()
                delta[:3, 3] = origin + step[3:] - delta[:3, :3] @ origin
            candidate = delta @ transform
            after = nearest(target_scene, points @ candidate[:3, :3].T + candidate[:3, 3])[2]
            if np.mean(np.minimum(after, radius) ** 2) > np.mean(np.minimum(distance, radius) ** 2) * (1 + 1e-7):
                break
            transform = candidate
            if np.linalg.norm(delta[:3, :3] - np.eye(3)) < 1e-8 and np.linalg.norm(delta[:3, 3]) < scale * 1e-9:
                break
        return transform

    for index, mesh in enumerate(meshes):
        check()
        entry = dict(index=index, matched=index == reference_index,
                     reason='Образец' if index == reference_index else '', rotation_degrees=0.)
        report['matches'].append(entry)
        if index == reference_index:
            continue
        progress(f'Сравнение формы детали {index + 1}/{len(meshes)} с образцом…')
        max_area = max(mesh.area, reference.area)
        # Conservative, rotation-invariant rejection. Final acceptance still
        # requires both surfaces and aligned dimensions to fit the mm tolerance.
        area_allowance = .02 * max_area + 8 * tolerance * np.sqrt(max_area) + 24 * tolerance ** 2
        if abs(mesh.area - reference.area) > area_allowance:
            entry['reason'] = 'Площадь поверхности отличается: другая форма или размер.'
            continue
        source_center = mesh.bounds.mean(axis=0)
        source_points = surface_points(mesh, sample_count, seed=42) - source_center
        source_vertices = vertex_sample(mesh, source_center)
        source_scene = scene_for(mesh, source_center)

        def verify(transform):
            rotation, offset = transform[:3, :3], transform[:3, 3]
            forward = distances_report(target_scene, source_points @ rotation.T + offset,
                                       source_vertices @ rotation.T + offset)
            reverse = distances_report(source_scene, (target_points - offset) @ rotation,
                                       (target_vertices - offset) @ rotation)
            rotated_vertices = (mesh.vertices - source_center) @ rotation.T
            size_error = float(np.max(np.abs(np.ptp(rotated_vertices, axis=0) - reference.extents)))
            matched = max(forward['max_mm'], reverse['max_mm']) <= limit and size_error <= 2 * limit
            return matched, dict(forward=forward, reverse=reverse, size_error_mm=size_error)

        candidates = [np.eye(4)]
        if mesh.vertices.shape == reference.vertices.shape and np.array_equal(mesh.faces, reference.faces):
            # Exact vertex correspondence is only a candidate, never an acceptance shortcut.
            candidates.insert(0, rigid_fit(mesh.vertices - source_center, reference.vertices - center))
        found, best = None, None
        for candidate in candidates:
            matched, quality = verify(candidate)
            if matched:
                found, best = candidate, quality
                break
        if found is None:
            source_axes = np.linalg.eigh(np.cov(source_points.T))[1]
            for order in permutations(range(3)):
                for signs in product((-1, 1), repeat=3):
                    rotation = target_axes[:, order] @ np.diag(signs) @ source_axes.T
                    if np.linalg.det(rotation) > 0:
                        candidate = np.eye(4)
                        candidate[:3, :3] = rotation
                        candidate[:3, 3] = target_points.mean(axis=0) - rotation @ source_points.mean(axis=0)
                        candidates.append(candidate)
            ranked = []
            coarse = source_points[:700]
            for number, candidate in enumerate(candidates):
                check()
                candidate = refine(candidate, coarse, max(scale * .2, limit), 6, point_to_point=True)
                candidate = refine(candidate, coarse, max(scale * .06, limit), 12)
                distance = nearest(target_scene, coarse @ candidate[:3, :3].T + candidate[:3, 3])[2]
                ranked.append((float(np.mean(np.minimum(distance, scale * .1) ** 2)), number, candidate))
            ranked.sort(key=lambda item: item[:2])
            for _, _, candidate in ranked[:4]:
                for radius in (max(scale * .03, limit), max(scale * .005, limit), limit):
                    candidate = refine(candidate, source_points, radius, 25)
                matched, quality = verify(candidate)
                if best is None or max(quality['forward']['max_mm'], quality['reverse']['max_mm']) < max(best['forward']['max_mm'], best['reverse']['max_mm']):
                    best = quality
                if matched:
                    found, best = candidate, quality
                    break
        if best is not None:
            entry.update(best)
        if found is None:
            entry['reason'] = 'Двусторонняя проверка поверхности не уложилась в допуск.'
            continue
        rotation = found[:3, :3]
        # Preserve the final AABB center, not only the rotation pivot: they differ
        # when an asymmetric object is reoriented.
        rotated = (mesh.vertices - source_center) @ rotation.T
        rotated_center = (rotated.min(axis=0) + rotated.max(axis=0)) * .5
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = source_center - rotation @ source_center - rotated_center
        matrices[index] = matrix
        entry.update(matched=True, reason='Поверхности совпадают в заданном допуске.',
                     rotation_degrees=float(np.rad2deg(Rotation.from_matrix(rotation).magnitude())))
    unmatched = [item['index'] + 1 for item in report['matches'] if not item['matched']]
    if unmatched:
        report['warnings'].append('Оставлены без изменений: детали ' + ', '.join(map(str, unmatched)) + '.')
    report['warnings'].append('Сходство проверено по выборкам поверхности и вершинам; это не доказательство полной идентичности тел. Симметричные детали могут иметь несколько равноправных ориентаций.')
    check()
    return matrices, report
