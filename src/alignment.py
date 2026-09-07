"""Deterministic, area-sampled rigid registration against CAD triangles."""
from itertools import permutations, product

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from project_store import validate_mesh


def surface_points(mesh, count, seed=42):
    rng = np.random.default_rng(seed)
    triangles = mesh.triangles[rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)]
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])


def rigid_fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    transform = np.eye(4)
    transform[:3, :3] = vt.T @ correction @ u.T
    transform[:3, 3] = b - transform[:3, :3] @ a
    return transform


def align_surfaces(cad, scan, cad_markers=(), scan_markers=(), settings=None,
                   log=lambda message: None, cancelled=lambda: False):
    """Return an aligned copy and quality data; never mutate either input mesh.

    Coverage is the fraction of sampled *scan area* within the capture tolerance.
    Distances and RMSE are to triangle interiors, not nearest CAD vertices.
    """
    settings = settings or {}
    validate_mesh(cad)
    validate_mesh(scan)
    def check():
        if cancelled():
            raise InterruptedError("Совмещение отменено.")
    check()
    scale = float(np.max(cad.extents))
    mode = int(settings.get("search_time", 1))
    count = int(settings.get("samples", (12000, 30000, 60000)[mode]))
    tolerance = float(settings.get("tolerance", 0)) or scale * .002
    minimum = float(settings.get("min_fitness", .3))
    if count < 100 or not np.isfinite(tolerance) or tolerance <= 0 or not 0 < minimum <= 1:
        raise ValueError("Некорректные параметры совмещения.")
    # Work near the origin: ray queries are float32, while transforms stay float64.
    target_center, source_center = cad.bounds.mean(axis=0), scan.bounds.mean(axis=0)
    source = surface_points(scan, count) - source_center
    target = surface_points(cad, count) - target_center
    triangle_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(cad.vertices - target_center, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(cad.faces, dtype=np.int64)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(triangle_mesh)

    def closest(points):
        check()
        result = scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)), nthreads=2)
        q = result["points"].numpy().astype(float)
        n = result["primitive_normals"].numpy().astype(float)
        return q, n, np.linalg.norm(points - q, axis=1)

    def score(matrix, points, radius):
        _, _, distances = closest(points @ matrix[:3, :3].T + matrix[:3, 3])
        # Penalize unmatched area: a tiny perfect patch cannot beat a full match.
        return float(np.mean(np.minimum(distances, radius) ** 2))

    weak_geometry = False
    def refine(matrix, points, radius, iterations, point_to_point=False):
        nonlocal weak_geometry
        matrix = matrix.copy()
        for _ in range(iterations):
            check()
            p = points @ matrix[:3, :3].T + matrix[:3, 3]
            q, n, distance = closest(p)
            keep = distance < radius
            if np.count_nonzero(keep) < 12:
                break
            p, q, n, d = p[keep], q[keep], n[keep], distance[keep]
            if point_to_point:
                delta = rigid_fit(p, q)
            else:
                center = p.mean(axis=0)
                # Scale rotation columns so planar degeneracy has a meaningful rank.
                a = np.column_stack((np.cross((p - center) / scale, n), n))
                b = np.einsum("ij,ij->i", q - p, n)
                weight = np.maximum(0, 1 - (d / radius) ** 2)
                step, _, rank, _ = np.linalg.lstsq(a * weight[:, None], b * weight, rcond=1e-7)
                weak_geometry |= rank < 6
                omega = step[:3] / scale
                length = np.linalg.norm(omega)
                if length > .2:
                    step *= .2 / length
                    omega = step[:3] / scale
                delta = np.eye(4)
                delta[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
                delta[:3, 3] = center + step[3:] - delta[:3, :3] @ center
            candidate = delta @ matrix
            # Avoid divergence near edges or when a coarse orientation is wrong.
            if score(candidate, points, radius) > score(matrix, points, radius) * (1 + 1e-7):
                break
            matrix = candidate
            if np.linalg.norm(delta[:3, :3] - np.eye(3)) < 1e-7 and np.linalg.norm(delta[:3, 3]) < scale * 1e-8:
                break
        return matrix

    marked = bool(len(cad_markers) or len(scan_markers))
    ambiguous = False
    if marked:
        cm, sm = np.asarray(cad_markers, dtype=float), np.asarray(scan_markers, dtype=float)
        if cm.shape != sm.shape or cm.ndim != 2 or cm.shape[1] != 3 or len(cm) < 3 or not np.isfinite(cm).all() or not np.isfinite(sm).all():
            raise ValueError("Нужно не менее трёх пар конечных координат маркеров.")
        if min(np.linalg.matrix_rank(cm - cm.mean(axis=0)), np.linalg.matrix_rank(sm - sm.mean(axis=0))) < 2:
            raise ValueError("Маркеры не должны лежать на одной прямой.")
        log("Начальное положение — по парным маркерам.")
        transform = rigid_fit(sm - source_center, cm - target_center)
    else:
        log("Поиск ориентации: равномерные выборки по площади поверхности.")
        candidates = [np.eye(4)]
        # Preserve a good existing placement as well as trying centered models.
        placed = np.eye(4)
        placed[:3, 3] = source_center - target_center
        candidates.append(placed)
        if mode > 0:
            _, target_axes = np.linalg.eigh(np.cov(target.T))
            _, source_axes = np.linalg.eigh(np.cov(source.T))
            for order in permutations(range(3)):
                for signs in product((-1, 1), repeat=3):
                    rotation = target_axes[:, order] @ np.diag(signs) @ source_axes.T
                    if np.linalg.det(rotation) > 0:
                        matrix = np.eye(4)
                        matrix[:3, :3] = rotation
                        matrix[:3, 3] = target.mean(axis=0) - rotation @ source.mean(axis=0)
                        candidates.append(matrix)
            # Local shape features supply candidates when a partial scan has different PCA axes.
            check()
            voxel = scale * (.04 if mode == 1 else .025)
            def features(points):
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(voxel)
                cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
                feature = o3d.pipelines.registration.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
                return cloud, feature
            sc, sf = features(source)
            tc, tf = features(target)
            if min(len(sc.points), len(tc.points)) >= 10:
                try:
                    global_fit = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                        sc, tc, sf, tf, o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=voxel * 2))
                    if np.isfinite(global_fit.transformation).all():
                        candidates.append(global_fit.transformation.copy())
                except RuntimeError as exc:
                    log(f"Поиск по признакам пропущен: {exc}")
        coarse = source[:min(2000, len(source))]
        ranked = []
        for index, candidate in enumerate(candidates):
            check()
            candidate = refine(candidate, coarse, scale * .15, 8, point_to_point=True)
            candidate = refine(candidate, coarse, scale * .08, 15)
            ranked.append((score(candidate, coarse, scale * .04), candidate))
            if index % 8 == 0:
                log(f"Проверены ориентации: {index + 1}/{len(candidates)}")
        ranked.sort(key=lambda pair: pair[0])
        transform = ranked[0][1]
        for candidate_score, candidate in ranked[1:]:
            if np.linalg.norm(candidate[:3, :3] - transform[:3, :3]) > .2:
                ambiguous = candidate_score <= ranked[0][0] * 1.1 + (scale * 1e-5) ** 2
                break
    weak_geometry = False
    if settings.get("do_icp", True):
        for radius in (max(scale * .08, tolerance), max(scale * .02, tolerance), max(scale * .005, tolerance), tolerance):
            log(f"Подгонка к треугольникам CAD, допуск захвата {radius:.4g} мм…")
            transform = refine(transform, source, radius, 40 if mode < 2 else 65)
    _, _, distances = closest(source @ transform[:3, :3].T + transform[:3, 3])
    inliers = distances <= tolerance
    coverage = float(inliers.mean())
    if coverage < minimum:
        raise ValueError(f"Совмещение не принято: {coverage:.1%} площади скана в допуске {tolerance:.4g} мм (нужно {minimum:.0%}). Проверьте маркеры и допуск.")
    metrics = dict(rmse=float(np.sqrt(np.mean(distances[inliers] ** 2))), coverage=coverage,
                   median=float(np.median(distances)), p95=float(np.percentile(distances, 95)),
                   tolerance=tolerance, samples=count, ambiguous=ambiguous, weak_geometry=weak_geometry)
    world = transform.copy()
    world[:3, 3] += target_center - world[:3, :3] @ source_center
    metrics["transform"] = world.tolist()
    result = scan.copy()
    result.apply_transform(world)
    result.metadata["alignment"] = metrics
    if ambiguous:
        log("[!] Несколько ориентаций дают близкий результат. Для симметричной детали уточните положение маркерами.")
    if weak_geometry:
        log("[!] Геометрия слабо ограничивает часть перемещений; проверьте положение по маркерам.")
    log(f"RMSE: {metrics['rmse']:.5f} мм; площадь скана в допуске: {coverage:.1%}; P95: {metrics['p95']:.5f} мм.")
    check()
    return result, metrics
