"""Rigid placement and bounded, conservative box packing without GUI objects.

Matrices map the original world coordinates to the proposed world coordinates.
Platform XY is centred at zero, with build height 0..dim[2]. Edge margins apply
in XY; clearance raises the bottom above Z=0. No operation scales a CAD model.
Packing uses occupied AABBs (including supplied supports), not exact nesting.
"""
from itertools import permutations, product
import numpy as np
import trimesh
from scipy.spatial import QhullError


MAX_PARTS = 200
MAX_GEOMETRY_BYTES = 512 * 1024**2
MAX_HULL_POINTS = 2048
MAX_FREE_BOXES = 4096
MAX_SEARCH_STEPS = 1_000_000
_EPS = 1e-8


def _check(cancelled):
    if cancelled():
        raise InterruptedError('Размещение отменено; исходные детали сохранены.')


def _number(value, name, minimum=0., maximum=1e9):
    if (isinstance(value, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f'{name}: требуется конечное число от {minimum:g} до {maximum:g}.')
    return float(value)


def _integer(value, name, minimum, maximum):
    result = _number(value, name, minimum, maximum)
    if int(result) != result:
        raise ValueError(f'{name}: требуется целое число.')
    return int(result)


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name}: требуется логическое значение.')
    return bool(value)


def _points(mesh):
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not len(mesh.vertices):
        raise ValueError('Для размещения требуется непустая треугольная сетка.')
    if mesh.vertices.nbytes + mesh.faces.nbytes > MAX_GEOMETRY_BYTES:
        raise ValueError('Слишком большая геометрия для расчёта размещения (более 512 МБ).')
    if (mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or mesh.faces.min() < 0
            or mesh.faces.max() >= len(mesh.vertices) or not np.isfinite(mesh.vertices).all()):
        raise ValueError('Геометрия содержит некорректные координаты или индексы.')
    return np.asarray(mesh.vertices)[mesh.referenced_vertices]


def _meshes(values, name, *, empty=False):
    result = list(values)
    if (not empty and not result) or len(result) > MAX_PARTS:
        raise ValueError(f'{name}: выберите от {0 if empty else 1} до {MAX_PARTS} деталей.')
    if sum(m.vertices.nbytes + m.faces.nbytes for m in result if isinstance(m, trimesh.Trimesh)) > MAX_GEOMETRY_BYTES:
        raise ValueError('Суммарная геометрия превышает бюджет 512 МБ.')
    return result, [_points(mesh) for mesh in result]


def _bounds(points, rotation=None, center=None, cancelled=lambda: False):
    rotation = np.eye(3) if rotation is None else rotation
    center = np.zeros(3) if center is None else center
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for start in range(0, len(points), 100_000):
        _check(cancelled)
        cloud = (points[start:start + 100_000] - center) @ rotation.T
        lower = np.minimum(lower, cloud.min(axis=0))
        upper = np.maximum(upper, cloud.max(axis=0))
    return np.array([lower, upper])


def _matrix(rotation, center, displacement=None):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = center - rotation @ center
    if displacement is not None:
        result[:3, 3] += displacement
    return result


def _orthogonal_rotations():
    result = []
    for axes in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            rotation = np.eye(3)[list(axes)] * np.asarray(signs)[:, None]
            if np.linalg.det(rotation) > .5:
                result.append(rotation)
    result.sort(key=lambda value: float(np.linalg.norm(value - np.eye(3))))
    return result


def _cloud(points):
    ids = np.linspace(0, len(points) - 1, min(len(points), MAX_HULL_POINTS), dtype=int)
    ids = np.unique(np.concatenate((ids, points.argmin(axis=0), points.argmax(axis=0))))
    return points[ids]


def _obb_rotation(points, angle_digits, cancelled):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    cloud = _cloud(points) - center
    scale = float(np.ptp(cloud, axis=0).max())
    if scale <= np.finfo(float).tiny:
        raise ValueError('У детали нет ненулевого размера.')
    _check(cancelled)
    try:
        transform, _ = trimesh.bounds.oriented_bounds(cloud / scale, angle_digits=angle_digits)
    except (ValueError, np.linalg.LinAlgError, QhullError) as exc:
        raise ValueError('Не удалось определить ориентированные габариты детали.') from exc
    _check(cancelled)
    rotation = transform[:3, :3].copy()
    if np.linalg.det(rotation) < 0:
        rotation[0] *= -1
    return rotation


def _metrics(mesh, bounds, rotation, angle):
    extents = bounds[1] - bounds[0]
    downward = mesh.face_normals @ rotation[2] < -np.cos(np.deg2rad(angle))
    return dict(height_mm=float(extents[2]), footprint_mm2=float(np.prod(extents[:2])),
                overhang_area_mm2=float(mesh.area_faces[downward].sum()),
                bbox_volume_mm3=float(np.prod(extents)))


def minimum_oriented_bounds(mesh, *, angle_digits=1, progress=lambda message: None,
                            cancelled=lambda: False):
    """Return a searched OBB orientation, not a certified global minimum.

    A bounded vertex sample drives the hull search; final extents always use all
    referenced vertices. The box centre stays at the original AABB centre.
    """
    _check(cancelled)
    precision = _integer(angle_digits, 'Точность поиска', 0, 2)
    points = _points(mesh)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    progress('Поиск ориентированного габаритного параллелепипеда…')
    rotation = _obb_rotation(points, precision, cancelled)
    local = _bounds(points, rotation, center, cancelled)
    displacement = -local.mean(axis=0)
    bounds = local + center + displacement
    matrix = _matrix(rotation, center, displacement)
    return matrix, dict(bounds=bounds, extents=bounds[1] - bounds[0], approximate=True,
                        method='convex_hull_orientation_search',
                        warnings=['Минимальные габариты найдены ограниченным поиском ориентаций; глобальный минимум не гарантирован.'],
                        **_metrics(mesh, bounds, rotation, 45.))


def orientation_candidates(mesh, *, objective='height', overhang_angle_deg=45., max_candidates=48,
                           progress=lambda message: None, cancelled=lambda: False):
    _check(cancelled)
    keys = dict(height='height_mm', footprint='footprint_mm2', support='overhang_area_mm2', bbox='bbox_volume_mm3')
    if objective not in keys:
        raise ValueError('Неизвестная цель оптимизации ориентации.')
    angle = _number(overhang_angle_deg, 'Угол нависания', 0., 90.)
    limit = _integer(max_candidates, 'Число вариантов', 1, 128)
    points = _points(mesh)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    progress('Подготовка вариантов ориентации…')
    obb = _obb_rotation(points, 1, cancelled)
    rotations = [(np.eye(3), 'Исходная ориентация')]
    rotations.extend((rotation @ obb, f'Габариты: вариант {i + 1}')
                     for i, rotation in enumerate(_orthogonal_rotations()))
    areas = mesh.area_faces
    count = min(24, len(areas))
    largest = np.argpartition(areas, len(areas) - count)[-count:]
    for face in largest[np.argsort(areas[largest])[::-1]]:
        normal = mesh.face_normals[face]
        if np.linalg.norm(normal) > .9:
            rotation = trimesh.geometry.align_vectors(normal, [0., 0., -1.])[:3, :3]
            rotations.append((rotation, f'На грань {int(face) + 1}'))
    rotations.extend((rotation, f'Оси: вариант {i + 1}') for i, rotation in enumerate(_orthogonal_rotations()))
    result, seen = [], set()
    for rotation, label in rotations:
        _check(cancelled)
        key = tuple(np.round(rotation, 8).ravel())
        if key in seen:
            continue
        seen.add(key)
        bounds = _bounds(points, rotation, center, cancelled) + center
        metrics = _metrics(mesh, bounds, rotation, angle)
        result.append(dict(matrix=_matrix(rotation, center), bounds=bounds, extents=bounds[1] - bounds[0],
                           label=label, score=metrics[keys[objective]], **metrics))
        progress(f'Оценка ориентаций: {len(result)}/{min(limit, len(rotations))}')
        if len(result) >= limit:
            break
    result.sort(key=lambda item: (item['score'], item['height_mm'], item['footprint_mm2']))
    return result


def optimize_orientation(mesh, **parameters):
    candidates = orientation_candidates(mesh, **parameters)
    best = candidates[0]
    report = {key: value for key, value in best.items() if key != 'matrix'}
    report.update(candidates_tested=len(candidates), approximate=True,
                  warnings=['Ориентация выбрана из ограниченного набора. Площадь нависаний — оценка по нормалям, без расчёта поддержек.'])
    return best['matrix'].copy(), report


def _platform(platform, margin, clearance):
    if not isinstance(platform, dict):
        raise ValueError('Выберите платформу построения.')
    dim = np.asarray(platform.get('dim', []), dtype=float)
    if dim.shape != (3,) or not np.isfinite(dim).all() or np.any(dim <= 0):
        raise ValueError('Габариты платформы должны быть тремя положительными числами.')
    bounds = np.array([[-dim[0] / 2 + margin, -dim[1] / 2 + margin, clearance],
                       [dim[0] / 2 - margin, dim[1] / 2 - margin, dim[2]]])
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError('Отступы не оставляют места внутри платформы.')
    zones = []
    if platform.get('use_zones', False):
        for zone in platform.get('zones', []):
            if not isinstance(zone, dict):
                raise ValueError('Некорректная запретная зона платформы.')
            x, y = (_number(zone.get(axis, 0), f'Зона {axis}', -1e9, 1e9) for axis in ('x', 'y'))
            radius = _number(zone.get('r', 5), 'Размер зоны', 1e-9, 1e9)
            if zone.get('full_h', False):
                low, high = 0., float(dim[2])
            else:
                low = _number(zone.get('zmin', 0), 'Зона Zmin', -1e9, 1e9)
                high = _number(zone.get('zmax', 0), 'Зона Zmax', -1e9, 1e9)
                if high < low:
                    raise ValueError('В запретной зоне Zmax меньше Zmin.')
                if high - low < .001:
                    midpoint = (low + high) / 2
                    low, high = midpoint - .0005, midpoint + .0005
            zones.append(np.array([[x - radius, y - radius, low], [x + radius, y + radius, high]]))
    return bounds, zones


def _overlap(a, b):
    return bool(np.all(a[0] < b[1] - _EPS) and np.all(b[0] < a[1] - _EPS))


def _prune(boxes, cancelled):
    if len(boxes) > MAX_FREE_BOXES * 2:
        raise ValueError('Не удалось найти размещение в пределах бюджета свободных областей. Уменьшите число деталей или зон.')
    if not boxes:
        return []
    array = np.asarray(boxes)
    order = np.argsort(np.prod(array[:, 1] - array[:, 0], axis=1))[::-1]
    kept = []
    for index in order:
        _check(cancelled)
        if kept:
            previous = array[kept]
            if np.any(np.all(previous[:, 0] <= array[index, 0] + _EPS, axis=1)
                      & np.all(previous[:, 1] >= array[index, 1] - _EPS, axis=1)):
                continue
        kept.append(int(index))
        if len(kept) > MAX_FREE_BOXES:
            raise ValueError('Не удалось найти размещение в пределах бюджета свободных областей. Уменьшите число деталей или зон.')
    return [array[index] for index in kept]


def _subtract(boxes, occupied, cancelled):
    result = []
    for box in boxes:
        _check(cancelled)
        if not _overlap(box, occupied):
            result.append(box)
            continue
        for axis in range(box.shape[1]):
            for side in (0, 1):
                cut = occupied[side, axis]
                if box[0, axis] + _EPS < cut < box[1, axis] - _EPS:
                    piece = box.copy()
                    piece[1 - side, axis] = cut
                    result.append(piece)
    return _prune(result, cancelled)


def _blocked(box, gap):
    result = box.copy()
    result[0] -= gap
    result[1] += gap
    return result


def _placement_rotations(allow_rotation, dimensions):
    if not allow_rotation:
        return [np.eye(3)]
    if dimensions == 2:
        return [np.eye(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])]
    return _orthogonal_rotations()


def _variants(points, rotations, cancelled):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    variants, seen = [], set()
    for rotation in rotations:
        bounds = _bounds(points, rotation, center, cancelled)
        extents = bounds[1] - bounds[0]
        key = tuple(np.round(extents, 8))
        if key not in seen:
            variants.append((rotation, bounds, extents))
            seen.add(key)
    return center, variants


def _arrange(point_clouds, platform, *, dimensions, gap, margin, clearance, rotations,
             obstacles, progress, cancelled, prefer_center=False):
    domain, zones = _platform(platform, margin, clearance)
    _, obstacle_points = _meshes(obstacles, 'Препятствия', empty=True)
    static_blocked = [_bounds(points, cancelled=cancelled) for points in obstacle_points] + zones
    blocked = list(static_blocked)
    variants = [_variants(points, rotations[index], cancelled) for index, points in enumerate(point_clouds)]
    order = sorted(range(len(variants)), key=lambda i: -max(float(np.prod(v[2][:dimensions])) for v in variants[i][1]))
    matrices, placed = [None] * len(variants), [None] * len(variants)
    # Retain the free regions after each placement. Replaying every previous
    # subtraction for every rotation makes even a hundred empty boxes slow.
    free_placed = [domain[:, :dimensions].copy()]
    if dimensions == 3:
        for obstacle in static_blocked:
            free_placed = _subtract(free_placed, _blocked(obstacle, gap), cancelled)
    steps = 0
    for count, index in enumerate(order):
        _check(cancelled)
        progress(f'Поиск размещения: {count + 1}/{len(order)}')
        center, choices = variants[index]
        best = None
        free_by_height = {}
        expanded_blocked = np.asarray([_blocked(box, gap) for box in blocked])
        for rotation, local, size in choices:
            if size[2] > domain[1, 2] - domain[0, 2] + _EPS:
                continue
            height_key = float(size[2]) if dimensions == 2 else None
            if height_key not in free_by_height:
                free = list(free_placed)
                if dimensions == 2:
                    for obstacle in static_blocked:
                        expanded = _blocked(obstacle, gap)
                        if (expanded[1, 2] <= clearance + _EPS
                                or expanded[0, 2] >= clearance + size[2] - _EPS):
                            continue
                        free = _subtract(free, expanded[:, :dimensions], cancelled)
                        if not free:
                            break
                free_by_height[height_key] = free
            free = free_by_height[height_key]
            for region in free:
                steps += 1
                if steps > MAX_SEARCH_STEPS:
                    raise ValueError('Не удалось найти размещение за допустимое число попыток. Уменьшите количество деталей или варианты вращения.')
                _check(cancelled)
                if np.any(size[:dimensions] > region[1] - region[0] + _EPS):
                    continue
                origin = np.array([region[0, 0], region[0, 1], clearance if dimensions == 2 else region[0, 2]])
                if prefer_center:
                    origin[:2] = np.clip(-size[:2] / 2, region[0, :2], region[1, :2] - size[:2])
                proposed = np.array([origin, origin + size])
                key = (origin[2], float(np.linalg.norm(proposed.mean(axis=0)[:2])), origin[1], origin[0]) if prefer_center else (
                    origin[2], origin[1], origin[0], float(np.prod(size[:dimensions])))
                if best is not None and key >= best[0]:
                    continue
                # A final 3D guard includes the actual obstacle height even in
                # 2D mode and independently verifies the free-space arithmetic.
                if len(expanded_blocked) and np.any(
                        np.all(proposed[0] < expanded_blocked[:, 1] - _EPS, axis=1)
                        & np.all(expanded_blocked[:, 0] < proposed[1] - _EPS, axis=1)):
                    continue
                displacement = origin - (local[0] + center)
                best = (key, _matrix(rotation, center, displacement), proposed)
        if best is None:
            raise ValueError(f'Не удалось найти размещение детали {index + 1} с заданными зазорами и препятствиями. '
                             'Попробуйте другой порядок, меньшие отступы или меньшее число деталей.')
        matrices[index], placed[index] = best[1], best[2]
        blocked.append(best[2])
        free_placed = _subtract(free_placed, _blocked(best[2], gap)[:, :dimensions], cancelled)
    _check(cancelled)
    bounds = np.asarray(placed)
    extent = bounds[:, 1].max(axis=0) - bounds[:, 0].min(axis=0)
    warnings = ['Используется последовательная упаковка габаритных параллелепипедов. Точное вложение поверхностей и глобальный оптимум не рассчитываются.']
    if zones:
        warnings.append('Цилиндрические запретные зоны учитываются по описанным прямоугольникам; их углы также зарезервированы.')
    if dimensions == 3:
        warnings.append('Объёмная упаковка допускает детали над платформой; пригодность опор и технологии печати проверяется отдельно.')
    return matrices, dict(bounds=bounds, platform_bounds=domain, dimensions=dimensions, gap_mm=gap,
                          margin_mm=margin, clearance_mm=clearance, attempts=steps, approximate=True,
                          height_mm=float(extent[2]), footprint_mm2=float(np.prod(extent[:2])),
                          bbox_volume_mm3=float(np.prod(extent)), warnings=warnings)


def pack_meshes(meshes, platform, *, dimensions=2, gap_mm=2., margin_mm=5., clearance_mm=0.,
                allow_rotation=True, obstacles=(), progress=lambda message: None, cancelled=lambda: False):
    """Place parts independently, returning matrices in the input part order.

    Pass combined part-and-support meshes and every unselected obstacle, even
    hidden ones. A greedy-search failure does not prove packing is impossible.
    """
    _check(cancelled)
    sources, points = _meshes(meshes, 'Размещаемые детали')
    dimensions = _integer(dimensions, 'Размерность упаковки', 2, 3)
    gap = _number(gap_mm, 'Зазор')
    margin = _number(margin_mm, 'Отступ от краёв')
    clearance = _number(clearance_mm, 'Высота над платформой')
    rotations = _placement_rotations(_flag(allow_rotation, 'Разрешить вращение'), dimensions)
    matrices, report = _arrange(points, platform, dimensions=dimensions, gap=gap, margin=margin,
                                clearance=clearance, rotations=[rotations] * len(points), obstacles=obstacles,
                                progress=progress, cancelled=cancelled)
    report['overhang_area_mm2'] = 0.
    for source, matrix, bounds in zip(sources, matrices, report['bounds']):
        _check(cancelled)
        report['overhang_area_mm2'] += _metrics(source, bounds, matrix[:3, :3], 45.)['overhang_area_mm2']
    _check(cancelled)
    return matrices, report


def fit_platform(meshes, platform, *, margin_mm=0., clearance_mm=0., allow_rotation=False,
                  obstacles=(), progress=lambda message: None, cancelled=lambda: False):
    """Move the selected group as one rigid object, with its bottom on clearance."""
    _check(cancelled)
    sources, points = _meshes(meshes, 'Размещаемые детали')
    margin = _number(margin_mm, 'Отступ от краёв')
    clearance = _number(clearance_mm, 'Высота над платформой')
    cloud = np.concatenate(points)
    rotations = [np.eye(3)]
    if _flag(allow_rotation, 'Разрешить вращение'):
        progress('Подбор ориентации группы под платформу…')
        obb = _obb_rotation(cloud, 1, cancelled)
        rotations.extend(_orthogonal_rotations())
        rotations.extend(rotation @ obb for rotation in _orthogonal_rotations())
    matrices, report = _arrange([cloud], platform, dimensions=2, gap=0., margin=margin,
                                clearance=clearance, rotations=[rotations], obstacles=obstacles,
                                progress=progress, cancelled=cancelled, prefer_center=True)
    report['warnings'] = ['Выбранная группа перемещена без масштабирования; взаимное расположение деталей сохранено.'] + report['warnings'][1:]
    report['group_bounds'] = report['bounds'][0]
    report['bounds'] = np.asarray([_bounds(point @ matrices[0][:3, :3].T + matrices[0][:3, 3], cancelled=cancelled)
                                  for point in points])
    report['overhang_area_mm2'] = 0.
    for source, bounds in zip(sources, report['bounds']):
        _check(cancelled)
        report['overhang_area_mm2'] += _metrics(source, bounds, matrices[0][:3, :3], 45.)['overhang_area_mm2']
    _check(cancelled)
    return [matrices[0].copy() for _ in points], report
