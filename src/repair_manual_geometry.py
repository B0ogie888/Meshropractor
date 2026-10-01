"""Explicit mesh edits for the repair workspace; inputs are never modified.

Selection IDs are zero-based indices into the supplied mesh. Vertex indices stay
stable for local edits; clipping and component/Boolean operations may reindex.
"""
from copy import deepcopy

import numpy as np
import trimesh

from mesh_repair import inspect_mesh
from project_store import validate_mesh


def _checkpoint(cancelled, progress=None, message=''):
    if cancelled():
        raise InterruptedError('Редактирование модели отменено')
    if progress is not None:
        progress(message)
    if cancelled():
        raise InterruptedError('Редактирование модели отменено')


def _ids(values, size, label, count=None):
    array = np.asarray([] if values is None else values)
    if (array.ndim != 1 or not len(array) or array.dtype.kind not in 'iu'
            or np.any(array < 0) or np.any(array >= size)):
        raise ValueError(f'{label}: выберите существующие индексы без отрицательных значений.')
    result = array.astype(np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError(f'{label}: повторяющиеся индексы недопустимы.')
    if count is not None and len(result) != count:
        raise ValueError(f'{label}: требуется ровно {count}.')
    return result


def _vector(value, label):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label}: нужны три конечные координаты.') from exc
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{label}: нужны три конечные координаты.')
    return value.copy()


def _tolerance(points):
    return max(float(np.ptp(points, axis=0).max()) * 1e-10, 1e-12)


def _valid_triangles(vertices, faces):
    triangles = vertices[faces]
    lengths = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=1)
    # Relative to each triangle, so small valid features on a large part survive.
    scale = np.maximum(np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                       np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1))
    if np.any(lengths <= np.maximum(scale * scale * 1e-12, np.finfo(float).tiny)):
        raise ValueError('Операция создаёт вырожденный треугольник; измените выбранные точки.')


def _edge_keys(edges, vertex_count):
    edges = np.sort(edges, axis=1)
    return edges[:, 0] * vertex_count + edges[:, 1]


def _append_faces(mesh, faces):
    faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    if not len(faces):
        raise ValueError('Не удалось построить треугольники.')
    _valid_triangles(mesh.vertices, faces)
    ordered = np.ascontiguousarray(np.sort(mesh.faces, axis=1))
    additions = np.ascontiguousarray(np.sort(faces, axis=1))
    row_type = np.dtype((np.void, ordered.dtype.itemsize * 3))
    old_rows, new_rows = ordered.view(row_type).ravel(), additions.view(row_type).ravel()
    if len(np.unique(new_rows)) != len(new_rows) or np.isin(new_rows, old_rows).any():
        raise ValueError('Такой треугольник уже существует; дублирование граней запрещено.')
    new_edges = trimesh.geometry.faces_to_edges(faces)
    keys = _edge_keys(new_edges, len(mesh.vertices))
    touched = np.isin(_edge_keys(mesh.edges, len(mesh.vertices)), keys)
    incidents = {}
    for a, b in mesh.edges[touched]:
        incidents.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b)))
    for a, b in new_edges:
        edge = (int(a), int(b))
        previous = incidents.setdefault(tuple(sorted(edge)), [])
        if len(previous) >= 2:
            raise ValueError('Добавляемые грани создают ребро с более чем двумя треугольниками.')
        if edge in previous:
            raise ValueError('Направление новой грани противоречит нормалям соседних треугольников.')
        previous.append(edge)
    mesh.faces = np.vstack((mesh.faces, faces))
    return len(faces)


def _boundary(mesh):
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    selected = counts[mesh.edges_unique_inverse] == 1
    return mesh.edges[selected], mesh.edges_face[selected]


def _loops(edges, cancelled):
    """Follow directed boundaries; branched or inconsistently wound loops fail."""
    following, incoming = {}, {}
    for a, b in edges:
        a, b = int(a), int(b)
        if a in following or b in incoming:
            raise ValueError('Контур разветвлён или имеет несогласованные нормали. Сначала исправьте его.')
        following[a], incoming[b] = b, a
    if following.keys() != incoming.keys():
        raise ValueError('Выбранная граница не образует замкнутый простой контур.')
    unseen, result = set(following), []
    while unseen:
        _checkpoint(cancelled)
        first = min(unseen)
        ring, current = [], first
        while current in unseen:
            if len(ring) % 1024 == 0:
                _checkpoint(cancelled)
            unseen.remove(current)
            ring.append(current)
            current = following[current]
        if current != first or len(ring) < 3:
            raise ValueError('Выбранная граница не образует простой контур.')
        result.append(ring)
    return result


def _simple_polygon(points, tolerance, cancelled):
    """Reject crossings/touches in a projected contour before triangulation."""
    count = len(points)
    end = np.roll(points, -1, axis=0)
    if np.any(np.linalg.norm(end - points, axis=1) <= tolerance):
        raise ValueError('Контур содержит совпадающие соседние точки.')

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    for index in range(count):
        if index % 64 == 0:
            _checkpoint(cancelled)
        others = np.arange(index + 2, count if index else count - 1)
        if not len(others):
            continue
        a, b = points[index], end[index]
        c, d = points[others], end[others]
        overlap = (np.minimum(a, b) <= np.maximum(c, d) + tolerance).all(axis=1)
        overlap &= (np.minimum(c, d) <= np.maximum(a, b) + tolerance).all(axis=1)
        c, d = c[overlap], d[overlap]
        if not len(c):
            continue
        scale = np.maximum(np.linalg.norm(b - a), np.linalg.norm(d - c, axis=1))
        epsilon = tolerance * scale
        ab_c, ab_d = cross(b - a, c - a), cross(b - a, d - a)
        cd_a, cd_b = cross(d - c, a - c), cross(d - c, b - c)
        intersects = (np.minimum(ab_c, ab_d) <= epsilon) & (np.maximum(ab_c, ab_d) >= -epsilon)
        intersects &= (np.minimum(cd_a, cd_b) <= epsilon) & (np.maximum(cd_a, cd_b) >= -epsilon)
        if intersects.any():
            raise ValueError('Контур пересекает или касается сам себя; автоматическая крышка неоднозначна.')


def _triangulate(mesh, loops, cancelled, normal=None):
    """Triangulate oriented contours, including concavity and nested cut holes."""
    try:
        import manifold3d
    except ImportError as exc:
        raise ValueError('Для заполнения контуров требуется модуль manifold3d.') from exc
    ids = np.concatenate(loops)
    points = mesh.vertices[ids]
    center = points.mean(axis=0)
    if normal is None:
        local = mesh.vertices[loops[0]] - center
        normal = np.cross(local, np.roll(local, -1, axis=0)).sum(axis=0)
    length = np.linalg.norm(normal)
    if length <= np.finfo(float).tiny:
        raise ValueError('Невозможно определить плоскость выбранного контура.')
    normal = np.asarray(normal) / length
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(axis, normal)
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u)))
    polygons = [(mesh.vertices[ring] - center) @ basis for ring in loops]
    tolerance = _tolerance(points)
    for polygon in polygons:
        _simple_polygon(polygon, tolerance, cancelled)
    _checkpoint(cancelled)
    triangles = manifold3d.triangulate(polygons, epsilon=tolerance, allow_convex=False)
    _checkpoint(cancelled)
    return ids[np.asarray(triangles, dtype=np.int64)]


def _selected_hole(mesh, face_ids, vertex_ids, cancelled):
    boundary, owners = _boundary(mesh)
    if not len(boundary):
        raise ValueError('У модели нет открытых граничных рёбер.')
    seeds = set()
    if vertex_ids is not None and np.asarray(vertex_ids).size:
        selected = _ids(vertex_ids, len(mesh.vertices), 'Вершины отверстия')
        if not np.isin(selected, boundary).all():
            raise ValueError('Выбранная вершина не принадлежит открытому контуру.')
        seeds.update(map(int, selected))
    if face_ids is not None and np.asarray(face_ids).size:
        selected = _ids(face_ids, len(mesh.faces), 'Грани отверстия')
        seeds.update(map(int, boundary[np.isin(owners, selected)].ravel()))
    if not seeds:
        raise ValueError('Выберите граничную вершину или грань у нужного отверстия.')
    # Inspect only the chosen connected boundary. A defect elsewhere must not
    # prevent a user from closing this one otherwise-valid hole.
    adjacent = {}
    for a, b in boundary:
        adjacent.setdefault(int(a), set()).add(int(b))
        adjacent.setdefault(int(b), set()).add(int(a))
    component, pending = set(), [next(iter(seeds))]
    while pending:
        if len(component) % 256 == 0:
            _checkpoint(cancelled)
        current = pending.pop()
        if current in component:
            continue
        component.add(current)
        pending.extend(adjacent[current] - component)
    if not seeds <= component:
        raise ValueError('Выбрано несколько отверстий. Укажите только один контур.')
    selected_edges = boundary[np.isin(boundary[:, 0], list(component))]
    rings = _loops(selected_edges, cancelled)
    if len(rings) != 1:
        raise ValueError('Выбранная граница неоднозначна.')
    return rings[0][::-1]  # Reverse the existing boundary's edge directions.


def _clip(mesh, parameters, cancelled):
    point = _vector(parameters.get('point'), 'Точка плоскости')
    normal = _vector(parameters.get('normal'), 'Нормаль плоскости')
    magnitude = float(np.max(np.abs(normal)))
    if magnitude <= np.finfo(float).tiny:
        raise ValueError('Нормаль плоскости не может быть нулевой.')
    normal /= magnitude
    normal /= np.linalg.norm(normal)
    side = parameters.get('side', 'positive')
    if side not in ('positive', 'negative'):
        raise ValueError('Сторона отсечения: positive или negative.')
    if side == 'negative':
        normal *= -1
    cap = parameters.get('cap', False)
    if not isinstance(cap, (bool, np.bool_)):
        raise ValueError('Параметр cap должен быть логическим значением.')
    distances = (mesh.vertices - point) @ normal
    if np.min(distances) >= -_tolerance(mesh.vertices):
        return mesh, 0, 0
    if np.max(distances) <= 0:
        raise ValueError('Плоскость удаляет всю модель. Измените положение или сторону отсечения.')
    # Work near zero for distant CAD coordinates. Quantization only welds the
    # coincident new edge intersections created independently by adjacent faces.
    center = mesh.bounds.mean(axis=0)
    scale = max(float(np.ptp(mesh.vertices, axis=0).max()), np.finfo(float).tiny)
    local = mesh.copy()
    local.vertices = (mesh.vertices - center) / scale
    local_point = (point - center) / scale
    _checkpoint(cancelled)
    result = trimesh.intersections.slice_mesh_plane(local, normal, local_point, cap=False)
    if not len(result.faces):
        raise ValueError('После отсечения не осталось треугольников.')
    result.merge_vertices(digits_vertex=12)
    result.remove_unreferenced_vertices()
    if cap:
        boundary, _ = _boundary(result)
        on_plane = np.abs((result.vertices - local_point) @ normal) <= 1e-9
        edges = boundary[on_plane[boundary].all(axis=1)]
        if len(edges):
            loops = [ring[::-1] for ring in _loops(edges, cancelled)]
            _append_faces(result, _triangulate(result, loops, cancelled, normal=-normal))
        if mesh.is_watertight and not result.is_watertight:
            raise ValueError('Не удалось получить замкнутую крышку сечения. Исходная модель сохранена.')
    _checkpoint(cancelled)
    result.vertices = result.vertices * scale + center
    result.metadata = deepcopy(mesh.metadata)
    deleted = int(np.count_nonzero((distances[mesh.faces] < 0).any(axis=1)))
    return result, len(result.faces) - (len(mesh.faces) - deleted), deleted


def edit_mesh(source, operation, parameters, face_ids=None, vertex_ids=None,
              progress=lambda message: None, cancelled=lambda: False):
    """Return a detached edited mesh and counts; raise ValueError on invalid edits.

    ``delete_faces``: face_ids. ``add_triangle``: three ordered vertex_ids.
    ``bridge``: vertex_ids=[a,b,c,d] describing boundary edges (a,b), (c,d).
    ``move_vertices``: vertex_ids and delta=[x,y,z], or absolute for one vertex.
    ``clip``: point, normal, side='positive'|'negative', cap=False.
    ``fill_hole``: boundary vertex_ids or incident face_ids selecting one loop.
    """
    _checkpoint(cancelled, progress, 'Подготовка ручного исправления…')
    validate_mesh(source)
    if not isinstance(parameters, dict):
        raise ValueError('Параметры операции должны быть словарём.')
    mesh = source.copy()
    mesh.metadata = deepcopy(source.metadata)
    before = inspect_mesh(mesh)
    added = deleted = moved = 0
    if operation == 'delete_faces':
        selected = _ids(face_ids, len(mesh.faces), 'Удаляемые грани')
        if len(selected) == len(mesh.faces):
            raise ValueError('Нельзя удалить все треугольники детали.')
        keep = np.ones(len(mesh.faces), dtype=bool)
        keep[selected] = False
        mesh.update_faces(keep)
        deleted = len(selected)
    elif operation == 'add_triangle':
        selected = _ids(vertex_ids, len(mesh.vertices), 'Вершины треугольника', 3)
        added = _append_faces(mesh, [selected])
    elif operation == 'bridge':
        selected = _ids(vertex_ids, len(mesh.vertices), 'Вершины двух рёбер', 4)
        boundary, _ = _boundary(mesh)
        directed = {tuple(sorted(edge)): tuple(map(int, edge)) for edge in boundary}
        try:
            a, b = directed[tuple(sorted(selected[:2]))]
            c, d = directed[tuple(sorted(selected[2:]))]
        except KeyError as exc:
            raise ValueError('Оба выбранных ребра должны принадлежать открытой границе.') from exc
        added = _append_faces(mesh, _triangulate(mesh, [[b, a, d, c]], cancelled))
        if added != 2:
            raise ValueError('Мост должен содержать два невырожденных треугольника.')
    elif operation == 'move_vertices':
        selected = _ids(vertex_ids, len(mesh.vertices), 'Перемещаемые вершины')
        if ('delta' in parameters) == ('absolute' in parameters):
            raise ValueError('Задайте либо смещение delta, либо абсолютные координаты absolute.')
        original = mesh.vertices[selected].copy()
        if 'absolute' in parameters:
            if len(selected) != 1:
                raise ValueError('Абсолютные координаты задаются для одной вершины.')
            mesh.vertices[selected] = _vector(parameters['absolute'], 'Координаты вершины')
        else:
            mesh.vertices[selected] += _vector(parameters['delta'], 'Смещение вершин')
        affected = np.isin(mesh.faces, selected).any(axis=1)
        _valid_triangles(mesh.vertices, mesh.faces[affected])
        moved = int(np.count_nonzero(np.any(mesh.vertices[selected] != original, axis=1)))
    elif operation == 'fill_hole':
        loop = _selected_hole(mesh, face_ids, vertex_ids, cancelled)
        added = _append_faces(mesh, _triangulate(mesh, [loop], cancelled))
    elif operation == 'clip':
        mesh, added, deleted = _clip(mesh, parameters, cancelled)
    else:
        raise ValueError(f'Неизвестная операция исправления: {operation}')
    _checkpoint(cancelled, progress, 'Проверка результата ручного исправления…')
    validate_mesh(mesh)
    after = inspect_mesh(mesh)
    changed = (not np.array_equal(mesh.vertices, source.vertices)
               or not np.array_equal(mesh.faces, source.faces))
    return mesh, dict(operation=operation, changed=changed, before=before, after=after,
                     added_faces=added, deleted_faces=deleted, moved_vertices=moved,
                     vertex_ids_preserved=operation != 'clip')


def split_components(source, return_face_maps=False, max_components=None):
    """Split by shared edges, retaining open components without automatic repair.

    With ``return_face_maps=True``, return ``(meshes, face_maps)`` where
    ``face_maps[i][j]`` is the original face index of face ``j`` in mesh ``i``.
    Maps come from topology, so coincident but disconnected shells remain
    distinguishable. Every source face occurs in exactly one component.
    ``max_components`` optionally limits the count before allocating submeshes.
    """
    validate_mesh(source)
    if max_components is not None and (
            isinstance(max_components, (bool, np.bool_))
            or not isinstance(max_components, (int, np.integer)) or max_components < 1):
        raise ValueError('Максимальное число фрагментов должно быть положительным целым числом.')
    working = source.copy()
    groups = trimesh.graph.connected_components(
        edges=working.face_adjacency, nodes=np.arange(len(working.faces)), min_len=1)
    if max_components is not None and len(groups) > max_components:
        raise ValueError(f'Слишком много фрагментов: {len(groups)} (максимум {max_components}); '
                         'сначала удалите шум.')
    face_maps = [np.sort(np.asarray(group, dtype=np.int64)).copy() for group in groups]
    result = list(working.submesh(face_maps, only_watertight=False, repair=False))
    for component in result:
        component.metadata = deepcopy(source.metadata)
        validate_mesh(component)
    return (result, face_maps) if return_face_maps else result


def combine_meshes(meshes, boolean=True):
    """Union closed solids via Manifold, or explicitly group with boolean=False."""
    meshes = list(meshes)
    if len(meshes) < 2:
        raise ValueError('Выберите как минимум две детали для объединения.')
    for mesh in meshes:
        validate_mesh(mesh)
    if not isinstance(boolean, (bool, np.bool_)):
        raise ValueError('Параметр boolean должен быть логическим значением.')
    copies = [mesh.copy() for mesh in meshes]
    if boolean:
        if not all(mesh.is_volume for mesh in copies):
            raise ValueError('Булево объединение требует замкнутых деталей с согласованными наружными нормалями.')
        if 'manifold' not in trimesh.boolean.engines_available:
            raise ValueError('Для булева объединения требуется установленный модуль manifold3d.')
        bounds = np.asarray([mesh.bounds for mesh in copies])
        lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
        center, scale = (lower + upper) / 2, float((upper - lower).max())
        for mesh in copies:
            mesh.vertices = (mesh.vertices - center) / scale
        try:
            result = trimesh.boolean.union(copies, engine='manifold', check_volume=True)
            validate_mesh(result)
            if not result.is_volume:
                raise ValueError('Результат не является замкнутым ориентированным телом.')
        except Exception as exc:
            raise ValueError('Не удалось выполнить корректное булево объединение: ' + str(exc)) from exc
        result.vertices = result.vertices * scale + center
    else:
        result = trimesh.util.concatenate(copies)
    validate_mesh(result)
    result.metadata = dict(combination='boolean_union' if boolean else 'group',
                           source_metadata=[deepcopy(mesh.metadata) for mesh in meshes])
    return result
