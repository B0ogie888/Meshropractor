"""Bounded repair-tool operations on private meshes, without UI dependencies.

All dimensions are millimetres. Algorithms which change topology operate on a
whole part. Diagnostics retain original face IDs. VTK cancellation is observed
at its progress callbacks and between stages; a single native step may finish
before it notices cancellation.
"""
from copy import deepcopy
import math

import numpy as np
import trimesh

from project_store import validate_mesh


MAX_INPUT_FACES = 10_000_000
MAX_OUTPUT_FACES = 4_000_000
MAX_VOXELS = 4_000_000
MAX_CONTACT_PAIRS = 20_000_000
MAX_QUERY_CANDIDATES = 250_000
MAX_SHELL_PAIRS = 10_000
MAX_GEOMETRY_BYTES = 512 * 1024 * 1024
OPERATIONS = frozenset(('normals', 'stitch', 'holes', 'noise', 'duplicates', 'smooth',
                        'clean_smooth', 'decimate', 'subdivide', 'remesh', 'wrap',
                        'slivers', 'overlaps'))
SELECTION_OPERATIONS = frozenset(('normals', 'smooth', 'slivers', 'overlaps'))
PARAMETER_KEYS = {
    'normals': {'flip'}, 'stitch': {'tolerance_mm'},
    'holes': {'max_diameter_mm', 'max_boundary_vertices'},
    'noise': {'min_faces', 'min_volume_mm3', 'keep_largest'}, 'duplicates': set(),
    'smooth': {'iterations', 'relaxation', 'preserve_boundary'},
    'clean_smooth': {'iterations', 'relaxation', 'preserve_boundary'},
    'decimate': {'target_ratio', 'preserve_topology'}, 'subdivide': {'iterations'},
    'remesh': {'target_edge_mm', 'iterations'}, 'wrap': {'voxel_size_mm'},
    'slivers': {'min_angle_deg'}, 'overlaps': {'tolerance_mm'},
}


def _check(cancelled):
    if cancelled():
        raise InterruptedError('Операция исправления отменена; исходная модель сохранена.')


def _number(parameters, key, default, low, high, *, integer=False, positive=False):
    value = parameters.get(key, default)
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'Некорректный параметр {key}.')
    if not np.isfinite(value) or not low <= value <= high or (positive and value <= 0):
        raise ValueError(f'Параметр {key} должен быть в пределах {low:g}…{high:g}.')
    if integer and int(value) != value:
        raise ValueError(f'Параметр {key} должен быть целым числом.')
    return int(value) if integer else float(value)


def _flag(parameters, key, default):
    value = parameters.get(key, default)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'Параметр {key} должен быть логическим.')
    return bool(value)


def _to_polydata(mesh):
    import pyvista as pv
    faces = np.empty((len(mesh.faces), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = mesh.faces
    return pv.PolyData(np.asarray(mesh.vertices).copy(), faces.ravel())


def _from_polydata(data):
    import pyvista as pv
    wrapped = pv.wrap(data)
    faces = np.asarray(wrapped.faces)
    if not len(faces):
        raise ValueError('Операция не оставила треугольников. Исходная модель сохранена.')
    if len(faces) % 4 or np.any(faces.reshape(-1, 4)[:, 0] != 3):
        raise ValueError('Алгоритм вернул нетреугольную сетку.')
    return trimesh.Trimesh(np.asarray(wrapped.points).copy(), faces.reshape(-1, 4)[:, 1:].copy(), process=False)


def _native(algorithm, title, progress, cancelled):
    last = [-1]
    def changed(obj, _event):
        if cancelled():
            obj.SetAbortExecute(True)
            return
        percent = int(obj.GetProgress() * 100)
        if percent >= last[0] + 10:
            progress(f'{title}: {percent}%')
            last[0] = percent
    _check(cancelled)
    progress(title + '…')
    observer = algorithm.AddObserver('ProgressEvent', changed)
    try:
        algorithm.Update()
    finally:
        algorithm.RemoveObserver(observer)
    _check(cancelled)
    if algorithm.GetErrorCode():
        raise ValueError(f'{title}: алгоритм завершился с ошибкой.')
    return algorithm.GetOutput()


def _clean(mesh):
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-12))
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _weld(mesh, tolerance, progress, cancelled):
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData
    algorithm = vtkCleanPolyData()
    algorithm.SetInputData(_to_polydata(mesh))
    algorithm.PointMergingOn()
    algorithm.ToleranceIsAbsoluteOn()
    algorithm.SetAbsoluteTolerance(tolerance)
    algorithm.ConvertPolysToLinesOff()
    algorithm.ConvertLinesToPointsOff()
    algorithm.ConvertStripsToPolysOff()
    result = _clean(_from_polydata(_native(algorithm, 'Сшивка близких вершин', progress, cancelled)))
    if len(result.vertices) == len(mesh.vertices) and len(result.faces) == len(mesh.faces):
        # vtkCleanPolyData reindexes even an already clean mesh. Avoid turning
        # that no-op into an edit and invalidating face-linked selections.
        return mesh
    return result


def _orient_shells(mesh, progress, cancelled):
    """Orient closed material boundaries by containment, preserving face IDs.

    A cavity is a negatively oriented shell, not another outward solid. Winding
    is repaired first; relative shell orientation is decided only after ruling
    out contacts and crossings. Open patches have no unambiguous inside.
    """
    progress('Согласование ориентации соседних треугольников…')
    trimesh.repair.fix_winding(mesh)
    _check(cancelled)
    groups = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1,
                                                nodes=np.arange(len(mesh.faces)))
    shells, open_count = [], 0
    for group in groups:
        _check(cancelled)
        if len(group) < 4:
            open_count += 1
            continue
        body = mesh.submesh([group], append=True, repair=False)
        if not body.is_watertight or not body.is_winding_consistent:
            open_count += 1
            continue
        bounds = body.bounds.copy()
        center = bounds.mean(axis=0)
        scale = float(np.ptp(bounds, axis=0).max())
        if scale <= np.finfo(float).tiny:
            raise ValueError('Невозможно определить объём замкнутой оболочки.')
        body.vertices = (body.vertices - center) / scale
        volume = float(body.volume)
        if not np.isfinite(volume) or abs(volume) <= 1e-15:
            raise ValueError('Объём замкнутой оболочки близок к нулю; сначала исправьте её геометрию.')
        samples = np.unique(np.concatenate((np.argmin(body.vertices, axis=0),
                                            np.argmax(body.vertices, axis=0))))
        faces = np.linspace(0, len(body.faces) - 1, min(12, len(body.faces)), dtype=int)
        points = np.vstack((body.vertices[samples], body.vertices[body.faces[faces]].mean(axis=1)))
        shells.append(dict(ids=group, mesh=body, bounds=bounds, center=center,
                           scale=scale, volume=volume, samples=points, depth=0))

    if len(shells) > 1:
        from rtree.index import Index, Property
        from vtkmodules.vtkCommonMath import vtkMatrix4x4
        from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter
        tree = Index(((i, np.concatenate(s['bounds']), None) for i, s in enumerate(shells)),
                     properties=Property(dimension=3))
        tested = 0

        def inside(inner, outer):
            # Both inputs remain near zero, even for CAD coordinates far from
            # the origin. Multiple representatives expose numerical ambiguity.
            points = (inner['samples'] * inner['scale'] / outer['scale']
                      + (inner['center'] - outer['center']) / outer['scale'])
            states = []
            for point in points:
                _check(cancelled)
                states.append(bool(outer['mesh'].contains(point[None, :])[0]))
            if any(states) != all(states):
                raise ValueError('Вложенность оболочек неоднозначна. Проверьте пересечения и касания перед исправлением нормалей.')
            return all(states)

        try:
            for i, first in enumerate(shells):
                _check(cancelled)
                progress(f'Проверка вложенности замкнутых оболочек: {i + 1}/{len(shells)}')
                for j in tree.intersection(np.concatenate(first['bounds'])):
                    if j <= i:
                        continue
                    tested += 1
                    if tested > MAX_SHELL_PAIRS:
                        raise ValueError('Слишком много соприкасающихся границ оболочек для анализа нормалей. Разделите модель на меньшие группы.')
                    second = shells[j]
                    # OBB contact detection rejects even contacts which a small
                    # sample of containment rays could otherwise miss.
                    collision = vtkCollisionDetectionFilter()
                    for index, shell in enumerate((first, second)):
                        if 'polydata' not in shell:
                            shell['polydata'] = _to_polydata(shell['mesh'])
                        collision.SetInputData(index, shell['polydata'])
                        matrix = vtkMatrix4x4()
                        matrix.Identity()
                        for axis in range(3):
                            matrix.SetElement(axis, axis, shell['scale'] / first['scale'])
                            matrix.SetElement(axis, 3, (shell['center'][axis] - first['center'][axis]) / first['scale'])
                        collision.SetMatrix(index, matrix)
                    collision.SetCollisionModeToFirstContact()
                    collision.GenerateScalarsOff()
                    collision.SetBoxTolerance(1e-9)
                    collision.SetCellTolerance(1e-18)
                    _native(collision, 'Проверка контакта оболочек', lambda _: None, cancelled)
                    if collision.GetNumberOfContacts():
                        raise ValueError('Замкнутые оболочки пересекаются или касаются. Сначала устраните контакты; автоматическая ориентация полостей неоднозначна.')
                    for inner, outer in ((first, second), (second, first)):
                        if (np.all(inner['bounds'][0] >= outer['bounds'][0])
                                and np.all(inner['bounds'][1] <= outer['bounds'][1])
                                and inside(inner, outer)):
                            inner['depth'] += 1
        finally:
            tree.close()

    faces = mesh.faces.copy()
    for shell in shells:
        expected_positive = shell['depth'] % 2 == 0
        if (shell['volume'] > 0) != expected_positive:
            faces[shell['ids']] = faces[shell['ids'], ::-1]
    mesh.faces = faces
    warnings = []
    if open_count:
        warnings.append(f'Открытые или неоднозначные фрагменты: {open_count}. Согласованы соседние грани; направление наружу для них не определяется.')
    return dict(closed_shells=len(shells), cavity_shells=sum(s['depth'] % 2 for s in shells),
                warnings=warnings)


def _smooth(mesh, iterations, relaxation, preserve_boundary, face_ids, progress, cancelled):
    from scipy.sparse import coo_matrix
    progress('Подготовка соседства вершин для сглаживания…')
    edges = mesh.edges_unique
    n = len(mesh.vertices)
    movable = np.ones(n, bool)
    if face_ids is not None:
        chosen = np.zeros(len(mesh.faces), bool)
        chosen[face_ids] = True
        movable[:] = False
        movable[mesh.faces[chosen].ravel()] = True
        # Pin the border of a partial edit: unselected triangles stay unchanged.
        movable[mesh.faces[~chosen].ravel()] = False
    if preserve_boundary:
        counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
        movable[edges[counts != 2].ravel()] = False
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    degree = np.bincount(rows, minlength=n)
    movable &= degree > 0
    weights = 1. / np.maximum(degree[rows], 1)
    adjacency = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    vertices = mesh.vertices.copy()
    # A positive and a negative pass limit the usual Laplacian shrinkage. This
    # is geometric smoothing, not a guarantee of exact volume preservation.
    for index in range(iterations):
        _check(cancelled)
        for amount in (relaxation, -relaxation / (1. - .1 * relaxation)):
            delta = adjacency @ vertices - vertices
            vertices[movable] += amount * delta[movable]
        progress(f'Сглаживание: {index + 1}/{iterations}')
    mesh.vertices = vertices
    return mesh


def _decimate(mesh, target_ratio, preserve_topology, progress, cancelled):
    if target_ratio >= 1 or len(mesh.faces) <= 4:
        return mesh
    from vtkmodules.vtkFiltersCore import vtkDecimatePro
    algorithm = vtkDecimatePro()
    algorithm.SetInputData(_to_polydata(mesh))
    algorithm.SetTargetReduction(1. - target_ratio)
    algorithm.SetPreserveTopology(preserve_topology)
    algorithm.SetSplitting(not preserve_topology)
    algorithm.SetBoundaryVertexDeletion(not preserve_topology)
    algorithm.SetFeatureAngle(45.)
    algorithm.SetAccumulateError(True)
    return _clean(_from_polydata(_native(algorithm, 'Сокращение числа треугольников', progress, cancelled)))


def _subdivide(mesh, iterations, progress, cancelled):
    if len(mesh.faces) * (4 ** iterations) > MAX_OUTPUT_FACES:
        raise ValueError(f'Разбиение превысит предел {MAX_OUTPUT_FACES:,} треугольников. Уменьшите число проходов.')
    for index in range(iterations):
        _check(cancelled)
        progress(f'Разбиение треугольников: {index + 1}/{iterations}')
        vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
    return mesh


def _fill_holes(mesh, diameter, limit, progress, cancelled):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from repair_manual_geometry import _boundary, _loops, _triangulate, _append_faces
    progress('Поиск открытых контуров…')
    boundary, _ = _boundary(mesh)
    skipped = dict(too_large=0, too_many_vertices=0, nonplanar=0, ambiguous=0)
    if not len(boundary):
        return mesh, 0, skipped
    # An unavailable backend is an installation error, not an ambiguous hole.
    try:
        import manifold3d  # noqa: F401
    except ImportError as exc:
        raise ValueError('Для заполнения контуров требуется модуль manifold3d.') from exc
    _, compact = np.unique(boundary, return_inverse=True)
    pairs = compact.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(int(compact.max()) + 1,) * 2).tocsr()
    _, labels = connected_components(graph, directed=False)
    edge_labels = labels[pairs[:, 0]]
    _, counts = np.unique(edge_labels, return_counts=True)
    order = np.argsort(edge_labels, kind='stable')
    offsets = np.r_[0, np.cumsum(counts)]
    candidates, additions, last_percent = [], 0, -5
    for index in range(len(counts)):
        _check(cancelled)
        percent = index * 100 // max(len(counts), 1)
        if percent >= last_percent + 5:
            progress(f'Проверка контуров: {percent}%')
            last_percent = percent
        edges = boundary[order[offsets[index]:offsets[index+1]]]
        # Components are handled separately: a branched boundary elsewhere
        # must not prevent filling this otherwise unambiguous closed loop.
        try:
            rings = _loops(edges, cancelled)
        except ValueError:
            skipped['ambiguous'] += 1
            continue
        if len(rings) != 1:
            skipped['ambiguous'] += 1
            continue
        loop = rings[0][::-1]
        if len(loop) > limit:
            skipped['too_many_vertices'] += 1
            continue
        points = mesh.vertices[loop]
        actual_diameter = float(np.linalg.norm(points[:, None] - points[None, :], axis=2).max())
        if actual_diameter > diameter:
            skipped['too_large'] += 1
            continue
        local = points - points.mean(axis=0)
        _, _, axes = np.linalg.svd(local, full_matrices=False)
        if np.max(np.abs(local @ axes[-1])) > max(1e-6, actual_diameter * 1e-10):
            skipped['nonplanar'] += 1
            continue
        try:
            cap = _triangulate(mesh, [loop], cancelled)
        except ValueError:
            skipped['ambiguous'] += 1
            continue
        if not len(cap):
            skipped['ambiguous'] += 1
            continue
        additions += len(cap)
        if len(mesh.faces) + additions > MAX_INPUT_FACES:
            raise ValueError('Заполнение отверстий превысит бюджет треугольников.')
        candidates.append(cap)
    if not candidates:
        return mesh, 0, skipped
    _check(cancelled)
    progress('Проверка ориентации и сшивка крышек отверстий…')
    try:
        # Validate all independent caps together, scanning the original mesh
        # only once in the usual case of many valid small holes.
        _append_faces(mesh, np.vstack(candidates))
        filled = len(candidates)
    except ValueError:
        # One cap may coincide with existing geometry or touch another sheet.
        # Keep other valid holes repairable; _append_faces validates before edit.
        filled = 0
        for cap in candidates:
            _check(cancelled)
            try:
                _append_faces(mesh, cap)
                filled += 1
            except ValueError:
                skipped['ambiguous'] += 1
    return mesh, filled, skipped


def _noise(mesh, min_faces, min_volume, keep_largest, progress, cancelled):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    progress('Поиск связных фрагментов…')
    edges = mesh.edges_unique
    graph = coo_matrix((np.ones(len(edges), np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(len(mesh.vertices), len(mesh.vertices))).tocsr()
    _, labels = connected_components(graph, directed=False)
    face_labels = labels[mesh.faces[:, 0]]
    ids, counts = np.unique(face_labels, return_counts=True)
    order = np.argsort(face_labels, kind='stable')
    offsets = np.r_[0, np.cumsum(counts)]
    areas = np.add.reduceat(mesh.area_faces[order], offsets[:-1])
    largest = ids[np.argmax(areas)]
    keep = np.ones(len(mesh.faces), bool)
    removed, open_ignored, last_percent = 0, 0, -5
    for index, (component, count) in enumerate(zip(ids, counts)):
        _check(cancelled)
        if keep_largest and component == largest:
            continue
        use = order[offsets[index]:offsets[index+1]]
        small = count < min_faces
        if min_volume > 0 and not small:
            part = mesh.submesh([use], append=True, repair=False)
            if part.is_watertight and part.is_winding_consistent:
                small = abs(part.volume) < min_volume
            else:
                open_ignored += 1
        if small:
            keep[use] = False
            removed += 1
        percent = (index + 1) * 100 // max(len(ids), 1)
        if percent >= last_percent + 5:
            progress(f'Проверка фрагментов: {percent}%')
            last_percent = percent
    if not keep.any():
        raise ValueError('Порог удалит всю модель. Включите сохранение крупнейшего фрагмента или уменьшите порог.')
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh, removed, open_ignored


def _wrap(mesh, pitch, progress, cancelled):
    from scipy.ndimage import binary_closing, binary_fill_holes
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkFlyingEdges3D
    from vtkmodules.vtkImagingHybrid import vtkSampleFunction
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    spans = np.asarray(mesh.extents, dtype=float)
    raw = np.ceil(spans / pitch) + 5
    if not np.isfinite(raw).all() or np.any(raw > MAX_VOXELS):
        raise ValueError('Размер ячейки слишком мал: превышен бюджет воксельной сетки.')
    dims = tuple(map(int, raw))
    count = math.prod(dims)
    if count > MAX_VOXELS:
        minimum = pitch * (count / MAX_VOXELS) ** (1 / 3)
        raise ValueError(f'Оболочке требуется {count:,} ячеек (предел {MAX_VOXELS:,}). '
                         f'Увеличьте размер ячейки примерно до {minimum:.6g} мм или более.')
    origin = mesh.bounds[0] - 2 * pitch
    high = origin + (np.asarray(dims) - 1) * pitch
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(_to_polydata(mesh))
    sampler = vtkSampleFunction()
    sampler.SetImplicitFunction(distance)
    sampler.SetModelBounds(*(value for axis in range(3) for value in (origin[axis], high[axis])))
    sampler.SetSampleDimensions(*dims)
    sampler.ComputeNormalsOff()
    sampler.SetOutputScalarTypeToFloat()
    sampled = _native(sampler, 'Воксельная оболочка: расстояния до поверхности', progress, cancelled)
    values = vtk_to_numpy(sampled.GetPointData().GetScalars()).reshape(dims, order='F')
    if not np.isfinite(values).all():
        raise ValueError('Не удалось вычислить конечные расстояния для оболочки.')
    progress('Воксельная оболочка: замыкание промежутков и заполнение объёма…')
    shell = np.abs(values) <= pitch * np.sqrt(3) / 2
    shell = binary_closing(shell, iterations=1)
    _check(cancelled)
    filled = binary_fill_holes(shell)
    _check(cancelled)
    if not filled.any():
        raise ValueError('При таком размере ячейки оболочка исчезла. Уменьшите размер ячейки.')
    base = filled[:-1, :-1, :-1]
    active = np.zeros(base.shape, bool)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                active |= base != filled[x:x+dims[0]-1, y:y+dims[1]-1, z:z+dims[2]-1]
    if int(active.sum()) * 5 > MAX_OUTPUT_FACES:
        raise ValueError('Поверхность оболочки превысит бюджет треугольников. Увеличьте размер ячейки.')
    image = vtkImageData()
    image.SetDimensions(*dims)
    image.SetOrigin(*origin)
    image.SetSpacing(pitch, pitch, pitch)
    image.GetPointData().SetScalars(numpy_to_vtk(np.asarray(filled, dtype=np.uint8).ravel(order='F'), deep=True))
    contour = vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, .5)
    contour.ComputeNormalsOff()
    result = _clean(_from_polydata(_native(contour, 'Построение поверхности оболочки', progress, cancelled)))
    trimesh.repair.fix_normals(result, multibody=True)
    return result, dict(voxel_dimensions=dims, voxel_count=count, voxel_size_mm=pitch)


def _overlaps(mesh, ids, tolerance, progress, cancelled):
    """Bound each broad-phase allocation before calling R-tree's vector query."""
    from rtree.index import Index, Property
    from mesh_intersections import triangle_contacts
    n = len(mesh.faces)
    low, high = np.empty((n, 3)), np.empty((n, 3))
    for start in range(0, n, 100_000):
        _check(cancelled)
        triangles = mesh.vertices[mesh.faces[start:start + 100_000]]
        low[start:start + len(triangles)] = triangles.min(axis=1)
        high[start:start + len(triangles)] = triangles.max(axis=1)
    progress('Построение пространственного индекса треугольников…')
    tree = Index((np.arange(n, dtype=np.int64), low, high), properties=Property(dimension=3))
    selected = np.arange(n) if ids is None else ids
    overlaps, crossings = np.zeros(n, bool), np.zeros(n, bool)
    checked, pending, candidates, last_percent = 0, [], 0, -5
    def examine(batch):
        nonlocal checked
        batch = np.asarray(batch, dtype=np.int64)
        partners, counts = tree.intersection_v(low[batch]-tolerance, high[batch]+tolerance)
        owners = np.repeat(batch, counts.astype(np.int64))
        use = partners > owners if ids is None else partners != owners
        owners, partners = owners[use], partners[use]
        checked += len(owners)
        if checked > MAX_CONTACT_PAIRS:
            raise ValueError('Слишком много потенциальных нахлёстов. Уменьшите число треугольников или анализируйте выделенный участок.')
        for offset in range(0, len(owners), 16_384):
            _check(cancelled)
            a, b = owners[offset:offset+16_384], partners[offset:offset+16_384]
            fa, fb = mesh.faces[a], mesh.faces[b]
            ov, ix = triangle_contacts(mesh.vertices[fa], mesh.vertices[fb], tolerance)
            shared = (fa[:, :, None] == fb[:, None, :]).any(axis=2).sum(axis=1)
            ix &= shared < 2
            overlaps[a[ov]] = True
            overlaps[b[ov]] = True
            crossings[a[ix]] = True
            crossings[b[ix]] = True
    try:
        for index, face in enumerate(selected):
            if index % 256 == 0:
                _check(cancelled)
                percent = index * 100 // max(len(selected), 1)
                if percent >= last_percent + 5:
                    progress(f'Нахлёсты и пересечения: {percent}%')
                    last_percent = percent
            count = tree.count(np.concatenate((low[face]-tolerance, high[face]+tolerance)))
            if count > MAX_QUERY_CANDIDATES:
                raise ValueError('Слишком плотное скопление треугольников для анализа в пределах памяти. Сначала удалите дубликаты или сократите сетку.')
            if pending and (len(pending) >= 256 or candidates + count > MAX_QUERY_CANDIDATES):
                examine(pending)
                pending, candidates = [], 0
            pending.append(face)
            candidates += count
        if pending:
            examine(pending)
    finally:
        tree.close()
    mask = overlaps | crossings
    if ids is not None:
        allowed = np.zeros(n, bool)
        allowed[ids] = True
        mask &= allowed
    return dict(selected_faces=np.flatnonzero(mask), overlap_faces=np.flatnonzero(overlaps),
                intersection_faces=np.flatnonzero(crossings), checked_pairs=checked)


def repair_mesh(source, operation, parameters=None, face_ids=None,
                progress=lambda message: None, cancelled=lambda: False):
    """Return ``(private_mesh, report)``; never write to source or a model file.

    ``target_ratio`` is the fraction to retain, not the fraction to remove.
    Partial smoothing pins shared vertices, so unselected faces cannot move.
    Hole filling is intentionally limited to simple planar boundaries.
    Remeshing approximates density by subdivision/smoothing/decimation; wrap
    changes dimensions by an amount dependent on its voxel size.
    """
    _check(cancelled)
    if operation not in OPERATIONS:
        raise ValueError('Неизвестная операция исправления.')
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError('Параметры операции должны быть словарём.')
    if parameters.keys() - PARAMETER_KEYS[operation]:
        raise ValueError('Передан неизвестный параметр операции: ' + ', '.join(map(str, parameters.keys() - PARAMETER_KEYS[operation])))
    if not isinstance(source, trimesh.Trimesh) or len(source.faces) > MAX_INPUT_FACES:
        raise ValueError(f'Операция поддерживает не более {MAX_INPUT_FACES:,} треугольников.')
    if source.vertices.nbytes + source.faces.nbytes > MAX_GEOMETRY_BYTES:
        raise ValueError('Геометрия превышает бюджет входных данных (512 МиБ). Уменьшите сетку перед этой операцией.')
    validate_mesh(source)
    ids = None
    if face_ids is not None:
        raw = np.asarray(list(face_ids) if isinstance(face_ids, (set, frozenset)) else face_ids)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw):
            raise ValueError('Выберите хотя бы один треугольник; индексы должны быть целыми.')
        if raw.min() < 0 or raw.max() >= len(source.faces):
            raise ValueError('Индексы выделенных треугольников вне модели.')
        ids = np.unique(raw.astype(np.int64))
        if len(ids) == len(source.faces):
            ids = None
        elif operation not in SELECTION_OPERATIONS:
            raise ValueError('Эта операция изменяет топологию и применяется к детали целиком.')
    progress('Подготовка копии модели…')
    mesh = source.copy()
    report = dict(operation=operation, changed=False, before=dict(vertices=len(source.vertices), faces=len(source.faces)), warnings=[])
    if operation == 'normals':
        flip = _flag(parameters, 'flip', False)
        if flip:
            faces = mesh.faces.copy()
            if ids is None:
                faces = faces[:, ::-1].copy()
            else:
                faces[ids] = faces[ids, ::-1]
            mesh.faces = faces
        else:
            original = mesh.faces.copy() if ids is not None else None
            orientation = _orient_shells(mesh, progress, cancelled)
            report['warnings'].extend(orientation.pop('warnings'))
            report.update(orientation)
            if ids is not None:
                original[ids] = mesh.faces[ids]
                mesh.faces = original
                report['warnings'].append('Для частичного исправления ориентация сверяется со всей деталью; дефекты вне выделения остаются.')
    elif operation == 'stitch':
        tolerance = _number(parameters, 'tolerance_mm', .01, 0, 100, positive=True)
        mesh = _weld(mesh, tolerance, progress, cancelled)
    elif operation == 'duplicates':
        before = len(mesh.faces)
        # Raw STL may contain distinct indices at exactly the same position.
        # Compare geometric triangles, ignoring winding, without welding the
        # remaining model as an unexpected side effect of this command.
        _, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        triangles = np.sort(inverse[mesh.faces], axis=1)
        _, first = np.unique(triangles, axis=0, return_index=True)
        keep = np.zeros(len(mesh.faces), bool)
        keep[first] = True
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        report['removed_faces'] = before - len(mesh.faces)
    elif operation == 'holes':
        diameter = _number(parameters, 'max_diameter_mm', 10., 0, 1_000_000, positive=True)
        limit = _number(parameters, 'max_boundary_vertices', 256, 3, 512, integer=True)
        mesh, count, skipped = _fill_holes(mesh, diameter, limit, progress, cancelled)
        report['holes_filled'] = count
        report['holes_skipped'] = skipped
        report['warnings'].append('Заполняются простые плоские контуры заданного диаметра, включая вогнутые. '
                                  'Неплоские, разветвлённые, самопересекающиеся и неоднозначные контуры пропускаются; конструктивные отверстия также могут быть закрыты.')
        if any(skipped.values()):
            report['warnings'].append(f"Пропущено: по диаметру {skipped['too_large']}, по числу рёбер {skipped['too_many_vertices']}, "
                                      f"неплоских {skipped['nonplanar']}, неоднозначных {skipped['ambiguous']}.")
    elif operation == 'noise':
        min_faces = _number(parameters, 'min_faces', 10, 0, MAX_INPUT_FACES, integer=True)
        volume = _number(parameters, 'min_volume_mm3', 0., 0, 1e18)
        mesh, count, open_ignored = _noise(mesh, min_faces, volume, _flag(parameters, 'keep_largest', True), progress, cancelled)
        report['removed_components'] = count
        if open_ignored:
            report['warnings'].append(f'Объём не применялся к {open_ignored} открытым или несогласованным фрагментам; для них действует только число граней.')
    elif operation in ('smooth', 'clean_smooth'):
        count = _number(parameters, 'iterations', 10, 1, 200, integer=True)
        relaxation = _number(parameters, 'relaxation', .1, 0, .5, positive=True)
        if operation == 'clean_smooth':
            mesh = _weld(mesh, 0., progress, cancelled)
        mesh = _smooth(mesh, count, relaxation, _flag(parameters, 'preserve_boundary', True), ids, progress, cancelled)
        report['warnings'].append('Сглаживание изменяет координаты вершин и не гарантирует сохранение размеров или объёма.')
    elif operation == 'decimate':
        ratio = _number(parameters, 'target_ratio', .5, .001, 1.)
        preserve = _flag(parameters, 'preserve_topology', True)
        mesh = _decimate(mesh, ratio, preserve, progress, cancelled)
        report['actual_ratio'] = len(mesh.faces) / len(source.faces)
        report['warnings'].append('Целевая плотность приблизительная; сохранение топологии и границ может ограничить сокращение.')
    elif operation == 'subdivide':
        count = _number(parameters, 'iterations', 1, 1, 3, integer=True)
        mesh = _subdivide(mesh, count, progress, cancelled)
    elif operation == 'remesh':
        edge = _number(parameters, 'target_edge_mm', 1., 0, 1_000_000, positive=True)
        count = _number(parameters, 'iterations', 2, 1, 20, integer=True)
        minimum_edge = math.sqrt(mesh.area * 4 / (np.sqrt(3) * MAX_OUTPUT_FACES))
        if edge < minimum_edge:
            raise ValueError('Заданная длина ребра превысит бюджет треугольников. Увеличьте длину.')
        target = max(4, math.ceil((mesh.area / edge) * 4 / (np.sqrt(3) * edge)))
        while len(mesh.faces) < target and len(mesh.faces) * 4 <= MAX_OUTPUT_FACES:
            mesh = _subdivide(mesh, 1, progress, cancelled)
        if len(mesh.faces) > target:
            mesh = _decimate(mesh, max(.001, target / len(mesh.faces)), True, progress, cancelled)
        mesh = _smooth(mesh, count, .1, True, None, progress, cancelled)
        report['target_faces'] = target
        report['warnings'].append('Это приближение плотности через разбиение, сглаживание и сокращение. Равносторонние треугольники, точная длина каждого ребра и неизменность формы не гарантируются.')
    elif operation == 'wrap':
        pitch = _number(parameters, 'voxel_size_mm', 1., 0, 1_000_000, positive=True)
        mesh, info = _wrap(mesh, pitch, progress, cancelled)
        report.update(info)
        report['warnings'].append('Воксельная оболочка меняет размеры, может объединить близкие поверхности и закрыть небольшие зазоры. Большое отверстие открытой поверхности может превратиться в край утолщённой оболочки.')
    elif operation == 'slivers':
        threshold = _number(parameters, 'min_angle_deg', 5., 0, 60., positive=True)
        selected = np.arange(len(mesh.faces)) if ids is None else ids
        result = []
        for start in range(0, len(selected), 100_000):
            _check(cancelled)
            batch = selected[start:start+100_000]
            triangles = mesh.vertices[mesh.faces[batch]]
            area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
            minimum = np.full(len(batch), np.pi)
            for corner in range(3):
                a = triangles[:, (corner+1) % 3] - triangles[:, corner]
                b = triangles[:, (corner+2) % 3] - triangles[:, corner]
                minimum = np.minimum(minimum, np.arctan2(area2, np.einsum('ij,ij->i', a, b)))
            result.extend(batch[np.degrees(minimum) < threshold].tolist())
        report['selected_faces'] = np.asarray(result, dtype=np.int64)
    elif operation == 'overlaps':
        tolerance = _number(parameters, 'tolerance_mm', 1e-8, 0, 1., positive=True)
        report.update(_overlaps(mesh, ids, tolerance, progress, cancelled))
    _check(cancelled)
    validate_mesh(mesh)
    if len(mesh.faces) > MAX_INPUT_FACES:
        raise ValueError('Результат превышает бюджет треугольников.')
    mesh.metadata = deepcopy(source.metadata)
    report['after'] = dict(vertices=len(mesh.vertices), faces=len(mesh.faces))
    report['changed'] = not (np.array_equal(source.vertices, mesh.vertices) and np.array_equal(source.faces, mesh.faces))
    if operation == 'normals':
        report['faces_flipped'] = int(np.count_nonzero(np.any(source.faces != mesh.faces, axis=1)))
    report['bounds_delta_mm'] = float(np.max(np.abs(source.bounds - mesh.bounds)))
    progress('Операция завершена.')
    return mesh, report
