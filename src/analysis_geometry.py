"""Read-only sampled inspections. Calculations never change project meshes."""
import math
import numpy as np
import trimesh
import open3d as o3d
from part_supports import support_mesh


def check_cancel(cancelled):
    if cancelled and cancelled(): raise InterruptedError('Анализ отменён.')


def prepared(mesh):
    result = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces.copy(), process=False)
    result.merge_vertices(digits_vertex=8); result.update_faces(result.nondegenerate_faces())
    result.update_faces(result.unique_faces()); result.remove_unreferenced_vertices()
    return result


def ray_scene(mesh):
    center = mesh.bounds.mean(axis=0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices - center, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32))))
    return scene, center


def wall_samples(mesh, count=2000, threshold=1., cancelled=None):
    if not 1 <= int(count) <= 20000 or not np.isfinite(threshold) or threshold <= 0: raise ValueError('Некорректные параметры толщины.')
    mesh = prepared(mesh); check_cancel(cancelled)
    rng = np.random.default_rng(42)
    faces = rng.choice(len(mesh.faces), size=int(count), p=mesh.area_faces / mesh.area)
    uv = rng.random((int(count), 2)); uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    triangles = mesh.triangles[faces]
    points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    normals = mesh.face_normals[faces]
    scale = max(float(mesh.extents.max()), 1e-9); epsilon = scale * 1e-5
    scene, center = ray_scene(mesh)
    # Use inward surface normals, not the nearest unrelated exterior point.
    rays = np.c_[points - center - normals * epsilon, -normals].astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    values = result['t_hit'].numpy().astype(float) + epsilon
    opposite = result['primitive_normals'].numpy()
    valid = np.isfinite(values) & (values > epsilon * 2) & ((opposite * normals).sum(axis=1) < -.1)
    values[~valid] = np.nan; check_cancel(cancelled)
    finite = values[np.isfinite(values)]
    return dict(points=points, values=values, source_faces=faces, count=len(values), hits=len(finite),
                minimum=float(finite.min()) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                thin=int(np.count_nonzero(finite < threshold)), threshold=float(threshold),
                reliable=bool(mesh.is_volume))


def point_thickness(mesh, point, normal):
    source = prepared(mesh); scene, center = ray_scene(source)
    normal = np.array(normal, dtype=float, copy=True)
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length < 1e-12: raise ValueError('Нормаль не определена.')
    normal /= length
    epsilon = max(float(source.extents.max()), 1e-9) * 1e-5
    ray = np.r_[np.asarray(point) - center - normal * epsilon, -normal].astype(np.float32)[None]
    hit = scene.cast_rays(o3d.core.Tensor(ray)); value = float(hit['t_hit'].numpy()[0]) + epsilon
    opposite = hit['primitive_normals'].numpy()[0]
    if not np.isfinite(value) or value <= epsilon * 2 or opposite @ normal > -.1:
        raise ValueError('Противоположная стенка вдоль нормали не найдена. Проверьте нормали и замкнутость.')
    return np.asarray(point) - normal * value, value


def collisions(records, *, direction=None, travel=20., steps=12, progress=None, cancelled=None):
    if len(records) > 50: raise ValueError('За один анализ выберите не более 50 деталей.')
    meshes = [prepared(r['mesh']) for r in records]
    result = []
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            check_cancel(cancelled)
            if progress: progress(f'Проверка: {records[i]["filename"]} / {records[j]["filename"]}')
            a, b = meshes[i], meshes[j]
            if not a.is_volume or not b.is_volume:
                result.append(dict(first=i, second=j, status='Объёмная проверка недоступна: открытая или несогласованная сетка.', volume=None)); continue
            offsets = [0.] if direction is None else np.linspace(0., float(travel), int(steps) + 1)
            hit = False
            for offset in offsets:
                check_cancel(cancelled); moved = a.copy()
                if direction is not None: moved.apply_translation(np.asarray(direction) * offset)
                if np.any(np.minimum(moved.bounds[1], b.bounds[1]) - np.maximum(moved.bounds[0], b.bounds[0]) <= 1e-7): continue
                intersection = trimesh.boolean.intersection([moved, b], engine='manifold')
                volume = abs(float(intersection.volume)) if not intersection.is_empty else 0.
                if volume > max(a.volume, b.volume) * 1e-10:
                    result.append(dict(first=i, second=j, status='Пересечение' if direction is None else 'Препятствие при перемещении первой детали', volume=volume, offset=float(offset)))
                    hit = True; break
            if not hit: result.append(dict(first=i, second=j, status='Объёмного пересечения не найдено', volume=0.))
    return result


def cavities(mesh, cancelled=None):
    mesh = prepared(mesh); components = mesh.split(only_watertight=False); shells = []
    for i, component in enumerate(components):
        check_cancel(cancelled)
        shells.append(dict(index=i + 1, closed=bool(component.is_watertight),
            signed_volume=float(component.volume) if component.is_watertight else None,
            area=float(component.area), triangles=len(component.faces)))
    return dict(shells=shells, inward_shells=sum(s['closed'] and s['signed_volume'] < 0 for s in shells),
                reliable=bool(mesh.is_watertight and mesh.is_winding_consistent))


def slice_distribution(records, layer_height=.05, samples=80, progress=None, cancelled=None):
    if not np.isfinite(layer_height) or layer_height <= 0: raise ValueError('Высота слоя должна быть положительной.')
    meshes = [prepared(r['mesh']) for r in records]
    for record in records:
        meshes += [prepared(support_mesh(group)) for group in record.get('supports', []) if len(group['faces'])]
    low = min(m.bounds[0, 2] for m in meshes); high = max(m.bounds[1, 2] for m in meshes)
    layers = max(1, int(math.ceil(max(0., high) / layer_height)))
    # Mid-layer planes avoid degeneracies on horizontal end faces.
    total = max(1, int(math.ceil((high - low) / layer_height)))
    indices = np.unique(np.linspace(0, total - 1, min(int(samples), total)).astype(int))
    heights = low + (indices + .5) * layer_height
    areas, perimeters, open_paths = [], [], 0
    for z in heights:
        check_cancel(cancelled)
        if progress: progress(f'Срез Z={z:.3f} мм')
        area, perimeter = 0., 0.
        for mesh in meshes:
            if not mesh.bounds[0, 2] < z < mesh.bounds[1, 2]: continue
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if section is None: continue
            path, _ = section.to_2D()
            perimeter += float(path.length)
            if path.is_closed:
                area += float(sum(poly.area for poly in path.polygons_full))
            else: open_paths += 1
        areas.append(area); perimeters.append(perimeter)
    return dict(z=heights.tolist(), area_mm2=areas, perimeter_mm=perimeters, layers=layers,
                layer_height=layer_height, open_paths=open_paths,
                note='Площади суммируются с поддержками; пересечения не объединяются. График — выборка слоёв.')


def build_estimate(parts, params):
    from display_tools import scene_statistics, material_estimate
    values = np.asarray([params[key] for key in ('layer_height', 'volume_rate', 'layer_seconds', 'setup_minutes', 'hour_price', 'density', 'material_price', 'fixed_cost')], dtype=float)
    if not np.isfinite(values).all() or (values[:2] <= 0).any() or (values[2:] < 0).any(): raise ValueError('Некорректные параметры оценки.')
    statistics = scene_statistics(parts)
    height = max((max([p['mesh'].bounds[1, 2]] + [support_mesh(g).bounds[1, 2] for g in p.get('supports', []) if len(g['faces'])]) for p in parts), default=0.)
    layers = max(0, math.ceil(max(0., height) / params['layer_height']))
    seconds = params['setup_minutes'] * 60 + layers * params['layer_seconds'] + statistics['total_mm3'] / params['volume_rate']
    mass, material = material_estimate(statistics['total_mm3'], params['density'], params['material_price'])
    machine = seconds / 3600 * params['hour_price']
    return dict(**statistics, height_mm=max(0., height), layers=layers, seconds=seconds,
                mass_g=mass, material_cost=material, machine_cost=machine,
                total_cost=material + machine + params['fixed_cost'], params=dict(params))
