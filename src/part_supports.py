"""Serializable child support groups owned by a slicer part."""
from copy import deepcopy
from uuid import uuid4
import numpy as np
import trimesh


def support_mesh(group):
    return trimesh.Trimesh(group['vertices'], group['faces'], process=False)


def make_group(mesh, surface_faces=(), kind='Точечные', params=None, contacts=0):
    return dict(id=str(uuid4()), kind=kind, surface_faces=list(map(int, surface_faces)),
                params=deepcopy(params or {}), contacts=int(contacts), visible=True,
                vertices=np.asarray(mesh.vertices).copy(), faces=np.asarray(mesh.faces).copy())


def transformed(groups, matrix):
    result = deepcopy(groups)
    for group in result:
        mesh = support_mesh(group)
        mesh.apply_transform(matrix)
        group['vertices'], group['faces'] = mesh.vertices.copy(), mesh.faces.copy()
    return result


def combined_mesh(part):
    meshes = [part['mesh']] + [support_mesh(group) for group in part.get('supports', []) if len(group['faces'])]
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def validate_groups(groups, face_count):
    if not isinstance(groups, list): raise ValueError('Некорректный список поддержек детали.')
    seen = set()
    from project_store import validate_mesh
    for group in groups:
        if not isinstance(group.get('id'), str) or group['id'] in seen: raise ValueError('Некорректный ID поддержки.')
        seen.add(group['id'])
        ids = np.asarray(group.get('surface_faces', []))
        if ids.size and (ids.dtype.kind not in 'iu' or ids.min() < 0 or ids.max() >= face_count):
            raise ValueError('Поверхность поддержки отсутствует в детали.')
        vertices, faces = np.asarray(group['vertices']), np.asarray(group['faces'])
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
            raise ValueError('Некорректная сетка поддержки.')
        if len(faces):
            if faces.min() < 0 or faces.max() >= len(vertices): raise ValueError('Индексы сетки поддержки вне диапазона.')
            validate_mesh(support_mesh(group))
        elif len(vertices): raise ValueError('В поддержке есть вершины без граней.')


def sync_actors(window):
    plotter = window.ui.slicer_plotter
    if plotter is None: return
    expected = set()
    for row, part in enumerate(window.slicer_parts):
        source = plotter.actors.get(part['actor_name'])
        for group in part.get('supports', []):
            if not len(group['faces']): continue
            name = 'part_support_' + group['id']
            expected.add(name)
            actor = plotter.actors.get(name)
            if actor is None:
                actor = plotter.add_mesh(window.trimesh_to_pyvista(support_mesh(group)), name=name,
                                         color='#86be68', pickable=False)
            actor.SetVisibility(bool(source and source.GetVisibility() and group.get('visible', True)))
            actor.GetMapper().RemoveAllClippingPlanes()
            for plane in window.ui.section_panel._planes: actor.GetMapper().AddClippingPlane(plane)
    for name in list(plotter.actors):
        if name.startswith('part_support_') and name not in expected: plotter.remove_actor(name)


def remove_actors(window, groups):
    if window.ui.slicer_plotter:
        for group in groups: window.ui.slicer_plotter.remove_actor('part_support_' + group['id'])
