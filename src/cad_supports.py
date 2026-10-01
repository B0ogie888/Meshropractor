"""Keep child supports attached to CAD faces across tessellation changes.

Whole CAD faces have exact stable IDs. A selection covering only part of a CAD
face cannot be transferred exactly to another tessellation: retain the physical
supports and their local reference points, but require explicit region selection.
No BRep or duplicate support geometry is stored in the binding.
"""
from copy import deepcopy
import hashlib

import numpy as np

from cad_state import CAD_KEY, require_native
from part_supports import validate_groups


def _ids(value, maximum, label):
    values = np.asarray(value)
    if values.ndim != 1 or (values.size and (values.dtype.kind not in 'iu'
            or values.min() < 0 or values.max() >= maximum)):
        raise ValueError(f'Некорректные индексы {label} поддержки.')
    return np.unique(values.astype(np.int64))


def _brep_digest(payload):
    return hashlib.sha256(memoryview(np.ascontiguousarray(payload['brep'])).cast('B')).hexdigest()


def _binding(group, mesh, payload, digest):
    selected = _ids(group.get('surface_faces', []), len(mesh.faces), 'треугольников')
    owners = payload['face_ids']
    faces, counts = np.unique(owners[selected], return_counts=True)
    totals = np.bincount(owners, minlength=len(payload['face_info']))
    whole = faces[counts == totals[faces]]
    partial = faces[counts != totals[faces]]
    partial_triangles = selected[np.isin(owners[selected], partial)]
    # CAD-local coordinates survive translations, rotations and affine scaling
    # of the owning part without updating its child support binding.
    centers = mesh.vertices[mesh.faces[partial_triangles]].mean(axis=1)
    inverse = np.linalg.inv(payload['matrix'])
    centers = centers @ inverse[:3, :3].T + inverse[:3, 3]
    return dict(version=1, brep_digest=digest, cad_face_ids=faces,
                full_face_ids=whole, partial_face_ids=partial,
                partial_triangle_face_ids=owners[partial_triangles].copy(),
                partial_triangle_centers=centers, requires_reselect=False, notice='')


def bind_support_surface(group, mesh):
    """Mutate and return a child group; ordinary/edited meshes use triangle IDs."""
    validate_groups([group], len(mesh.faces))
    if CAD_KEY not in mesh.metadata:
        group.pop('cad_binding', None)
        return group
    try:
        payload = require_native(mesh)
    except ValueError:
        group.pop('cad_binding', None)
        return group
    group['cad_binding'] = _binding(group, mesh, payload, _brep_digest(payload))
    return group


def _validate_binding(binding, group, mesh, payload, digest):
    if (not isinstance(binding, dict) or type(binding.get('version')) is not int
            or binding['version'] != 1 or binding.get('brep_digest') != digest):
        raise ValueError('Привязка поддержки относится к другой или повреждённой CAD-модели.')
    count = len(payload['face_info'])
    all_faces = _ids(binding.get('cad_face_ids'), count, 'поверхностей CAD')
    whole = _ids(binding.get('full_face_ids'), count, 'целых поверхностей CAD')
    partial = _ids(binding.get('partial_face_ids'), count, 'частичных поверхностей CAD')
    if (len(np.intersect1d(whole, partial))
            or not np.array_equal(np.union1d(whole, partial), all_faces)):
        raise ValueError('Полные и частичные поверхности поддержки не согласованы.')
    point_faces = np.asarray(binding.get('partial_triangle_face_ids'))
    centers = np.asarray(binding.get('partial_triangle_centers'))
    if (point_faces.ndim != 1 or point_faces.dtype.kind not in 'iu'
            or centers.shape != (len(point_faces), 3) or centers.dtype.kind not in 'iuf'
            or not np.isfinite(centers).all()
            or not np.array_equal(np.unique(point_faces), partial)):
        raise ValueError('Повреждены опорные точки частичной области поддержки.')
    pending = binding.get('requires_reselect')
    if type(pending) is not bool:
        raise ValueError('Некорректное состояние привязки CAD-поддержки.')
    actual = _binding(group, mesh, payload, digest)
    if (not np.array_equal(actual['full_face_ids'], whole)
            or not np.array_equal(actual['partial_face_ids'], [] if pending else partial)):
        raise ValueError('Привязка CAD-поддержки не соответствует выбранным треугольникам.')
    return whole, partial, pending


def rebind_supports(groups, source, newmesh):
    """Return independent support groups for a new proxy of the same placed BRep.

    This does not regenerate supports or move them. The caller must show a
    binding's ``notice`` when ``requires_reselect`` is true before region edits.
    It intentionally refuses unrelated BRep or changed placement, even when CAD
    face numbers happen to coincide.
    """
    old, new = require_native(source), require_native(newmesh)
    digest = _brep_digest(old)
    if (digest != _brep_digest(new) or len(old['face_info']) != len(new['face_info'])
            or not np.array_equal(old['matrix'], new['matrix'])):
        raise ValueError('Перепривязка поддержек допустима только при смене триангуляции той же CAD-модели в том же положении.')
    validate_groups(groups, len(source.faces))
    result = deepcopy(groups)
    unchanged = (old['proxy_digest'] == new['proxy_digest']
                 and np.array_equal(old['face_ids'], new['face_ids']))
    for group in result:
        binding = group.get('cad_binding')
        if binding is None:
            binding = _binding(group, source, old, digest)
            group['cad_binding'] = binding
        whole, partial, pending = _validate_binding(binding, group, source, old, digest)
        if unchanged and not pending:
            continue
        matches = np.flatnonzero(np.isin(new['face_ids'], whole))
        if not np.array_equal(np.unique(new['face_ids'][matches]), whole):
            raise ValueError('В новой триангуляции отсутствует CAD-поверхность поддержки.')
        group['surface_faces'] = matches.tolist()
        needs_selection = bool(len(partial) or not len(whole))
        binding['requires_reselect'] = needs_selection
        binding['notice'] = ('Геометрия поддержек сохранена. Частичная область CAD не переносится точно между '
                             'триангуляциями; выберите область заново перед редактированием поддержек.'
                             if needs_selection else '')
    validate_groups(result, len(newmesh.faces))
    return result
