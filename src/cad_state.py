"""Serializable native-CAD provenance attached to an immutable mesh snapshot.

This module does not parse BRep. It verifies that the tessellation and its CAD
payload still match their recorded fingerprints before native operations run.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import trimesh

CAD_KEY = 'cad_native'


def _hash_value(hasher, value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf' or (value.dtype.kind == 'f' and not np.isfinite(value).all()):
            raise ValueError('Данные CAD должны содержать только конечные числовые массивы.')
        array = np.ascontiguousarray(value)
        hasher.update(b'array:' + str((array.dtype.str, array.shape)).encode())
        if array.size:
            hasher.update(memoryview(array).cast('B'))
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError('Ключи данных CAD должны быть строками.')
        hasher.update(b'{')
        for key in sorted(value):
            _hash_value(hasher, key); _hash_value(hasher, value[key])
        hasher.update(b'}')
    elif isinstance(value, (list, tuple)):
        hasher.update(b'[')
        for item in value:
            _hash_value(hasher, item)
        hasher.update(b']')
    else:
        if isinstance(value, np.generic):
            value = value.item()
        try:
            hasher.update(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))
        except (TypeError, ValueError) as exc:
            raise ValueError('Данные CAD содержат неподдерживаемое или неконечное значение.') from exc
        hasher.update(b'\0')


def mesh_digest(mesh):
    """Hash exact vertex/face arrays; unrelated metadata does not affect validity."""
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Требуется треугольная сетка модели.')
    hasher = hashlib.sha256()
    _hash_value(hasher, (mesh.vertices, mesh.faces))
    return hasher.hexdigest()


def _payload_digest(payload):
    hasher = hashlib.sha256()
    _hash_value(hasher, {key: value for key, value in payload.items() if key != 'payload_digest'})
    return hasher.hexdigest()


def _matrix(value):
    try:
        matrix = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('Матрица CAD должна содержать конечные числа.') from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Матрица CAD должна быть конечной матрицей 4×4.')
    if not np.array_equal(matrix[3], [0., 0., 0., 1.]):
        raise ValueError('Проективное преобразование CAD не поддерживается.')
    singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if singular[-1] <= np.finfo(float).eps * singular[0]:
        raise ValueError('Вырожденное преобразование CAD не поддерживается.')
    return matrix


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _validate(payload, face_count):
    if not isinstance(payload, dict) or not _integer(payload.get('version')) or payload['version'] != 1:
        raise ValueError('Неизвестная или повреждённая версия данных CAD.')
    if payload.get('invalidated', False):
        raise ValueError('Связь с CAD утрачена после изменения сетки.')
    brep = payload.get('brep')
    if not isinstance(brep, np.ndarray) or brep.dtype != np.uint8 or brep.ndim != 1 or not brep.size:
        raise ValueError('В проекте отсутствуют корректные бинарные данные BRep.')
    _matrix(payload.get('matrix'))
    bodies, face_info = payload.get('bodies'), payload.get('face_info')
    if (not isinstance(bodies, list) or not all(isinstance(item, dict) for item in bodies)
            or not isinstance(face_info, list) or not face_info or not all(isinstance(item, dict) for item in face_info)):
        raise ValueError('Некорректное описание тел или поверхностей CAD.')
    for name in ('face_ids', 'body_ids'):
        values = payload.get(name)
        if not isinstance(values, np.ndarray) or values.dtype != np.int64 or values.shape != (face_count,):
            raise ValueError('Связь поверхностей CAD не соответствует треугольникам сетки.')
    face_ids, body_ids = payload['face_ids'], payload['body_ids']
    if (np.any(face_ids < 0) or np.any(face_ids >= len(face_info))
            or np.any(body_ids < -1) or np.any(body_ids >= len(bodies))):
        raise ValueError('Индекс тела или поверхности CAD вне допустимого диапазона.')
    # -1 denotes a surface with no owning solid (an open shell, for example).
    declared = np.full(len(face_info), -2, dtype=np.int64)
    for index, info in enumerate(face_info):
        if 'body_id' in info:
            body = info['body_id']
            if not _integer(body) or not -1 <= body < len(bodies):
                raise ValueError('Некорректная принадлежность поверхности телу CAD.')
            declared[index] = body
    assigned = declared[face_ids]
    if np.any((assigned != -2) & (assigned != body_ids)):
        raise ValueError('Принадлежность треугольников телам CAD не согласована с поверхностями.')
    for name in ('proxy_digest', 'payload_digest'):
        value = payload.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise ValueError('Контрольная сумма CAD отсутствует или повреждена.')


def require_native(mesh):
    """Return a verified payload or explain why native CAD operations are unsafe."""
    if not isinstance(mesh, trimesh.Trimesh) or CAD_KEY not in mesh.metadata:
        raise ValueError('У модели нет исходной CAD-геометрии. Загрузите STEP для этой операции.')
    payload = mesh.metadata[CAD_KEY]
    _validate(payload, len(mesh.faces))
    if payload['proxy_digest'] != mesh_digest(mesh):
        raise ValueError('Сетка CAD была изменена. Исходный BRep больше не соответствует модели; работайте с сеткой или загрузите STEP заново.')
    if payload['payload_digest'] != _payload_digest(payload):
        raise ValueError('Данные или привязки CAD изменены: контрольная сумма не совпадает.')
    return payload


def cad_status(mesh):
    """Return ``native``, ``modified`` or ``mesh`` without loading OpenCascade."""
    if not isinstance(mesh, trimesh.Trimesh) or CAD_KEY not in mesh.metadata:
        return 'mesh'
    try:
        require_native(mesh)
    except (ValueError, TypeError, IndexError, KeyError, np.linalg.LinAlgError):
        return 'modified'
    return 'native'


def attach_native(mesh, payload):
    """Bind a newly tessellated mesh to a private, validated copy of its payload."""
    from project_store import validate_mesh
    validate_mesh(mesh)
    if not isinstance(payload, dict):
        raise ValueError('Данные CAD должны быть словарём.')
    attached = deepcopy(payload)
    attached['matrix'] = _matrix(attached.get('matrix'))
    attached['proxy_digest'] = mesh_digest(mesh)
    attached['payload_digest'] = _payload_digest(attached)
    _validate(attached, len(mesh.faces))
    mesh.metadata[CAD_KEY] = attached
    return mesh


def apply_cad_transform(mesh, matrix):
    """Apply an affine transform in place, preserving only an already-valid CAD binding.

    Invalid bindings stay invalid even if a later transform happens to restore
    the original vertex coordinates. Neither a transform nor annotation changes
    can silently certify a modified tessellation as native CAD again.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError('Требуется треугольная сетка модели.')
    transform = _matrix(matrix)
    status = cad_status(mesh)
    payload = None
    if status == 'native':
        payload = deepcopy(mesh.metadata[CAD_KEY])
        with np.errstate(over='ignore', invalid='ignore'):
            payload['matrix'] = _matrix(transform @ payload['matrix'])
    # Refuse overflow before mutating the mesh, including ordinary STL inputs.
    with np.errstate(over='ignore', invalid='ignore'):
        for start in range(0, len(mesh.vertices), 100000):
            points = mesh.vertices[start:start + 100000] @ transform[:3, :3].T + transform[:3, 3]
            if not np.isfinite(points).all():
                raise ValueError('Преобразование создаёт неконечные координаты.')
    marking_data = mesh.metadata.get('marking_plans')
    marking_valid = False
    if marking_data:
        from marking_geometry import digest as marking_digest, transform_plans
        marking_valid = marking_data.get('digest') == marking_digest(mesh)
    mesh.apply_transform(transform)
    if marking_valid:
        transform_plans(mesh, transform, True)
    if payload is not None:
        payload['proxy_digest'] = mesh_digest(mesh)
        payload['payload_digest'] = _payload_digest(payload)
        mesh.metadata[CAD_KEY] = payload
    elif status == 'modified' and isinstance(mesh.metadata.get(CAD_KEY), dict):
        payload = deepcopy(mesh.metadata[CAD_KEY])
        payload['invalidated'] = True
        mesh.metadata[CAD_KEY] = payload
    return mesh


def strip_native(mesh):
    """Return an independent mesh copy without its native-CAD binding."""
    result = mesh.copy()
    result.metadata.pop(CAD_KEY, None)
    return result


def cad_face_triangles(mesh, triangle_id):
    """Find proxy triangles belonging to the same original CAD face as a hit."""
    payload = require_native(mesh)
    if not _integer(triangle_id) or not 0 <= triangle_id < len(mesh.faces):
        raise ValueError('Индекс выбранного треугольника вне модели.')
    return np.flatnonzero(payload['face_ids'] == payload['face_ids'][triangle_id]).astype(np.int64)
