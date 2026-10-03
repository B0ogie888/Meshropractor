"""Persistent colour and image layers, independent of CAD geometry and VTK."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image
import trimesh

KEY = 'surface_appearance'
MAX_PIXELS = 16 * 1024**2


def read_image(path):
    with Image.open(path) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError('Изображение слишком велико: максимум 16 мегапикселей.')
        rgba = source.convert('RGBA')
        image = Image.new('RGB', rgba.size, 'white')
        image.paste(rgba, mask=rgba.getchannel('A'))
        image.thumbnail((2048, 2048))
        return np.asarray(image).copy()


def validate_appearance(value, faces):
    if not isinstance(value, dict): raise ValueError('Некорректные данные текстур.')
    reference = np.asarray(value.get('faces'))
    colors = np.asarray(value.get('colors'))
    if reference.shape != (faces, 3) or reference.dtype.kind not in 'iu':
        raise ValueError('Привязка текстур не соответствует сетке.')
    if colors.shape != (faces, 4) or colors.dtype != np.uint8:
        raise ValueError('Некорректные цвета треугольников.')
    layers = value.get('layers')
    if not isinstance(layers, list) or len(layers) > 32: raise ValueError('Максимум 32 текстуры на деталь.')
    seen = set()
    for layer in layers:
        if not isinstance(layer, dict): raise ValueError('Некорректный слой текстуры.')
        identifier = layer.get('id')
        if not isinstance(identifier, str) or identifier in seen: raise ValueError('Некорректный ID текстуры.')
        seen.add(identifier)
        image, uv, mask = (np.asarray(layer.get(k)) for k in ('image', 'uv', 'mask'))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1 or image.shape[0] * image.shape[1] > MAX_PIXELS:
            raise ValueError('Некорректное изображение текстуры.')
        if uv.shape != (faces, 3, 2) or not np.isfinite(uv).all(): raise ValueError('Некорректная UV-развёртка.')
        if mask.shape != (faces,) or mask.dtype != bool: raise ValueError('Некорректный выбор поверхности текстуры.')
        if not isinstance(layer.get('params'), dict): raise ValueError('Некорректные параметры текстуры.')
        if not isinstance(layer.get('name'), str) or not isinstance(layer.get('path'), str) or not isinstance(layer.get('visible'), bool):
            raise ValueError('Некорректное описание текстуры.')


def appearance(mesh):
    value = mesh.metadata.get(KEY)
    if value is None: return None
    try:
        validate_appearance(value, len(mesh.faces))
        # Mirroring may reverse corners; geometry edits must not reuse unrelated faces.
        if not np.array_equal(np.sort(value['faces'], axis=1), np.sort(mesh.faces, axis=1)): return None
    except (ValueError, TypeError, KeyError): return None
    return value


def fresh(mesh, base=(211, 211, 211, 255)):
    value = appearance(mesh)
    if value is not None:
        value = deepcopy(value)
        old = value['faces']
        order = np.argmax(mesh.faces[:, :, None] == old[:, None, :], axis=2)
        for layer in value['layers']:
            layer['uv'] = np.take_along_axis(layer['uv'], order[:, :, None], axis=1)
        value['faces'] = np.asarray(mesh.faces).copy()
        return value
    colors = np.tile(np.asarray(base, dtype=np.uint8), (len(mesh.faces), 1))
    visual = mesh.visual
    if getattr(visual, 'defined', False) and visual.kind in ('face', 'vertex'):
        colors = np.asarray(visual.face_colors, dtype=np.uint8).copy()
    value = dict(faces=np.asarray(mesh.faces).copy(), colors=colors, layers=[])
    if getattr(visual, 'kind', None) == 'texture' and visual.uv is not None:
        material = visual.material
        image = getattr(material, 'image', None)
        if image is None: image = getattr(material, 'baseColorTexture', None)
        if image is not None:
            value['layers'].append(dict(id=str(uuid4()), name='Импортированная текстура',
                image=np.asarray(Image.fromarray(np.asarray(image)).convert('RGB')), uv=np.asarray(visual.uv)[mesh.faces].copy(),
                mask=np.ones(len(mesh.faces), dtype=bool), visible=True, path='', params={'projection': 'Исходная UV'}))
    return value


def face_mask(mesh, ids=None):
    mask = np.ones(len(mesh.faces), dtype=bool) if ids is None else np.zeros(len(mesh.faces), dtype=bool)
    if ids is not None:
        raw = np.asarray(list(ids))
        if raw.size and raw.dtype.kind not in 'iu': raise ValueError('Индексы поверхностей должны быть целыми.')
        indices = raw.astype(np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= len(mask)): raise ValueError('Поверхность отсутствует в детали.')
        mask[indices] = True
    if not mask.any(): raise ValueError('Выберите поверхности.')
    return mask


def project_uv(mesh, params, mask):
    points = np.asarray(mesh.triangles)
    low, high = points[mask].reshape(-1, 3).min(axis=0), points[mask].reshape(-1, 3).max(axis=0)
    size = np.maximum(high - low, 1e-12)
    normalized = (points - low) / size
    mode = params.get('projection', 'По граням')
    if mode in ('XY', 'XZ', 'YZ'):
        uv = normalized[:, :, {'XY': [0, 1], 'XZ': [0, 2], 'YZ': [1, 2]}[mode]]
    elif mode in ('Цилиндр', 'Сфера'):
        centered = points - (low + high) / 2
        u = np.arctan2(centered[:, :, 1], centered[:, :, 0]) / (2 * np.pi) + .5
        crossing = np.ptp(u, axis=1) > .5
        u[crossing] = np.where(u[crossing] < .5, u[crossing] + 1, u[crossing])
        v = normalized[:, :, 2] if mode == 'Цилиндр' else .5 + np.arcsin(np.clip(centered[:, :, 2] / np.maximum(np.linalg.norm(centered, axis=2), 1e-12), -1, 1)) / np.pi
        uv = np.stack([u, v], axis=2)
    elif mode == 'По граням':
        axis = np.abs(mesh.face_normals).argmax(axis=1)
        uv = np.empty((len(mesh.faces), 3, 2))
        for a, pair in enumerate(([1, 2], [0, 2], [0, 1])): uv[axis == a] = normalized[axis == a][:, :, pair]
    else: raise ValueError('Выберите поддерживаемую проекцию.')
    repeat = np.asarray([params.get('repeat_u', 1.), params.get('repeat_v', 1.)], dtype=float)
    offset = np.asarray([params.get('offset_u', 0.), params.get('offset_v', 0.)], dtype=float)
    angle = float(params.get('angle', 0.))
    if not np.isfinite(np.r_[repeat, offset, angle]).all() or (repeat <= 0).any() or (repeat > 8).any() or (np.abs(offset) > 1).any():
        raise ValueError('Повторение: 0.1–8; смещение: −1–1; параметры должны быть конечными.')
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return ((uv - .5) @ rotation.T) * repeat + .5 + offset


def add_layer(mesh, image, params=None, ids=None, name='Текстура', path='', base=(211, 211, 211, 255)):
    result = mesh.copy(); value = fresh(mesh, base); mask = face_mask(mesh, ids)
    image = np.asarray(image)
    params = deepcopy(params or {'projection': 'По граням'})
    if len(value['layers']) >= 32: raise ValueError('На детали уже 32 текстуры. Удалите ненужные.')
    layer = dict(id=str(uuid4()), name=name, image=image.copy(), uv=project_uv(mesh, params, mask),
                 mask=mask, visible=True, path=str(path), params=params)
    value['layers'].append(layer)
    validate_appearance(value, len(mesh.faces)); result.metadata[KEY] = value
    return result, layer['id']


def paint(mesh, color, ids=None, base=(211, 211, 211, 255)):
    result = mesh.copy(); value = fresh(mesh, base)
    rgba = np.asarray(color)
    if rgba.shape != (4,) or not np.isfinite(rgba).all() or (rgba < 0).any() or (rgba > 255).any(): raise ValueError('Некорректный цвет.')
    mask = face_mask(mesh, ids); value['colors'][mask] = rgba.astype(np.uint8)
    # Painting covers images in the painted region while keeping other areas.
    for layer in value['layers']: layer['mask'][mask] = False
    result.metadata[KEY] = value
    return result


def change_layer(mesh, identifier, action, ids=None, image=None, params=None, name=None):
    result = mesh.copy(); value = fresh(mesh)
    layer = next((item for item in value['layers'] if item['id'] == identifier), None)
    if layer is None: raise ValueError('Текстура отсутствует на детали.')
    if action == 'delete': value['layers'].remove(layer)
    elif action == 'clear': layer['mask'][face_mask(mesh, ids)] = False
    elif action == 'invert': layer['visible'] = not layer.get('visible', True)
    elif action == 'edit':
        if image is not None: layer['image'] = np.asarray(image).copy()
        if name is not None: layer['name'] = name
        if params is not None:
            layer['params'] = deepcopy(params)
            mask = layer['mask'] if layer['mask'].any() else np.ones(len(mesh.faces), dtype=bool)
            layer['uv'] = project_uv(mesh, params, mask)
    else: raise ValueError('Неизвестная операция текстуры.')
    validate_appearance(value, len(mesh.faces)); result.metadata[KEY] = value
    return result


def bake_colors(mesh, base=(211, 211, 211, 255)):
    """Bake triangle colours into an atlas; every triangle keeps its own texel."""
    value = fresh(mesh, base); count = len(mesh.faces); side = int(np.ceil(np.sqrt(count)))
    if side * side > MAX_PIXELS // 16: raise ValueError('Слишком много треугольников для атласа.')
    image = np.full((side, side, 3), 255, dtype=np.uint8)
    image.reshape(-1, 3)[:count] = effective_colors(mesh, base)[:, :3]
    image = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)
    uv = np.column_stack(((np.arange(count) % side + .5) / side, 1 - (np.arange(count) // side + .5) / side))
    layer = dict(id=str(uuid4()), name=Path('Цвета детали').name, image=image,
                 uv=np.repeat(uv[:, None, :], 3, axis=1), mask=np.ones(count, dtype=bool),
                 visible=True, path='', params={'projection': 'Атлас цветов'})
    value['layers'] = [layer]; value['colors'][:] = 255
    result = mesh.copy(); result.metadata[KEY] = value
    return result, layer['id']


def sample_image(image, uv):
    uv = np.asarray(uv)
    # Repeat; exact upper edge belongs to the final pixel of the preceding tile.
    wrapped = np.mod(uv, 1.)
    wrapped[np.isclose(wrapped, 0.) & (uv > 0)] = 1.
    x = np.clip(np.rint(wrapped[..., 0] * (image.shape[1] - 1)).astype(int), 0, image.shape[1] - 1)
    y = np.clip(np.rint((1 - wrapped[..., 1]) * (image.shape[0] - 1)).astype(int), 0, image.shape[0] - 1)
    return image[y, x]


def effective_colors(mesh, base=(211, 211, 211, 255)):
    value = fresh(mesh, base); colors = value['colors'].copy()
    for layer in value['layers']:
        if not layer.get('visible', True): continue
        mask = layer['mask']; colors[mask, :3] = sample_image(layer['image'], layer['uv'][mask].mean(axis=1))
    return colors


def split_colors(mesh, supports=(), base=(211, 211, 211, 255)):
    """Split colour regions with source-face provenance and unambiguous child supports."""
    from part_supports import support_mesh
    colors = effective_colors(mesh, base)
    palette, labels = np.unique(colors, axis=0, return_inverse=True)
    if len(palette) > 256: raise ValueError('Более 256 цветов. Сначала покрасьте поверхности в нужные цвета.')
    results = []
    for label, color in enumerate(palette):
        ids = np.flatnonzero(labels == label); vertices, inverse = np.unique(mesh.faces[ids], return_inverse=True)
        child = trimesh.Trimesh(mesh.vertices[vertices], inverse.reshape(-1, 3), process=False)
        child.metadata = deepcopy(mesh.metadata); child.metadata.pop('cad_native', None)
        child.metadata.pop(KEY, None)
        child = paint(child, color)
        results.append(dict(mesh=child, color=color.tolist(), supports=[], source_faces=ids))
    for group in supports:
        ids = np.asarray(group.get('surface_faces', []), dtype=int)
        if ids.size:
            owner_labels = np.unique(labels[ids])
            if len(owner_labels) != 1:
                raise ValueError('Одна группа поддержек относится к разным цветам. Разделите или удалите эту группу перед разделением детали.')
            owner = int(owner_labels[0])
        else:
            point = support_mesh(group).centroid
            owner = int(labels[np.argmin(np.linalg.norm(mesh.triangles_center - point, axis=1))])
        attached = deepcopy(group); attached.pop('cad_binding', None)
        mapping = {int(old): new for new, old in enumerate(results[owner]['source_faces'])}
        attached['surface_faces'] = [mapping[int(old)] for old in ids]
        results[owner]['supports'].append(attached)
    return results
