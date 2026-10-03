"""Build a cached display atlas without changing source face IDs or CAD meshes."""
import numpy as np
import pyvista as pv
from PIL import Image
from texture_geometry import fresh


def display_data(mesh, source):
    value = fresh(mesh)
    layers = [layer for layer in value['layers'] if layer.get('visible', True) and layer['mask'].any()]
    if not layers:
        source.cell_data['_display_colors'] = value['colors']
        return source, None, 'face'
    plans = []
    for layer in layers:
        used = layer['uv'][layer['mask']]
        low = np.floor(used.reshape(-1, 2).min(axis=0))
        high = np.maximum(np.ceil(used.reshape(-1, 2).max(axis=0)), low + 1)
        tiles = (high - low).astype(int)
        if (tiles > 32).any(): raise ValueError('Слишком большой диапазон UV-развёртки.')
        image = layer['image']
        tile_size = np.minimum([image.shape[1], image.shape[0]], 1024)
        plans.append([layer, low, tiles, tile_size])
    widths = [int(p[2][0] * p[3][0]) + 4 for p in plans]
    heights = [int(p[2][1] * p[3][1]) + 4 for p in plans]
    width, height = max(widths) + 8, sum(heights) + 8
    factor = min(1., (16 * 1024**2 / (width * height))**.5, 8192 / height, 8192 / width)
    for p in plans: p[3] = np.maximum(1, (p[3] * factor).astype(int))
    width = max(int(p[2][0] * p[3][0]) + 4 for p in plans) + 8
    height = sum(int(p[2][1] * p[3][1]) + 4 for p in plans) + 8
    atlas = np.full((height, width, 3), 255, dtype=np.uint8)
    uv = np.full((len(mesh.faces), 3, 2), [2 / (width - 1), 1 - 2 / (height - 1)], dtype=float)
    colors = value['colors'].copy()
    top = 6
    for layer, low, tiles, tile_size in plans:
        tile = np.asarray(Image.fromarray(layer['image']).resize(tuple(tile_size), Image.Resampling.BILINEAR))
        tiled = np.tile(tile, (int(tiles[1]), int(tiles[0]), 1))
        h, w = tiled.shape[:2]
        atlas[top - 2:top + h + 2, 4:4 + w + 4] = np.pad(tiled, ((2, 2), (2, 2), (0, 0)), mode='edge')
        normalized = (layer['uv'] - low) / tiles
        coordinates = np.empty_like(normalized)
        coordinates[..., 0] = (6 + normalized[..., 0] * (w - 1)) / (width - 1)
        coordinates[..., 1] = 1 - (top + (1 - normalized[..., 1]) * (h - 1)) / (height - 1)
        mask = layer['mask']; uv[mask] = coordinates[mask]; colors[mask, :3] = 255
        top += h + 4
    # Explode only the display proxy: exact cell order is retained for picking.
    count = len(mesh.faces)
    data = pv.PolyData(np.asarray(mesh.triangles).reshape(-1, 3),
                       np.column_stack([np.full(count, 3), np.arange(count * 3).reshape(-1, 3)]).ravel())
    for name, array in source.cell_data.items(): data.cell_data[name] = array
    if 'Normals' in source.point_data:
        data.point_data['Normals'] = source.point_data['Normals'][mesh.faces].reshape(-1, 3)
        data.point_data.active_normals_name = 'Normals'
    data.active_texture_coordinates = uv.reshape(-1, 2)
    data.cell_data['_display_colors'] = colors
    texture = pv.Texture(atlas); texture.InterpolateOn(); texture.RepeatOff()
    return data, texture, 'face'
