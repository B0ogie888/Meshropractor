"""Validated duplication offsets shared by virtual previews and final copies."""
from itertools import product
import math
import numpy as np


def duplicate_plan(meshes, params, operation='Дублировать'):
    if not meshes: raise ValueError('Нет выбранных деталей.')
    counts = np.asarray(params['counts'],float)
    if counts.ndim!=1 or not len(counts) or not np.isfinite(counts).all() or np.any(counts<1) or np.any(counts!=np.floor(counts)):
        raise ValueError('Количество должно быть положительным целым.')
    matrix = bool(params.get('matrix_layout')) or operation=='Пакетное дублирование'
    if matrix and counts.shape!=(3,): raise ValueError('Задайте количество по X, Y и Z.')
    counts = [int(n) for n in counts]
    cells = math.prod(counts) if matrix else counts[0]+1
    copies = cells-1
    if copies*len(meshes)>1000: raise ValueError('За один раз можно создать не более 1000 новых деталей.')
    estimate = sum(m.vertices.nbytes+m.faces.nbytes for m in meshes)*copies
    if estimate>512*1024**2: raise ValueError('Массив слишком велик (более 512 МБ геометрии). Уменьшите количество копий.')
    if params.get('matrix_layout'):
        gaps = np.asarray(params['gaps'],float)
        if gaps.shape!=(3,) or not np.isfinite(gaps).all() or np.any(gaps<0):
            raise ValueError('Промежутки должны быть неотрицательными конечными числами.')
        bounds = np.asarray([m.bounds for m in meshes])
        steps = bounds[:,1].max(axis=0)-bounds[:,0].min(axis=0)+gaps
    else:
        steps = np.asarray(params['values'],float)
        if steps.shape!=(3,) or not np.isfinite(steps).all(): raise ValueError('Шаг должен содержать три конечные координаты.')
    positions = (cell for cell in product(*(range(n) for n in counts)) if any(cell)) if matrix else ([i,i,i] for i in range(1,counts[0]+1))
    offsets = np.asarray([steps*cell for cell in positions],float).reshape((-1,3))
    return dict(offsets=offsets,steps=steps,cells=cells,new_parts=copies*len(meshes),
                total_parts=cells*len(meshes),bytes=estimate)
