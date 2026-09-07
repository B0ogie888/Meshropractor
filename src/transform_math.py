"""World-coordinate transforms used by the detailed placement dialogs."""
import numpy as np
from scipy.spatial.transform import Rotation


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(vector).all() or length < 1e-10:
        raise ValueError("Направление должно задаваться ненулевым вектором.")
    return vector / length


def plane_from_points(points):
    points = np.asarray(points, dtype=float)
    normal = unit(np.cross(points[1] - points[0], points[2] - points[0]))
    return points[0].copy(), normal


def selection_bounds(meshes):
    bounds = np.array([mesh.bounds for mesh in meshes])
    return np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])


def anchor(bounds, modes, custom):
    return np.array([bounds[0, i] if mode == 0 else bounds[:, i].mean() if mode == 1 else
                     bounds[1, i] if mode == 2 else custom[i] for i, mode in enumerate(modes)])


def matrices(meshes, operation, params):
    bounds = selection_bounds(meshes)
    result = []
    for mesh in meshes:
        matrix = np.eye(4)
        individual = params.get('individual', False)
        center = mesh.bounds.mean(axis=0) if individual else np.asarray(params.get('center', bounds.mean(axis=0)), dtype=float)
        if operation == 'Перемещать':
            start = anchor(mesh.bounds if individual else bounds, params['anchor_modes'], params['anchor_custom'])
            delta = np.asarray(params['target']) - start if params['absolute'] else np.asarray(params['values'], dtype=float)
            if params['along_line']:
                direction = unit(np.asarray(params['line_b']) - params['line_a'])
                delta = direction * np.dot(delta, direction)
            matrix[:3, 3] = delta
        else:
            if operation == 'Вращать':
                if params['along_line']:
                    direction = unit(np.asarray(params['line_b']) - params['line_a'])
                    rotation = Rotation.from_rotvec(direction * np.deg2rad(params['line_angle'])).as_matrix()
                    center = np.asarray(params['line_a'], dtype=float)
                else:
                    rotation = Rotation.from_euler('xyz', params['values'], degrees=True).as_matrix()
            elif operation == 'Масштабировать':
                factors = np.asarray(params['values'], dtype=float)
                if np.any(factors <= 0) or not np.isfinite(factors).all():
                    raise ValueError("Множители масштаба должны быть положительными.")
                rotation = np.diag(factors)
            elif operation == 'Отзеркалить':
                normal = unit(params['normal'])
                rotation = np.eye(3) - 2 * np.outer(normal, normal)
            else:
                raise ValueError("Неизвестное преобразование.")
            matrix[:3, :3] = rotation
            matrix[:3, 3] = center - rotation @ center
            if params.get('keep_z', False):
                transformed_z = mesh.vertices @ matrix[2, :3] + matrix[2, 3]
                matrix[2, 3] += mesh.bounds[0, 2] - transformed_z.min()
        if not np.isfinite(matrix).all():
            raise ValueError("Преобразование содержит некорректные числа.")
        result.append(matrix)
    return result
