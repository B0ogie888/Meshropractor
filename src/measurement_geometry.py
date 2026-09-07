"""Metric calculations in model coordinates, with explicit degenerate-input checks."""
import numpy as np


def distance(a, b):
    delta = np.asarray(b) - a
    return float(np.linalg.norm(delta)), delta


def project_plane(point, origin, normal):
    normal = np.array(normal, dtype=float, copy=True)
    length = np.linalg.norm(normal)
    if length < 1e-12: raise ValueError('Нормаль плоскости не определена.')
    normal /= length
    signed = float((np.asarray(point) - origin) @ normal)
    return np.asarray(point) - signed * normal, abs(signed)


def circle(points):
    points = np.asarray(points, dtype=float)
    a, b = points[1] - points[0], points[2] - points[0]
    normal = np.cross(a, b)
    squared = normal @ normal
    if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10 or squared < (np.linalg.norm(a) * np.linalg.norm(b) * 1e-6) ** 2:
        raise ValueError('Для окружности нужны три разные точки, не лежащие на одной прямой.')
    center = points[0] + (a @ a * np.cross(b, normal) + b @ b * np.cross(normal, a)) / (2 * squared)
    return center, float(np.linalg.norm(points[0] - center)), normal / np.sqrt(squared)


def angle(points):
    points = np.asarray(points, dtype=float)
    a, b = points[0] - points[1], points[2] - points[1]
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-10:
        raise ValueError('Для угла укажите три разные точки; вторая — вершина угла.')
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def normal_angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.asarray(a) @ b) / np.linalg.norm(a) / np.linalg.norm(b), 0, 1))))
