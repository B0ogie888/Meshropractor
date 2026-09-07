"""Serializable display-only section planes (kept half-space >= 0)."""
from copy import deepcopy
import numpy as np

AXES = {"XY": [0., 0., 1.], "XZ": [0., 1., 0.], "YZ": [1., 0., 0.]}


def default_sections():
    return [dict(active=False, axis=axis, normal=normal.copy(), position=0., step=1.,
                 cut="+", color=color) for (axis, normal), color in zip(AXES.items(), ("#ef6b62", "#66c997", "#61a9ef"))]


def validate_sections(sections):
    if not isinstance(sections, list) or len(sections) > 6:
        raise ValueError("Можно использовать не более шести секущих плоскостей.")
    result = deepcopy(sections)
    for plane in result:
        if not isinstance(plane, dict):
            raise ValueError("Некорректная секущая плоскость.")
        n = np.asarray(plane.get("normal"), dtype=float)
        if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-10:
            raise ValueError("Некорректная нормаль секущей плоскости.")
        if plane.get("axis") not in (*AXES, "Произв.") or plane.get("cut") not in ("+", "−"):
            raise ValueError("Некорректный тип сечения.")
        if not isinstance(plane.get("active"), bool):
            raise ValueError("Некорректная активность сечения.")
        if not np.isfinite(float(plane.get("position", np.nan))) or not np.isfinite(float(plane.get("step", np.nan))) or float(plane["step"]) <= 0:
            raise ValueError("Некорректная позиция или шаг сечения.")
        color = plane.get("color", "")
        if not isinstance(color, str) or len(color) != 7 or color[0] != "#" or any(c not in "0123456789abcdefABCDEF" for c in color[1:]):
            raise ValueError("Некорректный цвет сечения.")
        n /= np.linalg.norm(n)
        if plane["axis"] in AXES:
            n = np.asarray(AXES[plane["axis"]])
        plane["normal"] = n.tolist()
    return result


def clipping_plane(section):
    from vtkmodules.vtkCommonDataModel import vtkPlane
    normal = np.asarray(section["normal"], dtype=float)
    plane = vtkPlane()
    plane.SetOrigin(*(normal * section["position"]))
    plane.SetNormal(*(normal * (-1 if section["cut"] == "+" else 1)))
    return plane


def projected_range(bounds, normal):
    from itertools import product
    corners = np.array(list(product(*zip(bounds[0], bounds[1]))))
    positions = corners @ np.asarray(normal)
    return float(positions.min()), float(positions.max())
