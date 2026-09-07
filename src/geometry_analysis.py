"""CPU geometry measurements independent of the user interface."""
import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh

from project_store import validate_mesh


def sample_surface(mesh, count, seed=42):
    """Sample by triangle area; count=0 uses the original CAD vertices."""
    geometry = validate_mesh(trimesh.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:], process=False))
    if count == 0:
        cloud = pv.PolyData(geometry.vertices.copy())
        cloud.point_data["Normals"] = geometry.vertex_normals.copy()
    else:
        # Local RNG avoids changing application-global NumPy randomness.
        rng = np.random.default_rng(seed)
        faces = rng.choice(len(geometry.faces), size=count, p=geometry.area_faces / geometry.area)
        uv = rng.random((count, 2))
        uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
        triangles = geometry.triangles[faces]
        points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
        cloud = pv.PolyData(points)
        cloud.point_data["Normals"] = geometry.face_normals[faces].copy()
    cloud.point_data.active_normals_name = "Normals"
    return cloud


def validate_deformation(source, result):
    faces = result.faces.reshape(-1, 4)[:, 1:]
    geometry = validate_mesh(trimesh.Trimesh(vertices=result.points, faces=faces, process=False))
    scale = max(float(np.max(geometry.extents)), 1e-12)
    if np.any(geometry.area_faces <= scale * scale * 1e-14):
        raise ValueError("Деформация создала вырожденные треугольники. Уменьшите коэффициент.")
    original = source.points[faces]
    deformed = result.points[faces]
    before = np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0])
    after = np.cross(deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0])
    if np.any(np.einsum("ij,ij->i", before, after) <= 0):
        raise ValueError("Обнаружен переворот треугольников. Уменьшите коэффициент или сгладьте поле.")
    check = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(geometry.vertices), o3d.utility.Vector3iVector(geometry.faces))
    if check.is_self_intersecting():
        raise ValueError("Результат содержит самопересечения. Проверьте исходную сетку и коэффициент.")


def compute_heatmap(cad_mesh, scan_mesh):
    validate_mesh(cad_mesh)
    validate_mesh(scan_mesh)
    if not cad_mesh.is_watertight:
        raise ValueError("Для знаковой карты нужен замкнутый CAD без отверстий. Исправьте сетку CAD.")
    geometry = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(cad_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(cad_mesh.faces, dtype=np.int32)))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(geometry)
    chunks = []
    for start in range(0, len(scan_mesh.vertices), 100_000):
        points = o3d.core.Tensor(np.asarray(scan_mesh.vertices[start:start + 100_000], dtype=np.float32))
        chunks.append(scene.compute_signed_distance(points).numpy())
    return np.concatenate(chunks)
