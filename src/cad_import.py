"""STEP -> millimetre triangle mesh using OpenCascade, loaded only when needed."""
from pathlib import Path
import math

import numpy as np
import trimesh


def load_step(path, linear_deflection=0.05, angular_deflection=0.25):
    if not math.isfinite(linear_deflection) or linear_deflection <= 0:
        raise ValueError("Точность триангуляции должна быть положительной (мм).")
    if not math.isfinite(angular_deflection) or not 0 < angular_deflection < math.pi:
        raise ValueError("Некорректная угловая точность триангуляции.")
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRep import BRep_Tool
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_REVERSED
        from OCP.TopoDS import TopoDS
        from OCP.TopLoc import TopLoc_Location
    except ImportError as exc:
        raise ImportError("Для STEP установите зависимости проекта: python -m pip install -r requirements.txt") from exc

    reader = STEPControl_Reader()
    if reader.ReadFile(str(Path(path).resolve())) != IFSelect_RetDone:
        raise ValueError("Не удалось прочитать STEP. Проверьте формат и целостность файла.")
    # OCCT length-unit values are expressed in mm; 1.0 selects millimetres.
    reader.SetSystemLengthUnit(1.0)
    if reader.TransferRoots() == 0:
        raise ValueError("STEP не содержит импортируемой геометрии.")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("STEP содержит пустую геометрию.")
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    if not mesher.IsDone():
        raise ValueError("Не удалось построить сетку STEP.")
    vertices, triangles = [], []
    offset = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_count = 0
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            raise ValueError("Одна из поверхностей STEP не триангулируется. Импорт прерван, чтобы не потерять поверхность.")
        transform = location.Transformation()
        vertices.extend(triangulation.Node(i).Transformed(transform).Coord()
                        for i in range(1, triangulation.NbNodes() + 1))
        reverse = (face.Orientation() == TopAbs_REVERSED) != transform.IsNegative()
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            if reverse: b, c = c, b
            triangles.append((a + offset - 1, b + offset - 1, c + offset - 1))
        offset += triangulation.NbNodes()
        face_count += 1
        explorer.Next()
    if not triangles:
        raise ValueError("STEP не содержит поверхностей: одних кривых недостаточно для импорта детали.")
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(triangles), process=False)
    # Join exact/seam duplicates produced per face, without a tolerance tied to mesh resolution.
    mesh.merge_vertices(digits_vertex=9)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    body_count = 0
    while solids.More():
        body_count += 1
        solids.Next()
    mesh.metadata.update(source_format="STEP", source_name=Path(path).name, units="mm",
                         linear_deflection_mm=float(linear_deflection), angular_deflection_rad=float(angular_deflection),
                         cad_face_count=face_count, cad_body_count=body_count)
    return mesh
