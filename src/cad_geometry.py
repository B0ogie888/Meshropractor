"""Native OpenCascade geometry behind an ordinary, editable Trimesh proxy.

BRep is kept in local coordinates without triangulations. Proxy triangles map
to deterministic body/face indices; affine transforms are kept separately.
Every public CAD operation rejects a proxy whose mesh has been edited.
"""
from copy import deepcopy
import io
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import trimesh


MAX_INPUT_BYTES = 256 * 1024**2
MAX_BREP_BYTES = 128 * 1024**2
MAX_NATIVE_FACES = 100_000
MAX_BODIES = 10_000
MAX_TRIANGLES = 5_000_000
MAX_VERTICES = 10_000_000
MAX_PROXY_BYTES = 512 * 1024**2


def _check(cancelled):
    if cancelled and cancelled():
        raise InterruptedError('Операция CAD отменена; исходная модель сохранена.')


def _progress(progress, message):
    if progress:
        progress(message)


def _precision(linear, angular):
    for value in (linear, angular):
        if isinstance(value, (bool, complex, np.bool_, np.complexfloating)) or not isinstance(value, (int, float, np.number)):
            raise ValueError('Точность триангуляции должна задаваться конечными числами.')
    if not math.isfinite(linear) or linear <= 0:
        raise ValueError('Точность триангуляции должна быть положительной (мм).')
    if not math.isfinite(angular) or not 0 < angular < math.pi:
        raise ValueError('Некорректная угловая точность триангуляции.')
    # Prevent accidental requests far below modelling precision from causing
    # unbounded native mesher allocations before its output can be inspected.
    if linear < 1e-6 or angular < 1e-4:
        raise ValueError('Точность слишком высокая для безопасной триангуляции: минимум 0.000001 мм и 0.0001 радиана.')
    return float(linear), float(angular)


def _serialize(shape):
    from OCP.BRepTools import BRepTools
    from OCP.TopTools import TopTools_FormatVersion_VERSION_3
    stream = io.BytesIO()
    BRepTools.Write_s(shape, stream, False, False, TopTools_FormatVersion_VERSION_3)
    data = stream.getvalue()
    if not data or len(data) > MAX_BREP_BYTES:
        raise ValueError('Нативная CAD-геометрия превышает допустимый размер 128 МБ.')
    return np.frombuffer(data, dtype=np.uint8).copy()


def _deserialize(data):
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    array = np.asarray(data)
    if array.dtype != np.uint8 or array.ndim != 1 or not 0 < len(array) <= MAX_BREP_BYTES:
        raise ValueError('Некорректные данные нативной CAD-геометрии.')
    shape = TopoDS_Shape()
    try:
        BRepTools.Read_s(shape, io.BytesIO(array.tobytes()), BRep_Builder())
    except Exception as exc:
        raise ValueError('Не удалось прочитать сохранённую CAD-геометрию.') from exc
    if shape.IsNull():
        raise ValueError('Сохранённая CAD-геометрия пуста.')
    return shape


def _faces(shape):
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    explorer, result = TopExp_Explorer(shape, TopAbs_FACE), []
    while explorer.More():
        if len(result) >= MAX_NATIVE_FACES:
            raise ValueError('Слишком много CAD-граней (более 100 000). Разделите документ.')
        result.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return result


def _topology(shape, cancelled=None):
    from OCP.BRep import BRep_Builder
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.TopoDS import TopoDS_Compound, TopoDS
    bodies = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        _check(cancelled)
        if len(bodies) >= MAX_BODIES:
            raise ValueError('Слишком много CAD-тел. Разделите документ.')
        solid = explorer.Current()
        bodies.append(dict(shape=solid, faces=_faces(solid), is_solid=True))
        explorer.Next()
    # Do not discard open sheets or faces beside closed solids in an assembly.
    explorer = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)
    sheets, builder, loose = TopoDS_Compound(), BRep_Builder(), []
    builder.MakeCompound(sheets)
    while explorer.More():
        _check(cancelled)
        face = TopoDS.Face_s(explorer.Current())
        builder.Add(sheets, face)
        loose.append(face)
        if len(loose) > MAX_NATIVE_FACES:
            raise ValueError('Слишком много CAD-поверхностей. Разделите документ.')
        explorer.Next()
    if loose:
        bodies.append(dict(shape=sheets, faces=loose, is_solid=False))
    if not bodies or sum(len(body['faces']) for body in bodies) > MAX_NATIVE_FACES:
        raise ValueError('CAD не содержит поверхностей или превышает лимит 100 000 граней.')
    return bodies


def _face_properties(face, *, detailed=False):
    from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_REVERSED, TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_Circle
    adaptor = BRepAdaptor_Surface(face)
    kind = adaptor.GetType()
    title = kind.name.removeprefix('GeomAbs_').lower()
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    result = dict(type=title, area_mm2=float(props.Mass()), center=list(props.CentreOfMass().Coord()))
    if kind == GeomAbs_Plane:
        surface = adaptor.Plane()
        normal = np.asarray(surface.Axis().Direction().Coord())
        if face.Orientation() == TopAbs_REVERSED:
            normal *= -1
        result.update(normal=normal.tolist(), point=list(surface.Location().Coord()))
    elif kind in (GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus):
        surface = {GeomAbs_Cylinder: adaptor.Cylinder, GeomAbs_Cone: adaptor.Cone,
                   GeomAbs_Sphere: adaptor.Sphere, GeomAbs_Torus: adaptor.Torus}[kind]()
        result.update(axis=list(surface.Axis().Direction().Coord()), location=list(surface.Location().Coord()))
        if kind == GeomAbs_Cone:
            result.update(radius_mm=float(surface.RefRadius()), semi_angle_deg=float(np.degrees(surface.SemiAngle())))
        elif kind == GeomAbs_Torus:
            result.update(major_radius_mm=float(surface.MajorRadius()), minor_radius_mm=float(surface.MinorRadius()))
        else:
            result['radius_mm'] = float(surface.Radius())
    if detailed:
        circles, seen = [], set()
        explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while explorer.More():
            curve = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()))
            if curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                key = tuple(np.round((*circle.Location().Coord(), circle.Radius()), 9))
                if key not in seen:
                    circles.append(dict(radius_mm=float(circle.Radius()), center=list(circle.Location().Coord()),
                                        axis=list(circle.Axis().Direction().Coord())))
                    seen.add(key)
            explorer.Next()
        result['circular_edges'] = circles
    return result


def _tessellate(shape, linear, angular, *, progress=None, cancelled=None, labels=None):
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopAbs import TopAbs_REVERSED
    bodies = _topology(shape, cancelled)
    BRepTools.Clean_s(shape)
    _check(cancelled)
    _progress(progress, 'Триангуляция CAD-поверхностей…')
    mesher = BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    _check(cancelled)
    if not mesher.IsDone():
        raise ValueError('Не удалось построить отображение CAD.')
    pieces, infos, body_info = [], [], []
    vertex_count = triangle_count = 0
    for body_id, body in enumerate(bodies):
        _check(cancelled)
        label = (labels or [])[body_id] if labels and body_id < len(labels) else {}
        description = dict(id=body_id, name=label.get('name') or f"{'Тело' if body['is_solid'] else 'Поверхности'} {body_id + 1}",
                           is_solid=body['is_solid'], face_count=len(body['faces']))
        if label.get('color') is not None:
            description['color'] = deepcopy(label['color'])
        body_info.append(description)
        for local_face, face in enumerate(body['faces']):
            _check(cancelled)
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is None or triangulation.NbTriangles() == 0:
                raise ValueError('Одна из CAD-граней не триангулируется. Импорт остановлен, чтобы не потерять поверхность.')
            face_id = len(infos)
            vertex_count += triangulation.NbNodes()
            triangle_count += triangulation.NbTriangles()
            if (vertex_count > MAX_VERTICES or triangle_count > MAX_TRIANGLES
                    or vertex_count * 24 + triangle_count * 40 > MAX_PROXY_BYTES):
                raise ValueError('Слишком плотная триангуляция CAD. Уменьшите качество (лимит 5 млн треугольников / 512 МБ).')
            info = dict(id=face_id, body_id=body_id, **_face_properties(face))
            face_labels = label.get('faces', [])
            if local_face < len(face_labels):
                for key in ('name', 'color'):
                    if face_labels[local_face].get(key) is not None:
                        info[key] = deepcopy(face_labels[local_face][key])
            infos.append(info)
            pieces.append((triangulation, location.Transformation(), face.Orientation(), face_id, body_id))
    vertices = np.empty((vertex_count, 3), dtype=np.float64)
    triangles = np.empty((triangle_count, 3), dtype=np.int64)
    face_ids, body_ids = np.empty(triangle_count, np.int64), np.empty(triangle_count, np.int64)
    v_offset = f_offset = 0
    for triangulation, transform, orientation, face_id, body_id in pieces:
        _check(cancelled)
        _progress(progress, f'CAD-грань {face_id + 1}/{len(pieces)}')
        for i in range(1, triangulation.NbNodes() + 1):
            if i % 8192 == 0: _check(cancelled)
            vertices[v_offset + i - 1] = triangulation.Node(i).Transformed(transform).Coord()
        reverse = (orientation == TopAbs_REVERSED) != transform.IsNegative()
        for i in range(1, triangulation.NbTriangles() + 1):
            if i % 8192 == 0: _check(cancelled)
            a, b, c = triangulation.Triangle(i).Get()
            triangles[f_offset + i - 1] = np.asarray((a, c, b) if reverse else (a, b, c)) + v_offset - 1
        count = triangulation.NbTriangles()
        face_ids[f_offset:f_offset + count] = face_id
        body_ids[f_offset:f_offset + count] = body_id
        v_offset += triangulation.NbNodes()
        f_offset += count
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.merge_vertices(digits_vertex=9)
    keep = mesh.nondegenerate_faces()
    mesh.update_faces(keep)
    face_ids, body_ids = face_ids[keep], body_ids[keep]
    mesh.remove_unreferenced_vertices()
    if len(np.unique(face_ids)) != len(infos):
        raise ValueError('После сшивки триангуляции пропала CAD-грань. Выберите другую точность.')
    return mesh, face_ids, body_ids, body_info, infos


def _from_brep(brep, linear, angular, *, metadata=None, matrix=None, labels=None, progress=None, cancelled=None):
    from cad_state import attach_native, apply_cad_transform
    _check(cancelled)
    shape = _deserialize(brep)
    # A local error d can grow by at most the largest singular value under an
    # affine transform. The angular setting remains in local CAD coordinates;
    # nonuniform scaling does not preserve angles.
    scale = float(np.linalg.svd(np.asarray(matrix)[:3, :3], compute_uv=False).max()) if matrix is not None else 1.
    local_linear, _ = _precision(linear / scale, angular)
    mesh, face_ids, body_ids, bodies, face_info = _tessellate(shape, local_linear, angular, progress=progress,
                                                           cancelled=cancelled, labels=labels)
    mesh.metadata = {key: deepcopy(value) for key, value in (metadata or {}).items() if key != 'cad_native'}
    mesh.metadata.update(source_format='STEP', units='mm', linear_deflection_mm=linear,
                         angular_deflection_rad=angular, cad_face_count=len(face_info),
                         cad_body_count=sum(body['is_solid'] for body in bodies),
                         linear_deflection_space='world', native_linear_deflection_mm=local_linear,
                         angular_deflection_space='local_cad')
    payload = dict(version=1, brep=np.asarray(brep, dtype=np.uint8), matrix=np.eye(4), face_ids=face_ids,
                   body_ids=body_ids, bodies=bodies, face_info=face_info)
    attach_native(mesh, payload)
    if matrix is not None and not np.array_equal(matrix, np.eye(4)):
        apply_cad_transform(mesh, matrix)
    _check(cancelled)
    return mesh


def _labels(payload):
    labels = deepcopy(payload['bodies'])
    for label in labels:
        label['faces'] = []
    for info in payload['face_info']:
        labels[info['body_id']]['faces'].append({key: info[key] for key in ('name', 'color') if key in info})
    return labels


def retessellate(mesh, linear_deflection=.05, angular_deflection=.25, *, progress=None, cancelled=None):
    from cad_state import require_native
    linear, angular = _precision(linear_deflection, angular_deflection)
    _check(cancelled)
    payload = require_native(mesh)
    return _from_brep(payload['brep'], linear, angular, metadata=mesh.metadata, matrix=payload['matrix'],
                      labels=_labels(payload), progress=progress, cancelled=cancelled)


def split_bodies(mesh, *, progress=None, cancelled=None):
    from cad_state import require_native
    _check(cancelled)
    payload = require_native(mesh)
    bodies = _topology(_deserialize(payload['brep']), cancelled)
    labels, result = _labels(payload), []
    for index, body in enumerate(bodies):
        _check(cancelled)
        _progress(progress, f'Выделение CAD-тела {index + 1}/{len(bodies)}')
        metadata = {key: value for key, value in mesh.metadata.items() if key != 'cad_native'}
        metadata = dict(metadata, cad_body_name=labels[index]['name'])
        if labels[index].get('color') is not None:
            metadata['cad_color'] = labels[index]['color']
        else:
            metadata.pop('cad_color', None)
        result.append(_from_brep(_serialize(body['shape']), float(mesh.metadata.get('linear_deflection_mm', .05)),
                                 float(mesh.metadata.get('angular_deflection_rad', .25)), metadata=metadata,
                                 matrix=payload['matrix'], labels=[labels[index]], progress=progress, cancelled=cancelled))
    _check(cancelled)
    return result


def _transform_native(shape, matrix):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_Transform
    from OCP.gp import gp_GTrsf, gp_Trsf
    if np.array_equal(matrix, np.eye(4)):
        return shape, None
    linear = np.asarray(matrix)[:3, :3]
    gram = linear.T @ linear
    scale2 = np.trace(gram) / 3
    if np.allclose(gram, np.eye(3) * scale2, rtol=1e-10, atol=max(scale2, 1e-300) * 1e-12):
        transform = gp_Trsf()
        transform.SetValues(*map(float, np.asarray(matrix)[:3].ravel()))
        builder = BRepBuilderAPI_Transform(shape, transform, True)
    else:
        transform = gp_GTrsf()
        for i in range(3):
            for j in range(4):
                transform.SetValue(i + 1, j + 1, float(matrix[i, j]))
        builder = BRepBuilderAPI_GTransform(shape, transform, True)
    if not builder.IsDone() or builder.Shape().IsNull():
        raise ValueError('Не удалось применить преобразование к CAD-геометрии.')
    return builder.Shape(), builder


def native_info(mesh, *, progress=None, cancelled=None):
    from cad_state import require_native
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopoDS import TopoDS
    _check(cancelled)
    payload = require_native(mesh)
    original = _deserialize(payload['brep'])
    bodies = _topology(original, cancelled)
    _progress(progress, 'Точный анализ CAD-геометрии…')
    transformed, builder = _transform_native(original, payload['matrix'])
    area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(transformed, area)
    volume, faces, body_info = 0., [], []
    for index, body in enumerate(bodies):
        _check(cancelled)
        description = deepcopy(payload['bodies'][index])
        if body['is_solid']:
            current = builder.ModifiedShape(body['shape']) if builder else body['shape']
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(current, props)
            description['volume_mm3'] = float(props.Mass())
            volume += props.Mass()
        for face in body['faces']:
            _check(cancelled)
            face_id = len(faces)
            current_face = TopoDS.Face_s(builder.ModifiedShape(face)) if builder else face
            info = {key: value for key, value in payload['face_info'][face_id].items()
                    if key in ('id', 'body_id', 'name', 'color')}
            info.update(_face_properties(current_face, detailed=True))
            faces.append(info)
        body_info.append(description)
    valid = bool(BRepCheck_Analyzer(transformed).IsValid())
    _check(cancelled)
    return dict(area_mm2=float(area.Mass()), volume_mm3=float(volume) if all(b['is_solid'] for b in bodies) else None,
                solid_volume_mm3=float(volume),
                solid_body_count=sum(b['is_solid'] for b in bodies), body_count=len(bodies), face_count=len(faces),
                is_valid=valid, bodies=body_info, face_info=faces)


def _name(label):
    # FindAttribute(output_handle) is unsafe with this OCP binding; attribute
    # iteration returns the actual typed handle without an output argument.
    from OCP.TDF import TDF_AttributeIterator
    from OCP.TDataStd import TDataStd_Name
    iterator = TDF_AttributeIterator(label)
    while iterator.More():
        attribute = iterator.Value()
        if isinstance(attribute, TDataStd_Name):
            return attribute.Get().ToExtString()
        iterator.Next()
    return None


def _color(color_tool, shape):
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf
    color = Quantity_Color()
    for mode in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        if color_tool.GetColor(shape, mode, color):
            return [float(color.Red()), float(color.Green()), float(color.Blue())]
    return None


def _read_step(path, progress=None, cancelled=None):
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
    from OCP.TDF import TDF_LabelSequence
    from OCP.IFSelect import IFSelect_RetDone
    _check(cancelled)
    source = Path(path).resolve()
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_INPUT_BYTES:
        raise ValueError('STEP отсутствует, пуст или превышает допустимый размер 256 МБ.')
    _progress(progress, 'Чтение STEP с CAD-телами и поверхностями…')
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError('Не удалось прочитать STEP. Проверьте формат и целостность файла.')
    reader.Reader().SetSystemLengthUnit(1.)
    document = TDocStd_Document(TCollection_ExtendedString('BinXCAF'))
    _check(cancelled)
    if not reader.Transfer(document):
        raise ValueError('STEP не содержит импортируемой CAD-геометрии.')
    shape = reader.Reader().OneShape()
    if shape.IsNull():
        raise ValueError('STEP содержит пустую геометрию.')
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    root_labels = []
    for index in range(1, roots.Length() + 1):
        label = roots.Value(index)
        root_labels.append((XCAFDoc_ShapeTool.GetShape_s(label), _name(label)))
    labels = []
    for body in _topology(shape, cancelled):
        label = dict(faces=[])
        for root_shape, root_name in root_labels:
            if root_shape.IsSame(body['shape']):
                label['name'] = root_name
                break
        label['color'] = _color(color_tool, body['shape'])
        for face in body['faces']:
            label['faces'].append(dict(color=_color(color_tool, face)))
        labels.append(label)
    _check(cancelled)
    return shape, labels


def export_step(meshes, path, *, progress=None, cancelled=None):
    """Write genuine BRep geometry atomically; no triangle-to-solid conversion."""
    from cad_state import require_native
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorGen, XCAFDoc_ColorSurf
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.Interface import Interface_Static
    _check(cancelled)
    meshes = list(meshes)
    if not meshes or len(meshes) > MAX_BODIES:
        raise ValueError('Выберите от 1 до 10 000 нативных CAD-деталей.')
    payloads = [require_native(mesh) for mesh in meshes]
    document = TDocStd_Document(TCollection_ExtendedString('BinXCAF'))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    body_count = face_count = 0
    for index, payload in enumerate(payloads):
        _check(cancelled)
        _progress(progress, f'Подготовка CAD-экспорта: {index + 1}/{len(payloads)}')
        original = _deserialize(payload['brep'])
        transformed, builder = _transform_native(original, payload['matrix'])
        if not BRepCheck_Analyzer(transformed).IsValid():
            raise ValueError('CAD-геометрия некорректна после преобразования; STEP не записан.')
        topology, labels = _topology(original, cancelled), _labels(payload)
        for body_id, body in enumerate(topology):
            _check(cancelled)
            current = builder.ModifiedShape(body['shape']) if builder and body['is_solid'] else body['shape']
            # A synthetic compound of loose faces is not part of the builder's
            # history. Transform it independently when exporting sheets.
            if builder and not body['is_solid']:
                current, _ = _transform_native(body['shape'], payload['matrix'])
            label = shape_tool.AddShape(current, False)
            info = labels[body_id]
            TDataStd_Name.Set_s(label, TCollection_ExtendedString(info.get('name') or f'Body {body_count + 1}', True))
            if info.get('color') is not None:
                color_tool.SetColor(label, Quantity_Color(*map(float, info['color'][:3]), Quantity_TOC_RGB), XCAFDoc_ColorGen)
            for face, face_label in zip(body['faces'], info['faces']):
                current_face = builder.ModifiedShape(face) if builder else face
                if face_label.get('color') is not None or face_label.get('name'):
                    sub = shape_tool.AddSubShape(label, current_face)
                    if sub.IsNull(): continue
                    if face_label.get('name'):
                        TDataStd_Name.Set_s(sub, TCollection_ExtendedString(face_label['name'], True))
                    if face_label.get('color') is not None:
                        color_tool.SetColor(sub, Quantity_Color(*map(float, face_label['color'][:3]), Quantity_TOC_RGB), XCAFDoc_ColorSurf)
            body_count += 1
            face_count += len(body['faces'])
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.step', dir=destination.parent)
    os.close(fd)
    previous_unit = Interface_Static.CVal_s('write.step.unit')
    try:
        _check(cancelled)
        Interface_Static.SetCVal_s('write.step.unit', 'MM')
        writer = STEPCAFControl_Writer()
        writer.SetNameMode(True)
        writer.SetColorMode(True)
        _progress(progress, 'Запись STEP с точной CAD-геометрией…')
        if not writer.Transfer(document, STEPControl_AsIs):
            raise ValueError('Не удалось подготовить STEP.')
        _check(cancelled)
        if writer.Write(str(temporary)) != IFSelect_RetDone:
            raise ValueError('Не удалось записать STEP.')
        _check(cancelled)
        with open(temporary, 'r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Interface_Static.SetCVal_s('write.step.unit', previous_unit or 'MM')
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dict(parts=len(meshes), body_count=body_count, face_count=face_count, path=str(destination),
                warnings=['Экспортированы нативные CAD-тела и поверхности. Сеточные поддержки в STEP не включены.'])
