"""Surface-attached marking plans and closed relief meshes; no GUI objects."""
from copy import deepcopy
import hashlib
import math
import numpy as np
import trimesh
from shapely import constrained_delaunay_triangles, union_all
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, translate, scale
from analysis_geometry import check_cancel, ray_scene

KEY = 'marking_plans'
MAX_FACES = 250_000


def digest(mesh):
    h = hashlib.sha256()
    for values in (mesh.vertices, mesh.faces): h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def plans(mesh):
    data = mesh.metadata.get(KEY, {})
    if not data: return []
    if not isinstance(data,dict): raise ValueError('Некорректный план маркировки.')
    if data.get('digest') != digest(mesh):
        raise ValueError('Геометрия изменилась. Старые области маркировки нужно задать заново.')
    result = deepcopy(data.get('items', []))
    if not isinstance(result, list) or len(result) > 50: raise ValueError('Некорректный список областей маркировки.')
    for item in result:
        if not isinstance(item,dict) or not isinstance(item.get('name'),str) or not isinstance(item.get('params'),dict):
            raise ValueError('Некорректная область маркировки.')
        try:
            frame = np.asarray(item.get('frame'),float)
            if frame.shape!=(4,4) or not np.isfinite(frame).all() or abs(np.linalg.det(frame[:3,:3]))<1e-10:
                raise ValueError()
            values = item['params']
            for key in ('width','area_height','depth','text_size','resolution','spacing','angle','shift_x','shift_y','threshold','raster_size'):
                if key in values and (not isinstance(values[key],(float,int)) or not math.isfinite(values[key])): raise ValueError()
            if values.get('content','text') not in ('text','image','datamatrix') or values.get('align','center') not in ('left','center','right'):
                raise ValueError()
            for key in ('text','font','code','image'):
                if key in values and not isinstance(values[key],str): raise ValueError()
            if len(values.get('image',''))>2*1024**2: raise ValueError()
        except (TypeError,ValueError): raise ValueError('Некорректные параметры области маркировки.') from None
    return result


def store_plans(mesh, items):
    if len(items) > 50: raise ValueError('Допускается до 50 областей на детали.')
    if items: mesh.metadata[KEY] = dict(version=1, digest=digest(mesh), items=deepcopy(items))
    else: mesh.metadata.pop(KEY, None)


def transform_plans(mesh, matrix, valid):
    data = mesh.metadata.get(KEY)
    if not data or not valid: return
    for item in data.get('items', []):
        item['frame'] = (matrix @ np.asarray(item['frame'], float)).tolist()
    data['digest'] = digest(mesh)


def silhouette(params):
    shape = None
    for points in params.get('contours', []):
        poly = Polygon(points)
        if not poly.is_valid: poly = poly.buffer(0)
        shape = poly if shape is None else shape.symmetric_difference(poly)
    if shape is None or shape.is_empty: raise ValueError('Маркировка не содержит контуров.')
    width, height = float(params['width']), float(params['area_height'])
    if not .01 <= min(width, height) or max(width, height) > 10000:
        raise ValueError('Размер области должен быть от 0,01 до 10000 мм.')
    bounds = shape.bounds; w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
    if min(w, h) < 1e-9: raise ValueError('Пустой рисунок или текст.')
    shape = translate(shape, -(bounds[0]+bounds[2])/2, -(bounds[1]+bounds[3])/2)
    content = params.get('content', 'text')
    factor = float(params.get('text_size', 5.)) / float(params.get('glyph_height', h)) if content == 'text' else min(width/w, height/h)*.90
    if content=='datamatrix': factor = min(width/(w+4),height/(h+4))  # two quiet modules on every edge
    if params.get('fit', True): factor = min(factor, .9*width/w, .9*height/h)
    shape = scale(shape, factor, factor, origin=(0, 0))
    if params.get('circular') and content=='text':
        # Bend the baseline into an arc; each contour is densified before the
        # mapping so long straight font segments also follow the circle.
        radius = min(width, height) * .33
        if shape.bounds[2]-shape.bounds[0] > radius*math.tau*.9:
            raise ValueError('Текст длиннее окружности. Уменьшите шрифт или увеличьте область.')
        from shapely.ops import transform
        def bend(x, y, z=None):
            x, y = np.asarray(x), np.asarray(y)
            return (radius+y)*np.sin(x/radius), (radius+y)*np.cos(x/radius)
        shape = transform(bend, shape.segmentize(max(.1, radius/35)))
    elif params.get('align', 'center') != 'center':
        a, _, b, _ = shape.bounds
        shape = translate(shape, -width*.45-a if params['align']=='left' else width*.45-b)
    shape = rotate(shape, float(params.get('angle', 0)), origin=(0, 0))
    shape = translate(shape, float(params.get('shift_x', 0)), float(params.get('shift_y', 0)))
    region = Point(0, 0).buffer(1, quad_segs=64)
    region = scale(region, width/2, height/2) if params.get('circular') else box(-width/2, -height/2, width/2, height/2)
    if content=='datamatrix':
        if params.get('circular'): raise ValueError('Data Matrix требует прямоугольной области.')
        quiet = box(-(w+4)*factor/2,-(h+4)*factor/2,(w+4)*factor/2,(h+4)*factor/2)
        quiet = rotate(quiet,float(params.get('angle',0)),origin=(0,0))
        quiet = translate(quiet,float(params.get('shift_x',0)),float(params.get('shift_y',0)))
        if not region.buffer(1e-8).covers(quiet):
            raise ValueError('Data Matrix со свободной зоной выходит за область. Уберите сдвиг/поворот или измените размеры области.')
    shape = shape.intersection(region)
    if shape.is_empty: raise ValueError('Маркировка вышла за границы области.')
    return shape


def mask_contours(mask):
    """Union horizontal runs, keeping holes and avoiding one solid per pixel."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or max(mask.shape) > 384: raise ValueError('Слишком большой растр маркировки.')
    runs = []
    for y, row in enumerate(mask):
        changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
        for left, right in zip(changes[::2], changes[1::2]):
            runs.append(box(left, -y-1, right, -y))
    if not runs: raise ValueError('Рисунок пуст после пороговой обработки.')
    # Diagonal black pixels meet at a single point. Extruding that junction
    # creates a non-manifold edge; a 0.2% module inset separates those contacts.
    shape = union_all(runs).buffer(-.002, join_style='mitre')
    polygons = [shape] if shape.geom_type == 'Polygon' else shape.geoms
    result = []
    for polygon in polygons:
        result.append(list(polygon.exterior.coords))
        result.extend(list(ring.coords) for ring in polygon.interiors)
    return result


def build_relief(source, params, progress=lambda _:None, cancelled=lambda:False):
    check_cancel(cancelled)
    frame = np.asarray(params['frame'], float)
    if frame.shape != (4,4) or not np.isfinite(frame).all() or abs(np.linalg.det(frame[:3,:3])) < 1e-10:
        raise ValueError('Не задана корректная область на поверхности.')
    depth = float(params.get('depth', .4))
    step = float(params.get('resolution', 1.))
    if not .005 <= depth <= 1000 or not .05 <= step <= 20: raise ValueError('Некорректная глубина или точность проекции.')
    shape = silhouette(params)
    polygons = [shape] if shape.geom_type == 'Polygon' else [p for p in shape.geoms if p.geom_type=='Polygon']
    local_source = source.copy(); local_source.apply_transform(np.linalg.inv(frame))
    projected = bool(params.get('project', True))
    if projected:
        scene, center = ray_scene(local_source)
        import open3d as o3d
    ztop = local_source.bounds[1,2] + max(float(local_source.extents.max())*.02, 1.)
    pieces = []; face_count = 0
    progress('Построение рельефа на поверхности…')
    for polygon in polygons:
        check_cancel(cancelled)
        triangles = [np.asarray(t.exterior.coords)[:3] for t in constrained_delaunay_triangles(polygon).geoms]
        if not triangles: continue
        points = np.asarray(triangles).reshape(-1, 2)
        points, inv = np.unique(points, axis=0, return_inverse=True)
        verts = np.c_[points, np.zeros(len(points))]; faces = inv.reshape(-1,3)
        if projected:
            for _ in range(9):
                edges = verts[faces[:,[0,1,2]]] - verts[faces[:,[1,2,0]]]
                if np.linalg.norm(edges, axis=2).max() <= step: break
                if len(faces)*8 + face_count > MAX_FACES:
                    raise ValueError('Слишком подробная маркировка. Увеличьте шаг проекции или упростите рисунок.')
                verts, faces = trimesh.remesh.subdivide(verts, faces)
                check_cancel(cancelled)
            rays = np.c_[verts[:,:2], np.full(len(verts), ztop), np.zeros((len(verts),2)), -np.ones(len(verts))]
            rays[:,:3] -= center
            hits = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
            distances = hits['t_hit'].numpy().astype(float)
            normals = hits['primitive_normals'].numpy()
            surface = ztop - distances
            limit = max(float(params['width']), float(params['area_height']))
            if (not np.isfinite(surface).all() or (normals[:,2] < .05).any() or (np.abs(surface) > limit).any()):
                raise ValueError('Часть маркировки не попадает на выбранную сторону детали. Уменьшите область или поверните вид.')
        else: surface = np.zeros(len(verts))
        piece = trimesh.creation.extrude_triangulation(verts[:,:2], faces, 1.)
        # Extrusion may reorder vertices; sample unique XY once, map both caps.
        xy_lookup = {tuple(xy): z for xy, z in zip(verts[:,:2], surface)}
        z = np.array([xy_lookup[tuple(p[:2])] for p in piece.vertices])
        upper = piece.vertices[:,2] > .5
        inset = min(.05, depth*.2)
        if params.get('engrave') or params.get('through'):
            bottom = np.full(len(z), local_source.bounds[0,2]-inset) if params.get('through') else z-depth
            piece.vertices[:,2] = np.where(upper, z+inset, bottom)
        else: piece.vertices[:,2] = np.where(upper, z+depth, z-inset)
        piece.apply_transform(frame)
        piece.fix_normals(multibody=True)
        if not piece.is_volume: raise ValueError('Проекция дала некорректный рельеф. Уменьшите область или шаг проекции.')
        face_count += len(piece.faces)
        if face_count > MAX_FACES: raise ValueError('Маркировка превышает 250 тысяч треугольников.')
        pieces.append(piece)
    if not pieces: raise ValueError('Не удалось построить маркировку.')
    result = trimesh.util.concatenate(pieces)
    check_cancel(cancelled)
    return result


def merge_relief(source, relief, params, progress=lambda _:None, cancelled=lambda:False):
    from model_tool_geometry import boolean
    progress('Объединение маркировки с деталью…')
    # Refuse a floating relief, including a planar projection above a curved face.
    boolean([source, relief], 'intersection', cancelled)
    result = boolean([source, relief], 'difference' if params.get('engrave') or params.get('through') else 'union', cancelled)
    result.metadata = {k: deepcopy(v) for k,v in source.metadata.items() if k not in (KEY, 'surface_appearance')}
    # CAD bindings deliberately retain their old digest: cad_status marks them
    # modified instead of exporting a STEP that omits the new relief.
    return result
