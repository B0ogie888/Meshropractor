"""Mesh editing and implicit structures. Independent from GUI, inputs immutable."""
import numpy as np
import trimesh
from scipy import ndimage
from analysis_geometry import prepared, ray_scene, check_cancel

MAX_CELLS = 3_000_000
MAX_FACES = 2_000_000


def number(params,key,default,minimum=0,positive=True):
    value=float(params.get(key,default))
    if not np.isfinite(value) or value<minimum or (positive and value==minimum):
        raise ValueError('Некорректный параметр: '+key)
    return value


def solid(source):
    mesh=prepared(source)
    if not mesh.is_volume: raise ValueError('Нужна замкнутая сетка с согласованными нормалями. Сначала выполните исправление.')
    return mesh


def boolean(meshes,kind,cancelled=None):
    meshes=[solid(m) for m in meshes]
    if len(meshes)<2: raise ValueError('Выберите минимум две детали; первая — основная, остальные — инструменты.')
    bounds=np.asarray([m.bounds for m in meshes]);center=(bounds[:,0].min(0)+bounds[:,1].max(0))/2
    scale=float(np.ptp(bounds.reshape(-1,3),axis=0).max())
    for mesh in meshes: mesh.vertices=(mesh.vertices-center)/scale
    check_cancel(cancelled)
    result=getattr(trimesh.boolean,kind)(meshes,engine='manifold')
    check_cancel(cancelled)
    if result.is_empty or not result.is_volume: raise ValueError('Результат пустой или не является замкнутым телом.')
    result.vertices=result.vertices*scale+center;return result


def field_grid(source,step,padding,progress,cancelled):
    mesh=solid(source)
    lower=np.floor((mesh.bounds[0]-padding)/step)*step
    shape=np.ceil((mesh.bounds[1]+padding-lower)/step).astype(int)+1
    if np.prod(shape.astype(object))>MAX_CELLS:
        raise ValueError(f'Слишком мелкая ячейка: {int(np.prod(shape)):,} узлов (предел {MAX_CELLS:,}). Увеличьте шаг расчёта.')
    axes=[lower[i]+np.arange(shape[i])*step for i in range(3)]
    scene,center=ray_scene(mesh)
    import open3d as o3d
    field=np.empty(int(np.prod(shape)),dtype=np.float32)
    for start in range(0,len(field),65536):
        check_cancel(cancelled)
        ids=np.arange(start,min(start+65536,len(field)))
        coords=np.array(np.unravel_index(ids,shape)).T
        points=np.column_stack([axes[i][coords[:,i]] for i in range(3)])
        field[start:start+len(ids)]=scene.compute_signed_distance(o3d.core.Tensor((points-center).astype(np.float32))).numpy()
        progress(f'Расчёт поля: {100*min(start+65536,len(field))//len(field)}%')
    return field.reshape(shape),axes,lower


def contour(field,origin,step,cancelled):
    check_cancel(cancelled)
    if not (np.min(field)<0<np.max(field)): raise ValueError('Параметры не оставляют поверхности. Измените толщину или размер ячейки.')
    import pyvista as pv
    grid=pv.ImageData(dimensions=field.shape,spacing=(step,)*3,origin=origin)
    grid.point_data['distance']=np.asarray(field,dtype=np.float32).ravel(order='F')
    # Avoid the degenerate triangles FlyingEdges creates when the isosurface
    # passes exactly through a grid node. The displacement is < 0.01% of a cell.
    surface=grid.contour([step*1e-4],scalars='distance',method='flying_edges').triangulate()
    check_cancel(cancelled)
    if surface.n_cells>MAX_FACES: raise ValueError('Результат превышает 2 млн треугольников. Увеличьте расчётную ячейку.')
    mesh=trimesh.Trimesh(surface.points,np.asarray(surface.faces).reshape(-1,4)[:,1:],process=True)
    mesh.update_faces(mesh.nondegenerate_faces());mesh.update_faces(mesh.unique_faces());mesh.remove_unreferenced_vertices()
    if mesh.volume<0: mesh.invert()
    if not mesh.is_volume: raise ValueError('Не получено корректное замкнутое тело. Измените шаг или параметры.')
    return mesh


def implicit(source,operation,params,progress,cancelled):
    step=number(params,'step',.4)
    wall=number(params,'wall',1.,positive=False)
    offset=number(params,'offset',1.,minimum=-1e6)
    radius=number(params,'radius',1.)
    gap=number(params,'gap',.2,positive=False)
    padding=max(wall,gap+wall,abs(offset),radius)*2+step*3
    field,axes,origin=field_grid(source,step,padding,progress,cancelled)
    if operation in ('hollow','shell_core','formfit') and wall<2*step:
        raise ValueError('Толщина должна быть минимум две расчётные ячейки; уменьшите шаг.')
    if operation=='hollow': result=np.maximum(field,-field-wall)
    elif operation=='shell_core':
        return [contour(np.maximum(field,-field-wall),origin,step,cancelled),contour(field+wall,origin,step,cancelled)]
    elif operation=='formfit': result=np.maximum(field-gap-wall,-field+gap)
    elif operation in ('round','round_offset'):
        if radius<2*step: raise ValueError('Радиус должен быть минимум две расчётные ячейки.')
        inside=field<(offset if operation=='round_offset' else 0)
        eroded=ndimage.distance_transform_edt(inside,sampling=step)>radius
        check_cancel(cancelled)
        if not eroded.any(): raise ValueError('Радиус слишком велик: сердцевина исчезла.')
        result=ndimage.distance_transform_edt(~eroded,sampling=step)-radius
    else:
        cell=number(params,'cell',5.);thickness=number(params,'thickness',1.)
        if thickness<2*step or cell<=thickness*2:
            raise ValueError('Толщина рёбер должна быть ≥ 2 ячеек расчёта, размер структуры — > 2 толщин.')
        # Coordinates relative to part bounds keep patterns stable at large world coordinates.
        local=[(axis-source.bounds[0,i])/cell for i,axis in enumerate(axes)]
        x,y,z=local[0][:,None,None],local[1][None,:,None],local[2][None,None,:]
        d=[np.abs((v+.5)%1-.5)*cell for v in (x,y,z)]
        if operation=='honeycomb':
            side=cell/np.sqrt(3);xx=x*cell;yy=y*cell
            lattice=np.full((len(axes[0]),len(axes[1]),1),np.inf)
            col=np.rint(xx/(1.5*side))
            for dc in (-1,0,1):
                c=col+dc;row=np.rint(yy/cell-.5*(c%2))
                for dr in (-1,0,1):
                    dx=xx-c*1.5*side;dy=yy-(row+dr+.5*(c%2))*cell
                    # Regular hexagon with a horizontal pair of vertices.
                    edge=np.maximum(np.abs(dy),np.maximum(np.abs(np.sqrt(3)*dx+dy)/2,np.abs(np.sqrt(3)*dx-dy)/2))-cell/2
                    lattice=np.minimum(lattice,np.abs(edge))
            lattice=lattice-thickness/2
        elif operation in ('slice_lattice','tetra_slices'):
            if operation=='slice_lattice':
                stripe=np.where((np.floor(z)%2)==0,d[0],d[1])
                lattice=np.sqrt(stripe**2+d[2]**2)-thickness/2
            else:
                diagonal=np.abs(((x+np.where(np.floor(z)%2==0,y,-y))+.5)%1-.5)*cell/np.sqrt(2)
                lattice=np.sqrt(diagonal**2+d[2]**2)-thickness/2
        elif operation=='tetra':
            lattice=np.full(field.shape,np.inf,dtype=np.float32)
            # Four periodic body-diagonal families; geometric beams, not DSM material profiles.
            for sy,sz in ((1,1),(-1,1),(1,-1),(-1,-1)):
                u=(x-sy*y);v=(x-sz*z)
                for a in (-1,0,1):
                    for b in (-1,0,1):
                        du=(u+.5)%1-.5+a;dv=(v+.5)%1-.5+b
                        lattice=np.minimum(lattice,cell*np.sqrt(np.maximum(0,(2*du**2+2*dv**2-2*du*dv)/3)))
            lattice-=thickness/2
        else:
            lattice=np.minimum(np.sqrt(d[0]**2+d[1]**2),np.minimum(np.sqrt(d[0]**2+d[2]**2),np.sqrt(d[1]**2+d[2]**2)))-thickness/2
        result=np.maximum(field,lattice)
        if wall:
            if wall<2*step: raise ValueError('Наружная стенка должна быть ≥ 2 расчётных ячеек или равна нулю.')
            result=np.minimum(result,np.maximum(field,-field-wall))
    check_cancel(cancelled);progress('Построение поверхности…')
    return [contour(result,origin,step,cancelled)]


def extruded_surface(source,ids,distance):
    if not ids: raise ValueError('Выделите поверхность перед запуском команды.')
    patch=source.submesh([np.asarray(ids,dtype=int)],append=True,repair=False)
    patch=prepared(patch)
    vector=np.sum(patch.face_normals*patch.area_faces[:,None],axis=0)
    if np.linalg.norm(vector)<1e-9: raise ValueError('Средняя нормаль не определена. Выберите поверхность с одним направлением.')
    vector=vector/np.linalg.norm(vector)*distance
    count=len(patch.vertices);vertices=np.vstack([patch.vertices,patch.vertices+vector])
    edges=patch.edges[np.bincount(patch.edges_unique_inverse)[patch.edges_unique_inverse]==1]
    if not len(edges): raise ValueError('Нужен открытый участок поверхности, а не вся замкнутая оболочка.')
    faces=np.vstack([patch.faces[:,::-1],patch.faces+count,np.c_[edges,edges[:,1]+count],np.c_[edges[:,0],edges[:,1]+count,edges[:,0]+count]])
    prism=trimesh.Trimesh(vertices,faces,process=True)
    if prism.volume<0: prism.invert()
    if not prism.is_volume: raise ValueError('Участок не образует корректную призму. Выберите плоскую связную поверхность.')
    return prism,vector


def label_geometry(params,cancelled):
    from shapely.geometry import Polygon
    from shapely import constrained_delaunay_triangles
    shape=None
    for contour_points in params.get('text_contours',[]):
        polygon=Polygon(contour_points)
        if not polygon.is_valid:polygon=polygon.buffer(0)
        if not polygon.is_empty:shape=polygon if shape is None else shape.symmetric_difference(polygon)
    if shape is None or shape.is_empty:raise ValueError('Текст не образует контуров. Измените текст или шрифт.')
    height=number(params,'height',1.);pieces=[]
    for polygon in ([shape] if shape.geom_type=='Polygon' else shape.geoms):
        check_cancel(cancelled)
        triangles=[np.asarray(triangle.exterior.coords)[:3] for triangle in constrained_delaunay_triangles(polygon).geoms]
        if not triangles:continue
        vertices=np.asarray(triangles).reshape(-1,2);faces=np.arange(len(vertices)).reshape(-1,3)
        vertices,inverse=np.unique(vertices,axis=0,return_inverse=True)
        piece=trimesh.creation.extrude_triangulation(vertices,inverse[faces],height)
        piece.apply_translation(params['position']);pieces.append(piece)
    if not pieces:raise ValueError('Не удалось построить текст.')
    return trimesh.util.concatenate(pieces)


def calculate(records,operation,params,selection=None,vertex_ids=None,progress=lambda _:None,cancelled=lambda:False):
    selection=selection or {};items=[];mode='replace';warnings=[]
    if not records or len(records)>200:raise ValueError('Выберите от 1 до 200 деталей для одной операции.')
    if sum(len(record['mesh'].faces) for record in records)>MAX_FACES:raise ValueError('Выбранные модели превышают бюджет 2 млн треугольников.')
    adds=operation in ('surface_array','struts','rapidfit','formfit') or (operation=='label' and params.get('standalone'))
    if not adds and any(record.get('supports') for record in records):
        raise ValueError('Сначала удалите поддержки изменяемой детали, чтобы не потерять привязки к поверхностям.')
    if adds:mode='add'
    if operation in ('merge','union','difference','intersection','remove_volume'):
        source=[r['mesh'] for r in records]
        mesh=trimesh.util.concatenate([m.copy() for m in source]) if operation=='merge' else boolean(source,'difference' if operation=='remove_volume' else operation,cancelled)
        mesh.metadata={}
        return dict(mode='merge',items=[dict(row=records[0]['row'],meshes=[mesh],report=dict(changed=True,warnings=['Выбранные детали заменяются одной сеткой; инструменты вычитания удаляются из сцены.']))])
    for record in records:
        check_cancel(cancelled);source=record['mesh'];progress(record['name'])
        if operation in ('hollow','shell_core','round','round_offset','honeycomb','lattice','slice_lattice','tetra','tetra_slices','formfit'):
            meshes=implicit(source,operation,params,progress,cancelled)
            warnings=['Приближённая сетка по полю расстояний. Точность ограничена шагом расчёта; проверьте размеры и тонкие элементы.']
        elif operation=='cut':
            from repair_manual_geometry import edit_mesh
            meshes=[edit_mesh(source,'clip',dict(point=params['point'],normal=params['normal'],side=side,cap=True),cancelled=cancelled)[0] for side in ('positive','negative')]
        elif operation=='offset':
            mesh=prepared(source);offset=number(params,'offset',1.,minimum=-1e6)
            mesh.vertices=np.asarray(mesh.vertices)+mesh.vertex_normals*offset
            if not mesh.is_volume:raise ValueError('Смещение требует замкнутой согласованной сетки.')
            meshes=[mesh];warnings=['Смещение вершин по нормалям: возможны самопересечения и изменение толщины. Проверьте результат диагностикой.']
        elif operation=='extrude':
            distance=number(params,'distance',2.,minimum=-1e6)
            if not distance:raise ValueError('Выдвижение не должно быть нулевым.')
            prism,_=extruded_surface(source,selection.get(record['row']),abs(distance))
            if distance<0:
                prism,_=extruded_surface(source,selection.get(record['row']),distance)
            meshes=[boolean([source,prism],'union' if distance>0 else 'difference',cancelled)]
        elif operation=='surface_array':
            ids=selection.get(record['row']);distance=number(params,'distance',2.,minimum=-1e6)
            count=int(params.get('count',3))
            if not ids or not 1<=count<=100:raise ValueError('Выберите поверхность; число копий — от 1 до 100.')
            if count*len(ids)>MAX_FACES:raise ValueError('Массив превышает 2 млн треугольников. Уменьшите число копий.')
            patch=source.submesh([ids],append=True,repair=False);normal=np.sum(patch.face_normals*patch.area_faces[:,None],axis=0)
            if np.linalg.norm(normal)<1e-9:raise ValueError('Нормаль поверхности не определена.')
            meshes=[]
            for i in range(1,count+1):
                mesh=patch.copy();mesh.apply_translation(normal/np.linalg.norm(normal)*distance*i);meshes.append(mesh)
            mode='add';warnings=['Копии поверхности — открытые сетки, не замкнутые детали.']
        elif operation=='perforate':
            source=solid(source);radius=number(params,'radius',1.);pitch=number(params,'cell',5.)
            if pitch<=2*radius:raise ValueError('Шаг отверстий должен быть больше их диаметра.')
            axis=int(params.get('axis',2));others=[i for i in range(3) if i!=axis]
            positions=[np.arange(source.bounds[0,i]+pitch/2,source.bounds[1,i],pitch) for i in others]
            if len(positions[0])*len(positions[1])>400:raise ValueError('Слишком много отверстий: предел 400. Увеличьте шаг.')
            cutters=[]
            for a in positions[0]:
                for b in positions[1]:
                    check_cancel(cancelled)
                    center=source.bounds.mean(0);center[others]=[a,b]
                    cylinder=trimesh.creation.cylinder(radius,source.extents[axis]+radius*4,sections=32)
                    if axis!=2:cylinder.apply_transform(trimesh.geometry.align_vectors([0,0,1],np.eye(3)[axis]))
                    cylinder.apply_translation(center);cutters.append(cylinder)
            if not cutters:raise ValueError('При заданном шаге отверстия не попадают в деталь.')
            meshes=[boolean([source,trimesh.util.concatenate(cutters)],'difference',cancelled)]
        elif operation=='label':
            text=label_geometry(params,cancelled)
            if params.get('standalone'):meshes=[text];mode='add'
            else:meshes=[boolean([source,text],'difference' if params.get('engrave') else 'union',cancelled)]
        elif operation=='struts':
            start,end=np.asarray(params['start']),np.asarray(params['end']);direction=end-start
            if np.linalg.norm(direction)<1e-6:raise ValueError('Концы распорки должны различаться.')
            rod=trimesh.creation.cylinder(number(params,'radius',1.),np.linalg.norm(direction),sections=48)
            rod.apply_transform(trimesh.geometry.align_vectors([0,0,1],direction));rod.apply_translation((start+end)/2)
            meshes=[rod];mode='add'
        elif operation=='rapidfit':
            wall=number(params,'wall',2.);gap=number(params,'gap',.2,positive=False)
            bounds=source.bounds.copy();size=source.extents+2*(wall+gap);size[2]=source.extents[2]+wall
            center=bounds.mean(0);center[2]-=wall/2
            outer=trimesh.creation.box(size);outer.apply_translation(center)
            inner=trimesh.creation.box(source.extents+[2*gap,2*gap,2*wall]);center=bounds.mean(0);center[2]+=wall
            inner.apply_translation(center);meshes=[boolean([outer,inner],'difference',cancelled)];mode='add'
            warnings=['Прямоугольный открытый лоток по габаритам. Это собственный геометрический фиксатор, без расчёта прочности или эквивалентности RapidFit.']
        else:raise ValueError('Неизвестная операция: '+operation)
        for mesh in meshes:
            check_cancel(cancelled)
            if len(mesh.faces)>MAX_FACES:raise ValueError('Слишком большой результат: предел 2 млн треугольников.')
            # A modified mesh must never keep stale CAD/texture provenance.
            mesh.metadata={}
        if sum(len(item['meshes']) for item in items)+len(meshes)>200:
            raise ValueError('За один раз создаётся не более 200 частей. Уменьшите число выбранных деталей или копий.')
        if sum(len(mesh.faces) for item in items for mesh in item['meshes'])+sum(len(mesh.faces) for mesh in meshes)>MAX_FACES:
            raise ValueError('Суммарный результат превышает 2 млн треугольников.')
        items.append(dict(row=record['row'],meshes=meshes,report=dict(changed=True,before=dict(faces=len(source.faces)),after=dict(faces=sum(len(m.faces) for m in meshes)),warnings=warnings)))
    return dict(mode=mode,items=items)
