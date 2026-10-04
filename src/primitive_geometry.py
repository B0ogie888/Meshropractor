"""Parametric solids with explicit chordal tolerance and bounded mesh complexity."""
from itertools import product
import math
import numpy as np
import trimesh

KINDS = ('Параллелепипед','Цилиндр','Труба','Конус','Пирамида','Призма','Сфера','Тор',
         'Скруглённый параллелепипед','Параллелепипед с фасками')
MAX_FACES = 250_000
FIELDS = {
    KINDS[0]: [('x','X — ширина',10),('y','Y — глубина',10),('h','Z — высота',10)],
    KINDS[1]: [('r','R — радиус',5),('h','H — высота',10)],
    KINDS[2]: [('r','R1 — наружный радиус',5),('inner','R2 — внутренний радиус',2.5),('h','H — высота',10)],
    KINDS[3]: [('r','R1 — нижний радиус',5),('top','R2 — верхний радиус',2.5),('h','H — высота',10)],
    KINDS[4]: [('x','X — ширина основания',10),('y','Y — глубина основания',10),('h','H — высота',10)],
    KINDS[5]: [('r','R — радиус основания',5),('h','H — высота',10),('sides','N — стороны основания',6)],
    KINDS[6]: [('r','R — радиус',5)],
    KINDS[7]: [('r','R1 — радиус кольца',10),('tube','R2 — радиус сечения',2.5)],
    KINDS[8]: [('x','X — ширина',10),('y','Y — глубина',10),('h','Z — высота',10),('fillet','R — скругление',1)],
    KINDS[9]: [('x','X — ширина',10),('y','Y — глубина',10),('h','Z — высота',10),('chamfer','C — фаска',1)],
}


def validate(kind, values):
    if kind not in FIELDS: raise ValueError('Неизвестная форма.')
    for key,_,_ in FIELDS[kind]:
        value = float(values[key])
        if not math.isfinite(value) or value<0 or (value==0 and key!='top'):
            raise ValueError('Размеры должны быть положительными конечными числами.')
    if kind=='Труба' and values['inner']>=values['r']: raise ValueError('Внутренний радиус должен быть меньше наружного.')
    if kind=='Тор' and values['tube']>=values['r']: raise ValueError('Радиус сечения должен быть меньше радиуса кольца.')
    if kind==KINDS[8] and 2*values['fillet']>min(values['x'],values['y'],values['h']):
        raise ValueError('Радиус скругления не должен превышать половину меньшего размера.')
    if kind==KINDS[9] and 2*values['chamfer']>=min(values['x'],values['y'],values['h']):
        raise ValueError('Фаска должна быть меньше половины меньшего размера.')
    if kind=='Призма' and (int(values['sides'])!=values['sides'] or not 3<=values['sides']<=128):
        raise ValueError('Число сторон должно быть от 3 до 128.')


def circle_segments(radius, tolerance):
    if not math.isfinite(tolerance) or tolerance<=0: raise ValueError('Допуск должен быть больше нуля.')
    # Stable for a very small tolerance/radius, without cancellation in acos.
    angle = 2*math.asin(math.sqrt(min(tolerance/radius,2.)/2))
    if angle==0: raise ValueError('Слишком малая величина допуска. Увеличьте допуск.')
    number = max(12,math.ceil(math.pi/angle))
    if number>2048: raise ValueError('Слишком малая величина допуска. Увеличьте допуск (максимум 2048 сегментов).')
    return ((number+3)//4)*4


def tessellation(kind, values, settings):
    validate(kind,values)
    mode = settings.get('mode','tolerance'); tolerance = float(settings.get('tolerance',.01))
    if mode not in ('tolerance','segments'): raise ValueError('Неизвестный способ построения сетки.')
    manual = int(settings.get('segments',96))
    if not 12<=manual<=2048: raise ValueError('Число сегментов должно быть от 12 до 2048.')
    radius = max(values.get('r',0),values.get('top',0),values.get('fillet',0))
    n = m = 0
    if kind not in (KINDS[0],KINDS[4],KINDS[5],KINDS[9]):
        if kind=='Тор': radius += values['tube']
        curved_twice = kind in ('Тор','Сфера',KINDS[8])
        n = circle_segments(radius,tolerance/(2 if curved_twice else 1)) if mode=='tolerance' else manual
        m = circle_segments(values['tube'],tolerance/2) if kind=='Тор' and mode=='tolerance' else n
    if kind in (KINDS[0],KINDS[4],KINDS[9]): faces = {KINDS[0]:12,KINDS[4]:6,KINDS[9]:44}[kind]
    elif kind=='Призма': faces = 4*int(values['sides'])
    elif kind=='Труба': faces = 8*n
    elif kind=='Конус' and values['top']==0: faces = 2*n
    elif kind=='Тор': faces = 2*n*m
    elif kind in ('Сфера',KINDS[8]): faces = 2*n*(max(2,n//2)-1)+(8*n if kind==KINDS[8] else 48)
    else: faces = 4*n
    if faces>MAX_FACES: raise ValueError('Сетка превышает 250 000 треугольников. Увеличьте допуск или уменьшите число сегментов.')
    return dict(n=n,m=m,faces=faces)


def sphere(radius, n):
    latitude = max(4,((n//2+1)//2)*2)
    angle = np.linspace(0,np.pi,latitude+1)
    return trimesh.creation.revolve(np.c_[radius*np.sin(angle),-radius*np.cos(angle)],sections=n)


def build_primitive(kind, values, center, settings=None):
    values = {key:float(value) for key,value in values.items()}
    settings = dict(settings or {}); quality = tessellation(kind,values,settings)
    center = np.asarray(center,float)
    if center.shape!=(3,) or not np.isfinite(center).all(): raise ValueError('Центр должен содержать три конечные координаты.')
    n,m = quality['n'],quality['m']; h = values.get('h',0)
    if kind==KINDS[0]: mesh = trimesh.creation.box([values['x'],values['y'],h])
    elif kind=='Цилиндр': mesh = trimesh.creation.cylinder(values['r'],h,sections=n)
    elif kind=='Труба':
        r,inner = values['r'],values['inner']
        mesh = trimesh.creation.revolve([[inner,-h/2],[r,-h/2],[r,h/2],[inner,h/2],[inner,-h/2]],sections=n)
    elif kind=='Конус':
        mesh = trimesh.creation.revolve([[0,-h/2],[values['r'],-h/2],[values['top'],h/2],[0,h/2]],sections=n)
    elif kind=='Пирамида':
        x,y = values['x']/2,values['y']/2
        mesh = trimesh.Trimesh([[-x,-y,-h/2],[x,-y,-h/2],[x,y,-h/2],[-x,y,-h/2],[0,0,h/2]],
            [[0,2,1],[0,3,2],[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    elif kind=='Призма': mesh = trimesh.creation.cylinder(values['r'],h,sections=int(values['sides']))
    elif kind=='Сфера': mesh = sphere(values['r'],n)
    elif kind=='Тор': mesh = trimesh.creation.torus(values['r'],values['tube'],major_sections=n,minor_sections=m)
    elif kind==KINDS[8]:
        radius = values['fillet']; half = np.array([values['x'],values['y'],h])/2-radius
        directions = sphere(1.,n).vertices.copy(); directions[np.abs(directions)<1e-12]=0
        points = []
        for direction in directions:
            signs = [(sign,) if sign else (-1,1) for sign in np.sign(direction)]
            for corner in product(*signs): points.append(half*np.asarray(corner)+radius*direction)
        mesh = trimesh.convex.convex_hull(np.asarray(points))
    else:
        half = np.array([values['x'],values['y'],h])/2; c = values['chamfer']; points = []
        for signs in product((-1,1),repeat=3):
            for axis in range(3):
                p = half-c; p[axis]+=c; points.append(p*np.asarray(signs))
        mesh = trimesh.convex.convex_hull(np.asarray(points))
    if len(mesh.faces)>MAX_FACES or not mesh.is_volume: raise ValueError('Не удалось построить корректное замкнутое тело.')
    mesh.apply_translation(center)
    mesh.metadata['primitive'] = dict(kind=kind,parameters=values,center=center.tolist(),mesh_settings=settings)
    return mesh
