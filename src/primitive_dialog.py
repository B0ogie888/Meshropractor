"""Engineering create-part dialog with linked dimensions and a dimensioned sketch."""
import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QSignalBlocker
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor, QPalette
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QWidget,QLabel,
    QComboBox,QDoubleSpinBox,QSpinBox,QDialogButtonBox,QGroupBox,QPushButton)
from app_branding import app_icon
from primitive_geometry import KINDS,FIELDS,build_primitive,tessellation


class PrimitiveSketch(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent); self.mesh = None; self.kind = ''; self.values = {}
        self.setMinimumSize(240,250)
        self.setToolTip('Схема параметров. Все размеры задаются в миллиметрах.')

    def set_parameters(self,kind,values):
        self.kind,self.values = kind,values
        try: self.mesh = build_primitive(kind,values,[0,0,0],dict(mode='segments',segments=64 if kind=='Тор' else 32))
        except ValueError: self.mesh = None
        self.update()

    def paintEvent(self,event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        owner = self
        while owner is not None and not hasattr(owner,'engineering_theme'): owner = owner.parentWidget()
        dark = owner.engineering_theme.mode=='dark' if owner is not None else self.palette().color(QPalette.Window).lightness()<100
        ink = QColor('#e6ede7' if dark else '#252b2b')
        painter.setBrush(Qt.NoBrush)
        if self.mesh is None:
            painter.setPen(ink); painter.drawText(self.rect(),Qt.AlignCenter,'Проверьте размеры фигуры'); return
        vertices = self.mesh.vertices
        profile = None
        if self.kind in (KINDS[8],KINDS[9]):
            x,y = self.values['x']/2,self.values['y']/2
            h = self.values['h']; size = self.values.get('fillet',self.values.get('chamfer'))
            if self.kind==KINDS[8]:
                corners = ((x-size,y-size,0),(-x+size,y-size,90),
                           (-x+size,-y+size,180),(x-size,-y+size,270))
                profile = np.vstack([np.c_[cx+size*np.cos(angles),cy+size*np.sin(angles)]
                    for cx,cy,start in corners
                    for angles in [np.deg2rad(np.linspace(start,start+90,17))]])
            else:
                profile = np.array([(x,y-size),(x-size,y),(-x+size,y),(-x,y-size),
                                    (-x,-y+size),(-x+size,-y),(x-size,-y),(x,-y+size)])
            # Simplified parameter sketches show the rounded/bevelled vertical
            # corners without the surface triangulation or extra inset borders.
            vertices = np.vstack([np.c_[profile,np.full(len(profile),z)] for z in (-h/2,h/2)])
        # An orthographic isometric view: X runs to the right, Y to the left,
        # and the nearest vertical edge stays between the two visible sides.
        projection = np.array([[1,-1,0],[-1/math.sqrt(3),-1/math.sqrt(3),-2/math.sqrt(3)]])/math.sqrt(2)
        view_direction = np.array([-1,-1,1])
        if self.kind=='Тор':
            tilt = math.radians(-45)
            rotation = np.array([[math.cos(tilt),-math.sin(tilt)],[math.sin(tilt),math.cos(tilt)]])
            projection = rotation@np.array([[1,0,0],[0,.5,-math.sqrt(3)/2]])
            view_direction = np.array([0,math.sqrt(3)/2,.5])
        flat = vertices@projection.T
        low,high = flat.min(0),flat.max(0)
        factor = min((self.width()-82)/max(high[0]-low[0],1e-8),(self.height()-84)/max(high[1]-low[1],1e-8))
        midpoint = (low+high)/2
        def screen(point,offset=(0,0)):
            p = (np.asarray(point)@projection.T-midpoint)*factor
            return QPointF(p[0]+self.width()/2+offset[0],p[1]+self.height()/2+offset[1])
        points = [screen(p) for p in vertices]
        painter.setPen(QPen(ink,1.3))
        p = self.values; h = p.get('h',0); r = p.get('r',0)
        circular = self.kind in ('Цилиндр','Труба','Конус')
        top = screen([0,0,h/2]); bottom = screen([0,0,-h/2])
        rx,ry = r*factor,r*factor/math.sqrt(3)
        top_radius = p['top'] if self.kind=='Конус' else r
        tx,ty = top_radius*factor,top_radius*factor/math.sqrt(3)

        def ellipse(center,x,y,front_only=False):
            rect = QRectF(center.x()-x,center.y()-y,2*x,2*y)
            if front_only: painter.drawArc(rect,180*16,180*16)
            elif x>0: painter.drawEllipse(rect)

        if circular:
            # Analytic ellipses avoid exposing tessellation in a parameter diagram.
            ellipse(top,tx,ty); ellipse(bottom,rx,ry,True)
            for sign in (-1,1):
                painter.drawLine(top+QPointF(sign*tx,0),bottom+QPointF(sign*rx,0))
            if self.kind=='Труба':
                ellipse(top,p['inner']*factor,p['inner']*factor/math.sqrt(3))
        elif self.kind=='Сфера':
            center = screen([0,0,0]); sphere_radius = r*factor
            ellipse(center,sphere_radius,sphere_radius)
            ellipse(center,sphere_radius,sphere_radius*.3,True)
        elif profile is not None:
            upper = [screen([px,py,h/2]) for px,py in profile]
            lower = [screen([px,py,-h/2]) for px,py in profile]
            path = QPainterPath(); path.moveTo(upper[0])
            for point in upper[1:]: path.lineTo(point)
            path.closeSubpath(); painter.drawPath(path)
            delta = np.roll(profile,-1,axis=0)-profile
            visible = delta[:,0]-delta[:,1]>1e-9
            for index,shown in enumerate(visible):
                if shown: painter.drawLine(lower[index],lower[(index+1)%len(lower)])
            if self.kind==KINDS[8]:
                edges = (min(range(len(upper)),key=lambda i:upper[i].x()),
                         max(range(len(upper)),key=lambda i:upper[i].x()))
            else: edges = [i for i in range(len(upper)) if visible[i] or visible[i-1]]
            for index in edges: painter.drawLine(upper[index],lower[index])
        else:
            normals = self.mesh.face_normals; front = normals@view_direction>1e-9
            for faces,edge in zip(self.mesh.face_adjacency,self.mesh.face_adjacency_edges):
                a,b = faces
                flat_a = np.max(np.abs(normals[a]))>1-1e-8
                flat_b = np.max(np.abs(normals[b]))>1-1e-8
                fillet_boundary = self.kind==KINDS[8] and flat_a!=flat_b and (front[a] if flat_a else front[b])
                if (front[a] or front[b]) and (front[a]!=front[b] or normals[a]@normals[b]<.75 or fillet_boundary):
                    painter.drawLine(points[edge[0]],points[edge[1]])

        if self.kind=='Тор':
            # The dashed circles identify the ring centreline and the circular
            # tube section, so the two radii have unambiguous reference points.
            angles = np.linspace(0,2*math.pi,129)
            tube = p['tube']
            pen = QPen(ink,1); pen.setDashPattern([5,4]); painter.setPen(pen)
            for curve in (np.c_[r*np.cos(angles),r*np.sin(angles),np.zeros(len(angles))],
                          np.c_[r+tube*np.cos(angles),np.zeros(len(angles)),tube*np.sin(angles)]):
                path = QPainterPath(); path.moveTo(screen(curve[0]))
                for point in curve[1:]: path.lineTo(screen(point))
                path.closeSubpath(); painter.drawPath(path)

        font = painter.font(); font.setBold(False); painter.setFont(font)

        def label_at(point,label):
            metrics = painter.fontMetrics()
            width,height = metrics.horizontalAdvance(label)+2,metrics.height()
            painter.drawText(QRectF(point.x()-width/2,point.y()-height/2,width,height),Qt.AlignCenter,label)

        def arrow(point,u,sign):
            normal = np.array([-u[1],u[0]])
            for side in (-1,1):
                tip = sign*5*u+side*2.5*normal
                painter.drawLine(point,point+QPointF(*tip))

        def dimension_screen(source_a,source_b,label,offset=(0,0)):
            a,b = source_a+QPointF(*offset),source_b+QPointF(*offset)
            painter.setPen(QPen(ink,1))
            if offset!=(0,0):
                painter.drawLine(source_a,a); painter.drawLine(source_b,b)
            painter.drawLine(a,b)
            dx,dy = b.x()-a.x(),b.y()-a.y(); length = math.hypot(dx,dy)
            if length>2:
                u = np.array([dx,dy])/length
                normal = np.array([-u[1],u[0]])
                if np.dot(normal,offset)<0: normal *= -1
                for point,sign in ((a,1),(b,-1)): arrow(point,u,sign)
                distance = painter.fontMetrics().height()/2+5
                label_at((a+b)/2+QPointF(*(normal*distance)),label)

        def dimension(a,b,label,offset=(0,0)):
            dimension_screen(screen(a),screen(b),label,offset)

        def radius(center,edge,label,above=True,label_position=None):
            painter.setPen(QPen(ink,1)); painter.drawLine(center,edge)
            delta = np.array([edge.x()-center.x(),edge.y()-center.y()])
            length = np.linalg.norm(delta)
            if length>2:
                u = delta/length; normal = np.array([-u[1],u[0]])
                if (normal[1]>0)==above: normal *= -1
                arrow(edge,u,-1)
                label_at(label_position if label_position is not None else (center+edge)/2+QPointF(*(normal*(painter.fontMetrics().height()/2+3))),label)
            else: label_at(label_position if label_position is not None else center+QPointF(0,-painter.fontMetrics().height()),label+' = 0' if length<1e-8 else label)

        if self.kind in (KINDS[0],KINDS[4],KINDS[8],KINDS[9]):
            x,y = p['x']/2,p['y']/2
            dimension([-x,-y,-h/2],[x,-y,-h/2],'X',(16,10))
            dimension([-x,y,-h/2],[-x,-y,-h/2],'Y',(-16,10))
            if self.kind==KINDS[4]:
                apex = screen([0,0,h/2]); corner = screen([x,-y,-h/2])
                height_x = corner.x()+17
                height_top = QPointF(height_x,apex.y())
                height_bottom = QPointF(height_x,screen([0,0,-h/2]).y())
                painter.setPen(QPen(ink,1))
                painter.drawLine(apex,height_top); painter.drawLine(corner,height_bottom)
                dimension_screen(height_bottom,height_top,'H')
            else: dimension([x,-y,-h/2],[x,-y,h/2],'Z',(17,0))
            if self.kind in (KINDS[8],KINDS[9]):
                size = p.get('fillet',p.get('chamfer'))
                if self.kind==KINDS[8]:
                    corner = np.array([-x+size,-y+size,h/2])
                    radius(screen(corner),screen(corner-[size/math.sqrt(2),size/math.sqrt(2),0]),'R',
                           label_position=screen(corner)+QPointF(0,-painter.fontMetrics().height()))
                else:
                    leader=screen([-x+1.6*size,-y+1.6*size,h/2])
                    radius(leader,screen([-x+size/2,-y+size/2,h/2]),'C',
                           label_position=leader+QPointF(0,-painter.fontMetrics().height()))
        elif circular:
            direction = QPointF(.8, .6/math.sqrt(3))
            if self.kind=='Цилиндр': radius(top,top+direction*(r*factor),'R')
            else:
                radius(bottom,bottom+direction*(r*factor),'R1')
                inner = p['inner'] if self.kind=='Труба' else top_radius
                radius(top,top+direction*(inner*factor),'R2')
            height_x = max(rx,tx)
            painter.drawLine(top+QPointF(tx,0),top+QPointF(height_x,0))
            painter.drawLine(bottom+QPointF(rx,0),bottom+QPointF(height_x,0))
            dimension_screen(top+QPointF(height_x,0),bottom+QPointF(height_x,0),'H',(17,0))
        elif self.kind=='Сфера':
            radius(center,center+QPointF(sphere_radius*.8,-sphere_radius*.6),'R')
        elif self.kind=='Тор':
            radius(screen([0,0,0]),screen([-r,0,0]),'R1',above=False)
            radius(screen([r,0,0]),screen([r,0,-p['tube']]),'R2',above=False)
        else:
            radius(screen([0,0,h/2]),screen([r,0,h/2]),'R')
            # Use the right silhouette edge, which is visible for any N. Drawing
            # the height there leaves the side faces free of dimension lines.
            angles = np.arange(int(p['sides']))*2*math.pi/int(p['sides'])
            ring = np.c_[r*np.cos(angles),r*np.sin(angles),np.zeros(len(angles))]
            edge = max(ring,key=lambda point:screen(point).x())
            dimension(edge+[0,0,-h/2],edge+[0,0,h/2],'H')
        painter.end()


class PrimitiveDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle('Создать деталь'); self.setWindowIcon(app_icon()); self.setMinimumWidth(660)
        self.setStyleSheet('''QDialog {background:#fafbf8; color:#252b2b;}
            QLabel#PrimitiveTitle {font-size:18px; font-weight:600;}
            QLabel#PrimitiveHint {color:#5b6463;}
            QGroupBox {border:1px solid #cdd3cf; margin-top:8px; padding-top:12px;}
            QGroupBox::title {subcontrol-origin:margin; left:10px; padding:0 4px;}
        ''')
        self.inputs = {}; self.diameters = {}; self.saved = {}; self.previous_kind = None
        outer = QVBoxLayout(self); outer.setContentsMargins(18,18,18,16); outer.setSpacing(12)
        title = QLabel('СОЗДАТЬ ДЕТАЛЬ'); title.setObjectName('PrimitiveTitle'); outer.addWidget(title)
        self.kind = QComboBox(); self.kind.addItems(KINDS); outer.addWidget(self.kind)
        body = QHBoxLayout(); self.form_widget = QWidget(); self.form = QFormLayout(self.form_widget)
        self.form.setContentsMargins(0,0,8,0); body.addWidget(self.form_widget,1)
        self.sketch = PrimitiveSketch(); body.addWidget(self.sketch,1); outer.addLayout(body)
        mesh = QGroupBox('Построение сетки'); mesh_form = QFormLayout(mesh)
        self.mode = QComboBox(); self.mode.addItems(['По допуску','По числу сегментов'])
        mesh_form.addRow('Детализация',self.mode)
        self.tolerance = self.spin(.01,.00001,1000); self.tolerance.setDecimals(5); self.tolerance.setSuffix(' мм')
        mesh_form.addRow('Допуск поверхности',self.tolerance)
        self.segments = QSpinBox(); self.segments.setRange(12,2048); self.segments.setValue(96); self.segments.setSingleStep(4)
        self.segments.setKeyboardTracking(False); mesh_form.addRow('Сегменты окружности',self.segments)
        hint = QLabel('Допуск — максимальное отклонение сетки от криволинейной поверхности. Меньше допуск — больше треугольников.')
        hint.setObjectName('PrimitiveHint'); hint.setWordWrap(True); mesh_form.addRow(hint)
        self.mesh_form = mesh_form; outer.addWidget(mesh)
        center = QGroupBox('Центр в координатах сцены'); center_layout = QHBoxLayout(center); self.origin = []
        for axis,value in zip('XYZ',(0,0,5)):
            box = self.spin(value,-1e6,1e6); box.setPrefix(axis+': '); center_layout.addWidget(box); self.origin.append(box)
        outer.addWidget(center)
        plate = QPushButton('Поставить низ на Z = 0'); plate.clicked.connect(self.on_plate); outer.addWidget(plate)
        self.status = QLabel(); self.status.setWordWrap(True); outer.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText('Создать'); self.buttons.button(QDialogButtonBox.Cancel).setText('Закрыть')
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject); outer.addWidget(self.buttons)
        self.kind.currentIndexChanged.connect(self.shape_changed)
        for widget in (self.mode,self.tolerance,self.segments):
            signal = widget.currentIndexChanged if widget is self.mode else widget.valueChanged
            signal.connect(self.update_quality)
        self.shape_changed(); self.resize(710,self.sizeHint().height())

    @staticmethod
    def spin(value,low=.001,high=1e6):
        box = QDoubleSpinBox(); box.setDecimals(4); box.setRange(low,high); box.setValue(value)
        box.setSuffix(' мм'); box.setKeyboardTracking(False); return box

    def values(self): return {key:widget.value() for key,widget in self.inputs.items()}

    def shape_changed(self,*_):
        if self.previous_kind is not None: self.saved[self.previous_kind] = self.values()
        while self.form.rowCount(): self.form.removeRow(0)
        self.inputs.clear(); self.diameters.clear(); kind = self.kind.currentText(); self.previous_kind = kind
        for key,label,default in FIELDS[kind]:
            if key=='sides':
                box = QSpinBox(); box.setRange(3,128); box.setValue(6); box.setKeyboardTracking(False)
            else: box = self.spin(default,0 if key=='top' else .001)
            box.setValue(self.saved.get(kind,{}).get(key,default)); self.inputs[key] = box
            self.form.addRow(label,box); box.valueChanged.connect(self.update_geometry)
        for key,label in [('r','D1 — наружный диаметр' if kind=='Труба' else 'D1 — нижний диаметр' if kind=='Конус' else 'D — диаметр'),
                          ('inner','D2 — внутренний диаметр'),('top','D2 — верхний диаметр')]:
            if key not in self.inputs or kind=='Тор': continue
            diameter = self.spin(self.inputs[key].value()*2,0 if key=='top' else .002,2e6)
            self.diameters[key] = diameter; self.form.addRow(label,diameter)
            diameter.valueChanged.connect(lambda value,k=key:self.radius_changed(k,value/2))
        self.update_geometry()

    def radius_changed(self,key,value):
        self.inputs[key].setValue(value)

    def update_geometry(self,*_):
        for key,diameter in self.diameters.items():
            with QSignalBlocker(diameter): diameter.setValue(2*self.inputs[key].value())
        self.sketch.set_parameters(self.kind.currentText(),self.values()); self.update_quality()

    def settings(self):
        return dict(mode='tolerance' if self.mode.currentIndex()==0 else 'segments',
                    tolerance=self.tolerance.value(),segments=self.segments.value())

    def update_quality(self,*_):
        curved = self.kind.currentText() not in (KINDS[0],KINDS[4],KINDS[5],KINDS[9])
        self.mode.setEnabled(curved)
        self.mesh_form.setRowVisible(self.tolerance,curved and self.mode.currentIndex()==0)
        self.mesh_form.setRowVisible(self.segments,curved and self.mode.currentIndex()==1)
        try:
            quality = tessellation(self.kind.currentText(),self.values(),self.settings())
            segments = f" / {quality['n']} сегментов" if quality['n'] else ''
            self.status.setText(f"Сетка: ≈ {quality['faces']:,} треугольников{segments}. Единицы — мм.")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except ValueError as exc:
            self.status.setText(str(exc)); self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def on_plate(self):
        values = self.values(); kind = self.kind.currentText()
        half = values['r'] if kind=='Сфера' else values['tube'] if kind=='Тор' else values['h']/2
        self.origin[2].setValue(half)

    def parameters(self):
        return dict(kind=self.kind.currentText(),primitive=self.values(),mesh_settings=self.settings(),
                    center=[box.value() for box in self.origin])
