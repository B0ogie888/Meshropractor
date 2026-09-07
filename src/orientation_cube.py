"""A translucent clickable view cube beside the viewport's coordinate triad."""
import numpy as np
from PySide6.QtCore import Qt, QPointF, QEvent
from PySide6.QtGui import QPainter, QPolygonF, QColor, QPen
from PySide6.QtWidgets import QWidget


class OrientationCube(QWidget):
    def __init__(self, plotter):
        super().__init__(plotter)
        self.plotter = plotter
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setFixedSize(138, 138)
        self.setMouseTracking(True)
        self.setToolTip('Грань куба — вид по нормали. Двойной щелчок — изометрия.')
        self.faces = []
        self.hover = None
        plotter.installEventFilter(self)
        self.observer = plotter.camera.AddObserver('ModifiedEvent', self.camera_changed)
        self.reposition()
        self.show()

    def reposition(self):
        self.move(5, self.parentWidget().height() - self.height() - 5)
        self.raise_()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize: self.reposition()
        return False

    def camera_changed(self, *_):
        actor = self.plotter.actors.get('plat_base')
        if actor:
            actor.GetProperty().SetOpacity(.12 if self.plotter.camera.position[2] < 0 else 1.)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        matrix = self.plotter.camera.GetViewTransformMatrix()
        rotation = np.array([[matrix.GetElement(i, j) for j in range(3)] for i in range(3)])
        def project(point):
            p = rotation @ point
            return QPointF(69 + p[0] * 31, 65 - p[1] * 31)
        faces = []
        for axis in range(3):
            for sign in (-1, 1):
                normal = np.eye(3)[axis] * sign
                if (rotation @ normal)[2] <= 1e-8: continue
                u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
                polygon = QPolygonF([project(normal + a * u + b * v) for a, b in ((-1,-1), (1,-1), (1,1), (-1,1))])
                faces.append(((rotation @ normal)[2], polygon, axis, sign))
        self.faces = []
        for _, polygon, axis, sign in sorted(faces, key=lambda x: x[0]):
            self.faces.append((polygon, axis, sign))
            painter.setPen(QPen(QColor('#172b3b'), 2.0))
            shade = (QColor(51, 75, 94, 225), QColor(67, 92, 111, 225), QColor(83, 108, 126, 225))[axis]
            painter.setBrush(QColor(30, 130, 192, 245) if self.hover == (axis, sign) else shade)
            painter.drawPolygon(polygon)
            painter.setPen(QColor('#ffffff'))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            center = polygon.boundingRect().center()
            painter.drawText(center + QPointF(-8, 4), 'XYZ'[axis] + ('+' if sign > 0 else '−'))
        for axis, color in enumerate(('#de4d47', '#56a847', '#449be2')):
            start = project(np.zeros(3)) + QPointF(0, 23)
            end = project(np.eye(3)[axis] * 1.65) + QPointF(0, 23)
            painter.setPen(QPen(QColor(color), 1.8))
            painter.drawLine(start, end)
            painter.drawText(end + QPointF(2, 0), 'XYZ'[axis])

    def face_at(self, point):
        return next(((axis, sign) for polygon, axis, sign in reversed(self.faces) if polygon.containsPoint(point, Qt.OddEvenFill)), None)

    def mouseMoveEvent(self, event):
        self.hover = self.face_at(event.position())
        self.setCursor(Qt.PointingHandCursor if self.hover else Qt.ArrowCursor)
        self.update()

    def leaveEvent(self, event):
        self.hover = None
        self.update()

    def orient(self, axis=None, sign=1):
        camera = self.plotter.camera
        focus = np.asarray(camera.focal_point)
        distance = max(np.linalg.norm(np.asarray(camera.position) - focus), 1.)
        direction = np.array([1., 1., 1.]) / np.sqrt(3) if axis is None else np.eye(3)[axis] * sign
        camera.position = focus + direction * distance
        camera.up = (0, 1, 0) if axis == 2 else (0, 0, 1)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            face = self.face_at(event.position())
            if face: self.orient(*face)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.orient()

    def dispose(self):
        self.plotter.camera.RemoveObserver(self.observer)
