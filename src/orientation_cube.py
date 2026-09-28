"""A translucent clickable view cube beside the viewport's coordinate triad."""
import numpy as np
from PySide6.QtCore import Qt, QPointF, QEvent, QTimer
from PySide6.QtGui import QPainter, QPolygonF, QColor, QPen, QRegion, QPainterPath, QPainterPathStroker, QPixmap
from PySide6.QtWidgets import QWidget


class OrientationCube(QWidget):
    def __init__(self, plotter):
        super().__init__(plotter)
        self.plotter = plotter
        # VTK owns a native child window on Windows. A non-native Qt overlay
        # can paint correctly in grab() yet be hidden behind that window.
        self.setAttribute(Qt.WA_NativeWindow)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setFixedSize(138, 138)
        self.setMouseTracking(True)
        self.setToolTip('Грань куба — вид по нормали. Двойной щелчок — изометрия.')
        self.faces = []
        self.hover = None
        self._frame_key = None
        self._pixmap = None
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(16)
        self._frame_timer.timeout.connect(self._refresh_frame)
        plotter.installEventFilter(self)
        self.observer = plotter.camera.AddObserver('ModifiedEvent', self.camera_changed)
        self.reposition()
        self._refresh_frame()
        self.show()

    def reposition(self):
        self.move(5, self.parentWidget().height() - self.height() - 5)
        self.raise_()

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Show):
            self.reposition()
            self._frame_timer.start(0)
        return False

    def camera_changed(self, *_):
        actor = self.plotter.actors.get('plat_base')
        if actor:
            opacity = .12 if self.plotter.camera.position[2] < 0 else 1.
            if actor.GetProperty().GetOpacity() != opacity:
                actor.GetProperty().SetOpacity(opacity)
        if self.isVisible() and not self._frame_timer.isActive(): self._frame_timer.start(16)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._pixmap is not None: painter.drawPixmap(0, 0, self._pixmap)

    def _refresh_frame(self):
        self._frame_timer.stop()
        matrix = self.plotter.camera.GetViewTransformMatrix()
        rotation = np.array([[matrix.GetElement(i, j) for j in range(3)] for i in range(3)])
        key = (tuple(rotation.ravel()), self.hover, self.plotter.background_color.int_rgb, self.devicePixelRatioF())
        if key == self._frame_key: return
        self._frame_key = key
        ratio = self.devicePixelRatioF()
        self._pixmap = QPixmap(round(self.width()*ratio), round(self.height()*ratio))
        self._pixmap.setDevicePixelRatio(ratio)
        painter = QPainter(self._pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(*self.plotter.background_color.int_rgb))
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
        mask = QRegion()
        for _, polygon, axis, sign in sorted(faces, key=lambda x: x[0]):
            self.faces.append((polygon, axis, sign))
            mask = mask.united(QRegion(polygon.toPolygon()))
            painter.setPen(QPen(QColor('#77818a'), .8))
            shade = (QColor(80, 100, 115, 18), QColor(80, 100, 115, 28), QColor(80, 100, 115, 10))[axis]
            painter.setBrush(QColor(30, 130, 192, 65) if self.hover == (axis, sign) else shade)
            painter.drawPolygon(polygon)
            painter.setPen(QColor('#485864'))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            center = polygon.boundingRect().center()
            painter.drawText(center + QPointF(-8, 4), 'XYZ'[axis] + ('+' if sign > 0 else '−'))
        # Rear edges remain visible through the pale faces, as in a wire cube.
        painter.setPen(QPen(QColor('#a4abb1'), .6, Qt.DotLine))
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for a in (-1, 1):
                for b in (-1, 1):
                    start = np.zeros(3); start[others] = (a, b); start[axis] = -1
                    end = start.copy(); end[axis] = 1
                    painter.drawLine(project(start), project(end))
        for axis, color in enumerate(('#de4d47', '#56a847', '#449be2')):
            start = project(np.zeros(3)) + QPointF(0, 23)
            end = project(np.eye(3)[axis] * 1.65) + QPointF(0, 23)
            painter.setPen(QPen(QColor(color), 1.8))
            painter.drawLine(start, end)
            painter.drawText(end + QPointF(2, 0), 'XYZ'[axis])
            path = QPainterPath(start); path.lineTo(end)
            stroker = QPainterPathStroker(); stroker.setWidth(5)
            mask = mask.united(QRegion(stroker.createStroke(path).toFillPolygon().toPolygon()))
            mask = mask.united(QRegion(int(end.x()), int(end.y()) - 14, 18, 19))
        # Clip the native window to the cube and triad: no rectangular panel
        # covers the model or intercepts clicks outside the orientation control.
        painter.end()
        if mask != self.mask(): self.setMask(mask)
        self.update()

    def face_at(self, point):
        return next(((axis, sign) for polygon, axis, sign in reversed(self.faces) if polygon.containsPoint(point, Qt.OddEvenFill)), None)

    def mouseMoveEvent(self, event):
        hover = self.face_at(event.position())
        if hover == self.hover: return
        self.hover = hover
        self.setCursor(Qt.PointingHandCursor if self.hover else Qt.ArrowCursor)
        self._refresh_frame()

    def leaveEvent(self, event):
        if self.hover is None: return
        self.hover = None
        self._refresh_frame()

    def orient(self, axis=None, sign=1):
        camera = self.plotter.camera
        focus = np.asarray(camera.focal_point)
        distance = max(np.linalg.norm(np.asarray(camera.position) - focus), 1.)
        direction = np.array([1., 1., 1.]) / np.sqrt(3) if axis is None else np.eye(3)[axis] * sign
        camera.position = focus + direction * distance
        camera.up = (0, 1, 0) if axis == 2 else (0, 0, 1)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        self._refresh_frame()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            face = self.face_at(event.position())
            if face: self.orient(*face)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.orient()

    def dispose(self):
        self._frame_timer.stop()
        self.plotter.camera.RemoveObserver(self.observer)
