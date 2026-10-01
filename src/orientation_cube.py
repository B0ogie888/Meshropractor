"""Clickable orientation cube composed into the scene's existing VTK frame."""
import numpy as np

from PySide6.QtCore import QObject, Qt, QPointF, QRect, QEvent
from PySide6.QtGui import QPolygonF
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkTextActor


SIZE = 138
MARGIN = 5
SCALE = 28.
ORIGIN = np.full(3, -1.)
AXIS_COLORS = ((.87, .30, .28), (.34, .66, .28), (.14, .47, .93))


def _polydata(vertices, cells, *, lines=False):
    points = vtkPoints()
    for point in vertices:
        points.InsertNextPoint(*point)
    indices = vtkCellArray()
    for cell in cells:
        indices.InsertNextCell(len(cell))
        for index in cell:
            indices.InsertCellPoint(index)
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetLines(indices) if lines else data.SetPolys(indices)
    return data


def _actor(data, color, *, opacity=1., width=1.):
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(data)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetPickable(False)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetOpacity(opacity)
    prop.SetLineWidth(width)
    prop.LightingOff()
    return actor


class OrientationCube(QObject):
    """A transparent VTK layer; mouse events are handled by the scene widget.

    The main renderer's StartEvent updates the small camera before the same
    render window draws this layer. There is no second native window, paint
    event, timer or buffer swap for the cube.
    """

    def __init__(self, plotter):
        super().__init__(plotter)
        self.plotter = plotter
        self.faces = []
        self.hover = None
        self._visible = True
        self._pressed = False
        self._disposed = False
        self._frame_key = None
        self._prior_cursor = None
        self.renderer = vtkRenderer()
        self.renderer.SetInteractive(False)
        self.renderer.SetPreserveColorBuffer(True)
        self.renderer.SetPreserveDepthBuffer(False)
        window = plotter.render_window
        layer = window.GetNumberOfLayers()
        self.renderer.SetLayer(layer)
        window.SetNumberOfLayers(layer + 1)
        window.AddRenderer(self.renderer)
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelProjection(True)
        camera.SetParallelScale(SIZE / (2 * SCALE))
        camera.SetClippingRange(.1, 20.)

        self.face_actors = {}
        self._face_vertices = {}
        for axis in range(3):
            for sign in (-1, 1):
                normal = np.eye(3)[axis] * sign
                u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
                vertices = [normal + a * u + b * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                cells = [(0, 1, 2, 3) if sign > 0 else (3, 2, 1, 0)]
                actor = _actor(_polydata(vertices, cells), (.31, .39, .45), opacity=(.07, .11, .04)[axis])
                actor.GetProperty().BackfaceCullingOn()
                self.renderer.AddActor(actor)
                self.face_actors[axis, sign] = actor
                self._face_vertices[axis, sign] = vertices

        vertices, edges = [], []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for a in (-1, 1):
                for b in (-1, 1):
                    start = np.zeros(3)
                    start[others] = (a, b)
                    start[axis] = -1
                    end = start.copy()
                    end[axis] = 1
                    edges.append((len(vertices), len(vertices) + 1))
                    vertices.extend((start, end))
        self.wire = _actor(_polydata(vertices, edges, lines=True), (.57, .62, .66), width=.9)
        self.renderer.AddActor(self.wire)
        self.axis_actors = []
        self.labels = []
        for axis, color in enumerate(AXIS_COLORS):
            endpoint = ORIGIN.copy()
            endpoint[axis] = 1
            actor = _actor(_polydata([ORIGIN, endpoint], [(0, 1)], lines=True), color, width=2.)
            self.renderer.AddActor(actor)
            self.axis_actors.append(actor)
            label = vtkTextActor()
            label.SetInput('XYZ'[axis])
            label.SetTextScaleModeToNone()
            label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            prop = label.GetTextProperty()
            prop.SetColor(*color)
            prop.SetFontSize(10)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToCentered()
            label.SetPickable(False)
            self.renderer.AddViewProp(label)
            self.labels.append(label)

        plotter.installEventFilter(self)
        self._observed_camera = plotter.camera
        self.observer = self._observed_camera.AddObserver('ModifiedEvent', self.camera_changed)
        self.render_observer = plotter.renderer.AddObserver('StartEvent', self._sync_camera, 1.)
        self._sync_camera()

    def geometry(self):
        return QRect(MARGIN, self.plotter.height() - SIZE - MARGIN, SIZE, SIZE)

    def setVisible(self, visible):
        self._visible = bool(visible)
        self.renderer.SetDraw(self._visible)
        self._set_hover(None, render=False)
        self.plotter.render()

    def isVisible(self):
        return self._visible and not self._disposed

    def camera_changed(self, *_):
        actor = self.plotter.actors.get('plat_base')
        if actor:
            opacity = .12 if self.plotter.camera.position[2] < 0 else 1.
            if actor.GetProperty().GetOpacity() != opacity:
                actor.GetProperty().SetOpacity(opacity)

    def _sync_camera(self, *_):
        if self._disposed:
            return
        # Hardware selection renders colour IDs; this purely visual layer must
        # neither cover those IDs nor acquire a selectable model face ID.
        visible = self._visible and not self.plotter.renderer.GetSelector()
        self.renderer.SetDraw(visible)
        if not visible:
            return
        self.renderer.SetUseFXAA(self.plotter.renderer.GetUseFXAA())
        self.camera_changed()
        width, height = max(self.plotter.width(), 1), max(self.plotter.height(), 1)
        self.renderer.SetViewport(MARGIN / width, MARGIN / height,
                                  (MARGIN + SIZE) / width, (MARGIN + SIZE) / height)
        matrix = self.plotter.camera.GetViewTransformMatrix()
        rotation = np.array([[matrix.GetElement(i, j) for j in range(3)] for i in range(3)])
        key = tuple(rotation.ravel())
        if key == self._frame_key:
            return
        self._frame_key = key
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(*(rotation[2] * 8.))
        camera.SetFocalPoint(0., 0., 0.)
        camera.SetViewUp(*rotation[1])

        def project(point):
            point = rotation @ point
            return QPointF(SIZE / 2 + point[0] * SCALE, SIZE / 2 - point[1] * SCALE)

        faces = []
        for (axis, sign), vertices in self._face_vertices.items():
            depth = rotation[2, axis] * sign
            if depth > 1e-8:
                faces.append((depth, QPolygonF([project(point) for point in vertices]), axis, sign))
        self.faces = [(polygon, axis, sign) for _, polygon, axis, sign in sorted(faces, key=lambda face: face[0])]
        start = project(ORIGIN)
        for axis, label in enumerate(self.labels):
            endpoint = ORIGIN.copy()
            endpoint[axis] = 1
            end = project(endpoint)
            direction = end - start
            length = np.hypot(direction.x(), direction.y())
            if length < 1e-6:
                direction = end - project(np.zeros(3))
                length = np.hypot(direction.x(), direction.y())
            if length < 1e-6:
                direction, length = QPointF(0, -1), 1.
            position = end + direction * (12 / length)
            label.GetPositionCoordinate().SetValue(position.x() / SIZE, 1. - position.y() / SIZE)

    def face_at(self, point):
        return next(((axis, sign) for polygon, axis, sign in reversed(self.faces)
                     if polygon.containsPoint(point, Qt.OddEvenFill)), None)

    def _set_hover(self, face, *, render=True):
        if face == self.hover:
            return
        if self.hover is not None:
            old = self.face_actors[self.hover].GetProperty()
            old.SetColor(.31, .39, .45)
            old.SetOpacity((.07, .11, .04)[self.hover[0]])
        if face is not None:
            prop = self.face_actors[face].GetProperty()
            prop.SetColor(.12, .51, .75)
            prop.SetOpacity(.25)
            if self._prior_cursor is None:
                self._prior_cursor = self.plotter.cursor()
            self.plotter.setCursor(Qt.PointingHandCursor)
        elif self._prior_cursor is not None:
            self.plotter.setCursor(self._prior_cursor)
            self._prior_cursor = None
        self.hover = face
        if render:
            self.plotter.render()

    def eventFilter(self, obj, event):
        if obj is not self.plotter or not self.isVisible():
            return False
        kind = event.type()
        if kind == QEvent.Leave:
            self._set_hover(None)
        elif kind in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            local = event.position() - self.geometry().topLeft()
            face = self.face_at(local)
            if kind == QEvent.MouseMove:
                if self._pressed:
                    return True
                if not event.buttons():
                    self._set_hover(face)
                return False
            if event.button() == Qt.LeftButton and face is not None:
                self._pressed = True
                self._set_hover(None, render=False)
                self.orient(None if kind == QEvent.MouseButtonDblClick else face[0], face[1])
                event.accept()
                return True
        elif kind == QEvent.MouseButtonRelease and self._pressed and event.button() == Qt.LeftButton:
            self._pressed = False
            event.accept()
            return True
        return False

    def orient(self, axis=None, sign=1):
        camera = self.plotter.camera
        focus = np.asarray(camera.focal_point)
        distance = max(np.linalg.norm(np.asarray(camera.position) - focus), 1.)
        direction = np.array([1., 1., 1.]) / np.sqrt(3) if axis is None else np.eye(3)[axis] * sign
        camera.position = focus + direction * distance
        camera.up = (0, 1, 0) if axis == 2 else (0, 0, 1)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def dispose(self):
        if self._disposed:
            return
        self._set_hover(None, render=False)
        self._disposed = True
        self.plotter.removeEventFilter(self)
        self._observed_camera.RemoveObserver(self.observer)
        self.plotter.renderer.RemoveObserver(self.render_observer)
        self.plotter.render_window.RemoveRenderer(self.renderer)
