"""Right-drag orbit/roll and unfilled guides in the existing scene frame."""
import math
from PySide6.QtCore import QObject, QEvent, QPointF, QRect, Qt, QTimer
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D


class SceneOutline(QObject):
    """Line-only screen overlay: no native child widget, fill or extra frame."""
    def __init__(self, plotter):
        super().__init__(plotter)
        self.plotter = plotter
        self._rect = QRect()
        self.data = vtkPolyData()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(self.data)
        self.actor = vtkActor2D()
        self.actor.SetMapper(mapper)
        self.actor.SetPickable(False)
        self.actor.SetVisibility(False)
        self.actor.GetProperty().SetColor(.25, .6, .95)

    def _lines(self, vertices, segments):
        ratio = self.plotter.devicePixelRatioF()
        height = self.plotter.render_window.GetSize()[1]
        points, lines = vtkPoints(), vtkCellArray()
        for x, y in vertices:
            points.InsertNextPoint(x * ratio, height - 1 - y * ratio, 0.)
        for segment in segments:
            lines.InsertNextCell(len(segment))
            for index in segment: lines.InsertCellPoint(index)
        self.data.SetPoints(points)
        self.data.SetLines(lines)
        self.actor.GetProperty().SetLineWidth(max(1., ratio))
        if self.actor.GetVisibility(): self.plotter.render()

    def setGeometry(self, rect):
        self._rect = QRect(rect)
        self._lines([(rect.left(), rect.top()), (rect.right(), rect.top()),
                     (rect.right(), rect.bottom()), (rect.left(), rect.bottom())], [(0, 1, 2, 3, 0)])

    def geometry(self):
        return QRect(self._rect)

    def circle(self, center, radius):
        vertices, segments = [], []
        for dash in range(64):
            start = len(vertices)
            for i in range(4):
                angle = (dash + .65 * i / 3) * math.tau / 64
                vertices.append((center.x() + radius * math.cos(angle),
                                 center.y() + radius * math.sin(angle)))
            segments.append(tuple(range(start, len(vertices))))
        background = self.plotter.renderer.GetBackground()
        color = .45 if sum(background) > 1.5 else .8
        self.actor.GetProperty().SetColor(color, color, color)
        self._lines(vertices, segments)

    def show(self):
        # Project replacement can clear every prop from the renderer.
        if not self.plotter.renderer.HasViewProp(self.actor):
            self.plotter.renderer.AddViewProp(self.actor)
        self.actor.SetVisibility(True)
        self.plotter.render()

    def hide(self, *, render=True):
        visible = self.actor.GetVisibility()
        self.actor.SetVisibility(False)
        if visible and render: self.plotter.render()

    def dispose(self):
        self.plotter.renderer.RemoveViewProp(self.actor)


class SceneNavigation(QObject):
    def __init__(self, plotter, click=None):
        super().__init__(plotter)
        self.plotter, self.click = plotter, click
        self.guide = SceneOutline(plotter)
        self.start = self.last = None
        self.dragged = False
        self.interacting = False
        self.mode = None
        plotter.installEventFilter(self)

    def circle_geometry(self):
        return QPointF(self.plotter.width() / 2, self.plotter.height() / 2), min(self.plotter.width(), self.plotter.height()) / 3

    def cancel(self):
        if self.start is None: return
        self.guide.hide(render=False)
        self.start = self.last = None
        if self.interacting:
            self.plotter.iren.interactor.InvokeEvent('EndInteractionEvent')
        self.interacting = False
        self.plotter.render()

    def move(self, point):
        dx, dy = point.x() - self.last.x(), point.y() - self.last.y()
        if dx == 0 and dy == 0: return
        camera = self.plotter.camera
        if self.mode == 'orbit':
            camera.Azimuth(-200 * dx / max(1, self.plotter.width()))
            camera.Elevation(200 * dy / max(1, self.plotter.height()))
            camera.OrthogonalizeViewUp()
        else:
            before, after = self.last - self.center, point - self.center
            # A roll drag passing through the centre has no defined angle.
            if math.hypot(after.x(), after.y()) > 2 and math.hypot(before.x(), before.y()) > 2:
                delta = math.atan2(after.y(), after.x()) - math.atan2(before.y(), before.x())
                delta = math.atan2(math.sin(delta), math.cos(delta))
                camera.Roll(-math.degrees(delta))
        self.last = QPointF(point)
        self.plotter.reset_camera_clipping_range()
        self.plotter.iren.interactor.InvokeEvent('InteractionEvent')
        self.plotter.render()

    def eventFilter(self, obj, event):
        if obj is not self.plotter: return False
        kind = event.type()
        if kind in (QEvent.Hide, QEvent.WindowDeactivate, QEvent.FocusOut):
            self.cancel()
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape and self.start is not None:
            self.cancel()
            return True
        if kind == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            if event.buttons() & (Qt.LeftButton | Qt.MiddleButton): return False
            self.start = self.last = QPointF(event.position())
            self.center, radius = self.circle_geometry()
            offset = self.start - self.center
            self.mode = 'orbit' if math.hypot(offset.x(), offset.y()) <= radius else 'roll'
            self.dragged = self.interacting = False
            self.guide.circle(self.center, radius)
            self.guide.show()
            event.accept()
            return True
        if kind == QEvent.MouseMove and self.start is not None:
            if not event.buttons() & Qt.RightButton:
                self.cancel()
                return False
            if not self.dragged and (event.position() - self.start).manhattanLength() >= 4:
                self.dragged = self.interacting = True
                self.plotter.iren.interactor.InvokeEvent('StartInteractionEvent')
            if self.dragged: self.move(event.position())
            event.accept()
            return True
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.RightButton and self.start is not None:
            clicked = not self.dragged and (event.position() - self.start).manhattanLength() < 4
            if self.dragged: self.move(event.position())
            position = event.globalPosition().toPoint()
            self.cancel()
            if clicked and self.click:
                QTimer.singleShot(0, lambda: self.click(position))
            event.accept()
            return True
        return False

    def dispose(self):
        self.cancel()
        self.plotter.removeEventFilter(self)
        self.guide.dispose()
