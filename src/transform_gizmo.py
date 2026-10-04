"""Thin world-axis handles with planar translation and screen-plane rotation.

Qt owns the complete left-button gesture, so VTK cannot leave camera rotation
active after a handle drag. Picking uses projected contours with a pixel-sized
hit margin; thin lines remain easy to grab at any zoom or display scale.
"""
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QObject, QEvent, Qt, QTimer
from vtkmodules.vtkRenderingCore import vtkRenderer
from transform_math import unit


COLORS = ('#ef5350', '#72c64b', '#448aff')
PLANE_EXTENT = .84  # From the origin to the base of each arrowhead.


def ray_plane(start, direction, origin, normal):
    denominator = float(np.dot(direction, normal))
    if abs(denominator) < 1e-8: return None
    return start + direction * (np.dot(origin - start, normal) / denominator)


def axis_drag_normal(axis, view):
    """Most camera-facing plane containing a translation axis."""
    normal = view - axis * np.dot(view, axis)
    return unit(normal) if np.linalg.norm(normal) > 1e-8 else None


def signed_angle(before, after, normal):
    before, after = unit(before), unit(after)
    return float(np.arctan2(np.dot(normal, np.cross(before, after)), np.dot(before, after)))


def rotate_about(center, normal, angle):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(normal * angle).as_matrix()
    matrix[:3, 3] = center - matrix[:3, :3] @ center
    return matrix


class TransformGizmo(QObject):
    def __init__(self, plotter, operation, callback, released):
        super().__init__(plotter)
        self.plotter, self.operation = plotter, operation
        self.callback, self.released = callback, released
        self.origin = np.zeros(3)
        self.axes = np.eye(3)
        self.matrix = np.eye(4)
        self.snap = None
        self.along_line = False
        self.length = 1.
        self.actors, self.contours, self.planes = {}, {}, {}
        self.drag = None
        self.hover = None
        self.removed = False
        self._geometry_key = None
        self._view_normal = np.array([0., 0., 1.])
        self._cursor = plotter.cursor()
        self._tracking = plotter.hasMouseTracking()
        # A depth-independent overlay keeps thin lines uniform under MSAA.
        # Large polygon offsets distort polylines at grazing camera angles.
        self.renderer = vtkRenderer()
        self.layer = plotter.render_window.GetNumberOfLayers()
        self.renderer.SetLayer(self.layer)
        self.renderer.SetPreserveColorBuffer(True)
        self.renderer.SetPreserveDepthBuffer(False)
        self.renderer.SetInteractive(False)
        self.renderer.SetActiveCamera(plotter.camera)
        self.renderer.SetViewport(plotter.renderer.GetViewport())
        plotter.render_window.SetNumberOfLayers(self.layer + 1)
        plotter.render_window.AddRenderer(self.renderer)
        plotter.setMouseTracking(True)
        plotter.installEventFilter(self)
        self.camera_observer = plotter.camera.AddObserver('ModifiedEvent', self.camera_changed)

    @property
    def dragging(self): return self.drag is not None

    def configure(self, origin, axes, matrix, length, along_line=False, snap=None):
        if self.dragging:
            if self.operation == 'Перемещать':
                offset = np.eye(4)
                offset[:3, 3] = matrix[:3, 3] - self.drag['matrix'][:3, 3]
                for actors in self.actors.values():
                    for actor in actors: actor.user_matrix = offset
            return
        self.origin, self.axes, self.matrix = np.array(origin), np.array(axes), np.array(matrix)
        self.length, self.along_line, self.snap = length, along_line, snap
        self.sync_geometry()

    def camera_basis(self):
        camera = self.plotter.camera
        direction = np.asarray(camera.position) - camera.focal_point
        # Camera setters emit ModifiedEvent between position/focal/up changes.
        # Those intermediate frames can momentarily have a parallel view-up.
        if np.linalg.norm(direction) > 1e-8: self._view_normal = unit(direction)
        normal = self._view_normal
        right = np.cross(camera.up, normal)
        if np.linalg.norm(right) < 1e-8:
            reference = [0, 0, 1] if abs(normal[2]) < .9 else [0, 1, 0]
            right = np.cross(reference, normal)
        right = unit(right)
        return normal, right, np.cross(normal, right)

    def project(self, points):
        renderer = self.plotter.renderer
        matrix = pv.array_from_vtkmatrix(self.plotter.camera.GetCompositeProjectionTransformMatrix(
            renderer.GetTiledAspectRatio(), 0, 1))
        points = np.atleast_2d(points)
        clip = np.c_[points, np.ones(len(points))] @ matrix.T
        with np.errstate(divide='ignore', invalid='ignore'): ndc = clip[:, :3] / clip[:, 3:4]
        width, height = renderer.GetSize()
        x, y = renderer.GetOrigin()
        result = np.c_[x + (ndc[:, 0] + 1) * width / 2,
                       self.plotter.render_window.GetSize()[1] - 1 - y - (ndc[:, 1] + 1) * height / 2]
        return result / self.plotter.devicePixelRatioF()

    def ray(self, point):
        renderer = self.plotter.renderer
        ratio = self.plotter.devicePixelRatioF()
        x, y = point
        result = []
        for z in (0, 1):
            renderer.SetDisplayPoint(x * ratio, self.plotter.render_window.GetSize()[1] - 1 - y * ratio, z)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint())
            result.append(world[:3] / world[3])
        return result[0], unit(result[1] - result[0])

    def camera_changed(self, *_):
        if not self.removed and not self.dragging: self.sync_geometry()

    def sync_geometry(self):
        self.renderer.SetUseFXAA(self.plotter.renderer.GetUseFXAA())
        normal, right, up = self.camera_basis()
        height = max(1, self.plotter.renderer.GetSize()[1] / self.plotter.devicePixelRatioF())
        camera = self.plotter.camera
        # Bound the handles in screen pixels without changing the model's scale.
        world_height = (2 * camera.parallel_scale if camera.parallel_projection else
                        2 * abs(np.dot(np.asarray(camera.position) - self.origin, normal)) *
                        np.tan(np.deg2rad(camera.view_angle / 2)))
        per_pixel = max(world_height / height, 1e-9)
        if self.operation == 'Перемещать': size = np.clip(self.length * .4, per_pixel * 64, per_pixel * 96)
        else: size = np.clip(self.length * .58, per_pixel * 86, per_pixel * 140)
        self.size = float(size)
        key = (tuple(self.origin), tuple(self.axes.flat), self.size, self.along_line,
               tuple(normal), tuple(right), tuple(self.plotter.renderer.GetBackground()))
        if key == self._geometry_key: return
        self._geometry_key = key
        self.contours, self.planes = {}, {}
        for actors in self.actors.values():
            for actor in actors: actor.visibility = False
        if self.operation == 'Перемещать':
            for index, axis in enumerate(self.axes):
                if self.along_line and index != 2: continue
                name = 'axis_' + 'xyz'[index]
                points = self.origin + np.array([0., .86])[:, None] * size * axis
                self.line(name, points, COLORS[index], width=1.5)
                self.contours[name] = (np.vstack([self.origin, self.origin + size * axis]), axis)
                self.mesh(name + '_tip', pv.Cone(center=self.origin + .92 * size * axis,
                    direction=axis, height=.16 * size, radius=.035 * size, resolution=12), COLORS[index])
            if not self.along_line:
                for a, b in ((0, 1), (0, 2), (1, 2)):
                    name = 'plane_' + 'xyz'[a] + 'xyz'[b]
                    uv = np.array([[0., 0.], [PLANE_EXTENT, 0.],
                                   [PLANE_EXTENT, PLANE_EXTENT], [0., PLANE_EXTENT]])
                    points = self.origin + size * (uv[:, :1] * self.axes[a] + uv[:, 1:] * self.axes[b])
                    normal_plane = unit(np.cross(self.axes[a], self.axes[b]))
                    self.planes[name] = (a, b, normal_plane, points)
                    color = COLORS[3 - a - b]
                    self.mesh(name, pv.PolyData(points, faces=[4, 0, 1, 2, 3]), color, opacity=.22)
                    self.line(name + '_edge', np.vstack([points, points[0]]), color, width=1.)
        else:
            for index, axis in enumerate(self.axes):
                if self.along_line and index != 2: continue
                a, b = self.axes[(index + 1) % 3], self.axes[(index + 2) % 3]
                self.ring('ring_' + 'xyz'[index], axis, a, b, size, COLORS[index])
            if not self.along_line:
                # The platform stays white in both themes; neutral grey is
                # readable on it as well as on either viewport background.
                color = '#6c7875'
                self.ring('ring_screen', normal, right, up, size * 1.22, color)
        self.highlight(self.hover)

    def ring(self, name, normal, a, b, radius, color):
        angles = np.linspace(0, 2 * np.pi, 193)
        points = self.origin + radius * (np.cos(angles)[:, None] * a + np.sin(angles)[:, None] * b)
        self.contours[name] = (points, normal)
        self.line(name, points, color, width=1.5)

    def mesh(self, name, data, color, opacity=1., width=None):
        if name not in self.actors:
            actor = self.plotter.add_mesh(data, color=color, opacity=opacity, lighting=False,
                pickable=False, reset_camera=False, render=False, name='transform_gizmo_' + name,
                line_width=width or 1., render_lines_as_tubes=False)
            self.plotter.renderer.RemoveActor(actor)
            self.renderer.AddActor(actor)
            self.actors[name] = [actor]
        else:
            actor = self.actors[name][0]
            actor.mapper.dataset = data
        actor.visibility = True
        actor.user_matrix = np.eye(4)
        actor.prop.color = color
        actor.prop.opacity = opacity
        if width: actor.prop.line_width = width
        actor._gizmo_color, actor._gizmo_opacity = color, opacity

    def line(self, name, points, color, width):
        # Supplying lines at construction avoids implicit vertex cells (dots).
        data = pv.PolyData(points, lines=np.r_[len(points), np.arange(len(points))])
        self.mesh(name, data, color, width=width)

    def hit(self, point):
        # Lines have priority over translucent plane interiors.
        closest = None
        for name, (points, normal) in self.contours.items():
            projected = self.project(points)
            start, segment = projected[:-1], np.diff(projected, axis=0)
            squared = np.einsum('ij,ij->i', segment, segment)
            valid = np.isfinite(squared) & (squared > .01)
            if not valid.any(): continue
            t = np.zeros(len(squared))
            t[valid] = np.clip(np.einsum('ij,ij->i', point - start, segment)[valid] / squared[valid], 0, 1)
            distances = np.linalg.norm(start + t[:, None] * segment - point, axis=1)
            distances[~valid] = np.inf
            index = int(np.argmin(distances))
            distance = distances[index]
            if distance <= 7 and (closest is None or distance < closest[0]):
                world = points[index] + t[index] * (points[index + 1] - points[index])
                closest = distance, name, world, normal
        if closest is not None: return closest[1:]
        start, direction = self.ray(point)
        planes = []
        for name, (a, b, normal, points) in self.planes.items():
            world = ray_plane(start, direction, self.origin, normal)
            if world is None: continue
            uv = (world - self.origin) @ self.axes[[a, b]].T / self.size
            if np.all(uv >= 0) and np.all(uv <= PLANE_EXTENT):
                distance = np.dot(world - start, direction)
                if distance >= 0: planes.append((float(distance), name, world, normal))
        # Where the enlarged planes overlap, grab the one nearest the camera.
        return min(planes, key=lambda item: item[0])[1:] if planes else None

    def highlight(self, name):
        for key, actors in self.actors.items():
            active = key == name or key == str(name) + '_edge' or key == str(name) + '_tip'
            for actor in actors:
                actor.prop.color = '#e0cb32' if active else actor._gizmo_color
                actor.prop.opacity = .4 if active and key.startswith('plane_') and not key.endswith('_edge') else actor._gizmo_opacity

    def begin(self, point):
        hit = self.hit(point)
        if hit is None: return False
        name, world, normal = hit
        start, direction = self.ray(point)
        view = self.camera_basis()[0]
        if name.startswith('axis_'):
            normal = axis_drag_normal(normal, view)
            if normal is None: return False
        initial = ray_plane(start, direction, self.origin, normal)
        if not name.startswith('ring_') and initial is None: return False
        self.drag = dict(name=name, normal=normal, start=point.copy(), matrix=self.matrix.copy(),
                         origin=self.origin.copy(), initial=initial, world=world,
                         last_angle=0., accumulated=0.)
        if name.startswith('ring_'):
            tangent = unit(np.cross(normal, world - self.origin))
            screen_tangent = self.project([world + tangent * self.size])[0] - self.project([world])[0]
            self.drag['tangent'] = screen_tangent
            self.drag['fallback'] = abs(np.dot(normal, direction)) < .08 or initial is None
        self.hover = name
        self.highlight(name)
        self.plotter.setCursor(Qt.ClosedHandCursor)
        self.plotter.iren.interactor.InvokeEvent('StartInteractionEvent')
        self.plotter.render()
        return True

    def move(self, point):
        drag = self.drag
        if drag is None: return
        name, normal = drag['name'], drag['normal']
        start, direction = self.ray(point)
        current = ray_plane(start, direction, drag['origin'], normal)
        if name.startswith('ring_'):
            if drag['fallback']:
                tangent = drag['tangent']
                angle = float(np.dot(point - drag['start'], tangent) / max(np.dot(tangent, tangent), 1.))
            else:
                if current is None or np.linalg.norm(current - drag['origin']) < 1e-9: return
                angle = signed_angle(drag['initial'] - drag['origin'], current - drag['origin'], normal)
                difference = angle - drag['last_angle']
                drag['accumulated'] += np.arctan2(np.sin(difference), np.cos(difference))
                drag['last_angle'] = angle
                angle = drag['accumulated']
            if self.snap: angle = np.deg2rad(round(np.rad2deg(angle) / self.snap) * self.snap)
            matrix = rotate_about(drag['origin'], normal, angle) @ drag['matrix']
        else:
            if current is None: return
            delta = current - drag['initial']
            if name.startswith('axis_'):
                axis = self.axes['xyz'.index(name[-1])]
                distance = np.dot(delta, axis)
                if self.snap: distance = round(distance / self.snap) * self.snap
                delta = axis * distance
            else:
                a, b = self.planes[name][:2]
                coordinates = delta @ self.axes[[a, b]].T
                if self.snap: coordinates = np.round(coordinates / self.snap) * self.snap
                delta = coordinates @ self.axes[[a, b]]
            matrix = drag['matrix'].copy()
            matrix[:3, 3] += delta
        self.callback(matrix)
        self.plotter.iren.interactor.InvokeEvent('InteractionEvent')

    def finish(self, cancel=False):
        if self.drag is None: return
        matrix = self.drag['matrix']
        self.drag = None
        for actors in self.actors.values():
            for actor in actors: actor.user_matrix = np.eye(4)
        if cancel: self.callback(matrix)
        self.plotter.setCursor(self._cursor)
        self.plotter.iren.interactor.InvokeEvent('EndInteractionEvent')
        self.released()

    def eventFilter(self, obj, event):
        if obj is not self.plotter or self.removed: return False
        kind = event.type()
        if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if event.buttons() & (Qt.RightButton | Qt.MiddleButton): return False
            if self.begin(np.array([event.position().x(), event.position().y()])):
                event.accept(); return True
        if kind == QEvent.MouseMove:
            point = np.array([event.position().x(), event.position().y()])
            if self.dragging:
                if not event.buttons() & Qt.LeftButton: self.finish()
                else: self.move(point)
                event.accept(); return True
            hit = self.hit(point)
            name = hit[0] if hit else None
            if name != self.hover:
                self.hover = name; self.highlight(name)
                self.plotter.setCursor(Qt.OpenHandCursor if name else self._cursor)
                self.plotter.render()
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and self.dragging:
            self.move(np.array([event.position().x(), event.position().y()]))
            self.finish(); event.accept(); return True
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape and self.dragging:
            self.finish(cancel=True); event.accept(); return True
        if kind in (QEvent.Hide, QEvent.WindowDeactivate, QEvent.FocusOut) and self.dragging: self.finish()
        if kind == QEvent.Resize: QTimer.singleShot(0, self.camera_changed)
        return False

    def remove(self):
        self.removed = True
        if self.dragging: self.plotter.iren.interactor.InvokeEvent('EndInteractionEvent')
        self.drag = None
        self.plotter.removeEventFilter(self)
        self.plotter.camera.RemoveObserver(self.camera_observer)
        self.plotter.setCursor(self._cursor)
        self.plotter.setMouseTracking(self._tracking)
        for actors in self.actors.values():
            for actor in actors:
                self.renderer.RemoveActor(actor)
                self.plotter.remove_actor(actor, render=False)
        self.actors.clear()
        self.plotter.render_window.RemoveRenderer(self.renderer)
        if self.plotter.render_window.GetNumberOfLayers() == self.layer + 1:
            self.plotter.render_window.SetNumberOfLayers(self.layer)
        self.deleteLater()
