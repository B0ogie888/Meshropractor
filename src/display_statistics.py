"""Event-driven statistics of selected parts, rendered in the same VTK frame."""
import itertools

import numpy as np
from PySide6.QtCore import QObject, QEvent, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QWidget
from vtkmodules.vtkCommonCore import VTK_FONT_FILE
from vtkmodules.vtkRenderingCore import vtkTextActor

from display_ribbon import STATISTICS
from part_supports import support_mesh


class LiveStatistics(QObject):
    def __init__(self, tools):
        super().__init__(tools)
        self.tools = tools
        self.actors = {}
        self.lines = []
        self.data = {}
        self._support_cache = {}
        self._plotter = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.update)

    def request(self):
        if any(self.tools.state[key] for key in STATISTICS): self.timer.start()

    def calculate(self):
        from display_tools import solid_volume, material_estimate
        window, plotter = self.tools.window, self.tools.plotter
        parts_volume, supports_volume, unknown = 0., 0., 0
        bounds, rows, used = [], window.selected_slicer_rows(), set()
        for row in rows:
            part = window.slicer_parts[row]
            actor = plotter.actors.get(part['actor_name'])
            vtk_matrix = actor.GetMatrix() if actor is not None else None
            matrix = np.array([[vtk_matrix.GetElement(i, j) for j in range(4)] for i in range(4)]) if vtk_matrix else np.eye(4)
            scale = abs(float(np.linalg.det(matrix[:3, :3])))
            volume = solid_volume(part['mesh'])
            parts_volume += (volume or 0.) * scale
            unknown += int(volume is None)
            components = [part['mesh'].bounds]
            for group in part.get('supports', []):
                if not len(group['faces']): continue
                key = (id(group['vertices']), id(group['faces']))
                used.add(key)
                if key not in self._support_cache:
                    mesh = support_mesh(group)
                    self._support_cache[key] = (group['vertices'], group['faces'], solid_volume(mesh), mesh.bounds.copy())
                _, _, support_volume, support_bounds = self._support_cache[key]
                supports_volume += (support_volume or 0.) * scale
                unknown += int(support_volume is None)
                components.append(support_bounds)
            for lo, hi in components:
                corners = np.array(list(itertools.product(*zip(lo, hi))))
                world = corners @ matrix[:3, :3].T + matrix[:3, 3]
                bounds.append(np.array([world.min(axis=0), world.max(axis=0)]))
        self._support_cache = {key: value for key, value in self._support_cache.items() if key in used}
        total = parts_volume + supports_volume
        height = max(0., max((value[1, 2] for value in bounds), default=0.))
        platform = self.tools.active_platform()
        usage, packing = None, None
        if platform:
            x, y, z = platform['dim']
            usage = total / (x * y * z) * 100.
            packing = total / (x * y * height) * 100. if height > 1e-12 else (0. if total == 0 else None)
        mass, cost = material_estimate(total, self.tools._density, self.tools._price)
        return dict(count=len(rows), parts_mm3=parts_volume, supports_mm3=supports_volume,
                    total_mm3=total, unknown=unknown, height_mm=height, usage_percent=usage,
                    packing_percent=packing, mass_g=mass, cost=cost)

    def update(self):
        self.timer.stop()
        plotter = self.tools.plotter
        if plotter is None: return
        if plotter is not self._plotter:
            if isinstance(self._plotter, QWidget): self._plotter.removeEventFilter(self)
            self._plotter = plotter
            if isinstance(plotter, QWidget): plotter.installEventFilter(self)
            self.actors.clear()
        enabled = any(self.tools.state[key] for key in STATISTICS)
        if not enabled:
            for actor in self.actors.values(): actor.SetVisibility(False)
            self.lines = []
            plotter.render()
            return
        self.data = data = self.calculate()
        partial = '≥ ' if data['unknown'] else ''
        lines = [('Выбранные детали', str(data['count']))]
        if self.tools.state['packing_density']:
            percent = lambda value: '—' if value is None else f'{partial}{value:.2f} %'
            lines += [('Использование объёма платформы', percent(data['usage_percent'])),
                      ('Текущая плотность размещения', percent(data['packing_percent'])),
                      ('Высота сборки', f"{data['height_mm']:.2f} мм")]
        if self.tools.state['volume']:
            lines += [('Объём деталей', f"{partial}{data['parts_mm3']:.2f} мм³"),
                      ('Объём поддержек', f"{partial}{data['supports_mm3']:.2f} мм³"),
                      ('Суммарный объём', f"{partial}{data['total_mm3']:.2f} мм³")]
        if self.tools.state['material_cost']:
            lines += [('Масса материала', f"{partial}{data['mass_g']:.3f} г"),
                      ('Стоимость материала', f"{partial}{data['cost']:.2f}")]
        if data['unknown']: lines.append(('Неопределённые объёмы', str(data['unknown'])))
        self.lines = lines
        preferences = getattr(self.tools.window.ui, 'display_preferences', None)
        color = QColor(getattr(preferences, 'background', '#ffffff'))
        background = (color.redF(), color.greenF(), color.blueF())
        foreground = (.08, .08, .08) if color.lightnessF() > .5 else (.95, .95, .95)
        for column, name in enumerate(('labels', 'values')):
            key = 'scene_statistics_' + name
            actor = self.actors.get(name)
            if actor is None or key not in plotter.actors:
                actor = vtkTextActor(); actor.SetPickable(False)
                prop = actor.GetTextProperty(); prop.SetFontSize(13); prop.SetVerticalJustificationToTop()
                if name == 'values': prop.SetJustificationToRight()
                if self.tools._font_file:
                    prop.SetFontFamily(VTK_FONT_FILE); prop.SetFontFile(self.tools._font_file)
                actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                plotter.add_actor(actor, name=key, pickable=False, reset_camera=False, render=False)
                self.actors[name] = actor
            actor.SetVisibility(True)
            actor.SetInput('\n'.join(line[column] for line in lines))
            prop = actor.GetTextProperty(); prop.SetColor(*foreground)
            prop.SetBackgroundColor(*background); prop.SetBackgroundOpacity(.7)
        self.position()
        plotter.render()

    def position(self):
        if not self.lines or len(self.actors) != 2: return
        plotter = self.tools.plotter
        width = max(plotter.width(), 1) if isinstance(plotter, QWidget) else 1000
        font = QFont(); font.setPixelSize(16); metrics = QFontMetrics(font)
        labels = max(metrics.horizontalAdvance(line[0]) for line in self.lines)
        values = max(metrics.horizontalAdvance(line[1]) for line in self.lines)
        right = 1. - 14. / width
        self.actors['values'].SetPosition(right, .97)
        self.actors['labels'].SetPosition(max(.01, right - (labels + values + 24.) / width), .97)

    def eventFilter(self, obj, event):
        if obj is self._plotter and event.type() == QEvent.Resize:
            self.position()
        return False
