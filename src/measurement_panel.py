"""Interactive distances, circles and angles on original part surfaces."""
import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QComboBox, QLabel, QPushButton, QCheckBox, QListWidget, QApplication
from measurement_geometry import distance, project_plane, circle, angle, normal_angle


class MeasurementPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.window = self.workspace = None
        self.active = None
        self.hits = []
        self.names = []
        self.serial = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        self.tabs = QTabWidget()
        self.modes = []
        for title, values in [('Расстояние', ['Точка — точка', 'Точка — плоскость', 'Точка — поверхность детали', 'Плоскость — плоскость']),
                              ('Окружность', ['Радиус и диаметр по 3 точкам']), ('Угол', ['Три точки (вторая — вершина)', 'Между плоскостями'])]:
            page = QWidget()
            row = QHBoxLayout(page)
            combo = QComboBox()
            combo.addItems(values)
            row.addWidget(combo)
            self.tabs.addTab(page, title)
            self.modes.append(combo)
        layout.addWidget(self.tabs)
        self.hint = QLabel('Укажите режим измерения и нажмите «Выбрать». Единицы: мм и градусы.')
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.results = QListWidget()
        self.results.setMinimumHeight(80)
        self.results.setMaximumHeight(155)
        self.results.setWordWrap(True)
        layout.addWidget(self.results)
        self.hidden = QCheckBox('Скрыть измерения в сцене')
        self.hidden.toggled.connect(self.set_hidden)
        layout.addWidget(self.hidden)
        row = QHBoxLayout()
        self.start_button = QPushButton('Выбрать')
        self.start_button.clicked.connect(self.start)
        self.clear_button = QPushButton('Очистить')
        self.clear_button.clicked.connect(self.clear)
        copy_button = QPushButton('Копировать')
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText('\n'.join(self.results.item(i).text() for i in range(self.results.count()))))
        for button in (self.start_button, self.clear_button, copy_button): row.addWidget(button)
        layout.addLayout(row)

    def bind(self, window, workspace):
        self.window, self.workspace = window, workspace

    def start(self):
        if not self.window or self.window._busy(): return
        if self.window.ui.slicer_plotter is None or not self.window.slicer_parts:
            self.hint.setText('Сначала загрузите деталь.')
            return
        self.workspace.set_mode('part')
        self.window.ui.section_panel.manipulate.setChecked(False)
        self.active = (self.tabs.currentIndex(), self.modes[self.tabs.currentIndex()].currentIndex())
        self.hits = []
        self.hint.setText('Выберите три точки на поверхности; Esc — завершить.' if self.active[0] == 1 or self.active == (2, 0)
                          else 'Выберите первую точку/поверхность, затем вторую. Esc — завершить.')
        self.workspace.plotter.setCursor(Qt.CrossCursor)
        self.workspace.plotter.setFocus()

    def stop(self):
        self.active = None
        self.hits = []
        if self.window and self.window.ui.slicer_plotter:
            self.window.ui.slicer_plotter.remove_actor('measurement_pending')

    def clear(self):
        self.stop()
        plotter = self.window.ui.slicer_plotter if self.window else None
        if plotter:
            for name in list(plotter.actors):
                if name.startswith('measurement_'): plotter.remove_actor(name)
            plotter.render()
        self.names.clear()
        self.results.clear()

    def set_hidden(self, hidden):
        if self.window and self.window.ui.slicer_plotter:
            plotter = self.window.ui.slicer_plotter
            for name, actor in plotter.actors.items():
                if name.startswith('measurement_'): actor.SetVisibility(not hidden)
            plotter.render()

    def pick(self, position):
        hit = self.workspace.picker(position, selected_only=False)
        if hit is None: return
        self.hits.append(hit)
        self.workspace.plotter.add_mesh(pv.PolyData(np.array([item[2] for item in self.hits])), color='#9b2ec8',
                                       point_size=12, render_points_as_spheres=True, name='measurement_pending', pickable=False)
        count = 3 if self.active[0] == 1 or self.active == (2, 0) else 2
        if len(self.hits) < count:
            self.hint.setText(f'Выбрано точек: {len(self.hits)} из {count}. Укажите следующую.')
            return
        try: self.calculate()
        except (ValueError, FloatingPointError) as exc: self.hint.setText(str(exc) + ' Начните выбор заново.')
        self.hits = []
        self.workspace.plotter.remove_actor('measurement_pending')

    def calculate(self):
        points = np.array([hit[2] for hit in self.hits])
        normals = [self.window.slicer_parts[row]['mesh'].face_normals[face] for row, face, _ in self.hits]
        mode = self.active
        geometry = None
        if mode == (0, 0):
            value, delta = distance(*points)
            text = f'{value:.4f} мм; ΔX={delta[0]:.4f}, ΔY={delta[1]:.4f}, ΔZ={delta[2]:.4f}'
        elif mode == (0, 1):
            end, value = project_plane(points[0], points[1], normals[1])
            points[1] = end
            text = f'До плоскости грани: {value:.4f} мм (продолжение плоскости)'
        elif mode == (0, 2):
            data = self.window.slicer_parts[self.hits[1][0]]['mesh_pv']
            _, end = data.find_closest_cell(points[0], return_closest_point=True)
            points[1] = end
            value, _ = distance(*points)
            text = f'До поверхности детали: {value:.4f} мм'
        elif mode == (0, 3):
            degrees = normal_angle(*normals)
            if degrees > 1e-4:
                raise ValueError(f'Плоскости пересекаются (угол {degrees:.4f}°); расстояние между их продолжениями равно 0')
            points[1], value = project_plane(points[0], points[1], normals[1])
            text = f'Между параллельными плоскостями: {value:.4f} мм'
        elif mode[0] == 1:
            center, radius, normal = circle(points)
            u = (points[0] - center) / radius
            v = np.cross(normal, u)
            t = np.linspace(0, 2 * np.pi, 129)
            ring = center + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
            geometry = pv.lines_from_points(ring)
            text = f'R={radius:.4f} мм; Ø={2*radius:.4f} мм; центр ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})'
        elif mode == (2, 0):
            text = f'Угол: {angle(points):.4f}°'
        else:
            text = f'Угол между плоскостями: {normal_angle(*normals):.4f}°'
        if geometry is None: geometry = pv.lines_from_points(points)
        plotter = self.workspace.plotter
        self.serial += 1
        name = f'measurement_{self.serial}'
        plotter.add_mesh(geometry, color='#b34ed6', line_width=3, name=name, pickable=False)
        plotter.add_mesh(pv.PolyData(points), color='#9b2ec8', point_size=10, render_points_as_spheres=True, name=name + '_points', pickable=False)
        if hasattr(plotter, 'add_point_labels'):
            plotter.add_point_labels([points.mean(axis=0)], [text], name=name + '_label', font_size=12,
                                    always_visible=True, point_size=0, shape_opacity=.75)
        self.results.addItem(text)
        self.results.scrollToBottom()
        self.hint.setText('Измерение готово. Можно выбрать следующие точки или нажать Esc.')
        self.set_hidden(self.hidden.isChecked())
