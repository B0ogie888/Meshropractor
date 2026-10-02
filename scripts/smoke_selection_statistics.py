"""Native visible part picking, hardware rectangle selection and statistics HUD."""
from contextlib import ExitStack
from pathlib import Path
import itertools
import json
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')

import numpy as np
import trimesh
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import Meshropractor
import app_updater
from part_supports import make_group
from project_store import ProjectState


def main():
    output = ROOT / 'output/selection-statistics-smoke'; output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle('Fusion')
    report = dict(status='running', checks=[])
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(Meshropractor, 'QSettings', lambda *args: settings))
        for name in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, name, lambda *args: None))
        window = Meshropractor.MainWindow(); window.resize(1600, 1000); window.show()
        try:
            first = trimesh.creation.box([10, 10, 10]); first.apply_translation([-20, 0, 10])
            second = trimesh.creation.box([10, 10, 10]); second.apply_translation([20, 0, 15])
            support = trimesh.creation.box([2, 2, 5]); support.apply_translation([-20, 0, 2.5])
            window.restore_project(ProjectState(page='slicer',
                platforms=[dict(name='Build', dim=[100, 100, 100], is_default=True)],
                parts=[dict(mesh=first, filename='Первая.stl', platform='Build', supports=[make_group(support)]),
                       dict(mesh=second, filename='Вторая.stl', platform='Build')]))
            window.ui.scene_tabs.setCurrentIndex(1)
            plotter = window.ui.slicer_plotter
            plotter.camera_position = [(0, 0, 200), (0, 0, 0), (0, 1, 0)]
            plotter.camera.parallel_projection = True; plotter.camera.parallel_scale = 60
            plotter.reset_camera_clipping_range(); plotter.render(); QTest.qWait(100)
            tools = window.display_tools; tools._density, tools._price = 2., 50.
            for key in ('volume', 'material_cost', 'packing_density'): window.ui.display_buttons[key].click()
            def pixel(point):
                renderer = plotter.renderer; renderer.SetWorldPoint(*point, 1); renderer.WorldToDisplay()
                x, y, _ = renderer.GetDisplayPoint(); ratio = plotter.devicePixelRatioF()
                return QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))
            def click(position, modifiers=Qt.NoModifier):
                QTest.mouseClick(plotter, Qt.LeftButton, modifiers, position); QTest.qWait(100)
            click(pixel(first.centroid))
            assert window.selected_slicer_rows() == [0]
            assert tools.statistics.data['total_mm3'] == 1020.
            assert tools.statistics.data['height_mm'] == 15.
            click(pixel(second.centroid), Qt.ShiftModifier)
            assert window.selected_slicer_rows() == [0, 1]
            click(pixel(first.centroid), Qt.ControlModifier)
            assert window.selected_slicer_rows() == [1]
            report['checks'].append('Native part click, Shift addition and Ctrl toggle synchronize checkboxes and statistics')
            click(QPoint(10, 10))
            assert window.selected_slicer_rows() == []
            assert tools.statistics.data['total_mm3'] == 0.
            report['checks'].append('Empty click clears every part and displays zero statistics')
            points = [pixel(corner) for mesh in (first, second) for corner in itertools.product(*zip(*mesh.bounds))]
            start = QPoint(min(p.x() for p in points) - 15, min(p.y() for p in points) - 15)
            end = QPoint(max(p.x() for p in points) + 15, max(p.y() for p in points) + 15)
            assert window.workspace_tools.picker(start, selected_only=False) is None
            QTest.mousePress(plotter, Qt.LeftButton, Qt.NoModifier, start)
            move = QMouseEvent(QEvent.MouseMove, QPointF(end), QPointF(plotter.mapToGlobal(end)),
                               Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
            app.sendEvent(plotter, move)
            QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, end); QTest.qWait(100)
            assert window.selected_slicer_rows() == [0, 1], window.selected_slicer_rows()
            data = tools.statistics.data
            assert data['total_mm3'] == 2020.
            assert data['height_mm'] == 20.
            assert abs(data['usage_percent'] - .202) < 1e-9
            assert abs(data['packing_percent'] - 1.01) < 1e-9
            assert abs(data['cost'] - .202) < 1e-9
            report['checks'].append('Hardware rectangle selects both visible parts; support volume, build height and density are correct')
            matrix = np.eye(4); matrix[2, 3] = 10.
            window.preview_transforms({0: matrix, 1: matrix}); QTest.qWait(100)
            assert tools.statistics.data['height_mm'] == 30.
            window.preview_transforms({0: np.eye(4), 1: np.eye(4)}); QTest.qWait(100)
            report['checks'].append('Statistics and selection outlines follow transformation previews')
            plotter.screenshot(str(output / 'scene.png'))
            ribbon = window.ui.magics_ribbon
            index = next(i for i in range(ribbon.count()) if ribbon.tabText(i).casefold() == 'отображение')
            ribbon.setCurrentIndex(index)
            panel = ribbon.widget(index)
            panel.horizontalScrollBar().setValue(panel.horizontalScrollBar().maximum())
            QTest.qWait(100)
            panel.grab().save(str(output / 'ribbon.png'))
            window.resize(1300, 850); QTest.qWait(100)
            plotter.screenshot(str(output / 'scene-resized.png'))
            report['status'] = 'ok'
        finally:
            (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            window.dirty = False; window.close(); window.deleteLater(); app.processEvents()


if __name__ == '__main__': main()
