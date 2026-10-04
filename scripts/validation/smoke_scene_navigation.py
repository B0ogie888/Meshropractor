"""Real Qt right-drag gestures and unfilled VTK guides in both workspaces."""
from pathlib import Path
from contextlib import ExitStack
import json
import os
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import main_window
from project_store import ProjectState


def main():
    output = ROOT / 'output/scene-navigation-smoke'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checks=[])
    app = QApplication([])
    app.setStyle('Fusion')
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window, 'QSettings', lambda *args: settings))
        window = main_window.MainWindow()
        window.resize(1600, 1000)
        window.show()
        mesh = trimesh.creation.box([15, 20, 25])
        mesh.apply_translation([0, 0, 22.5])
        try:
            state = ProjectState(page='slicer', platforms=[dict(name='Build', dim=[100, 100, 100], is_default=True)],
                parts=[dict(mesh=mesh, filename='Part.stl', platform='Build')],
                models=[dict(key='CAD_0', kind='CAD', name='CAD', mesh=mesh.copy(), style={})])
            window.restore_project(state)
            window.ui.scene_tabs.setCurrentIndex(1)
            QTest.qWait(100)
            tools = window.workspace_tools
            plotter = window.ui.slicer_plotter
            assert plotter.actors['plat_bounds'].mapper.dataset.n_faces_strict == 0
            tools.set_mode('triangle')
            selected_before = window.selected_slicer_rows()

            def press(plotter, position):
                QTest.mousePress(plotter, Qt.RightButton, Qt.NoModifier, position)
            def move(plotter, position):
                event = QMouseEvent(QEvent.MouseMove, QPointF(position), QPointF(plotter.mapToGlobal(position)),
                                    Qt.NoButton, Qt.RightButton, Qt.NoModifier)
                app.sendEvent(plotter, event)
            def release(plotter, position):
                QTest.mouseRelease(plotter, Qt.RightButton, Qt.NoModifier, position)
                QTest.qWait(30)

            for name, current in [('slicer', plotter), ('predeformation', window.ui.plotter)]:
                window.ui.stack.setCurrentWidget(window.ui.page_slicer if name == 'slicer' else window.ui.page_predef)
                QTest.qWait(100)
                current.camera_position = [(0, 0, 200), (0, 0, 0), (0, 1, 0)]
                current.camera.parallel_projection = True
                current.camera.parallel_scale = 80
                current.render()
                navigation = current._scene_navigation
                center, radius = navigation.circle_geometry()
                inside = center.toPoint()
                before_camera = list(current.camera_position)
                QTest.mousePress(current, Qt.LeftButton, Qt.AltModifier, inside)
                event = QMouseEvent(QEvent.MouseMove,QPointF(inside+QPoint(40,20)),
                    QPointF(current.mapToGlobal(inside+QPoint(40,20))),Qt.NoButton,Qt.LeftButton,Qt.AltModifier)
                app.sendEvent(current,event)
                QTest.mouseRelease(current,Qt.LeftButton,Qt.AltModifier,inside+QPoint(40,20))
                QTest.mouseMove(current,inside+QPoint(70,40))
                np.testing.assert_allclose(list(current.camera_position),before_camera)
                assert current.iren.interactor.GetInteractorStyle().GetState()==0
                before, focus = np.array(current.camera.position), current.camera.focal_point
                press(current, inside)
                assert navigation.guide.actor.GetVisibility()
                current.screenshot(str(output / (name + '-circle.png')))
                move(current, inside + QPoint(70, 45))
                release(current, inside + QPoint(70, 45))
                assert not np.allclose(current.camera.position, before)
                assert current.camera.focal_point == focus
                assert not navigation.guide.actor.GetVisibility()
                assert not tools.selection
                assert window.selected_slicer_rows() == selected_before
                current.camera_position = [(0, 0, 200), (0, 0, 0), (0, 1, 0)]
                r = radius + 45
                start = QPoint(round(center.x() + r), round(center.y()))
                end = QPoint(round(center.x() + r * .866), round(center.y() + r * .5))
                position, focus = current.camera.position, current.camera.focal_point
                press(current, start)
                move(current, end)
                release(current, end)
                assert current.camera.position == position and current.camera.focal_point == focus
                assert current.camera.up[0] < -.4
                assert tools.menu is None or not tools.menu.isVisible()
                current.screenshot(str(output / (name + '-roll.png')))
                report['checks'].append(name + ': right drag orbits inside and rolls outside without Alt or changing selection')
                cube = tools.cube if name=='slicer' else window.ui.def_cube
                point = cube.geometry().center()
                assert cube.face_at(point-cube.geometry().topLeft()) is not None
                before_camera = list(current.camera_position)
                QTest.mousePress(current,Qt.LeftButton,Qt.NoModifier,point)
                event = QMouseEvent(QEvent.MouseMove,QPointF(point+QPoint(30,15)),
                    QPointF(current.mapToGlobal(point+QPoint(30,15))),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
                app.sendEvent(current,event)
                QTest.mouseRelease(current,Qt.LeftButton,Qt.NoModifier,point+QPoint(30,15))
                assert not np.allclose(list(current.camera_position),before_camera)
                report['checks'].append(name + ': left scene drag cannot rotate or stick; left cube drag still rotates')

            window.ui.stack.setCurrentWidget(window.ui.page_slicer)
            QTest.qWait(100)
            rectangle = tools.rubber
            plotter.render()
            before = plotter.screenshot()
            rectangle.setGeometry(QRect(70, 70, 170, 140))
            rectangle.show()
            after = plotter.screenshot(str(output / 'unfilled-rectangle.png'))
            ratio = plotter.devicePixelRatioF()
            # Interior pixels are identical: even a translucent fill would fail.
            np.testing.assert_array_equal(before[int(85*ratio):int(195*ratio), int(85*ratio):int(225*ratio)],
                                          after[int(85*ratio):int(195*ratio), int(85*ratio):int(225*ratio)])
            assert np.any(before != after)
            rectangle.hide()
            report['checks'].append('Selection rectangle changes only its outline; build bounds contain no polygons')
            tools.set_mode('part')
            press(plotter, QPoint(plotter.width() // 2, plotter.height() // 2))
            release(plotter, QPoint(plotter.width() // 2, plotter.height() // 2))
            assert tools.menu is not None and tools.menu.isVisible()
            tools.menu.close()
            report['checks'].append('Plain right click still opens the radial menu')
            report['status'] = 'passed'
        except Exception:
            import traceback
            report['error'] = traceback.format_exc()
            report['status'] = 'failed'
            raise
        finally:
            (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            window.dirty = False
            window.close()
            app.processEvents()


if __name__ == '__main__': main()
