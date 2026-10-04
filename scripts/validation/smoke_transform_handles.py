"""Native Qt/VTK handle drags, constraints, camera behavior and both themes."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QSettings, QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
import main_window, app_updater
from project_store import ProjectState


def mouse(plotter, kind, point, button=Qt.NoButton, buttons=Qt.NoButton):
    local = QPointF(*point)
    global_point = QPointF(plotter.mapToGlobal(local.toPoint()))
    event = QMouseEvent(kind, local, global_point, button, buttons, Qt.NoModifier)
    QApplication.sendEvent(plotter, event)


def drag(plotter, start, end):
    mouse(plotter, QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton)
    for t in np.linspace(.1, 1, 10):
        mouse(plotter, QEvent.MouseMove, start + t * (end - start), buttons=Qt.LeftButton)
    mouse(plotter, QEvent.MouseButtonRelease, end, Qt.LeftButton)


def selectable(gizmo, name):
    if name.startswith('plane_'):
        a, b = gizmo.planes[name][:2]
        candidates = [gizmo.origin + gizmo.size * (u * gizmo.axes[a] + v * gizmo.axes[b])
                      for u in np.linspace(.15, .75, 13) for v in np.linspace(.15, .75, 13)]
    else: candidates = gizmo.contours[name][0][1:-1]
    for world in candidates:
        point = gizmo.project([world])[0]
        if not (0 <= point[0] < gizmo.plotter.width() and 0 <= point[1] < gizmo.plotter.height()): continue
        hit = gizmo.hit(point)
        if hit and hit[0] == name: return world, point
    raise AssertionError('No hittable region: ' + name)


def main():
    output = ROOT / 'output/transform-handles-smoke'; output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle('Fusion'); report = dict(status='running', checks=[])
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat); settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window, 'QSettings', lambda *args: settings))
        for method in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, method, lambda *args: None))
        window = main_window.MainWindow(); window.resize(1450, 950); window.show()
        try:
            source = trimesh.creation.box([10, 8, 6]); source.apply_translation([0, 0, 3])
            window.restore_project(ProjectState(page='slicer', platforms=[dict(name='Build', dim=[100,100,100], is_default=True)],
                parts=[dict(mesh=source, filename='box.stl', platform='Build', style=dict(is_selected=True))]))
            window.ui.scene_tabs.setCurrentIndex(1); window.dirty=False; window.reset_history()
            plotter = window.ui.slicer_plotter
            plotter.camera_position = [(40,40,32),(0,0,3),(0,0,1)]
            plotter.reset_camera_clipping_range()
            QTest.qWait(100)
            vertices = source.vertices.copy(); camera = np.array(plotter.camera.position)
            window.run_slicer_tool('Перемещать'); session = window._transform_session; dialog = session.dialog
            dialog.snap.setChecked(False); session.update_preview()
            layers = session.gizmo.layer
            assert len(session.gizmo.planes) == 3
            for name,(a,b,normal,points) in session.gizmo.planes.items():
                np.testing.assert_allclose(points[0],session.gizmo.origin)
                coordinates=(points-session.gizmo.origin)@session.gizmo.axes[[a,b]].T/session.gizmo.size
                np.testing.assert_allclose(coordinates.max(axis=0),[.84,.84])
            assert all(0 < session.gizmo.actors[name][0].prop.opacity < 1 for name in session.gizmo.planes)
            assert all(a.mapper.dataset.n_verts == 0 for key, actors in session.gizmo.actors.items()
                       if key.endswith('_edge') or key.startswith('axis_') and not key.endswith('_tip') for a in actors)
            for mode in ('light', 'dark'):
                window.engineering_theme.set_mode(mode); QTest.qWait(80)
                session.update_preview(); plotter.screenshot(str(output / f'move-scene-{mode}.png'))
            for name in ('plane_xy', 'plane_xz', 'plane_yz'):
                dialog.set_values(dialog.values, [0,0,0]); dialog.sync_move(False); session.update_preview()
                gizmo = session.gizmo
                world, start = selectable(gizmo, name)
                a, b = gizmo.planes[name][:2]
                delta = 2 * gizmo.axes[a] + 3 * gizmo.axes[b]
                drag(plotter, start, gizmo.project([world + delta])[0])
                assert not gizmo.dragging
                np.testing.assert_allclose(dialog.numbers(dialog.values), delta, atol=.01)
                np.testing.assert_allclose(window.slicer_parts[0]['mesh'].vertices, vertices)
            dialog.set_values(dialog.values, [0,0,0]); dialog.sync_move(False); dialog.snap.setChecked(True); dialog.snap_step.setValue(1)
            session.update_preview(); gizmo=session.gizmo
            world = gizmo.origin + .7 * gizmo.size * gizmo.axes[0]; start = gizmo.project([world])[0]
            assert gizmo.hit(start)[0] == 'axis_x'
            drag(plotter, start, gizmo.project([world + [2.7,0,0]])[0])
            np.testing.assert_allclose(dialog.numbers(dialog.values), [3,0,0], atol=.001)
            world,start=selectable(gizmo,'plane_xy')
            mouse(plotter,QEvent.MouseButtonPress,start,Qt.LeftButton,Qt.LeftButton)
            mouse(plotter,QEvent.MouseMove,gizmo.project([world+[2,3,0]])[0],buttons=Qt.LeftButton)
            QTest.keyClick(plotter,Qt.Key_Escape)
            assert not gizmo.dragging
            np.testing.assert_allclose(dialog.numbers(dialog.values),[3,0,0],atol=.001)
            np.testing.assert_allclose(plotter.camera.position, camera)
            session.apply(True); QTest.qWait(50)
            assert plotter.render_window.GetNumberOfLayers() == layers
            np.testing.assert_allclose(window.slicer_parts[0]['mesh'].vertices, vertices + [3,0,0])
            window.travel_history(-1)
            np.testing.assert_allclose(window.slicer_parts[0]['mesh'].vertices, vertices)
            report['checks'].append('Real left-button drags on XY/XZ/YZ constrain the omitted axis; arrow snap, live preview, Apply and Undo work without camera movement')
            print('PLANAR_MOVE_OK', flush=True)

            window.run_slicer_tool('Вращать'); session=window._transform_session; dialog=session.dialog
            dialog.snap.setChecked(False); session.update_preview(); QTest.qWait(60)
            assert dialog.values[0].mapTo(dialog, dialog.values[0].rect().topLeft()).x() < 85
            assert dialog.center[0].mapTo(dialog, dialog.center[0].rect().topLeft()).x() < 95
            for mode in ('light','dark'):
                window.engineering_theme.set_mode(mode); QTest.qWait(80); session.update_preview()
                dialog.grab().save(str(output / f'rotation-dialog-{mode}.png'))
                plotter.screenshot(str(output / f'rotation-scene-{mode}.png'))
            for name in ('ring_x','ring_y','ring_z','ring_screen'):
                dialog.set_values(dialog.values,[0,0,0]); session.update_preview()
                gizmo=session.gizmo; world,start=selectable(gizmo,name); normal=gizmo.contours[name][1]
                assert gizmo.actors[name][0].mapper.dataset.n_verts == 0
                expected=Rotation.from_rotvec(normal*np.deg2rad(30)).as_matrix()
                end=gizmo.project([gizmo.origin+expected@(world-gizmo.origin)])[0]
                drag(plotter,start,end)
                actual=Rotation.from_euler('xyz',dialog.numbers(dialog.values),degrees=True).as_matrix()
                np.testing.assert_allclose(actual,expected,atol=2e-5)
                np.testing.assert_allclose(window.slicer_parts[0]['mesh'].vertices,vertices)
            dialog.set_values(dialog.values,[0,0,0]); dialog.snap.setChecked(True); dialog.snap_step.setValue(45); session.update_preview()
            gizmo=session.gizmo; world,start=selectable(gizmo,'ring_screen'); normal=gizmo.contours['ring_screen'][1]
            rot=Rotation.from_rotvec(normal*np.deg2rad(32)).as_matrix()
            drag(plotter,start,gizmo.project([gizmo.origin+rot@(world-gizmo.origin)])[0])
            actual=Rotation.from_euler('xyz',dialog.numbers(dialog.values),degrees=True).as_matrix()
            np.testing.assert_allclose(actual,Rotation.from_rotvec(normal*np.pi/4).as_matrix(),atol=2e-5)
            # Edge-on axis ring remains draggable in a top view.
            dialog.snap.setChecked(False); dialog.set_values(dialog.values,[0,0,0]); plotter.view_xy(); session.update_preview()
            gizmo=session.gizmo; world,start=selectable(gizmo,'ring_x')
            mouse(plotter,QEvent.MouseButtonPress,start,Qt.LeftButton,Qt.LeftButton)
            assert gizmo.dragging
            tangent=gizmo.drag['tangent']; end=start+.35*tangent
            mouse(plotter,QEvent.MouseMove,end,buttons=Qt.LeftButton)
            mouse(plotter,QEvent.MouseButtonRelease,end,Qt.LeftButton)
            assert np.linalg.norm(dialog.numbers(dialog.values))>1 and not gizmo.dragging
            # Line mode offers only the configured axis, with no unrelated screen ring.
            dialog.accept_points('line',[[0,0,0],[1,2,3]]); session.update_preview()
            assert set(session.gizmo.contours)=={'ring_z'}
            dialog.reject(); QTest.qWait(50)
            assert not any(name.startswith('transform_gizmo_') for name in plotter.actors)
            assert window._transform_session is None
            assert plotter.render_window.GetNumberOfLayers() == layers
            report['checks'].append('X/Y/Z and camera-normal rings rotate via real mouse events; screen-axis 45° snapping, top-view edge-on drag, line mode, cancel cleanup, close fields and both themes pass')
            print('ROTATION_RINGS_OK',flush=True)
            report['status']='passed'
        except Exception:
            import traceback
            report.update(status='failed',error=traceback.format_exc()); raise
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            window.dirty=False; window.close(); window.deleteLater(); app.processEvents()
    print('TRANSFORM_HANDLES_SMOKE_OK',flush=True)


if __name__=='__main__':
    import faulthandler
    faulthandler.dump_traceback_later(60,repeat=True)
    main()
