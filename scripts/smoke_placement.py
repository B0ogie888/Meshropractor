"""Native Windows placement regression with isolated preferences and test models.

    python scripts/smoke_placement.py --output output/placement-smoke

Exercises real Qt mouse events, VTK picking, geometry workers and undo/redo.
Starts and closes only its own window; does not build EXEs or check for updates.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def run_native(output):
    import numpy as np
    import trimesh
    from PySide6.QtCore import QPoint, QSettings, QTimer, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QScrollArea
    sys.path.insert(0, str(ROOT / 'src'))
    import app_updater
    import Meshropractor
    from part_supports import make_group
    from placement_ribbon import PLACEMENT_COMMANDS
    from project_history import digest
    from project_store import ProjectState

    app = QApplication([])
    if app.platformName() != 'windows':
        raise RuntimeError('This check requires native Windows Qt/OpenGL.')
    app.setStyle('Fusion')
    report = dict(status='running', checks=[], screenshots=[], settings='temporary', updates='disabled')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='meshropractor-placement-smoke-') as folder, ExitStack() as patches:
        QSettings.setDefaultFormat(QSettings.IniFormat)
        for scope in (QSettings.UserScope, QSettings.SystemScope):
            QSettings.setPath(QSettings.IniFormat, scope, folder)
        settings = QSettings(str(Path(folder) / 'smoke.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(Meshropractor, 'QSettings', new=lambda *args: settings))

        def disabled_update(self, *args, **kwargs):
            pass

        for name in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, name, new=disabled_update))
        window = Meshropractor.MainWindow()
        window.resize(1600, 1000)
        window.show(); window.raise_(); window.activateWindow()
        platform = dict(id='smoke-platform', name='Smoke platform', dim=[60, 50, 40], is_default=True, use_zones=False, zones=[])

        def wait_until(predicate, message, timeout=35):
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() >= deadline:
                    session = getattr(window, '_placement_session', None)
                    detail = session.dialog.status.text() + '\n' + session.dialog.report.toPlainText() if session else ''
                    raise AssertionError(message + '\n' + detail)
                QTest.qWait(10)
            app.processEvents()

        def box(center, dimensions):
            mesh = trimesh.creation.box(extents=dimensions)
            mesh.apply_translation(center)
            return mesh

        def snapshot():
            return [dict(mesh=part['mesh'].copy(), supports=deepcopy(part.get('supports', [])),
                         platform=part.get('platform'), filename=part['filename']) for part in window.slicer_parts]

        def same_mesh(left, right):
            np.testing.assert_array_equal(left.vertices, right.vertices)
            np.testing.assert_array_equal(left.faces, right.faces)
            assert digest(left.metadata) == digest(right.metadata)

        def assert_snapshot(saved):
            assert len(saved) == len(window.slicer_parts)
            for part, original in zip(window.slicer_parts, saved):
                same_mesh(part['mesh'], original['mesh'])
                assert digest(part.get('supports', [])) == digest(original['supports'])
                assert part.get('platform') == original['platform']
                assert part['filename'] == original['filename']

        def fixture(two_selected=False, support=False, tilt=False):
            first = box([-10, 0, 13], [18, 11, 7])
            if tilt:
                matrix = trimesh.transformations.euler_matrix(.4, -.6, .2)
                matrix[:3, 3] = first.bounds.mean(axis=0) - matrix[:3, :3] @ first.bounds.mean(axis=0)
                first.apply_transform(matrix)
            children = [make_group(box([-10, 0, 5], [2, 2, 8]), [0])] if support else []
            parts = [dict(mesh=first, filename='Selected.stl', supports=children,
                          style=dict(is_selected=True, is_visible=True, color='#63b7db', last_visible_mode='shaded_wire')),
                     dict(mesh=box([18, 0, 13], [10, 8, 8]), filename='Second.stl',
                          style=dict(is_selected=two_selected, is_visible=True, color='#c66b43', last_visible_mode='shaded')),
                     dict(mesh=box([22, 17, 2], [4, 4, 4]), filename='Hidden.stl', platform=platform['name'],
                          style=dict(is_selected=False, is_visible=False, color='#7d868d', last_visible_mode='shaded'))]
            window.restore_project(ProjectState(parts=parts, platforms=[deepcopy(platform)], page='slicer'))
            window.ui.scene_tabs.setCurrentIndex(0)
            index = next(i for i in range(window.ui.magics_ribbon.count())
                         if window.ui.magics_ribbon.tabText(i).strip().casefold() == 'расположение')
            window.ui.magics_ribbon.setCurrentIndex(index)
            plotter = window.ui.slicer_plotter
            plotter.camera_position = [(40, -65, 52), (0, 0, 10), (0, 0, 1)]
            plotter.reset_camera(); plotter.render(); app.processEvents()
            window.dirty = False
            window.reset_history()
            return snapshot()

        def actor_matrix(row):
            matrix = window.ui.slicer_plotter.actors[window.slicer_parts[row]['actor_name']].GetUserMatrix()
            return np.eye(4) if matrix is None else np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])

        def open_tool(operation):
            button = window.ui.placement_buttons[operation]
            parent = button.parentWidget()
            while parent is not None:
                if isinstance(parent, QScrollArea):
                    parent.ensureWidgetVisible(button); break
                parent = parent.parentWidget()
            app.processEvents()
            assert button.isEnabled(), operation + ' is disabled'
            QTest.mouseClick(button, Qt.LeftButton)
            wait_until(lambda: getattr(window, '_placement_session', None) is not None, operation + ' did not open')
            return window._placement_session

        def prepared(session):
            wait_until(lambda: window._job is None and not session.dialog.running, 'Worker did not finish')
            wait_until(lambda: session.result is not None, 'Worker returned no result', timeout=2)
            assert session.dialog.apply.isEnabled()
            assert window.history.index == 0, 'Preview created a history entry'

        def assert_preview(saved, matrices):
            assert_snapshot(saved)
            for row, matrix in matrices.items():
                np.testing.assert_allclose(actor_matrix(row), matrix, atol=1e-10)

        def apply_history(session, saved):
            matrices = {row: matrix.copy() for row, matrix in session.result['matrices'].items()}
            target = session.result.get('platform')
            QTest.mouseClick(session.dialog.apply, Qt.LeftButton)
            wait_until(lambda: window._placement_session is None, 'Apply did not close')
            assert window.history.index == 1 and len(window.history.entries) == 2, 'Apply must create one undo step'
            for row, (part, original) in enumerate(zip(window.slicer_parts, saved)):
                matrix = matrices.get(row, np.eye(4))
                np.testing.assert_allclose(part['mesh'].vertices,
                    trimesh.transform_points(original['mesh'].vertices, matrix), atol=1e-8)
                np.testing.assert_array_equal(part['mesh'].faces, original['mesh'].faces)
                assert part.get('platform') == (target['name'] if target and row in matrices else original['platform'])
                for group, old in zip(part.get('supports', []), original['supports']):
                    np.testing.assert_allclose(group['vertices'], trimesh.transform_points(old['vertices'], matrix), atol=1e-8)
                    assert group['id'] == old['id'] and group['surface_faces'] == old['surface_faces']
            result = snapshot()
            window.ui.action_undo.trigger()
            assert window.history.index == 0
            assert_snapshot(saved)
            window.ui.action_redo.trigger()
            assert window.history.index == 1
            assert_snapshot(result)

        def capture(widget, name):
            path = output / name
            assert widget.grab().save(str(path))
            report['screenshots'].append(str(path))

        def scene(name):
            path = output / name
            picture = window.ui.slicer_plotter.screenshot(str(path), return_img=True)
            assert picture.shape[0] > 100 and np.ptp(picture) > 50
            report['screenshots'].append(str(path))

        def visible_face():
            mesh = window.slicer_parts[0]['mesh']
            plotter = window.ui.slicer_plotter
            renderer, ratio = plotter.renderer, plotter.devicePixelRatioF()
            matrix = actor_matrix(0)
            centers = trimesh.transform_points(mesh.triangles_center, matrix)
            normals = mesh.face_normals @ matrix[:3, :3].T
            facing = np.einsum('ij,ij->i', normals, np.asarray(plotter.camera.position) - centers)
            for face in np.argsort(facing)[::-1]:
                if facing[face] <= 0: continue
                point = centers[face]
                renderer.SetWorldPoint(*point, 1.); renderer.WorldToDisplay()
                x, y, _ = renderer.GetDisplayPoint()
                pixel = QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))
                hit = window.workspace_tools.picker(pixel, rows=[0])
                if hit and hit[1] == face: return pixel, int(face)
            raise AssertionError('No visible triangle available for native picking')

        def check():
            try:
                saved = fixture(support=True)
                assert set(window.ui.placement_buttons) == set(PLACEMENT_COMMANDS)
                assert len(window.ui.placement_buttons) == 13
                for button in window.ui.placement_buttons.values():
                    assert button.isEnabled() and not button.icon().isNull()
                capture(window.ui.magics_ribbon, 'placement-ribbon.png')
                report['checks'].append('13 ribbon actions are active and have icons')
                session = open_tool('free_move')
                pixel, _ = visible_face()
                plotter = window.ui.slicer_plotter
                QTest.mousePress(plotter, Qt.LeftButton, Qt.NoModifier, pixel)
                assert session.drag is not None, 'Native press did not start free drag'
                destination = pixel + QPoint(30, -18)
                QTest.mouseMove(plotter, destination)
                assert np.linalg.norm(session.result['matrices'][0][:3, 3]) > .1
                QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, destination)
                assert session.drag is None
                assert_preview(saved, session.result['matrices'])
                child = plotter.actors['part_support_' + saved[0]['supports'][0]['id']]
                np.testing.assert_allclose(np.array([[child.GetUserMatrix().GetElement(i, j) for j in range(4)] for i in range(4)]), actor_matrix(0))
                capture(session.dialog, 'free-move-dialog.png'); scene('free-move-preview.png')
                apply_history(session, saved)
                report['checks'].append('Native free dragging: source unchanged, support preview, apply, undo/redo')

                saved = fixture()
                session = open_tool('free_move')
                session.dialog.delta[2].setValue(1.25)
                session.dialog.plane.setCurrentIndex(1)
                session.dialog.fields['snap_mm'].setValue(.5)
                pixel, _ = visible_face()
                QTest.mousePress(plotter, Qt.LeftButton, Qt.NoModifier, pixel)
                assert session.drag is not None
                destination = pixel + QPoint(37, -16)
                QTest.mouseMove(plotter, destination)
                QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, destination)
                delta = np.asarray(session.dialog.parameters()['delta'])
                assert abs(delta[2] - 1.25) < 1e-10, 'XY snapping changed the locked Z offset'
                assert np.linalg.norm(delta[:2]) > .2
                np.testing.assert_allclose(delta[:2] / .5, np.round(delta[:2] / .5), atol=1e-10)
                assert_preview(saved, session.result['matrices'])
                capture(session.dialog, 'xy-snap-dialog.png')
                report['checks'].append('Native XY drag with 0.5 mm snapping preserves the existing Z=1.25 offset')
                session.dialog.delta[0].setValue(5.)
                assert_preview(saved, session.result['matrices'])
                QTest.mouseClick(session.dialog.close_button, Qt.LeftButton)
                wait_until(lambda: window._placement_session is None, 'Cancel did not close')
                assert_snapshot(saved)
                np.testing.assert_array_equal(actor_matrix(0), np.eye(4))
                assert window.history.index == 0
                report['checks'].append('Cancel removes free-move preview without changing geometry or history')

                for side in (0, 1):
                    saved = fixture(tilt=True)
                    session = open_tool('top_bottom')
                    session.dialog.side.setCurrentIndex(side)
                    QTest.mouseClick(session.dialog.pick, Qt.LeftButton)
                    pixel, face = visible_face()
                    QTest.mouseClick(window.ui.slicer_plotter, Qt.LeftButton, Qt.NoModifier, pixel)
                    prepared(session)
                    assert session.surface == face
                    matrix = session.result['matrices'][0]
                    np.testing.assert_allclose(matrix[:3, :3] @ saved[0]['mesh'].face_normals[face],
                                               [0, 0, -1 if side == 0 else 1], atol=1e-8)
                    expected = trimesh.transform_points(saved[0]['mesh'].vertices, matrix)
                    assert abs(expected[:, 2].min()) < 1e-8
                    assert_preview(saved, session.result['matrices'])
                    capture(session.dialog, f'surface-{side}-dialog.png'); scene(f'surface-{side}-preview.png')
                    apply_history(session, saved)
                report['checks'].append('Native surface picks orient the selected normal to both -Z/+Z and place the part on Z=0')

                saved = fixture(two_selected=True)
                session = open_tool('auto_arrange')
                QTest.mouseClick(session.dialog.prepare, Qt.LeftButton)
                prepared(session)
                assert_preview(saved, session.result['matrices'])
                for row, matrix in session.result['matrices'].items():
                    points = trimesh.transform_points(saved[row]['mesh'].vertices, matrix)
                    assert np.all(points.min(axis=0) >= np.array([-28, -23, 0]) - 1e-8)
                    assert np.all(points.max(axis=0) <= np.array([28, 23, 40]) + 1e-8)
                capture(session.dialog, 'auto-arrange-dialog.png')
                window.ui.slicer_plotter.reset_camera(); window.ui.slicer_plotter.render()
                scene('auto-arrange-preview.png')
                apply_history(session, saved)
                report['checks'].append('Async arrangement: both parts within the platform, hidden obstacle unchanged, one undo step')

                fixture(support=True)
                state = window.capture_project()
                other_platform = dict(platform, id='smoke-platform-b', name='Smoke platform B', dim=[70, 55, 40])
                state.platforms.append(other_platform)
                state.parts[0]['platform'] = platform['name']
                state.parts[1]['platform'] = other_platform['name']
                state.parts[1]['supports'] = [make_group(box([18, 0, 5], [2, 2, 8]), [0])]
                state.parts[2]['platform'] = platform['name']
                state.parts[2]['style']['is_visible'] = True
                window.restore_project(state)
                window.ui.scene_tabs.setCurrentIndex(1)
                window.reset_history()
                saved = snapshot()

                def assert_visibility(expected):
                    for row, visible in enumerate(expected):
                        part = window.slicer_parts[row]
                        actual = bool(window.ui.slicer_plotter.actors[part['actor_name']].GetVisibility())
                        assert actual == visible, f"Part {row}: visibility {actual}, expected {visible}; scene {window.ui.scene_tabs.currentIndex()}"
                        for group in part.get('supports', []):
                            assert bool(window.ui.slicer_plotter.actors['part_support_' + group['id']].GetVisibility()) == visible

                assert_visibility([True, False, True])
                session = open_tool('auto_arrange')
                session.dialog.platform.setCurrentIndex(1)
                QTest.mouseClick(session.dialog.prepare, Qt.LeftButton)
                prepared(session)
                assert_visibility([True, True, False])
                assert window.ui.scene_tabs.currentIndex() == 1, 'Preview switched away from the original platform tab'
                assert_preview(saved, session.result['matrices'])
                window.ui.slicer_plotter.reset_camera(); window.ui.slicer_plotter.render()
                scene('cross-platform-preview.png')
                QTest.mouseClick(session.dialog.preview, Qt.LeftButton, Qt.NoModifier, QPoint(8, session.dialog.preview.height() // 2))
                assert not session.dialog.preview.isChecked(), 'Native preview checkbox click was not accepted'
                assert_visibility([True, False, True])
                np.testing.assert_array_equal(actor_matrix(0), np.eye(4))
                QTest.mouseClick(session.dialog.preview, Qt.LeftButton, Qt.NoModifier, QPoint(8, session.dialog.preview.height() // 2))
                assert session.dialog.preview.isChecked(), 'Native preview checkbox click was not accepted'
                assert_visibility([True, True, False])
                assert_preview(saved, session.result['matrices'])
                QTest.mouseClick(session.dialog.close_button, Qt.LeftButton)
                wait_until(lambda: window._placement_session is None, 'Cross-platform cancel did not close')
                assert_visibility([True, False, True])
                assert_snapshot(saved)
                assert window.history.index == 0 and window.ui.scene_tabs.currentIndex() == 1
                np.testing.assert_array_equal(actor_matrix(0), np.eye(4))
                report['checks'].append('A-to-B preview shows target parts/supports and hides unrelated A parts; preview off and cancel restore A without assignment changes')

                saved = fixture(tilt=True)
                session = open_tool('compare_orientations')
                QTest.mouseClick(session.dialog.prepare, Qt.LeftButton)
                prepared(session)
                assert session.dialog.table.rowCount() > 1
                first = actor_matrix(0).copy()
                variants = session.result['variants']
                different = [i for i, entry in enumerate(variants) if not np.allclose(entry['matrix'], first)]
                variant = next((i for i in different if abs(variants[i]['height_mm'] - variants[0]['height_mm']) > .01), different[0])
                table = session.dialog.table
                item = table.item(variant, 0)
                table.scrollToItem(item); app.processEvents()
                QTest.mouseClick(table.viewport(), Qt.LeftButton, Qt.NoModifier, table.visualItemRect(item).center())
                np.testing.assert_allclose(actor_matrix(0), session.result['variants'][variant]['matrix'])
                assert not np.allclose(actor_matrix(0), first), 'Table selection did not update VTK preview'
                assert_preview(saved, session.result['matrices'])
                capture(session.dialog, 'compare-dialog.png'); scene('compare-preview.png')
                apply_history(session, saved)
                report['checks'].append('Orientation table native click changes VTK UserMatrix and applies the selected candidate')
                report['status'] = 'ok'
            except Exception:
                report['status'] = 'error'; report['error'] = traceback.format_exc()
                print(report['error'], flush=True)
            finally:
                try:
                    if window._job is not None:
                        window.cancel_current_job()
                        wait_until(lambda: window._job is None, 'Worker cancellation did not finish')
                    session = getattr(window, '_placement_session', None)
                    if session is not None: session.dialog.reject()
                    window.dirty = False; window.close()
                except Exception:
                    report['cleanup_error'] = traceback.format_exc(); report['status'] = 'error'
                (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                app.quit()
        QTimer.singleShot(800, check)
        app.exec()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/placement-smoke'))
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args._worker: return 0 if run_native(output)['status'] == 'ok' else 1
    if os.name != 'nt': raise SystemExit('This check requires Windows.')
    # Mark the run before spawning so a native crash cannot leave a stale success.
    (output / 'report.json').write_text('{"status":"running"}', encoding='utf-8')
    try:
        result = subprocess.run([sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()),
            '--_worker', '--output', str(output)], cwd=ROOT, capture_output=True, text=True,
            encoding='utf-8', errors='replace', env={**os.environ, 'PYTHONUTF8': '1'}, timeout=180)
        (output / 'native.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        print(json.dumps(dict(status='ok' if result.returncode == 0 else 'error', exit_code=result.returncode,
                             report=str(output / 'report.json'), log=str(output / 'native.log'))))
        return bool(result.returncode)
    except subprocess.TimeoutExpired:
        failure = dict(status='error', error='Our diagnostic child exceeded 180 seconds and was stopped.')
        (output / 'report.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        print(json.dumps(failure)); return 1


if __name__ == '__main__':
    raise SystemExit(main())
