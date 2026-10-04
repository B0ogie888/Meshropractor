"""Native Windows regression for repair previews, history and mouse editing.

    python scripts/validation/smoke_repair_tools.py --output output/repair-tools-smoke

Opens its own window with synthetic models and temporary preferences. Geometry,
workers, rendering and picking are real. No EXE is built or update requested.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def run_native(output):
    import numpy as np
    import trimesh
    from PySide6.QtCore import QPoint, QSettings, QTimer, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QScrollArea

    sys.path.insert(0, str(ROOT / 'src'))
    import app_updater
    import main_window
    from project_history import digest
    from project_store import ProjectState

    app = QApplication([])
    if app.platformName() != 'windows':
        raise RuntimeError('The smoke check requires native Windows Qt/OpenGL.')
    app.setStyle('Fusion')
    report = dict(status='running', checks=[], screenshots=[], settings='temporary', updates='disabled')
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='meshropractor-repair-smoke-') as folder, ExitStack() as patches:
        QSettings.setDefaultFormat(QSettings.IniFormat)
        for scope in (QSettings.UserScope, QSettings.SystemScope):
            QSettings.setPath(QSettings.IniFormat, scope, folder)
        settings = QSettings(str(Path(folder) / 'smoke.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window, 'QSettings', new=lambda *args: settings))

        def disabled_update(self, *args, **kwargs):
            pass

        for name in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, name, new=disabled_update))
        window = main_window.MainWindow()
        window.resize(1600, 1000)
        window.show()
        window.raise_()
        window.activateWindow()

        def wait_until(predicate, message, timeout=25):
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() >= deadline:
                    raise AssertionError(message + ': ' + window.ui.status_label.text())
                QTest.qWait(10)
            app.processEvents()

        def same_mesh(left, right):
            np.testing.assert_array_equal(left.vertices, right.vertices)
            np.testing.assert_array_equal(left.faces, right.faces)
            assert digest(left.metadata) == digest(right.metadata)

        def box(center=(0, 0, 10), size=10):
            mesh = trimesh.creation.box(extents=[size] * 3)
            mesh.apply_translation(center)
            return mesh

        def fixture(first=None, second=None, all_selected=False, third=None):
            first = first if first is not None else box()
            second = second if second is not None else box((20, 0, 10), 8)
            parts = [dict(mesh=first, filename='Selected.stl', style=dict(is_selected=True, color='#d3d3d3', last_visible_mode='shaded_wire')),
                     dict(mesh=second, filename='Untouched.stl', style=dict(is_selected=all_selected, color='#c66b43', last_visible_mode='shaded'))]
            if third is not None:
                parts[1]['filename'] = 'Second selected.stl'
                parts.append(dict(mesh=third, filename='Untouched.stl', style=dict(is_selected=False, color='#c66b43', last_visible_mode='shaded')))
            window.restore_project(ProjectState(parts=parts, page='slicer'))
            window.dirty = False
            window.reset_history()
            index = next(i for i in range(window.ui.magics_ribbon.count())
                         if window.ui.magics_ribbon.tabText(i).strip().casefold() == 'исправление')
            window.ui.magics_ribbon.setCurrentIndex(index)
            plotter = window.ui.slicer_plotter
            plotter.camera_position = [(30, -45, 42), (7, 0, 10), (0, 0, 1)]
            plotter.reset_camera()
            plotter.render()
            app.processEvents()
            return first, second

        def open_tool(operation):
            button = window.ui.repair_buttons[operation]
            parent = button.parentWidget()
            while parent is not None:
                if isinstance(parent, QScrollArea):
                    parent.ensureWidgetVisible(button)
                    break
                parent = parent.parentWidget()
            app.processEvents()
            assert button.isEnabled(), operation + ' ribbon button is disabled'
            QTest.mouseClick(button, Qt.LeftButton)
            wait_until(lambda: getattr(window, '_repair_session', None) is not None, operation + ' did not open')
            session = window._repair_session
            assert session.operation == operation
            return session

        def prepare(session):
            before = [(record['mesh'], record['mesh'].copy()) for record in session.records]
            QTest.mouseClick(session.dialog.prepare, Qt.LeftButton)
            wait_result(session)
            for source, saved in before:
                same_mesh(source, saved)
            for record in session.records:
                assert window.slicer_parts[record['row']]['mesh'] is record['mesh'], 'Preview replaced project geometry'
            assert session.dialog.apply.isEnabled(), session.dialog.status.text() + '\n' + session.dialog.report.toPlainText()
            assert session.preview_actors, 'Prepared result was not rendered'
            assert window.history.index == 0, 'Preview added a history entry'

        def wait_result(session):
            wait_until(lambda: window._job is None and not session.dialog.running,
                       'Repair worker did not finish')
            wait_until(lambda: session.result is not None,
                       session.dialog.status.text() + '\n' + session.dialog.report.toPlainText(), timeout=2)

        def apply_with_history(session, original, check_result, untouched=None):
            QTest.mouseClick(session.dialog.apply, Qt.LeftButton)
            wait_until(lambda: window._repair_session is None, 'Apply did not close session')
            assert window.history.index == 1 and len(window.history.entries) == 2, 'Apply must be one undo step'
            check_result()
            if untouched is not None:
                same_mesh(next(part['mesh'] for part in window.slicer_parts if part['filename'] == 'Untouched.stl'), untouched)
            result = [part['mesh'].copy() for part in window.slicer_parts]
            window.ui.action_undo.trigger()
            assert window.history.index == 0
            assert len(window.slicer_parts) == len(original)
            for part, mesh in zip(window.slicer_parts, original):
                same_mesh(part['mesh'], mesh)
            window.ui.action_redo.trigger()
            assert window.history.index == 1
            for part, mesh in zip(window.slicer_parts, result):
                same_mesh(part['mesh'], mesh)
            assert not any(name.startswith('repair_preview_') for name in window.ui.slicer_plotter.actors)

        def capture(widget, name):
            path = output / name
            assert widget.grab().save(str(path)), 'Failed to save ' + name
            report['screenshots'].append(str(path))

        def scene(name):
            path = output / name
            image = window.ui.slicer_plotter.screenshot(str(path), return_img=True)
            assert image.shape[0] > 100 and np.ptp(image) > 50, 'Empty VTK screenshot'
            report['screenshots'].append(str(path))

        def projected(point):
            plotter = window.ui.slicer_plotter
            renderer = plotter.renderer
            renderer.SetWorldPoint(*point, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            ratio = plotter.devicePixelRatioF()
            return QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))

        def visible_pick(vertices=None, face=False):
            mesh = window.slicer_parts[0]['mesh']
            camera = np.asarray(window.ui.slicer_plotter.camera.position)
            facing = np.einsum('ij,ij->i', mesh.face_normals, camera - mesh.triangles_center)
            for index in np.argsort(facing)[::-1]:
                if facing[index] <= 0:
                    continue
                options = [None] if face else mesh.faces[index]
                for vertex in options:
                    if vertices is not None and vertex not in vertices:
                        continue
                    point = mesh.triangles_center[index] if face else .97 * mesh.vertices[vertex] + .03 * mesh.triangles_center[index]
                    pixel = projected(point)
                    hit = window.workspace_tools.picker(pixel, rows=[0])
                    if hit is None:
                        continue
                    _, picked_face, location = hit
                    picked_vertex = int(mesh.faces[picked_face][np.argmin(np.linalg.norm(mesh.vertices[mesh.faces[picked_face]] - location, axis=1))])
                    if (face and picked_face == index) or (not face and picked_vertex == vertex):
                        return pixel, int(index if face else vertex)
            raise AssertionError('Could not project a visible target for native picking')

        def check():
            try:
                # The ribbon itself, then real asynchronous preview/apply/history.
                for operation in ('duplicates', 'normals', 'subdivide'):
                    first = box()
                    if operation == 'duplicates':
                        first.faces = np.vstack((first.faces, first.faces[:1]))
                    first, other = fixture(first)
                    saved = [first.copy(), other.copy()]
                    if operation == 'duplicates':
                        capture(window.ui.magics_ribbon, 'repair-ribbon.png')
                    session = open_tool(operation)
                    if operation == 'normals':
                        session.dialog.flags['flip'].setChecked(True)
                    prepare(session)
                    if operation == 'duplicates':
                        capture(session.dialog, 'repair-duplicates-dialog.png')
                        scene('repair-preview-scene.png')
                    def expected(op=operation):
                        mesh = window.slicer_parts[0]['mesh']
                        assert len(mesh.faces) == (48 if op == 'subdivide' else 12)
                        if op == 'normals':
                            assert mesh.volume < 0
                    apply_with_history(session, saved, expected, untouched=saved[1])
                    same_mesh(first, saved[0])
                    report['checks'].append(operation + ': preview, apply, undo/redo, unselected part preserved')

                first, other = fixture()
                saved = first.copy()
                session = open_tool('subdivide')
                prepare(session)
                session.dialog.close_button.click()
                wait_until(lambda: window._repair_session is None, 'Cancel did not close')
                same_mesh(window.slicer_parts[0]['mesh'], saved)
                assert window.history.index == 0
                actor = window.ui.slicer_plotter.actors[window.slicer_parts[0]['actor_name']]
                assert actor.GetVisibility() and not any(name.startswith('repair_preview_') for name in window.ui.slicer_plotter.actors)
                report['checks'].append('Closing preview restores original actor and leaves history unchanged')

                first, other = fixture(trimesh.util.concatenate([box(), box((13, 0, 10), 4)]))
                session = open_tool('split')
                prepare(session)
                def split_expected():
                    assert len(window.slicer_parts) == 3
                    assert sorted(len(part['mesh'].faces) for part in window.slicer_parts) == [12, 12, 12]
                apply_with_history(session, [first.copy(), other.copy()], split_expected, untouched=other)
                report['checks'].append('Split creates separate parts in one undo step')

                third = box((24, 0, 10), 4)
                first, second = fixture(box(), box((4, 0, 10)), all_selected=True, third=third)
                session = open_tool('unify')
                prepare(session)
                def union_expected():
                    assert len(window.slicer_parts) == 2
                    assert window.slicer_parts[0]['mesh'].is_volume
                    assert abs(window.slicer_parts[0]['mesh'].volume - 1400) < .01
                apply_with_history(session, [first.copy(), second.copy(), third.copy()], union_expected, untouched=third)
                report['checks'].append('Actual Manifold union removes intersection and preserves unselected part')

                for operation in ('move_vertices', 'drag_vertices', 'delete_faces', 'fill_hole'):
                    first = box()
                    boundary_vertices = None
                    if operation == 'fill_hole':
                        removed = int(np.argmax(first.face_normals[:, 2]))
                        boundary_vertices = set(map(int, first.faces[removed]))
                        first.update_faces(np.arange(len(first.faces)) != removed)
                    first, other = fixture(first)
                    saved = [first.copy(), other.copy()]
                    session = open_tool(operation)
                    session.dialog.pick_button.click()
                    pixel, picked = visible_pick(vertices=boundary_vertices, face=operation == 'delete_faces')
                    plotter = window.ui.slicer_plotter
                    if operation == 'drag_vertices':
                        QTest.mousePress(plotter, Qt.LeftButton, Qt.NoModifier, pixel)
                        assert session.drag is not None, 'Native press did not begin vertex dragging'
                        destination = pixel + QPoint(12, -8)
                        QTest.mouseMove(plotter, destination)
                        assert np.linalg.norm(session.drag['last'] - session.drag['source']) > .01
                        same_mesh(first, saved[0])
                        QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, destination)
                        wait_result(session)
                        assert session.drag is None and session.dialog.apply.isEnabled()
                    else:
                        QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, pixel)
                        assert session.dialog.selected_ids() == [picked], 'Native pick returned a different ID'
                        if operation == 'move_vertices':
                            session.dialog.vectors['delta'][0].setValue(.3)
                        prepare(session)
                    same_mesh(first, saved[0])
                    if operation in ('move_vertices', 'drag_vertices'):
                        expected_vertices = saved[0].vertices.copy()
                        expected_vertices[picked] += np.array(session.dialog.parameters()['delta'])
                        def manual_expected(vertices=expected_vertices):
                            np.testing.assert_allclose(window.slicer_parts[0]['mesh'].vertices, vertices, atol=1e-10)
                    else:
                        def manual_expected(op=operation):
                            assert len(window.slicer_parts[0]['mesh'].faces) == (11 if op == 'delete_faces' else 12)
                    capture(session.dialog, 'repair-' + operation + '-dialog.png')
                    scene('repair-' + operation + '-scene.png')
                    apply_with_history(session, saved, manual_expected, untouched=saved[1])
                    same_mesh(first, saved[0])
                    report['checks'].append(operation + ': native pick/drag, preview, apply, undo/redo')
                report['status'] = 'ok'
            except Exception:
                report['status'] = 'error'
                report['error'] = traceback.format_exc()
                print(report['error'], flush=True)
            finally:
                try:
                    if window._job is not None:
                        window.cancel_current_job()
                        wait_until(lambda: window._job is None, 'Cancellation did not finish')
                    session = getattr(window, '_repair_session', None)
                    if session is not None:
                        session.dialog.reject()
                    window.dirty = False
                    window.close()
                except Exception:
                    report['cleanup_error'] = traceback.format_exc()
                    report['status'] = 'error'
                (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                app.quit()

        QTimer.singleShot(800, check)
        app.exec()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/repair-tools-smoke'))
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args._worker:
        return 0 if run_native(output)['status'] == 'ok' else 1
    if os.name != 'nt':
        raise SystemExit('This smoke check requires Windows.')
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
        print(json.dumps(failure))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
