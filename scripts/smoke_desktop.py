"""Manual Windows/OpenGL smoke check; briefly opens and closes the application."""
import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh
from PySide6.QtCore import QTimer, QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Meshropractor import MainWindow
from project_store import ProjectState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screenshot', default='output/desktop-smoke.png')
    parser.add_argument('--sections', action='store_true')
    parser.add_argument('--tools', action='store_true')
    args = parser.parse_args()
    app = QApplication([])
    app.setStyle('Fusion')
    window = MainWindow()
    window.resize(1600, 950)
    window.show()
    failures = []

    def check():
        try:
            cad = trimesh.creation.icosphere(subdivisions=2, radius=10)
            scan = cad.copy(); scan.apply_scale(1.01)
            window.restore_project(ProjectState(models=[
                dict(key='CAD_0', kind='CAD', name='CAD.stl', mesh=cad, style={}),
                dict(key='Scan_0', kind='Scan', name='Scan.stl', mesh=scan, style={}),
            ]))
            window.ui.tabs.setCurrentIndex(2)
            window.ui.plotter.reset_camera()
            window.ui.plotter.render()
            app.processEvents()
            path = Path(args.screenshot).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            image = window.ui.plotter.screenshot(str(path.with_name(path.stem + '-scene.png')), return_img=True)
            if np.ptp(image) == 0:
                raise RuntimeError('3D scene is a uniform image')
            if not window.grab().save(str(path)):
                raise RuntimeError('Screenshot failed')
            if args.sections:
                outer = trimesh.creation.icosphere(subdivisions=3, radius=10)
                inner = trimesh.creation.icosphere(subdivisions=3, radius=7)
                inner.invert()
                hollow = trimesh.util.concatenate([outer, inner])
                window.restore_project(ProjectState(parts=[dict(mesh=hollow, filename='Hollow shell.step',
                    style={'last_visible_mode': 'shaded', 'color': '#80b9df'})], page='slicer'))
                plotter = window.ui.slicer_plotter
                plotter.camera_position = [(30, 35, 25), (0, 0, 0), (0, 0, 1)]
                plotter.reset_camera()
                app.processEvents()
                before = plotter.screenshot(return_img=True)
                panel = window.ui.section_panel
                panel.table.cellWidget(0, 0).setChecked(True)
                panel.table.cellWidget(1, 0).setChecked(True)
                app.processEvents()
                after = plotter.screenshot(str(path.with_name('sections-scene.png')), return_img=True)
                if np.mean(np.abs(before.astype(float) - after.astype(float))) < 1:
                    raise RuntimeError('Clipping did not change the rendered image')
                panel.select_row(0)
                panel.manipulate.setChecked(True)
                if panel.widget is None:
                    raise RuntimeError('Interactive plane was not created')
                panel.table.cellWidget(0, 1).setCurrentText('Произв.')
                panel.widget.SetNormal(0.3, 0.2, 1)
                panel.widget.SetOrigin(0, 0, 2)
                panel.widget.InvokeEvent('InteractionEvent')
                if abs(panel.sections[0]['position']) < 1:
                    raise RuntimeError('Plane interaction did not update section position')
                panel.align_camera()
                panel.manipulate.setChecked(False)
                panel.table.cellWidget(0, 1).setCurrentText('XY')
                panel.table.cellWidget(0, 4).setValue(0)
                plotter.camera_position = [(30, 35, 25), (0, 0, 0), (0, 0, 1)]
                plotter.reset_camera()
                panel.manipulate.setChecked(True)
                app.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                app.processEvents()
                panel.grab().save(str(path.with_name('sections-controls.png')))
                window.grab().save(str(path.with_name('sections-panel.png')))
                plotter.screenshot(str(path.with_name('sections-widget.png')))
                np.testing.assert_array_equal(window.slicer_parts[0]['mesh'].vertices, hollow.vertices)
                print('Section clipping, plane interaction and original mesh preservation OK', flush=True)
            if args.tools:
                from import_dialog import StepImportDialog
                window.restore_project(ProjectState(parts=[dict(mesh=trimesh.creation.box(extents=[10, 15, 20]), filename='Tool test.stl')], page='slicer'))
                window.dirty = False
                window.reset_history()
                window.ui.magics_ribbon.setCurrentIndex(1)
                window.ui.slicer_plotter.reset_camera()
                app.processEvents()
                window.ui.magics_ribbon.grab().save(str(path.with_name('tools-ribbon.png')))
                window.ui.ribbon_btns['Перемещать'].click()
                session = window._transform_session
                session.dialog.values[0].setValue(8)
                session.update_preview()
                app.processEvents()
                matrix = window.ui.slicer_plotter.actors['slicer_part_0'].GetMatrix()
                if abs(matrix.GetElement(0, 3) - 8) > 1e-6:
                    raise RuntimeError(f'Live transformation preview did not update the actor: {matrix.GetElement(0, 3)}, {session.dialog.status.text()}, {session.dialog.parameters()}')
                session.dialog.grab().save(str(path.with_name('tools-move-dialog.png')))
                session.apply(True)
                np.testing.assert_allclose(window.slicer_parts[0]['mesh'].bounds.mean(axis=0), [8, 0, 0])
                window.ui.action_undo.trigger()
                np.testing.assert_allclose(window.slicer_parts[0]['mesh'].bounds.mean(axis=0), [0, 0, 0])
                window.ui.action_redo.trigger()
                np.testing.assert_allclose(window.slicer_parts[0]['mesh'].bounds.mean(axis=0), [8, 0, 0])
                window.run_slicer_tool('Перемещать')
                session = window._transform_session
                session.dialog.snap.setChecked(False)
                plotter = window.ui.slicer_plotter
                renderer = plotter.renderer
                arrow = session.gizmo._arrows[0]
                point = np.asarray(arrow.center)
                def display_point(world):
                    renderer.SetWorldPoint(*world, 1)
                    renderer.WorldToDisplay()
                    return tuple(round(v) for v in renderer.GetDisplayPoint()[:2])
                start = display_point(point)
                end = display_point(point + [5, 0, 0])
                interactor = plotter.iren.interactor
                interactor.SetEventInformation(*start)
                interactor.InvokeEvent('MouseMoveEvent')
                interactor.InvokeEvent('LeftButtonPressEvent')
                interactor.SetEventInformation(*end)
                interactor.InvokeEvent('MouseMoveEvent')
                app.processEvents()
                interactor.InvokeEvent('LeftButtonReleaseEvent')
                app.processEvents()
                if np.linalg.norm(session.dialog.numbers(session.dialog.values)) < .01:
                    raise RuntimeError('Mouse drag did not update the translation gizmo')
                session.dialog.reject()
                step_dialog = StepImportDialog(window)
                step_dialog.show()
                app.processEvents()
                step_dialog.grab().save(str(path.with_name('step-quality-dialog.png')))
                step_dialog.close()
                window.ui.magics_ribbon.setCurrentIndex(0)
                app.processEvents()
                window.ui.magics_ribbon.grab().save(str(path.with_name('home-ribbon.png')))
                for operation, slug in [('Вращать', 'rotate'), ('Масштабировать', 'scale'), ('Отзеркалить', 'mirror')]:
                    window.run_slicer_tool(operation)
                    session = window._transform_session
                    app.processEvents()
                    session.dialog.grab().save(str(path.with_name(f'transform-{slug}.png')))
                    window.ui.slicer_plotter.screenshot(str(path.with_name(f'transform-{slug}-scene.png')))
                    if operation == 'Вращать':
                        session.dialog.snap.setChecked(False)
                        ring = session.gizmo._circles[2]
                        points = ring.mapper.dataset.points
                        point = np.array(points[len(points) // 2]) + np.array(session.gizmo.origin)
                        vector = point - np.array(session.gizmo.origin)
                        theta = np.deg2rad(15)
                        end_point = np.array(session.gizmo.origin) + [np.cos(theta) * vector[0] - np.sin(theta) * vector[1],
                            np.sin(theta) * vector[0] + np.cos(theta) * vector[1], vector[2]]
                        interactor.SetEventInformation(*display_point(point))
                        interactor.InvokeEvent('MouseMoveEvent')
                        interactor.InvokeEvent('LeftButtonPressEvent')
                        interactor.SetEventInformation(*display_point(end_point))
                        interactor.InvokeEvent('MouseMoveEvent')
                        app.processEvents()
                        interactor.InvokeEvent('LeftButtonReleaseEvent')
                        app.processEvents()
                        if np.linalg.norm(session.dialog.numbers(session.dialog.values)) < .1:
                            raise RuntimeError('Mouse drag did not update the rotation gizmo')
                    if operation == 'Масштабировать':
                        session.dialog.fit.setChecked(True)
                        session.dialog.fit_target.setValue(4)
                        session.start_pick('measure', 2)
                        renderer = window.ui.slicer_plotter.renderer
                        picker = window.ui.slicer_plotter.iren.interactor.GetPicker()
                        center = window.slicer_parts[0]['mesh'].bounds.mean(axis=0)
                        z = window.slicer_parts[0]['mesh'].bounds[1, 2]
                        for x in (center[0] - 1, center[0] + 1):
                            renderer.SetWorldPoint(x, center[1], z, 1)
                            renderer.WorldToDisplay()
                            px, py, _ = renderer.GetDisplayPoint()
                            picker.Pick(px, py, 0, renderer)
                        if session.picking is not None or abs(session.dialog.measured.value() - 2) > .01:
                            raise RuntimeError(f'Surface point picking failed: {session.dialog.measured.value()}')
                        session.update_preview()
                        np.testing.assert_allclose(session.dialog.parameters()['values'], [2, 2, 2], atol=.01)
                    session.dialog.reject()
                print('Tools ribbon, live preview, Undo/Redo and STEP settings UI OK', flush=True)
            print(f'Desktop OpenGL smoke OK: {path}', flush=True)
        except Exception as exc:
            failures.append(exc)
            print(f'Desktop smoke failed: {exc}', flush=True)
        finally:
            window.dirty = False
            window.close()
            app.quit()

    QTimer.singleShot(400, check)
    app.exec()
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
