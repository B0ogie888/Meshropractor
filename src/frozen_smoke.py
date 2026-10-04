"""Explicit distribution verification (--self-test), with real Qt/VTK rendering."""
from pathlib import Path
import json
import tempfile


def check_preparation(window, output, report):
    """Exercise 0.3 assets and handles using only the packaged runtime."""
    import numpy as np
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from primitive_geometry import KINDS, FIELDS, build_primitive
    from primitive_dialog import PrimitiveDialog
    from pystrich.datamatrix import DataMatrixData, DataMatrixEncoder

    for kind in KINDS:
        values = {key: default for key, label, default in FIELDS[kind]}
        primitive = build_primitive(kind, values, [0, 0, 0], dict(mode='segments', segments=24))
        assert primitive.is_watertight and primitive.is_volume
    encoder = DataMatrixEncoder(DataMatrixData('Meshropractor 0.3', auto_encoding=True))
    assert encoder.get_imagedata(cellsize=4).startswith(b'\x89PNG')
    report['checks'].append('Ten primitive meshes and bundled Data Matrix encoder execute')
    for theme in ('light', 'dark'):
        window.engineering_theme.set_mode(theme)
        dialog = PrimitiveDialog(window); dialog.show(); QTest.qWait(50)
        assert dialog.grab().save(str(output / f'create-{theme}.png'))
        dialog.reject(); dialog.deleteLater()
        window.run_slicer_tool('Дублировать')
        session = window._duplicate_session
        assert session is not None and session.actors
        assert all(0 < actor.prop.opacity < 1 for actor in session.actors.values())
        assert session.dialog.grab().save(str(output / f'duplicate-{theme}.png'))
        session.dialog.reject(); QTest.qWait(50)
    report['checks'].append('Both themes display primitive schemes and live virtual copies')

    plotter = window.ui.slicer_plotter
    center = window.slicer_parts[0]['mesh'].bounds.mean(axis=0)
    plotter.camera_position = [center + [60, 60, 60], center, [0, 0, 1]]
    plotter.reset_camera_clipping_range()
    window.run_slicer_tool('Перемещать')
    session = window._transform_session
    session.dialog.snap.setChecked(False); session.update_preview()
    gizmo = session.gizmo
    assert len(gizmo.planes) == 3
    origin = gizmo.origin.copy()
    a, b = gizmo.planes['plane_xy'][:2]
    choices = [origin + gizmo.size * (u * gizmo.axes[a] + v * gizmo.axes[b])
               for u in (.3, .5, .7) for v in (.3, .5, .7)]
    world = next(point for point in choices if (hit := gizmo.hit(gizmo.project([point])[0])) and hit[0] == 'plane_xy')
    start = gizmo.project([world])[0]
    end = gizmo.project([world + [2, 3, 0]])[0]
    for kind, point, button, buttons in (
        (QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
        (QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton),
        (QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)):
        local = QPointF(*point)
        event = QMouseEvent(kind, local, QPointF(plotter.mapToGlobal(local.toPoint())), button, buttons, Qt.NoModifier)
        QApplication.sendEvent(plotter, event)
    np.testing.assert_allclose(session.dialog.numbers(session.dialog.values), [2, 3, 0], atol=.01)
    plotter.screenshot(str(output / 'move-planes.png'))
    session.dialog.reject(); QTest.qWait(50)
    window.run_slicer_tool('Вращать')
    session = window._transform_session
    assert set(session.gizmo.contours) == {'ring_x', 'ring_y', 'ring_z', 'ring_screen'}
    plotter.screenshot(str(output / 'rotation-rings.png'))
    session.dialog.reject(); QTest.qWait(50)
    report['checks'].append('Packaged planar mouse drag and four rotation rings render correctly')


def run(output):
    import numpy as np
    import torch
    import trimesh
    import rtree
    from PySide6.QtCore import QSettings
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    import main_window
    from cad_import import load_step
    from cad_state import require_native
    from project_store import ProjectState, save_project, load_project
    from repair_manual_geometry import edit_mesh
    from mesh_full_repair import run_engine
    from geometry_analysis import compute_heatmap
    from support_geometry import generate_supports
    from support_tools import DEFAULTS
    from cad_state import apply_cad_transform
    from cad_supports import bind_support_surface
    from part_supports import make_group

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checks=[], version=None)
    from app_version import APP_VERSION
    report['version'] = APP_VERSION
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    with tempfile.TemporaryDirectory(prefix='meshropractor-smoke-') as temporary:
        folder = Path(temporary)
        settings = QSettings(str(folder / 'settings.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        original_settings = main_window.QSettings
        main_window.QSettings = lambda *args: settings
        window = None
        try:
            index = rtree.index.Index()
            index.insert(7, (0, 0, 1, 1))
            assert list(index.intersection((.2, .2, .3, .3))) == [7]
            assert torch.tensor([1., 2.]).sum().item() == 3.
            report['checks'].append('Bundled R-tree and PyTorch execute')
            step = folder / 'проверка.step'
            writer = STEPControl_Writer()
            assert writer.Transfer(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), STEPControl_AsIs) == IFSelect_RetDone
            assert writer.Write(str(step)) == IFSelect_RetDone
            mesh = load_step(step)
            require_native(mesh)
            np.testing.assert_allclose(mesh.extents, [10, 20, 30], atol=1e-6)
            assert mesh.is_watertight
            report['checks'].append('STEP import retains native BREP faces and millimetres')
            matrix = np.eye(4)
            matrix[2, 3] = 10
            mesh = apply_cad_transform(mesh, matrix)
            bottom = np.flatnonzero(mesh.face_normals[:, 2] < -.99)
            record = dict(row=0, mesh=mesh, filename=step.name, platform='Build')
            results = generate_supports([record], {0: bottom.tolist()}, DEFAULTS)
            assert results and results[0]['contacts'] > 0
            group = bind_support_surface(make_group(results[0]['mesh'], bottom.tolist()), mesh)
            assert group['cad_binding']
            report['checks'].append('Open3D generates supports bound to a native CAD face')
            archive = folder / 'project.mrp'
            state = ProjectState(page='slicer', parts=[dict(mesh=mesh, filename=step.name, supports=[group])])
            save_project(archive, state)
            state = load_project(archive)
            require_native(state.parts[0]['mesh'])
            report['checks'].append('Project roundtrip retains CAD geometry')
            cut, _ = edit_mesh(trimesh.creation.box(), 'clip',
                            dict(point=[0, 0, 0], normal=[0, 0, 1], cap=True))
            assert cut.is_watertight and abs(cut.volume - .5) < 1e-6
            report['checks'].append('Bundled Shapely and capped plane clipping execute')
            broken = trimesh.creation.icosphere(subdivisions=2)
            broken.update_faces(np.arange(len(broken.faces) - 1))
            repaired, _ = run_engine(broken, 3, lambda value: None, lambda: False)
            assert repaired.is_watertight and repaired.is_winding_consistent
            report['checks'].append('Independent bundled repair executable closes a hole')
            window = main_window.MainWindow()
            window.resize(1400, 900)
            window.show()
            window.restore_project(state)
            window.ui.scene_tabs.setCurrentIndex(1)
            QTest.qWait(300)
            plotter = window.ui.slicer_plotter
            plotter.reset_camera()
            plotter.render()
            image = plotter.screenshot(str(output / 'slicer.png'))
            assert np.ptp(image[..., :3]) > 30
            assert np.unique(image[..., :3].reshape(-1, 3), axis=0).shape[0] > 20
            assert window.grab().save(str(output / 'application.png'))
            report['checks'].append('Installed application opens and renders the slicer scene')
            check_preparation(window, output, report)
            window.restore_project(ProjectState(page='predef', models=[
                dict(mesh=mesh, kind='CAD', key='CAD_0', name=step.name, style={})]))
            QTest.qWait(200)
            window.ui.plotter.reset_camera()
            window.ui.plotter.render()
            window.ui.plotter.screenshot(str(output / 'predeformation.png'))
            report['checks'].append('Native CAD displays in predeformation')
            scan = trimesh.Trimesh(np.asarray(mesh.vertices) + [0, 0, .03], mesh.faces[::3], process=False)
            assert not scan.is_watertight and np.isfinite(compute_heatmap(mesh, scan)).all()
            report['checks'].append('Deviation map computes for native CAD and an open scan')
            if window.updater.start_timer.isActive():
                raise AssertionError('Self-test must not check for updates')
            report['status'] = 'passed'
        except Exception:
            import traceback
            report['status'] = 'failed'
            report['error'] = traceback.format_exc()
            raise
        finally:
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if window is not None:
                window.dirty = False
                window.close()
                app.processEvents()
            main_window.QSettings = original_settings
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
