"""Explicit distribution verification (--self-test), with real Qt/VTK rendering."""
from pathlib import Path
import json
import tempfile


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
    import Meshropractor
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
        original_settings = Meshropractor.QSettings
        Meshropractor.QSettings = lambda *args: settings
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
            window = Meshropractor.MainWindow()
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
            Meshropractor.QSettings = original_settings
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
