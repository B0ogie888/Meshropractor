"""Native Qt/VTK regression for CAD and display tools; isolated preferences, no builds."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')

import numpy as np
import trimesh
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from OCP.BRep import BRep_Builder
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
import Meshropractor
import app_updater
from background_tasks import FunctionWorker
from cad_state import cad_status, require_native
from cad_geometry import export_step
from cad_import import load_step
from geometry_analysis import compute_heatmap
from project_store import ProjectState, save_project, load_project
from support_geometry import generate_supports
from support_tools import DEFAULTS


def main():
    output = ROOT / 'output/cad-display-smoke'; output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checks=[], screenshots=[])
    app = QApplication([]); app.setStyle('Fusion')
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(Meshropractor, 'QSettings', lambda *args: settings))
        for name in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, name, lambda *args: None))
        window = Meshropractor.MainWindow(); window.resize(1600, 1000); window.show()
        def wait_job():
            deadline = time.monotonic() + 60
            while window._job is not None:
                if time.monotonic() > deadline: raise AssertionError('Background job timed out')
                QTest.qWait(10)
            app.processEvents()
        def shot(name, widget=None):
            QTest.qWait(100)
            target = output / (name + '.png')
            assert (widget or window).grab().save(str(target))
            report['screenshots'].append(str(target))
            if widget is None:
                plotter = window.ui.plotter if window.ui.stack.currentWidget() is window.ui.page_predef else window.ui.slicer_plotter
                if plotter is not None:
                    scene_path = output / (name + '-scene.png')
                    plotter.screenshot(str(scene_path))
                    report['screenshots'].append(str(scene_path))
        def select_tab(title):
            ribbon = window.ui.magics_ribbon
            ribbon.setCurrentIndex(next(i for i in range(ribbon.count()) if ribbon.tabText(i) == title))
        try:
            shape = TopoDS_Compound(); builder = BRep_Builder(); builder.MakeCompound(shape)
            builder.Add(shape, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 10), gp_Dir(0, 0, 1)), 6, 12).Shape())
            builder.Add(shape, BRepPrimAPI_MakeBox(gp_Pnt(16, -4, 10), 8, 8, 9).Shape())
            path = Path(folder) / 'проверка.step'
            writer = STEPControl_Writer(); writer.Transfer(shape, STEPControl_AsIs); writer.Write(str(path))
            window.restore_project(ProjectState(page='slicer'))
            with patch('project_controller.QFileDialog.getOpenFileName', return_value=(str(path), '')), \
                 patch('import_dialog.StepImportDialog.exec', lambda self: 1):
                window.import_slicer_part(); wait_job()
            assert len(window.slicer_parts) == 2
            assert all(cad_status(p['mesh']) == 'native' for p in window.slicer_parts)
            report['checks'].append('Native STEP import retains and separates two BREP bodies')
            plotter = window.ui.slicer_plotter
            mesh = window.slicer_parts[0]['mesh']
            plotter.camera_position = [(25, -40, -35), (4, 0, 14), (0, 0, 1)]
            plotter.reset_camera(); plotter.render(); app.processEvents()
            bottom = int(np.flatnonzero(mesh.face_normals[:, 2] < -.99)[0])
            center = mesh.triangles_center[bottom]
            renderer = plotter.renderer; renderer.SetWorldPoint(*center, 1); renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint(); ratio = plotter.devicePixelRatioF()
            position = QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))
            QTest.mouseClick(window.workspace_tools.buttons['cad_face'], Qt.LeftButton)
            QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, position); app.processEvents()
            selected = window.workspace_tools.selection.get(0, set())
            assert selected and all(mesh.face_normals[list(selected), 2] < -.99)
            assert len(np.unique(require_native(mesh)['face_ids'][list(selected)])) == 1
            report['checks'].append('Real mouse pick selects a complete CAD face')
            records = window.workspace_tools.supports.records()
            window.start_job(FunctionWorker(generate_supports, records, {0: sorted(selected)}, DEFAULTS),
                             window.workspace_tools.supports.append_results); wait_job()
            group = window.slicer_parts[0]['supports'][0]; group_id = group['id']; geometry = group['vertices'].copy()
            assert 'cad_binding' in group and len(window.slicer_parts) == 2
            window.workspace_tools.supports.open(3)
            shot('cad-supports')
            window.workspace_tools.supports.panel.finish()
            select_tab('ИНСТРУМЕНТЫ'); window.cad_tools.open()
            window.cad_tools.properties(); wait_job()
            assert 'Площадь:' in window.cad_tools.dialog.details.toPlainText()
            shot('cad-tools', window.cad_tools.dialog)
            with patch('import_dialog.StepImportDialog.exec', lambda self: 1), \
                 patch('import_dialog.StepImportDialog.values', lambda self: (.01, .08)):
                window.cad_tools.quality(); wait_job()
            group = window.slicer_parts[0]['supports'][0]
            assert group['id'] == group_id and cad_status(window.slicer_parts[0]['mesh']) == 'native'
            np.testing.assert_array_equal(group['vertices'], geometry)
            report['checks'].append('Supports stay children of CAD through retessellation')
            window.cad_tools.dialog.close()
            window.apply_slicer_tool('Перемещать', dict(values=[2, 0, 0], pivot=0), [0])
            exported = Path(folder) / 'export.step'
            export_step([window.slicer_parts[0]['mesh']], exported)
            np.testing.assert_allclose(load_step(exported).bounds, window.slicer_parts[0]['mesh'].bounds, atol=.02)
            archive = Path(folder) / 'native.mrp'; save_project(archive, window.capture_project())
            restored = load_project(archive); window.restore_project(restored)
            assert all(cad_status(p['mesh']) == 'native' for p in window.slicer_parts)
            report['checks'].append('Placement, STEP export and project roundtrip preserve native CAD')
            select_tab('ОТОБРАЖЕНИЕ'); plotter = window.ui.slicer_plotter
            plotter.camera_position = [(45, -65, 45), (8, 0, 10), (0, 0, 1)]; plotter.reset_camera()
            for key in ('smooth', 'grid', 'dimensions', 'part_name'):
                window.ui.display_buttons[key].click()
            assert not window.display_tools.state['simplified']
            assert window.display_tools.overlays
            shot('display-tab'); shot('display-ribbon', window.ui.magics_ribbon)
            assert window.display_tools.capture_image().save(str(output / 'scene.png'))
            count = window.display_tools.rebuild_count
            for _ in range(5): window.display_tools.on_scene_changed()
            assert count == window.display_tools.rebuild_count
            report['checks'].append('Display toggles and cached overlays work in native VTK; PNG export works')
            with patch('project_controller.QFileDialog.getOpenFileName', return_value=(str(path), '')), \
                 patch('import_dialog.StepImportDialog.exec', lambda self: 1):
                window.load_cad(); wait_job()
            window.ui.stack.setCurrentWidget(window.ui.page_predef)
            cad = window.cad_mesh
            assert cad_status(cad) == 'native' and len(require_native(cad)['bodies']) == 2
            scan = trimesh.Trimesh(vertices=np.asarray(cad.vertices) + [0, 0, .03], faces=cad.faces[::3], process=False)
            assert not scan.is_watertight and np.isfinite(compute_heatmap(cad, scan)).all()
            window.ui.plotter.reset_camera(); shot('predeformation-step')
            report['checks'].append('Predeformation imports combined BREP and computes deviations with an open scan')
            report['status'] = 'ok'
        except Exception:
            report['status'] = 'failed'; report['error'] = traceback.format_exc()
            raise
        finally:
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if window._job is not None: window.cancel_current_job(); wait_job()
            window.dirty = False; window.close(); app.processEvents()


if __name__ == '__main__': main()
