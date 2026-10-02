"""Check native Windows QVTK drops, STEP choices and predeformation; no builds."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')

import trimesh
import numpy as np
from PySide6.QtCore import QMimeData, QPoint, QPointF, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
import Meshropractor
import app_updater
from cad_state import cad_status
from import_dialog import StepImportDialog
from project_store import ProjectState


def main():
    output = ROOT / 'output/scene-drop-smoke'; output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle('Fusion')
    report = dict(status='running', checks=[])
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(Meshropractor, 'QSettings', lambda *args: settings))
        for name in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, name, lambda *args: None))
        patches.enter_context(patch('mesh_repair.request_repair', return_value=False))
        window = Meshropractor.MainWindow(); window.resize(1400, 950); window.show()
        def wait():
            deadline = time.monotonic() + 45
            while window._job is not None or window.drop_imports.pending:
                if time.monotonic() > deadline: raise AssertionError('Drop import timed out')
                QTest.qWait(10)
            QTest.qWait(100)
        def drop(plotter, path):
            mime = QMimeData(); mime.setUrls([QUrl.fromLocalFile(str(path))])
            enter = QDragEnterEvent(QPoint(100, 100), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            app.sendEvent(plotter, enter)
            event = QDropEvent(QPointF(100, 100), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            app.sendEvent(plotter, event)
            assert enter.isAccepted() and event.isAccepted(), 'QVTK widget rejected the file'
        original_exec = StepImportDialog.exec
        as_mesh = False
        def step_dialog(dialog):
            def choose():
                if as_mesh: QTest.mouseClick(dialog.mesh, Qt.LeftButton)
                dialog.grab().save(str(output / ('step-mesh.png' if as_mesh else 'step-brep.png')))
                QTest.mouseClick(dialog.buttons.button(QDialogButtonBox.Ok), Qt.LeftButton)
            QTimer.singleShot(200, choose)
            return original_exec(dialog)
        patches.enter_context(patch.object(StepImportDialog, 'exec', step_dialog))
        try:
            stl = Path(folder) / 'деталь.stl'; trimesh.creation.box().export(stl)
            step = Path(folder) / 'CAD.step'
            writer = STEPControl_Writer(); writer.Transfer(BRepPrimAPI_MakeBox(5, 6, 7).Shape(), STEPControl_AsIs)
            writer.Write(str(step))
            window.restore_project(ProjectState(page='slicer'))
            drop(window.ui.slicer_plotter, stl); wait()
            assert len(window.slicer_parts) == 1
            report['checks'].append('STL loads through a real QVTK drop event')
            drop(window.ui.slicer_plotter, step); wait()
            assert cad_status(window.slicer_parts[-1]['mesh']) == 'native'
            report['checks'].append('STEP dialog accepts BREP and retains CAD')
            as_mesh = True
            drop(window.ui.slicer_plotter, step); wait()
            assert cad_status(window.slicer_parts[-1]['mesh']) == 'mesh'
            report['checks'].append('STEP dialog switches to a plain STL mesh')
            plotter = window.ui.slicer_plotter
            np.testing.assert_allclose(plotter.camera.focal_point, [0, 0, 0])
            window.ui.tbl_parts.cellClicked.emit(1, 0)
            np.testing.assert_allclose(plotter.camera.focal_point, [0, 0, 0])
            point = window.slicer_parts[-1]['mesh'].centroid
            renderer = plotter.renderer; renderer.SetWorldPoint(*point, 1); renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint(); ratio = plotter.devicePixelRatioF()
            position = QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))
            hit = window.workspace_tools.picker(position, selected_only=False)
            assert hit is not None, 'Projected part is not pickable'
            expected = plotter.actors[window.slicer_parts[hit[0]]['actor_name']].center
            QTest.mouseDClick(plotter, Qt.LeftButton, Qt.NoModifier, position)
            QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, position)
            np.testing.assert_allclose(plotter.camera.focal_point, expected)
            # The initial STL was much smaller than this STEP; make background visible.
            direction = np.asarray(plotter.camera.position) - np.asarray(plotter.camera.focal_point)
            plotter.camera.position = np.asarray(expected) + 60. * direction / np.linalg.norm(direction)
            plotter.reset_camera_clipping_range(); plotter.render(); app.processEvents()
            empty = QPoint(10, 10)
            assert window.workspace_tools.picker(empty, selected_only=False) is None
            QTest.mouseDClick(plotter, Qt.LeftButton, Qt.NoModifier, empty)
            QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, empty)
            np.testing.assert_allclose(plotter.camera.focal_point, [0, 0, 0])
            report['checks'].append('Native viewport double-click changes pivot; empty space resets plate center')
            window.ui.slicer_plotter.screenshot(str(output / 'slicer-scene.png'))
            window.ui.stack.setCurrentWidget(window.ui.page_predef)
            as_mesh = False
            drop(window.ui.plotter, step); wait()
            assert cad_status(window.cad_mesh) == 'native'
            report['checks'].append('STEP drop imports BREP into predeformation')
            window.ui.plotter.screenshot(str(output / 'predeformation-scene.png'))
            report['status'] = 'ok'
        finally:
            (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            window.drop_imports.clear()
            if window._job is not None:
                window.cancel_current_job(); wait()
            window.dirty = False; window.close(); window.deleteLater(); app.processEvents()


if __name__ == '__main__':
    main()
