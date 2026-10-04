"""Native region placement, light/dark editor, editable plans and atomic marking."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform=='win32' else 'xcb')
import numpy as np
import trimesh
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent, QPalette, QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox
import main_window
import app_updater
from marking_geometry import KEY, plans
from project_store import ProjectState, save_project, load_project


def main():
    output = ROOT/'output/marking-smoke'; output.mkdir(parents=True, exist_ok=True)
    import faulthandler
    faulthandler.dump_traceback_later(30,repeat=True)
    app = QApplication([]); app.setStyle('Fusion')
    report = dict(status='running', checks=[])
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder)/'settings.ini'), QSettings.IniFormat)
        settings.setFallbacksEnabled(False); settings.setValue('appearance/theme','dark')
        patches.enter_context(patch.object(main_window,'QSettings',lambda *args:settings))
        for method in ('start','check','manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController,method,lambda *args:None))
        window = main_window.MainWindow(); window.resize(1600,1050); window.show()
        mesh = trimesh.creation.box([50,40,10]); mesh.apply_translation([0,0,5])
        base = ProjectState(page='slicer',platforms=[dict(name='Build',dim=[100,100,100],is_default=True)],
            parts=[dict(mesh=mesh,filename='Деталь.stl',platform='Build')])
        plotter = None

        def move(point, buttons):
            event = QMouseEvent(QEvent.MouseMove,QPointF(point),QPointF(plotter.mapToGlobal(point)),
                                Qt.NoButton,buttons,Qt.NoModifier)
            app.sendEvent(plotter,event)

        def screen(point):
            renderer = plotter.renderer; renderer.SetWorldPoint(*point,1); renderer.WorldToDisplay()
            x,y,_ = renderer.GetDisplayPoint(); ratio = plotter.devicePixelRatioF()
            return QPoint(round(x/ratio),round((plotter.render_window.GetSize()[1]-1-y)/ratio))

        def wait_job(session, timeout=30):
            end = time.monotonic()+timeout
            while window._job is not None or session.dialog.running or session.dialog.preview_running:
                assert time.monotonic()<end, 'Calculation timeout'
                app.processEvents(); time.sleep(.01)
            QTest.qWait(30)

        def prepare(session):
            print('PREPARE',session.dialog.content,session.dialog.shape.currentIndex(),flush=True)
            session.prepare(); wait_job(session)
            assert session.relief is not None, session.dialog.status.text()
            assert session.relief.is_volume
            assert session.dialog.apply.isEnabled()
            print('PREVIEW OK',flush=True)

        def open_editor():
            window.model_tools.open('label'); app.processEvents()
            session = window._repair_session; assert session is not None, window.ui.status_label.text()
            session.dialog.flags['auto_preview'].setChecked(False)
            return session

        try:
            window.restore_project(base); window.ui.scene_tabs.setCurrentIndex(1); window.reset_history()
            QTest.qWait(100)
            plotter = window.ui.slicer_plotter
            plotter.camera_position = [(0,0,150),(0,0,0),(0,1,0)]
            plotter.camera.parallel_projection = True; plotter.camera.parallel_scale = 45
            plotter.render()
            # Actual part hit must never enter VTK's trackball-rotate state.
            center = screen([0,0,10]); camera = np.asarray(list(plotter.camera_position))
            QTest.mousePress(plotter,Qt.LeftButton,Qt.NoModifier,center)
            move(center+QPoint(25,20),Qt.LeftButton)
            QTest.mouseRelease(plotter,Qt.LeftButton,Qt.NoModifier,center+QPoint(25,20))
            move(center+QPoint(70,40),Qt.NoButton)
            np.testing.assert_allclose(list(plotter.camera_position),camera)
            assert plotter.iren.interactor.GetInteractorStyle().GetState()==0
            QTest.mouseClick(plotter,Qt.LeftButton,Qt.NoModifier,center)
            assert window.selected_slicer_rows()==[0]
            session = open_editor(); dialog = session.dialog
            assert dialog.palette().color(QPalette.Window).lightness()<100
            dialog.text.setPlainText('О8'); dialog.fields['text_size'].setValue(5)
            start,end = screen([-16,8,10]),screen([16,-8,10])
            QTest.mousePress(plotter,Qt.LeftButton,Qt.NoModifier,start)
            move(end,Qt.LeftButton)
            QTest.mouseRelease(plotter,Qt.LeftButton,Qt.NoModifier,end)
            assert len(session.items)==1, dialog.status.text()
            np.testing.assert_allclose(list(plotter.camera_position),camera)
            prepare(session)
            assert session.preview_actor is not None
            plotter.screenshot(str(output/'surface-preview.png'))
            np.testing.assert_array_equal(window.slicer_parts[0]['mesh'].vertices,mesh.vertices)
            # Auto-preview must neither disable the editor nor steal the caret.
            import threading
            from marking_geometry import build_relief
            gate = threading.Event()
            def slow_preview(source,params,**kwargs):
                while not gate.wait(.01):
                    if kwargs['cancelled'](): raise InterruptedError()
                return build_relief(source,params,**kwargs)
            dialog.flags['auto_preview'].setChecked(True)
            dialog.activateWindow(); dialog.text.setFocus(); dialog.text.setPlainText('AB')
            cursor = dialog.text.textCursor(); cursor.movePosition(QTextCursor.End); dialog.text.setTextCursor(cursor)
            with patch('marking_previews.build_relief',slow_preview):
                until = time.monotonic()+3
                while not dialog.preview_running:
                    assert time.monotonic()<until; app.processEvents(); time.sleep(.01)
                assert window._job is None and dialog.editor.isEnabled() and dialog.text.hasFocus()
                QTest.keyClicks(dialog.text,'C'); assert dialog.text.toPlainText()=='ABC'
                gate.set()
                until = time.monotonic()+5
                while session.relief is None or dialog.preview_running or session.timer.isActive():
                    assert time.monotonic()<until,dialog.status.text(); app.processEvents(); time.sleep(.01)
                assert dialog.text.hasFocus() and session.params['text']=='ABC'
            # Closing during the debounce interval must show the latest draft,
            # not just the previous completed preview from the editor cache.
            dialog.text.setPlainText('PENDING'); dialog.reject(); app.processEvents()
            until = time.monotonic()+5
            while window.marking_previews.worker is not None or any(r['name'] not in plotter.actors for r in window.marking_previews.records.values()):
                assert time.monotonic()<until; app.processEvents(); time.sleep(.01)
            assert plans(window.slicer_parts[0]['mesh'])[0]['params']['text']=='PENDING'
            assert window.marking_previews.records
            visibility = window.ui.tbl_parts.cellWidget(0,window.COL_VISIBLE).findChild(QCheckBox)
            visibility.setChecked(False); app.processEvents()
            assert all(not plotter.actors[r['name']].GetVisibility() for r in window.marking_previews.records.values())
            visibility.setChecked(True); app.processEvents()
            assert all(plotter.actors[r['name']].GetVisibility() for r in window.marking_previews.records.values())
            session = open_editor(); dialog = session.dialog
            assert dialog.text.toPlainText()=='PENDING'
            dialog.text.setPlainText('О8'); prepare(session)
            report['checks'].append('Typing keeps focus/caret during auto-preview; close-before-preview shows latest draft; hiding the part hides drafts too')
            for mode in ('dark','light'):
                window.engineering_theme.set_mode(mode); QTest.qWait(60)
                dialog.grab().save(str(output/f'editor-{mode}.png'))
                assert (dialog.palette().color(QPalette.Window).lightness()<100)==(mode=='dark')
                for index,label in ((1,'image'),(2,'projection'),(3,'barcode')):
                    dialog.tabs.setCurrentIndex(index); QTest.qWait(30)
                    dialog.grab().save(str(output/f'{label}-{mode}.png'))
                dialog.tabs.setCurrentIndex(0)
            report['checks'].append('LMB selects/draws without camera rotation; first-open dark theme and live light/dark switching')

            dialog.shape.setCurrentIndex(1); prepare(session)
            dialog.shape.setCurrentIndex(0)
            # SVG imports with alpha; save/reopen keeps both the pixels and preview.
            svg = Path(folder)/'mark.svg'
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><path fill="black" fill-rule="evenodd" d="M5 5H35V35H5Z M13 13V27H27V13Z"/></svg>',encoding='utf-8')
            dialog.tabs.setCurrentIndex(1)
            with patch('marking_dialog.QFileDialog.getOpenFileName',return_value=(str(svg),'')):
                dialog.load_image()
            prepare(session)
            values = dialog.values(); dialog.load_values(values)
            assert dialog.image_label.pixmap() and not dialog.image_label.pixmap().isNull()
            dialog.tabs.setCurrentIndex(3); dialog.code.setPlainText('MP-03-128'); prepare(session)
            assert not dialog.code_preview.pixmap().isNull()
            dialog.tabs.setCurrentIndex(0); dialog.method.setCurrentIndex(1); prepare(session)
            target = Path(folder)/'relief.stl'
            with patch('marking_tools.QFileDialog.getSaveFileName',return_value=(str(target),'')):
                session.export()
            assert trimesh.load(target,force='mesh').is_volume
            report['checks'].append('Cyrillic/circular text, transparent SVG, Data Matrix and engraving preview produce closed meshes; STL export succeeds')

            session.save(); assert len(plans(window.slicer_parts[0]['mesh']))==1
            session.dialog.reject(); app.processEvents()
            assert window.marking_previews.records
            assert all(window.marking_previews.plotter.actors[r['name']].GetPickable()==0
                       for r in window.marking_previews.records.values())
            window.travel_history(-1); assert plans(window.slicer_parts[0]['mesh'])[0]['params']['text']=='PENDING'
            window.travel_history(1); assert len(plans(window.slicer_parts[0]['mesh']))==1
            path = Path(folder)/'marking.mrp'; save_project(path,window.capture_project())
            loaded = load_project(path); assert plans(loaded.parts[0]['mesh'])[0]['params']['text']=='О8'
            window.restore_project(loaded); app.processEvents()
            until = time.monotonic()+5
            while any(r['name'] not in plotter.actors for r in window.marking_previews.records.values()):
                assert time.monotonic()<until; app.processEvents(); time.sleep(.01)
            assert window.marking_previews.records
            session = open_editor(); prepare(session)
            volume = window.slicer_parts[0]['mesh'].volume
            session.apply()
            end = time.monotonic()+30
            while window._repair_session is not None:
                assert time.monotonic()<end, session.dialog.status.text()
                app.processEvents(); time.sleep(.01)
            result = window.slicer_parts[0]['mesh']
            assert len(window.slicer_parts)==1 and result.is_volume and result.volume<volume
            assert KEY not in result.metadata
            window.travel_history(-1); assert window.slicer_parts[0]['mesh'].volume==volume
            assert len(plans(window.slicer_parts[0]['mesh']))==1
            window.travel_history(1); assert window.slicer_parts[0]['mesh'].volume<volume
            report['checks'].append('Planned regions save in MRP and reopen editable; apply changes one part; both plan and geometry support Undo/Redo')
            # Closing the main window safely cancels an independent plan worker.
            gate.clear(); window.marking_previews.cache.clear()
            with patch('marking_previews.build_relief',slow_preview):
                window.restore_project(loaded); app.processEvents()
                until = time.monotonic()+5
                while window.marking_previews.worker is None:
                    assert time.monotonic()<until; app.processEvents(); time.sleep(.01)
                window.dirty=False; window.close()
                while window.isVisible():
                    assert time.monotonic()<until; app.processEvents(); time.sleep(.01)
                assert window.marking_previews.worker is None
            report['checks'].append('Main-window close cancels a running draft preview before disposing the scene')
            report['status'] = 'passed'
        except Exception:
            import traceback
            report['status']='failed'; report['error']=traceback.format_exc(); raise
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            if getattr(window,'_repair_session',None) is not None:
                window.cancel_current_job()
                until = time.monotonic()+10
                while window._job is not None and time.monotonic()<until: app.processEvents(); time.sleep(.02)
                window._repair_session.dialog.reject()
            until = time.monotonic()+10
            while window.marking_previews.worker is not None and time.monotonic()<until:
                app.processEvents(); time.sleep(.01)
            window.dirty=False; window.close(); window.deleteLater(); app.processEvents()
    print('MARKING_SMOKE_OK')


if __name__=='__main__': main()
