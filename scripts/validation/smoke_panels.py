"""Native regression for adjacent pre-deformation panels and the part inspector."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform=='win32' else 'xcb')
import numpy as np
import trimesh
from PySide6.QtCore import QSettings, Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
import main_window
import app_updater
from cad_geometry import _from_brep, _serialize
from cad_state import cad_status
from project_store import ProjectState
from placement_tools import PlacementSession
from repair_tools import RepairSession


def main():
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT/'output/new-panels-smoke'; output.mkdir(parents=True, exist_ok=True)
    errors = []; report = dict(status='running', checks=[])
    def hook(*args):
        errors.append(''.join(traceback.format_exception(*args))); traceback.print_exception(*args)
    sys.excepthook = hook
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        def settings(team, name):
            result = QSettings(str(Path(folder)/(name+'.ini')), QSettings.IniFormat)
            result.setFallbacksEnabled(False); return result
        patches.enter_context(patch.object(main_window, 'QSettings', settings))
        for method in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, method, lambda *args: None))
        window = main_window.MainWindow(); ui=window.ui
        window.resize(1600,1000); window.show()
        try:
            cad = _from_brep(_serialize(BRepPrimAPI_MakeBox(12,24,18).Shape()), .05, .25)
            mesh = trimesh.creation.box([20,30,40]); mesh.apply_translation([-35,0,20])
            window.restore_project(ProjectState(page='slicer',
                platforms=[dict(name='Build', dim=[150,150,150], is_default=True), dict(name='Second', dim=[100,100,100], is_default=True)],
                parts=[dict(mesh=cad, filename='Корпус.step', platform='Build'),
                       dict(mesh=mesh, filename='Опора.stl', platform='Build'),
                       dict(mesh=mesh.copy(), filename='Другая платформа.stl', platform='Second')],
                models=[dict(key='CAD_0', kind='CAD', name='Корпус.step', mesh=cad.copy(), style={})]))
            ui.scene_tabs.setCurrentIndex(1); QTest.qWait(120)
            inspector=ui.part_inspector; browser=ui.parts_browser
            def select(row, modifiers=Qt.NoModifier):
                rect=browser.list.visualRect(browser.model.index(row))
                QTest.mouseClick(browser.list.viewport(), Qt.LeftButton, modifiers, rect.topLeft()+QPoint(65,20)); QTest.qWait(50)
            select(0)
            assert inspector.title.text()=='Корпус.step'
            assert inspector.values['type'].text()=='CAD / BREP'
            assert inspector.values['bodies'].text()=='1' and inspector.values['faces'].text()=='6'
            np.testing.assert_allclose([float(v.text()) for v in inspector.axis_values], [12,24,18])
            with patch('part_inspector.cad_status', side_effect=AssertionError('Rehashed unchanged CAD')):
                inspector.refresh(); inspector.refresh()
            select(1, Qt.ControlModifier)
            assert inspector.values['type'].text()=='CAD: 1 · сетки: 1'
            bounds=np.asarray([cad.bounds,mesh.bounds])
            np.testing.assert_allclose([float(v.text()) for v in inspector.axis_values],bounds[:,1].max(0)-bounds[:,0].min(0))
            assert inspector.values['triangles'].text()=='24'
            select(1)
            assert inspector.values['type'].text()=='STL / сетка'
            assert inspector.values['units'].text()=='мм'
            report['checks'].append('Live STL/BREP/multiple selection, bounds, CAD bodies/faces, mm units, no repeated CAD hashing')

            plotter=ui.slicer_plotter
            plotter.camera_position=[(0,0,200),(0,0,0),(0,1,0)]
            plotter.camera.parallel_projection=True; plotter.camera.parallel_scale=65
            plotter.reset_camera_clipping_range(); plotter.render(); QTest.qWait(80)
            def pixel(point):
                renderer=plotter.renderer; renderer.SetWorldPoint(*point,1); renderer.WorldToDisplay()
                x,y,_=renderer.GetDisplayPoint(); ratio=plotter.devicePixelRatioF()
                return QPoint(round(x/ratio),round((plotter.render_window.GetSize()[1]-1-y)/ratio))
            QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, pixel(cad.bounds.mean(0))); QTest.qWait(80)
            assert window.selected_slicer_rows()==[0] and inspector.title.text()=='Корпус.step'
            QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, QPoint(15,15)); QTest.qWait(80)
            assert window.selected_slicer_rows()==[] and inspector.empty.isVisible()
            assert all(not b.isEnabled() for b in inspector.buttons.values())
            report['checks'].append('Real VTK scene click and empty click update inspector and disable empty-selection actions')

            select(0); window.reset_history()
            inspector.buttons['Перемещать'].click(); QTest.qWait(80)
            session=window._transform_session; assert session.operation=='Перемещать'
            before=window.slicer_parts[0]['mesh'].bounds.copy()
            session.dialog.set_values(session.dialog.values,[10,0,0]); session.dialog.sync_move(False)
            session.dialog.apply_button.click(); QTest.qWait(80)
            np.testing.assert_allclose(window.slicer_parts[0]['mesh'].bounds,before+[10,0,0])
            session.dialog.reject(); QTest.qWait(50)
            assert inspector.buttons['Вращать'].isEnabled()
            inspector.buttons['Вращать'].click(); QTest.qWait(80)
            session=window._transform_session; assert session.operation=='Вращать'
            session.dialog.set_values(session.dialog.values,[0,0,90]); session.dialog.apply_button.click(); QTest.qWait(80)
            session.dialog.reject(); QTest.qWait(50)
            np.testing.assert_allclose([float(v.text()) for v in inspector.axis_values],[24,12,18])
            assert cad_status(window.slicer_parts[0]['mesh'])=='native'
            window.undo_action(); QTest.qWait(80)
            np.testing.assert_allclose([float(v.text()) for v in inspector.axis_values],[12,24,18])
            window.redo_action(); QTest.qWait(80)
            np.testing.assert_allclose([float(v.text()) for v in inspector.axis_values],[24,12,18])
            with patch.object(window,'run_slicer_tool') as action:
                inspector.buttons['Масштабировать'].click(); inspector.buttons['Отзеркалить'].click()
                assert [c.args[0] for c in action.call_args_list]==['Масштабировать','Отзеркалить']
            with patch.object(window,'save_selected_slicer_parts') as save:
                inspector.save_button.click(); save.assert_called_once_with()
            inspector.cad_button.click(); QTest.qWait(40)
            assert window.cad_tools.dialog.isVisible(); window.cad_tools.dialog.close()
            report['checks'].append('Move/rotate buttons apply real native-CAD transforms; bounds follow apply/undo/redo; CAD and export handlers connected')

            for factory, operation in ((PlacementSession, 'free_move'), (RepairSession, 'delete_faces')):
                session = factory(window, operation, [0]); QTest.qWait(50)
                assert not inspector.import_button.isEnabled()
                assert all(not button.isEnabled() for button in inspector.buttons.values())
                session.dialog.reject(); QTest.qWait(50)
                assert inspector.import_button.isEnabled()
                assert all(button.isEnabled() for button in inspector.buttons.values())
            report['checks'].append('Inspector actions unlock after cancelling placement and repair sessions')

            ui.scene_tabs.setCurrentIndex(2); QTest.qWait(50)
            assert inspector.empty.isVisible()  # Hidden-platform selections cannot leak into context.
            select(0)
            assert inspector.title.text()=='Другая платформа.stl'
            ui.scene_tabs.setCurrentIndex(1); QTest.qWait(50)
            assert inspector.empty.isVisible()
            select(0)
            inspector.close_button.click(); QTest.qWait(40)
            assert not ui.slicer_rails.is_open(2) and not ui.inspector_button.isChecked()
            ui.inspector_button.click(); QTest.qWait(50)
            assert ui.slicer_rails.is_open(2) and ui.slicer_splitter.sizes()[2]>0
            report['checks'].append('Active platform filtering, hide/reopen inspector, scrollbar on short windows')

            for mode in ('light','dark'):
                window.engineering_theme.set_mode(mode); QTest.qWait(60)
                window.resize(1600,1000); ui.show_slicer(); QTest.qWait(60)
                assert ui.inspector_scroll.horizontalScrollBar().maximum()==0
                assert ui.scene_tabs.tabRect(0).bottom()+1 == ui.scene_tabs.height()
                assert ui.scene_tabs.mapTo(window,QPoint()).y()+ui.scene_tabs.height() == window.statusBar().y()
                window.screen().grabWindow(window.winId()).save(str(output/f'slicer-{mode}.png'))
                ui.stack.setCurrentWidget(ui.page_predef); QTest.qWait(70)
                workflow=ui.page_predef.layout().itemAt(0).widget()
                assert workflow.geometry().bottom()+1==ui.main_splitter.mapTo(ui.page_predef,QPoint()).y()
                assert ui.tabs.geometry()==ui.right_panel.rect()
                assert ui.right_panel.x()-(ui._def_center_container.x()+ui._def_center_container.width())==ui.main_splitter.handleWidth()
                assert ui.plotter.geometry()==ui._def_center_container.rect()
                for i in range(4):
                    ui.predef_steps[i].click(); QTest.qWait(30)
                    page=ui.tabs.widget(i)
                    assert page.mapTo(ui.right_panel,QPoint())==QPoint(0,0)
                    assert page.size()==ui.right_panel.size()
                ui.predef_steps[0].click(); QTest.qWait(40)
                window.screen().grabWindow(window.winId()).save(str(output/f'predef-{mode}.png'))
            ui.show_slicer()
            for width,height in ((1000,720),(800,680),(1920,1080)):
                window.resize(width,height); QTest.qWait(60)
                assert window.width()==width
                assert ui.inspector_scroll.horizontalScrollBar().maximum()==0
                if height<800: assert ui.inspector_scroll.verticalScrollBar().maximum()>0
            report['checks'].append('Both themes, 800/1000/1920 widths, pre-deformation panes meet scene and workflow without outer padding')
            assert not errors, errors
            ui.set_inspector_visible(False)
            window.dirty=False; window.close(); app.processEvents()
            restored=main_window.MainWindow()
            assert not restored.ui.slicer_rails.is_open(2)
            restored.dirty=False; restored.close(); restored.deleteLater()
            report['checks'].append('Inspector visibility restored after restart')
            report['status']='passed'
        except Exception:
            report['error']=traceback.format_exc(); raise
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            window.dirty=False; window.close(); window.deleteLater(); app.processEvents()
    print('NEW_PANELS_SMOKE_OK')


if __name__=='__main__': main()
