"""Native command dialogs, previews, atomic application and persistence."""
from contextlib import ExitStack
from pathlib import Path
import os,sys,tempfile,time,json
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','windows' if sys.platform=='win32' else 'xcb')
import numpy as np,trimesh
from PySide6.QtCore import QSettings,QRect
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QCheckBox
import main_window,app_updater
from project_store import ProjectState,save_project,load_project
from model_tool_ribbon import GROUPS


def main():
    output=ROOT/('output/model-tools-smoke');output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);checks=[]
    with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
        settings=QSettings(str(Path(folder)/'settings.ini'),QSettings.IniFormat);settings.setFallbacksEnabled(False)
        stack.enter_context(patch.object(main_window,'QSettings',lambda *args:settings))
        for method in ('start','check','manual_check'):stack.enter_context(patch.object(app_updater.UpdateController,method,lambda *args:None))
        window=main_window.MainWindow();window.resize(1300,950);window.show()
        try:
            a=trimesh.creation.box([10,10,10]);a.apply_translation([0,0,5]);b=a.copy();b.apply_translation([5,0,0])
            base=ProjectState(page='slicer',platforms=[dict(name='Build',dim=[100,100,100],is_default=True)],parts=[dict(mesh=a,filename='A.stl',platform='Build'),dict(mesh=b,filename='B.stl',platform='Build')])
            def reset(two=False):
                window.restore_project(base);window.ui.scene_tabs.setCurrentIndex(1);window.reset_history()
                window.ui.tbl_parts.cellWidget(1,1).findChild(QCheckBox).setChecked(two)
                window.workspace_tools.selection={0:set(np.flatnonzero(a.face_normals[:,2]>.9).tolist())}
                app.processEvents()
            reset();tabs=window.ui.magics_ribbon;tabs.setCurrentIndex(1)
            QTest.qWait(100);window.grab(QRect(0,0,1300,tabs.height()+45)).save(str(output/'tools-left.png'))
            tabs.widget(1).horizontalScrollBar().setValue(tabs.widget(1).horizontalScrollBar().maximum());QTest.qWait(40)
            window.grab(QRect(0,0,1300,tabs.height()+45)).save(str(output/'tools-right.png'))
            for op in [op for _,ops in GROUPS for op in ops]:
                reset(op in ('merge','boolean','remove_volume'))
                if op=='boolean':window.model_tools.buttons[op].menu().actions()[0].trigger()
                else:window.model_tools.buttons[op].click()
                app.processEvents();session=window._repair_session
                assert session is not None,op
                assert session.dialog.isVisible() and not window.ui.magics_ribbon.isEnabled(),op
                session.dialog.parameters();session.dialog.reject();app.processEvents()
                assert window._repair_session is None and window.ui.magics_ribbon.isEnabled(),op
            checks.append('All 22 commands open real parameter dialogs; closing restores controls')
            def prepare(op,params=None,two=False):
                reset(two);window.model_tools.open(op);session=window._repair_session
                assert session is not None,op
                for key,value in (params or {}).items():session.dialog.fields[key].setValue(value)
                session.dialog.prepare.click();end=time.monotonic()+40
                while window._job is not None:
                    assert time.monotonic()<end,'timeout '+op;QTest.qWait(10)
                app.processEvents();assert session.result is not None,session.dialog.report.toPlainText()
                assert session.preview_actors and session.dialog.apply.isEnabled(),op
                return session
            session=prepare('union',two=True);assert abs(session.result['items'][0]['meshes'][0].volume-1500)<.001
            session.dialog.apply.click();app.processEvents();assert len(window.slicer_parts)==1
            window.travel_history(-1);assert len(window.slicer_parts)==2
            window.travel_history(1);assert len(window.slicer_parts)==1
            session=prepare('hollow',dict(step=.5,wall=1.5));session.dialog.grab().save(str(output/'hollow-dialog.png'))
            assert np.array_equal(window.slicer_parts[0]['mesh'].vertices,a.vertices)
            session.dialog.apply.click();app.processEvents();assert window.slicer_parts[0]['mesh'].volume<1000
            window.travel_history(-1);assert window.slicer_parts[0]['mesh'].volume==1000
            session=prepare('cut');session.dialog.apply.click();app.processEvents();assert len(window.slicer_parts)==3
            assert abs(window.slicer_parts[0]['mesh'].volume+window.slicer_parts[1]['mesh'].volume-1000)<.001
            # Interactive marking is exercised separately by smoke_marking.py.
            path=Path(folder)/'model-tools.mrp';save_project(path,window.capture_project());loaded=load_project(path)
            assert len(loaded.parts)==3 and loaded.parts[1]['mesh'].is_volume
            checks.append('Real union/hollow/cut calculations preview without changing sources; application and Undo/Redo/MRP roundtrip succeed')
        finally:
            if getattr(window,'_repair_session',None):window._repair_session.dialog.reject()
            window.dirty=False;window.close();window.deleteLater();app.processEvents()
    (output/'report.json').write_text(json.dumps(dict(status='passed',checks=checks),ensure_ascii=False,indent=2),encoding='utf-8')
    print('MODEL_TOOLS_SMOKE_OK')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        error=traceback.format_exc();(ROOT/'output/model-tools-smoke/error.txt').write_text(error,encoding='utf-8');print(error,file=sys.__stderr__);raise SystemExit(1)
