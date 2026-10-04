"""Native virtual-copy lifecycle, placement gaps and remembered move anchors."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','windows' if sys.platform=='win32' else 'xcb')
import numpy as np
import trimesh
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
import main_window,app_updater
from project_store import ProjectState,save_project,load_project
from part_supports import make_group


def wait(app,predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while not predicate() and time.monotonic()<deadline:
        app.processEvents(); time.sleep(.01)
    assert predicate(),'Timed out waiting for GUI state'


def main():
    output=ROOT/'output/duplicate-move-smoke'; output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]); app.setStyle('Fusion'); report=dict(status='running',checks=[])
    with tempfile.TemporaryDirectory() as folder,ExitStack() as patches:
        settings=QSettings(str(Path(folder)/'settings.ini'),QSettings.IniFormat); settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window,'QSettings',lambda *args:settings))
        for method in ('start','check','manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController,method,lambda *args:None))
        window=main_window.MainWindow(); window.resize(1450,950); window.show()
        try:
            source=trimesh.creation.box([10,6,10]); source.apply_translation([0,0,5])
            child=trimesh.creation.box([2,2,2]); child.apply_translation([6,0,5])
            state=ProjectState(page='slicer',platforms=[dict(name='Build',dim=[100,100,100],is_default=True)],
                parts=[dict(mesh=source,filename='part.stl',platform='Build',supports=[make_group(child)],
                            style=dict(is_selected=True,is_visible=True))])
            window.restore_project(state); window.ui.scene_tabs.setCurrentIndex(1); window.dirty=False; window.reset_history()
            window.ui.slicer_plotter.view_isometric(); window.ui.slicer_plotter.reset_camera()
            before=window.slicer_parts[0]['mesh'].vertices.copy(); key=window.history.key
            window.run_slicer_tool('Дублировать'); session=window._duplicate_session; dialog=session.dialog
            assert len(session.actors)==2
            assert all(not a.pickable and 0<a.prop.opacity<1 for a in session.actors.values())
            original_ghost=session.actors[(0,'part',0)]
            dialog.counts[0].setValue(3); dialog.counts[1].setValue(2)
            wait(app,lambda:len(session.actors)==10)
            assert session.actors[(0,'part',0)] is original_ghost
            assert all(a.mapper is original_ghost.mapper for (row,tag,index),a in session.actors.items() if tag=='part')
            dialog.gaps[0].setValue(8)
            wait(app,lambda: any(abs(a.user_matrix[0,3]-20)<1e-7 for (row,tag,index),a in session.actors.items() if tag=='part'))
            assert len(window.capture_project().parts)==1 and not window.dirty and window.history.key==key
            np.testing.assert_array_equal(window.slicer_parts[0]['mesh'].vertices,before)
            for mode in ('light','dark'):
                window.engineering_theme.set_mode(mode); QTest.qWait(50)
                dialog.grab().save(str(output/f'duplicate-{mode}.png'))
                window.ui.slicer_plotter.screenshot(str(output/f'virtual-{mode}.png'))
            dialog.preview.setChecked(False); wait(app,lambda:not session.actors)
            dialog.preview.setChecked(True); wait(app,lambda:len(session.actors)==10)
            dialog.counts[0].setValue(1001); wait(app,lambda:not dialog.apply_button.isEnabled())
            assert not session.actors and len(window.slicer_parts)==1
            dialog.counts[0].setValue(3); wait(app,lambda:dialog.apply_button.isEnabled())
            offsets=[a.user_matrix[:3,3].copy() for (row,tag,index),a in session.actors.items() if tag=='part']
            dialog.apply_button.click(); app.processEvents()
            assert window._duplicate_session is None and len(window.slicer_parts)==6
            assert not any(name.startswith('duplicate_preview_') for name in window.ui.slicer_plotter.actors)
            for part,offset in zip(window.slicer_parts[1:],offsets):
                np.testing.assert_allclose(part['mesh'].bounds.mean(0),source.bounds.mean(0)+offset)
                np.testing.assert_allclose(np.asarray(part['supports'][0]['vertices']).mean(0),child.vertices.mean(0)+offset)
                assert part['platform']=='Build' and part['supports'][0]['id']!=state.parts[0]['supports'][0]['id']
            path=Path(folder)/'copies.mrp'; save_project(path,window.capture_project()); assert len(load_project(path).parts)==6
            window.travel_history(-1); assert len(window.slicer_parts)==1
            window.travel_history(1); assert len(window.slicer_parts)==6
            window.travel_history(-1); window.dirty=False; window.reset_history(); key=window.history.key
            window.run_slicer_tool('Дублировать'); session=window._duplicate_session
            assert [w.value() for w in session.dialog.counts]==[3,2,1]
            session.dialog.counts[0].setValue(4); session.dialog.reject(); QTest.qWait(100)
            assert window._duplicate_session is None and len(window.slicer_parts)==1 and window.history.key==key
            assert not any(name.startswith('duplicate_preview_') for name in window.ui.slicer_plotter.actors)
            report['checks'].append('Matrix gaps include child supports; live non-pickable shared-geometry ghosts never enter project/history; confirm, cancel, Undo/Redo and MRP pass')
            print('DUPLICATE_PREVIEW_OK',flush=True)

            window.run_slicer_tool('Перемещать'); session=window._transform_session; dialog=session.dialog
            assert all(not w.isEnabled() and w.value()==0 for w in dialog.anchor_custom)
            dialog.anchor_groups[2].button(0).click(); dialog.snap_step.setValue(7.5)
            assert json.loads(settings.value('move_preferences_v1'))['anchor_modes']==[1,1,0]
            for mode in ('light','dark'):
                window.engineering_theme.set_mode(mode); QTest.qWait(50)
                dialog.grab().save(str(output/f'move-{mode}.png'))
            dialog.reject(); app.processEvents()
            other=source.copy(); other.apply_translation([0,0,20])
            window.restore_project(ProjectState(page='slicer',parts=[dict(mesh=other,filename='second.stl',style=dict(is_selected=True))]))
            window.run_slicer_tool('Перемещать'); session=window._transform_session; dialog=session.dialog
            assert [g.checkedId() for g in dialog.anchor_groups]==[1,1,0]
            assert dialog.snap_step.value()==7.5 and dialog.target[2].value()==20
            assert [w.value() for w in dialog.values]==[0,0,0] and all(w.value()==0 for w in dialog.anchor_custom)
            dialog.anchor_groups[2].button(3).click(); assert dialog.anchor_custom[2].isEnabled()
            dialog.anchor_custom[2].setValue(7.25); dialog.reject(); app.processEvents()
            window.run_slicer_tool('Перемещать'); dialog=window._transform_session.dialog
            assert dialog.anchor_groups[2].checkedId()==3 and dialog.anchor_custom[2].value()==7.25
            dialog.anchor_groups[2].button(0).click()
            assert not dialog.anchor_custom[2].isEnabled() and dialog.anchor_custom[2].value()==0
            dialog.reject(); app.processEvents()
            report['checks'].append('Move custom coordinates are disabled/faint/zero until selected; anchor modes, custom coordinates and snapping automatically persist for subsequent parts while movement starts at zero')
            print('MOVE_PREFERENCES_OK',flush=True)
            window.dirty=False; window.run_slicer_tool('Дублировать'); session=window._duplicate_session
            session.dialog.counts[0].setValue(4); window.dirty=False; window.close(); QTest.qWait(100)
            assert window._duplicate_session is None and not window.isVisible()
            report['checks'].append('Main-window close removes pending virtual previews safely; diagrams and controls inspected in both themes')
            report['status']='passed'
        except Exception:
            import traceback
            report.update(status='failed',error=traceback.format_exc()); raise
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            window.dirty=False; window.close(); window.deleteLater(); app.processEvents()
    print('DUPLICATE_MOVE_SMOKE_OK')


if __name__=='__main__':
    import faulthandler
    faulthandler.dump_traceback_later(60,repeat=True)
    main()
