"""Native create-part parameter diagrams, linked R/D, themes and project history."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','windows' if sys.platform=='win32' else 'xcb')
from PySide6.QtCore import QSettings,QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
import main_window,app_updater
from primitive_dialog import PrimitiveDialog
from primitive_geometry import KINDS
from project_store import ProjectState,save_project,load_project


def main():
    output=ROOT/'output/primitives-smoke'; output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]); app.setStyle('Fusion'); report=dict(status='running',checks=[])
    with tempfile.TemporaryDirectory() as folder,ExitStack() as patches:
        settings=QSettings(str(Path(folder)/'settings.ini'),QSettings.IniFormat); settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window,'QSettings',lambda *args:settings))
        for method in ('start','check','manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController,method,lambda *args:None))
        window=main_window.MainWindow(); window.resize(1450,950); window.show()
        try:
            base=ProjectState(page='slicer',platforms=[dict(name='Build',dim=[100,100,100],is_default=True)])
            window.restore_project(base); window.ui.scene_tabs.setCurrentIndex(1); window.reset_history()
            dialog=PrimitiveDialog(window); dialog.show(); QTest.qWait(100)
            for kind in KINDS:
                dialog.kind.setCurrentText(kind); dialog.on_plate(); app.processEvents()
                assert dialog.sketch.mesh is not None,kind
                for key,diameter in dialog.diameters.items():
                    original=dialog.inputs[key].value(); diameter.setValue(original*2+1)
                    assert dialog.inputs[key].value()==original+.5
                    dialog.inputs[key].setValue(original); assert diameter.value()==original*2
                params=dialog.parameters(); window.apply_slicer_tool('Создать',params)
                part=window.slicer_parts[-1]; assert part['platform']=='Build' and part['mesh'].is_volume
                assert abs(part['mesh'].bounds[0,2])<1e-7
                for mode in ('light','dark'):
                    window.engineering_theme.set_mode(mode); QTest.qWait(30)
                    dialog.grab().save(str(output/f'{KINDS.index(kind):02}-{mode}.png'))
                assert dialog.buttons.buttons()[0].isEnabled(),kind
            count=len(window.slicer_parts); window.travel_history(-1); assert len(window.slicer_parts)==count-1
            window.travel_history(1); assert len(window.slicer_parts)==count
            path=Path(folder)/'primitives.mrp'; save_project(path,window.capture_project())
            loaded=load_project(path); assert len(loaded.parts)==len(KINDS)
            assert all(p['mesh'].is_volume for p in loaded.parts)
            dialog.close(); dialog.deleteLater(); app.processEvents()
            # Exercise the actual ribbon entry, not only the standalone dialog.
            def accept_create():
                active=app.activeModalWidget(); assert isinstance(active,PrimitiveDialog)
                active.kind.setCurrentText('Труба'); active.accept()
            QTimer.singleShot(100,accept_create); window.run_slicer_tool('Создать')
            assert len(window.slicer_parts)==count+1
            report.update(status='passed',checks=['All ten dimensioned diagrams render in light/dark themes',
                'Radius/diameter links, plate placement and closed meshes in active platform',
                'Ribbon opens the new dialog; creating parts supports Undo/Redo and MRP'])
        except Exception:
            import traceback
            report.update(status='failed',error=traceback.format_exc()); raise
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            window.dirty=False; window.close(); window.deleteLater(); app.processEvents()
    print('PRIMITIVES_SMOKE_OK')


if __name__=='__main__': main()
