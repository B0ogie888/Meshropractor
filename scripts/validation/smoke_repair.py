"""Real Qt repair wizard smoke; optionally render a verified external CAD report."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import trimesh
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from main_window import MainWindow
from project_store import ProjectState
from repair_wizard import RepairWizard
from mesh_diagnostics import diagnose_mesh

parser=argparse.ArgumentParser()
parser.add_argument('--cad'); parser.add_argument('--repaired'); parser.add_argument('--report')
args=parser.parse_args()
app=QApplication([]); app.setStyle('Fusion')
window=MainWindow(); window.resize(1500,950); window.show()
failures=[]

def check():
    try:
        source=trimesh.load(args.cad,force='mesh') if args.cad else trimesh.creation.box()
        if not args.cad: source.update_faces(np.arange(11))
        repaired=trimesh.load(args.repaired,force='mesh') if args.repaired else trimesh.creation.box()
        report=json.loads(Path(args.report).read_text(encoding='utf8')) if args.report else dict(
            before=diagnose_mesh(source),after=diagnose_mesh(repaired),changed=True,acceptable=True)
        window.restore_project(ProjectState(models=[dict(key='CAD_1',kind='CAD',name=Path(args.cad).name if args.cad else 'Test CAD',mesh=source)],
                                            parts=[dict(mesh=trimesh.creation.box(),filename='Another part.stl')],page='slicer'))
        wizard=RepairWizard(window); wizard.models.setCurrentIndex(1)
        wizard.show_repair((repaired,report)); wizard.resize(880,860); wizard.show()
        app.processEvents()
        assert wizard.models.count()==2
        assert wizard.apply.isEnabled()
        assert window.ui.ribbon_btns['Автоисправление'].isEnabled()
        wizard.grab().save(str(Path('output/repair-wizard.png').resolve()))
        wizard.models.setCurrentIndex(0)
        assert wizard.repaired is None and not wizard.apply.isEnabled()
        wizard.close(); wizard.deleteLater()
        print('REPAIR_WIZARD_SMOKE_OK',flush=True)
    except Exception:
        failures.append(traceback.format_exc()); print(failures[-1],flush=True)
    finally:
        window.dirty=False; window.close(); app.quit()

QTimer.singleShot(1000,check)
app.exec()
if failures: raise SystemExit(1)
