"""Native Windows/Linux GUI regression for the complete texture ribbon; no builds."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import tempfile
import traceback
import faulthandler
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
import main_window
import app_updater
from cad_import import load_step
from cad_state import cad_status, require_native
from part_supports import make_group, validate_groups
from project_store import ProjectState, save_project, load_project
from texture_geometry import KEY, appearance, effective_colors
from texture_dialog import TextureDialog


def main():
    output = ROOT / ('output/texture-smoke'); output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle('Fusion'); report = dict(status='running', checks=[])
    diagnostic = (output / 'native-error.log').open('w', encoding='utf-8', buffering=1)
    faulthandler.enable(file=diagnostic)
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        settings = QSettings(str(Path(folder) / 'settings.ini'), QSettings.IniFormat); settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window, 'QSettings', lambda *args: settings))
        for method in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, method, lambda *args: None))
        writer = STEPControl_Writer(); writer.Transfer(BRepPrimAPI_MakeBox(20, 20, 20).Shape(), STEPControl_AsIs)
        path = Path(folder) / 'box.step'; writer.Write(str(path))
        cad = load_step(path)
        from cad_state import apply_cad_transform
        apply_cad_transform(cad, trimesh.transformations.translation_matrix([0, 0, 5]))
        other = trimesh.creation.box([15, 15, 15]); other.apply_translation([40, 10, 12.5])
        support = trimesh.creation.box([2, 2, 5]); support.apply_translation([5, 5, 2.5])
        bottom = np.flatnonzero(cad.face_normals[:, 2] < -.99).tolist()
        group = make_group(support, bottom); support_id = group['id']
        diagnostic.write('Creating window\n')
        window = main_window.MainWindow()
        diagnostic.write('Window created\n')
        window.resize(1600, 1000); window.show()
        diagnostic.write('Window shown\n')
        try:
            window.restore_project(ProjectState(page='slicer',
                platforms=[dict(name='Build', dim=[100, 100, 100], is_default=True)],
                parts=[dict(mesh=cad, filename='CAD.step', platform='Build', supports=[group]),
                       dict(mesh=other, filename='Mesh.stl', platform='Build')]))
            window.ui.scene_tabs.setCurrentIndex(1); window.reset_history()
            ribbon = window.ui.magics_ribbon
            ribbon.setCurrentIndex(next(i for i in range(ribbon.count()) if ribbon.tabText(i) == 'ТЕКСТУРЫ'))
            plotter = window.ui.slicer_plotter
            plotter.camera_position = [(75, -90, 90), (20, 10, 10), (0, 0, 1)]; plotter.reset_camera(); plotter.render()
            tools = window.texture_tools; buttons = tools.buttons
            def select(rows):
                for row in range(len(window.slicer_parts)):
                    window.ui.tbl_parts.cellWidget(row, 1).findChild(QCheckBox).setChecked(row in rows)
            def click(op):
                buttons[op].click(); app.processEvents(); QTest.qWait(30)
            def current(row=0): return appearance(window.slicer_parts[row]['mesh'])
            select([0]); selected = np.flatnonzero(cad.face_normals[:, 1] < -.99).tolist()
            window.workspace_tools.edit_selection(0, selected, 'replace')
            image = np.full((128, 128, 3), [30, 150, 210], dtype=np.uint8)
            image[::16] = [240, 210, 70]; image[:, ::16] = [240, 210, 70]
            image_path = Path(folder) / 'grid.png'; Image.fromarray(image).save(image_path)
            with patch('texture_tools.QFileDialog.getOpenFileName', return_value=(str(image_path), '')), patch.object(TextureDialog, 'exec', lambda self: 1): click('new')
            assert len(current()['layers']) == 1 and current()['layers'][0]['mask'].sum() == len(selected)
            assert cad_status(window.slicer_parts[0]['mesh']) == 'native'
            assert window.workspace_tools.selection[0] == set(selected)
            shown = plotter.actors['slicer_part_0'].mapper.dataset
            assert shown.n_cells == len(cad.faces)
            np.testing.assert_allclose(shown.points.reshape(-1, 3, 3), window.slicer_parts[0]['mesh'].triangles)
            report['checks'].append('New image applies only to CAD surface; BREP, support ownership and picked cell IDs remain valid')
            preview = TextureDialog(current()['layers'][0], window); preview.show(); QTest.qWait(80)
            preview.grab().save(str(output / 'texture-dialog.png')); preview.close(); preview.deleteLater()
            window.workspace_tools.clear_selection()
            plotter.screenshot(str(output / 'textured-scene.png')); window.grab().save(str(output / 'texture-ribbon.png'))
            before = deepcopy(current()['layers'][0]['uv'])
            def accept_edit(dialog):
                dialog.name.setText('Изменённая'); dialog.fields['angle'].setValue(35); dialog.fields['repeat_u'].setValue(2)
                return 1
            with patch.object(TextureDialog, 'exec', accept_edit): click('edit')
            assert current()['layers'][0]['name'] == 'Изменённая'
            assert not np.array_equal(current()['layers'][0]['uv'], before)
            click('copy'); assert tools.clipboard['name'] == 'Изменённая'
            select([1]); click('paste'); assert len(current(1)['layers']) == 1
            assert current(1)['layers'][0]['id'] != current()['layers'][0]['id']
            window.travel_history(-1); assert appearance(window.slicer_parts[1]['mesh']) is None
            window.travel_history(1); assert len(current(1)['layers']) == 1
            assert window.slicer_parts[0]['supports'][0]['id'] == support_id
            report['checks'].append('Edit, independent copy/paste and actual GUI Undo/Redo retain textures and supports')
            select([0]); tools.active = (0, current()['layers'][0]['id'])
            with patch.object(QDialog, 'exec', lambda self: 1): click('select')
            assert window.selected_slicer_rows() == [0] and window.workspace_tools.selection[0] == set(selected)
            chosen = current()['layers'][0]['id']; click('clear'); assert not current()['layers'][0]['mask'].any()
            window.travel_history(-1); assert current()['layers'][0]['mask'].sum() == len(selected)
            tools.active = (0, chosen); click('invert'); assert not current()['layers'][0]['visible']
            click('invert'); assert current()['layers'][0]['visible']
            assert plotter.actors['slicer_part_0'].GetTexture() is not None
            click('show'); assert plotter.actors['slicer_part_0'].GetTexture() is None
            click('colors'); assert plotter.actors['slicer_part_0'].GetTexture() is not None
            Image.fromarray(np.full((128, 128, 3), [50, 220, 80], dtype=np.uint8)).save(image_path)
            click('update'); np.testing.assert_array_equal(current()['layers'][0]['image'][0, 0], [50, 220, 80])
            report['checks'].append('Texture selection, partial deletion, visibility inversion and image reload work through real ribbon buttons')
            # A CAD face remains selectable after image rendering explodes only the display proxy.
            window.workspace_tools.set_mode('cad_face'); window.workspace_tools.clear_selection()
            mesh = window.slicer_parts[0]['mesh']; face = selected[0]; center = mesh.triangles_center[face]
            renderer = plotter.renderer; renderer.SetWorldPoint(*center, 1); renderer.WorldToDisplay(); x, y, _ = renderer.GetDisplayPoint()
            ratio = plotter.devicePixelRatioF(); pos = QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))
            hit = window.workspace_tools.picker(pos); assert hit is not None and hit[0] == 0
            QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, pos); app.processEvents()
            picked = window.workspace_tools.selection[0]
            assert len(np.unique(require_native(mesh)['face_ids'][list(picked)])) == 1
            with patch('texture_tools.QColorDialog.getColor', return_value=QColor(240, 60, 40)): click('paint_faces')
            np.testing.assert_array_equal(effective_colors(window.slicer_parts[0]['mesh'])[list(picked)], np.tile([240, 60, 40, 255], (len(picked), 1)))
            assert cad_status(window.slicer_parts[0]['mesh']) == 'native'
            select([1])
            with patch('texture_tools.QColorDialog.getColor', return_value=QColor(60, 100, 220)): click('paint_part')
            np.testing.assert_array_equal(effective_colors(window.slicer_parts[1]['mesh']), np.tile([60, 100, 220, 255], (12, 1)))
            click('bake'); assert current(1)['layers'][0]['params']['projection'] == 'Атлас цветов'
            click('delete'); assert not current(1)['layers']
            report['checks'].append('Native mouse CAD-face painting and whole-part painting, baking and texture deletion are functional')
            select([0]); window.workspace_tools.clear_selection()
            saved = Path(folder) / 'textures.mrp'; save_project(saved, window.capture_project()); image_path.unlink()
            restored = load_project(saved); window.restore_project(restored)
            assert window.display_tools.state['texture'] and window.ui.texture_buttons['show'].isChecked()
            assert current()['layers'][0]['image'][0, 0].tolist() == [50, 220, 80]
            assert cad_status(window.slicer_parts[0]['mesh']) == 'native'; validate_groups(window.slicer_parts[0]['supports'], len(cad.faces))
            report['checks'].append('Project roundtrip embeds image, UV, colors and visibility; external image file is no longer required')
            select([0]); window.workspace_tools.clear_selection(); click('split')
            assert len(window.slicer_parts) >= 3
            assert sum(len(part['supports']) for part in window.slicer_parts) == 1
            for part in window.slicer_parts: validate_groups(part['supports'], len(part['mesh'].faces))
            window.travel_history(-1); assert len(window.slicer_parts) == 2 and cad_status(window.slicer_parts[0]['mesh']) == 'native'
            report['checks'].append('Color splitting remaps support provenance and Undo restores original native CAD')
            report['status'] = 'passed'
        finally:
            (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            window.dirty = False; window.close(); window.deleteLater(); app.processEvents()
    print('TEXTURE_SMOKE_OK')


if __name__ == '__main__':
    try: main()
    except BaseException:
        error = traceback.format_exc()
        (ROOT / 'output/texture-smoke/error.txt').write_text(error, encoding='utf-8')
        print(error, file=sys.__stderr__, flush=True)
        raise SystemExit(1)
