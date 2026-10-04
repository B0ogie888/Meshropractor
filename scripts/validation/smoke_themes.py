"""Native light/dark UI, compact geometry and pre-deformation control regression."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
import numpy as np
import trimesh
from PySide6.QtCore import QSettings, Qt, QPoint
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QCheckBox, QPushButton, QLabel
import main_window
import app_updater
from project_store import ProjectState
from radial_menu import RadialMenu
from display_settings import DisplaySettingsDialog, DisplaySettingsController


def main():
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT / 'output/new-theme-smoke'; output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checks=[])
    errors = []
    import traceback
    def exception_hook(*args):
        errors.append(''.join(traceback.format_exception(*args)))
        traceback.print_exception(*args)
    sys.excepthook = exception_hook
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        def settings(team, name):
            result = QSettings(str(Path(folder)/(name+'.ini')), QSettings.IniFormat)
            result.setFallbacksEnabled(False); return result
        patches.enter_context(patch.object(main_window, 'QSettings', settings))
        for name in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, name, lambda *args: None))
        window = main_window.MainWindow(); ui = window.ui
        window.resize(1600, 1000); window.show()
        try:
            box = trimesh.creation.box([30, 22, 18]); box.apply_translation([0, 0, 14])
            scan = box.copy(); scan.apply_translation([22, 30, 3])
            window.restore_project(ProjectState(page='slicer',
                platforms=[dict(name='Build', dim=[150, 150, 150], is_default=True)],
                parts=[dict(mesh=box, filename='Кронштейн.stl', platform='Build')],
                models=[dict(key='CAD_0', kind='CAD', name='Эталон.step', mesh=box.copy(), style={}),
                        dict(key='Scan_0', kind='Scan', name='Скан.stl', mesh=scan, style={})]))
            ui.scene_tabs.setCurrentIndex(1); QTest.qWait(100)
            def compact_positions():
                root = ui.slicer_left_scroll.widget()
                return [widget.mapTo(root, QPoint()).y() for widget in
                        (ui.cb_plat, ui.parts_browser.counter, ui.parts_browser.list, ui.part_controls)]
            window.resize(1600, 850); QTest.qWait(60); baseline = compact_positions()
            for height in (1100, 1500):
                window.resize(1600, height); QTest.qWait(60)
                assert compact_positions() == baseline, (height, baseline, compact_positions())
            window.showMaximized(); QTest.qWait(80)
            assert compact_positions() == baseline
            group = ui.slicer_normal_groups[0]
            group.toggle_button.click(); group.toggle_button.click(); QTest.qWait(60)
            assert compact_positions() == baseline
            window.showNormal(); window.resize(1600, 1000); QTest.qWait(60)
            report['checks'].append('Compact sidebar positions unchanged at heights 850/1100/1500, maximized and after collapse/reopen')

            ui.stack.setCurrentWidget(ui.page_predef); QTest.qWait(100)
            cad, scan_browser, heat, result = ui.predef_browsers
            assert [len(b.model.entries) for b in ui.predef_browsers] == [1,1,0,0]
            def click(browser, x=65, modifiers=Qt.NoModifier):
                rect = browser.list.visualRect(browser.model.index(0))
                QTest.mouseClick(browser.list.viewport(), Qt.LeftButton, modifiers,
                    QPoint(x if x>=0 else rect.width()+x, rect.top()+20)); QTest.qWait(50)
            controls = ui.predef_controls
            click(cad); assert len(controls.rows()) == 1
            click(scan_browser, modifiers=Qt.ControlModifier); assert len(controls.rows()) == 2
            controls.hide_button.click(); QTest.qWait(50)
            assert all(not window.actors[k].GetVisibility() for k in ('CAD_0','Scan_0'))
            controls.show_button.click(); QTest.qWait(50)
            assert all(window.actors[k].GetVisibility() for k in ('CAD_0','Scan_0'))
            click(scan_browser, -52); assert not window.actors['Scan_0'].GetVisibility()
            click(scan_browser, -52); assert window.actors['Scan_0'].GetVisibility()
            with patch('main_window.QColorDialog.getColor', return_value=QColor('#997744')):
                click(cad, -22)
            assert ui.mesh_colors['CAD_0'] == '#997744'
            controls.opacity.setValue(35); QTest.qWait(50)
            assert all(abs(window.actors[k].GetProperty().GetOpacity()-.65)<1e-6 for k in ('CAD_0','Scan_0'))
            controls.set_color(QColor('#99acaa')); QTest.qWait(50)
            window.undo_action(); QTest.qWait(50)
            assert ui.mesh_colors['CAD_0'] == '#997744'
            window.redo_action(); QTest.qWait(50)
            assert ui.mesh_colors['CAD_0'] == '#99acaa'
            controls.extra_columns.setChecked(True)
            assert all(b.stack.currentWidget() is b.table for b in ui.predef_browsers)
            controls.extra_columns.setChecked(False)
            controls.set_transparency(0); controls.set_shading(controls.shading.findData('shaded'))
            assert ui.page_predef.layout().itemAt(0).widget().height() < 65
            report['checks'].append('Pre-deformation cards: selection/Ctrl, eyes, colour swatches, bulk visibility/colour/opacity, Undo/Redo, full tables')

            hm = box.copy(); hm.apply_translation([0, 0, 28])
            key = window.add_def_table_item(ui.tbl_heat, 'Карта отклонений', 'Heatmap', actor_key='Heatmap_0')
            deviations = np.linspace(-.2, .3, len(hm.vertices))
            window._render_heatmap(key, hm, deviations)
            corrected = box.copy(); corrected.apply_translation([-40, 0, 0])
            key = window.add_def_table_item(ui.tbl_res, 'Компенсация', 'Result', actor_key='Result_0')
            window.show_mesh(key, corrected); QTest.qWait(60)
            click(heat); assert window.active_heatmap_key == 'Heatmap_0'
            click(result); assert window.active_result_key == 'Result_0'
            assert window.result_mesh is corrected
            lut = window.actors['Heatmap_0'].mapper.lookup_table.values.copy()
            report['checks'].append('Heatmap and result cards activate their real datasets, legend and export target')

            scene_color = tuple(window.actors['CAD_0'].GetProperty().GetColor())
            vertices = window.scene_models['CAD_0']['mesh'].vertices.copy()
            window.reset_history()
            for mode in ('light', 'dark', 'light', 'dark'):
                if window.engineering_theme.mode != mode: ui.theme_button.click()
                QTest.qWait(100)
                assert window.engineering_theme.mode == mode
                assert window.settings.value('appearance/theme', 'light') == mode
                assert tuple(window.actors['CAD_0'].GetProperty().GetColor()) == scene_color
                np.testing.assert_array_equal(vertices, window.scene_models['CAD_0']['mesh'].vertices)
                np.testing.assert_array_equal(lut, window.actors['Heatmap_0'].mapper.lookup_table.values)
                for scalar_bar in ui.plotter.scalar_bars.values():
                    ink_value = scalar_bar.GetLabelTextProperty().GetColor()[0]
                    assert ink_value < .2 if mode == 'light' else ink_value > .8
                ink = '#252b2b' if mode == 'light' else '#e6ede7'
                assert ui.part_controls.hide_button.palette().color(ui.part_controls.hide_button.foregroundRole()).name() == ink
                assert ui.predef_controls.hide_button.palette().color(ui.predef_controls.hide_button.foregroundRole()).name() == ink
                assert 'тёмную' in ui.theme_button.toolTip() if mode == 'light' else 'светлую' in ui.theme_button.toolTip()
                for i, step in enumerate(ui.predef_steps):
                    step.click(); QTest.qWait(30)
                    assert ui.tabs.currentIndex() == i and ui.tabs.widget(i).widget().isVisible()
                    assert ui.tabs.widget(i).height() > 600
                    assert ui.tabs.widget(i).horizontalScrollBar().maximum() == 0, (i, mode)
                    window.screen().grabWindow(window.winId()).save(str(output/f'predef-{mode}-{i}.png'))
                ui.show_slicer(); QTest.qWait(60)
                for i in range(ui.magics_ribbon.count()):
                    ui.magics_ribbon.setCurrentIndex(i); QTest.qWait(15)
                ui.magics_ribbon.setCurrentIndex(0)
                QTest.qWait(40)
                window.screen().grabWindow(window.winId()).save(str(output/f'slicer-{mode}.png'))
                menu = RadialMenu(window)
                menu.popup(ui.slicer_plotter.mapToGlobal(ui.slicer_plotter.rect().center())); QTest.qWait(70)
                assert menu.circle_color.alpha() < 200
                expected = '#18241d' if mode == 'light' else '#e6ede7'
                assert all(b.palette().color(b.foregroundRole()).name()==expected for b in menu.buttons)
                menu.grab().save(str(output/f'radial-{mode}.png'))
                QTest.keyClick(menu, Qt.Key_Escape); assert not menu.isVisible(); menu.deleteLater()
                ui.stack.setCurrentWidget(ui.page_predef)
            report['checks'].append('Live sun/moon toggles: panels, four workflow stages, all ribbons, native scenes, radial transparency/contrast, geometry and material colours preserved')
            controller = DisplaySettingsController(window)
            dialog = DisplaySettingsDialog(ui.display_preferences, controller.apply, window)
            dialog.show(); QTest.qWait(60)
            assert dialog.theme_choice.currentData() == 'dark'
            dialog.theme_choice.setCurrentIndex(0); dialog.reject()
            assert window.engineering_theme.mode == 'dark'
            dialog.show(); dialog.theme_choice.setCurrentIndex(0)
            dialog.buttons.button(QDialogButtonBox.Apply).click(); QTest.qWait(60)
            assert window.engineering_theme.mode == 'light'
            assert dialog.background_color == ui.display_preferences.background
            dialog.theme_choice.setCurrentIndex(1); dialog._set_background('#22332a'); dialog.apply(); QTest.qWait(60)
            assert ui.display_preferences.background == '#22332a'
            ui.theme_button.click(); QTest.qWait(30)
            ui.theme_button.click(); QTest.qWait(30)
            assert ui.display_preferences.background == '#22332a'
            window.screen().grabWindow(dialog.winId()).save(str(output/'settings-dark.png'))
            dialog.reject(); dialog.deleteLater()
            report['checks'].append('Settings: theme choice, cancel without changes, Apply, synchronized scene background')
            assert not errors, errors
            window.dirty = False; window.close(); app.processEvents()
            restored = main_window.MainWindow()
            assert restored.engineering_theme.mode == 'dark'
            assert restored.ui.display_preferences.background == '#22332a'
            restored.dirty = False; restored.close(); restored.deleteLater()
            report['checks'].append('New dark theme and scene background restored after reopening')
            report['status']='passed'
        except Exception:
            report['error']=traceback.format_exc(); raise
        finally:
            (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            window.dirty = False; window.close(); window.deleteLater(); app.processEvents()
    print('NEW_THEMES_SMOKE_OK')


if __name__ == '__main__': main()
