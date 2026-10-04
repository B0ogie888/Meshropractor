"""Native parity, shared projects, appearance controls and live cube interaction."""
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
from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QToolButton, QDialog, QVBoxLayout, QLabel
import main_window
import app_updater
from project_store import ProjectState, save_project, load_project


def inventory(window):
    ui = window.ui
    return dict(commands={k: b.isEnabled() for k, b in ui.ribbon_btns.items()},
                tabs=[ui.magics_ribbon.tabText(i) for i in range(ui.magics_ribbon.count())],
                sliders=sorted(ui.sliders), operations={name: sorted(getattr(ui, name)) for name in
                ('model_tool_buttons', 'repair_buttons', 'texture_buttons', 'display_buttons', 'analysis_buttons')})


def main():
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT / 'output/new-ui-smoke'; output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'checks': []}
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        # Isolate real desktop preferences while validating the unified UI.
        def settings(team, name):
            s = QSettings(str(Path(folder) / (name + '.ini')), QSettings.IniFormat)
            s.setFallbacksEnabled(False); return s
        patches.enter_context(patch.object(main_window, 'QSettings', settings))
        for method in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, method, lambda *args: None))
        restored_window = main_window.MainWindow()
        baseline = inventory(restored_window)
        window = main_window.MainWindow()
        window.resize(1600, 1000); window.show()
        try:
            expected = {**baseline, 'commands': {name: enabled for name, enabled in baseline['commands'].items()
                                                if name != 'CAD / STEP'}}
            assert inventory(window) == expected
            assert len(baseline['tabs']) == 11
            assert hasattr(window.ui.part_inspector, 'cad_button')
            assert window.settings.fileName() == restored_window.settings.fileName()
            assert window.ui.display_preferences.background == '#e6eae5'
            assert window.windowTitle().startswith('Meshropractor —')
            assert 'Classic' not in window.windowTitle() and 'New' not in window.windowTitle()
            assert window.ui.menu_btn.isHidden()
            window.ui.new_log_button.click(); QTest.qWait(20)
            assert window.ui.log_dock.isVisible()
            window.ui.new_log_button.click()
            report['inventory'] = baseline
            report['checks'].append('Default unified interface: 11 ribbons, contextual CAD inspector, single title/settings store')
            QTest.qWait(100); window.grab().save(str(output / 'new-home.png'))
            assert window.ui.btn_new_project.palette().color(window.ui.btn_new_project.foregroundRole()).name() == '#252b2b'
            window.ui.show_slicer(); QTest.qWait(80)
            browser = window.ui.parts_browser
            assert browser.stack.currentWidget() is browser.empty
            window.screen().grabWindow(window.winId()).save(str(output/'new-empty-screen.png'))
            a = trimesh.creation.box([30, 25, 20]); a.apply_translation([0, 0, 16])
            b = trimesh.creation.cylinder(radius=12, height=30); b.apply_translation([40, 0, 15])
            state = ProjectState(page='slicer', platforms=[dict(name='Build', dim=[150, 150, 150], is_default=True)],
                parts=[dict(mesh=a, filename='Кронштейн.stl', platform='Build'), dict(mesh=b, filename='Втулка.stl', platform='Build')],
                models=[dict(key='CAD_0', kind='CAD', name='CAD', mesh=a.copy(), style={}),
                        dict(key='Scan_0', kind='Scan', name='Скан', mesh=b.copy(), style={})])
            window.restore_project(state); window.ui.scene_tabs.setCurrentIndex(1); window.reset_history()
            QTest.qWait(100)
            panel = window.ui.part_controls
            plotter = window.ui.slicer_plotter
            actors = lambda: [plotter.actors[p['actor_name']] for p in window.slicer_parts]
            browser = window.ui.parts_browser
            def row_click(index, x=65, modifiers=Qt.NoModifier):
                rect = browser.list.visualRect(browser.model.index(index))
                QTest.mouseClick(browser.list.viewport(), Qt.LeftButton, modifiers,
                                 QPoint(x if x>=0 else rect.width()+x, rect.top()+20))
                QTest.qWait(40)
            assert len(browser.model.entries) == 2
            row_click(0); assert window.selected_slicer_rows() == [0]
            row_click(1, modifiers=Qt.ControlModifier); assert window.selected_slicer_rows() == [0, 1]
            row_click(0, -52); assert not actors()[0].GetVisibility() and actors()[1].GetVisibility()
            row_click(0, -52); assert actors()[0].GetVisibility()
            with patch('main_window.QColorDialog.getColor', return_value=QColor('#997744')):
                row_click(0, -22)
            assert np.allclose(actors()[0].GetProperty().GetColor(), [153/255,119/255,68/255])
            assert not hasattr(browser, 'search')
            browser.none_button.click(); QTest.qWait(30); assert window.selected_slicer_rows() == []
            browser.all_button.click(); QTest.qWait(30); assert window.selected_slicer_rows() == [0, 1]
            browser.list.setCurrentIndex(browser.model.index(0))
            QTest.keyClick(browser.list, Qt.Key_Space); QTest.qWait(30)
            assert window.selected_slicer_rows() == [1]
            QTest.keyClick(browser.list, Qt.Key_A, Qt.ControlModifier); QTest.qWait(30)
            assert window.selected_slicer_rows() == [0, 1]
            report['checks'].append('Compact browser: click/Ctrl/keyboard selection, select-all/clear, per-row eye and colour; menu and search removed, operation log retained')
            panel.hide_button.click(); QTest.qWait(50)
            assert all(not actor.GetVisibility() for actor in actors())
            assert window.selected_slicer_rows() == [0, 1]
            panel.show_button.click(); QTest.qWait(50)
            assert all(actor.GetVisibility() for actor in actors())
            panel.opacity.setValue(50); QTest.qWait(50)
            assert all(abs(actor.GetProperty().GetOpacity() - .5) < 1e-6 for actor in actors())
            with patch('part_controls.QColorDialog.getColor', return_value=QColor('#333333')) as color_picker:
                panel.color_button.click()
                assert color_picker.call_count == 1
            QTest.qWait(50)
            assert all(np.allclose(window._style_for(window.ui.tbl_parts, r)['color'], [.2,.2,.2]) for r in (0, 1))
            panel.shading.setCurrentIndex(panel.shading.findData('shaded'))
            panel.set_shading(panel.shading.currentIndex()); QTest.qWait(50)
            assert all(p['last_visible_mode'] == 'shaded' for p in window.slicer_parts)
            window.undo_action(); QTest.qWait(50)
            assert all(p['last_visible_mode'] == 'transparent' for p in window.slicer_parts)
            window.redo_action(); QTest.qWait(50)
            panel.extra_columns.setChecked(True)
            assert browser.stack.currentWidget() is window.ui.tbl_parts
            assert all(not window.ui.tbl_parts.isColumnHidden(c) for c in range(7))
            panel.extra_columns.setChecked(False)
            assert browser.stack.currentWidget() is browser.list
            report['checks'].append('Selection-aware hide/show, colour, shading and transparency update actual actors; Undo/Redo and individual table controls work')
            saved = Path(folder) / 'shared.mrp'; save_project(saved, window.capture_project())
            restored_window.restore_project(load_project(saved))
            assert len(restored_window.slicer_parts) == 2
            assert np.allclose(restored_window._style_for(restored_window.ui.tbl_parts, 0)['color'], [.2,.2,.2])
            save_project(saved, restored_window.capture_project()); window.restore_project(load_project(saved)); QTest.qWait(100)
            assert np.allclose(window._style_for(window.ui.tbl_parts, 0)['color'], [.2,.2,.2])
            report['checks'].append('MRP round-trip between two application windows preserves models and display properties')
            for name, page, plotter, cube in (
                    ('slicer', window.ui.page_slicer, window.ui.slicer_plotter, window.workspace_tools.cube),
                    ('predef', window.ui.page_predef, window.ui.plotter, window.ui.def_cube)):
                window.ui.stack.setCurrentWidget(page); QTest.qWait(100)
                plotter.camera_position = 'iso'; plotter.render(); QTest.qWait(50)
                assert cube.enable_drag and cube.faces
                before = tuple(cube._frame_key)
                plotter.camera.Azimuth(25); plotter.render(); QTest.qWait(50)
                assert tuple(cube._frame_key) != before
                polygon, axis, sign = cube.faces[-1]
                local = polygon.boundingRect().center().toPoint()
                # The average of a convex face's vertices is safely inside it.
                local = QPoint(round(sum(p.x() for p in polygon)/len(polygon)), round(sum(p.y() for p in polygon)/len(polygon)))
                point = cube.geometry().topLeft() + local
                old = np.array(plotter.camera.position)
                QTest.mousePress(plotter, Qt.LeftButton, Qt.NoModifier, point)
                end = point + QPoint(24, 16)
                move = QMouseEvent(QEvent.MouseMove, QPointF(end), QPointF(plotter.mapToGlobal(end)), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
                app.sendEvent(plotter, move); QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, end); QTest.qWait(60)
                assert not np.allclose(old, plotter.camera.position), name
                polygon, axis, sign = cube.faces[-1]
                point = cube.geometry().topLeft() + QPoint(round(sum(p.x() for p in polygon)/len(polygon)), round(sum(p.y() for p in polygon)/len(polygon)))
                QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, point); QTest.qWait(50)
                direction = np.array(plotter.camera.position) - np.array(plotter.camera.focal_point)
                direction /= np.linalg.norm(direction)
                np.testing.assert_allclose(direction, np.eye(3)[axis]*sign, atol=1e-6)
                cube.orient(); plotter.reset_camera(); plotter.render(); QTest.qWait(50)
                window.grab().save(str(output / ('new-' + name + '.png')))
                pixels=plotter.screenshot(str(output / ('new-' + name + '-scene.png')))
                assert pixels.std() > 3, (name, 'blank scene')
                report['checks'].append(name + ': live cube follows camera, drag rotates scene and face click positions camera')
            window.ui.stack.setCurrentWidget(window.ui.page_slicer)
            for width in (1600, 1300, 1000, 800):
                window.resize(width, 1000); QTest.qWait(80)
                assert window.width() == width, (width, window.width())
                sidebar = window.ui.slicer_left_scroll
                assert sidebar.horizontalScrollBar().maximum() == 0, (width, sidebar.widget().minimumSizeHint().width())
                for i in range(window.ui.magics_ribbon.count()):
                    window.ui.magics_ribbon.setCurrentIndex(i); QTest.qWait(10)
                    for button in window.ui.magics_ribbon.widget(i).findChildren(QToolButton):
                        if button.toolButtonStyle() == Qt.ToolButtonTextUnderIcon:
                            assert button.iconSize() == QSize(28, 28)
                window.grab().save(str(output / f'new-width-{width}.png'))
            report['checks'].append('All ribbons accessible at 800/1000/1300/1600, common 28px icons')
            window.resize(1600,1000); window.ui.show_slicer()
            window.ui.magics_ribbon.setCurrentIndex(0)
            panel.set_transparency(0)
            panel.set_color(QColor('#99acaa'))
            panel.shading.setCurrentIndex(panel.shading.findData('shaded'))
            panel.set_shading(panel.shading.currentIndex())
            window.raise_(); window.activateWindow(); QTest.qWait(100)
            window.grab().save(str(output/'new-slicer-final.png'))
            window.screen().grabWindow(window.winId()).save(str(output/'new-slicer-screen.png'))
            window.ui.slicer_left_scroll.grab().save(str(output/'new-project-sidebar.png'))
            for index in (1, 10):
                window.ui.magics_ribbon.setCurrentIndex(index); QTest.qWait(60)
                window.screen().grabWindow(window.winId()).save(str(output/f'new-ribbon-{index}-screen.png'))
            report['status'] = 'passed'
        except Exception:
            import traceback
            report['error'] = traceback.format_exc(); raise
        finally:
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            for w in (window, restored_window): w.dirty=False; w.close(); w.deleteLater()
            app.processEvents()
    print('UI_SMOKE_OK')


if __name__ == '__main__':
    main()
