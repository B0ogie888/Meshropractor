"""Visible theme repaint and compact command columns in the native Qt/VTK app."""
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
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
import trimesh
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtGui import QPalette, QFontMetrics, QFont, QPixmap, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton, QFrame, QStyle, QStyleOptionToolButton
import main_window
import app_updater
from project_store import ProjectState
from ui_theme import THEME_COLORS


def command_image(button, state):
    """Exercise the actual widget's QSS for each interaction state."""
    option = QStyleOptionToolButton(); button.initStyleOption(option)
    option.state &= ~(QStyle.State_MouseOver | QStyle.State_Sunken | QStyle.State_On | QStyle.State_HasFocus)
    option.state |= QStyle.State_Enabled | state
    pixmap = QPixmap(button.size()); pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    button.style().drawComplexControl(QStyle.CC_ToolButton, option, painter, button)
    painter.end()
    return pixmap.toImage()


def main():
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT / 'output/theme-ribbons-smoke'; output.mkdir(parents=True, exist_ok=True)
    errors = []; report = dict(status='running', checks=[], pixels=[])
    def hook(*args):
        errors.append(''.join(traceback.format_exception(*args))); traceback.print_exception(*args)
    sys.excepthook = hook
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        def settings(team, name):
            result = QSettings(str(Path(folder) / (name + '.ini')), QSettings.IniFormat)
            result.setFallbacksEnabled(False); return result
        patches.enter_context(patch.object(main_window, 'QSettings', settings))
        for method in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, method, lambda *args: None))
        window = main_window.MainWindow(); ui = window.ui
        window.resize(1600, 1000); window.show()
        try:
            window.restore_project(ProjectState(page='slicer', parts=[dict(
                mesh=trimesh.creation.box([20, 30, 40]), filename='Корпус.stl', platform=None)]))
            ui.parts_browser.select_all(); QTest.qWait(200)
            ui.slicer_plotter.view_isometric(); ui.slicer_plotter.reset_camera(); ui.slicer_plotter.render()
            for mode in ('dark', 'light', 'dark', 'light'):
                ui.theme_button.click(); QTest.qWait(150)
                assert window.engineering_theme.mode == mode
                assert ui.stack.currentWidget() is ui.page_slicer
                # Capture the already painted native window. QWidget.grab() would
                # force a fresh paint and could hide this delayed-repaint defect.
                screenshot = window.screen().grabWindow(window.winId())
                screenshot.save(str(output / f'slicer-{mode}.png'))
                image = screenshot.toImage(); scale = image.width() / window.width()
                samples = {}
                for name, widget, point in (
                    ('left', ui.slicer_left_scroll.widget(), QPoint(5, 50)),
                    ('right', ui.part_inspector, QPoint(5, 100)),
                ):
                    pixel = widget.mapTo(window, point)
                    samples[name] = image.pixelColor(round(pixel.x()*scale), round(pixel.y()*scale)).name()
                report['pixels'].append(dict(mode=mode, **samples))
                if samples['right'] != THEME_COLORS[mode]['panel']:
                    widget = ui.part_inspector; chain = []
                    while widget:
                        chain.append(dict(name=widget.objectName(), cls=type(widget).__name__,
                            auto=widget.autoFillBackground(), color=widget.palette().color(QPalette.Window).name(),
                            qss=widget.styleSheet()[:200]))
                        widget = widget.parentWidget()
                    report['chain'] = chain
                assert all(value == THEME_COLORS[mode]['panel'] for value in samples.values()), samples
            report['checks'].append('Visible sidebars repaint on every theme toggle without hiding or changing workspaces')

            report['ribbons'] = []
            for mode in ('light', 'dark'):
                window.engineering_theme.set_mode(mode)
                for name, button in ui.texture_buttons.items():
                    icon = button.icon().pixmap(48, 48).toImage()
                    assert not icon.isNull(), name
                    # The pictogram is line art with negative space; no opaque tile.
                    solid = sum(icon.pixelColor(x, y).alpha() > 240
                                for x in range(icon.width()) for y in range(icon.height()))
                    assert 0 < solid < icon.width() * icon.height() * .45, (mode, name, solid)
                for name in ('build_risk', 'triangle_colors', 'smooth'):
                    icon = ui.display_buttons[name].icon().pixmap(48, 48).toImage()
                    solid = sum(icon.pixelColor(x, y).alpha() > 240
                                for x in range(icon.width()) for y in range(icon.height()))
                    assert 0 < solid < icon.width() * icon.height() * .30, (mode, name, solid)
                window.showMaximized(); QTest.qWait(100)
                command_tops = set()
                for index in range(ui.magics_ribbon.count()):
                    ui.magics_ribbon.setCurrentIndex(index); QTest.qWait(30)
                    scroll = ui.magics_ribbon.currentWidget(); panel = scroll.widget()
                    title = ui.magics_ribbon.tabText(index)
                    buttons = [b for b in panel.findChildren(QToolButton) if b.text()]
                    for button in buttons:
                        assert button.font().pixelSize() == 12 and button.font().weight() == QFont.Normal
                        if button.toolButtonStyle() == Qt.ToolButtonTextUnderIcon:
                            assert button.iconSize().width() == 28
                            command_tops.add(button.mapTo(ui.magics_ribbon, QPoint()).y())
                        metrics = QFontMetrics(button.font())
                        assert max(metrics.horizontalAdvance(line) for line in button.text().split('\n')) + 8 <= button.width()
                        for state in (QStyle.State_MouseOver, QStyle.State_Sunken, QStyle.State_On):
                            image = command_image(button, state)
                            assert image.pixelColor(0, image.height()//2).name() == '#e5d943', (title, button.text(), state)
                            fill = ('#f2efc6' if mode == 'light' else '#424a32') if state == QStyle.State_MouseOver else '#e5d943'
                            assert image.pixelColor(4, 4).name() == fill, (title, button.text(), state)
                    for divider in panel.findChildren(QFrame):
                        if divider.frameShape() != QFrame.VLine and divider.width() != 1: continue
                        line_x = divider.mapTo(panel, QPoint()).x()
                        for button in buttons:
                            start = button.mapTo(panel, QPoint()).x()
                            assert not start <= line_x < start + button.width(), (title, button.text())
                    report['ribbons'].append(dict(theme=mode, title=title, commands=len(buttons),
                        minimum_width=panel.minimumSizeHint().width(), viewport=scroll.viewport().width()))
                    panel.grab().save(str(output / f'ribbon-{index:02d}-{mode}.png'))
                    if title == 'ТЕКСТУРЫ':
                        ui.magics_ribbon.grab().save(str(output / f'textures-{mode}.png'))
                        ui.title_bar.grab().save(str(output / f'titlebar-{mode}.png'))
                    if title == 'ИНСТРУМЕНТЫ':
                        assert 'CAD / STEP' not in ui.tools_buttons
                        assert scroll.horizontalScrollBar().maximum() == 0
                        window.screen().grabWindow(window.winId()).save(str(output/f'tools-full-{mode}.png'))
                    if index == 0:
                        window.screen().grabWindow(window.winId()).save(str(output/f'main-full-{mode}.png'))
                assert len(command_tops) == 1, command_tops
                # The unused header area has the same surface as the ribbon.
                screenshot = window.screen().grabWindow(window.winId()).toImage()
                scale = screenshot.width()/window.width()
                tabs = ui.magics_ribbon
                point = tabs.mapTo(window, QPoint(tabs.width()-30, tabs.tabBar().height()//2))
                header_color = screenshot.pixelColor(round(point.x()*scale), round(point.y()*scale)).name()
                assert header_color == THEME_COLORS[mode]['panel'], (mode, header_color, point)
                window.showNormal(); QTest.qWait(40)
            report['checks'].append('All tabs: common yellow hover/pressed/checked states in both themes; original 12px font and 28px icons; no divider overlap')
            report['checks'].append('Tools fit the maximized 2560px screen without horizontal scrolling')
            report['checks'].append('All command rows share one top edge; no CAD/STEP ribbon command; unused header matches panel in both themes')
            report['checks'].append('All 15 texture pictograms retain transparent negative space in both themes')
            report['checks'].append('Display warning, triangle colours and smooth sphere retain internal details in both themes; all ribbons captured')

            groups = (
                ('ИНСТРУМЕНТЫ', window.model_tools, ui.model_tool_buttons, (('extrude', 'offset', 'round_offset'),)),
                ('ИСПРАВЛЕНИЕ', window.repair_tools, ui.repair_buttons,
                 (('unify', 'split', 'remove_small'), ('slivers', 'duplicates', 'overlaps'))),
            )
            for mode in ('light', 'dark'):
                if window.engineering_theme.mode != mode: ui.theme_button.click()
                for width in (1600, 800):
                    window.resize(width, 1000); QTest.qWait(80)
                    for title, controller, buttons, columns in groups:
                        tab = next(i for i in range(ui.magics_ribbon.count()) if ui.magics_ribbon.tabText(i) == title)
                        ui.magics_ribbon.setCurrentIndex(tab); QTest.qWait(60)
                        scroll = ui.magics_ribbon.currentWidget()
                        for operations in columns:
                            column = buttons[operations[0]].parentWidget()
                            assert column.property('ribbonCompactColumn')
                            previous = None
                            for operation in operations:
                                button = buttons[operation]
                                assert button.parentWidget() is column
                                assert button.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
                                assert button.iconSize().width() == 16 and not button.icon().isNull()
                                assert '\n' not in button.text()
                                assert QFontMetrics(button.font()).horizontalAdvance(button.text()) + 32 <= button.width()
                                if previous:
                                    assert button.x() == previous.x() and button.y() > previous.geometry().bottom()
                                previous = button
                            scroll.ensureWidgetVisible(column); QTest.qWait(50)
                            for operation in operations:
                                button = buttons[operation]
                                top_left = button.mapTo(scroll.viewport(), QPoint())
                                assert scroll.viewport().rect().contains(top_left)
                                assert scroll.viewport().rect().contains(top_left + button.rect().bottomRight())
                                with patch.object(controller, 'open') as action:
                                    QTest.mouseClick(button, Qt.LeftButton)
                                    action.assert_called_once_with(operation)
                        if width == 1600:
                            name = 'tools' if title == 'ИНСТРУМЕНТЫ' else 'repair'
                            window.screen().grabWindow(window.winId()).save(str(output / f'{name}-{mode}.png'))
            assert ui.model_tool_buttons['round_offset'].text() == 'Заокругленное смещение'
            report['checks'].append('Three compact columns: labels, icons, no overlap/clipping and all nine commands clickable at 800/1600 in both themes')
            assert not errors, errors
            report['status'] = 'passed'
        except Exception:
            report['status'] = 'failed'; report['error'] = traceback.format_exc(); raise
        finally:
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            window.dirty = False; window.close(); window.deleteLater(); app.processEvents()
    print('THEME_RIBBONS_SMOKE_OK')


if __name__ == '__main__': main()
