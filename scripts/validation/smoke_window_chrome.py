"""Native captions, consistent ribbon typography, and recoverable side panels."""
from contextlib import ExitStack
from pathlib import Path
import ctypes
import json
import os
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
import trimesh
from PIL import Image, ImageColor
from PySide6.QtCore import Qt, QSettings, QPoint
from PySide6.QtGui import QFont, QFontMetrics, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton, QDialog, QMessageBox
import main_window
import app_updater
from project_store import ProjectState
from display_settings import DisplaySettingsDialog
from ui_theme import THEME_COLORS


def caption_attributes(dialog):
    get = ctypes.windll.dwmapi.DwmGetWindowAttribute
    get.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
    get.restype = ctypes.c_long
    result = {}
    for number in (20, 35, 36):
        value = ctypes.c_uint32()
        hr = get(int(dialog.winId()), number, ctypes.byref(value), ctypes.sizeof(value))
        result[number] = value.value if hr == 0 else None
    return result


def capture_frame(dialog):
    """Print our own HWND including the non-client caption (Qt grabs omit it)."""
    from ctypes import wintypes
    user, gdi = ctypes.windll.user32, ctypes.windll.gdi32
    declarations = ((user.GetWindowRect, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
        (user.GetWindowDC, [wintypes.HWND], wintypes.HDC),
        (user.ReleaseDC, [wintypes.HWND, wintypes.HDC], ctypes.c_int),
        (user.PrintWindow, [wintypes.HWND, wintypes.HDC, wintypes.UINT], wintypes.BOOL),
        (gdi.CreateCompatibleDC, [wintypes.HDC], wintypes.HDC),
        (gdi.CreateCompatibleBitmap, [wintypes.HDC, ctypes.c_int, ctypes.c_int], wintypes.HBITMAP),
        (gdi.SelectObject, [wintypes.HDC, wintypes.HGDIOBJ], wintypes.HGDIOBJ),
        (gdi.DeleteObject, [wintypes.HGDIOBJ], wintypes.BOOL),
        (gdi.DeleteDC, [wintypes.HDC], wintypes.BOOL),
        (gdi.GetDIBits, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                        ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT], ctypes.c_int))
    for function, args, result in declarations: function.argtypes=args; function.restype=result
    hwnd = int(dialog.winId()); rect=wintypes.RECT()
    assert user.GetWindowRect(hwnd, ctypes.byref(rect))
    width, height = rect.right-rect.left, rect.bottom-rect.top
    dc=user.GetWindowDC(hwnd); memory=gdi.CreateCompatibleDC(dc)
    bitmap=gdi.CreateCompatibleBitmap(dc,width,height); previous=gdi.SelectObject(memory,bitmap)
    try:
        assert user.PrintWindow(hwnd,memory,2)
        gdi.SelectObject(memory,previous)
        import struct
        info=ctypes.create_string_buffer(struct.pack('<IiiHHIIiiII',40,width,-height,1,32,0,0,0,0,0,0))
        data=ctypes.create_string_buffer(width*height*4)
        assert gdi.GetDIBits(memory,bitmap,0,height,data,info,0)==height
        return Image.frombytes('RGB',(width,height),data.raw,'raw','BGRX')
    finally:
        gdi.SelectObject(memory,previous); gdi.DeleteObject(bitmap); gdi.DeleteDC(memory); user.ReleaseDC(hwnd,dc)


def drag(grip, dx):
    start = grip.rect().center()
    global_start = grip.mapToGlobal(start)
    QTest.mousePress(grip, Qt.LeftButton, Qt.NoModifier, start)
    # Explicit global coordinates keep this test valid when the panel layout
    # changes under the captured pointer during a drag.
    for fraction in (.25, .5, .75, 1):
        global_point = global_start + QPoint(round(dx*fraction), 0)
        point = grip.mapFromGlobal(global_point)
        event = QMouseEvent(QMouseEvent.MouseMove, point, global_point,
                            Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(grip, event); QTest.qWait(25)
    QTest.mouseRelease(grip, Qt.LeftButton, Qt.NoModifier, grip.mapFromGlobal(global_start+QPoint(dx,0))); QTest.qWait(60)


def main():
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT/'output/window-chrome-smoke'; output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checks=[], captions=[]); errors = []
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
        window = main_window.MainWindow(); ui = window.ui
        window.resize(1600,1000); window.show()
        try:
            window.restore_project(ProjectState(page='slicer', parts=[dict(
                mesh=trimesh.creation.box([20,30,40]), filename='Корпус.stl', platform=None)]))
            ui.parts_browser.select_all(); QTest.qWait(100)
            ui.slicer_plotter.view_isometric(); ui.slicer_plotter.reset_camera()
            for mode in ('dark', 'light'):
                window.engineering_theme.set_mode(mode); QTest.qWait(80)
                for index in range(ui.magics_ribbon.count()):
                    assert ui.magics_ribbon.tabIcon(index).isNull()
                    ui.magics_ribbon.setCurrentIndex(index); QTest.qWait(20)
                    for button in ui.magics_ribbon.widget(index).findChildren(QToolButton):
                        if not button.text(): continue
                        font = button.font()
                        assert font.pixelSize() == 12 and font.weight() == QFont.Normal, (button.text(), font.toString())
                        assert font.family() == 'Segoe UI', font.toString()
                        metrics = QFontMetrics(font)
                        assert max(metrics.horizontalAdvance(line) for line in button.text().split('\n')) < button.width()
                        if button.toolButtonStyle() == Qt.ToolButtonTextUnderIcon:
                            assert 28 + len(button.text().split('\n'))*metrics.lineSpacing() + 8 <= button.height()
                ui.magics_ribbon.setCurrentIndex(0)
                settings_dialog = DisplaySettingsDialog(ui.display_preferences, window.display_settings.apply, window)
                for dialog in (settings_dialog, QDialog(window), QMessageBox(QMessageBox.Information, 'Сведения', 'Проверка темы', parent=window)):
                    dialog.show(); QTest.qWait(90)
                    frame = dialog.frameGeometry()
                    capture = capture_frame(dialog) if sys.platform == 'win32' else dialog.grab()
                    if dialog is settings_dialog: capture.save(str(output/f'settings-{mode}.png'))
                    if sys.platform == 'win32':
                        values = caption_attributes(dialog); report['captions'].append(dict(mode=mode, values=values))
                        assert values[20] == int(mode=='dark'), values
                        if sys.getwindowsversion().build >= 22000:
                            # Caption colours are documented as Set-only DWM
                            # attributes; verify the actual non-client pixels.
                            scale = capture.width/frame.width(); rgb = ImageColor.getrgb(THEME_COLORS[mode]['panel'])
                            count = sum(capture.getpixel((x,y)) == rgb
                                for y in range(round(8*scale),round(24*scale)) for x in range(20,capture.width-20))
                            assert count > 100, (mode, count)
                    if dialog is settings_dialog:
                        opposite = 'light' if mode=='dark' else 'dark'
                        window.engineering_theme.set_mode(opposite); QTest.qWait(80)
                        if sys.platform == 'win32':
                            assert caption_attributes(dialog)[20] == int(opposite=='dark')
                        window.engineering_theme.set_mode(mode)
                    dialog.close(); dialog.deleteLater(); QTest.qWait(30)
                for page, rails in ((ui.page_slicer, ui.slicer_rails), (ui.page_predef, ui.predef_rails)):
                    ui.stack.setCurrentWidget(page); QTest.qWait(100)
                    for side in (0,2):
                        grip, compact = rails.grips[side], rails.compact_grips[side]
                        assert grip.isHidden() and compact.isVisible() and compact.width() == 8
                        was = rails.splitter.sizes()[side]
                        QTest.mouseClick(compact, Qt.LeftButton); QTest.qWait(80)
                        assert not rails.is_open(side) and grip.isVisible() and compact.isHidden()
                        assert rails.splitter.sizes()[1] > 0
                        QTest.keyClick(grip, Qt.Key_Space); QTest.qWait(80)
                        assert rails.is_open(side) and abs(rails.splitter.sizes()[side]-was) < 5
                        assert grip.isHidden() and compact.isVisible()
                        QTest.mouseClick(compact, Qt.LeftButton); QTest.qWait(60)
                        drag(grip, 420 if side==0 else -440)
                        assert rails.is_open(side), (side, rails.splitter.sizes())
                        assert grip.isHidden() and compact.isVisible()
                        drag(compact, -550 if side==0 else 550)
                        assert not rails.is_open(side)
                    label = 'slicer' if page is ui.page_slicer else 'predef'
                    window.screen().grabWindow(window.winId()).save(str(output/f'collapsed-{label}-{mode}.png'))
                    for side in (0,2): rails.set_visible(side, True)
                    QTest.qWait(80)
                    assert rails.splitter.x() == 0 and rails.splitter.width() == rails.container.width()
                ui.show_slicer(); QTest.qWait(80)
                window.screen().grabWindow(window.winId()).save(str(output/f'workspace-{mode}.png'))
            report['checks'].append('All ribbon command fonts match at 12px regular in both themes; labels fit')
            report['checks'].append('Settings, dialog and message-box native captions match theme, including live changes')
            report['checks'].append('Labelled rails only when collapsed; 8px three-dot grips beside open panels; click/keyboard/drag in both workspaces')
            report['checks'].append('All ribbon tab headers are text-only; open panels reserve no outer rail space')
            for width in (800, 1100, 1920):
                window.resize(width,1000); QTest.qWait(100)
                assert window.width() == width, window.minimumSizeHint().width()
                assert all(ui.slicer_rails.grips[i].isHidden() and ui.slicer_rails.compact_grips[i].isVisible() for i in (0,2))
            ui.slicer_rails.set_visible(0, False); ui.slicer_rails.set_visible(2, False)
            ui.predef_rails.set_visible(0, False)
            window.dirty = False; window.close(); app.processEvents()
            restored = main_window.MainWindow(); restored.show(); restored.ui.show_slicer(); QTest.qWait(120)
            assert not restored.ui.slicer_rails.is_open(0) and not restored.ui.slicer_rails.is_open(2)
            assert not restored.ui.predef_rails.is_open(0)
            restored.dirty=False; restored.close(); restored.deleteLater()
            assert not errors, errors
            report['checks'].append('Widths 800/1100/1920 and collapsed-state restoration after restart')
            report['status']='passed'
        except Exception:
            report['status']='failed'; report['error']=traceback.format_exc(); raise
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            window.dirty=False; window.close(); window.deleteLater(); app.processEvents()
    print('WINDOW_CHROME_SMOKE_OK')


if __name__=='__main__': main()
