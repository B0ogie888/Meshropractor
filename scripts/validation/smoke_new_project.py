"""Open the real home dialog, switch its theme live and enter both workspaces."""
from contextlib import ExitStack
from pathlib import Path
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
from PySide6.QtCore import QEvent, QSettings, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import main_window
import app_updater
from ui_base import DialogNewProject
from ui_theme import THEME_COLORS


def main(startup_theme='light'):
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT / 'output/new-project-smoke'; output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        def settings(team, name):
            result = QSettings(str(Path(folder) / (name + '.ini')), QSettings.IniFormat)
            result.setFallbacksEnabled(False)
            if not result.contains('appearance/theme'):
                result.setValue('appearance/theme', startup_theme)
            return result
        patches.enter_context(patch.object(main_window, 'QSettings', settings))
        patches.enter_context(patch.object(app_updater.UpdateController, 'start', lambda *args: None))
        window = main_window.MainWindow(); window.show(); app.processEvents()
        errors = []
        try:
            for target in ('slicer', 'predef', 'cancel'):
                window.ui.stack.setCurrentWidget(window.ui.page_start)
                def interact():
                    dialog = app.activeModalWidget()
                    try:
                        assert isinstance(dialog, DialogNewProject)
                        assert dialog.selected_mode is None
                        # Check the first paint before toggling: a mode change
                        # would otherwise mask a theme-on-construction defect.
                        initial = window.engineering_theme.mode
                        shot = dialog.grab().toImage()
                        shot.save(str(output / (initial + '-first-open.png')))
                        assert shot.pixelColor(4, 4).name() == THEME_COLORS[initial]['panel'], (
                            initial, shot.pixelColor(4, 4).name(), dialog.styleSheet())
                        for mode in ('light', 'dark', 'light'):
                            window.engineering_theme.set_mode(mode); QTest.qWait(70)
                            shot = dialog.grab().toImage()
                            assert shot.pixelColor(4, 4).name() == THEME_COLORS[mode]['panel']
                            for button in (dialog.btn_slicer, dialog.btn_predef):
                                assert button.isEnabled() and button.iconSize().width() == 48
                                assert button.size().width() >= button.minimumSizeHint().width()
                                icon = button.icon().pixmap(48, 48).toImage()
                                colors = {icon.pixelColor(x, y).getRgb()[:3] for x in range(48) for y in range(48)
                                          if icon.pixelColor(x, y).alpha() > 240}
                                expected = QColor(THEME_COLORS[mode]['ink']).getRgb()[:3]
                                assert colors and all(max(abs(a-b) for a, b in zip(color, expected)) <= 1
                                                      for color in colors), (mode, colors)
                            shot.save(str(output / (mode + '.png')))
                        if target == 'cancel':
                            QTest.keyClick(dialog, Qt.Key_Escape)
                        else:
                            button = getattr(dialog, 'btn_' + target)
                            button.setFocus(); QTest.keyClick(button, Qt.Key_Space)
                    except BaseException as exc:
                        errors.append(exc)
                        if dialog: dialog.reject()
                QTimer.singleShot(80, interact)
                window.ui.btn_new_project.click()
                if errors: raise errors[0]
                page = window.ui.page_start if target == 'cancel' else getattr(window.ui, 'page_' + target)
                assert window.ui.stack.currentWidget() is page, target
        finally:
            window.close(); window.deleteLater()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
    print('NEW_PROJECT_OK: both workspaces, cancel, live light/dark theme, icons and keyboard')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', choices=('light', 'dark'), default='light')
    main(parser.parse_args().theme)
