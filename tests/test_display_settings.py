import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from display_settings import (DisplayPreferences, DisplaySettingsController, DisplaySettingsDialog,
                              apply_to_plotter, load_preferences, save_preferences)
from settings_icons import SETTINGS_COMMANDS

APP = QApplication.instance() or QApplication([])


class DisplaySettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = QSettings(str(Path(self.directory.name) / 'settings.ini'), QSettings.IniFormat)

    def test_preferences_survive_restart_and_invalid_values_fall_back(self):
        chosen = DisplayPreferences('#142536', 'fxaa', 4, False)
        save_preferences(self.settings, chosen)
        restarted = QSettings(self.settings.fileName(), QSettings.IniFormat)
        self.assertEqual(load_preferences(restarted), chosen)
        restarted.setValue('display/preferences', json.dumps(dict(background='bad colour', anti_aliasing='ssaa',
                                                                 samples=-4, interactive_edges='false')))
        self.assertEqual(load_preferences(restarted), DisplayPreferences())
        restarted.setValue('display/preferences', '[invalid json')
        self.assertEqual(load_preferences(restarted), DisplayPreferences())

    def test_rendering_preferences_leave_geometry_and_camera_untouched(self):
        model = object()
        camera = object()
        plotter = SimpleNamespace(_viewport_performance=Mock(), set_background=Mock(), render=Mock(),
                                  actors={'part': model}, camera=camera)
        chosen = DisplayPreferences('#123456', 'msaa', 8, False)
        apply_to_plotter(plotter, chosen)
        plotter.set_background.assert_called_once_with('#123456')
        plotter._viewport_performance.configure.assert_called_once_with(
            anti_aliasing='msaa', samples=8, interactive_edges=False)
        plotter.render.assert_called_once_with()
        self.assertIs(plotter.camera, camera)
        self.assertIs(plotter.actors['part'], model)
        apply_to_plotter(None, chosen)  # lazy scene has not been opened yet

    def test_dialog_cancel_does_not_change_preferences_and_apply_is_explicit(self):
        applied = []
        dialog = DisplaySettingsDialog(DisplayPreferences(), applied.append)
        self.addCleanup(dialog.deleteLater)
        dialog._set_background('#273849')
        dialog.antialiasing.setCurrentIndex(3)
        dialog.interactive_edges.setChecked(False)
        dialog.reject()
        self.assertEqual(applied, [])
        dialog.buttons.button(QDialogButtonBox.Apply).click()
        self.assertEqual(applied, [DisplayPreferences('#273849', 'msaa', 8, False)])
        dialog.buttons.button(QDialogButtonBox.RestoreDefaults).click()
        self.assertEqual(dialog.preferences(), DisplayPreferences())
        self.assertEqual(len(applied), 1)
        dialog.accept()
        self.assertEqual(applied[-1], DisplayPreferences())

    def test_settings_ribbon_commands_have_icons_and_real_handlers(self):
        from Meshropractor import MainWindow
        update_calls = []
        with patch('app_updater.UpdateController.manual_check', lambda controller: update_calls.append(controller)):
            window = MainWindow()
        try:
            self.assertEqual(window.ui.display_preferences, load_preferences(window.settings))
            window.settings = self.settings
            for name in SETTINGS_COMMANDS:
                self.assertTrue(window.ui.ribbon_btns[name].isEnabled(), name)
                self.assertFalse(window.ui.ribbon_btns[name].icon().isNull(), name)
            self.assertNotIn('Язык', window.ui.ribbon_btns)
            settings_calls = []
            with patch('display_settings.DisplaySettingsDialog.exec',
                       lambda dialog: settings_calls.append(dialog.preferences()) or 0):
                window.ui.ribbon_btns['Параметры'].click()
                self.assertEqual(len(settings_calls), 1)
            texts = []
            with patch.object(window.display_settings, '_show_text', lambda title, html: texts.append((title, html))):
                for name in ('Горячие клавиши', 'Справка', 'О программе'):
                    window.ui.ribbon_btns[name].click()
                self.assertEqual(len(texts), 3)
                self.assertIn('Ctrl+S', texts[0][1])
            window.ui.ribbon_btns['Проверить обновления'].click()
            self.assertEqual(update_calls, [window.updater])
            plotter = SimpleNamespace(set_background=Mock(), render=Mock(), _viewport_performance=Mock())
            window.ui.slicer_plotter = plotter
            preferences = DisplayPreferences('#778899', 'none', 4, False)
            window.display_settings.apply(preferences)
            self.assertEqual(load_preferences(self.settings), preferences)
            self.assertEqual(window.ui.display_preferences, preferences)
            plotter.set_background.assert_called_once_with('#778899')
            # The other scene is still lazy and will use the same stored values.
            self.assertIsNone(window.ui.plotter)
            window.ui.slicer_plotter = None
        finally:
            window.dirty = False
            window.close()
            window.deleteLater()
            APP.processEvents()


if __name__ == '__main__':
    unittest.main()
