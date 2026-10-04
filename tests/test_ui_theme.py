import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QWidget, QDialog, QLineEdit, QPushButton
from ui_theme import EngineeringTheme, translate_style, theme_style, PANEL, INK, ACCENT

APP = QApplication.instance() or QApplication([])


class EngineeringThemeTests(unittest.TestCase):
    def test_dialog_first_paint_uses_saved_dark_theme_without_a_toggle(self):
        window = QWidget()
        window.settings = SimpleNamespace(value=lambda key, default: 'dark')
        theme = EngineeringTheme(window)
        try:
            for _ in range(2):
                dialog = QDialog(window); dialog.setObjectName('NewProjectDialog')
                dialog.setStyleSheet('QDialog#NewProjectDialog {background: #fafbf8; color: #252b2b;}')
                dialog.resize(120, 80); dialog.show(); APP.processEvents()
                self.assertEqual(dialog.grab().toImage().pixelColor(4, 4).name(), '#282f2f')
                dialog.close(); dialog.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        finally:
            APP.removeEventFilter(theme)
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_dark_chrome_keeps_contrast_on_accent_and_selectors(self):
        source = 'QToolButton:checked {background:#e5d943;color:#252b2b;} QDialog {background:#fafbf8;color:#252b2b;}'
        dark = theme_style(source, 'dark')
        self.assertIn('QToolButton:checked {background: #e5d943;color: #252b2b;}', dark)
        self.assertIn('QDialog {background: #282f2f;color: #e6ede7;}', dark)
        self.assertEqual(theme_style(source, 'light'), source)

    def test_repeated_modes_use_original_styles_and_leave_swatches_intact(self):
        window = QWidget(); button = QPushButton('Action', window)
        button.setStyleSheet('QPushButton {background:#333;color:white;}')
        swatch = QPushButton(window); swatch.setFixedWidth(24); swatch.setStyleSheet('background:#333;')
        theme = EngineeringTheme(window)
        try:
            light = button.styleSheet()
            for mode in ('dark', 'light', 'dark', 'light'):
                theme.mode = mode; theme.style_widget(button); theme.style_widget(swatch)
                self.assertEqual(button.styleSheet(), theme_style(light, mode))
                self.assertEqual(swatch.styleSheet(), 'background:#333;')
            # Runtime styles from an existing controller also get the active theme.
            theme.mode = 'dark'; button.setStyleSheet('QPushButton {color:white;background:#444;}')
            theme.style_widget(button)
            self.assertIn('#e6ede7', button.styleSheet()); self.assertIn('#282f2f', button.styleSheet())
        finally:
            APP.removeEventFilter(theme); window.deleteLater(); APP.processEvents()

    def test_styles_preserve_pseudo_selectors_and_are_idempotent(self):
        style = 'QMenu::item:selected, QPushButton:checked {background: #b31b1b; color: white; padding: 4px;} QDialog {background:#333;}'
        result = translate_style(style)
        self.assertIn('QMenu::item:selected, QPushButton:checked', result)
        self.assertIn('background: ' + ACCENT, result)
        self.assertIn('color: ' + INK, result)
        self.assertIn('padding: 4px', result)
        self.assertIn('background: ' + PANEL, result)
        self.assertEqual(translate_style(result), result)
        self.assertIn('background: ' + ACCENT,
                      translate_style('QToolButton:checked {background:#244650;}'))

    def test_new_children_are_styled_without_touching_classic_or_data(self):
        classic = QWidget(); new = QWidget()
        classic.setStyleSheet('background:#333;color:white;')
        theme = EngineeringTheme(new)
        try:
            dialog = QDialog(new)
            # Real modeling dialogs expose a QLineEdit named `text`.
            dialog.text = QLineEdit(dialog)
            dialog.setStyleSheet('QDialog{background:#333;color:white;}')
            theme.style_widget(dialog)
            self.assertIn(PANEL, dialog.styleSheet())
            swatch = QPushButton(dialog)
            swatch.setProperty('preserveThemeColors', True)
            swatch.setStyleSheet('background:#333;')
            theme.style_widget(swatch)
            self.assertEqual(swatch.styleSheet(), 'background:#333;')
            self.assertEqual(classic.styleSheet(), 'background:#333;color:white;')
            self.assertFalse(theme.belongs(classic))
            self.assertTrue(theme.belongs(dialog.text))
        finally:
            APP.removeEventFilter(theme)
            new.close(); classic.close()
            new.deleteLater(); classic.deleteLater(); APP.processEvents()


if __name__ == '__main__':
    unittest.main()
