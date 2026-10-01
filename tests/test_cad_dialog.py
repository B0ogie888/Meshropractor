"""Qt contracts for native STEP import and modeless CAD controls."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import math
from pathlib import Path
import sys
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cad_dialog import CADToolsDialog, cad_icon
from import_dialog import StepImportDialog

APP = QApplication.instance() or QApplication([])


class CADDialogTests(unittest.TestCase):
    def dialog(self, cls, *args, **kwargs):
        dialog = cls(*args, **kwargs)
        def cleanup():
            dialog.close()
            dialog.deleteLater()
            APP.processEvents()
        self.addCleanup(cleanup)
        return dialog

    def test_import_defaults_keep_native_bodies_and_legacy_quality_values(self):
        dialog = self.dialog(StepImportDialog)
        self.assertEqual(dialog.options(), {'native': True, 'split_bodies': True})
        self.assertTrue(dialog.split_bodies.isEnabled())
        linear, angle = dialog.values()
        self.assertEqual(linear, .05)
        self.assertAlmostEqual(angle, .25, places=5)
        dialog.linear.setValue(.0123)
        dialog.angle.setValue(12.5)
        self.assertEqual(dialog.values(), (.0123, math.radians(12.5)))
        self.assertGreater(dialog.linear.minimum(), 0)
        self.assertGreater(dialog.angle.minimum(), 0)
        self.assertLess(dialog.angle.maximum(), 180)

    def test_mesh_only_disables_body_split_but_restores_user_preference(self):
        dialog = self.dialog(StepImportDialog)
        dialog.mesh.setChecked(True)
        self.assertFalse(dialog.split_bodies.isEnabled())
        self.assertEqual(dialog.options(), {'native': False, 'split_bodies': False})
        dialog.native.setChecked(True)
        self.assertTrue(dialog.split_bodies.isEnabled())
        self.assertTrue(dialog.options()['split_bodies'])
        dialog.split_bodies.setChecked(False)
        dialog.mesh.setChecked(True)
        dialog.native.setChecked(True)
        self.assertEqual(dialog.options(), {'native': True, 'split_bodies': False})

    def test_predeformation_forbids_split_without_disabling_native_geometry(self):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        dialog = self.dialog(StepImportDialog, parent, allow_split=False)
        self.assertIs(dialog.parentWidget(), parent)
        self.assertFalse(dialog.split_bodies.isEnabled())
        self.assertEqual(dialog.options(), {'native': True, 'split_bodies': False})
        dialog.mesh.setChecked(True)
        dialog.native.setChecked(True)
        dialog.split_bodies.setChecked(True)
        self.assertFalse(dialog.split_bodies.isEnabled())
        self.assertEqual(dialog.options(), {'native': True, 'split_bodies': False})

    def test_import_buttons_accept_and_cancel(self):
        accepted = self.dialog(StepImportDialog)
        accepted.buttons.button(QDialogButtonBox.Ok).click()
        self.assertEqual(accepted.result(), QDialog.Accepted)
        cancelled = self.dialog(StepImportDialog)
        cancelled.buttons.button(QDialogButtonBox.Cancel).click()
        self.assertEqual(cancelled.result(), QDialog.Rejected)

    def test_cad_dialog_selection_updates_and_multiple_export_remains_available(self):
        parent = QWidget()
        self.addCleanup(parent.deleteLater)
        dialog = self.dialog(CADToolsDialog, parent)
        self.assertIs(dialog.parentWidget(), parent)
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.windowModality(), Qt.NonModal)
        buttons = (dialog.retessellate, dialog.split, dialog.export, dialog.convert)
        self.assertTrue(all(not button.isEnabled() for button in buttons))
        dialog.update_selection('Выбрано 3 CAD-детали. Тел: 5. Граней: 24. Геометрия корректна.', True, False)
        self.assertTrue(dialog.export.isEnabled())
        self.assertTrue(dialog.retessellate.isEnabled())
        self.assertTrue(dialog.convert.isEnabled())
        self.assertFalse(dialog.split.isEnabled())
        self.assertIn('Тел: 5', dialog.summary.text())
        dialog.update_selection('Одна CAD-модель с двумя телами.', True, True)
        self.assertTrue(all(button.isEnabled() for button in buttons))
        dialog.set_info('Тип: цилиндр\nПлощадь: 20 мм²\nРадиус: 2 мм\nОбъём: 30 мм³')
        dialog.update_selection('Выбрана треугольная сетка.', False, True)
        self.assertTrue(all(not button.isEnabled() for button in buttons))
        self.assertEqual(dialog.details.toPlainText(), '')

    def test_cad_actions_emit_once_without_accepting_or_closing_modeless_dialog(self):
        dialog = self.dialog(CADToolsDialog)
        dialog.update_selection('CAD: 2 тела', True, True)
        dialog.show()
        events = (dialog.retessellate_requested, dialog.split_requested,
                  dialog.export_requested, dialog.convert_requested)
        spies = [QSignalSpy(event) for event in events]
        for index, button in enumerate((dialog.retessellate, dialog.split, dialog.export, dialog.convert)):
            button.click()
            self.assertEqual([spy.count() for spy in spies], [int(i <= index) for i in range(4)])
            self.assertEqual(spies[index].at(0), [])
            self.assertTrue(dialog.isVisible())
        dialog.buttons.button(QDialogButtonBox.Close).click()
        self.assertFalse(dialog.isVisible())

    def test_cad_information_is_readonly_plain_text_and_icon_has_multiple_resolutions(self):
        dialog = self.dialog(CADToolsDialog)
        info = '<body>Деталь & корпус</body>\nТел: 2\nГрань №7: цилиндр\nПлощадь: 20 мм²\nРадиус: 2 мм\nОбъём: 30 мм³'
        dialog.set_info(info)
        self.assertEqual(dialog.details.toPlainText(), info)
        self.assertTrue(dialog.details.isReadOnly())
        self.assertEqual(dialog.summary.textFormat(), Qt.PlainText)
        icon = cad_icon(40)
        self.assertFalse(icon.isNull())
        self.assertTrue({24, 32, 40, 48, 64, 96}.issubset({size.width() for size in icon.availableSizes()}))
        pixmap = icon.pixmap(40, 40)
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.toImage().pixelColor(20, 20).alpha(), 0)


if __name__ == '__main__':
    unittest.main()
