"""Offline updater UI and project-save integration tests; no installer is run."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import ctypes
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from app_updater import UpdateController, launch_installer
from update_backend import ReleaseUpdate

APP = QApplication.instance() or QApplication([])


class InstallerLauncherTests(unittest.TestCase):
    """Replace the entire Windows DLL loader: these tests cannot launch a file."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.installer = Path(self.temp.name) / 'обновление программы.exe'
        self.installer.write_bytes(b'not an executable')
        self.shell_execute = Mock(return_value=33)
        windll = SimpleNamespace(shell32=SimpleNamespace(ShellExecuteW=self.shell_execute))
        mock_loader = patch('app_updater.ctypes.windll', windll, create=True)
        mock_loader.start()
        self.addCleanup(mock_loader.stop)
        platform = patch('app_updater.sys.platform', 'win32')
        platform.start()
        self.addCleanup(platform.stop)

    def test_unicode_parameters_and_pointer_sized_window_handle(self):
        handle = (1 << 48) + 123 if ctypes.sizeof(ctypes.c_void_p) == 8 else (1 << 30) + 123
        launch_installer(self.installer, handle)
        self.shell_execute.assert_called_once_with(
            handle, 'open', str(self.installer.resolve()),
            '/SP- /SILENT /NORESTART /UPDATE=1', str(self.installer.parent.resolve()), 1)
        self.assertEqual(self.shell_execute.argtypes,
                         [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                          ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int])
        self.assertIs(self.shell_execute.restype, ctypes.c_void_p)
        self.assertEqual(self.shell_execute.argtypes[0](handle).value, handle)
        for value in self.shell_execute.call_args.args[1:5]:
            self.assertIsInstance(value, str)

    def test_windows_failure_codes_raise_and_success_codes_are_accepted(self):
        for code in (None, 0, 2, 5, 31, 32):
            with self.subTest(code=code):
                self.shell_execute.return_value = code
                with self.assertRaises(OSError):
                    launch_installer(self.installer)
        for code in (33, 256, (1 << 40) + 33):
            with self.subTest(code=code):
                self.shell_execute.return_value = code
                launch_installer(self.installer)

    def test_missing_file_and_non_executable_never_reach_shell(self):
        text_file = self.installer.with_suffix('.txt')
        text_file.write_text('test', encoding='utf-8')
        for path in (self.installer.with_name('missing.exe'), text_file):
            with self.subTest(path=path), self.assertRaises(OSError):
                launch_installer(path)
        self.shell_execute.assert_not_called()


class UpdaterWindow(QMainWindow):
    def __init__(self, directory):
        super().__init__()
        self.ui = SimpleNamespace(btn_check_updates=QPushButton(self), status_label=QLabel(self))
        self.log = Mock()
        self._confirm_discard = Mock(return_value=True)
        self._job = self._transform_session = None
        self.closed = False
        self.updater = UpdateController(self, directory=directory)

    def closeEvent(self, event):
        if not self.updater.allow_close():
            event.ignore()
            return
        if not getattr(self, '_update_exit', False) and not self._confirm_discard(self.close):
            event.ignore()
            return
        self.closed = True
        self.updater.on_closed()
        event.accept()


class UpdaterTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.release = ReleaseUpdate('99.0.0', 'v99.0.0', 'Meshropractor-Setup-99.0.0-x64.exe',
                                     'https://github.com/B0ogie888/Meshropractor/releases/download/v99.0.0/installer.exe',
                                     4 * 1024**3, None,
                                     'https://github.com/B0ogie888/Meshropractor/releases/tag/v99.0.0')
        self.installer = self.directory / self.release.asset_name
        # This intentionally is not executable. Backend and launch are always mocked.
        self.installer.write_bytes(b'not a real installer')
        for name in ('check_for_update', 'download_release', 'verify_installer', 'launch_installer'):
            mocked = patch('app_updater.' + name)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)
        self.check_for_update.return_value = None
        self.download_release.return_value = self.installer
        self.verify_installer.return_value = self.installer
        for name in ('warning', 'information'):
            mocked = patch('app_updater.QMessageBox.' + name)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)
        self.gates = []

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return
            time.sleep(.005)
        self.fail('Qt operation did not finish within the timeout')

    def wait_worker(self):
        self.wait_for(lambda: self.updater.worker is None)

    def hold_download(self):
        gate = threading.Event()
        self.gates.append(gate)

        def download(update, directory, *, progress, cancelled):
            progress(3 * 1024**3, 4 * 1024**3)
            while not gate.wait(.005):
                if cancelled():
                    raise InterruptedError()
            if cancelled():
                raise InterruptedError()
            return self.installer

        self.download_release.side_effect = download
        return gate

    def cleanup_window(self):
        for gate in self.gates:
            gate.set()
        self.updater.start_timer.stop()
        self.updater.offer_timer.stop()
        self.updater.pending_offer = None
        if self.updater.worker:
            self.updater.cancel()
            self.wait_worker()
        if getattr(self.window, '_job', None):
            self.window._job.requestInterruption()
            self.wait_for(lambda: self.window._job is None)
        if hasattr(self.window, 'dirty'):
            self.window.dirty = False
        self.window._confirm_discard = Mock(return_value=True)
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()


class UpdateControllerTests(UpdaterTestBase):
    def setUp(self):
        super().setUp()
        self.window = UpdaterWindow(self.directory)
        self.updater = self.window.updater
        self.updater._ask = Mock(return_value=False)
        self.addCleanup(self.cleanup_window)
        self.window.show()
        APP.processEvents()

    def ready_to_install(self):
        self.updater.available = self.release
        self.updater.ready = self.installer

    def test_start_schedules_nonblocking_check(self):
        self.updater.start()
        self.assertTrue(self.updater.start_timer.isActive())
        self.check_for_update.assert_not_called()
        self.updater.start_timer.stop()
        self.updater.start_timer.timeout.emit()
        self.wait_worker()
        self.check_for_update.assert_called_once()

    def test_background_check_has_no_popup_without_update(self):
        called_from = []
        self.check_for_update.side_effect = lambda *args, **kw: called_from.append(QThread.currentThread())
        self.updater.check()
        self.wait_worker()
        self.assertNotEqual(called_from[0], APP.thread())
        self.assertIsNone(self.updater.pending_offer)
        self.updater._ask.assert_not_called()
        self.information.assert_not_called()
        self.warning.assert_not_called()

    def test_background_network_error_only_logs_and_restores_button(self):
        self.check_for_update.side_effect = OSError('offline test')
        self.updater.check()
        self.wait_worker()
        self.assertIn('offline test', self.window.log.call_args.args[0])
        self.assertTrue(self.window.ui.btn_check_updates.isEnabled())
        self.assertIsNone(self.updater.pending_offer)
        self.warning.assert_not_called()

    def test_manual_error_is_reported(self):
        self.check_for_update.side_effect = OSError('offline test')
        self.window.ui.btn_check_updates.click()
        self.wait_worker()
        self.updater._deliver_offer()
        self.warning.assert_called_once()

    def test_decline_download_never_starts_download(self):
        self.check_for_update.return_value = self.release
        self.updater.check()
        self.wait_worker()
        self.updater._deliver_offer()
        self.updater._ask.assert_called_once()
        self.download_release.assert_not_called()
        self.launch_installer.assert_not_called()

    def test_offer_waits_until_project_operation_finishes(self):
        self.check_for_update.return_value = self.release
        self.updater.check()
        self.wait_worker()
        self.window._job = object()
        self.updater._deliver_offer()
        self.updater._ask.assert_not_called()
        self.assertIsNotNone(self.updater.pending_offer)
        self.window._job = None
        self.updater._deliver_offer()
        self.updater._ask.assert_called_once()

    def test_progress_handles_more_than_two_gigabytes_and_cancel(self):
        self.hold_download()
        self.updater.available = self.release
        self.updater.start_download()
        self.wait_for(lambda: self.updater.progress_dialog.value() == 750)
        self.assertIn('3.00', self.updater.progress_dialog.labelText())
        self.assertIn('75%', self.updater.progress_dialog.labelText())
        self.updater.progress_dialog.canceled.emit()
        self.wait_worker()
        self.assertIsNone(self.updater.progress_dialog)
        self.assertIsNone(self.updater.ready)
        self.assertIsNone(self.updater.pending_offer)
        self.verify_installer.assert_not_called()
        self.launch_installer.assert_not_called()

    def test_install_offer_only_after_worker_has_finished_and_decline_retains_file(self):
        gate = self.hold_download()
        self.updater.available = self.release
        self.updater.start_download()
        self.wait_for(lambda: self.updater.progress_dialog.value() == 750)
        # QThread.result precedes QThread.finished, including with a nested UI loop.
        self.updater.worker.result.emit(self.installer)
        APP.processEvents()
        self.assertIsNotNone(self.updater.worker)
        self.assertIsNone(self.updater.pending_offer)
        self.updater._ask.assert_not_called()
        gate.set()
        self.wait_worker()
        self.updater._ask.side_effect = lambda *args: self.assertIsNone(self.updater.worker) or False
        self.updater._deliver_offer()
        self.assertEqual(self.updater.ready, self.installer)
        self.assertTrue(self.installer.exists())
        self.assertIn('Установить', self.window.ui.btn_check_updates.text())
        self.verify_installer.assert_not_called()
        self.launch_installer.assert_not_called()
        self.window.ui.btn_check_updates.click()
        self.updater._deliver_offer()
        self.assertEqual(self.updater._ask.call_count, 2)
        self.check_for_update.assert_not_called()

    def test_cancel_unsaved_changes_prevents_verification_and_launch(self):
        self.ready_to_install()
        self.updater._ask.return_value = True
        self.window._confirm_discard.return_value = False
        self.updater.offer_install()
        self.window._confirm_discard.assert_called_once()
        self.verify_installer.assert_not_called()
        self.launch_installer.assert_not_called()
        self.assertFalse(self.window.closed)

    def test_verification_failure_keeps_window_open(self):
        self.ready_to_install()
        self.verify_installer.side_effect = ValueError('Checksum changed')
        self.updater.install_now()
        self.wait_worker()
        self.updater._deliver_offer()
        self.warning.assert_called_once()
        self.launch_installer.assert_not_called()
        self.assertFalse(self.window.closed)
        self.assertTrue(self.window.isVisible())
        self.assertIsNone(self.updater.ready)
        self.check_for_update.return_value = self.release
        self.window.ui.btn_check_updates.click()
        self.wait_worker()
        self.updater._deliver_offer()
        self.assertEqual(self.updater._ask.call_args.args[2], 'Скачать')

    def test_new_check_clears_outdated_pending_offer(self):
        self.check_for_update.return_value = self.release
        self.updater.check()
        self.wait_worker()
        self.assertIsNotNone(self.updater.pending_offer)
        self.check_for_update.return_value = None
        self.updater.check()
        self.wait_worker()
        self.updater._deliver_offer()
        self.updater._ask.assert_not_called()
        self.assertIsNone(self.updater.pending_offer)

    def test_closed_controller_does_not_start_more_operations(self):
        self.ready_to_install()
        self.window.close()
        self.updater.start()
        self.updater.check()
        self.updater.manual_check()
        self.updater.start_download()
        self.updater.install_now()
        APP.processEvents()
        self.assertFalse(self.updater.start_timer.isActive())
        self.assertIsNone(self.updater.worker)
        self.assertIsNone(self.updater.progress_dialog)
        self.check_for_update.assert_not_called()
        self.download_release.assert_not_called()
        self.verify_installer.assert_not_called()
        self.launch_installer.assert_not_called()

    def test_launch_failure_keeps_window_open(self):
        self.ready_to_install()
        self.launch_installer.side_effect = OSError('UAC rejected')
        self.updater.install_now()
        self.wait_worker()
        self.warning.assert_called_once()
        self.assertFalse(getattr(self.window, '_update_exit', False))
        self.assertTrue(self.window.isVisible())

    def test_successful_launch_closes_only_after_verification(self):
        self.ready_to_install()
        calls = []
        self.verify_installer.side_effect = lambda *a, **kw: calls.append('verified')
        self.launch_installer.side_effect = lambda *a: calls.append('launched')
        self.updater.install_now()
        self.wait_worker()
        self.assertEqual(calls, ['verified', 'launched'])
        self.assertTrue(self.window.closed)
        self.assertTrue(self.updater.closed)
        self.assertTrue(self.window._update_exit)
        self.window._confirm_discard.assert_called_once()

    def test_close_waits_for_download_thread_before_accepting_close(self):
        self.hold_download()
        self.updater.available = self.release
        self.updater.start_download()
        self.wait_for(lambda: self.updater.progress_dialog.value() == 750)
        self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(self.updater.closing)
        self.assertFalse(self.window.closed)
        self.wait_for(lambda: self.window.closed)
        self.assertIsNone(self.updater.worker)
        self.assertTrue(self.updater.closed)
        self.launch_installer.assert_not_called()


class UpdateProjectSaveTests(UpdaterTestBase):
    """Exercise the real project save/discard flow and its asynchronous continuation."""
    def setUp(self):
        super().setUp()
        from Meshropractor import MainWindow
        self.window = MainWindow()
        self.window.add_to_recent = Mock()
        self.updater = self.window.updater
        self.updater.directory = self.directory
        self.updater.available = self.release
        self.updater.ready = self.installer
        self.updater._ask = Mock(return_value=True)
        # Real project edits are necessary: the save worker flushes history and
        # recomputes dirty from the snapshot, rather than trusting a manual flag.
        self.window.ui.sb_factor.setValue(self.window.ui.sb_factor.value() + .2)
        self.window.flush_history()
        self.assertTrue(self.window.dirty)
        self.addCleanup(self.cleanup_window)

    def test_real_discard_prompt_cancel_keeps_project_and_installer(self):
        with patch('project_controller.QMessageBox.question', return_value=QMessageBox.Cancel):
            self.updater.offer_install()
        self.assertTrue(self.window.dirty)
        self.assertEqual(self.updater.ready, self.installer)
        self.assertIsNone(self.updater.worker)
        self.launch_installer.assert_not_called()

    def test_cancel_save_as_does_not_install(self):
        with patch('project_controller.QMessageBox.question', return_value=QMessageBox.Save), \
             patch('project_controller.QFileDialog.getSaveFileName', return_value=('', '')):
            self.updater.offer_install()
        APP.processEvents()
        self.assertTrue(self.window.dirty)
        self.assertIsNone(self.window._job)
        self.assertIsNone(self.window._after_save)
        self.verify_installer.assert_not_called()
        self.launch_installer.assert_not_called()

    def test_failed_save_does_not_resume_installation(self):
        self.window.project_path = str(self.directory / 'failed.mrp')
        with patch('project_controller.QMessageBox.question', return_value=QMessageBox.Save), \
             patch('project_controller.save_project', side_effect=OSError('disk full')):
            self.updater.offer_install()
            self.wait_for(lambda: self.window._job is None)
            APP.processEvents()
        self.assertTrue(self.window.dirty)
        self.assertIsNone(self.window._after_save)
        self.verify_installer.assert_not_called()
        self.launch_installer.assert_not_called()

    def test_successful_project_save_resumes_verification_and_installation(self):
        self.window.project_path = str(self.directory / 'saved.mrp')
        gate = threading.Event()
        self.gates.append(gate)
        calls = []

        def save(*args):
            if not gate.wait(5):
                raise RuntimeError('Test save was not released')
            calls.append('saved')

        self.verify_installer.side_effect = lambda *a, **kw: calls.append('verified')
        self.launch_installer.side_effect = lambda *a: calls.append('launched')
        with patch('project_controller.QMessageBox.question', return_value=QMessageBox.Save) as question, \
             patch('project_controller.save_project', side_effect=save):
            self.updater.offer_install()
            self.assertIsNotNone(self.window._job)
            self.verify_installer.assert_not_called()
            self.launch_installer.assert_not_called()
            gate.set()
            self.wait_for(lambda: self.updater.closed)
            question.assert_called_once()
        self.assertEqual(calls, ['saved', 'verified', 'launched'])
        self.assertFalse(self.window.dirty)
        self.assertIsNone(self.window._job)
        self.assertIsNone(self.window._after_save)


if __name__ == '__main__':
    unittest.main()
