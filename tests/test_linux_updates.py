"""Platform-specific release discovery and Linux UI; no real network or install."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import io
import json
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from app_updater import UpdateController
from app_version import APP_VERSION
import update_backend as backend
from qt_test_cleanup import delete_widget

APP = QApplication.instance() or QApplication([])


def release(version='0.3.1', tag='v0.3.1', name=None):
    name = name or f'meshropractor_{version}_amd64.deb'
    return dict(tag_name=tag, assets=[dict(name=name, size=123456, state='uploaded',
        browser_download_url=f'https://github.com/{backend.REPOSITORY}/releases/download/{tag}/{name}')])


class LinuxReleaseTests(unittest.TestCase):
    def check(self, items, current='0.3', platform='linux'):
        with patch.object(backend, '_open_url', return_value=io.BytesIO(json.dumps(items).encode())):
            return backend.check_for_update(current, platform=platform)

    def test_linux_uses_deb_version_and_ignores_windows_and_wrong_architecture(self):
        self.assertIsNone(self.check([release(name='Meshropractor-Setup-9.0-x64.exe'),
                                     release(name='meshropractor_9.0_arm64.deb')]))
        self.assertEqual(self.check([release('0.3.9'), release('0.3.10', tag='v3.10')]).version, '0.3.10')
        self.assertIsNone(self.check([release('0.3.0')]))
        mixed = release()
        mixed['assets'] += release(name='Meshropractor-Setup-0.4-x64.exe')['assets']
        self.assertEqual(self.check([mixed]).version, '0.3.1')
        self.assertEqual(self.check([mixed], platform='win32').version, '0.4')

    def test_linux_stable_release_and_url_validation(self):
        for change in ('draft', 'prerelease', 'bad_url', 'ambiguous'):
            item = release()
            if change in ('draft', 'prerelease'): item[change] = True
            elif change == 'bad_url': item['assets'][0]['browser_download_url'] = 'https://github.com/other/app/file.deb'
            else: item['assets'] += release('0.3.2', tag=item['tag_name'])['assets']
            self.assertIsNone(self.check([item]), change)


class LinuxUpdateTests(unittest.TestCase):
    def setUp(self):
        self.window = QMainWindow()
        self.window.ui = SimpleNamespace(btn_check_updates=QPushButton(self.window))
        self.window.log = Mock()
        self.controller = UpdateController(self.window)
        self.platform = patch('app_updater.sys', SimpleNamespace(platform='linux'))
        self.platform.start()

    def tearDown(self):
        self.controller.on_closed()
        self.platform.stop()
        delete_widget(self.window)

    def test_startup_check_enabled_and_passes_current_version_and_platform(self):
        controller = self.controller
        controller.start()
        self.assertTrue(controller.start_timer.isActive())
        controller.start_timer.stop()
        with patch('app_updater.check_for_update', return_value=None) as request:
            controller.check(manual=True)
            deadline = time.monotonic() + 5
            while controller.worker and time.monotonic() < deadline:
                APP.processEvents(); time.sleep(.01)
            self.assertIsNone(controller.worker)
            self.assertEqual(request.call_args.args, (APP_VERSION,))
            self.assertEqual(request.call_args.kwargs['platform'], 'linux')
        self.assertIn(APP_VERSION, self.window.ui.btn_check_updates.text())
        with patch('app_updater.QMessageBox.information') as info:
            controller.pending_offer()
            self.assertIn(APP_VERSION, info.call_args.args[2])

    def test_new_linux_version_offers_release_without_windows_download(self):
        controller = self.controller
        controller.available = backend._release_update(release(), platform='linux')
        with patch.object(controller, '_ask', return_value=True) as ask, \
             patch('app_updater.download_release') as download, \
             patch('app_updater.QDesktopServices.openUrl', return_value=True) as open_url:
            controller.offer_download()
            self.assertIn('0.3.1', ask.call_args.args[1])
            self.assertIn(APP_VERSION, ask.call_args.args[1])
            self.assertEqual(open_url.call_args.args[0].toString(), controller.available.page_url)
            download.assert_not_called()
