"""A Linux desktop must not download or offer a Windows executable."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from app_updater import UpdateController

APP = QApplication.instance() or QApplication([])


class LinuxUpdateTests(unittest.TestCase):
    def test_startup_is_quiet_and_manual_check_opens_releases(self):
        window = QMainWindow()
        window.ui = SimpleNamespace(btn_check_updates=QPushButton(window))
        window.log = Mock()
        controller = UpdateController(window)
        with patch('app_updater.sys', SimpleNamespace(platform='linux')), \
             patch('app_updater.check_for_update') as request, \
             patch('app_updater.QDesktopServices.openUrl', return_value=True) as open_url:
            controller.start()
            self.assertFalse(controller.start_timer.isActive())
            controller.check()
            open_url.assert_not_called()
            controller.manual_check()
            request.assert_not_called()
            self.assertIsNone(controller.worker)
            self.assertEqual(open_url.call_args.args[0].toString(),
                             'https://github.com/B0ogie888/Meshropractor/releases')
        window.close()
