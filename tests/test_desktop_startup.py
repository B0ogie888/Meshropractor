"""Settings migration and real source/pythonw splash lifecycle, without installers."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from app_settings import load_settings
from startup_bridge import SplashProcess, reveal_after_splash
from startup_splash import (StartupSplash, ASSEMBLY_DURATION, LOGO_REVEAL_DURATION,
                            LOGO_HOLD_DURATION, FADE_OUT_DURATION)

APP = QApplication.instance() or QApplication([])


class DesktopStartupTests(unittest.TestCase):
    def test_migrate_new_settings_once_preserve_old_values_and_user_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            def settings(team, name):
                result = QSettings(str(Path(folder)/(name+'.ini')), QSettings.IniFormat)
                result.setFallbacksEnabled(False)
                return result
            old = settings('', 'Meshropractor')
            old.setValue('platforms_json', 'old platform')
            old.setValue('recent_files', 'old project')
            new = settings('', 'MeshropractorNew')
            new.setValue('platforms_json', 'working platform')
            new.setValue('appearance/theme', 'dark')
            new.setValue('panels/state', b'panel layout')
            migrated = load_settings(settings)
            self.assertEqual(migrated.value('platforms_json'), 'working platform')
            self.assertEqual(migrated.value('recent_files'), 'old project')
            self.assertEqual(migrated.value('appearance/theme'), 'dark')
            self.assertEqual(migrated.value('panels/state'), b'panel layout')
            self.assertEqual(migrated.value('migration/before_0_3/platforms_json'), 'old platform')
            migrated.setValue('appearance/theme', 'light')
            self.assertEqual(load_settings(settings).value('appearance/theme'), 'light')
            self.assertEqual(new.value('appearance/theme'), 'dark')

    def test_splash_does_not_finish_until_ready_and_stages_never_go_backwards(self):
        now = [0.]
        splash = StartupSplash(clock=lambda: now[0], reduce_motion=False)
        try:
            splash.show(); APP.processEvents()
            splash.set_stage('scene'); splash.set_stage('geometry')
            self.assertEqual(splash.stage_index, 2)
            now[0] = 120.; splash.tick()
            self.assertTrue(splash.isVisible())
            self.assertIsNone(splash.finished_at)
            splash.finish(); now[0] = 120.2; splash.tick()
            self.assertTrue(splash.isVisible())
            self.assertEqual(splash.logo_opacity(), 0.)
            splash.finish()  # duplicate ready signals must not extend the animation
            now[0] = 120. + ASSEMBLY_DURATION + LOGO_REVEAL_DURATION / 2
            self.assertAlmostEqual(splash.logo_opacity(), .5)
            reveal_done = 120. + ASSEMBLY_DURATION + LOGO_REVEAL_DURATION
            for offset in (.01, LOGO_HOLD_DURATION - .01):
                now[0] = reveal_done + offset; splash.tick()
                self.assertEqual(splash.logo_opacity(), 1.)
                self.assertEqual(splash.windowOpacity(), 1.)
                self.assertTrue(splash.isVisible())
            now[0] = reveal_done + LOGO_HOLD_DURATION + FADE_OUT_DURATION / 2
            splash.tick()
            self.assertTrue(splash.isVisible())
            self.assertAlmostEqual(splash.windowOpacity(), .5, delta=.01)
            now[0] = 120. + splash.completion_duration + .01; splash.tick()
            self.assertFalse(splash.isVisible())
        finally:
            splash.close(); splash.deleteLater(); APP.processEvents()

    def test_reduced_motion_shows_complete_logo_without_moving_or_fading(self):
        now = [0.]
        splash = StartupSplash(clock=lambda: now[0], reduce_motion=True)
        try:
            splash.show(); splash.finish()
            self.assertEqual(splash.logo_opacity(), 1.)
            now[0] = LOGO_HOLD_DURATION - .01; splash.tick()
            self.assertTrue(splash.isVisible())
            self.assertEqual(splash.windowOpacity(), 1.)
            now[0] = LOGO_HOLD_DURATION + .01; splash.tick()
            self.assertFalse(splash.isVisible())
        finally:
            splash.close(); splash.deleteLater(); APP.processEvents()

    def test_source_worker_exits_on_ready_and_on_parent_pipe_eof(self):
        # Both python.exe and pythonw.exe must work: pythonw has sys.stdin=None.
        executables = [sys.executable]
        pythonw = Path(sys.executable).with_name('pythonw.exe')
        if sys.platform == 'win32' and pythonw.exists(): executables.append(str(pythonw))
        for executable in executables:
            for ready in (False, True):
                with self.subTest(executable=executable, ready=ready):
                    process = subprocess.Popen([executable, str(ROOT/'src/Meshropractor.py'), '--splash-worker'],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding='utf-8', env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
                    try:
                        process.stdin.write('{"event":"theme","value":"dark"}\n')
                        process.stdin.write('{"event":"stage","value":"interface"}\n')
                        process.stdin.flush()
                        if ready:
                            process.stdin.write('{"event":"ready"}\n'); process.stdin.flush()
                            code = process.wait(timeout=15)
                        else:
                            process.stdin.close()
                            code = process.wait(timeout=15)
                        self.assertEqual(code, 0, process.stderr.read())
                    finally:
                        if process.poll() is None: process.kill(); process.wait(timeout=5)
                        for stream in (process.stdin, process.stdout, process.stderr):
                            if stream is not None and not stream.closed: stream.close()

    def test_failed_splash_spawn_does_not_prevent_application_start(self):
        splash = SplashProcess()
        with patch('startup_bridge.subprocess.Popen', side_effect=OSError('test unavailable')):
            with self.assertLogs(level='ERROR'):
                splash.start()
        splash.stage('geometry'); splash.ready(); splash.close()
        self.assertIsNone(splash.process)

    def test_main_window_stays_hidden_until_worker_finishes(self):
        window, splash = QWidget(), Mock()
        splash.process.poll.return_value = None
        revealed = Mock()
        timer = reveal_after_splash(window, splash, on_revealed=revealed)
        try:
            splash.ready.assert_called_once()
            QTest.qWait(80)
            self.assertFalse(window.isVisible())
            revealed.assert_not_called()
            splash.process.poll.return_value = 0
            QTest.qWait(80)
            self.assertTrue(window.isVisible())
            self.assertFalse(timer.isActive())
            revealed.assert_called_once()
            QTest.qWait(40)
            revealed.assert_called_once()
        finally:
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_unavailable_and_crashed_splashes_do_not_delay_window(self):
        for process in (None, Mock()):
            with self.subTest(process=process):
                if process is not None: process.poll.return_value = 1
                window, splash = QWidget(), Mock()
                splash.process = process
                try:
                    reveal_after_splash(window, splash)
                    QTest.qWait(60)
                    self.assertTrue(window.isVisible())
                finally:
                    window.close(); window.deleteLater()
                    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_stalled_splash_recovers_to_main_window(self):
        window, splash = QWidget(), Mock()
        splash.process.poll.return_value = None
        try:
            with self.assertLogs(level='WARNING'):
                timer = reveal_after_splash(window, splash, timeout_ms=40)
                QTest.qWait(100)
            splash.close.assert_called_once()
            self.assertTrue(window.isVisible())
            self.assertFalse(timer.isActive())
        finally:
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_entry_help_is_lightweight_and_hides_obsolete_editions(self):
        result = subprocess.run([sys.executable, str(ROOT/'src/Meshropractor.py'), '--help'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--ui', result.stdout)

    def test_failed_window_construction_closes_splash_and_reports_error(self):
        # Isolated QApplication: exercise the actual launcher exception path.
        script = r'''
import sys, types
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QMessageBox
import desktop_launcher
window = Mock(side_effect=RuntimeError('window construction failed'))
sys.modules['main_window'] = types.SimpleNamespace(MainWindow=window)
settings, splash, dialog = Mock(), Mock(), Mock()
settings.value.return_value = 'light'
with patch('app_settings.load_settings', return_value=settings), \
     patch('startup_bridge.SplashProcess', return_value=splash), \
     patch('PySide6.QtWidgets.QMessageBox.exec', return_value=0) as execute, \
     patch('PySide6.QtWidgets.QMessageBox.setDetailedText') as details:
    assert desktop_launcher.main([]) == 1
    splash.start.assert_called_once_with(theme='light')
    splash.ready.assert_not_called()
    assert splash.close.called
    execute.assert_called_once()
    assert 'window construction failed' in details.call_args.args[0]
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT/'src',
                                capture_output=True, text=True, timeout=15,
                                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
