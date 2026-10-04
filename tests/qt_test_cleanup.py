"""Isolate desktop preferences and finish deletion outside QApplication.exec."""
from pathlib import Path
import os
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from PySide6.QtCore import QCoreApplication, QEvent, QSettings
import shiboken6
import main_window

_profile = tempfile.TemporaryDirectory(prefix='meshropractor-tests-')


def test_settings(organization, application):
    settings = QSettings(str(Path(_profile.name) / (application + '.ini')), QSettings.IniFormat)
    settings.setFallbacksEnabled(False)
    return settings


# MainWindow passes this factory to load_settings; use real INI persistence
# without reading or writing the developer's Windows registry preferences.
main_window.QSettings = test_settings


def delete_widget(widget):
    widget.deleteLater()
    # processEvents alone does not drain DeferredDelete outside an event loop.
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(widget), 'Qt test window was not destroyed'
