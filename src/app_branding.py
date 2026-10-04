"""Shared application identity and artwork for source and packaged launches."""
from pathlib import Path
import sys

from PySide6.QtGui import QIcon, QPixmap

APP_NAME = 'Meshropractor'
APP_ID = 'b0ogie888.meshropractor'
ASSETS = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1])) / 'assets'


def app_icon():
    """Use the multi-resolution ICO for Windows captions, taskbar and Alt+Tab."""
    return QIcon(str(ASSETS / 'logo.ico'))


def logo_pixmap():
    return QPixmap(str(ASSETS / 'logo.png'))


def set_process_identity():
    if sys.platform == 'win32':
        import ctypes
        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        setter(APP_ID)
