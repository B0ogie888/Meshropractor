"""Use the Windows move/resize loop so the custom title bar supports Aero Snap."""
import sys
import ctypes
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QAbstractButton, QApplication


def enable_snap(window):
    if sys.platform != 'win32' or QApplication.platformName() == 'offscreen': return
    user32 = ctypes.windll.user32
    get_style = user32.GetWindowLongPtrW
    set_style = user32.SetWindowLongPtrW
    get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    set_style.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    set_style.restype = ctypes.c_ssize_t
    hwnd = int(window.winId())
    # Native resizable/minimizable/maximizable window; Qt retains the custom caption.
    set_style(hwnd, -16, get_style(hwnd, -16) | 0x00040000 | 0x00010000 | 0x00020000)


def title_event(window, obj, event):
    if event.type() not in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick): return False
    if event.button() != Qt.LeftButton or not hasattr(obj, 'parentWidget'): return False
    node = obj
    while node is not None and node is not window.ui.title_bar:
        if isinstance(node, QAbstractButton): return False
        node = node.parentWidget()
    if node is None: return False
    if event.type() == QEvent.MouseButtonDblClick:
        window.showNormal() if window.isMaximized() else window.showMaximized()
        return True
    return bool(window.windowHandle() and window.windowHandle().startSystemMove())
