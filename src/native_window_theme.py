"""Theme Windows non-client captions without replacing native window controls."""
import ctypes
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication


def colorref(color):
    value = QColor(color)
    return value.red() | value.green() << 8 | value.blue() << 16


def apply_native_caption(widget, mode, panel, ink):
    if (sys.platform != 'win32' or QApplication.platformName() != 'windows'
            or not widget.isWindow() or not widget.testAttribute(Qt.WA_WState_Created)
            or widget.windowFlags() & Qt.FramelessWindowHint
            or widget.windowType() in (Qt.Popup, Qt.ToolTip)):
        return
    hwnd = int(widget.winId())
    signature = (hwnd, mode, panel, ink)
    if getattr(widget, '_engineering_caption', None) == signature: return
    try:
        set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        set_attribute.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        set_attribute.restype = ctypes.c_long
        def attribute(number, value):
            data = ctypes.c_uint32(value)
            return set_attribute(hwnd, number, ctypes.byref(data), ctypes.sizeof(data)) == 0
        # Windows 11 exposes exact caption/text colours. Older Windows versions
        # can reject these attributes and retain the native light/dark fallback.
        # https://learn.microsoft.com/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
        dark = attribute(20, int(mode == 'dark'))
        if not dark: dark = attribute(19, int(mode == 'dark'))
        caption = attribute(35, colorref(panel))
        attribute(36, colorref(ink))
        attribute(34, colorref(panel))
        if dark or caption:
            widget._engineering_caption = signature
            redraw = ctypes.windll.user32.RedrawWindow
            redraw.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            redraw.restype = ctypes.c_int
            redraw(hwnd, None, None, 0x001 | 0x040 | 0x100 | 0x400)
    except (AttributeError, OSError):
        # Native decoration is owned by the platform; keep the Qt UI usable.
        return
