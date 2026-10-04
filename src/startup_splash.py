"""Animated Qt splash with a live parent pipe; deliberately no 3D dependencies."""
import json
import math
import os
import sys
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from app_version import APP_VERSION
from app_branding import app_icon, logo_pixmap

STAGES = ('geometry', 'interface', 'scene')
LABELS = ('Инициализация геометрии', 'Загрузка интерфейса', 'Подготовка рабочей сцены')
COLORS = {
    'light': ('#f4f5ef', '#242d29', '#65716a', '#d5dcd1', '#e9ede3'),
    'dark': ('#202725', '#edf0e8', '#a9b4ab', '#404b43', '#2b352f'),
}

# Finish the mesh sweep, reveal the mark, then give it a full second on screen.
# These run on the worker's Qt timer; application initialization never sleeps.
ASSEMBLY_DURATION = .30
LOGO_REVEAL_DURATION = .55
LOGO_HOLD_DURATION = 1.0
FADE_OUT_DURATION = .20


def reduced_motion():
    if sys.platform == 'win32':
        import ctypes
        enabled = ctypes.c_int(1)
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
            return not enabled.value
    return False


class StartupSplash(QWidget):
    closed = Signal()

    def __init__(self, *, clock=time.monotonic, reduce_motion=None):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle('Meshropractor — Готовим пространство')
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.clock, self.started = clock, clock()
        self.finished_at = None
        self.mode, self.stage_index = 'light', 0
        self.reduce_motion = reduced_motion() if reduce_motion is None else reduce_motion
        self.logo = logo_pixmap()
        self.points = []
        for side in range(6):
            for a in range(7):
                for b in range(7):
                    axis = side // 2
                    point = [0., 0., 0.]
                    point[axis] = 1. if side % 2 else -1.
                    point[(axis + 1) % 3] = a / 3 - 1
                    point[(axis + 2) % 3] = b / 3 - 1
                    self.points.append(point)
        screen = QApplication.primaryScreen().availableGeometry()
        scale = min(1., (screen.width() - 24) / 740, (screen.height() - 24) / 458)
        self.resize(round(740 * scale), round(458 * scale))
        self.move(screen.center() - self.rect().center())
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def set_theme(self, mode):
        self.mode = mode if mode in COLORS else 'light'
        self.update()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
        self.closed.emit()

    def set_stage(self, stage):
        if stage in STAGES and self.finished_at is None:
            self.stage_index = max(self.stage_index, STAGES.index(stage))
            self.update()

    def finish(self):
        if self.finished_at is None:
            self.finished_at = self.clock()
            self.setAccessibleDescription('Рабочая среда готова')
            self.update()

    @property
    def completion_duration(self):
        if self.reduce_motion:
            return LOGO_HOLD_DURATION
        return ASSEMBLY_DURATION + LOGO_REVEAL_DURATION + LOGO_HOLD_DURATION + FADE_OUT_DURATION

    def logo_opacity(self):
        if self.finished_at is None:
            return 0.
        if self.reduce_motion:
            return 1.
        progress = max(0., min(1.,
            (self.clock() - self.finished_at - ASSEMBLY_DURATION) / LOGO_REVEAL_DURATION))
        return progress * progress * (3 - 2 * progress)

    def tick(self):
        if self.finished_at is not None:
            elapsed = self.clock() - self.finished_at
            duration = self.completion_duration
            if elapsed >= duration:
                self.timer.stop()
                self.close()
                return
            if not self.reduce_motion and elapsed > duration - FADE_OUT_DURATION:
                self.setWindowOpacity(max(0., (duration - elapsed) / FADE_OUT_DURATION))
        if not self.reduce_motion or self.finished_at is not None:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.scale(self.width() / 740, self.height() / 458)
        bg, ink, muted, line, soft = map(QColor, COLORS[self.mode])
        accent = QColor('#e5d943')
        painter.fillRect(QRectF(0, 0, 740, 458), bg)
        painter.setPen(QPen(line, 1))
        painter.drawRect(QRectF(.5, .5, 739, 457))
        painter.drawLine(0, 66, 740, 66)
        painter.drawLine(0, 415, 740, 415)
        painter.fillRect(QRectF(0, 0, 86, 3), accent)

        def text(x, y, w, h, value, size=12, color=ink, bold=False, mono=False, align=Qt.AlignLeft):
            font = QFont('Consolas' if mono else 'Segoe UI')
            font.setPixelSize(size)
            font.setWeight(QFont.DemiBold if bold else QFont.Normal)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QRectF(x, y, w, h), align | Qt.AlignVCenter, value)

        painter.drawPixmap(QRectF(25, 12, 43, 43), self.logo, QRectF(self.logo.rect()))
        text(80, 19, 245, 29, 'MESHROPRACTOR', 13, bold=True)
        text(535, 19, 175, 29, 'VERSION / ' + APP_VERSION, 11, muted, mono=True, align=Qt.AlignRight)
        text(28, 96, 350, 20, 'PRECISION IN EVERY LAYER', 11, muted, mono=True)
        text(28, 135, 350, 44, 'Готовим', 38, bold=True)
        text(28, 177, 350, 44, 'пространство.', 38, bold=True)
        text(28, 238, 340, 22, 'Подготовка моделей.', 12, muted)
        text(28, 260, 340, 22, 'Контроль формы. Точное построение.', 12, muted)
        painter.fillRect(QRectF(28, 310, 6, 6), accent)
        ready = self.finished_at is not None
        label = 'Рабочая среда готова' if ready else LABELS[self.stage_index]
        self.setAccessibleName('Готовим пространство. ' + label)
        text(44, 299, 365, 27, label, 12)

        elapsed = max(0., self.clock() - self.started)
        t = 2.5 if self.reduce_motion else elapsed
        completion = self.logo_opacity()
        scan = (t * .19) % 1
        if ready:
            # Complete the current pass instead of abruptly extinguishing its
            # highlighted points when the real scene sends the ready signal.
            stopped = max(0., self.finished_at - self.started)
            scan_start = (stopped * .19) % 1
            scan = scan_start + (1 - scan_start) * min(1., (self.clock() - self.finished_at) / ASSEMBLY_DURATION)
            t = 2.5 if self.reduce_motion else stopped
        angle = -.55 + (.0 if self.reduce_motion else t * .1)
        cosine, sine = math.cos(angle), math.sin(angle)
        spread = 1 + max(0., 1 - t / 2.5) * .48
        projected = []
        for i, (x, y, z) in enumerate(self.points):
            rx, ry = x * cosine - y * sine, x * sine + y * cosine
            projected.append((558 + rx * 66 * spread, 203 + (ry * .45 - z * .82) * 66 * spread, ry * .8 + z * .3, i))
        painter.setPen(Qt.NoPen)
        for x, y, depth, index in sorted(projected, key=lambda p: p[2]):
            painter.setOpacity((.14 + .38 * (depth + 1.5) / 3) * (1 - completion * .85))
            painter.setBrush(ink)
            painter.drawEllipse(QPointF(x, y), 1.1, 1.1)
            if abs((index % 49) / 49 - scan) < .045:
                painter.setOpacity(.9 * (1 - completion))
                painter.fillRect(QRectF(x - 1.5, y - 1.5, 3, 3), accent)
        painter.setPen(QPen(ink, .55))
        painter.setOpacity(.12 * (1 - completion * .85))
        for i, (x, y, _, _) in enumerate(projected):
            if i % 7 < 6 and i % 2 == 0:
                painter.drawLine(QPointF(x, y), QPointF(*projected[i+1][:2]))
        if completion:
            painter.setOpacity(completion)
            size = 210 if self.reduce_motion else 194 + 16 * completion
            painter.drawPixmap(QRectF(558-size/2, 203-size/2, size, size), self.logo, QRectF(self.logo.rect()))
        painter.setOpacity(1.)
        text(414, 302, 289, 20, 'GEOMETRY / MESH / LAYERS', 10, muted, mono=True, align=Qt.AlignCenter)
        text(28, 343, 540, 24, 'РАБОЧАЯ СРЕДА ГОТОВА' if ready else 'ЗАГРУЗКА РАБОЧЕЙ СРЕДЫ', 11, mono=True)
        text(594, 343, 116, 24, 'ГОТОВО' if ready else f'{self.stage_index + 1:02d} / 03', 13, mono=True, align=Qt.AlignRight)
        for i, title in enumerate(('Геометрия', 'Интерфейс', 'Сцена')):
            x = 28 + i * 231
            painter.fillRect(QRectF(x, 371, 219, 3), accent if ready or i < self.stage_index else soft)
            if not ready and i == self.stage_index:
                # Activity within this real stage; it is not a fabricated percent.
                pos = 0.5 if self.reduce_motion else .5 + .5 * math.sin(t * 1.8)
                painter.fillRect(QRectF(x + pos * 168, 371, 51, 3), accent)
            text(x, 382, 220, 18, f'{i+1:02d}  {title}', 11, ink if ready or i <= self.stage_index else muted)
        text(28, 426, 345, 20, 'ENGINEERING ENVIRONMENT', 10, muted, mono=True)
        text(446, 426, 264, 20, 'SYSTEM READY' if ready else 'INITIALIZING', 10, muted, mono=True, align=Qt.AlignRight)
        painter.end()


class ParentPipe:
    """Nonblocking pipe reads on Windows and Linux; EOF means parent has exited."""
    def __init__(self):
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            self.ctypes = ctypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.GetStdHandle.argtypes = [wintypes.DWORD]
            self.kernel.GetStdHandle.restype = wintypes.HANDLE
            self.kernel.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                                ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            self.kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            self.handle = self.kernel.GetStdHandle(-10)

    def read(self):
        if sys.platform == 'win32':
            from ctypes import wintypes
            count = wintypes.DWORD()
            if not self.kernel.PeekNamedPipe(self.handle, None, 0, None, self.ctypes.byref(count), None):
                return b''
            if not count.value: return None
            buffer = self.ctypes.create_string_buffer(min(4096, count.value))
            if not self.kernel.ReadFile(self.handle, buffer, len(buffer), self.ctypes.byref(count), None):
                return b''
            return buffer.raw[:count.value]
        import select
        if not select.select([0], [], [], 0)[0]: return None
        return os.read(0, 4096)


def run_worker():
    app = QApplication(['Meshropractor splash'])
    app.setStyle('Fusion')
    window = StartupSplash()
    window.closed.connect(app.quit)
    pipe, buffer = ParentPipe(), bytearray()

    def receive():
        chunk = pipe.read()
        if chunk is None: return
        if not chunk:
            window.close(); app.quit(); return
        buffer.extend(chunk)
        if len(buffer) > 65536:
            window.close(); app.quit(); return
        while b'\n' in buffer:
            line, _, rest = buffer.partition(b'\n')
            buffer[:] = rest
            try: message = json.loads(line.decode('utf-8'))
            except (ValueError, UnicodeError): continue
            if not isinstance(message, dict): continue
            event = message.get('event')
            if event == 'theme':
                window.set_theme(message.get('value')); window.show()
            elif event == 'stage': window.set_stage(message.get('value'))
            elif event == 'ready': window.finish()

    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(receive)
    timer.start()
    return app.exec()
