"""JSON pipe to an independent Qt splash; no GUI calls from worker threads."""
import json
import logging
from pathlib import Path
import subprocess
import sys


def reveal_after_splash(window, splash, *, on_revealed=None, timeout_ms=8000):
    """Show the prepared window only after the splash worker has exited.

    Polling belongs to the GUI event loop, so initialization never sleeps and
    the worker owns the animation timing. A missing/crashed worker opens the
    application immediately; a stalled worker has a bounded recovery time.
    """
    from PySide6.QtCore import QElapsedTimer, QTimer
    from PySide6.QtWidgets import QApplication

    timer = QTimer(window)
    timer.setInterval(20)
    elapsed = QElapsedTimer()

    def check_finished():
        process = splash.process
        if process is not None and process.poll() is None:
            if elapsed.elapsed() < timeout_ms:
                return
            logging.warning('Splash did not finish; opening the prepared application')
            splash.close()
        timer.stop()
        window.show()
        window.raise_()
        window.activateWindow()
        if on_revealed is not None:
            on_revealed()

    timer.timeout.connect(check_finished)
    QApplication.instance().aboutToQuit.connect(timer.stop)
    splash.ready()
    elapsed.start()
    timer.start()
    # The first check also handles unavailable splash processes without an
    # extra animation-length delay. Retain the timer through its Qt parent.
    return timer


class SplashProcess:
    def __init__(self):
        self.process = None

    def start(self, theme='light'):
        if self.process is not None:
            return
        command = [sys.executable]
        if not getattr(sys, 'frozen', False):
            command.append(str(Path(__file__).with_name('Meshropractor.py')))
        command.append('--splash-worker')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            self.send(dict(event='theme', value=theme))
        except OSError:
            logging.exception('Splash unavailable; continuing application startup')

    def send(self, message):
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
            process.stdin.flush()
        except (OSError, ValueError):
            logging.debug('Splash pipe closed', exc_info=True)

    def stage(self, name):
        self.send(dict(event='stage', value=name))

    def ready(self):
        self.send(dict(event='ready'))

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        # EOF also closes the splash if startup aborted before a ready message.
        if process.stdin is not None:
            try: process.stdin.close()
            except OSError: pass
        try:
            process.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.)
