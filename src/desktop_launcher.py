"""Lightweight entry point; imports the 3D application after the animated splash."""
from pathlib import Path
import os
import subprocess
import sys


def launch():
    root = Path(__file__).resolve().parents[1]
    python = root / '.venv' / ('Scripts/pythonw.exe' if sys.platform == 'win32' else 'bin/python')
    if python.is_file() and Path(sys.prefix).resolve() != (root / '.venv').resolve():
        subprocess.Popen([str(python), str(root / 'src/Meshropractor.py')],
                         cwd=str(root), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return
    raise SystemExit(main())


def main(argv=None):
    import argparse
    from startup_logging import configure_windowed_logging
    log_path = configure_windowed_logging()
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Dispatch before importing Torch, VTK or OCP, including in the frozen EXE.
    if arguments == ['--splash-worker']:
        from startup_splash import run_worker
        return run_worker()
    if len(arguments) == 2 and arguments[0] == '--self-test':
        from frozen_smoke import run
        run(arguments[1])
        return 0
    parser = argparse.ArgumentParser(description='Meshropractor desktop')
    # Old installed shortcuts remain usable; both select the single interface.
    parser.add_argument('--ui', choices=('new', 'classic'), help=argparse.SUPPRESS)
    _, qt_arguments = parser.parse_known_args(arguments)
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QEventLoop, QTimer
    from app_branding import APP_NAME, app_icon, set_process_identity
    from app_settings import load_settings
    from app_version import APP_VERSION
    from startup_bridge import SplashProcess, reveal_after_splash
    set_process_identity()
    app = QApplication([sys.argv[0]] + qt_arguments)
    app.setStyle('Fusion')
    app.setApplicationName(APP_NAME)
    app.setOrganizationName('MeshropractorTeam')
    app.setApplicationVersion(APP_VERSION)
    app.setWindowIcon(app_icon())
    settings = load_settings()
    splash = SplashProcess()
    splash.start(theme=settings.value('appearance/theme', 'light'))
    app.aboutToQuit.connect(splash.close)
    try:
        splash.stage('geometry')
        from main_window import MainWindow
        from ui_theme import theme_palette
        mode = settings.value('appearance/theme', 'light')
        app.setPalette(theme_palette(mode if mode in ('light', 'dark') else 'light'))
        splash.stage('interface')
        window = MainWindow()
        splash.stage('scene')
        # Construct and polish while hidden. The splash completes its animation
        # first; only its process exit releases the real application window.
        window.ensurePolished()
        app.processEvents(QEventLoop.ExcludeUserInputEvents)
        reveal_after_splash(window, splash,
            on_revealed=lambda: QTimer.singleShot(600, window.updater.start))
        return app.exec()
    except Exception:
        import logging
        import traceback
        logging.exception('Application startup failed')
        splash.close()
        details = traceback.format_exc()
        if log_path: details += '\nЖурнал: ' + str(log_path)
        dialog = QMessageBox(QMessageBox.Critical, 'Meshropractor — ошибка запуска',
                             'Не удалось запустить приложение. Подробности доступны ниже.')
        dialog.setDetailedText(details)
        dialog.exec()
        return 1
    finally:
        splash.close()
