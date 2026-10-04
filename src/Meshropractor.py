"""Lightweight source/PyInstaller entry; application code lives in main_window."""
import sys
from desktop_launcher import main


def __getattr__(name):
    # Keep the former public import usable without loading CAD during startup.
    if name == 'MainWindow':
        from main_window import MainWindow
        return MainWindow
    raise AttributeError(name)


if __name__ == '__main__':
    sys.exit(main())
