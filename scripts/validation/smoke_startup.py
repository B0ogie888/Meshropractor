"""Capture the native animated splash in both themes at controlled times."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from startup_splash import StartupSplash, ASSEMBLY_DURATION, LOGO_REVEAL_DURATION


def main():
    output = ROOT/'output/startup-smoke'; output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    now = [0.]
    splash = StartupSplash(clock=lambda: now[0], reduce_motion=False)
    try:
        for mode in ('light', 'dark'):
            splash.set_theme(mode); splash.show(); QTest.qWait(40)
            for seconds in (0., 2., 4.):
                now[0] = seconds
                splash.update(); app.processEvents()
                splash.grab().save(str(output/f'{mode}-{seconds:g}.png'))
            splash.set_stage('interface'); splash.set_stage('scene')
            splash.finish()
            finished = now[0]
            for phase, offset in (('reveal', ASSEMBLY_DURATION + LOGO_REVEAL_DURATION / 2),
                                  ('ready', ASSEMBLY_DURATION + LOGO_REVEAL_DURATION + .5)):
                now[0] = finished + offset; splash.update(); app.processEvents()
                splash.grab().save(str(output/f'{mode}-{phase}.png'))
            assert splash.isVisible() and splash.stage_index == 2
            splash.finished_at = None
            splash.stage_index = 0
            now[0] = 0.
    finally:
        splash.close(); splash.deleteLater(); app.processEvents()
    print('STARTUP_ARTWORK_OK')


if __name__ == '__main__': main()
