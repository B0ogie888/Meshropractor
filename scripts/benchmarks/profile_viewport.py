"""Profile a continuous drag in a native Windows viewport, without building an EXE.

Run from the project environment, for example:
    python scripts/benchmarks/profile_viewport.py --page slicer --seconds 5 --output output/drag.json

Each run opens its own empty application window and uses temporary preferences.
No project is loaded, no update request is sent, and existing application processes
are untouched. The parent process enforces a timeout even if native rendering hangs.
"""
import argparse
from contextlib import ExitStack
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def statistics(values):
    """Seconds to milliseconds; empty samples are unavailable, never zero latency."""
    ordered = sorted(value * 1000 for value in values)
    if not ordered:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None,
                'p99_ms': None, 'max_ms': None}

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return dict(count=len(ordered), mean_ms=sum(ordered) / len(ordered),
                p50_ms=percentile(.5), p95_ms=percentile(.95),
                p99_ms=percentile(.99), max_ms=ordered[-1])


def intervals(timestamps):
    return statistics(b - a for a, b in zip(timestamps, timestamps[1:]))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def swap_interval(render_window):
    """Query only this window's current WGL context; never change driver settings."""
    if os.name != 'nt' or not render_window.IsA('vtkWin32OpenGLRenderWindow'):
        return None
    render_window.MakeCurrent()
    opengl = ctypes.WinDLL('opengl32')
    get_address = opengl.wglGetProcAddress
    get_address.argtypes = [ctypes.c_char_p]
    get_address.restype = ctypes.c_void_p
    address = get_address(b'wglGetSwapIntervalEXT')
    if address in (None, 0, 1, 2, 3, ctypes.c_void_p(-1).value):
        return None
    return ctypes.WINFUNCTYPE(ctypes.c_int)(address)()


def profile(args):
    # Importing --help or the statistics helpers does not initialize Qt/VTK.
    from PySide6 import __version__ as pyside_version
    from PySide6.QtCore import QEvent, QPointF, QSettings, Qt, QTimer
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QMainWindow
    import pyvista
    import pyvistaqt
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkCommonCore import vtkVersion

    sys.path.insert(0, str(ROOT / 'src'))
    from orientation_cube import OrientationCube
    from viewport_performance import ViewportPerformance

    app = QApplication([])
    if app.platformName() != 'windows':
        raise RuntimeError('This profile requires the native Windows Qt platform, not offscreen.')
    app.setStyle('Fusion')
    app.setApplicationName('Meshropractor viewport profile')
    state = dict(active=False, finished=False, pressed=False, error=None)
    samples = {name: [] for name in ('events', 'handlers', 'ends', 'render', 'geometry',
                                   'present', 'extra_present')}
    window = plotter = cube = performance = None
    observers = []
    original_frame = None
    frame_wrapped = False
    original_swap = None
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    report = dict(page=args.page, requested_seconds=args.seconds,
                  input_interval_ms=args.interval_ms, cube_visible=not args.hide_cube,
                  anti_aliasing=args.anti_aliasing, settings='temporary',
                  update_checks='disabled', warnings=[])

    with tempfile.TemporaryDirectory(prefix='meshropractor-profile-') as settings_dir, ExitStack() as patches:
        # Explicit organization/name constructors also use these process-local
        # paths. Disable fallback reads on the exact MainWindow settings object.
        QSettings.setDefaultFormat(QSettings.IniFormat)
        for scope in (QSettings.UserScope, QSettings.SystemScope):
            QSettings.setPath(QSettings.IniFormat, scope, settings_dir)
        try:
            if args.page == 'bare':
                window = QMainWindow()
                plotter = QtInteractor(window, auto_update=False)
                window.setCentralWidget(plotter)
                performance = plotter._viewport_performance = ViewportPerformance(plotter)
                cube = OrientationCube(plotter)
            else:
                import app_updater
                def disable_updates(self, *unused_args, **unused_kwargs):
                    pass

                for method in ('start', 'manual_check', 'check'):
                    # Qt signal connections must receive a real bound method.
                    # MagicMock in a QObject class can crash PySide's slot lookup.
                    patches.enter_context(patch.object(app_updater.UpdateController, method, new=disable_updates))
                import main_window
                settings = QSettings(str(Path(settings_dir) / 'profile.ini'), QSettings.IniFormat)
                settings.setFallbacksEnabled(False)
                patches.enter_context(patch.object(main_window, 'QSettings', new=lambda *unused_args: settings))
                window = main_window.MainWindow()
                if args.page == 'slicer':
                    window.ui.stack.setCurrentWidget(window.ui.page_slicer)
                    plotter = window.ui.slicer_plotter
                    cube = window.workspace_tools.cube
                else:
                    window.ui.stack.setCurrentWidget(window.ui.page_predef)
                    plotter = window.ui.plotter
                    cube = window.ui.def_cube
                performance = plotter._viewport_performance

            window.setWindowTitle('Meshropractor — viewport performance profile')
            window.resize(1200, 850)
            window.show()
            plotter.set_background('white')
            performance.configure(anti_aliasing=args.anti_aliasing, samples=4)
            cube.setVisible(not args.hide_cube)
            plotter.camera_position = 'iso'
            plotter.reset_camera()
            plotter.render()
            render_window = plotter.render_window
            original_swap = swap_interval(render_window)
            swap_accepted = None
            if args.swap_interval is not None:
                render_window.MakeCurrent()
                swap_accepted = bool(render_window.SetSwapControl(args.swap_interval))
                if not swap_accepted:
                    report['warnings'].append('The context rejected the requested swap interval.')
            capability_lines = render_window.ReportCapabilities().splitlines()
            report['environment'] = dict(
                python=sys.version.split()[0], pyside=pyside_version,
                vtk=vtkVersion.GetVTKVersion(), pyvista=pyvista.__version__,
                pyvistaqt=pyvistaqt.__version__, platform=app.platformName(),
                render_window=render_window.GetClassName(),
                opengl=[line.strip() for line in capability_lines
                        if line.lower().startswith(('opengl vendor', 'opengl renderer', 'opengl version'))],
                swap_interval_before=original_swap, swap_interval_requested=args.swap_interval,
                swap_request_accepted=swap_accepted, swap_interval_actual=swap_interval(render_window))

            def render_start(*_):
                if state['active']:
                    state['render_start'] = time.perf_counter()
                    state['render_ready'] = None

            def render_ready(*_):
                if state['active']:
                    state['render_ready'] = time.perf_counter()

            def render_end(*_):
                if not state['active']:
                    return
                now = time.perf_counter()
                start, ready = state.get('render_start'), state.get('render_ready')
                samples['ends'].append(now)
                if start is not None:
                    samples['render'].append(now - start)
                if ready is not None and start is not None:
                    samples['geometry'].append(ready - start)
                    samples['present'].append(now - ready)
                state['render_start'] = state['render_ready'] = None

            for event, callback in (('StartEvent', render_start), ('RenderEvent', render_ready),
                                    ('EndEvent', render_end)):
                observers.append(render_window.AddObserver(event, callback))
            original_frame = render_window.Frame

            def frame(*call_args, **call_kwargs):
                start = time.perf_counter()
                try:
                    return original_frame(*call_args, **call_kwargs)
                finally:
                    if state['active']:
                        samples['extra_present'].append(time.perf_counter() - start)

            try:
                # C++ virtual calls inside Render bypass this Python attribute.
                # Only additional Frame calls issued from Python are counted.
                render_window.Frame = frame
                frame_wrapped = True
            except (AttributeError, TypeError):
                report['warnings'].append('Python Frame calls could not be instrumented.')
            report['python_frame_instrumented'] = frame_wrapped

            def send(kind, point, button, buttons):
                event = QMouseEvent(kind, point, QPointF(plotter.mapToGlobal(point.toPoint())),
                                    button, buttons, Qt.NoModifier)
                QApplication.sendEvent(plotter, event)

            def finish(error=None):
                if state['finished']:
                    return
                state['finished'] = True
                state['active'] = False
                state['error'] = error
                state['stop'] = time.perf_counter()
                timer.stop()
                if state['pressed']:
                    state['pressed'] = False
                    try:
                        send(QEvent.MouseButtonRelease, state['last'], Qt.LeftButton, Qt.NoButton)
                    except RuntimeError:
                        pass  # The user may have closed this diagnostic window.
                app.quit()

            def begin():
                try:
                    screen = plotter.screen()
                    report['screen'] = dict(name=screen.name(), refresh_hz=screen.refreshRate(),
                        geometry=[screen.geometry().width(), screen.geometry().height()],
                        device_pixel_ratio=plotter.devicePixelRatioF())
                    report['viewport_size'] = list(render_window.GetSize())
                    report['scene_actors'] = list(plotter.actors)
                    report['renderer_count'] = render_window.GetRenderers().GetNumberOfItems()
                    report['multisamples'] = render_window.GetMultiSamples()
                    report['fxaa'] = bool(plotter.renderer.GetUseFXAA())
                    state['center'] = QPointF(plotter.width() * .55, plotter.height() * .45)
                    state['radius'] = (min(100., plotter.width() * .2), min(90., plotter.height() * .2))
                    state['last'] = state['center'] + QPointF(state['radius'][0], 0)
                    send(QEvent.MouseButtonPress, state['last'], Qt.LeftButton, Qt.LeftButton)
                    state['pressed'] = True
                    state['start'] = time.perf_counter()
                    state['active'] = True
                    timer.start(args.interval_ms)
                except Exception:
                    finish(traceback.format_exc())

            def tick():
                try:
                    now = time.perf_counter()
                    elapsed = now - state['start']
                    if elapsed >= args.seconds:
                        finish()
                        return
                    point = state['center'] + QPointF(state['radius'][0] * math.cos(elapsed * 1.2),
                                                      state['radius'][1] * math.sin(elapsed * 1.2))
                    # A zero-interval run must not flood VTK with stationary
                    # subpixel events which a physical mouse would never send.
                    if point.toPoint() == state['last'].toPoint():
                        return
                    samples['events'].append(now)
                    send(QEvent.MouseMove, point, Qt.NoButton, Qt.LeftButton)
                    samples['handlers'].append(time.perf_counter() - now)
                    state['last'] = point
                except Exception:
                    finish(traceback.format_exc())

            timer.timeout.connect(tick)
            # Warm up shaders and finish initial layouts outside the measurement.
            QTimer.singleShot(900, begin)
            app.exec()
            if not state['finished']:
                state['error'] = 'The diagnostic window was closed before measurement completed.'
            report.update(status='error' if state['error'] else 'ok', error=state['error'],
                elapsed_seconds=state.get('stop', time.perf_counter()) - state.get('start', time.perf_counter()),
                mouse_moves=len(samples['events']), full_frames=len(samples['ends']),
                mouse_intervals=intervals(samples['events']), mouse_handler=statistics(samples['handlers']),
                frame_intervals=intervals(samples['ends']), full_render=statistics(samples['render']),
                render_before_present=statistics(samples['geometry']), present=statistics(samples['present']),
                extra_python_frame=statistics(samples['extra_present']))
            report['timing_note'] = (
                'These are wall-clock VTK event phases, not isolated GPU geometry timings. '
                'Driver/VSync waits can occur before RenderEvent as well as inside Frame; '
                'frame_intervals measure the complete pacing seen by the event loop.')
            if report['status'] == 'ok' and (report['mouse_moves'] < 2 or report['full_frames'] < 2):
                report.update(status='error', error='Too few drag events or rendered frames to measure.')
        except Exception:
            report.update(status='error', error=traceback.format_exc())
        finally:
            state['active'] = False
            timer.stop()
            try:
                if plotter is not None and not getattr(plotter, '_closed', False):
                    render_window = plotter.render_window
                    for observer in observers:
                        render_window.RemoveObserver(observer)
                    if frame_wrapped:
                        render_window.Frame = original_frame
                    if args.swap_interval is not None and original_swap is not None:
                        render_window.MakeCurrent()
                        render_window.SetSwapControl(original_swap)
                if window is not None:
                    if args.page == 'bare':
                        if cube is not None:
                            cube.dispose()
                        if performance is not None:
                            performance.dispose()
                        if plotter is not None:
                            plotter.close()
                    else:
                        window.dirty = False
                    window.close()
                app.processEvents()
            except Exception:
                report['warnings'].append('Cleanup: ' + traceback.format_exc())
                report['status'] = 'error'
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--page', choices=('slicer', 'predef', 'bare'), default='slicer')
    parser.add_argument('--seconds', type=float, default=5, help='Drag duration in seconds (0.25-300).')
    parser.add_argument('--interval-ms', type=int, choices=(0, 8), default=8,
                        help='PreciseTimer input interval: 8 approximates 125 Hz, 0 has no waiting.')
    parser.add_argument('--output', type=Path, default=Path('output/profile-viewport.json'))
    parser.add_argument('--hide-cube', action='store_true', help='Hide only the orientation cube for comparison.')
    parser.add_argument('--anti-aliasing', choices=('msaa', 'none', 'fxaa'), default='msaa')
    parser.add_argument('--swap-interval', type=int, choices=(0, 1),
                        help='Explicit diagnostic override in this context only; default leaves VSync unchanged.')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not .25 <= args.seconds <= 300:
        parser.error('--seconds must be finite and between 0.25 and 300.')
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    if args._worker:
        report = profile(args)
        write_report(args.output, report)
        return 0 if report['status'] == 'ok' else 1
    if os.name != 'nt':
        raise SystemExit('Native viewport profiling is supported on Windows only.')
    command = [sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve()), '--_worker', '--page', args.page,
               '--seconds', str(args.seconds), '--interval-ms', str(args.interval_ms),
               '--anti-aliasing', args.anti_aliasing]
    if args.hide_cube:
        command.append('--hide-cube')
    if args.swap_interval is not None:
        command.extend(['--swap-interval', str(args.swap_interval)])
    # Only our child is terminated on timeout. This never finds/kills other
    # Python processes or an already-running Meshropractor application.
    with tempfile.TemporaryDirectory(prefix='meshropractor-profile-result-') as folder:
        # A failed worker must never pick up a successful report from an older run.
        worker_output = Path(folder) / 'result.json'
        command.extend(['--output', str(worker_output)])
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=args.seconds + 45,
                env={**os.environ, 'PYTHONUTF8': '1'})
            report = (json.loads(worker_output.read_text(encoding='utf-8')) if worker_output.exists()
                      else dict(status='error', error='The profile process did not produce a report.'))
            if completed.returncode:
                report.update(status='error', worker_exit_code=completed.returncode,
                              worker_output=(completed.stdout + completed.stderr)[-6000:])
        except subprocess.TimeoutExpired:
            report = dict(status='error', error='The diagnostic child exceeded its timeout and was stopped.',
                          timeout_seconds=args.seconds + 45)
    write_report(args.output, report)
    summary = {key: report.get(key) for key in ('status', 'page', 'elapsed_seconds', 'mouse_moves',
               'full_frames', 'frame_intervals', 'present', 'extra_python_frame', 'error')}
    summary['output'] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, separators=(',', ':')))
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
