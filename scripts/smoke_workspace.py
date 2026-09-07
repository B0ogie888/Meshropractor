"""Real Windows/VTK interaction regression for surface selection and contextual actions."""
import sys
from pathlib import Path
import traceback
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer, QPoint, QRect
from PySide6.QtTest import QTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Meshropractor import MainWindow
from project_store import ProjectState
from support_tools import SupportDialog, DEFAULTS
from support_geometry import generate_supports


app = QApplication([])
app.setStyle('Fusion')
window = MainWindow()
window.resize(1600, 950)
window.show()
failed = []
output = Path('output').resolve()
output.mkdir(exist_ok=True)


def check():
    try:
        cube = trimesh.creation.box([12, 12, 4])
        cube.apply_translation([0, 0, 15])
        window.restore_project(ProjectState(parts=[dict(mesh=cube, filename='Bracket.step', style={'last_visible_mode': 'shaded'})], page='slicer'))
        plotter = window.ui.slicer_plotter
        work = window.workspace_tools
        plotter.camera_position = [(27, 35, -15), (0, 0, 15), (0, 0, 1)]
        plotter.reset_camera()
        app.processEvents()

        def screen(point):
            plotter.renderer.SetWorldPoint(*point, 1)
            plotter.renderer.WorldToDisplay()
            x, y, _ = plotter.renderer.GetDisplayPoint()
            ratio = plotter.devicePixelRatioF()
            return QPoint(round(x / ratio), round((plotter.render_window.GetSize()[1] - 1 - y) / ratio))

        position = screen([1, 1, 13])
        hit = work.picker(position)
        assert hit is not None, 'Real cell picker missed the underside'
        work.buttons['triangle'].click()
        QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, position)
        assert len(work.selection[0]) == 1, 'Triangle click did not select one face'
        work.buttons['plane'].click()
        QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, position)
        assert len(work.selection[0]) == 2, 'Plane click did not grow across the diagonal'
        plotter.screenshot(str(output / 'workspace-selection-scene.png'))
        window.grab().save(str(output / 'workspace-selection-window.png'))
        work.buttons['brush'].click()
        work.radius.setValue(20)
        QTest.mouseClick(plotter, Qt.LeftButton, Qt.ControlModifier, position)
        assert not work.selection, 'Subtract brush failed'
        work.buttons['rectangle'].click()
        rect = QRect(position - QPoint(100, 100), position + QPoint(100, 100))
        QTest.mousePress(plotter, Qt.LeftButton, Qt.NoModifier, rect.topLeft())
        QTest.mouseMove(plotter, rect.bottomRight())
        QTest.mouseRelease(plotter, Qt.LeftButton, Qt.NoModifier, rect.bottomRight())
        assert work.selection, 'Hardware rectangle selection is empty'
        window.ui.section_panel.table.cellWidget(0, 0).setChecked(True)
        assert work.overlays[0][0].GetMapper().GetNumberOfClippingPlanes() == 1
        window.ui.section_panel.table.cellWidget(0, 0).setChecked(False)
        work.set_mode('part')
        QTest.mouseClick(plotter, Qt.RightButton, Qt.NoModifier, position)
        app.processEvents()
        assert work.menu and work.menu.isVisible(), 'Right click did not open the radial menu'
        work.menu.grab().save(str(output / 'workspace-radial.png'))
        work.menu.close()
        work.clear_selection()
        work.supports.preview_regions({0: None})
        plotter.screenshot(str(output / 'workspace-support-regions.png'))
        work.supports.clear_preview()
        records = work.supports.records()
        results = generate_supports(records, {0: None}, DEFAULTS, branching=True)
        assert results, 'No support geometry'
        work.supports.append_results(results)
        plotter.reset_camera()
        plotter.screenshot(str(output / 'workspace-supports-scene.png'))
        work.manual = dict(rows=[0], records=records, worlds={})
        # Pick a new contact at the corner of the original underside.
        work.supports.manual_at(screen([-4, -4, 13]))
        assert len(window.slicer_parts) == 1, 'Supports must not create extra part rows'
        assert len(window.slicer_parts[0]['supports']) == 2, 'Manual support was not attached'
        work.manual = None
        work.supports.open(3)
        panel = work.supports.panel
        app.processEvents()
        assert not panel.isHidden(), 'Manual support panel did not open'
        panel.grab().save(str(output / 'workspace-manual-panel.png'))
        window.grab().save(str(output / 'workspace-manual-window.png'))
        panel.settings_tabs.setCurrentIndex(1)
        app.processEvents()
        panel.grab().save(str(output / 'workspace-support-parameters.png'))
        panel.settings_tabs.setCurrentIndex(0)
        # Actual click on a projected face of the transparent cube.
        cube_widget = work.cube
        cube_widget.grab().save(str(output / 'workspace-view-cube.png'))
        face = next((item for item in cube_widget.faces if item[1:] == (2, -1)), cube_widget.faces[-1])
        QTest.mouseClick(cube_widget, Qt.LeftButton, Qt.NoModifier, face[0].boundingRect().center().toPoint())
        direction = np.asarray(plotter.camera.position) - plotter.camera.focal_point
        assert abs(direction[face[1]]) / np.linalg.norm(direction) > .99, 'View cube did not align camera'
        window.draw_platform(dict(dim=[25,25,40]))
        cube_widget.orient(2, -1)
        cube_widget.camera_changed()
        assert plotter.actors['plat_base'].GetProperty().GetOpacity() == .12, 'Platform is opaque from below'
        plotter.screenshot(str(output / 'workspace-platform-below.png'))
        cube_widget.orient(2, 1)
        assert plotter.actors['plat_base'].GetProperty().GetOpacity() == 1., 'Platform did not regain opacity'
        cube_widget.orient(2, -1)
        measurements = work.measurements
        measurements.tabs.setCurrentIndex(1)
        measurements.start()
        for point in ([2,0,13], [0,2,13], [-2,0,13]):
            QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, screen(point))
        assert measurements.results.count() == 1, 'Three-point circle did not complete'
        assert 'R=' in measurements.results.item(0).text()
        measurements.tabs.setCurrentIndex(0)
        measurements.start()
        for point in ([-3,-2,13], [3,-2,13]):
            QTest.mouseClick(plotter, Qt.LeftButton, Qt.NoModifier, screen(point))
        assert measurements.results.count() == 2, 'Point-to-point measurement did not complete'
        plotter.screenshot(str(output / 'workspace-measurements-scene.png'))
        measurements.grab().save(str(output / 'workspace-measurements-panel.png'))
        measurements.stop()
        panel.finish()
        window.ui.magics_ribbon.setCurrentIndex(6)
        app.processEvents()
        window.ui.magics_ribbon.grab().save(str(output / 'workspace-support-ribbon.png'))
        if sys.platform == 'win32':
            import ctypes
            get_style = ctypes.windll.user32.GetWindowLongPtrW
            get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
            get_style.restype = ctypes.c_ssize_t
            style = get_style(int(window.winId()), -16)
            assert style & 0x40000 and style & 0x10000, 'Native resize/maximize styles missing'
        print('WORKSPACE_SMOKE_OK')
    except Exception:
        failed.append(traceback.format_exc())
        print(failed[-1])
    finally:
        window.dirty = False
        window.close()
        app.quit()


QTimer.singleShot(800, check)
app.exec()
sys.exit(bool(failed))
