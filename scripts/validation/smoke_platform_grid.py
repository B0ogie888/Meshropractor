"""White build plate and origin-aligned 1/10 mm grid in the native renderer."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
import numpy as np
import trimesh
from PySide6.QtCore import QSettings
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import main_window
import app_updater
from display_tools import PREFIX
from project_store import ProjectState


def main():
    app = QApplication([]); app.setStyle('Fusion')
    output = ROOT/'output/platform-grid-smoke'; output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', checks=[])
    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        def settings(team, name):
            result = QSettings(str(Path(folder)/(name+'.ini')), QSettings.IniFormat)
            result.setFallbacksEnabled(False); return result
        patches.enter_context(patch.object(main_window, 'QSettings', settings))
        for method in ('start', 'check', 'manual_check'):
            patches.enter_context(patch.object(app_updater.UpdateController, method, lambda *args: None))
        window = main_window.MainWindow(); ui = window.ui
        window.resize(1600, 1000); window.show()
        try:
            part = trimesh.creation.box([20, 10, 10]); part.apply_translation([25, 25, 5])
            window.restore_project(ProjectState(page='slicer',
                platforms=[dict(name='220 mm', dim=[220, 220, 280], is_default=True),
                           dict(name='Rectangle', dim=[225, 143, 180], is_default=True)],
                parts=[dict(mesh=part, filename='20 x 10 mm.stl', platform='220 mm')]))
            ui.scene_tabs.setCurrentIndex(1); QTest.qWait(100)
            plotter, display = ui.slicer_plotter, window.display_tools
            ui.magics_ribbon.setCurrentIndex(next(i for i in range(ui.magics_ribbon.count())
                                                if ui.magics_ribbon.tabText(i) == 'ОТОБРАЖЕНИЕ'))
            assert not display.state['grid']
            original_points = window.slicer_parts[0]['mesh'].vertices.copy()
            original_faces = window.slicer_parts[0]['mesh'].faces.copy()
            for mode in ('light', 'dark'):
                window.engineering_theme.set_mode(mode)
                plotter.camera.position = (0, 0, 500); plotter.camera.focal_point = (0, 0, 0)
                plotter.camera.up = (0, 1, 0); plotter.camera.parallel_projection = True
                plotter.camera.parallel_scale = 145
                plotter.reset_camera_clipping_range(); plotter.render(); QTest.qWait(80)
                plate = plotter.actors['plat_base']; prop = plate.GetProperty()
                assert prop.GetColor() == (1., 1., 1.) and not prop.GetEdgeVisibility()
                assert not prop.GetLighting() and not plate.GetPickable()
                assert plate.mapper.dataset.n_cells == 1
                plotter.screenshot(str(output/f'plate-{mode}.png'))
                ui.display_buttons['grid'].click(); QTest.qWait(80)
                major = display.overlays[PREFIX+'grid_major']; minor = display.overlays[PREFIX+'grid_minor']
                assert major.mapper.dataset.n_cells == 46 and minor.mapper.dataset.n_cells == 396
                assert major.prop.line_width > minor.prop.line_width
                assert major.prop.opacity > minor.prop.opacity
                assert not major.GetPickable() and not minor.GetPickable()
                pixels = plotter.screenshot(str(output/f'grid-{mode}.png'))
                assert pixels.std() > 10
                # Camera movement must not rebuild the hundreds of grid lines.
                count = display.rebuild_count; actors = (major, minor)
                for _ in range(5): plotter.camera.Azimuth(6); plotter.render()
                assert display.rebuild_count == count
                assert all(display.overlays[PREFIX+name] is actor
                           for name, actor in zip(('grid_major', 'grid_minor'), actors))
                plotter.camera.position = (200, -230, -150); plotter.camera.up = (0, 0, 1)
                plotter.reset_camera_clipping_range(); plotter.render(); QTest.qWait(40)
                assert plate.prop.opacity == .12
                plotter.screenshot(str(output/f'underside-{mode}.png'))
                ui.display_buttons['grid'].click(); QTest.qWait(40)
                assert not any(name.startswith(PREFIX+'grid') for name in display.overlays)
                assert not plate.prop.show_edges
            np.testing.assert_array_equal(window.slicer_parts[0]['mesh'].vertices, original_points)
            np.testing.assert_array_equal(window.slicer_parts[0]['mesh'].faces, original_faces)
            ui.display_buttons['grid'].click()
            ui.scene_tabs.setCurrentIndex(2); QTest.qWait(60)
            for name in ('grid_major', 'grid_minor'):
                points = display.overlays[PREFIX+name].mapper.dataset.points
                np.testing.assert_allclose(points.min(axis=0)[:2], [-112.5, -71.5])
                np.testing.assert_allclose(points.max(axis=0)[:2], [112.5, 71.5])
            report['checks'] = ['White unlit single-quad plate without edges in both themes',
                '220 mm plate: 46 major and 396 minor segments, separate opacity/width, non-pickable',
                'Grid toggles cleanly; part mesh unchanged; camera motion reuses cached actors',
                'Transparent plate from below and grid clipping on a 225 x 143 mm platform']
            report['status'] = 'passed'
        except Exception:
            import traceback
            report['status'] = 'failed'; report['error'] = traceback.format_exc(); raise
        finally:
            (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            window.dirty = False; window.close(); window.deleteLater(); app.processEvents()
    print('PLATFORM_GRID_SMOKE_OK')


if __name__ == '__main__': main()
