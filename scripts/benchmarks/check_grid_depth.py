"""Render the same grid with/without its white plate to expose depth artifacts."""
from pathlib import Path
import json
import argparse
import sys
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from display_tools import DisplayTools


class GridScene:
    def __init__(self, plotter):
        self.plotter = plotter
        self.actors = []

    def _mesh_actor(self, name, data, **kwargs):
        actor = self.plotter.add_mesh(data, name=name, reset_camera=False, **kwargs)
        self.actors.append(actor)
        return actor

    def _label(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='store_true', help='Reproduce the previous renderer without the fix')
    args = parser.parse_args()
    output = ROOT / 'output/grid-depth' / ('before' if args.baseline else 'after')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
    try:
        plotter.set_background('white')
        plotter.enable_anti_aliasing('msaa', multi_samples=4)
        scene = GridScene(plotter)
        plate = plotter.add_mesh(pv.Plane(i_size=220, j_size=220, i_resolution=1, j_resolution=1),
                                 color='white', lighting=False, show_edges=False, name='plat_base')
        DisplayTools._build_metric_grid(scene, np.array([[-110., -110., 0.], [110., 110., 280.]]))
        if args.baseline:
            plate.GetShaderProperty().ClearAllShaderReplacements()
        views = (
            ('top', (0, 0, 700), 145),
            ('slanted', (210, -370, 350), 145),
            ('grazing', (320, -360, 90), 145),
            ('far', (620, -720, 420), 400),
            ('below', (210, -370, -350), 145),
        )
        for quality, samples in (('msaa', 4), ('msaa', 8), ('fxaa', 0), ('none', 0)):
            plotter.disable_anti_aliasing()
            if quality != 'none': plotter.enable_anti_aliasing(quality, multi_samples=samples)
            for parallel in (True, False):
                for name, position, scale in views:
                    plotter.camera.position = position
                    plotter.camera.focal_point = (0, 0, 0)
                    plotter.camera.up = (0, 1, 0) if name == 'top' else (0, 0, 1)
                    plotter.camera.parallel_projection = parallel
                    plotter.camera.parallel_scale = scale
                    plotter.reset_camera_clipping_range()
                    plate.prop.opacity = .12 if name == 'below' else 1.
                    plate.SetVisibility(False)
                    plotter.render()
                    reference = plotter.screenshot()
                    plate.SetVisibility(True)
                    plotter.render()
                    actual = plotter.screenshot()
                    if quality == 'msaa' and samples == 4 and parallel:
                        from PIL import Image
                        Image.fromarray(reference).save(output / f'{name}-without-plate.png')
                        Image.fromarray(actual).save(output / f'{name}-with-plate.png')
                    error = np.abs(reference.astype(float) - actual.astype(float)).max(axis=2)
                    grid_pixels = np.count_nonzero(reference.min(axis=2) < 245)
                    mismatch = np.count_nonzero(error > 8) / max(1, grid_pixels)
                    # A translucent white plate below the grid intentionally tints
                    # a few shared pixels; look for line loss above opaque plate.
                    if name != 'below' and not args.baseline:
                        assert mismatch < .01, (quality, samples, parallel, name, mismatch)
                    results.append(dict(quality=quality, samples=samples, parallel=parallel,
                                        view=name, mismatched_grid_fraction=mismatch,
                                        max_error=float(error.max())))
        # The depth fix must not draw the grid through parts resting on the plate.
        plate.prop.opacity = 1.
        plotter.camera.position = (0, 0, 700)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 145
        model = plotter.add_mesh(pv.Cube(center=(0, 0, 10), x_length=40, y_length=40, z_length=20),
                                color='#b33b32', lighting=False)
        plotter.reset_camera_clipping_range()
        plotter.render()
        with_grid = plotter.screenshot()
        for actor in scene.actors: actor.SetVisibility(False)
        plotter.render()
        without_grid = plotter.screenshot()
        np.testing.assert_array_equal(with_grid[330:370, 430:470], without_grid[330:370, 430:470])
        print(f'{len(results)} rendering comparisons; occlusion check passed. '
              f'Maximum grid loss above plate: {max(r["mismatched_grid_fraction"] for r in results if r["view"] != "below"):.4f}')
        (output / 'comparison.json').write_text(json.dumps(results, indent=2))
    finally:
        plotter.close()


if __name__ == '__main__':
    main()
