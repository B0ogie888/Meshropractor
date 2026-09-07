# Meshropractor

[Русский](README_RU.md) · [Changelog](CHANGELOG.md) · [Validation notes (RU)](docs/VALIDATION.md)

Meshropractor is a desktop application for preparing 3D parts and compensating
geometric deviations measured from scans. Version 0.2.5 brings together two workspaces:

- **Slicer:** STL/STEP import, part placement and transforms, section views, surface
  selection, automatic and manual supports, and geometric measurements.
- **Pre-deformation:** CAD/scan alignment, signed deviation maps, displacement-field
  training and compensated geometry exported as STL.

`.mrp` projects preserve geometry, results and settings, with session Undo/Redo.
Machine-job CLS output remains experimental and unavailable from the interface.
Capabilities and limitations are described below.

The interface uses PySide6 and PyVista/VTK; geometry calculations use Open3D and
Trimesh, STEP import uses OpenCascade, and the deformation field uses PyTorch with
Fourier features.

## Installation

For a packaged Windows build, use `Meshropractor-Setup-0.2.5-x64.exe` or copy the entire
`dist\Meshropractor` folder including `_internal`; a separate Python installation is
not required. For PyInstaller and Inno Setup packaging, see
[Windows build instructions (RU)](docs/BUILD_WINDOWS.md). Source setup follows below.

Tested on Windows with Python 3.12. Direct dependencies are pinned in `requirements.txt`.
NVIDIA GPU acceleration is optional; training falls back to CPU. Open3D raycasting
runs on CPU in batches regardless of the PyTorch training device.

```powershell
git clone https://github.com/B0ogie888/Meshropractor.git
cd Meshropractor
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install either the CUDA 12.1 build:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Or the CPU build:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies and launch:

```powershell
python -m pip install -r requirements.txt
python src/Meshropractor.py
```

## Workflow

1. Create a deformation project. CAD accepts STL/STEP/STP; scans accept STL.
   STEP units are converted to mm, with selectable tessellation deflection (default
   0.05 mm) and angular deflection in degrees (default about 14.324°), in the same dialog.
   Assembly bodies become a combined mesh with their placements preserved.
   STL coordinates are interpreted as mm; their units are not detected automatically.
2. Align automatically or place at least three non-collinear marker pairs. Registration
   uses area samples and closest CAD triangle points. Adjust the capture tolerance and
   minimum scan area fraction; results include inlier RMSE and whole-sample P95.
   Medium/long searches test 24 PCA orientations plus a local-feature candidate. Symmetric
   orientation ambiguity is reported; markers can resolve it. Cancellation is available.
3. Generate a signed deviation map; the CAD must be closed. Select a map in the
   table before placing measurement callouts.
4. Choose the network, surface sample count, deviation search limit and minimum
   coverage. Sampling affects training; disabling sampling uses the CAD vertices.
5. Calculate deformation or compensation with independent XY/Z factors.
6. Select a result to export to STL. Arrows show the actual applied displacement.
   Coverage, holdout RMSE and maximum distance to a confirmed measurement are shown
   below the results table.

Only one background operation runs at a time. Cancellation waits for the current
native geometry step rather than forcibly terminating a thread. Closing the window
waits for the task to stop.

## Slicer sections

The viewport toolbar selects triangles, connected planes, smooth patches, connected
shells, a surface brush or visible cells inside a rectangle. First select the target
parts. Selected faces are orange and follow clipping planes; Shift adds, Ctrl removes,
Alt allows camera navigation and Esc resets the tool. Face selection is temporary.
Click a part to select it; Shift/Ctrl modify part selection. Unload and selected export
use checked parts in the current scene. Right-click opens a radial move/rotate/export/
unload menu, while a right-button drag retains camera navigation. Unload supports undo.

The Supports ribbon generates columns or branching supports, places individual manual
contacts and previews downward overhang faces. Parameters include angle, spacing,
shaft/contact/foot sizes and platform Z. Selected faces can restrict the generated area.
Supports stop at the nearest surface below, or skip obstructed contacts in platform-only
mode. Support groups belong to their source part, follow its transforms and copies,
and participate in project saving, combined STL export and undo. The branching algorithm is original, not Materialise e-Stage;
branches contain intersecting closed bodies without a Boolean union. Ray checks do not
constitute full volumetric collision checks or print-process/strength validation.

Window dragging/resizing uses the native system loop for Windows Snap when enabled in
Windows settings. Double-click the title bar to maximize/restore. A status-bar Cancel
button is available during background jobs.

Manual Supports replaces the left pane with a part selector, a list of surface regions,
support types, a XY plan and editable parameters/profiles. Select faces, add a region and
rebuild its geometry. Available patterns are grid walls, walls along X, columns, diagonal
braces, perimeter contacts, truncated cones and branches. Groups can be hidden or deleted.
Choose None to keep the region without support geometry. Existing detached support parts
in older projects remain detached because those files contain no reliable parent link.

The translucent view cube aligns the camera when a face is clicked; double-click restores
an isometric view. The platform becomes 88% transparent with the camera below Z=0.
Measurements cover point distances and XYZ deltas, distance to a face's extended plane
or to the closest point of a part, parallel plane distances, three-point circles and
point/plane angles. Results appear in the pane and scene and can be copied or hidden.
Measurements are temporary and cleared on geometry changes; they use the triangle mesh,
not analytical CAD surfaces.

Import STL/STEP/STP through Import Part. Enable XY/XZ/YZ or arbitrary planes in the
Sections panel; up to six half-spaces can be combined. Choose the removed side (+/−),
position and step in mm. Move with the slider, step buttons or the interactive plane
(«Указать»); arbitrary planes can also rotate. «Выровнять» aligns the camera and
«Экспорт» exports the selected plane's contours of visible parts as VTP.

Clipping affects display only, preserving the original mesh and STL exports. No artificial
caps are generated. Planes are saved in `.mrp`; build platforms remain unclipped. STEP
becomes a triangle mesh: CAD feature editing and STEP export are not implemented.

Click a plane's cell to choose the row controlled by the slider. Hover does not change
the selected row; wheel events on unfocused fields do not edit another plane.

## Slicer tools and history

The Tools ribbon offers box/cylinder/sphere creation, duplication, XYZ copy arrays,
translation, rotation in degrees, scaling and mirroring. Check the parts in the current
scene's selection column. Rotations use world X, then Y, then Z; rotation/scale/mirror
can use group/individual/custom centers. Transform dialogs are modeless: Apply records
one history step and stays open, Yes applies and closes, Close discards only unapplied
preview. Create Copy preserves originals. Copy arrays preserve the
original cell and use explicit XYZ pitch (no collision-free packing).

Move supports linked absolute/relative coordinates, per-axis min/center/max/custom
anchors, individual origins, return to the opening position and two-point line constraints.
Rotation supports arbitrary lines, surface-picked centers and angular snapping on its
3D handles. Scale links factors, final dimensions and differences, with uniform scaling,
two-point measurement fitting and a persistent editable preset library. Mirror supports
principal or three-point/point-normal planes. Rotation and scale can preserve each part's
minimum Z. Points are picked on original surfaces; preview is temporarily hidden while
picking. Other project mutations and Undo/Redo are locked until the panel closes.
The Home ribbon now uses command icons above its labels.

Undo (`Ctrl+Z`) and Redo (`Ctrl+Y` / `Ctrl+Shift+Z`) restore project geometry, calculations,
markers and settings, including sections. Tool operations are atomic history steps.
Rapid field/slider edits coalesce until a 300 ms pause. History is session-local and
resets on New/Open; saving records a clean-state marker. Up to 30 steps are retained,
with a 256 MB geometry budget except for the current and immediately previous states.
Unchanged meshes share snapshot storage. Background jobs temporarily disable history.

## Projects

`Ctrl+S` saves the current project; Save As chooses a new path. A title-bar asterisk
indicates unsaved changes. The application offers to save before replacing a project
or closing the window.

`.mrp` 2.1 stores all results, deviation maps, vectors, markers, callouts, calculation
and display settings, slicer parts and platforms. Numeric arrays preserve vertex
indices and precision without an intermediate STL conversion. Saving replaces the
destination atomically. Invalid project files leave the current project intact.
Version 1.x and 2.0 projects can be opened; new saves use 2.1 and cannot be opened by older apps.

## Scope and limitations

STL/STEP import, STL export, section clipping, alignment, deviation analysis, neural compensation, scene/platform
management, Undo/Redo and project persistence are available. Separate report and
inspection modules, unimplemented ribbon commands and desktop CLS export are disabled.
The experimental CLS writer has not been validated against production machines.
Keep-out zones are visual guides; collision enforcement is not implemented.

The network fits observed geometry displacement; it is not a physical printing
simulation. Coverage is the fraction of rays with a valid hit, not a confidence
probability. Holdout RMSE does not guarantee manufacturing tolerances. Result checks
reject collapsed/inverted triangles and self-intersections. Large-part performance
and production accuracy require representative CAD/scan benchmarks.

## Development

```powershell
python -m unittest discover -s tests -v
python scripts/smoke_desktop.py --sections --tools
python -m pip install -r requirements-dev.txt
python -m PyInstaller Meshropractor.spec
```

The smoke script briefly opens a real Windows/OpenGL scene and writes screenshots
under `output/`. Controller tests run with real Qt/VTK objects and a non-rendering
plotter substitute for headless CI.

`project_store.py` owns archive validation and serialization; `project_controller.py`
and `background_tasks.py` coordinate lifecycle and jobs. `geometry_analysis.py`,
`ml_deformation.py` and `Workers_Meshropractor.py` implement calculations.

[Support the author on Boosty](https://boosty.to/boogie888) · Contact: theboogie888@gmail.com
