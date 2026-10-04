<p align="center">
  <img src="assets/logo.png" alt="Meshropractor" width="112">
</p>

# Meshropractor

**CAD and mesh preparation for additive manufacturing, with scan-based geometry compensation.**

[![Tests](https://github.com/B0ogie888/Meshropractor/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/B0ogie888/Meshropractor/actions/workflows/tests.yml)
[![Version](https://img.shields.io/badge/version-0.3-2563eb)](CHANGELOG.md)
[![Platform](https://img.shields.io/badge/platform-Windows_x64-475569)](#installation)
[![Linux](https://img.shields.io/badge/Linux-amd64-FCC624?logo=linux&logoColor=black)](#linux-application)
[![Python](https://img.shields.io/badge/python-3.12-3776ab)](requirements.txt)

[Русский](README_RU.md) · [Releases](https://github.com/B0ogie888/Meshropractor/releases) · [User guide](docs/USER_GUIDE.md) · [Changelog](CHANGELOG.md)

Meshropractor brings part preparation and geometric compensation into one desktop
application. Import STL or STEP, repair and arrange parts, select CAD surfaces for
supports, then compare a nominal model with a scan and export compensated geometry.

Two workspaces share project storage and geometry tools:

- **Slicer:** model preparation, build platforms, supports, sections and measurements.
- **Pre-deformation:** CAD/scan alignment, deviation analysis and displacement-field training.

## Contents

- [Capabilities](#capabilities)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Controls](#controls)
- [Formats](#formats)
- [Documentation](#documentation)
- [Development](#development)
- [Scope and limitations](#scope-and-limitations)
- [Feedback and support](#feedback-and-support)

## Capabilities

| Area | What you can do |
| --- | --- |
| **Native CAD** | Retain STEP BREP bodies and faces, change tessellation quality, select whole CAD faces and export transformed CAD as STEP. |
| **Mesh repair** | Diagnose boundaries, normals, fragments, overlaps and intersections; prepare automatic or manual repairs with preview and Undo/Redo. |
| **Placement** | Move, rotate, scale, mirror and duplicate parts; arrange them on a platform or in its volume and compare orientations. |
| **Creation and copies** | Build ten geometric shapes from dimensions and tessellation settings; arrange a matrix of copies with live translucent previews. |
| **Marking** | Define a surface area for text, images or Data Matrix; retain editable previews, merge relief into a part or export it. |
| **Supports** | Generate supports or define surface regions manually. Support groups belong to their part and follow its transforms. |
| **Textures and colors** | Apply image layers to STL/CAD surfaces, edit projections, copy textures, paint faces and split meshes by color. Images and mappings are embedded in projects with Undo/Redo. |
| **Analysis and reports** | Check volumetric intersections, sample wall thickness and slice areas, estimate time/cost, record actual measurements and export HTML/PDF/JSON/CSV reports. |
| **Inspection** | Combine clipping planes, measure geometry and view live selected-part volume, material cost and packing statistics. |
| **Compensation** | Align CAD and scan, build deviation maps, train a displacement field and export a compensated STL with independent XY/Z factors. |
| **Projects** | Save models, BREP, supports, platforms and calculation results in `.mrp`; undo and redo changes during the session. |

## Installation

### Windows application

Download a published Windows x64 installer from [Releases](https://github.com/B0ogie888/Meshropractor/releases).
Packaged builds do not require a separate Python installation. A portable distribution
uses the complete `dist\Meshropractor` folder, including `_internal`.

The current version is **0.3**, defined in [VERSION](VERSION). Published installers may
lag behind the source version. To build this version locally, follow the
[Windows packaging guide](docs/BUILD_WINDOWS.md) (Russian).

On Windows, the application checks stable releases for updates at startup. Downloading shows
progress and can be canceled; installation requires a separate confirmation.

### Linux application

Version **0.3** builds as `meshropractor_0.3_amd64.deb` for **Debian 12 and
Ubuntu 24.04, x86-64**. Python and the application dependencies are included;
the package uses CPU PyTorch. The desktop session needs OpenGL and X11, or XWayland
under Wayland.

Install a built package from its directory:

```bash
sudo apt install ./meshropractor_0.3_amd64.deb
meshropractor
```

It also appears in the application menu. Linux updates are installed through APT.
Startup and manual checks compare Linux package versions,
show the current version and offer the matching release page. See the [Linux packaging guide](docs/BUILD_LINUX.md)
(Russian) for building the package with Docker and validating it in a clean system.
Published release assets may lag behind the available source builds.

### Run from source

Meshropractor 0.3 uses one engineering interface with light and dark themes.
After installing dependencies, open `Meshropractor.pyw` in the repository root,
or run `python src/Meshropractor.py`. Existing `.mrp` projects remain supported;
previous interface preferences are migrated on first launch. An animated startup
screen shows real initialization stages. [Interface details](docs/UI_DESIGN.md) (Russian).
These changes are not yet included in a published installer.

The interface offers a compact command ribbon, contextual part properties and
light/dark themes. Drag arrows or XY/XZ/YZ planes extending from the origin to
the arrowheads to move parts; use axis rings or the outer screen ring to rotate.
Move anchor preferences are remembered for subsequent parts.

Builds use **Python 3.12**, Windows x64 or Linux amd64. A CUDA-capable NVIDIA GPU is optional;
training also runs on CPU. Dependencies are pinned in [requirements.txt](requirements.txt).

From PowerShell:

```powershell
git clone https://github.com/B0ogie888/Meshropractor.git
cd Meshropractor
py -3.12 -m venv .venv
```

Install **one** PyTorch build:

```powershell
# CPU
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Or, for NVIDIA CUDA 12.1:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies and launch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/Meshropractor.py
```

On Linux with Python 3.12 and the Qt/OpenGL system libraries installed:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/Meshropractor.py
```

## Quick start

### Prepare a part

1. Open the **Slicer** workspace and create a build platform with the required dimensions.
2. Import a model or drag STL/STEP files into the active viewport. For STEP, choose native BREP or a mesh and set tessellation tolerances.
3. Select parts in the scene. Use **Repair** for diagnostics and **Placement** for transforms or automatic arrangement.
4. Select surfaces and generate supports, or configure regions through **Manual Supports**.
5. Inspect sections and measurements, save the `.mrp` project and export the selected geometry.

### Compensate measured deviations

1. Open **Pre-deformation**, load nominal CAD (STL/STEP) and a scan (STL).
2. Align automatically or specify at least three non-collinear marker pairs.
3. Build a deviation map and review alignment quality and coverage.
4. Configure training and compensation factors, calculate the result and export it as STL.

See the [user guide](docs/USER_GUIDE.md) for tool parameters, repair review and result interpretation.

## Controls

These selection gestures apply to the slicer's **part selection** mode:

| Gesture | Action |
| --- | --- |
| Left-click a part / empty space | Select a part / clear all part selection. |
| Drag from empty space | Select parts with a rectangle. |
| Shift / Ctrl + click or rectangle | Add parts / toggle part selection. |
| Double-click a part / empty space | Orbit around the part / return to the build-plate center. |
| Alt + left-drag | Rotate the camera, including from empty space. |
| Right-drag starting inside the dashed circle | Orbit the scene in 3D, including during surface selection and measurements. |
| Right-drag starting outside the circle | Rotate the view clockwise/counterclockwise in the screen plane. |
| Click a view-cube face | Align the camera to that face. |
| Ctrl+S | Save the project. |
| Ctrl+Z / Ctrl+Y or Ctrl+Shift+Z | Undo / redo. |

Surface-selection tools use **Ctrl to subtract faces**. Full controls are available
in **Settings and Help → Keyboard shortcuts and controls** and the [user guide](docs/USER_GUIDE.md).

## Formats

| Format | Import | Export / storage |
| --- | --- | --- |
| **STL** | Slicer parts, nominal CAD and scans. Coordinates are interpreted as millimeters. | Parts with supports and deformation/compensation results. |
| **STEP / STP** | Native CAD bodies and faces, or a triangulated mesh; file units convert to millimeters. | Selected CAD bodies with their transforms while BREP remains valid. |
| **MRP 2.1** | Saved projects, including older 1.x and 2.0 archives. | Geometry, BREP, supports, platforms, results and settings. |
| **VTP / PNG** | — | Section contours / scene images. |

Mesh edits can invalidate BREP. Supports and compensated results are triangle meshes;
STEP export preserves the nominal CAD rather than converting those meshes to CAD.

## Documentation

| Guide | Topics | Language |
| --- | --- | --- |
| [User guide](docs/USER_GUIDE.md) | Workflows, supports, measurements, transforms and project history. | English |
| [Руководство пользователя](docs/USER_GUIDE_RU.md) | Полное описание рабочих инструментов. | Русский |
| [CAD / STEP](docs/CAD.md) | Bodies, faces, BREP, tessellation and STEP export. | Русский |
| [Mesh repair](docs/MESH_REPAIR.md) | Diagnostics, repair commands, tolerances and result review. | Русский |
| [Placement](docs/PLACEMENT.md) | Arrangement, orientation search and packing criteria. | Русский |
| [Create a part](docs/PRIMITIVES.md) | Shapes, dimension diagrams and tessellation settings. | Русский |
| [Matrix duplication](docs/DUPLICATION.md) | Copy counts, gaps and live virtual previews. | Русский |
| [Marking](docs/MARKING.md) | Editable surface areas, content, projection and relief. | Русский |
| [Interface design](docs/UI_DESIGN.md) | Themes, contextual panels, startup and application identity. | Русский |
| [Editing and structures](docs/MODEL_TOOLS.md) | Hollowing, cuts, perforations, booleans, text, lattices and fixtures. | Русский |
| [Display](docs/DISPLAY.md) | Sections, scene annotations and statistics. | Русский |
| [Textures and colors](docs/TEXTURES.md) | Image layers, projections, surface painting and color separation. | Русский |
| [Analysis and reports](docs/ANALYSIS_REPORTS.md) | Inspection methods, estimates, measurements and report formats. | Русский |
| [Ribbon icons](assets/ribbon/README.md) | Editable SVG files and shared icon sizing. | Русский |
| [Validation](docs/VALIDATION.md) | Tests, native scene checks and accuracy interpretation. | Русский |
| [Performance](docs/PERFORMANCE.md) | Rendering changes and benchmark methodology. | Русский |
| [Windows build](docs/BUILD_WINDOWS.md) | PyInstaller, Inno Setup and release packaging. | Русский |
| [Linux build](docs/BUILD_LINUX.md) | Debian/Ubuntu `.deb`, Docker builds and clean installation checks. | Русский |

## Development

See [repository structure and release workflow](docs/DEVELOPMENT.md) (Russian)
for entry points, script directories and artwork sources.

After setting up the source environment, run the regression suite:

```powershell
.\.venv\Scripts\python.exe scripts/validation/check_repository.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Native scene checks briefly open an application window and save reports and images in `output/`:

```powershell
.\.venv\Scripts\python.exe scripts/validation/smoke_cad_display.py
.\.venv\Scripts\python.exe scripts/validation/smoke_selection_statistics.py
```

Version 0.3 passed **417 local regression tests**, plus native Qt/VTK checks of
the themes, transforms, primitives and marking. GitHub Actions runs the regression suite on Windows; its current
status appears in the badge above. Test scope is documented in [Validation](docs/VALIDATION.md).

| Location | Responsibility |
| --- | --- |
| `src/` | Qt/VTK interface, CAD and mesh operations, background jobs and compensation. |
| `tests/` | Geometry, controller and UI regression tests. |
| `scripts/validation/` | Source, native scene and packaged application checks. |
| `scripts/benchmarks/` | Viewport profiling and grid rendering diagnostics. |
| `scripts/assets/` | SVG export and application icon packaging. |
| `packaging/linux/` | Linux build environment, Debian package and clean installation checks. |
| `docs/` | User and developer documentation. |
| `assets/` | Application icons and resources. |
| `licenses/repair-engine/` | Independent repair helper sources and license notices. |

For packaging, install [requirements-dev.txt](requirements-dev.txt):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Build the repair helper before the application; exact commands are in the
[Windows](docs/BUILD_WINDOWS.md) and [Linux](docs/BUILD_LINUX.md) build guides.

## Scope and limitations

- BREP is preserved for CAD operations; alignment, supports, measurements and deviation calculations use its tessellation.
- Packing uses conservative bounding boxes. Support geometry and compensation require review against the intended printing process.
- Compensation fits measured displacement; it does not simulate printing physics. Reported metrics do not certify manufacturing tolerances.
- Desktop CLS machine-job export remains disabled pending equipment validation. This application is not a complete Materialise Magics replacement.

Algorithm-specific limits are described in the linked guides. License and source
notices for the independent repair helper are in [licenses/repair-engine](licenses/repair-engine/README.md).

## Feedback and support

Report a bug or propose a feature through [GitHub Issues](https://github.com/B0ogie888/Meshropractor/issues).
For bugs, include the application version, steps to reproduce, expected behavior and
the relevant log excerpt.

Windowed builds write logs to `%LOCALAPPDATA%\Meshropractor\logs\Meshropractor.log`.
For a code change, describe the problem and validation in a pull request.

[Support the author on Boosty](https://boosty.to/boogie888) · [Email](mailto:theboogie888@gmail.com)

Scan the QR code to open the author's Boosty page:

<a href="https://boosty.to/boogie888"><img src="assets/qr_donate.png" alt="QR code — support Meshropractor on Boosty" width="200" height="200"></a>
