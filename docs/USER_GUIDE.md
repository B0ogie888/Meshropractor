# User guide

[Project overview](../README.md) · [Русский](USER_GUIDE_RU.md)

## Contents

- [Workflow](#workflow)
- [Display settings and help](#display-settings-and-help)
- [Slicer workspace](#slicer-workspace)
- [Slicer tools and history](#slicer-tools-and-history)
- [Projects](#projects)
- [Scope and limitations](#scope-and-limitations)

## Workflow

### Textures and colors

The **Textures** ribbon provides 15 working commands: image layers on parts or
selected STL/CAD surfaces, UV projections, layer selection/editing, copy/paste,
removal and visibility, surface painting, color baking and separation into meshes.
Images are embedded in `.mrp`; changes support Undo/Redo. Painting preserves BREP
and child supports; color separation creates meshes. Textures affect appearance
without adding printable relief. See [commands and limitations](TEXTURES.md) (Russian).

### Placement and repair

The **Placement** ribbon has 13 tools: precise transforms, mouse dragging, face
orientation, 2D/3D platform packing, orientation search and comparison, bounding-box
minimization, platform fitting without scaling, and orientation transfer to similar
parts. Background preparation, previews and Undo/Redo preserve the source meshes;
packing accounts for supports, existing parts and excluded zones.
See [placement commands and limitations](PLACEMENT.md).

The **Repair** ribbon provides 25 commands with individual icons: automatic repair,
normals, stitching, holes, duplicates and overlap detection, component separation and
Boolean union, manual triangles/bridges, plane clipping, vertex movement, reduction,
smoothing and remeshing. Operations use checked parts and prepare a background preview;
Apply commits a single undoable change. Pick vertices/faces in the scene or drag a vertex.
Wrapping and remeshing are approximate and require visual review before applying.
See the [command reference and limitations](MESH_REPAIR.md).

### Diagnostics and repair review

The **Repair wizard** is available from the slicer’s Repair tab and beside CAD/scan
import in Predeformation. Its dropdown includes every loaded part, child support group,
CAD, scan and result. Full analysis reports open boundaries, nonmanifold edges/vertices,
duplicate/degenerate faces, fragments, coplanar area overlaps and transverse intersections.
Counts use our own geometric criteria and need not match Magics.

Full repair runs in a cancellable local helper process, with optional quadric reduction
of overly dense meshes (about 400,000 faces by default). The result is independently
rechecked and compared with the original in both directions using every vertex plus
area-sampled surface points. Application is blocked if the measured distance exceeds
the configured tolerance (default 0.05 mm); this is a sampled estimate, not a certified
Hausdorff bound. Review/export the report and repaired copy before applying. Full repair
can close intentionally open surfaces and requires confirmation. Original files remain
unchanged; applying to one model supports undo. Full analysis of a multi-million-face STL
can take over ten minutes. See [repair details](MESH_REPAIR.md) and the
[independent helper source/license](../licenses/repair-engine/README.md).

Before loading, choose whether to run diagnostics and prepare repair, or import directly
without either step. Repair settings offer one to three passes and a hole diameter limit
(0.1 mm by default). Progress appears in the status bar/log; cancellation is checked
between stages and boundary components. Review the before/after counts before applying,
keeping the original mesh or canceling import. Repair welds coincident vertices
at 0.000001 mm precision and removes duplicate/degenerate faces. CAD and slicer parts
also receive caps for small planar convex holes: passes handle up to 4, 12 and 32 boundary
edges within the chosen diameter, stopping early when the mesh becomes closed;
scan boundaries are preserved. Larger/complex gaps, incorrect normals and self-intersections
are not automatically repaired by this mode. The source file is untouched and repair is
a separate Undo/Redo step. CAD from existing projects is reviewed before a deviation map
unless already reviewed in the session. A prompt precedes that work too: skip repair to
continue the map immediately, or cancel the calculation.

### Viewport performance

While rotating large models (100,000 faces or more), edge overlays are temporarily hidden
and restored on release. Geometry is not decimated. Static scenes render on changes and
the view cube renders in the same frame as the scene without a separate native window. See [performance measurements (RU)](PERFORMANCE.md).

### Pre-deformation

1. Create a deformation project. CAD accepts STL/STEP/STP; scans accept STL.
   STEP units are converted to mm, with selectable tessellation deflection (default
   0.05 mm) and angular deflection in degrees (default about 14.324°), in the same dialog.
   Assembly bodies remain one nominal CAD model, with BREP retained in native mode and a combined mesh for calculations; their placements are preserved.
   STL coordinates are interpreted as mm; their units are not detected automatically.
2. Align automatically or place at least three non-collinear marker pairs. Registration
   uses area samples and closest CAD triangle points. Adjust the capture tolerance and
   minimum scan area fraction; results include inlier RMSE and whole-sample P95.
   Medium/long searches test 24 PCA orientations plus a local-feature candidate. Symmetric
   orientation ambiguity is reported; markers can resolve it. Cancellation is available.
3. Generate a deviation map. Closed CAD uses inside/outside signs; open CAD uses
   closest-surface distance signed by the CAD face normal, with a notice in the log.
   This local sign can be ambiguous near gaps and sharp edges. Scans may be open.
   Select a map in the
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

## Display settings and help

The slicer's Settings and Help ribbon provides display preferences, keyboard shortcuts,
help, application information and a manual update check. Preferences include the scene
background, edge anti-aliasing (MSAA ×4 by default, ×8, FXAA or off) and hiding dense-mesh
edges during camera interaction. They persist between sessions and apply to both workspaces.
Anti-aliasing does not change geometry; smoother CAD silhouettes require finer tessellation
at import. The view cube's colored axes remain attached to one corner while rotating.

## Slicer workspace

### Surface selection

The viewport toolbar selects triangles, connected planes, smooth patches, connected
shells, a surface brush or visible cells inside a rectangle. First select the target
parts. Selected faces are orange and follow clipping planes; Shift adds, Ctrl removes,
Alt allows camera navigation and Esc resets the tool. Face selection is temporary.
Click a part to select it; Shift/Ctrl modify part selection. Unload and selected export
use checked parts in the current scene. Right-click opens a radial move/rotate/export/
unload menu, while a right-button drag retains camera navigation. Unload supports undo.

Holding the right button shows an unfilled dashed circle in both workspaces.
Start a right-drag inside it to orbit in 3D; start outside it to roll the view
clockwise/counterclockwise in the screen plane. The gesture mode stays fixed
until release, even if the pointer crosses the circle. This also works while
selecting surfaces, measuring or placing supports; Alt is optional for navigation.
Selection rectangles and the build-volume frame contain outlines only.

### Support generation

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

### Manual supports

Manual Supports replaces the left pane with a part selector, a list of surface regions,
support types, a XY plan and editable parameters/profiles. Select faces, add a region and
rebuild its geometry. Available patterns are grid walls, walls along X, columns, diagonal
braces, perimeter contacts, truncated cones and branches. Groups can be hidden or deleted.
Choose None to keep the region without support geometry. Existing detached support parts
in older projects remain detached because those files contain no reliable parent link.

### View cube and measurements

The view cube in both workspaces uses pale faces and thin edges without a rectangular
panel covering the viewport. Clicking a face aligns the camera; double-click restores
an isometric view. The platform becomes 88% transparent with the camera below Z=0.
Measurements cover point distances and XYZ deltas, distance to a face's extended plane
or to the closest point of a part, parallel plane distances, three-point circles and
point/plane angles. Results appear in the pane and scene and can be copied or hidden.
Measurements are temporary and cleared on geometry changes; they use the triangle mesh,
not analytical CAD surfaces.

### Import and sections

Import STL/STEP/STP through Import Part or drag files from Explorer into the active
viewport. The STEP dialog offers CAD/BREP or a plain STL mesh, with tessellation quality.
In navigation mode, double-click a part to orbit around its center; double-click empty
space to return to the build-plate center. Table selection does not change the pivot.
Use Alt + double-click while selecting surfaces or measuring.
Enable XY/XZ/YZ or arbitrary planes in the
Sections panel; up to six half-spaces can be combined. Choose the removed side (+/−),
position and step in mm. Move with the slider, step buttons or the interactive plane
(«Указать»); arbitrary planes can also rotate. «Выровнять» aligns the camera and
«Экспорт» exports the selected plane's contours of visible parts as VTP.

Clipping affects display only, preserving the original mesh and STL exports. No artificial
caps are generated. Planes are saved in `.mrp`; build platforms remain unclipped.

### Native CAD / STEP

STEP imports retain BREP by default. The slicer can import bodies as separate parts;
predeformation keeps an assembly together as the nominal CAD model. Select complete
CAD faces using the CAD button above the scene and generate supports owned by that part.
The CAD / STEP panel offers retessellation, body separation, exact CAD properties,
STEP export and explicit conversion to mesh. Placement, duplication, scaling and mirroring
retain BREP; mesh edits invalidate it and prevent exporting stale CAD. Projects embed
the BREP source. Calculations and supports use a mesh at the chosen tolerance.
See [CAD workflow and limitations](CAD.md).

### Display ribbon

25 functional commands cover camera views, smooth shading, simplified display, grids,
rulers, zones, bounds, mass centers, labels, colors, overhangs, geometric checks,
volume/material/packing estimates, PNG export, clipboard and printing. The three
statistics commands form a vertical list and update the top-right overlay for selected
parts and their supports. Set material density and price using the cost button's arrow.
Click a part to select it, click empty space to clear selection, or drag a rectangle
from empty space to select several parts. Shift adds, Ctrl toggles, and Alt lets you
rotate the scene from empty space. Display toggles
do not modify source geometry. Geometric checks are not a print simulation.
See [Display commands](DISPLAY.md).

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
Automatic packing accounts for keep-out zones using conservative bounding boxes. Manual transforms do not enforce collision avoidance.

The network fits observed geometry displacement; it is not a physical printing
simulation. Coverage is the fraction of rays with a valid hit, not a confidence
probability. Holdout RMSE does not guarantee manufacturing tolerances. Result checks
reject collapsed/inverted triangles and self-intersections. Large-part performance
and production accuracy require representative CAD/scan benchmarks.
