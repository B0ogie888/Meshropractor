# Independent mesh repair engine

`MeshRepairEngine.exe` is a standalone command-line program communicating through
NumPy vertex/face files. The desktop application launches it as a separate process;
this also permits cancellation and isolates native engine crashes.

- Adapter: `src/repair_engine_cli.py`, GPL-3.0-or-later. A source copy is included in
  the helper distribution under `source/`.
- PyMeshFix 0.18.1 / MeshFix: GPL-3.0-or-later; see `GPL-3.0.txt` and the complete
  corresponding upstream source archive `pymeshfix-0.18.1.tar.gz` in this directory.
  Upstream: https://github.com/pyvista/pymeshfix and https://github.com/MarcoAttene/MeshFix.
- NumPy and VTK are used for array exchange and optional quadric simplification.
  Their package license notices are included by the packaging hooks.

Build from the repository with Python 3.12, `requirements.txt`, `requirements-dev.txt`
and `python -m PyInstaller --noconfirm RepairEngine.spec`. The adapter accepts
`--input input.npz --output output.npz --passes 3 --target-faces 400000`.
Input arrays are named `vertices` (float64 N×3) and `faces` (int32 M×3).
Set `--target-faces 0` to disable simplification. The result includes the repaired
arrays and a JSON string named `passes`. No network service is used.

This notice covers the independent helper and its dependencies; it does not assign
a license to other project files. Retain this directory and the supplied sources
when redistributing the helper.
