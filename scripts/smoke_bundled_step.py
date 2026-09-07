"""Run with python -I -S: prove STEP works with only the packaged OCP, not the venv."""
from pathlib import Path
import sys
import tempfile
root = Path(__file__).resolve().parents[1] / 'dist' / 'Meshropractor' / '_internal'
sys.path.insert(0, str(root))
import OCP
assert Path(OCP.__file__).resolve().is_relative_to(root)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
from OCP.IFSelect import IFSelect_RetDone
with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / 'test.step')
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(2, 3, 4).Shape(), STEPControl_AsIs)
    assert writer.Write(path) == IFSelect_RetDone
    reader = STEPControl_Reader()
    assert reader.ReadFile(path) == IFSelect_RetDone
    reader.TransferRoots()
    assert not reader.OneShape().IsNull()
print('BUNDLED_STEP_OK:', OCP.__file__)
