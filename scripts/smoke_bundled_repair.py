"""Check the packaged R-tree DLL and independent native repair executable."""
from pathlib import Path
import subprocess
import tempfile
import sys
import numpy as np
import trimesh

root=Path(__file__).resolve().parents[1]
bundle=root/'dist'/'Meshropractor'/'_internal'
with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder)
    # Isolated interpreter: load R-tree only from the application distribution.
    code="""from pathlib import Path
import sys
root=Path(sys.argv[1]); sys.path.insert(0,str(root))
import rtree
assert Path(rtree.__file__).resolve().is_relative_to(root)
index=rtree.index.Index(); index.insert(7,(0,0,1,1))
assert list(index.intersection((.2,.2,.3,.3)))==[7]
print('BUNDLED_RTREE_OK')
"""
    subprocess.run([sys.executable,'-I','-S','-c',code,str(bundle)],cwd=folder,check=True)
    mesh=trimesh.creation.icosphere(subdivisions=3); mesh.update_faces(np.arange(len(mesh.faces)-1))
    np.savez(folder/'input.npz',vertices=mesh.vertices,faces=mesh.faces)
    with (folder/'engine.log').open('wb') as log:
        process=subprocess.run([str(bundle/'repair_engine'/'MeshRepairEngine.exe'),
                                '--input',str(folder/'input.npz'),'--output',str(folder/'output.npz'),
                                '--passes','3','--target-faces','500'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,
                               timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode: raise RuntimeError((folder/'engine.log').read_text(errors='replace'))
    with np.load(folder/'output.npz',allow_pickle=False) as data:
        repaired=trimesh.Trimesh(data['vertices'],data['faces'],process=False)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert len(repaired.faces)<len(mesh.faces)
    print('BUNDLED_REPAIR_OK: standalone executable, optimization, hole closure')
