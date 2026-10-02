"""Isolated, cancellable repair and independent quality checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import trimesh


def run_engine(source, passes, progress, cancelled, target_faces=0):
    if cancelled(): raise InterruptedError('Лечение отменено')
    with tempfile.TemporaryDirectory(prefix='meshropractor-repair-') as folder:
        root = Path(folder)
        np.savez(root/'input.npz', vertices=source.vertices, faces=source.faces)
        if getattr(sys, 'frozen', False):
            helper = 'MeshRepairEngine.exe' if sys.platform == 'win32' else 'MeshRepairEngine'
            command = [str(Path(sys._MEIPASS) / 'repair_engine' / helper)]
        else:
            command = [sys.executable, '-u', str(Path(__file__).with_name('repair_engine_cli.py'))]
        command += ['--input', str(root/'input.npz'), '--output', str(root/'output.npz'), '--passes', str(passes), '--target-faces', str(target_faces)]
        # A separate process also isolates native-library crashes and allows
        # immediate cancellation while MeshFix is inside a long C++ operation.
        with (root/'engine.log').open('w+b') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                with (root/'engine.log').open('rb') as reader:
                    last_heartbeat = time.monotonic()
                    message = 'Подготовка полного ремонта…'
                    pending = b''
                    while process.poll() is None:
                        if cancelled():
                            raise InterruptedError('Лечение отменено; исходная модель сохранена')
                        pending += reader.read()
                        lines = pending.split(b'\n')
                        pending = lines.pop()
                        for line in lines:
                            try:
                                marker = line.find(b'{"message":')
                                value = json.loads(line[marker:] if marker >= 0 else line)
                                message = value['message']
                                progress(message)
                            except (ValueError, KeyError):
                                pass
                        if time.monotonic()-last_heartbeat >= 15:
                            progress(message + ' Расчёт продолжается…')
                            last_heartbeat = time.monotonic()
                        time.sleep(.1)
                if process.returncode:
                    raise RuntimeError('Модуль ремонта завершился с ошибкой. Исходная модель сохранена.\n' +
                                       (root/'engine.log').read_text(errors='replace')[-1500:])
                with np.load(root/'output.npz', allow_pickle=False) as data:
                    return trimesh.Trimesh(data['vertices'], data['faces'], process=False), json.loads(str(data['passes']))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()


def compare_shape(source, repaired, progress=lambda message: None, cancelled=lambda: False):
    """Bidirectional vertex + area-sampled distance, not a certified Hausdorff bound."""
    import open3d as o3d
    origin = source.bounds.mean(axis=0)
    result = {}
    for name, query, target in (('source_to_result', source, repaired), ('result_to_source', repaired, source)):
        progress('Контроль сохранения формы: ' + ('исходная → исправленная' if name == 'source_to_result' else 'исправленная → исходная'))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(target.vertices-origin, dtype=np.float32)),
                            o3d.core.Tensor(np.asarray(target.faces, dtype=np.uint32)))
        samples, _ = trimesh.sample.sample_surface(query, min(100000, max(1000, len(query.faces))), seed=42)
        # Check every vertex, plus surface samples to detect bridges across holes.
        distances = []
        for points in (query.vertices, samples):
            for start in range(0, len(points), 100000):
                if cancelled(): raise InterruptedError('Контроль формы отменён')
                batch = np.asarray(points[start:start+100000]-origin, dtype=np.float32)
                distances.append(scene.compute_distance(o3d.core.Tensor(batch)).numpy())
        values = np.concatenate(distances)
        result[name] = dict(max_mm=float(values.max()), p95_mm=float(np.percentile(values, 95)),
                            rms_mm=float(np.sqrt(np.mean(values.astype(float)**2))), points=len(values))
    result['max_mm'] = max(result[name]['max_mm'] for name in ('source_to_result', 'result_to_source'))
    result['bounds_delta_mm'] = float(np.max(np.abs(source.bounds-repaired.bounds)))
    return result


def prepare_full_repair(source, kind='CAD', passes=3, tolerance_mm=.05,
                        progress=lambda message: None, cancelled=lambda: False, before=None,
                        target_faces=400000):
    from mesh_diagnostics import diagnose_mesh, serializable_report
    from project_store import validate_mesh
    from repair_operations import _orient_shells
    if not 1 <= passes <= 10 or not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('Некорректные параметры полного ремонта')
    validate_mesh(source)
    before = before or diagnose_mesh(source, progress=progress, cancelled=cancelled)
    if before['clean']:
        return source.copy(), dict(before=before, after=before, changed=False, defects=False,
                                   kind=kind, full=True, passes=[], holes=0, acceptable=True, warnings=[])
    progress('Полное исправление сетки…')
    repaired, records = run_engine(source, passes, progress, cancelled, target_faces)
    validate_mesh(repaired)
    # Keep inner closed shells inward: orienting every component outward would
    # turn a valid cavity into material without changing surface distances.
    orientation = _orient_shells(repaired, progress, cancelled)
    after = diagnose_mesh(repaired, progress=progress, cancelled=cancelled)
    shape = compare_shape(source, repaired, progress, cancelled)
    acceptable = shape['max_mm'] <= tolerance_mm
    repaired.metadata = deepcopy(source.metadata)
    repaired.metadata['repair'] = dict(before=serializable_report(before), after=serializable_report(after),
                                       shape=shape, tolerance_mm=tolerance_mm, passes=records,
                                       orientation=deepcopy(orientation))
    return repaired, dict(before=before, after=after, changed=True, defects=not after['clean'],
                          kind=kind, full=True, passes=records, holes=max(0,before['contours']-after['contours']),
                          shape=shape, tolerance_mm=tolerance_mm, acceptable=acceptable,
                          **orientation)
