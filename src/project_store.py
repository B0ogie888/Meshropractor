"""Validated, atomic project archives. No Qt/VTK objects or pickle on disk."""
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import trimesh
from section_geometry import default_sections, validate_sections


VERSION = "2.1"
MAX_ARCHIVE_BYTES = 8 * 1024**3


@dataclass
class ProjectState:
    models: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    platforms: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    sections: list = field(default_factory=default_sections)
    cad_pts: list = field(default_factory=list)
    scan_pts: list = field(default_factory=list)
    callouts: list = field(default_factory=list)
    active_result: str | None = None
    active_heatmap: str | None = None
    page: str = "predef"


def validate_mesh(mesh):
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Модель должна содержать вершины и треугольники.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Модель содержит некорректные координаты (NaN/Inf).")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Требуется треугольная сетка.")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError("Некорректные индексы треугольников.")
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Площадь модели должна быть положительной.")
    return mesh


def load_mesh(path, step_deflection=0.05, step_angle=0.25):
    if Path(path).suffix.lower() in (".step", ".stp"):
        from cad_import import load_step
        return validate_mesh(load_step(path, linear_deflection=step_deflection, angular_deflection=step_angle))
    return validate_mesh(trimesh.load(path, force="mesh"))


def validate_platforms(platforms):
    if not isinstance(platforms, list):
        raise ValueError("Некорректный список платформ.")
    names = set()
    for platform in platforms:
        name = platform.get("name", "").strip()
        dims = np.asarray(platform.get("dim", []), dtype=float)
        if not name or name in names or dims.shape != (3,) or not np.isfinite(dims).all() or (dims <= 0).any():
            raise ValueError("Платформам нужны уникальные имена и положительные габариты.")
        names.add(name)
        for zone in platform.get("zones", []):
            vals = np.asarray([zone.get(k, 0) for k in ("x", "y", "r", "zmin", "zmax")], dtype=float)
            if not np.isfinite(vals).all() or vals[2] <= 0 or vals[4] < vals[3]:
                raise ValueError("Некорректные размеры запретной зоны.")


def _json_value(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("Некорректный массив проекта.")
        key = f"array_{len(arrays)}"
        arrays[key] = value
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, arrays) for v in value]
    return value


def _restore_value(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            array = arrays[value["__array__"]].copy()
            if not np.isfinite(array).all():
                raise ValueError("Некорректный массив проекта.")
            return array
        return {k: _restore_value(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_value(v, arrays) for v in value]
    return value


def save_project(path, state, preview=b""):
    """Replace the destination only after a complete archive has been flushed."""
    validate_sections(state.sections)
    validate_platforms(state.platforms)
    destination = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                manifest = {k: v for k, v in vars(state).items() if k not in ("models", "parts")}
                manifest["version"] = VERSION
                for group in ("models", "parts"):
                    manifest[group] = []
                    for index, record in enumerate(getattr(state, group)):
                        mesh = validate_mesh(record["mesh"])
                        if group == 'parts':
                            from part_supports import validate_groups
                            validate_groups(record.get('supports', []), len(mesh.faces))
                        arrays = {"vertices": mesh.vertices, "faces": mesh.faces}
                        info = {k: v for k, v in record.items() if k != "mesh"}
                        info["metadata"] = mesh.metadata
                        info = _json_value(info, arrays)
                        name = f"meshes/{group}_{index}.npz"
                        payload = io.BytesIO()
                        np.savez(payload, **arrays)
                        archive.writestr(name, payload.getvalue())
                        manifest[group].append({"file": name, "info": info})
                archive.writestr("project.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False))
                if preview:
                    archive.writestr("preview.png", preview)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_legacy(archive, meta):
    state = ProjectState(platforms=meta.get("platforms", []), cad_pts=meta.get("cad_pts", []), scan_pts=meta.get("scan_pts", []))
    names = archive.namelist()
    for kind, name in (("CAD", "cad"), ("Scan", "scan"), ("Result", "result")):
        path = f"meshes/{name}.stl"
        if path in names:
            mesh = validate_mesh(trimesh.load(io.BytesIO(archive.read(path)), file_type="stl"))
            key = f"{kind}_0"
            state.models.append(dict(key=key, kind=kind, name=f"{kind}.stl", mesh=mesh, style={}))
            if kind == "Result":
                state.active_result = key
    files = sorted((p for p in names if p.startswith("meshes/slicer_part_") and p.endswith(".stl")), key=lambda p: int(Path(p).stem.rsplit("_", 1)[1]))
    for path in files:
        index = int(Path(path).stem.rsplit("_", 1)[1])
        infos = meta.get("slicer_parts_info", [])
        info = infos[index] if index < len(infos) else {"filename": meta.get(f"slicer_name_{index}", f"Part {index + 1}.stl")}
        mesh = validate_mesh(trimesh.load(io.BytesIO(archive.read(path)), file_type="stl"))
        state.parts.append(dict(info, mesh=mesh))
    state.page = "slicer" if state.parts and not state.models else "predef"
    return state


def load_project(path):
    """Read and validate everything before the caller changes its live project."""
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_ARCHIVE_BYTES:
            raise ValueError("Архив превышает допустимый размер распакованных данных (8 ГБ).")
        meta = json.loads(archive.read("project.json"))
        version = str(meta.get("version", "1.0"))
        if version.startswith("1."):
            state = _read_legacy(archive, meta)
        elif version in ('2.0', VERSION):
            state = ProjectState(**{k: meta[k] for k in vars(ProjectState()) if k in meta and k not in ("models", "parts")})
            for group in ("models", "parts"):
                for entry in meta.get(group, []):
                    with np.load(io.BytesIO(archive.read(entry["file"])), allow_pickle=False) as arrays:
                        info = _restore_value(entry["info"], arrays)
                        vertices, faces = arrays["vertices"], arrays["faces"]
                        if faces.dtype.kind not in "iu":
                            raise ValueError("Индексы треугольников должны быть целыми.")
                        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
                            raise ValueError("Некорректные индексы треугольников.")
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                        mesh.metadata = info.pop("metadata", {})
                        getattr(state, group).append(dict(info, mesh=validate_mesh(mesh)))
        else:
            raise ValueError(f"Версия проекта {version} не поддерживается.")
    validate_sections(state.sections)
    validate_platforms(state.platforms)
    from part_supports import validate_groups
    for part in state.parts: validate_groups(part.get('supports', []), len(part['mesh'].faces))
    support_ids = [group['id'] for part in state.parts for group in part.get('supports', [])]
    if len(support_ids) != len(set(support_ids)): raise ValueError('ID поддержек должны быть уникальны в проекте.')
    keys = set()
    for record in state.models:
        key = record["key"]
        if key in keys or record["kind"] not in ("CAD", "Scan", "Def", "Result", "Heatmap") or not key.startswith(record["kind"] + "_"):
            raise ValueError("Некорректный список моделей проекта.")
        keys.add(key)
        for field_name in ("vectors", "vector_origins"):
            values = record["mesh"].metadata.get(field_name)
            if values is not None and np.asarray(values).shape != (len(record["mesh"].vertices), 3):
                raise ValueError("Векторные данные не соответствуют вершинам сетки.")
        if record["kind"] == "Heatmap":
            values = np.asarray(record.get("deviations"))
            if values.shape != (len(record["mesh"].vertices),) or not np.isfinite(values).all():
                raise ValueError("Некорректная карта отклонений.")
    for key in (state.active_result, state.active_heatmap):
        if key is not None and key not in keys:
            raise ValueError("Активная модель отсутствует в проекте.")
    for points in (state.cad_pts, state.scan_pts):
        values = np.asarray(points, dtype=float)
        if values.size and (values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all()):
            raise ValueError("Некорректные маркеры проекта.")
    return state
