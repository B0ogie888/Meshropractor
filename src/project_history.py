"""Bounded project snapshots with shared immutable geometry and a saved-state marker."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import numpy as np
import trimesh
from PySide6.QtCore import QTimer


def digest(value):
    hasher = hashlib.sha256()
    def visit(item):
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            hasher.update(str((array.dtype.str, array.shape)).encode())
            if array.size: hasher.update(memoryview(array).cast('B'))
        elif isinstance(item, trimesh.Trimesh):
            visit(item.vertices); visit(item.faces); visit(item.metadata)
        elif isinstance(item, dict):
            hasher.update(b'{')
            for key in sorted(item):
                visit(str(key)); visit(item[key])
            hasher.update(b'}')
        elif isinstance(item, (list, tuple)):
            hasher.update(b'[')
            for element in item: visit(element)
            hasher.update(b']')
        elif hasattr(item, '__dataclass_fields__'):
            visit({f.name: getattr(item, f.name) for f in fields(item) if f.name != 'page'})
        else:
            hasher.update(json.dumps(item.item() if isinstance(item, np.generic) else item, ensure_ascii=False).encode())
            hasher.update(b'\0')
    visit(value)
    return hasher.hexdigest()


class ProjectHistory:
    def __init__(self, max_steps=30, max_bytes=256 * 1024**2):
        self.entries = []
        self.index = -1
        self.saved = None
        self.max_steps, self.max_bytes = max_steps, max_bytes

    def reset(self, state, saved=True):
        self.entries = []
        self.index = -1
        self.push(state, "Начальное состояние")
        self.saved = self.key if saved else None

    @property
    def key(self):
        return self.entries[self.index][0] if self.index >= 0 else None

    @property
    def dirty(self):
        return self.key != self.saved

    def push(self, state, label="Изменение проекта"):
        key = digest(state)
        if key == self.key:
            return False
        del self.entries[self.index + 1:]
        # Reuse unchanged meshes instead of retaining a copy for every display edit.
        pool = {}
        seen = set()
        for _, previous, _ in self.entries:
            for record in previous.models + previous.parts:
                if id(record['mesh']) in seen: continue
                seen.add(id(record['mesh']))
                pool[digest(record['mesh'])] = record['mesh']
        for record in state.models + state.parts:
            mesh_key = digest(record['mesh'])
            record['mesh'] = pool.setdefault(mesh_key, record['mesh'])
        self.entries.append((key, state, label))
        self.index = len(self.entries) - 1
        while len(self.entries) > 2 and (len(self.entries) > self.max_steps + 1 or self.bytes_used() > self.max_bytes):
            del self.entries[0]
            self.index -= 1
        return True

    def bytes_used(self):
        seen = set()
        total = 0
        def count(value):
            nonlocal total
            if id(value) in seen: return
            seen.add(id(value))
            if isinstance(value, np.ndarray): total += value.nbytes
            elif isinstance(value, trimesh.Trimesh):
                count(value.vertices); count(value.faces); count(value.metadata)
            elif isinstance(value, dict):
                for child in value.values(): count(child)
            elif isinstance(value, (list, tuple)):
                for child in value: count(child)
            elif hasattr(value, '__dataclass_fields__'): count(vars(value))
        count(self.entries)
        return total


class HistoryMixin:
    def init_history(self):
        self.history = ProjectHistory()
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(300)
        self._history_timer.timeout.connect(self.flush_history)
        self.ui.action_undo.setShortcut("Ctrl+Z")
        self.ui.action_redo.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        self.reset_history()

    def reset_history(self):
        if not hasattr(self, 'history'): return
        self._history_timer.stop()
        self.history.reset(self.capture_project(), saved=not self.dirty)
        self.update_history_actions()

    def queue_history(self):
        if hasattr(self, 'history') and not self._restoring:
            self._history_timer.start()
            self.update_history_actions()

    def flush_history(self, label="Изменение проекта"):
        if not hasattr(self, 'history') or self._restoring: return
        self._history_timer.stop()
        self.history.push(self.capture_project(), label)
        self.dirty = self.history.dirty
        self.update_title()
        self.update_history_actions()

    def update_history_actions(self):
        if not hasattr(self, 'history'): return
        ready = self._job is None and getattr(self, "_transform_session", None) is None
        pending = self._history_timer.isActive()
        self.ui.action_undo.setEnabled(ready and (self.history.index > 0 or pending))
        self.ui.action_redo.setEnabled(ready and not pending and self.history.index < len(self.history.entries) - 1)
        undo_label = self.history.entries[self.history.index][2] if self.history.index > 0 else "Изменение проекта"
        redo_label = self.history.entries[self.history.index + 1][2] if self.history.index + 1 < len(self.history.entries) else ""
        self.ui.action_undo.setToolTip(f"Отменить: {undo_label} (Ctrl+Z)")
        self.ui.action_redo.setToolTip(f"Повторить: {redo_label} (Ctrl+Y)")

    def travel_history(self, direction):
        if self._busy(): return
        self.flush_history()
        target = self.history.index + direction
        if not 0 <= target < len(self.history.entries): return
        cameras = []
        for plotter in (self.ui.plotter, self.ui.slicer_plotter):
            cameras.append(deepcopy(plotter.camera_position) if plotter is not None and hasattr(plotter, 'camera_position') else None)
        page = self.ui.stack.currentWidget()
        previous = self.capture_project()
        try:
            self.restore_project(deepcopy(self.history.entries[target][1]))
        except Exception as exc:
            self.restore_project(previous)
            self.log(f"Не удалось восстановить историю: {exc}")
            return
        self.history.index = target
        self.ui.stack.setCurrentWidget(page)
        for plotter, camera in zip((self.ui.plotter, self.ui.slicer_plotter), cameras):
            if plotter is not None and camera is not None:
                plotter.camera_position = camera
                plotter.render()
        self.dirty = self.history.dirty
        self.update_title()
        self.update_history_actions()
