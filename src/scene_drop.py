"""Local file drops reuse the normal cancellable import pipeline."""
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QEvent, QTimer, Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QWidget


class SceneDropController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.pending = deque()
        self.processing = False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._advance)
        for center in (window.ui._slicer_center_container, window.ui._def_center_container):
            center.setAcceptDrops(True)
        QApplication.instance().installEventFilter(self)

    @staticmethod
    def paths(mime):
        if not mime.hasUrls():
            return []
        result = []
        seen = set()
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() not in ('.stl', '.step', '.stp') or not path.is_file():
                continue
            identity = str(path.resolve()).casefold()
            if identity not in seen:
                seen.add(identity)
                result.append(str(path))
        return result

    def _scope(self, target):
        if not isinstance(target, QWidget):
            return None
        ui = self.window.ui
        for page, center, scope in (
                (ui.page_slicer, ui._slicer_center_container, 'slicer'),
                (ui.page_predef, ui._def_center_container, 'predef')):
            if ui.stack.currentWidget() is page and (target is center or center.isAncestorOf(target)):
                return scope
        return None

    def eventFilter(self, target, event):
        if event.type() not in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            return False
        scope = self._scope(target)
        if scope is None:
            return False
        paths = self.paths(event.mimeData())
        if not paths or self.processing or self.pending or self.window._job is not None:
            event.ignore()
            return True
        if any(getattr(self.window, name, None) is not None for name in
               ('_transform_session', '_repair_session', '_placement_session')):
            event.ignore()
            return True
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if event.type() == QEvent.Drop:
            self.enqueue(paths, scope)
        return True

    def enqueue(self, paths, scope):
        """Capture destination before dialogs/workers allow navigation elsewhere."""
        if self.window._busy() or self.pending or self.processing:
            return
        platform = None
        if scope == 'slicer':
            index = self.window.ui.scene_tabs.currentIndex()
            platforms = [p for p in self.window.platforms if p['is_default']]
            platform = platforms[index - 1]['name'] if 0 < index <= len(platforms) else None
        self.pending.append(dict(paths=list(paths), scope=scope, platform=platform,
                                 generation=self.window._generation, kind=None))
        # Show dialogs after Qt has completed the operating system's drop event.
        self.timer.start()

    def clear(self):
        self.pending.clear()
        self.timer.stop()

    def _advance(self):
        if self.processing or self.window._job is not None or QApplication.activeModalWidget() is not None:
            return
        if not self.pending:
            self.timer.stop()
            return
        batch = self.pending[0]
        if batch['generation'] != self.window._generation or self.window._closing:
            self.clear()
            return
        self.processing = True
        try:
            if batch['scope'] == 'predef' and batch['kind'] is None:
                if any(Path(path).suffix.lower() == '.stl' for path in batch['paths']):
                    choice, ok = QInputDialog.getItem(
                        self.window, 'Загрузка в предеформацию',
                        'Загрузить перетаскиваемые STL-модели как:',
                        ['Номинальная модель (CAD)', 'Фактическая модель (скан)'], 0, False)
                    if not ok:
                        self.clear()
                        return
                    batch['kind'] = 'Scan' if choice == 'Фактическая модель (скан)' else 'CAD'
                else:
                    batch['kind'] = 'CAD'
            path = batch['paths'].pop(0)
            if batch['scope'] == 'slicer':
                self.window._import_slicer_path(path, platform=batch['platform'])
            else:
                kind = 'CAD' if Path(path).suffix.lower() in ('.step', '.stp') else batch['kind']
                self.window._import_model(kind, path=path, replace=False)
        except Exception as exc:
            self.window.log(f'[!] Не удалось загрузить перетаскиваемый файл: {exc}')
        finally:
            if not batch['paths'] and self.pending and self.pending[0] is batch:
                self.pending.popleft()
            self.processing = False
