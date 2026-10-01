"""Asynchronous, opt-in GitHub download and Windows installer handoff."""
import ctypes
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QThread, QTimer, Signal, QStandardPaths, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from app_version import APP_VERSION
from update_backend import check_for_update, download_release, verify_installer


def launch_installer(path, parent_handle=0):
    """Let the installer request elevation normally; never suppress UAC/errors."""
    path = Path(path).resolve()
    if sys.platform != 'win32':
        raise OSError('Автоматическая установка доступна только в Windows.')
    if not path.is_file() or path.suffix.lower() != '.exe':
        raise OSError('Скачанный установщик не найден. Загрузите обновление повторно.')
    shell = ctypes.windll.shell32.ShellExecuteW
    shell.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                      ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_int]
    shell.restype = ctypes.c_void_p
    result = shell(parent_handle, 'open', str(path),
                   '/SP- /SILENT /NORESTART /UPDATE=1', str(path.parent), 1)
    if not result or result <= 32:
        raise OSError(f'Windows не запустила установщик (код {result or 0}). '
                      'Возможно, разрешение на установку было отклонено. Приложение остаётся открытым.')


class UpdateWorker(QThread):
    result = Signal(object)
    error = Signal(str)
    # Python integers avoid Qt's 32-bit int overflow for multi-gigabyte installers.
    progress = Signal(object, object)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            result = self.function(self)
            if not self.isInterruptionRequested(): self.result.emit(result)
        except InterruptedError:
            pass
        except Exception as exc:
            if not self.isInterruptionRequested(): self.error.emit(str(exc))


def size_text(size):
    return f'{size / 1024**3:.2f} ГБ' if size >= 1024**3 else f'{size / 1024**2:.1f} МБ'


class UpdateController(QObject):
    def __init__(self, window, directory=None):
        super().__init__(window)
        self.window = window
        self.directory = Path(directory) if directory else Path(
            QStandardPaths.writableLocation(QStandardPaths.GenericCacheLocation)) / 'Meshropractor' / 'updates'
        self.worker = None
        self.available = self.ready = None
        self.task = None
        self.outcome = self.failure = None
        self.cancelled = self.closing = self.closed = False
        self.manual = False
        self.progress_dialog = None
        self.pending_offer = None
        self.start_timer = QTimer(self)
        self.start_timer.setSingleShot(True)
        self.start_timer.timeout.connect(self.check)
        self.offer_timer = QTimer(self)
        self.offer_timer.setInterval(500)
        self.offer_timer.timeout.connect(self._deliver_offer)
        window.ui.btn_check_updates.clicked.connect(self.manual_check)

    def start(self):
        """Called once by the application entry point, after closing the splash."""
        if not self.closed: self.start_timer.start(1800)

    def manual_check(self):
        if self.ready and self.available:
            self._defer(self.offer_install)
        else:
            self.check(manual=True)

    def check(self, manual=False):
        if self.closed: return
        self.start_timer.stop()
        if self.worker:
            self.manual = self.manual or manual
            return
        self.manual = manual
        self.pending_offer = None
        self.offer_timer.stop()
        self._begin('check', lambda worker: check_for_update(APP_VERSION, cancelled=worker.isInterruptionRequested))

    def _begin(self, task, function):
        if self.worker or self.closed: return
        self.task = task
        self.outcome = self.failure = None
        self.cancelled = False
        worker = self.worker = UpdateWorker(function, self)
        worker.result.connect(self._result)
        worker.error.connect(self._error)
        worker.progress.connect(self._progress)
        worker.finished.connect(self._finished)
        self.window.ui.btn_check_updates.setEnabled(False)
        self.window.ui.btn_check_updates.setText('Проверка обновлений…' if task == 'check' else 'Обновление загружается…' if task == 'download' else 'Проверка установщика…')
        worker.start()

    def _result(self, value):
        self.outcome = value

    def _error(self, message):
        self.failure = message

    def _finished(self):
        task, worker = self.task, self.worker
        interrupted = self.cancelled or worker.isInterruptionRequested()
        self.worker = None
        worker.deleteLater()
        if self.progress_dialog:
            self.progress_dialog.canceled.disconnect(self.cancel)
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
            self.progress_dialog = None
        self._reset_button()
        if self.closing:
            self.closing = False
            QTimer.singleShot(0, self.window.close)
            return
        if self.closed or interrupted:
            return
        if self.failure:
            message = self.failure
            if task == 'verify':
                self.ready = None
                self._reset_button()
            self.window.log('[i] Обновление: ' + message)
            if task != 'check' or self.manual:
                self._defer(lambda: QMessageBox.warning(self.window, 'Обновление не выполнено', message))
            return
        if task == 'check':
            self.available = self.outcome
            if self.available: self._defer(self.offer_download)
            elif self.manual:
                self._defer(lambda: QMessageBox.information(self.window, 'Обновления',
                            f'Новых стабильных релизов с установщиком не найдено.\nТекущая версия: {APP_VERSION}.'))
        elif task == 'download':
            self.ready = Path(self.outcome)
            self._reset_button()
            self._defer(self.offer_install)
        elif task == 'verify':
            # Verification was window-modal: the previously approved project
            # save/discard decision cannot be invalidated by editing mid-check.
            self._perform_install()

    def _reset_button(self):
        self.window.ui.btn_check_updates.setEnabled(True)
        self.window.ui.btn_check_updates.setText('Установить скачанное обновление…' if self.ready else f'Проверить обновления · {APP_VERSION}')

    def _defer(self, callback):
        if self.closed: return
        self.pending_offer = callback
        self.offer_timer.start()

    def _deliver_offer(self):
        if self.closed or not self.pending_offer:
            self.offer_timer.stop()
            return
        if (not self.window.isVisible() or self.window.isMinimized() or self.worker
                or QApplication.activeModalWidget() is not None
                or getattr(self.window, '_job', None) is not None
                or getattr(self.window, '_transform_session', None) is not None):
            return
        callback, self.pending_offer = self.pending_offer, None
        self.offer_timer.stop()
        callback()

    def _ask(self, title, message, accept_text):
        dialog = QMessageBox(self.window)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle(title)
        dialog.setTextFormat(Qt.PlainText)
        dialog.setText(message)
        yes = dialog.addButton(accept_text, QMessageBox.AcceptRole)
        later = dialog.addButton('Позже', QMessageBox.RejectRole)
        dialog.setDefaultButton(later)
        dialog.exec()
        return dialog.clickedButton() is yes

    def offer_download(self):
        update = self.available
        if not update or self.closed: return
        if self._ask('Доступно обновление Meshropractor',
                     f'Доступна версия {update.version}.\nСейчас установлена {APP_VERSION}.\n'
                     f'Размер загрузки: {size_text(update.size)}.\n\nСкачать обновление из GitHub Releases?', 'Скачать'):
            self.start_download()

    def _show_progress(self, text, modal=False):
        dialog = self.progress_dialog = QProgressDialog(text, 'Отмена', 0, 1000, self.window)
        dialog.setWindowTitle('Обновление Meshropractor')
        dialog.setWindowModality(Qt.WindowModal if modal else Qt.NonModal)
        dialog.setAutoClose(False); dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setMinimumWidth(440)
        dialog.canceled.connect(self.cancel)
        dialog.show()

    def start_download(self):
        if self.closed or self.worker or not self.available: return
        self.ready = None
        self._show_progress('Подключение к GitHub…')
        self._begin('download', lambda worker: download_release(self.available, self.directory,
                    progress=worker.progress.emit, cancelled=worker.isInterruptionRequested))

    def _progress(self, received, total):
        if self.progress_dialog is None or self.cancelled: return
        if total > 0:
            self.progress_dialog.setRange(0, 1000)
            self.progress_dialog.setValue(min(1000, received * 1000 // total))
            text = f'{size_text(received)} из {size_text(total)} · {min(100, received * 100 // total)}%'
        else:
            self.progress_dialog.setRange(0, 0)
            text = size_text(received)
        self.progress_dialog.setLabelText('Загрузка обновления…\n' + text)

    def cancel(self):
        self.cancelled = True
        if self.worker: self.worker.requestInterruption()

    def offer_install(self):
        if self.closed or self.ready is None or not self.available or self.worker: return
        if self._ask('Обновление скачано',
                     f'Версия {self.available.version} готова к установке.\n\n'
                     'Приложение предложит сохранить проект и закроется. Установщик покажет ход обновления '
                     'и снова запустит Meshropractor. Windows может запросить разрешение на установку.\n\n'
                     'Установить обновление сейчас?', 'Установить'):
            self.install_now()

    def install_now(self):
        if self.closed or self.worker or self.ready is None: return
        if getattr(self.window, '_job', None) or getattr(self.window, '_transform_session', None):
            self._defer(self.offer_install)
            return
        # Reuse normal asynchronous saving. Cancelling Save/Discard/Save As or
        # a failed save never reaches verification or the installer launch.
        if not self.window._confirm_discard(self.install_now): return
        self._show_progress('Проверка целостности установщика…', modal=True)
        self.progress_dialog.setRange(0, 0)
        self._begin('verify', lambda worker: verify_installer(self.ready, self.available,
                                                             cancelled=worker.isInterruptionRequested))

    def _perform_install(self):
        try:
            launch_installer(self.ready, int(self.window.winId()))
        except Exception as exc:
            QMessageBox.warning(self.window, 'Не удалось начать установку', str(exc))
            return
        # Close only after Windows accepted the launch. The save/discard decision
        # was already made immediately before the window-modal verification.
        self.window._update_exit = True
        self.window.close()

    def allow_close(self):
        self.start_timer.stop(); self.offer_timer.stop()
        self.pending_offer = None
        if self.worker:
            self.closing = True
            self.cancel()
            self.window.ui.status_label.setText('Остановка загрузки обновления перед закрытием…')
            return False
        return True

    def on_closed(self):
        self.closed = True
        self.start_timer.stop(); self.offer_timer.stop()
