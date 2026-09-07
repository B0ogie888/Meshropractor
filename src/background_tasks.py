"""One result channel and Qt's native finished signal for every background task."""
import logging
from PySide6.QtCore import QThread, Signal


class FunctionWorker(QThread):
    result = Signal(object)
    error = Signal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function, self.args, self.kwargs = function, args, kwargs

    def run(self):
        try:
            value = self.function(*self.args, **self.kwargs)
            if not self.isInterruptionRequested():
                self.result.emit(value)
        except Exception as exc:
            logging.exception("Background operation failed")
            self.error.emit(str(exc))
