# Файл: debug_run.py
# ВРЕМЕННЫЙ диагностический запускатор. Запускайте ИМЕННО ЕГО вместо Meshropractor.py,
# пока не найдем причину краха (потом можно удалить).
#
# Что он делает:
#  1. Ставит переменную окружения KMP_DUPLICATE_LIB_OK=TRUE.
#     Это официальный обходной путь для одной из самых частых причин
#     "Process finished with exit code -1073741819 (0xC0000005)" в связках
#     numpy + scipy + Open3D + VTK на Windows: несколько разных версий
#     OpenMP-рантайма (libiomp5md.dll от Intel MKL/Open3D и libomp от других
#     библиотек) загружаются в один процесс и конфликтуют. Если после этого
#     краш ИСЧЕЗНЕТ - значит, дело было именно в этом (см. вывод ниже).
#  2. Включает faulthandler и пишет ВСЕ, что происходит, в файл crash_log.txt
#     рядом со скриптом. faulthandler умеет печатать питоновский traceback
#     даже при настоящем access violation (не всегда, но часто) - это даст
#     точное место краха, если шаг 1 не помог.
#
# ВАЖНО: запускайте из обычной консоли (cmd/PowerShell), КОМАНДОЙ:
#     python debug_run.py
# а не двойным кликом и не кнопкой Run в IDE, которая закрывает окно консоли
# сразу после краха - иначе вы не увидите вывод.

import os
import sys
import faulthandler

# --- Шаг 1: обход конфликта OpenMP-рантаймов ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Отключаем буферизацию вывода, чтобы все [DEBUG]-строки печатались СРАЗУ,
# а не терялись при аварийном завершении процесса.
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# --- Шаг 2: faulthandler в файл ---
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_log.txt")
log_file = open(log_path, "w", encoding="utf-8", buffering=1)
faulthandler.enable(file=log_file, all_threads=True)

print(f"[DEBUG] KMP_DUPLICATE_LIB_OK=TRUE, лог краша: {log_path}", flush=True)
print("[DEBUG] Запускаю Meshropractor.py ...", flush=True)

# Дублируем весь stdout/stderr также и в лог-файл, чтобы все [DEBUG]-строки
# из Meshropractor.py / UI_Meshropractor.py / Workers_Meshropractor.py
# гарантированно попали в crash_log.txt, даже если консоль исчезнет.
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = _Tee(sys.stdout, log_file)
sys.stderr = _Tee(sys.stderr, log_file)

# --- Шаг 3: запускаем реальное приложение ---
if __name__ == "__main__":
    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meshropractor.py"),
        run_name="__main__",
    )
