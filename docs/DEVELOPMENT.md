# Структура проекта и подготовка выпуска

## Один запуск, два способа открыть приложение

`Meshropractor.pyw` в корне — ярлык в виде Python-скрипта для запуска двойным
щелчком в Windows без консоли. Он выбирает проектную `.venv`, если она есть,
и передаёт управление общему launcher. Бизнес-логики в этом файле нет.

`src/Meshropractor.py` — короткая точка входа для терминала, IDE и PyInstaller.
Она используется командой `python src/Meshropractor.py` и также вызывает
`desktop_launcher.main`. Это не вторая версия программы. Прежний импорт
`from Meshropractor import MainWindow` оставлен совместимым, но новый код
импортирует окно из `main_window`.

Общий путь запуска:

```text
Meshropractor.pyw / src/Meshropractor.py / собранный EXE
  → desktop_launcher.py  (журнал, настройки, заставка)
    → main_window.py    (единственное главное окно)
      → ui_shell.py     (рабочая компоновка и темы)
        → ui_base.py    (общие контролы и диалоги)
```

Заставка импортирует только Qt и лёгкие вспомогательные модули.
Не переносите импорты Torch, VTK или OCP в точку входа: они задержат её показ.

## Каталоги

| Путь | Назначение |
| --- | --- |
| `src/` | Код приложения. Модули сгруппированы по префиксам `cad_`, `repair_`, `texture_`, `placement_` и т. д. |
| `src/main_window.py` | Главное окно, подключения контроллеров и обработчики действий. |
| `src/ui_base.py`, `src/ui_shell.py`, `src/ui_theme.py` | Базовые виджеты, компоновка, оформление. |
| `src/part_controls.py`, `src/parts_view.py`, `src/part_inspector.py`, `src/predef_panel.py` | Панели единственного интерфейса; прежний префикс `new_` удалён. |
| `src/workers.py`, `src/background_tasks.py` | Фоновые задачи и их сигналы. |
| `src/cls_slicer.py` | Экспериментальный экспорт CLS; ограничения описаны в руководстве. |
| `src/app_branding.py` | Общие логотип, многоразмерный значок окна и Windows AppUserModelID. |
| `assets/` | Ресурсы, которые входят в приложение: PNG/ICO, SVG, QR. |
| `assets/ribbon/` | Редактируемые иконки команд по вкладкам. |
| `scripts/validation/` | Нативные проверки интерфейса, моделей и готовых сборок. |
| `scripts/benchmarks/` | Замеры вращения сцены и диагностика глубины сетки. |
| `scripts/assets/` | Экспорт SVG и упаковка рабочего логотипа в PNG/ICO. |
| `tests/` | Автоматические регрессионные тесты unittest. |
| `packaging/linux/` | Docker, сборка и проверка пакета Debian/Ubuntu. |
| `docs/` | Руководства и отчёты о проверках. |
| `docs/design/` | Исходники утверждённого дизайна и история вариантов. В сборку не включаются. |
| `licenses/repair-engine/` | Лицензии, исходный архив зависимости и материалы отдельного модуля лечения. |

Файлы `Meshropractor.spec`, `RepairEngine.spec` и `Meshropractor.iss` остаются
в корне: это конфигурация сборки, команды и относительные пути рассчитаны
на запуск из корня репозитория. Параметры выпуска берутся из `VERSION`.

`build/`, `dist/`, `output/`, `.venv/`, `.idea/` и кеши Python исключены из Git.
Они не являются исходниками для коммита. Готовые установщики и пользовательские
настройки не удаляются при уборке дерева исходников.

## Проверки перед коммитом

```powershell
.venv\Scripts\python.exe scripts/validation/check_repository.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/validation/smoke_desktop_entry.py
.venv\Scripts\python.exe scripts/validation/smoke_ui.py
.venv\Scripts\python.exe scripts/validation/smoke_theme_ribbons.py
.venv\Scripts\python.exe scripts/validation/smoke_new_project.py
.venv\Scripts\python.exe scripts/validation/smoke_new_project.py --theme dark
.venv\Scripts\python.exe scripts/validation/smoke_startup.py
```

Первая команда не запускает Qt: проверяет синтаксис Python, локальные ссылки
документации, структуру SVG и наличие ресурсов. Она также включена в CI.
Нативные сценарии открывают собственные окна с временными настройками и
сохраняют результаты в `output/`. Проверки `smoke_frozen.py` и `smoke_bundled_*.py`
предназначены для уже собранного дистрибутива; сами EXE не создают.

`tests/qt_test_cleanup.py` направляет настройки тестовых главных окон во временный
INI-профиль. После `deleteLater()` явно обрабатываются отложенные удаления:
без основного цикла Qt одного `processEvents()` для этого недостаточно.
Это предотвращает накопление скрытых окон между тестами.
См. [документацию Qt](https://doc.qt.io/qt-6/qcoreapplication.html#processEvents).

## Изменение логотипа

Рабочий мастер: [meshropractor-mark-contoured.png](design/startup/meshropractor-mark-contoured.png).
После его редактирования выполните `python scripts/assets/export_branding.py`:
обновятся `assets/logo.png` и `assets/logo.ico`. Значок применяется к шапке,
заставке, окнам, Alt+Tab и панели задач Windows. PyInstaller и Inno Setup
используют тот же ICO, Linux-пакет — тот же PNG.

Обновление ресурсов не меняет уже собранные EXE и установщики. Их повторная
сборка выполняется отдельным шагом по [инструкции Windows](BUILD_WINDOWS.md)
или [инструкции Linux](BUILD_LINUX.md). После сборки обязательна проверка
готового приложения, а затем установка в чистой системе.

## Проверка иконок

`python scripts/validation/preview_ribbon_icons.py` создаёт в `output/icon-audit/`
обзор всех значков на светлом и тёмном фоне в размерах 28 и 48 px. Используется
тот же механизм перекрашивания, что в приложении: внутренние линии должны
отделяться прозрачными промежутками, иначе цветные грани сольются в силуэт.
См. [правила и исходники иконок](../assets/ribbon/README.md).
