# Сборка Windows

Из корня проекта, в подготовленном Python 3.12 окружении:

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm Meshropractor.spec
```

Результат: `dist\Meshropractor\Meshropractor.exe`. Для переноса нужна **вся папка**
`dist\Meshropractor`, включая `_internal`. Файл `.spec` включает ресурсы, Torch,
Open3D, PyVista, OpenCascade (`OCP`) и его DLL из `cadquery_ocp_novtk.libs` для STEP.
Дополнительные параметры `--onedir`, `--collect-all` и `--add-data` не нужны:
они уже заданы в `.spec`. Консоль оставлена включённой для диагностики.
UPX отключён для нативных библиотек геометрии и CUDA.
Скрипт также ограничивает PATH на время сборки: посторонние DLL, например ICU из
Poppler, не должны подменять системные библиотеки Qt. После изменения окружения
добавьте `--clean` к команде для полного пересоздания анализа зависимостей.

Установщик Inno Setup 7:

```powershell
& 'C:\Program Files\Inno Setup 7\ISCC.exe' Meshropractor.iss
```

Результат: `dist\installer\Meshropractor-Setup-0.2.5-x64.exe`.
Версия задаётся в `.iss` или параметром компилятора `/DMyAppVersion=0.2.5`.
Скрипт использует AppId существующей установки Meshropractor и копирует весь onedir,
сохраняя структуру `_internal`. Папки проектов и пользовательские настройки не удаляются.
Установщик не запускается автоматически после сборки.

Проверки готовой папки:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_frozen.py
.\.venv\Scripts\python.exe -I -S scripts/smoke_bundled_step.py
```

Первая проверка открывает и штатно закрывает EXE из посторонней рабочей папки.
Вторая проверяет обмен STEP упакованной библиотекой OCP без site-packages окружения.
Журнал запуска приложения: `output\frozen-startup.log`.
