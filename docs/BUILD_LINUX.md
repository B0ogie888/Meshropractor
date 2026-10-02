# Сборка и установка Linux 0.2.6

Linux-пакет — `meshropractor_0.2.6_amd64.deb`. Он содержит Python, Qt, VTK,
OpenCascade, CPU-версию PyTorch и отдельный модуль ремонта. Устанавливать Python
и зависимости через pip для готового пакета не нужно.

Размер установщика — около **735 МиБ**; установленная сборка занимает примерно
**4.4 ГиБ**, дополнительно нужны место для проектов и системных библиотек.

Целевая архитектура — **x86-64 (amd64)**. Сборка использует Debian 12 и glibc 2.36;
целевые системы — **Debian 12 и Ubuntu 24.04**. Для графики нужны OpenGL и X11.
В сеансе Wayland используется XWayland. Ubuntu 22.04, ARM и другие форматы
пакетов этой сборкой не покрываются.

Основное приложение использует системную `libstdc++6`, чтобы драйверы Mesa
новой системы не загружали старую C++-библиотеку из сборки. Этот случай описан
в [рекомендациях PyInstaller по совместимости Linux](https://pyinstaller.org/en/stable/usage.html#making-gnu-linux-apps-forward-compatible).

## Установка

Скопируйте `.deb` на Linux-компьютер и выполните в его каталоге:

```bash
sudo apt install ./meshropractor_0.2.6_amd64.deb
```

APT установит системные библиотеки и OpenGL-драйверы Mesa, перечисленные в пакете. Приложение появится
в меню программ как **Meshropractor**. Из терминала его можно открыть командой:

```bash
meshropractor
```

Сохраните `.sha256` рядом с установщиком для проверки файла:

```bash
sha256sum -c meshropractor_0.2.6_amd64.deb.sha256
```

Для обновления закройте приложение и установите новый `.deb` той же командой.
В Linux кнопка проверки обновлений открывает страницу GitHub Releases;
автоматическая загрузка и запуск Windows EXE используются только в Windows.

Удаление пакета:

```bash
sudo apt remove meshropractor
```

Приложение устанавливается в `/opt/meshropractor`; команда запуска —
`/usr/bin/meshropractor`. Проекты сохраняются в выбранных пользователем каталогах.
Настройки Qt — в `~/.config/MeshropractorTeam/`, журнал запуска —
`${XDG_STATE_HOME:-~/.local/state}/Meshropractor/logs/Meshropractor.log`.
Рабочий стол запускает приложение без окна терминала.

## Повторная сборка через Docker

PyInstaller собирает приложение для той ОС, на которой он запущен:
[официальная документация](https://pyinstaller.org/en/stable/operating-mode.html).
Поэтому из Windows используется **Linux-контейнер Docker Desktop**. Исходники
подключаются только для чтения, Windows-сборка в `dist/Meshropractor` сохраняется.

Из корня проекта в PowerShell:

```powershell
docker build --platform linux/amd64 -t meshropractor-linux-build:0.2.6 -f packaging/linux/Dockerfile .
New-Item -ItemType Directory -Force dist/linux | Out-Null
$projectPath = (Get-Location).Path
$linuxOutput = Join-Path $projectPath 'dist/linux'
docker run --rm --platform linux/amd64 --shm-size=1g `
  --mount "type=bind,source=$projectPath,target=/workspace,readonly" `
  --mount "type=bind,source=$linuxOutput,target=/out" `
  meshropractor-linux-build:0.2.6
```

На Linux те же исходники собираются так:

```bash
docker build --platform linux/amd64 -t meshropractor-linux-build:0.2.6 -f packaging/linux/Dockerfile .
mkdir -p dist/linux
docker run --rm --platform linux/amd64 --shm-size=1g \
  --mount "type=bind,source=$PWD,target=/workspace,readonly" \
  --mount "type=bind,source=$PWD/dist/linux,target=/out" \
  meshropractor-linux-build:0.2.6
```

Сначала собирается `RepairEngine.spec`, затем `Meshropractor.spec`. До упаковки
выполняется проверка готового приложения через Xvfb и Mesa. При ошибке пакет не
создаётся. Результаты — в `dist/linux/`: установщик, SHA-256, отчёт и снимки
`build-validation/`. GPL-исходники и уведомления отдельного модуля ремонта
включены в пакет, в том числе в `/usr/share/doc/meshropractor/repair-engine`.

## Проверка установленного пакета

Проверка использует одноразовый контейнер без Python, устанавливает `.deb` через
APT и запускает приложение от обычного пользователя. Выполняются импорт STEP/BREP,
поддержки на CAD-гранях, сохранение проекта, обрезка с закрытием сечения, ремонт,
отображение слайсера и предеформации, карта отклонений для открытого скана.

Пример для Debian из Linux-терминала:

```bash
docker run --rm --shm-size=1g \
  --mount "type=bind,source=$PWD,target=/workspace,readonly" \
  --mount "type=bind,source=$PWD/dist/linux,target=/out" \
  debian:12-slim bash /workspace/packaging/linux/validate.sh \
  /out/meshropractor_0.2.6_amd64.deb /out/debian-validation
```

Для Ubuntu замените образ на `ubuntu:24.04` и каталог отчёта на
`/out/ubuntu-validation`. Проверка выполняется с программным OpenGL; поведение
конкретных видеодрайверов и сеансов Wayland проверяется отдельно на рабочем ПК.

Встроенную проверку можно запустить и на установленной системе:

```bash
meshropractor --self-test /tmp/meshropractor-validation
```

Она использует временные настройки, не проверяет обновления, создаёт тестовую
геометрию и закрывает окно по завершении. Это проверка дистрибутива, не рабочего
пользовательского проекта.
