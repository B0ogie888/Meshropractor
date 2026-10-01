"""GitHub release discovery and verified, streaming installer downloads.

This module has no Qt dependency and never executes a downloaded file. Versions
come from the installer name: historical release tags in this repository contain
typos and must not accidentally turn an old installer into a newer version.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app_version import GITHUB_OWNER, GITHUB_REPO


REPOSITORY = f"{GITHUB_OWNER}/{GITHUB_REPO}"
RELEASES_URL = f"https://api.github.com/repos/{REPOSITORY}/releases"
NETWORK_TIMEOUT = 15
CHUNK_SIZE = 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_RELEASE_PAGES = 10
TRUSTED_HOSTS = frozenset({
    "api.github.com", "github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com", "github-releases.githubusercontent.com",
})
_VERSION = r"\d+(?:\.\d+){1,3}"
_INSTALLERS = (
    re.compile(rf"Meshropractor-Setup-({_VERSION})-x64\.exe", re.IGNORECASE),
    # Name used by the published 0.2.4 Inno Setup installer.
    re.compile(rf"Meshropractor_v({_VERSION})_Setup\.exe", re.IGNORECASE),
)


class UpdateError(RuntimeError):
    """An update cannot be checked or safely downloaded."""


@dataclass(frozen=True)
class ReleaseUpdate:
    version: str
    tag: str
    asset_name: str
    download_url: str
    size: int
    sha256: str | None
    page_url: str


def _version_key(value):
    if not isinstance(value, str) or not re.fullmatch(rf"[vV]?{_VERSION}", value):
        return None
    numbers = tuple(int(part) for part in value.lstrip("vV").split("."))
    return numbers + (0,) * (4 - len(numbers))


def _installer_version(name):
    if isinstance(name, str):
        for pattern in _INSTALLERS:
            match = pattern.fullmatch(name)
            if match:
                return match.group(1)
    return None


def _check_cancelled(cancelled):
    if cancelled():
        raise InterruptedError("Обновление отменено.")


def _validate_url(url):
    try:
        parsed = urlparse(url)
        valid = (parsed.scheme == "https" and parsed.hostname in TRUSTED_HOSTS
                 and parsed.port in (None, 443) and parsed.username is None
                 and parsed.password is None and not parsed.fragment
                 and not any(ord(char) < 32 for char in url))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise UpdateError("Ссылка обновления ведёт за пределы защищённых серверов GitHub.")
    return parsed


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        # Validate before urllib opens the destination, not after downloading.
        _validate_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _open_url(url, *, metadata=False):
    _validate_url(url)
    headers = {
        "User-Agent": "Meshropractor-Updater",
        "Accept": "application/vnd.github+json" if metadata else "application/octet-stream",
        "Accept-Encoding": "identity",
    }
    if metadata:
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    try:
        response = build_opener(_SafeRedirectHandler()).open(
            Request(url, headers=headers), timeout=NETWORK_TIMEOUT)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        if code in (403, 429):
            raise UpdateError("GitHub временно ограничил запросы. Повторите проверку позже.") from exc
        raise UpdateError(f"GitHub вернул ошибку HTTP {code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise UpdateError("Не удалось подключиться к GitHub. Проверьте интернет и повторите попытку.") from exc
    try:
        _validate_url(response.geturl())
        if response.getcode() != 200:
            raise UpdateError(f"Неожиданный ответ GitHub: HTTP {response.getcode()}.")
    except BaseException:
        response.close()
        raise
    return response


def _asset_update(release, asset):
    name = asset.get("name")
    version = _installer_version(name)
    size = asset.get("size")
    tag = release.get("tag_name")
    if (version is None or type(size) is not int or size < 2
            or not isinstance(tag, str) or not tag or len(tag) > 256
            or asset.get("state", "uploaded") != "uploaded"):
        return None
    url = asset.get("browser_download_url")
    try:
        parsed = _validate_url(url)
    except UpdateError:
        return None
    expected = f"/{REPOSITORY}/releases/download/{tag}/{name}"
    if parsed.hostname != "github.com" or unquote(parsed.path) != expected or parsed.query:
        return None
    digest = asset.get("digest")
    if digest is not None:
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
            return None
        digest = digest.split(":", 1)[1].lower()
    return ReleaseUpdate(version, tag, name, url, size, digest,
                         f"https://github.com/{REPOSITORY}/releases/tag/{quote(tag, safe='')}")


def _release_update(release):
    if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
        return None
    tag = release.get("tag_name", "")
    if not isinstance(tag, str) or re.search(
            r"(?:^|[-_.\d])(alpha|beta|rc|dev|preview|pre|snapshot|nightly)(?:\d|$|[-_.])",
            tag, re.IGNORECASE):
        return None
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        return None
    candidates = [candidate for asset in assets if isinstance(asset, dict)
                  if (candidate := _asset_update(release, asset)) is not None]
    if not candidates or len({_version_key(item.version) for item in candidates}) != 1:
        # Multiple installer versions in one release make the intended update ambiguous.
        return None
    return min(candidates, key=lambda item: ("-x64.exe" not in item.asset_name.lower(), item.asset_name))


def check_for_update(current_version: str, cancelled=lambda: False) -> ReleaseUpdate | None:
    """Return the newest stable installer newer than ``current_version``.

    Cancellation raises InterruptedError; offline/API failures raise UpdateError.
    No installer is downloaded here, and an unchanged version is never offered.
    """
    current = _version_key(current_version)
    if current is None:
        raise UpdateError(f"Некорректная текущая версия приложения: {current_version!r}.")
    newest = None
    for page in range(1, MAX_RELEASE_PAGES + 1):
        _check_cancelled(cancelled)
        with _open_url(f"{RELEASES_URL}?per_page=100&page={page}", metadata=True) as response:
            payload = bytearray()
            while True:
                _check_cancelled(cancelled)
                chunk = response.read(min(64 * 1024, MAX_METADATA_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > MAX_METADATA_BYTES:
                    raise UpdateError("Ответ GitHub слишком большой.")
        _check_cancelled(cancelled)
        try:
            releases = json.loads(payload)
        except (ValueError, UnicodeError) as exc:
            raise UpdateError("GitHub вернул некорректный список релизов.") from exc
        if not isinstance(releases, list):
            raise UpdateError("GitHub вернул некорректный список релизов.")
        for release in releases:
            candidate = _release_update(release)
            if candidate is not None and _version_key(candidate.version) > current:
                if newest is None or _version_key(candidate.version) > _version_key(newest.version):
                    newest = candidate
        if len(releases) < 100:
            return newest
    raise UpdateError("Слишком много страниц релизов GitHub. Откройте страницу релизов вручную.")


def _validate_update(update):
    # Revalidate even a caller-created ReleaseUpdate before touching the filesystem.
    candidate = _asset_update({"tag_name": update.tag}, {
        "name": update.asset_name, "size": update.size,
        "browser_download_url": update.download_url,
        "digest": f"sha256:{update.sha256}" if update.sha256 is not None else None,
    })
    if candidate is None or _version_key(candidate.version) != _version_key(update.version):
        raise UpdateError("Некорректные сведения об установщике обновления.")


def verify_installer(path: Path, update: ReleaseUpdate, cancelled=lambda: False) -> None:
    """Recheck an installer immediately before offering/executing installation."""
    _check_cancelled(cancelled)
    _validate_update(update)
    checksum = hashlib.sha256()
    received = 0
    signature = b""
    try:
        with Path(path).open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != update.size:
                raise UpdateError("Размер установщика изменился. Скачайте обновление повторно.")
            while True:
                _check_cancelled(cancelled)
                chunk = stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                received += len(chunk)
                signature = (signature + chunk)[:2]
                if received > update.size or (len(signature) == 2 and signature != b"MZ"):
                    raise UpdateError("Установщик повреждён. Скачайте обновление повторно.")
                checksum.update(chunk)
    except InterruptedError:
        raise
    except OSError as exc:
        raise UpdateError("Не удалось прочитать установщик. Скачайте обновление повторно.") from exc
    _check_cancelled(cancelled)
    if received != update.size or signature != b"MZ":
        raise UpdateError("Установщик повреждён. Скачайте обновление повторно.")
    if update.sha256 is not None and checksum.hexdigest() != update.sha256.lower():
        raise UpdateError("Контрольная сумма установщика не совпадает. Скачайте обновление повторно.")


def download_release(update: ReleaseUpdate, directory: Path,
                     progress=lambda received, total: None,
                     cancelled=lambda: False) -> Path:
    """Stream an installer into a private .part, verify it, and atomically publish.

    A failed/cancelled download only removes its own temporary file. Existing
    installers and other downloads in the directory are left intact on failure.
    A verified previously downloaded installer is reused without network access.
    The result is a path, not permission to execute it.
    """
    _check_cancelled(cancelled)
    _validate_update(update)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / update.asset_name
    if destination.is_file():
        try:
            verify_installer(destination, update, cancelled)
        except UpdateError:
            pass
        else:
            progress(update.size, update.size)
            return destination
    temporary = None
    received = 0
    signature = b""
    checksum = hashlib.sha256()
    try:
        with _open_url(update.download_url) as response:
            advertised_size = response.headers.get("Content-Length")
            if advertised_size is not None:
                try:
                    if int(advertised_size) != update.size:
                        raise ValueError()
                except (TypeError, ValueError) as exc:
                    raise UpdateError("Размер загрузки не совпадает с данными релиза GitHub.") from exc
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{update.asset_name}.",
                                             suffix=".part", dir=directory, delete=False) as output:
                temporary = Path(output.name)
                progress(0, update.size)
                while True:
                    _check_cancelled(cancelled)
                    chunk = response.read(CHUNK_SIZE)
                    _check_cancelled(cancelled)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > update.size:
                        raise UpdateError("Загруженный файл превышает размер установщика из релиза.")
                    signature = (signature + chunk)[:2]
                    if len(signature) == 2 and signature != b"MZ":
                        raise UpdateError("Загруженный файл не является установщиком Windows.")
                    checksum.update(chunk)
                    output.write(chunk)
                    progress(received, update.size)
                if received != update.size:
                    raise UpdateError("Загрузка оборвалась: установщик получен не полностью.")
                if signature != b"MZ":
                    raise UpdateError("Загруженный файл не является установщиком Windows.")
                if update.sha256 is not None and checksum.hexdigest() != update.sha256.lower():
                    raise UpdateError("Контрольная сумма установщика не совпадает. Повторите загрузку.")
                output.flush()
                os.fsync(output.fileno())
            _check_cancelled(cancelled)
            os.replace(temporary, destination)
            temporary = None
        return destination
    except (URLError, TimeoutError) as exc:
        raise UpdateError("Соединение прервано во время загрузки. Повторите попытку.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
