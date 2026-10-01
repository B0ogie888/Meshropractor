from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import update_backend as backend


PAYLOAD = b"MZ" + bytes(range(256)) * 12


def release(version="0.2.6", *, tag=None, legacy=False, **kwargs):
    tag = tag if tag is not None else "v" + version
    name = (f"Meshropractor_v{version}_Setup.exe" if legacy
            else f"Meshropractor-Setup-{version}-x64.exe")
    return dict(tag_name=tag, draft=False, prerelease=False, assets=[dict(
        name=name, state="uploaded", size=len(PAYLOAD),
        digest="sha256:" + hashlib.sha256(PAYLOAD).hexdigest(),
        browser_download_url=f"https://github.com/{backend.REPOSITORY}/releases/download/{tag}/{name}",
    )], **kwargs)


class Response(io.BytesIO):
    def __init__(self, payload=PAYLOAD, *, headers=None, url="https://release-assets.githubusercontent.com/test", status=200):
        super().__init__(payload)
        self.headers = headers or {}
        self.url = url
        self.status = status
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status


class ReleaseDiscoveryTests(unittest.TestCase):
    def check(self, releases, current="0.2.5"):
        response = Response(json.dumps(releases).encode())
        with patch.object(backend, "_open_url", return_value=response):
            result = backend.check_for_update(current)
        self.assertTrue(response.closed)
        self.assertTrue(all(0 < size <= 65536 for size in response.read_sizes))
        return result

    def test_numeric_order_installer_version_and_immutable_result(self):
        result = self.check([release("0.2.9"), release("0.2.10"), release("0.2.6")])
        self.assertEqual(result.version, "0.2.10")
        self.assertEqual(result.tag, "v0.2.10")
        self.assertEqual(result.sha256, hashlib.sha256(PAYLOAD).hexdigest())
        with self.assertRaises(AttributeError):
            result.version = "9.9.9"

    def test_no_update_for_equal_normalized_or_older_versions(self):
        self.assertIsNone(self.check([release("0.2.5.0"), release("0.2.4")], current="v0.2.5"))
        self.assertIsNone(self.check([]))

    def test_published_legacy_mistyped_tag_does_not_offer_downgrade(self):
        self.assertIsNone(self.check([release("0.2.4", tag="v.2.4.0", legacy=True)]))
        result = self.check([release("0.2.6", tag="v.2.6.0", legacy=True)])
        self.assertEqual(result.version, "0.2.6")

    def test_skip_draft_prerelease_and_bare_or_wrong_arch_executables(self):
        draft, beta, rc, bare, arm = [release("2.0.0") for _ in range(5)]
        draft["draft"] = True
        beta["prerelease"] = True
        rc["tag_name"] = "v2.0.0-rc1"
        bare["assets"][0]["name"] = "Meshropractor.exe"
        arm["assets"][0]["name"] = "Meshropractor-Setup-2.0.0-arm64.exe"
        self.assertIsNone(self.check([draft, beta, rc, bare, arm]))

    def test_asset_version_is_authoritative_but_ambiguous_release_is_ignored(self):
        mismatched = release("0.2.6", tag="v2.6.0")
        self.assertEqual(self.check([mismatched]).version, "0.2.6")
        mismatched["assets"].extend(release("0.2.7", tag="v2.6.0")["assets"])
        self.assertIsNone(self.check([mismatched]))

    def test_only_repo_scoped_trusted_assets_with_safe_names_and_digest(self):
        for field, value in [
            ("browser_download_url", "https://evil.example/install.exe"),
            ("browser_download_url", "https://github.com/attacker/other/releases/download/v0.2.6/Meshropractor-Setup-0.2.6-x64.exe"),
            ("name", "../Meshropractor-Setup-0.2.6-x64.exe"),
            ("digest", "sha256:bad"), ("size", -1), ("size", True),
            ("state", "new"),
        ]:
            with self.subTest(field=field, value=value):
                item = release()
                item["assets"][0][field] = value
                self.assertIsNone(self.check([item]))

    def test_pagination_inspects_older_publish_dates_too(self):
        responses = [Response(json.dumps([release("0.2.4")] * 100).encode()),
                     Response(json.dumps([release("0.2.7")]).encode())]
        with patch.object(backend, "_open_url", side_effect=responses) as opened:
            self.assertEqual(backend.check_for_update("0.2.5").version, "0.2.7")
        self.assertEqual(opened.call_count, 2)
        self.assertIn("page=2", opened.call_args.args[0])

    def test_cancel_bad_json_and_metadata_limit(self):
        with patch.object(backend, "_open_url") as opened:
            with self.assertRaises(InterruptedError):
                backend.check_for_update("0.2.5", cancelled=lambda: True)
            opened.assert_not_called()
        for payload in (b"<html>oops</html>", b'{"message":"invalid"}'):
            with patch.object(backend, "_open_url", return_value=Response(payload)):
                with self.assertRaises(backend.UpdateError):
                    backend.check_for_update("0.2.5")
        with patch.object(backend, "MAX_METADATA_BYTES", 16), patch.object(
                backend, "_open_url", return_value=Response(b" " * 100)):
            with self.assertRaises(backend.UpdateError):
                backend.check_for_update("0.2.5")


class InstallerDownloadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.directory = Path(self.folder.name)
        self.update = backend._release_update(release())
        self.path = self.directory / self.update.asset_name
        self.unrelated = self.directory / "another-download.part"
        self.unrelated.write_bytes(b"keep")

    def assert_no_partial(self):
        self.assertEqual(list(self.directory.glob("*.part")), [self.unrelated])

    def test_streaming_progress_verified_atomic_result(self):
        progress = []
        response = Response(headers={"Content-Length": str(len(PAYLOAD))})
        with patch.object(backend, "CHUNK_SIZE", 512), patch.object(
                backend, "_open_url", return_value=response):
            result = backend.download_release(self.update, self.directory,
                                               lambda got, total: progress.append((got, total)))
        self.assertEqual(result, self.path)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertEqual(progress[0], (0, len(PAYLOAD)))
        self.assertEqual(progress[-1], (len(PAYLOAD), len(PAYLOAD)))
        self.assertGreater(len(progress), 3)
        self.assertTrue(all(size == 512 for size in response.read_sizes))
        self.assertTrue(response.closed)
        self.assert_no_partial()
        backend.verify_installer(result, self.update)

    def test_truncated_corrupted_html_and_oversize_keep_existing_file(self):
        self.path.write_bytes(b"original older or damaged download")
        original = self.path.read_bytes()
        for payload in (PAYLOAD[:-1], PAYLOAD[:-1] + b"?", b"<html>" + PAYLOAD[6:], PAYLOAD + b"!"):
            with self.subTest(length=len(payload), prefix=payload[:2]):
                with patch.object(backend, "_open_url", return_value=Response(payload)):
                    with self.assertRaises(backend.UpdateError):
                        backend.download_release(self.update, self.directory)
                self.assertEqual(self.path.read_bytes(), original)
                self.assert_no_partial()

    def test_wrong_content_length_never_publishes(self):
        with patch.object(backend, "_open_url", return_value=Response(headers={"Content-Length": "123"})):
            with self.assertRaises(backend.UpdateError):
                backend.download_release(self.update, self.directory)
        self.assertFalse(self.path.exists())
        self.assert_no_partial()

    def test_cancel_mid_download_cleans_only_own_part(self):
        cancel = False

        def progress(received, total):
            nonlocal cancel
            cancel = received > 0

        with patch.object(backend, "CHUNK_SIZE", 512), patch.object(
                backend, "_open_url", return_value=Response()):
            with self.assertRaises(InterruptedError):
                backend.download_release(self.update, self.directory, progress, lambda: cancel)
        self.assertFalse(self.path.exists())
        self.assert_no_partial()

    def test_verified_cached_download_needs_no_network(self):
        self.path.write_bytes(PAYLOAD)
        with patch.object(backend, "_open_url") as opened:
            self.assertEqual(backend.download_release(self.update, self.directory), self.path)
            opened.assert_not_called()

    def test_invalid_cached_download_is_replaced_only_after_valid_download(self):
        self.path.write_bytes(b"MZbad")
        with patch.object(backend, "_open_url", return_value=Response()):
            backend.download_release(self.update, self.directory)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_no_digest_still_checks_size_and_executable_signature(self):
        update = replace(self.update, sha256=None)
        with patch.object(backend, "_open_url", return_value=Response()):
            backend.download_release(update, self.directory)
        backend.verify_installer(self.path, update)
        self.path.write_bytes(b"no" + PAYLOAD[2:])
        with self.assertRaises(backend.UpdateError):
            backend.verify_installer(self.path, update)

    def test_verify_after_download_detects_same_size_file_tampering(self):
        self.path.write_bytes(PAYLOAD[:-1] + b"?")
        with self.assertRaises(backend.UpdateError):
            backend.verify_installer(self.path, self.update)

    def test_verify_can_cancel_and_never_deletes_file(self):
        self.path.write_bytes(PAYLOAD)
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 3

        with patch.object(backend, "CHUNK_SIZE", 512):
            with self.assertRaises(InterruptedError):
                backend.verify_installer(self.path, self.update, cancelled)
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_caller_supplied_unsafe_name_is_rejected_before_network_or_write(self):
        with patch.object(backend, "_open_url") as opened:
            with self.assertRaises(backend.UpdateError):
                backend.download_release(replace(self.update, asset_name="../evil.exe"), self.directory)
            opened.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [self.unrelated])


class GitHubTransportTests(unittest.TestCase):
    def test_redirects_reject_http_credentials_other_hosts_and_non_https_ports(self):
        handler = backend._SafeRedirectHandler()
        source = Request("https://github.com/B0ogie888/Meshropractor/releases")
        for target in ("http://github.com/file", "https://evil.example/file",
                       "https://github.com.evil.example/file", "https://user:pw@github.com/file",
                       "https://github.com:444/file", "file:///C:/test.exe"):
            with self.subTest(target=target), self.assertRaises(backend.UpdateError):
                handler.redirect_request(source, None, 302, "Found", {}, target)
        destination = "https://release-assets.githubusercontent.com/test?token=signed"
        result = handler.redirect_request(source, None, 302, "Found", {}, destination)
        self.assertEqual(result.full_url, destination)

    def test_network_errors_use_user_readable_error_and_timeout(self):
        for error in (URLError("offline"), TimeoutError("timeout"),
                      HTTPError(backend.RELEASES_URL, 429, "rate limited", {}, None)):
            opener = Mock()
            opener.open.side_effect = error
            with patch.object(backend, "build_opener", return_value=opener):
                with self.assertRaises(backend.UpdateError):
                    backend._open_url(backend.RELEASES_URL, metadata=True)
            self.assertEqual(opener.open.call_args.kwargs["timeout"], backend.NETWORK_TIMEOUT)

    def test_unexpected_status_and_final_url_close_response(self):
        for response in (Response(status=206), Response(url="https://evil.example/installer.exe")):
            opener = Mock()
            opener.open.return_value = response
            with patch.object(backend, "build_opener", return_value=opener):
                with self.assertRaises(backend.UpdateError):
                    backend._open_url(backend.RELEASES_URL)
            self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
