"""Offline tests: python3 -m pytest -q tests/test_pi_q6.py"""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import register_pi_model as register
import src.serve as serve
from test_deployment_config import CONFIGS, EXAMPLE

CONFIG = CONFIGS["Qwen3-27B-Uncensored-Q6"]


class PiQ6Tests(unittest.TestCase):
    def setUp(self):
        patcher = patch("deployment.CONFIG_PATH", EXAMPLE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def document(self):
        return {"untouched": [1, 2], "providers": {
            "other": {"apiKey": "unchanged"},
            "llamacpp": {"models": [{"thinkingLevelMap": {"high": None}, "compat": {
                "thinkingFormat": "chat-template", "chatTemplateKwargs": {"preserve_thinking": True}}}]}}}

    def test_upsert_backup_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            original = self.document()
            path.write_text(json.dumps(original))
            generator = Mock(return_value="new-token")
            report = register.register(CONFIG, path=path, generator=generator)
            updated = json.loads(path.read_text())
            self.assertEqual(report["actual_ttl_seconds"], 3600)
            self.assertEqual((datetime.fromisoformat(report["next_refresh_at"]) -
                              datetime.fromisoformat(report["sign_time"])).total_seconds(), 3480)
            self.assertEqual(json.loads(Path(str(path) + ".bak").read_text()), original)
            provider = updated["providers"].pop("sagemaker-qwen3-q6")
            self.assertEqual(updated, original)
            self.assertEqual(provider["models"][0]["id"], "qwen38-27b-uncensored-q6-sm")
            self.assertEqual(provider["models"][0]["contextWindow"],
                             int(CONFIG["environment"]["SM_LLAMACPP_CTX_SIZE"]))
            self.assertNotIn("maxTokens", provider["models"][0])
            self.assertEqual(provider["models"][0]["compat"], original["providers"]["llamacpp"]["models"][0]["compat"])
            generator.return_value = "next-token"
            register.register(CONFIG, path=path, generator=generator)
            second = json.loads(path.read_text())
            self.assertEqual(second["providers"]["sagemaker-qwen3-q6"], provider)
            self.assertEqual(provider["apiKey"], register.token_command(register.token_path(path, CONFIG)))
            self.assertEqual(register.token_path(path, CONFIG).read_text(), "next-token\n")
            self.assertEqual(register.token_path(path, CONFIG).stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(Path(str(path) + ".bak").read_text()), original)
            self.assertNotIn("next-token", json.dumps(report))
            del second["providers"]["sagemaker-qwen3-q6"]
            self.assertEqual(second, original)

    def test_migrate_existing_provider_without_llamacpp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            provider = {"apiKey": "old-token", "models": [
                {"id": "custom", "contextWindow": 4096},
                {"id": "qwen38-27b-uncensored-q6-sm", "contextWindow": 65536,
                 "maxTokens": 1234, "compat": {"custom": True}}], "baseUrl": "unchanged"}
            original = {"providers": {"sagemaker-qwen3-q6": provider}}
            path.write_text(json.dumps(original))
            register.register(CONFIG, path=path, generator=Mock(return_value="fresh-token"))
            migrated = json.loads(path.read_text())
            expected = copy.deepcopy(provider)
            expected["apiKey"] = register.token_command(register.token_path(path, CONFIG))
            expected["models"][1]["contextWindow"] = int(
                CONFIG["environment"]["SM_LLAMACPP_CTX_SIZE"])
            self.assertEqual(migrated["providers"]["sagemaker-qwen3-q6"], expected)
            before = path.stat()
            register.register(CONFIG, path=path, generator=Mock(return_value="newer-token"))
            self.assertEqual(path.stat().st_ino, before.st_ino)
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)

    def test_context_sync_preserves_migration_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps(self.document()))
            register.register(CONFIG, path=path, generator=Mock(return_value="old-token"))
            backup = Path(str(path) + ".bak").read_bytes()
            before = json.loads(path.read_text())
            env = CONFIG["environment"]
            with patch.dict(env, {"SM_LLAMACPP_CTX_SIZE": "65536"}):
                with self.assertRaises(RuntimeError):
                    register.register(CONFIG, path=path, generator=Mock(side_effect=RuntimeError("offline")))
                self.assertEqual(json.loads(path.read_text()), before)
                self.assertEqual(register.token_path(path, CONFIG).read_text(), "old-token\n")
                register.register(CONFIG, path=path, generator=Mock(return_value="new-token"))
            before["providers"]["sagemaker-qwen3-q6"]["models"][0]["contextWindow"] = 65536
            self.assertEqual(json.loads(path.read_text()), before)
            self.assertEqual(Path(str(path) + ".bak").read_bytes(), backup)

    def test_invalid_context_leaves_files_intact(self):
        env = CONFIG["environment"]
        for value in (None, "", "invalid", "1.5", "0", "-1", 1.5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "models.json"
                path.write_text(json.dumps(self.document()))
                original = path.read_bytes()
                generator = Mock(return_value="token")
                replacement = {} if value is None else {"SM_LLAMACPP_CTX_SIZE": value}
                with patch.dict(env, replacement, clear=True):
                    with self.assertRaisesRegex(register.ContextConfigError, "SM_LLAMACPP_CTX_SIZE"):
                        register.register(CONFIG, path=path, generator=generator)
                generator.assert_not_called()
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(register.token_path(path, CONFIG).exists())
                self.assertFalse(Path(str(path) + ".bak").exists())

    def test_command_quoting_and_failed_refresh_retains_token(self):
        with tempfile.TemporaryDirectory(prefix="pi ' $token ") as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps(self.document()))
            register.register(CONFIG, path=path, generator=Mock(return_value="valid-token"))
            before = path.read_bytes()
            command = json.loads(before)["providers"]["sagemaker-qwen3-q6"]["apiKey"]
            self.assertEqual(subprocess.check_output(command[1:], shell=True, text=True), "valid-token\n")
            with self.assertRaises(RuntimeError):
                register.register(CONFIG, path=path, generator=Mock(side_effect=RuntimeError("offline")))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(register.token_path(path, CONFIG).read_text(), "valid-token\n")

    def test_failed_atomic_token_write_retains_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret"
            register.write_token(path, "old")
            with patch.object(register.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    register.write_token(path, "new")
            self.assertEqual(path.read_text(), "old\n")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_running_pi_runtime_refresh_and_sse(self):
        pi_dir = Path(os.environ.get("PI_TEST_INSTALL", "/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent"))
        if not shutil.which("node") or not (pi_dir / "dist/core/model-runtime.js").exists():
            self.skipTest("Requires Node and installed Pi (or PI_TEST_INSTALL)")
        with tempfile.TemporaryDirectory(prefix="pi ' $token ") as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps(self.document()))
            register.register(CONFIG, path=path, generator=Mock(return_value="first-token"))
            result = subprocess.run(
                ["node", str(Path(__file__).with_name("test_pi_dynamic_key.mjs")),
                 str(pi_dir), str(path), sys.executable],
                capture_output=True, text=True, timeout=45,
                env={**os.environ, "PI_OFFLINE": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("dynamic auth + SSE passed", result.stdout)

    def test_ttl_fallback_to_sdk_maximum(self):
        generator = Mock(side_effect=[ValueError("unsupported"), "token"])
        self.assertEqual(register.mint_token(3600, generator, region=CONFIG["region"]), ("token", 43200))
        self.assertEqual(generator.call_args.kwargs["expiry"].total_seconds(), 43200)

    def test_custom_ttl(self):
        self.assertEqual(register.mint_token(900, Mock(return_value="token"), region=CONFIG["region"]), ("token", 900))

    def test_generation_failure_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            original = json.dumps(self.document())
            path.write_text(original)
            with self.assertRaises(RuntimeError):
                register.register(CONFIG, path=path, generator=Mock(side_effect=RuntimeError("offline")))
            self.assertEqual(path.read_text(), original)
            self.assertFalse(Path(str(path) + ".bak").exists())

    def test_strict_arguments(self):
        for args in (["wrong"], ["Qwen3-27B-Uncensored-Q6", "--ttl-seconds", "0"]):
            with self.assertRaises(SystemExit) as error:
                register.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_wait_creating_then_ready(self):
        client = Mock()
        client.describe_endpoint.side_effect = [{"EndpointStatus": "Creating"}, {"EndpointStatus": "InService"}]
        sleep = Mock()
        serve.wait_for_endpoint(client, "Qwen3-27B-Uncensored-Q6", clock=Mock(return_value=0), sleep=sleep)
        sleep.assert_called_once_with(15)

    def test_wait_already_ready(self):
        client = Mock()
        client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        sleep = Mock()
        serve.wait_for_endpoint(client, "Qwen3-27B-Uncensored-Q6", sleep=sleep)
        sleep.assert_not_called()

    def test_failed_reason_and_timeout(self):
        client = Mock()
        client.describe_endpoint.return_value = {"EndpointStatus": "Failed", "FailureReason": "capacity"}
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            serve.wait_for_endpoint(client, "Qwen3-27B-Uncensored-Q6")
        client.describe_endpoint.return_value = {"EndpointStatus": "Creating"}
        with self.assertRaises(TimeoutError):
            serve.wait_for_endpoint(client, "Qwen3-27B-Uncensored-Q6", clock=Mock(side_effect=[0, 3600]))

    @patch.object(serve.subprocess, "run")
    def test_refresh_uses_reported_deadline(self, run):
        run.return_value = Mock(stdout='{"next_refresh_at": "2026-01-01T00:13:00+00:00"}\n', stderr="")
        deadline = datetime(2026, 1, 1, 0, 13, tzinfo=timezone.utc)
        self.assertEqual(serve.refresh(3600, "Qwen3-27B-Uncensored-Q6"), deadline)

    def test_credential_expiry_limits_schedule(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for remaining in (900, 420, 60):
            expiry = now + timedelta(seconds=remaining)
            report = register.refresh_schedule(now, 3600, expiry)
            deadline = datetime.fromisoformat(report["next_refresh_at"])
            self.assertEqual(deadline, expiry - timedelta(seconds=120))
            self.assertEqual(serve.refresh_delay(deadline, now), max(0, remaining - 120))
        report = register.refresh_schedule(now, 300, now + timedelta(seconds=900))
        self.assertEqual(serve.refresh_delay(datetime.fromisoformat(report["next_refresh_at"]), now), 180)

    def test_forced_login_refresh_and_failure_fallback(self):
        from botocore.credentials import LoginCredentialFetcher, RefreshableCredentials
        expiry = datetime.now(timezone.utc) + timedelta(seconds=600)
        metadata = {"access_key": "fresh1234", "secret_key": "secret", "token": "token",
                    "expiry_time": (expiry + timedelta(seconds=300)).isoformat()}
        fetcher = LoginCredentialFetcher("session", Mock(), Mock())
        credentials = RefreshableCredentials("old5678", "secret", "token", expiry,
                                            fetcher.refresh_credentials, "login")
        thresholds = []

        def refreshed(instance):
            thresholds.append(instance._REFRESH_THRESHOLD)
            return metadata

        with patch("botocore.session.Session") as session:
            session.return_value.get_credentials.return_value = credentials
            with patch.object(LoginCredentialFetcher, "refresh_credentials", refreshed):
                pinned, actual_expiry = register.signing_credentials()
            self.assertEqual(pinned.access_key, "fresh1234")
            self.assertEqual(actual_expiry, expiry + timedelta(seconds=300))
            self.assertEqual(thresholds, [float("inf")])
            self.assertEqual(fetcher._REFRESH_THRESHOLD, 300)
            with patch.object(LoginCredentialFetcher, "refresh_credentials", side_effect=OSError("secret")):
                pinned, actual_expiry = register.signing_credentials()
            self.assertEqual(pinned.access_key, "old5678")
            self.assertEqual(actual_expiry, expiry)


if __name__ == "__main__":
    unittest.main()
