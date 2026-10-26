"""Offline regression tests for configuration-selected serving."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_deployment_config import CONFIGS, EXAMPLE
import register_pi_model as register
import src.serve as serve


class ServeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("deployment.CONFIG_PATH", EXAMPLE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_custom_alias_and_region(self):
        config = {**CONFIGS["Qwen3-27B-Uncensored-FP8"],
                  "name": "custom-endpoint", "region": "eu-west-1",
                  "pi_model_alias": "custom-pi/custom/model"}
        with tempfile.TemporaryDirectory(prefix="config space ") as directory:
            path = Path(directory) / "unrelated-name.json"
            path.write_text(json.dumps({"custom": config}))
            with patch("deployment.CONFIG_PATH", path), \
                    patch.object(serve.subprocess, "run") as run, patch("boto3.client") as client, \
                    patch.object(serve, "wait_for_endpoint") as wait, \
                    patch.object(serve, "refresh", side_effect=KeyboardInterrupt) as refresh:
                self.assertEqual(serve.main(["custom"]), 0)
                self.assertEqual(run.call_args.args[0][-1], "custom")
                self.assertEqual(client.call_args.kwargs["region_name"], "eu-west-1")
                wait.assert_called_once_with(client.return_value, "custom-endpoint")
                refresh.assert_called_once_with(3600, "custom")
            models = Path(directory) / "models.json"
            models.write_text('{"providers": {}}')
            generator = Mock(return_value="token")
            register.register(config, path=models, generator=generator)
            self.assertEqual(generator.call_args.kwargs["region"], "eu-west-1")
            provider = json.loads(models.read_text())["providers"]["custom-pi"]
            self.assertIn("sagemaker.eu-west-1.amazonaws.com/endpoints/custom-endpoint/", provider["baseUrl"])
            self.assertEqual(provider["models"][0]["id"], "custom/model")

    def test_selection_through_startup_and_repeated_refresh(self):
        for alias in CONFIGS:
            with self.subTest(alias=alias), patch.object(serve.subprocess, "run") as run, \
                    patch("boto3.client") as client, \
                    patch.object(serve, "wait_for_endpoint") as wait, \
                    patch.object(serve, "refresh", side_effect=[Mock(), KeyboardInterrupt]) as refresh, \
                    patch.object(serve, "refresh_delay", return_value=10), \
                    patch.object(serve.time, "sleep"):
                self.assertEqual(serve.main([alias, "--ttl-seconds", "900"]), 0)
                self.assertEqual(run.call_args.args[0][-1:], [alias])
                self.assertEqual(Path(run.call_args.args[0][1]), serve.HERE / "create_endpoint.py")
                self.assertTrue(Path(run.call_args.args[0][1]).is_file())
                self.assertEqual(client.call_args.kwargs["region_name"], CONFIGS[alias]["region"])
                wait.assert_called_once_with(client.return_value, alias)
                self.assertEqual(refresh.call_count, 2)
                for call in refresh.call_args_list:
                    self.assertEqual(call.args, (900, alias))

    def test_wait_and_refresh_use_selected_alias(self):
        for alias in CONFIGS:
            client = Mock()
            client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
            serve.wait_for_endpoint(client, alias)
            client.describe_endpoint.assert_called_once_with(EndpointName=alias)
            with patch.object(serve.subprocess, "run") as run:
                run.return_value.stdout = '{"next_refresh_at":"2026-01-01T00:00:00+00:00"}\n'
                run.return_value.stderr = ""
                serve.refresh(900, alias)
                self.assertEqual(run.call_args.args[0][-3:], [alias, "--ttl-seconds", "900"])
                self.assertEqual(Path(run.call_args.args[0][1]), serve.HERE / "register_pi_model.py")
                self.assertTrue(Path(run.call_args.args[0][1]).is_file())
                run.return_value.check_returncode.assert_called_once()

    def test_invalid_arguments_have_no_side_effects(self):
        for args in ([], ["unknown"], ["Qwen3-27B-Uncensored-Q6", "--ttl-seconds", "0"]):
            with patch.object(serve.subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    serve.main(args)
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()

    def test_creation_failure_stops_before_registration(self):
        with patch.object(serve.subprocess, "run", side_effect=RuntimeError("creation failed")), \
                patch.object(serve, "refresh") as refresh:
            self.assertEqual(serve.main(["Qwen3-27B-Uncensored-Q6"]), 1)
            refresh.assert_not_called()

    def test_refresh_failure_retries_selected_alias(self):
        alias = "Qwen3-27B-Uncensored-FP8"
        with patch.object(serve.subprocess, "run"), patch("boto3.client"), \
                patch.object(serve, "wait_for_endpoint"), \
                patch.object(serve, "refresh", side_effect=[Mock(), RuntimeError("offline"), KeyboardInterrupt]) as refresh, \
                patch.object(serve, "refresh_delay", return_value=10), \
                patch.object(serve.time, "sleep") as sleep:
            self.assertEqual(serve.main([alias]), 0)
            self.assertEqual(refresh.call_count, 3)
            for call in refresh.call_args_list:
                self.assertEqual(call.args, (3600, alias))
            sleep.assert_any_call(30)

    def test_registration_cli_selects_alias(self):
        for alias, config in CONFIGS.items():
            with self.subTest(alias=alias), \
                    patch.object(register, "register", return_value={}) as registration, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(register.main([alias, "--ttl-seconds", "900"]), 0)
                registration.assert_called_once_with(config, 900)

    def test_registration_isolated_by_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            source = {"providers": {"llamacpp": {"models": [{
                "thinkingLevelMap": {"high": None},
                "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {}}}]}}}
            path.write_text(json.dumps(source))
            providers = set()
            for alias in CONFIGS:
                before = json.loads(path.read_text())
                report = register.register(path=path, config=CONFIGS[alias], generator=Mock(return_value=alias))
                name, model_id = register.provider_identity(CONFIGS[alias])
                self.assertNotIn(name, providers)
                providers.add(name)
                updated = json.loads(path.read_text())
                provider = updated["providers"].pop(name)
                self.assertEqual(updated, before)
                self.assertIn(f"/endpoints/{alias}/", provider["baseUrl"])
                self.assertEqual(provider["models"][0]["id"], model_id)
                self.assertEqual(provider["models"][0]["contextWindow"], register.context_window(CONFIGS[alias]))
                self.assertEqual(Path(report["token_path"]).read_text(), alias + "\n")
                register.register(path=path, config=CONFIGS[alias], generator=Mock(return_value="refreshed"))
                self.assertEqual(json.loads(path.read_text())["providers"][name], provider)

    def test_refresh_resolves_configured_identity_and_isolates_tokens(self):
        # Exercise serve.refresh -> registration CLI -> config loader -> real file writes.
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.json"
            models.write_text('{"providers": {}}')
            config_path = Path(directory) / "deployment_config.json"
            base = CONFIGS["Qwen3-27B-Uncensored-FP8"]
            configs = {key: {**base, "pi_model_alias": identity} for key, identity in
                       (("first", "custom-one/model/a"), ("second", "custom-two/model/b"))}
            config_path.write_text(json.dumps(configs))
            real_register = register.register

            def invoke(command, **kwargs):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = register.main(command[2:])
                self.assertEqual(status, 0)
                return Mock(stdout=output.getvalue(), stderr="")

            with patch("deployment.CONFIG_PATH", config_path), \
                    patch.object(serve.subprocess, "run", side_effect=invoke), \
                    patch.object(register, "register", side_effect=lambda config, ttl: real_register(
                        config, ttl, path=models, generator=lambda **kw: config["pi_model_alias"])):
                for alias in ("first", "second", "first"):
                    serve.refresh(900, alias)
            document = json.loads(models.read_text())
            self.assertEqual(set(document["providers"]), {"custom-one", "custom-two"})
            for config in configs.values():
                provider, model_id = config["pi_model_alias"].split("/", 1)
                self.assertEqual(document["providers"][provider]["models"][0]["id"], model_id)
                self.assertEqual(register.token_path(models, config).read_text().strip(),
                                 config["pi_model_alias"])

    def test_fp8_without_llamacpp_template(self):
        alias = "Qwen3-27B-Uncensored-FP8"
        provider = register.make_provider({"providers": {}}, "token", CONFIGS[alias])
        self.assertEqual(provider["models"][0]["id"], "/opt/ml/model")
        self.assertEqual(provider["models"][0]["contextWindow"], 32768)
        self.assertNotIn("compat", provider["models"][0])


if __name__ == "__main__":
    unittest.main()
