"""Offline regression tests: boto3 is replaced before any script main is run."""

import argparse
import json
import tempfile
import contextlib
import io
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import create_endpoint
import create_endpoint_config
import create_model
from deployment import load_config, model_container, parse_args, production_variant


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "deployment_config.example.json"
CONFIGS = json.loads(EXAMPLE.read_text())

# Load by path to avoid colliding with the installed Docker SDK package.
server_args = runpy.run_path(str(ROOT / "docker" / "adapter.py"))["server_args"]


class ClientError(Exception):
    def __init__(self, code="ValidationException"):
        self.response = {"Error": {"Code": code}}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("deployment.CONFIG_PATH", EXAMPLE)
        patcher.start()
        self.addCleanup(patcher.stop)
    def test_json_validation(self):
        original = next(iter(CONFIGS.values()))
        invalid = ["{", "[]", "null", "{}", '{"custom": []}',
                   json.dumps({"custom": {**original, "region": ""}}),
                   json.dumps({"custom": {**original, "environment": {"value": 123}}})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment_config.json"
            with patch("deployment.CONFIG_PATH", path):
                with self.assertRaises(argparse.ArgumentTypeError):
                    load_config("custom")
                for contents in invalid:
                    path.write_text(contents)
                    with self.subTest(contents=contents), self.assertRaises(argparse.ArgumentTypeError):
                        load_config("custom")
                custom = {**original, "name": "custom-endpoint", "region": "eu-west-1"}
                path.write_text(json.dumps({"custom": custom}))
                self.assertEqual(parse_args("test", ["custom"]).config, custom)
                self.assertEqual(production_variant(load_config("custom"))["ModelName"], "custom-endpoint")

    def test_pi_alias_validation(self):
        original = next(iter(CONFIGS.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment_config.json"
            for value in (None, "", " ", 123, {}, "model-only", "/model", "provider/",
                          "../model", "bad provider/model", "provider/model\n", "provider/\x00"):
                config = {**original, "pi_model_alias": value}
                if value is None:
                    del config["pi_model_alias"]
                path.write_text(json.dumps({"custom": config}))
                with self.subTest(value=value), patch("deployment.CONFIG_PATH", path):
                    with self.assertRaisesRegex(argparse.ArgumentTypeError, "pi_model_alias"):
                        load_config("custom")

    def test_malformed_json_rejected_before_aws_import(self):
        import register_pi_model
        import src.serve as serve
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment_config.json"
            for contents in ("{", "[]", "{}"):
                path.write_text(contents)
                for module in (create_model, create_endpoint_config, create_endpoint, serve, register_pi_model):
                    with self.subTest(module=module.__name__, contents=contents), \
                            patch("deployment.CONFIG_PATH", path), \
                            patch.dict(sys.modules, {"boto3": None}), \
                            contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as exc:
                            module.main(["Qwen3-27B-Uncensored-Q6"])
                        self.assertEqual(exc.exception.code, 2)

    def test_project_root_config_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            shutil.copytree(ROOT / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copyfile(EXAMPLE, project / "deployment_config.json")
            result = subprocess.run(
                [sys.executable, "-S", "-c",
                 "import sys; sys.path.insert(0, sys.argv[1]); from deployment import parse_args; "
                 "print(parse_args('test', [sys.argv[2]]).config['name'])",
                 str(project / "src"), "Qwen3-27B-Uncensored-Q6"],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "Qwen3-27B-Uncensored-Q6")

    def test_original_values(self):
        expected = {
            "Qwen3-27B-Uncensored-Q6": (
                "sagemaker-llamacpp-qwen38:b11347", "Q6_K", "ml.g7.2xlarge",
                {"SM_LLAMACPP_MODEL": "/opt/ml/model/Qwen3.8-27B-Uncensored-Q6_K.gguf",
                 "SM_LLAMACPP_MMPROJ": "/opt/ml/model/mmproj-Qwen3.8-27B-Uncensored-f16.gguf",
                 "SM_LLAMACPP_N_GPU_LAYERS": "999", "SM_LLAMACPP_CTX_SIZE": "131072",
                 "SM_LLAMACPP_JINJA": ""}),
            "Qwen3-27B-Uncensored-FP8": (
                "huggingface-vllm:latest", "FP8", "ml.g6e.xlarge",
                {"SM_VLLM_MODEL": "/opt/ml/model", "SM_VLLM_HOST": "0.0.0.0",
                 "SM_VLLM_MAX_MODEL_LEN": "32768", "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.90"}),
        }
        self.assertEqual(set(CONFIGS), set(expected))
        for name, (image, suffix, instance, env) in expected.items():
            with self.subTest(config=name):
                args = parse_args("test", [name])
                self.assertEqual(args.alias, name)
                self.assertEqual(args.config["name"], name)
                self.assertEqual(args.config["region"], "us-west-2")
                self.assertEqual(args.config["execution_role_arn"],
                                 "arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole")
                container = model_container(args.config)
                self.assertEqual(container, {
                    "Image": "ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/" + image,
                    "Environment": env,
                    "ModelDataSource": {"S3DataSource": {
                        "S3Uri": "s3://MODEL_BUCKET/Qwen3.8-27B-Uncensored-" + suffix + "/",
                        "S3DataType": "S3Prefix", "CompressionType": "None"}},
                })
                self.assertEqual(production_variant(CONFIGS[name]), {
                    "ModelName": name, "VariantName": "All", "InitialInstanceCount": 1,
                    "InstanceType": instance, "InitialVariantWeight": 1.0,
                    "ModelDataDownloadTimeoutInSeconds": 3600,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 300,
                })
        self.assertEqual(server_args(CONFIGS["Qwen3-27B-Uncensored-Q6"]["environment"]), [
            "/app/llama-server", "--host", "127.0.0.1", "--port", "58585",
            "--model", "/opt/ml/model/Qwen3.8-27B-Uncensored-Q6_K.gguf",
            "--n-gpu-layers", "999", "--ctx-size", "131072",
            "--mmproj", "/opt/ml/model/mmproj-Qwen3.8-27B-Uncensored-f16.gguf", "--jinja"])

    def test_cli_without_site_packages(self):
        for script in ("create_model.py", "create_endpoint_config.py", "create_endpoint.py", "serve.py", "register_pi_model.py"):
            for argv, status in ((["--help"], 0), (["unknown"], 2), ([str(ROOT / "deployment_config.json")], 2), (["--config", "Qwen3-27B-Uncensored-Q6"], 2), ([], 2)):
                with self.subTest(script=script, argv=argv):
                    result = subprocess.run([sys.executable, "-S", str(ROOT / "src" / script), *argv],
                                            cwd=tempfile.gettempdir(),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, status)
                    output = result.stdout + result.stderr
                    self.assertIn("ALIAS", output)

    def test_creation_requests(self):
        for name in CONFIGS:
            with self.subTest(config=name):
                client = Mock()
                client.exceptions.ClientError = ClientError
                desc = {"ModelName": name, "ModelArn": "mock", "CreationTime": "mock",
                        "ExecutionRoleArn": CONFIGS[name]["execution_role_arn"],
                        "Containers": [model_container(CONFIGS[name])]}
                endpoint_config = {"EndpointConfigName": name, "EndpointConfigArn": "mock",
                                   "CreationTime": "mock", "ProductionVariants": [production_variant(CONFIGS[name])]}
                client.describe_model.side_effect = [ClientError(), desc, desc]
                client.describe_endpoint_config.side_effect = [ClientError(), endpoint_config]
                client.describe_endpoint.side_effect = ClientError()
                boto = SimpleNamespace(client=Mock(return_value=client))
                with patch.dict(sys.modules, {"boto3": boto}), contextlib.redirect_stdout(io.StringIO()):
                    create_model.main([name])
                    create_endpoint_config.main([name])
                    self.assertEqual(create_endpoint.main([name]), 0)
                client.create_model.assert_called_once_with(
                    ModelName=name, ExecutionRoleArn=CONFIGS[name]["execution_role_arn"],
                    Tags=[], Containers=[model_container(CONFIGS[name])])
                client.create_endpoint_config.assert_called_once_with(
                    EndpointConfigName=name, ProductionVariants=[production_variant(CONFIGS[name])])
                client.create_endpoint.assert_called_once_with(EndpointName=name, EndpointConfigName=name)
                for call in boto.client.call_args_list:
                    self.assertEqual(call.args, ("sagemaker",))
                    self.assertEqual(call.kwargs, {"region_name": "us-west-2"})

    def test_model_update(self):
        for name in CONFIGS:
            for changed in (False, True):
                for use_env in (False, True):
                    with self.subTest(config=name, changed=changed, use_env=use_env):
                        client = Mock()
                        client.exceptions.ClientError = ClientError
                        desc = {"ModelName": name, "ModelArn": "mock", "CreationTime": "mock",
                                "ExecutionRoleArn": CONFIGS[name]["execution_role_arn"],
                                "Containers": [model_container(CONFIGS[name])]}
                        old = model_container(CONFIGS[name])
                        if changed:
                            old["Image"] = "old-image"
                        client.describe_model.side_effect = [{"PrimaryContainer": old}, desc]
                        argv = [name] + ([] if use_env else ["--update"])
                        with patch.dict(sys.modules, {"boto3": SimpleNamespace(client=Mock(return_value=client))}), \
                                patch.dict(os.environ, {"UPDATE_MODEL": "1" if use_env else "0"}), \
                                contextlib.redirect_stdout(io.StringIO()):
                            create_model.main(argv)
                        self.assertEqual(client.delete_model.call_count, int(changed))
                        self.assertEqual(client.create_model.call_count, int(changed))
                        if changed:
                            client.delete_model.assert_called_once_with(ModelName=name)


if __name__ == "__main__":
    unittest.main()
