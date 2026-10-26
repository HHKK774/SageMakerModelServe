"""Load deployment JSON and build SageMaker requests; no AWS calls at import time."""

import argparse
import json
import re
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "deployment_config.json"


def pi_identity(config):
    """Read the explicit Pi provider/model ID; model IDs may contain slashes."""
    alias = config.get("pi_model_alias")
    if not isinstance(alias, str):
        raise ValueError("pi_model_alias must be a provider/model-id string")
    provider, separator, model_id = alias.partition("/")
    if (not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", provider)
            or not model_id or any(char.isspace() or ord(char) < 32 or ord(char) == 127
                                   for char in model_id)):
        raise ValueError("pi_model_alias must contain a filename-safe provider and a nonempty model ID without whitespace")
    return provider, model_id


def load_config(alias):
    """Select an alias from the single project-root deployment configuration."""
    try:
        with CONFIG_PATH.open() as source:
            configs = json.load(source)
        if not isinstance(configs, dict) or not configs:
            raise ValueError("expected a nonempty JSON object keyed by alias")
        if alias not in configs:
            raise ValueError(f"unknown alias {alias!r}; available aliases: {', '.join(sorted(configs))}")
        config = configs[alias]
        if not isinstance(config, dict):
            raise ValueError("expected a JSON object")
        for key in ("name", "region", "execution_role_arn", "image", "model_data_s3", "instance_type"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        pi_identity(config)
        env = config.get("environment")
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError("environment must be an object of string values")
        return config
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{CONFIG_PATH} (alias {alias!r}): {exc}") from exc


def parse_args(description, argv=None, allow_update=False):
    parser = argparse.ArgumentParser(description=description)
    add_alias_argument(parser)
    if allow_update:
        parser.add_argument("--update", action="store_true",
                            help="Recreate an existing model if its container differs (also UPDATE_MODEL=1).")
    return resolve_config(parser, argv)


def add_alias_argument(parser):
    parser.add_argument("alias", metavar="ALIAS",
                        help="Required alias key in project-root deployment_config.json.")


def resolve_config(parser, argv=None):
    args = parser.parse_args(argv)
    try:
        args.config = load_config(args.alias)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def model_container(config):
    return {
        "Image": config["image"],
        "ModelDataSource": {
            "S3DataSource": {
                "S3Uri": config["model_data_s3"],
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        },
        "Environment": dict(config["environment"]),
    }


def production_variant(config):
    return {
        "ModelName": config["name"],
        "VariantName": "All",
        "InitialInstanceCount": 1,
        "InstanceType": config["instance_type"],
        "InitialVariantWeight": 1.0,
        "ModelDataDownloadTimeoutInSeconds": 3600,
        "ContainerStartupHealthCheckTimeoutInSeconds": 300,
    }
