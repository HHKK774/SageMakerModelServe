#!/usr/bin/env python3
"""Register a configured SageMaker endpoint in Pi; stdout ends with a JSON TTL report."""

import argparse
import copy
from datetime import datetime, timedelta, timezone
from functools import partial
from types import SimpleNamespace
import fcntl
import json
import os
from pathlib import Path
import shutil
import shlex
import sys
import tempfile

from deployment import add_alias_argument, resolve_config, pi_identity
MODELS_PATH = Path.home() / ".pi" / "agent" / "models.json"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("TTL must be a positive integer in seconds")
    return number


def signing_credentials():
    """Pin credentials and their matching expiry; never remove the login cache."""
    from botocore.credentials import Credentials, LoginCredentialFetcher
    from botocore.session import Session

    credentials = Session().get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS signing credentials found")
    if credentials.method == "login":
        # Snapshot before refreshing: even a mandatory refresh failure must allow
        # signing with the still-valid cached credentials, without retrying it.
        frozen = credentials._frozen_credentials
        expiry = credentials._expiry_time
        try:
            fetcher = credentials._refresh_using.__self__
            if not isinstance(fetcher, LoginCredentialFetcher):
                raise RuntimeError("Unsupported botocore login refresh callback")
            fetcher = copy.copy(fetcher)
            # Bypass only this fetcher's cached-access-token reuse check. Keep
            # botocore's DPoP, refresh-token rotation, and cache persistence intact.
            fetcher._REFRESH_THRESHOLD = float("inf")
            metadata = fetcher.refresh_credentials()
            frozen = Credentials(metadata["access_key"], metadata["secret_key"],
                                 metadata["token"]).get_frozen_credentials()
            expiry = datetime.fromisoformat(metadata["expiry_time"].replace("Z", "+00:00"))
        except Exception as exc:
            print(f"WARNING: forced login refresh failed ({type(exc).__name__}); "
                  "using cached credentials and their real expiry", file=sys.stderr, flush=True)
    else:
        print("WARNING: forced refresh unavailable for non-login credentials; "
              "using current credentials", file=sys.stderr, flush=True)
        frozen = credentials.get_frozen_credentials()
        expiry = getattr(credentials, "_expiry_time", None)
        if frozen.token and expiry is None:
            raise RuntimeError("Temporary signing credential expiry is unknown; refusing unsafe scheduling")
    if expiry is not None and expiry <= datetime.now(timezone.utc):
        raise RuntimeError("Signing credentials have expired")
    return Credentials(frozen.access_key, frozen.secret_key, frozen.token), expiry


def refresh_schedule(sign_time, token_ttl, credential_expiry):
    effective = sign_time + timedelta(seconds=token_ttl)
    if credential_expiry is not None:
        effective = min(effective, credential_expiry)
    return {
        "sign_time": sign_time.isoformat(),
        "credential_expires_at": credential_expiry.isoformat() if credential_expiry else None,
        "effective_expiry": effective.isoformat(),
        "next_refresh_at": (effective - timedelta(seconds=120)).isoformat(),
        "actual_ttl_seconds": (effective - sign_time).total_seconds(),
    }


def mint_token(requested_ttl, generator=None, maximum_ttl=None, *, region):
    if generator is None:
        from sagemaker.core.token_generator import TOKEN_DURATION, generate_token
        generator = generate_token
        maximum_ttl = int(TOKEN_DURATION)
    maximum_ttl = maximum_ttl or 43200
    # Try the requested lifetime first, then the SDK maximum and shorter lifetimes.
    candidates = dict.fromkeys([requested_ttl, maximum_ttl, 21600, 10800, 3600, 1800, 900, 300, 60, 1])
    for ttl in candidates:
        if ttl != requested_ttl and ttl > maximum_ttl:
            continue
        try:
            token = generator(region=region, expiry=timedelta(seconds=ttl))
            if not isinstance(token, str) or not token.strip() or token.startswith("Bearer "):
                raise ValueError("SDK did not return a nonempty bare token")
            return token, ttl
        except Exception as exc:
            # Do not print exception bodies: authentication errors may contain secrets.
            print(f"Token TTL {ttl}s rejected/failed ({type(exc).__name__}); trying fallback", file=sys.stderr, flush=True)
    raise RuntimeError("Token generation failed for all supported fallback lifetimes")


def provider_identity(config):
    return pi_identity(config)


def token_path(models_path, config):
    provider, _ = provider_identity(config)
    return Path(models_path).absolute().with_name(provider + ".token")


def token_command(path):
    # Pi executes !commands afresh for every request (not its cached auth.json path).
    return "!/bin/cat " + shlex.quote(str(Path(path).absolute()))


def write_token(path, token):
    """Readers see either complete token; mkstemp keeps secrets mode 0600."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(token + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        if path.read_text() != token + "\n":
            raise RuntimeError("Post-write token verification failed")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ContextConfigError(ValueError):
    pass


def context_window(config):
    env = config["environment"]
    key = "SM_VLLM_MAX_MODEL_LEN" if "SM_VLLM_MODEL" in env else "SM_LLAMACPP_CTX_SIZE"
    try:
        value = env[key]
        if not isinstance(value, str):
            raise ValueError
        size = int(value)
        if size <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ContextConfigError(
            f"{config['name']}: {key} must be a positive integer string"
        ) from exc
    return size


def make_provider(document, api_key, config):
    _, model_id = provider_identity(config)
    if "SM_VLLM_MODEL" in config["environment"]:
        # Do not copy llama.cpp-specific template options to vLLM.
        return {
            "baseUrl": f"https://runtime.sagemaker.{config['region']}.amazonaws.com/endpoints/{config['name']}/openai/v1",
            "api": "openai-completions",
            "apiKey": api_key,
            "models": [{"id": model_id, "name": f"{config['name']} (SageMaker)",
                        "contextWindow": context_window(config), "input": ["text"]}],
        }
    models = document["providers"]["llamacpp"]["models"]
    sources = [m for m in models if m.get("compat", {}).get("thinkingFormat") == "chat-template"
               and "chatTemplateKwargs" in m["compat"] and "thinkingLevelMap" in m]
    if len(sources) != 1:
        raise ValueError("Expected exactly one llamacpp model with thinkingLevelMap and chat-template compat")
    source = sources[0]
    return {
        "baseUrl": f"https://runtime.sagemaker.{config['region']}.amazonaws.com/endpoints/{config['name']}/openai/v1",
        "api": "openai-completions",
        "apiKey": api_key,
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
        "models": [{
            "id": model_id,
            "name": f"{config['name']} (SageMaker)",
            "contextWindow": context_window(config),
            "reasoning": True,
            "input": ["text", "image"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "thinkingLevelMap": copy.deepcopy(source["thinkingLevelMap"]),
            "compat": copy.deepcopy(source["compat"]),
        }],
    }


def register(config, requested_ttl=3600, path=MODELS_PATH, generator=None, credential_expiry=None):
    path = Path(path).absolute()
    provider_name, model_id = provider_identity(config)
    secret_path = token_path(path, config)
    api_key = token_command(secret_path)
    backup = path.with_name(path.name + ".bak")
    # A separate lock survives atomic replacement and serializes our refresh writers.
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = path.read_bytes()
        document = json.loads(original)
        size = context_window(config)
        if provider_name in document["providers"]:
            provider = document["providers"][provider_name]
            targets = [model for model in provider["models"] if model.get("id") == model_id]
            if len(targets) != 1:
                raise ValueError(f"Expected exactly one registered model with id {model_id}")
            targets[0]["contextWindow"] = size
            provider["apiKey"] = api_key
        else:
            document["providers"][provider_name] = make_provider(document, api_key, config)
        suffix = None
        if generator is None:
            from sagemaker.core.token_generator import generate_token
            credentials, credential_expiry = signing_credentials()
            suffix = credentials.access_key[-4:]
            generator = partial(generate_token, aws_credentials_provider=SimpleNamespace(load=lambda: credentials))
        sign_time = datetime.now(timezone.utc).replace(microsecond=0)
        token, ttl = mint_token(requested_ttl, generator, region=config["region"])
        report = refresh_schedule(sign_time, min(requested_ttl, ttl), credential_expiry)
        report.update(event="bearer_refresh", credential_access_key_suffix=suffix,
                      requested_ttl_seconds=requested_ttl, signing_ttl_seconds=ttl)
        if path.read_bytes() != original:
            raise RuntimeError("models.json changed during token generation; refusing to overwrite")
        # Publish before the command config, so it never points to a missing file.
        write_token(secret_path, token)
        report.update(provider=provider_name, models_path=str(path), token_path=str(secret_path),
                      backup_path=str(backup))
        # Once context/key are synced, refreshes replace only the token.
        if json.loads(original) == document:
            return report
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(document, out, indent=2, ensure_ascii=False)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            if path.read_bytes() != original:
                raise RuntimeError("models.json changed during token generation; refusing to overwrite")
            # Keep the original migration backup across later context updates.
            if not backup.exists():
                shutil.copy2(path, backup)
                backup.chmod(0o600)
            os.replace(temporary, path)
            with path.open() as written:
                if json.load(written) != document:
                    raise RuntimeError("Post-write JSON verification failed")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_alias_argument(parser)
    parser.add_argument("--ttl-seconds", type=positive_int, default=3600)
    args = resolve_config(parser, argv)
    try:
        report = register(args.config, args.ttl_seconds)
    except ContextConfigError as exc:
        print(f"Registration failed: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"Registration failed ({type(exc).__name__})", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
