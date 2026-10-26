#!/usr/bin/env python3
"""Ensure the selected deployment is InService, register it in Pi, and refresh its token.

Example (from repository root): python3 -m serve Qwen3-27B-Uncensored-Q6
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from deployment import add_alias_argument, resolve_config
from register_pi_model import positive_int

HERE = Path(__file__).resolve().parent


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def wait_for_endpoint(client, config_name, timeout=3600, interval=15, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while True:
        description = client.describe_endpoint(EndpointName=config_name)
        status = description["EndpointStatus"]
        log(f"Endpoint {config_name}: {status}")
        if status == "InService":
            return
        if status == "Failed":
            raise RuntimeError(f"Endpoint Failed: {description.get('FailureReason', 'not reported')}")
        if status not in {"Creating", "Updating", "SystemUpdating", "RollingBack"}:
            raise RuntimeError(f"Unexpected endpoint status: {status}")
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Endpoint did not become InService within 60 minutes")
        sleep(min(interval, remaining))


def refresh(ttl, alias):
    result = subprocess.run(
        [sys.executable, str(HERE / "register_pi_model.py"), alias,
         "--ttl-seconds", str(ttl)], capture_output=True, text=True, timeout=300,
    )
    if result.stdout:
        log(result.stdout.strip())
    if result.stderr:
        log(result.stderr.strip())
    result.check_returncode()
    report = json.loads(result.stdout.strip().splitlines()[-1])
    next_refresh = datetime.fromisoformat(report["next_refresh_at"])
    if next_refresh.tzinfo is None:
        raise ValueError("Refresh deadline must include a timezone")
    return next_refresh


def refresh_delay(next_refresh, now=None):
    now = now or datetime.now(timezone.utc)
    # A deadline already inside the safety window needs an immediate refresh.
    return max(0, (next_refresh - now).total_seconds())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_alias_argument(parser)
    parser.add_argument("--ttl-seconds", type=positive_int, default=3600)
    args = resolve_config(parser, argv)
    config = args.config
    try:
        subprocess.run([sys.executable, str(HERE / "create_endpoint.py"), args.alias], check=True)
        import boto3
        from botocore.config import Config
        client = boto3.client("sagemaker", region_name=config["region"],
                              config=Config(connect_timeout=10, read_timeout=30,
                                            retries={"max_attempts": 3}))
        wait_for_endpoint(client, config["name"])
        next_refresh = refresh(args.ttl_seconds, args.alias)
        while True:
            time.sleep(refresh_delay(next_refresh))
            try:
                next_refresh = refresh(args.ttl_seconds, args.alias)
                # Back off if fallback credentials are already in the safety
                # window, rather than busy-looping against a failing signin API.
                if refresh_delay(next_refresh) == 0:
                    time.sleep(30)
            except Exception as exc:
                log(f"Refresh failed ({type(exc).__name__}); retaining existing configuration, retry in 30s")
                time.sleep(30)
    except KeyboardInterrupt:
        log("Stopped")
        return 0
    except Exception as exc:
        log(f"Fatal: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
