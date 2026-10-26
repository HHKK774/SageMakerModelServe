#!/usr/bin/env python3
"""Poll a SageMaker realtime endpoint until creation succeeds or fails."""

import argparse
from datetime import datetime
import sys
import time

import boto3
from botocore.exceptions import ClientError

# ---------------- Config ----------------
REGION = "us-west-2"
ENDPOINT_NAME = "Qwen3-27B-Uncensored-FP8"
POLL_INTERVAL_SECONDS = 30
# -----------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=ENDPOINT_NAME, help="Endpoint name to check")
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL_SECONDS,
        help="Polling interval in seconds (default: 30)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    sagemaker = boto3.client("sagemaker", region_name=REGION)

    while True:
        try:
            resp = sagemaker.describe_endpoint(EndpointName=args.endpoint)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ValidationException":
                raise
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"[{timestamp}] Endpoint '{args.endpoint}' not found yet (pending).", flush=True)
        else:
            status = resp["EndpointStatus"]
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"[{timestamp}] Endpoint '{args.endpoint}' status: {status}", flush=True)
            if status == "InService":
                print(f"Endpoint '{args.endpoint}' is InService (creation succeeded).")
                return 0
            if status == "Failed":
                reason = resp.get("FailureReason", "no failure reason reported")
                print(f"Endpoint '{args.endpoint}' creation failed: {reason}", file=sys.stderr)
                return 1

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
