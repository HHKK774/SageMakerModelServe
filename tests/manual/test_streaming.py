#!/usr/bin/env python3
"""Test OpenAI-compatible streaming (SSE) chat completion against the SageMaker endpoint."""

import sys
import time
from datetime import timedelta

import boto3
from openai import OpenAI
from sagemaker.core.token_generator import generate_token

REGION = "us-west-2"
ENDPOINT_NAME = "Qwen3-27B-Uncensored-FP8"
MODEL_FIELD = "/opt/ml/model"  # documented value; local weights path as model name
FALLBACK_MODEL_FIELD = ENDPOINT_NAME


def wait_in_service(timeout_min=60) -> None:
    sm = boto3.client("sagemaker", region_name=REGION)
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
        print(f"endpoint status: {status}", flush=True)
        if status == "InService":
            return
        if status == "Failed":
            raise RuntimeError(f"endpoint Failed: {sm.describe_endpoint(EndpointName=ENDPOINT_NAME).get('FailureReason')}")
        time.sleep(15)
    raise TimeoutError(f"endpoint not InService after {timeout_min} min")


def run_streaming(client: OpenAI, model: str) -> None:
    print(f"\n=== Streaming request (model={model!r}) ===", flush=True)
    t0 = time.time()
    full_text = ""
    chunk_count = 0
    for chunk in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 20, one number per line."}],
        stream=True,
    ):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = delta.content or ""
        chunk_count += 1
        ts = time.time() - t0
        print(f"[chunk {chunk_count:3d}] t={ts:6.2f}s  {piece!r}", flush=True)
        full_text += piece
    print(f"\n--- Stream finished: {chunk_count} content chunks in {time.time() - t0:.2f}s ---")
    print("--- Assembled response ---")
    print(full_text)


def main() -> None:
    wait_in_service()
    token = generate_token(region=REGION, expiry=timedelta(minutes=5))
    base_url = (
        f"https://runtime.sagemaker.{REGION}.amazonaws.com"
        f"/endpoints/{ENDPOINT_NAME}/openai/v1"
    )
    client = OpenAI(base_url=base_url, api_key=token)

    try:
        run_streaming(client, MODEL_FIELD)
        return
    except Exception as e:  # noqa: BLE001 - try a fallback model field before failing
        print(f"\nFirst attempt with model={MODEL_FIELD!r} failed: {type(e).__name__}: {e}", file=sys.stderr)

    print("\nRetrying with model field = endpoint name...", file=sys.stderr)
    run_streaming(client, FALLBACK_MODEL_FIELD)


if __name__ == "__main__":
    main()
