#!/usr/bin/env python3
"""Create a SageMaker realtime endpoint for the selected Q6 or FP8 configuration."""

import sys

from deployment import parse_args


def main(argv=None) -> int:
    args = parse_args(__doc__, argv)
    endpoint_name = endpoint_config_name = args.config["name"]

    import boto3

    sagemaker = boto3.client("sagemaker", region_name=args.config["region"])

    try:
        desc = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print(f"Endpoint '{endpoint_name}' already exists, status: {status}")
        if status == "InService":
            print("Endpoint is InService, skipping creation.")
            return 0
        if status == "Failed":
            reason = desc.get("FailureReason", "no failure reason reported")
            print(f"Endpoint is Failed: {reason}", file=sys.stderr)
            return 1
        # Still Creating/Updating: leave it be; the wait loop can take over.
        print("Endpoint still in progress, nothing to do.")
        return 0
    except sagemaker.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise
        print(f"Creating endpoint '{endpoint_name}'...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        print("create_endpoint called; waiting for InService (see wait loop).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
