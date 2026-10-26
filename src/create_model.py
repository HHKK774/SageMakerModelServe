#!/usr/bin/env python3
"""Create a SageMaker Model using the selected Q6 or FP8 deployment configuration."""

import os
import sys

from deployment import model_container, parse_args


def create_model(sagemaker, config) -> None:
    config_name = config["name"]
    print(f"Creating model '{config_name}'...")
    sagemaker.create_model(
        ModelName=config_name,
        ExecutionRoleArn=config["execution_role_arn"],
        Tags=[],
        Containers=[model_container(config)],
    )


def main(argv=None) -> None:
    args = parse_args(__doc__, argv, allow_update=True)
    model_name = args.config["name"]
    container = model_container(args.config)

    # Parse first so --help/invalid config works even without boto3 or credentials.
    import boto3

    sagemaker = boto3.client("sagemaker", region_name=args.config["region"])

    # Models are immutable in SageMaker: to change settings, delete + recreate.
    force_update = args.update or os.environ.get("UPDATE_MODEL") == "1"

    try:
        existing = sagemaker.describe_model(ModelName=model_name)
        if force_update:
            cur_container = existing.get("Containers", [existing.get("PrimaryContainer", {})])[0]
            if (cur_container.get("Environment", {}) == container["Environment"]
                    and cur_container.get("Image") == container["Image"]):
                print(f"Model '{model_name}' already matches config, nothing to do.")
            else:
                print(f"Model '{model_name}' image or environment differs; deleting and recreating...")
                sagemaker.delete_model(ModelName=model_name)
                create_model(sagemaker, args.config)
        else:
            print(f"Model '{model_name}' already exists, nothing to do. (use --update to force re-creation)")
    except sagemaker.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ValidationException":
            create_model(sagemaker, args.config)
        elif code == "AccessDeniedException":
            err = e.response["Error"]
            print(f"AccessDenied: not enough permissions (action={err.get('Type', '?')}, message={err.get('Message', '?')}). Exiting.")
            sys.exit(1)
        else:
            raise

    desc = sagemaker.describe_model(ModelName=model_name)
    print("\n--- Created/updated model ---")
    print(f"ModelName:        {desc['ModelName']}")
    print(f"ModelArn:         {desc['ModelArn']}")
    print(f"CreationTime:     {desc['CreationTime']}")
    print(f"Image:            {desc['Containers'][0]['Image']}")
    src = desc["Containers"][0].get("ModelDataSource", {}).get("S3DataSource", {})
    print(f"ModelDataSource:  {src.get('S3Uri')} ({src.get('S3DataType')}/{src.get('CompressionType')})")
    print(f"ExecutionRoleArn: {desc['ExecutionRoleArn']}")


if __name__ == "__main__":
    sys.exit(main())
