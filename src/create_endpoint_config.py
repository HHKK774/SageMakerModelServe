#!/usr/bin/env python3
"""Create a SageMaker endpoint config for the selected Q6 or FP8 model."""

import sys

from deployment import parse_args, production_variant


def main(argv=None) -> None:
    args = parse_args(__doc__, argv)
    model_name = endpoint_config_name = args.config["name"]

    import boto3

    sagemaker = boto3.client("sagemaker", region_name=args.config["region"])

    # The model must exist before the endpoint config references it
    sagemaker.describe_model(ModelName=model_name)

    try:
        sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint config '{endpoint_config_name}' already exists, nothing to do.")
    except sagemaker.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise
        print(f"Creating endpoint config '{endpoint_config_name}'...")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[production_variant(args.config)],
        )

    desc = sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    pv = desc["ProductionVariants"][0]
    print("\n--- Created/updated endpoint config ---")
    print(f"EndpointConfigName: {desc['EndpointConfigName']}")
    print(f"EndpointConfigArn:  {desc['EndpointConfigArn']}")
    print(f"CreationTime:       {desc['CreationTime']}")
    print(f"ModelName:          {pv['ModelName']}")
    print(f"VariantName:        {pv['VariantName']}")
    print(f"InstanceType:       {pv['InstanceType']}")
    print(f"InitialInstanceCount: {pv['InitialInstanceCount']}")


if __name__ == "__main__":
    sys.exit(main())
