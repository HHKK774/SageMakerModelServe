#!/usr/bin/env python3
"""Attach an inline policy to the SageMaker execution role so realtime endpoint
container logs are shipped to CloudWatch (/aws/sagemaker/Endpoints/<endpoint>/...).
Without it, a failed container startup (CannotStartContainerError) leaves no logs."""

import json

import boto3

REGION = "us-west-2"
ACCOUNT_ID = "ACCOUNT_ID"  # Replace with your AWS account ID before running.
ROLE_NAME = "SageMakerExecutionRole"
POLICY_NAME = "SageMakerCloudWatchLogs"

POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams",
            ],
            "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:/aws/sagemaker/*",
        }
    ],
}


def main() -> None:
    iam = boto3.client("iam", region_name=REGION)
    try:
        iam.put_role_policy(
        RoleName=ROLE_NAME, PolicyName=POLICY_NAME, PolicyDocument=json.dumps(POLICY)
    )
        print(f"inline policy '{POLICY_NAME}' attached to role '{ROLE_NAME}'")
    except iam.exceptions.ClientError as e:
        print(f"error: {e}")


if __name__ == "__main__":
    main()
