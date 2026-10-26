#!/usr/bin/env python3
"""Fetch CloudWatch container logs for the endpoint variant (tail mode)."""

import boto3

REGION = "us-west-2"


def fetch_stream(logs, group: str, stream: str):
    events, token = [], None
    while True:
        kwargs = dict(logGroupName=group, logStreamNames=[stream], limit=100)
        if token:
            kwargs["nextToken"] = token
        r = logs.filter_log_events(**kwargs)
        events += r.get("events", [])
        token = r.get("nextToken")
        if not token:
            break
    return events


def main() -> None:
    logs = boto3.client("logs", region_name=REGION)
    groups = [
        g["logGroupName"]
        for g in logs.describe_log_groups(logGroupNamePrefix="/aws/sagemaker/Endpoints/")["logGroups"]
    ]
    if not groups:
        print("(no /aws/sagemaker/Endpoints log groups yet)")
        return
    for group in groups:
        streams = logs.describe_log_streams(logGroupName=group)["logStreams"]
        for s in streams:
            events = fetch_stream(logs, group, s["logStreamName"])
            print(f"=== {group} / {s['logStreamName']} ({len(events)} events) ===")
            for e in events[-400:]:
                print(e["message"])


if __name__ == "__main__":
    main()
