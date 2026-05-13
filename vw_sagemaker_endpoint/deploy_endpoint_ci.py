"""Create or update the SageMaker endpoint for the VW inference container."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _int_env(name: str) -> int:
    value = _required_env(name)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise RuntimeError(f"{name} must be >= 1")
    return parsed


def _endpoint_exists(client, endpoint_name: str) -> bool:
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        error = exc.response.get("Error", {})
        if error.get("Code") == "ValidationException":
            return False
        raise


def main() -> None:
    region = _required_env("AWS_REGION")
    image_uri = _required_env("IMAGE_URI")
    model_s3_path = _required_env("MODEL_S3_PATH")
    role_arn = _required_env("SAGEMAKER_ROLE_ARN")
    endpoint_name = _required_env("ENDPOINT_NAME")
    instance_type = _required_env("INSTANCE_TYPE")
    instance_count = _int_env("INSTANCE_COUNT")

    client = boto3.client("sagemaker", region_name=region)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_name = f"{endpoint_name}-model-{suffix}"
    endpoint_config_name = f"{endpoint_name}-config-{suffix}"

    print(f"Creating SageMaker model: {model_name}")
    print(f"Using image URI: {image_uri}")
    print(f"Using model artifact: {model_s3_path}")
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_path,
            "Environment": {
                "SAGEMAKER_PROGRAM": "serve.py",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            },
        },
    )

    print(f"Creating endpoint config: {endpoint_config_name}")
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    if _endpoint_exists(client, endpoint_name):
        print(f"Updating existing endpoint: {endpoint_name}")
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            RetainAllVariantProperties=False,
        )
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

    print("Waiting for endpoint to reach InService")
    waiter = client.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 80},
    )

    description = client.describe_endpoint(EndpointName=endpoint_name)
    print(f"Endpoint status: {description['EndpointStatus']}")
    print(f"Endpoint ARN: {description['EndpointArn']}")


if __name__ == "__main__":
    started = time.time()
    try:
        main()
    finally:
        print(f"Deployment script runtime_seconds={time.time() - started:.1f}")
