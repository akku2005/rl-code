# VW SageMaker Endpoint CI/CD

This folder contains the Docker inference container and CI/CD deployment script for serving the trained Vowpal Wabbit contextual-bandit model as an AWS SageMaker real-time endpoint.

The existing notebooks are not part of the deployment runtime. They continue to train and evaluate the model artifact. This folder starts from the packaged model artifact in S3:

```text
s3://aks-nvtabular-data/models/pl-aip-uplift/model.tar.gz
```

SageMaker extracts that archive into `/opt/ml/model/`. The archive must contain:

```text
pl_aip_uplift_model.vw
```

## Files

- `Dockerfile` builds the SageMaker inference image.
- `requirements.txt` installs VW, FastAPI, Gunicorn, Uvicorn, boto3, and runtime dependencies.
- `serve.py` exposes `GET /ping` and `POST /invocations` for SageMaker.
- `predictor.py` loads `/opt/ml/model/pl_aip_uplift_model.vw` with Python `vowpalwabbit` and ranks actions.
- `formatter.py` converts request JSON into multiline VW `cb_adf` format.
- `deploy_endpoint_ci.py` creates or updates the SageMaker Model, Endpoint Config, and Endpoint.
- `test_payload.json` is used by GitHub Actions and manual CLI tests.
- `.github/workflows/deploy-vw-endpoint.yml` builds, pushes, deploys, and smoke-tests the endpoint.

## Required AWS Services

- S3 for the trained `model.tar.gz` artifact.
- Amazon ECR for the Docker image.
- SageMaker for the real-time model endpoint.
- IAM for GitHub OIDC and SageMaker execution roles.
- CloudWatch Logs for container and endpoint diagnostics.

## Required GitHub Secrets

Create these repository secrets in GitHub:

```text
AWS_ROLE_TO_ASSUME
SAGEMAKER_ROLE_ARN
```

`AWS_ROLE_TO_ASSUME` is the IAM role GitHub Actions assumes through OIDC.

`SAGEMAKER_ROLE_ARN` is the SageMaker execution role passed to `CreateModel`. The GitHub OIDC role needs `iam:PassRole` permission for this role.

## Deployment Defaults

The workflow uses these defaults:

```text
AWS_REGION=ap-south-1
ECR_REPOSITORY=vw-rl-endpoint
ENDPOINT_NAME=vw-pl-aip-uplift-endpoint
MODEL_S3_PATH=s3://aks-nvtabular-data/models/pl-aip-uplift/model.tar.gz
INSTANCE_TYPE=ml.m5.large
INSTANCE_COUNT=1
```

## Run Deployment

Commit and push the deployment files to `main`:

```bash
git add vw_sagemaker_endpoint .github/workflows/deploy-vw-endpoint.yml
git commit -m "Add VW SageMaker endpoint deployment"
git push origin main
```

The workflow runs on pushes to `main` when `vw_sagemaker_endpoint/**` or the workflow file changes. It can also be started manually with `workflow_dispatch`.

## Manual Endpoint Test

After the endpoint is `InService`, invoke it manually:

```bash
aws sagemaker-runtime invoke-endpoint \
  --region ap-south-1 \
  --endpoint-name vw-pl-aip-uplift-endpoint \
  --content-type application/json \
  --body fileb://vw_sagemaker_endpoint/test_payload.json \
  response.json

cat response.json
```

## Package Model Artifact Manually

If `model.tar.gz` does not exist yet, create an archive that places the VW model at the archive root:

```bash
mkdir -p model_package
cp pl_aip_uplift_model.vw model_package/
tar -C model_package -czf model.tar.gz pl_aip_uplift_model.vw
aws s3 cp model.tar.gz s3://aks-nvtabular-data/models/pl-aip-uplift/model.tar.gz
```

SageMaker will extract this to:

```text
/opt/ml/model/pl_aip_uplift_model.vw
```

## One-Time AWS Setup

Create an IAM role for GitHub OIDC with trust limited to your GitHub repository and branch. The workflow role needs permissions for:

- `ecr:CreateRepository`
- `ecr:DescribeRepositories`
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`
- `s3:ListBucket` on `aks-nvtabular-data`
- `s3:GetObject` on `aks-nvtabular-data/models/pl-aip-uplift/model.tar.gz`
- `sagemaker:CreateModel`
- `sagemaker:CreateEndpointConfig`
- `sagemaker:CreateEndpoint`
- `sagemaker:UpdateEndpoint`
- `sagemaker:DescribeEndpoint`
- `sagemaker:InvokeEndpoint`
- `iam:PassRole` for the SageMaker execution role

The SageMaker execution role needs permission to pull from ECR, read the model artifact from S3, and write CloudWatch logs.

## Common Errors

`ECR permission denied`: the GitHub OIDC role is missing ECR write permissions or `ecr:GetAuthorizationToken`.

`iam:PassRole denied`: the GitHub OIDC role cannot pass `SAGEMAKER_ROLE_ARN` to SageMaker.

`model artifact not found`: `aws s3 ls s3://aks-nvtabular-data/models/pl-aip-uplift/model.tar.gz` fails or the role cannot read the object.

`endpoint failed due to Docker container startup`: check CloudWatch logs for missing Python dependencies, VW model load failures, or port binding issues.

`/ping health check failure`: the model was not loaded from `/opt/ml/model/pl_aip_uplift_model.vw`, or the container did not start on port `8080`.
