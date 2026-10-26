# SageMaker Model Serve

Manage SageMaker real-time inference endpoints for llama.cpp GGUF (Q6) and vLLM (FP8) from a shared deployment configuration. Register endpoints in Pi and refresh their access tokens automatically.

## Project layout

```text
serve.py                        Root module entry point
src/                            Deployment, Pi registration, token refresh, and operations
tests/                          Test suite
  manual/test_streaming.py       Streaming inference example
docker/                         llama.cpp container build context
deployment_config.example.json  Example deployment configuration
deployment_config.json          Local deployment configuration, ignored by Git
pytest.ini                      Test discovery and import paths
```

## Setup

Run commands from the repository root. Use Python 3.11+ on macOS or Linux; Pi registration uses `fcntl` and `/bin/cat`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install boto3 sagemaker pytest
```

Token generation requires a SageMaker SDK with `sagemaker.core.token_generator`. AWS login credential refresh requires botocore's `LoginCredentialFetcher`.

Prepare:

- AWS credentials through the standard AWS configuration or environment variables. For login credentials, complete AWS CLI login first. Temporary credentials need an accessible expiration time.
- An ECR image, an S3 model prefix, a SageMaker execution role, and GPU instance quota and capacity in the deployment region.
- Caller permissions for SageMaker operations, endpoint access, and `iam:PassRole`; execution-role permissions for ECR, S3 reads, and CloudWatch logs.
- An existing `~/.pi/agent/models.json`. For initial Q6 registration, `providers.llamacpp.models` must contain exactly one template model with `thinkingLevelMap`, `compat.thinkingFormat: "chat-template"`, and `compat.chatTemplateKwargs`. FP8 registration can start with `{"providers": {}}`.

## Configuration

Copy the example if the local configuration does not exist:

```bash
[ -e deployment_config.json ] || cp deployment_config.example.json deployment_config.json
```

Edit `deployment_config.json` with your account ID, model bucket, execution role, image URI, and deployment settings.

Scripts load **`deployment_config.json` from the project root**, resolved relative to the source files. The required CLI argument **`ALIAS`** selects a top-level JSON key.

The example includes `Qwen3-27B-Uncensored-Q6` and `Qwen3-27B-Uncensored-FP8`. The following table covers every example field. Script names refer to files under `src/`; **registration** means `register_pi_model.py`, both standalone and on each refresh invoked by `serve`.

| Field | Purpose and consuming stage/scripts |
| --- | --- |
| Top-level alias key | Required positional `ALIAS` for all three creation commands, `serve`, and registration; selects the entry, rather than naming an AWS resource directly. |
| `name` | Shared Model, EndpointConfig, and Endpoint name. `create_model.py` names the Model; `create_endpoint_config.py` references that Model and names its EndpointConfig; `create_endpoint.py` names the Endpoint and references that EndpointConfig. `serve.py` polls this Endpoint. Registration derives the initial display name and endpoint URL from it. |
| `pi_model_alias` | Required Pi identity in `provider/model-id` form, used by registration and every `serve` refresh for provider lookup, model ID, and the provider token filename. Not sent to AWS. |
| `region` | AWS client region for all creation scripts and `serve.py` polling; registration uses it for token generation and the initial endpoint URL. |
| `execution_role_arn` | Execution role passed to AWS by `create_model.py`. |
| `image` | Container image passed to AWS by `create_model.py`. |
| `model_data_s3` | Model-file prefix passed by `create_model.py` as `S3Prefix` with `CompressionType=None`, downloaded to `/opt/ml/model`. |
| `instance_type` | Instance type in the production variant created by `create_endpoint_config.py`. |
| `environment` | Entire string-valued object copied into the Model's container environment by `create_model.py`. Registration also reads the specific keys below. |
| `environment.SM_LLAMACPP_MODEL` | Q6 model file path for the container; passed through by `create_model.py`. |
| `environment.SM_LLAMACPP_MMPROJ` | Vision projector file path for the Q6 container; passed through by `create_model.py`. |
| `environment.SM_LLAMACPP_N_GPU_LAYERS` | GPU layer offload setting for the Q6 container; passed through by `create_model.py`. |
| `environment.SM_LLAMACPP_CTX_SIZE` | Q6 container context size; registration reads it as a positive integer string for Pi's `contextWindow`. |
| `environment.SM_LLAMACPP_JINJA` | Enables Jinja in the Q6 container by being present, including with an empty-string value; passed through by `create_model.py`. |
| `environment.SM_VLLM_MODEL` | FP8 container model path; its presence selects vLLM registration behavior. |
| `environment.SM_VLLM_HOST` | FP8 container bind address; passed through by `create_model.py`. |
| `environment.SM_VLLM_MAX_MODEL_LEN` | FP8 container context limit; registration reads it as a positive integer string for Pi's `contextWindow`. |
| `environment.SM_VLLM_GPU_MEMORY_UTILIZATION` | FP8 container GPU memory utilization setting; passed through by `create_model.py`. |

Match the Q6 model and projector paths to the files in S3. For FP8, use an image supporting the listed `SM_VLLM_*` variables. All environment keys are passed to model creation, even when also used for Pi metadata.

Every command loads and validates the same entry through `deployment.py`: `name`, `region`, `execution_role_arn`, `image`, `model_data_s3`, and `instance_type` must be nonempty strings, and `environment` must be an object of strings. `pi_model_alias` must contain a provider made of letters, digits, dots, underscores, or hyphens (starting with a letter or digit), followed by `/` and a nonempty model ID without whitespace or control characters. Model IDs may contain further slashes. This shared validation does not mean every stage sends all fields to AWS.

### Pi identity and refresh settings

The positional deployment `ALIAS` selects an entry; that entry's `pi_model_alias` explicitly selects its Pi identity. There is no nested `pi.*` configuration. Other Pi settings are derived from the deployment and the existing Pi template.

- The Q6 example configures `sagemaker-qwen3-q6/qwen38-27b-uncensored-q6-sm`.
- The FP8 example configures `sagemaker-qwen3-27b-uncensored-fp8//opt/ml/model`: the first slash separates the provider from model ID `/opt/ml/model`.
- Use distinct providers for independent deployments: the URL and token belong to the provider. Neither the deployment key, AWS `name`, nor container model path overrides `pi_model_alias`.
- Initial registration builds the SageMaker OpenAI-compatible URL from `region` and `name`. Q6 copies thinking/template settings from the existing Pi `llamacpp` template; FP8 creates a text-only model without that template.
- For an existing provider, registration requires exactly one matching model ID and updates only its `contextWindow` and the provider's `apiKey` command, preserving other metadata, including the existing URL.
- Token lifetime is controlled by CLI `--ttl-seconds`, not a JSON field. Signing credentials and their expiration come from AWS credentials, not the deployment entry.

## Deploy and run

Deploy in this order: **Model → EndpointConfig → Endpoint → InService → Pi registration/token refresh**. An EndpointConfig (the “create config” step) is an AWS resource, distinct from the local `deployment_config.json` file.

### Creation commands

Each command takes the positional alias selecting the local configuration entry:

| Command | Purpose | Existing-resource behavior |
| --- | --- | --- |
| `python3 src/create_model.py ALIAS` | Creates the SageMaker Model: execution role, image, model S3 source, and container environment. | Leaves an existing Model unchanged by default. `--update` or `UPDATE_MODEL=1` enables conditional deletion/recreation, detailed below. |
| `python3 src/create_endpoint_config.py ALIAS` | First checks that the Model exists, then creates an EndpointConfig referencing it and defining compute. | Leaves an existing EndpointConfig unchanged; still checks the Model first. |
| `python3 src/create_endpoint.py ALIAS` | Requests an Endpoint using the existing same-named EndpointConfig. It does not create the Model or EndpointConfig, wait for readiness, or register Pi. | Skips creation for `InService`; reports `Failed` and exits with status 1; leaves other existing statuses alone and returns successfully. |

The EndpointConfig has one production variant named `All`, with `InitialInstanceCount=1`, `InitialVariantWeight=1.0`, model download timeout `3600` seconds, and container startup health-check timeout `300` seconds. These values are fixed in `deployment.py`, not configurable JSON fields. Endpoint creation passes only `EndpointName` and `EndpointConfigName`, both from `name`, using the configured `region`.

### Start serving

Create the Model and EndpointConfig, then let `serve` create or reuse the Endpoint and run the token-refresh loop:

```bash
ALIAS=Qwen3-27B-Uncensored-Q6
python3 src/create_model.py "$ALIAS"
python3 src/create_endpoint_config.py "$ALIAS"
python3 -m serve "$ALIAS" --ttl-seconds 3600
```

The main entry point is `python3 -m serve ALIAS`. Pass the deployment key, not the `pi_model_alias` value; the selected entry supplies the Pi identity for initial registration and every refresh. `--ttl-seconds` accepts a positive integer and defaults to `3600`. View options with:

```bash
python3 -m serve --help
```

`serve` always invokes `src/create_endpoint.py ALIAS` first:

- **Absent Endpoint:** requests creation from the existing same-named EndpointConfig. Create the Model and EndpointConfig beforehand; `serve` does not invoke their creation scripts.
- **Existing `InService` Endpoint:** reuses it, then registers it in Pi and starts refreshes.
- **Existing Endpoint in progress:** waits for `InService`, polling every 15 seconds for up to 60 minutes. The wait loop accepts `Creating`, `Updating`, `SystemUpdating`, and `RollingBack`.
- **Failed Endpoint or unexpected wait status:** exits with an error rather than recreating the Endpoint.

Thus `serve` uses endpoint-creation settings (`name` and `region`) through its subprocess, plus those fields for polling. Its registration subprocess also uses the identity/context fields listed above. It does not apply image, role, S3 source, instance type, or container environment changes to an existing AWS deployment.

Refreshes are scheduled 120 seconds before the earlier of the effective token expiration and signing-credential expiration. The token generator tries the requested TTL first, then other supported lifetimes. After successful initial registration, refresh failures retain the current configuration and retry after 30 seconds. Initial registration failure exits the command. Each registration subprocess reloads the JSON, while endpoint polling uses the configuration loaded when `serve` started. Ctrl-C stops the local loop; the Endpoint remains running.

To create the endpoint and register it separately:

```bash
python3 src/create_endpoint.py "$ALIAS"
# Once the endpoint is InService, register or refresh its token once.
python3 src/register_pi_model.py "$ALIAS" --ttl-seconds 900
```

Registration updates `~/.pi/agent/models.json`, preserves other providers, and creates `models.json.bak` on the first configuration change. Each provider's token is stored in a `.token` file in the same directory with mode `0600`. Pi reads the token dynamically for each request.

### Apply configuration changes

Editing `deployment_config.json` does not automatically modify already-created AWS resources. Each creation stage captures its own settings; subsequent commands normally reuse the resources by name.

For an existing Model, request conditional recreation with:

```bash
python3 src/create_model.py "$ALIAS" --update
# Equivalent:
UPDATE_MODEL=1 python3 src/create_model.py "$ALIAS"
```

This compares only the existing container's **image and environment**. If either differs, it deletes and recreates the Model using all current Model settings, including role and S3 source. If both match, it does nothing—even if only `execution_role_arn` or `model_data_s3` changed. This operation affects the Model resource, not an already-running Endpoint.

For a new deployment with changed creation settings, use a new AWS `name` (optionally a new alias), then run `create_model.py`, `create_endpoint_config.py`, and `python3 -m serve` in order. A new alias alone still targets the old resources if `name` is unchanged. Set a distinct `pi_model_alias` provider for a separate Pi identity; changing the AWS name alone does not change Pi identity.

To reuse the same AWS name, manage the affected resource deletion/recreation or Endpoint update separately through SageMaker, then run the applicable creation commands in dependency order. The scripts include no EndpointConfig replacement or Endpoint update command; rerunning them against existing resources retains the existing configuration. An `instance_type` edit needs a new EndpointConfig and an Endpoint deployment using it. Image, role, S3 source, and environment edits need the corresponding Model and Endpoint deployment changes.

Registration changes are separate: the next registration/refresh reads the local context setting and updates Pi's `contextWindow`, but does not change the running container's context limit. Keep that metadata aligned with the deployed container. For an existing Pi provider, manage URL or model-identity changes in Pi's configuration separately; refresh preserves its URL and expects the configured model ID to already exist.

## Build the llama.cpp container

Use Docker with access to the upstream image and `docker/` as the build context:

```bash
docker build --platform linux/amd64 -t sagemaker-llamacpp-qwen38:b11347 docker/
```

Push the image to ECR and set its full URI in `deployment_config.json`. Inference requires a compatible NVIDIA GPU, CUDA runtime, and model files.

The adapter listens on port `8080`, serves `/ping` and `/health`, and forwards `/invocations` and OpenAI-compatible requests to the internal llama-server, including SSE streaming responses.

## Operations

Configure the region and resource constants in each helper script for your deployment; the default region is `us-west-2`.

```bash
# Poll endpoint status.
python3 src/check_endpoint_status.py --endpoint NAME

# Fetch CloudWatch container logs.
python3 src/fetch_logs.py

# Attach the CloudWatch logging inline policy to the execution role.
python3 src/setup_cloudwatch_logs.py
```

## Run tests

```bash
python3 -m pytest -q
```

Or use unittest with explicit import paths:

```bash
PYTHONPATH=src:docker:tests python3 -m unittest discover -s tests -v
```

The Pi runtime integration test requires Node.js and an installed Pi package containing `dist/core/model-runtime.js`. Set `PI_TEST_INSTALL` to the package directory:

```bash
export PI_TEST_INSTALL="$(npm root -g)/@earendil-works/pi-coding-agent"
python3 -m pytest -q tests/test_adapter.py tests/test_pi_q6.py::PiQ6Tests::test_running_pi_runtime_refresh_and_sse
```

For the streaming inference example, configure `REGION` and `ENDPOINT_NAME` in `tests/manual/test_streaming.py` for an FP8 endpoint and supply AWS credentials:

```bash
python3 -m pip install openai
python3 tests/manual/test_streaming.py
```
