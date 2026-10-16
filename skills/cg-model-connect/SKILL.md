---
name: cg-model-connect
description: Configure a model endpoint.
---

# Connect a model endpoint

## Prerequisite

Install `cg`.

Choose an endpoint, provider model id and API-key variable.

## Recipe

1. Set the chosen API-key environment variable in your shell.

2. Run the connection command below with your chosen values.

3. Inspect the connection with `cg model status`.

Connecting with `--consent` records consent without an endpoint request. <!-- enforced by: `cli_connect_registers_named_connection_with_chosen_provider_and_own_url` -->

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

Check disconnected status before supplying credentials.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
export CG_DATA_ROOT="$cg_example/data"
mkdir -p "$HOME"
cg model status --format text || test "$?" -eq 4
```

### Example output

```text
no model connection is active; run cg model auth login or cg model connect.
context: default goals: 0 decisions: 0 observations: 0 | pending understanding: 0 | raw kept: 0 chunks / 0 bytes
```

## Credential-dependent command

Supply your chosen credentials before running this command.

<!-- smoke: needs-connected-model -->
```text
cg model connect <name> --provider openrouter --endpoint https://openrouter.ai/api/v1 --env-key <VAR> --model <model> --consent
```

## Failure to handle

Disconnected model status exits 4. <!-- enforced by: `cli_status_without_connection_says_so_and_points_at_connect` -->

Its JSON guidance names `cg model auth login`. <!-- enforced by: `cli_status_without_connection_says_so_and_points_at_connect` -->

Check your endpoint and model id before connecting.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
