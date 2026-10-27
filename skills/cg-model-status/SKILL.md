---
name: cg-model-status
description: Inspect the selected model connection.
---

# Inspect a model connection

## Prerequisite

Install `cg`.

Choose the data root used for the connection.

## Recipe

1. Run `cg model status`.

2. Inspect the connection and spend fields.

3. Run `cg model status --format text` for readable output.

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
mkdir -p "$HOME"
export CG_DATA_ROOT="$cg_example/data"
cg model status --format text || test "$?" -eq 4
```

### Example output

```text
no model connection is active; run cg model auth login or cg model connect.
context: default goals: 0 decisions: 0 observations: 0 | pending understanding: 0 | raw kept: 0 chunks / 0 bytes
```

## Failure to handle

Disconnected model status exits 4. <!-- enforced by: `cli_status_without_connection_says_so_and_points_at_connect` -->

Its JSON guidance names `cg model auth login`. <!-- enforced by: `cli_status_without_connection_says_so_and_points_at_connect` -->

Set up the intended model connection before retrying.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
