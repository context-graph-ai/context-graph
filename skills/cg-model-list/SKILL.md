---
name: cg-model-list
description: List registered model entries.
---

# List registered models

## Prerequisite

Install `cg`.

## Recipe

1. Run `cg model list`.

2. Choose a returned model id.

3. Run `cg model show <id>`.

Model list includes registered local and connected model entries in default JSON. <!-- enforced by: `cli_model_list` -->

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
mkdir -p "$HOME"
CG_DATA_ROOT="$cg_example/data" cg model list --format json
```

### Example output

```text
[]
```

## Failure to handle

If the list is empty, run `cg model fetch --defaults`.

Then run `cg model list` again.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
