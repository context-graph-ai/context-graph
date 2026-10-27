---
name: cg-model-show
description: Inspect a registered model.
---

# Inspect one model

## Prerequisite

Install `cg`.

Choose an id from `cg model list`.

## Recipe

1. Run `cg model show <id>`.

2. Inspect the kind, connection and license fields.

3. Follow any manual installation instructions before fetching.

Model show reports the model kind, connection and license. <!-- enforced by: `cli_model_show` -->

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
mkdir -p "$HOME"
export CG_DATA_ROOT="$cg_example/data"
cg model fetch --defaults > "$cg_example/fetch.json"
cg model show nomic-embed-text-v1.5 --format json
```

### Example output

```text
{"connection":null,"credential_source":null,"endpoint":null,"id":"nomic-embed-text-v1.5","kind":"TextEmbedder","license":{"install_instructions":null,"manual_install_url":null,"requires_manual_acceptance":false,"spdx":"Apache-2.0","url":"https://www.apache.org/licenses/LICENSE-2.0"},"model":"nomic-embed-text-v1.5","provider":null}
```

## Failure to handle

A license-gated entry displays its manual installation instructions. <!-- enforced by: `cli_model_show_license_gated_renders_manual_instructions` -->

Showing that metadata changes no installation files. <!-- enforced by: `cli_model_show_license_gated_renders_manual_instructions` -->

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
