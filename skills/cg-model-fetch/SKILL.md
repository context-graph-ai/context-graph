---
name: cg-model-fetch
description: Prepare the local search model.
---

# Prepare the local search model

## Prerequisite

Install `cg`.

Allow network access for an uncached default model download.

## Recipe

1. Run `cg model fetch --defaults`.

2. Inspect the fetched and skipped entries.

3. Run `cg model show nomic-embed-text-v1.5`.

Default prefetch fetches Nomic and skips EmbeddingGemma for manual acceptance. <!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only` -->

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
mkdir -p "$HOME"
CG_DATA_ROOT="$cg_example/data" cg model fetch --defaults --format json
```

### Example output

```text
{"disclosed":[{"bytes":546938168,"id":"nomic-embed-text-v1.5","size":"547 MB"}],"fetched":[{"id":"nomic-embed-text-v1.5","sha":"9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c"}],"skipped":[{"id":"embeddinggemma-300m","reason":"ManualAcceptanceRequired"}]}
```

## Failure to handle

A partial Nomic cache returns `CacheIncomplete` before an outbound fetch. <!-- enforced by: `air_gap_with_partial_cache_and_no_network_returns_typed_artifact_missing_error_does_not_silently_attempt_fetch` -->

Check the cache before retrying; see [configuration](../../docs/configuration.md).

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
