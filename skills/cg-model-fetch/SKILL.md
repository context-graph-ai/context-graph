---
name: cg-model-fetch
description: Install the default local search model or adopt a complete verified artifact-only Nomic cache with explicit cg model fetch.
---
<!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

# Prepare the search model

## Prerequisites

Use an installed `cg` binary; Cargo fetches the pinned ContextDB git dependency when building it.
For an online fetch, allow access to the declared model source. For
cache adoption, obtain all four files of the pinned Nomic artifact from a successful
`cg model fetch --defaults` on an online host. Import itself never downloads missing search
artifacts.<!-- enforced by: `manifests_lockfile_and_readme_name_the_same_exact_contextdb_revision`, `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

## Recipe

1. On an online host, run `cg model fetch --defaults` to prepare the bundled search model.
2. To transfer an artifact-only cache, copy `model.safetensors`, `tokenizer.json`, `config.json`
   and `REVISION` into `nomic-embed-text-v1.5/` under the receiving host's `CG_MODEL_CACHE_ROOT`.
3. Run `cg model fetch nomic-embed-text-v1.5` with that cache root. Explicit fetch validates the
   complete local artifact before writing `catalog.toml`; a valid complete cache needs no source
   download during adoption.
4. Retry `cg import <project-directory> --search wait` to make captured history searchable.<!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable`, `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only`, `missing_nomic_artifact_blocks_background_and_wait_without_download_while_defer_skips_preflight` -->

The receiving layout before adoption is below. Keep the four files together from the same pinned
artifact; no hand-written catalog manifest is needed.<!-- enforced by: `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

```text
<CG_MODEL_CACHE_ROOT>/nomic-embed-text-v1.5/
  model.safetensors
  tokenizer.json
  config.json
  REVISION
```

## Reproducible cache-adoption example

Use a POSIX shell, Python 3.11+ and `cg` on `PATH`. The first fetch obtains the complete pinned
model into an isolated source cache on an online host. The copy then creates an artifact-only
receiving cache, and the second fetch adopts it. The final command prints the `model_id` from
the newly written manifest, the same field asserted by the adoption guard.<!-- enforced by: `prefetch_defaults_writes_artifacts_on_disk_for_apache_models_only`, `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->

```sh
cg_cache_example=$(mktemp -d)
CG_DATA_ROOT="$cg_cache_example/source-data" CG_MODEL_CACHE_ROOT="$cg_cache_example/source-cache" cg model fetch --defaults --format json > "$cg_cache_example/fetch.json"
mkdir -p "$cg_cache_example/receiving-cache/nomic-embed-text-v1.5"
for artifact in model.safetensors tokenizer.json config.json REVISION; do
  cp "$cg_cache_example/source-cache/nomic-embed-text-v1.5/$artifact" "$cg_cache_example/receiving-cache/nomic-embed-text-v1.5/$artifact"
done
CG_DATA_ROOT="$cg_cache_example/receiving-data" CG_MODEL_CACHE_ROOT="$cg_cache_example/receiving-cache" cg model fetch nomic-embed-text-v1.5 --format json > "$cg_cache_example/adopt.json"
python3 -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["model_id"])' "$cg_cache_example/receiving-cache/nomic-embed-text-v1.5/catalog.toml"
```

Exact final stdout:

```text
nomic-embed-text-v1.5
```

## Failure to handle

A partial or corrupt cache is refused instead of silently trusted. Inspect the reported artifact
failure; copying a manifest alongside corrupt bytes does not make those bytes valid.<!-- enforced by: `air_gap_with_partial_cache_and_no_network_returns_typed_artifact_missing_error_does_not_silently_attempt_fetch`, `air_gap_manifest_matching_corrupt_bytes_does_not_override_catalog_artifact_identity_validation`, `explicit_fetch_makes_a_complete_artifact_only_cache_usable` -->
