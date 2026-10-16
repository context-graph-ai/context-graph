# Download a development build

Development build, not a stable release. The first development build is published by the first run of the Build development binaries workflow. Until that hosted run succeeds, the download URLs are preparation instructions. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

## Select your architecture and download

| `uname -s` | `uname -m` | Target |
|---|---|---|
| Linux | x86_64 | x86_64-unknown-linux-gnu |
| Darwin | arm64 | aarch64-apple-darwin |

<!-- concept -->
Choose a new empty directory for the archive and extracted files. Have curl or wget, tar, python3 and a SHA-256 tool available.

```sh
set -eu
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) target=x86_64-unknown-linux-gnu ;;
  Darwin/arm64) target=aarch64-apple-darwin ;;
  *) echo "No development binary for this platform" >&2; exit 1 ;;
esac
base=https://github.com/context-graph-ai/context-graph/releases/download/dev-latest
curl -fLO "$base/cg-$target.tar.gz"
curl -fLO "$base/cg-$target.tar.gz.sha256"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c "cg-$target.tar.gz.sha256"
else
  shasum -a 256 -c "cg-$target.tar.gz.sha256"
fi
curl -fLO "$base/cg-$target.json"
python3 - "$target" <<'PY_RECEIPT'
import hashlib, json, pathlib, sys
stem = 'cg-' + sys.argv[1]
digest = hashlib.sha256(pathlib.Path(stem + '.tar.gz').read_bytes()).hexdigest()
receipt = json.loads(pathlib.Path(stem + '.json').read_text())
if receipt['sha256'] != digest:
    raise SystemExit('Receipt digest mismatch; download a complete matching asset set again')
print('Verified source commit:', receipt['context_graph_sha'])
PY_RECEIPT
tar -xzf "cg-$target.tar.gz"
./cg --version
export PATH="$PWD:$PATH"
cg model fetch --defaults
```

<!-- concept -->
For wget, replace the three curl lines above with:

```sh
wget "$base/cg-$target.tar.gz"
wget "$base/cg-$target.tar.gz.sha256"
wget "$base/cg-$target.json"
```

Stable URL pattern: `https://github.com/context-graph-ai/context-graph/releases/download/dev-latest/cg-<triple>.tar.gz` and `cg-<triple>.tar.gz.sha256` at the same base URL. Save the archive, checksum and receipt together before a later rolling update replaces them. Assets are replaced per platform in archive, checksum, receipt order. An interrupted replacement can leave that platform’s new archive with its previous checksum; checksum verification rejects that mismatch. The next successful run repairs the asset set. If a checksum or receipt check fails, wait for the running build to finish and download all three again, or use the newest complete run-named set listed in the release notes. Run-named checksum files name their matching run-named archive; compare that archive digest with the matching receipt before trusting its source commit. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

<!-- concept -->
Follow [first import and search](../README.md#first-import-and-search) after fetching the default model. For other architectures, use the [source fallback](../CONTRIBUTING.md#build-from-source).

The default Nomic model download is 546,938,168 bytes (about 547 MB). <!-- enforced by: `first_touch_lazy_fetch_observer_event_payload_carries_resolved_source_and_byte_budget` -->

Fallback names have the form `cg-<triple>-<sha12>-r<run_id>a<run_attempt>.tar.gz`, with matching `.tar.gz.sha256` and `.json` files. Download all three from one run group into a new directory. Verify its checksum, then compare the archive digest with that receipt before extracting. Every target in the run group is read back and verified before any stable file changes; existing run names refuse publication. Incomplete and unknown groups are never pruned. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

## Check identity and provenance

`dev-latest` is the single rolling release, with write-once run-named archive, checksum and receipt copies retained for the newest three complete publications plus the stable-receipt group. The tag commit is the bootstrap point on the default branch and is never moved; it is not the binary source. The receipt names the exact binary source commit, workflow run and run attempt. Keep the verified receipt to identify the downloaded commit. The binary version alone is insufficient to distinguish development commits. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

```sh
cat "cg-$target.json"
gh attestation verify "cg-$target.tar.gz" --repo context-graph-ai/context-graph
```

<!-- concept -->
For provenance verification, install and authenticate GitHub CLI (`gh`). Compare the archive digest with both the checksum and receipt, and compare `context_graph_sha` with the commit you intended to download.

| Receipt field | Meaning |
|---|---|
| `context_graph_sha` | Exact Context Graph source commit |
| `run_id` | Workflow run identity |
| `run_attempt` | Attempt within that workflow run |
| `contextdb_rev` | Exact ContextDB revision from Cargo.lock |
| `rustc_vV` | Full `rustc -vV` output |
| `target` | Binary target triple |
| `runner_image` | Runner OS, architecture, image and image version |
| `version` | Cargo package version, also printed by `cg --version` |
| `sha256` | Archive SHA-256 digest |

The build workflow is called after PR checks in the CI run of each push to this repository's dev branch, or by a verified maintainer dispatch on dev. Automatic delivery works whichever branch is the repository default branch. Publisher runs are serialized in one concurrency group with `queue: max`. It checks both native platforms and smokes each unpacked archive before publication. <!-- enforced by: `ci_runs_cargo_deny_and_cargo_expand_symbol_check` -->

<!-- concept -->
GitHub documents that queue as holding at most 100 pending runs and cancels arrivals beyond that bound.
