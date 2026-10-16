# Contributing

## Build from source

Have Git and rustup available. Build from source to install the CLI.

```sh
git clone https://github.com/context-graph-ai/context-graph.git
cd context-graph
cargo install --locked --path crates/cg-cli
cg model fetch --defaults
```

Follow the [first import and search](README.md#first-import-and-search) example after installation.

## How to contribute

Run `bash scripts/gate.sh` before submitting a change. Use Git and the Rust toolchain selected by `rust-toolchain.toml`.

Build individual targets from the repository root:

```sh
cargo build --workspace
cargo install --locked --path crates/cg-cli
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --doc --no-fail-fast
```

### Review documentation

Read [AGENTS.md](AGENTS.md) and the owning crate's guide before changing behavior. Every behavioral sentence in the README, crate READMEs, documentation and skills must carry an inline `<!-- enforced by: test_name -->` naming a live default-tier test that exercises the claim. Bind to the assertion, not a similar test title. Pure explanation needs no tag. Update help and public contracts together.

```sh
python3 scripts/validate-doc-coverage.py --all
bash scripts/check-doc-validator-negative-control.sh
bash scripts/generate-agent-bundle.sh
```

The generated `llms-full.txt` is ignored; do not commit it. Use synthetic fixtures and sanitized output. Never include private history, tokens or credentials in issues or PRs. Fork PR checks must not depend on repository secrets.

## Contributor reference

### Gate checks

| Step | What it checks |
|---|---|
| Formatting | Rust formatting matches rustfmt |
| Clippy | Workspace and all-target warnings are errors |
| Workspace tests | Behavioral proofs and deterministic test-estate audits |
| Doctests | Compiles documentation examples |
| JUnit duration report | Reports binary, measured seconds and ceiling for overruns |
| Historical baseline comparison | Rejects increases to recorded baseline values |
| Documentation coverage | Resolves claim tags and rejects unbound claims in changed pages; audits all pages without a usable base |
| Negative control | Proves the validator rejects a dead reference and an unbound claim |

Without cargo-nextest, the script runs workspace tests but skips the two checks requiring JUnit. Measured test duration must stay within recorded limits; missing measurements or baseline entries fail. Sleep/raw-clock call-site counts may only decrease; per-crate default-tier test inventory must match the recorded baseline exactly. Update inventory in the same change that adds or deliberately removes tests. Do not assert elapsed time or use sleep as synchronization.

### Live smokes

These contributor scripts use hosted models and can spend money. They exercise additional implementation paths beyond the presented commands. Read each script's required environment first; choose disposable data roots and provide keys locally, never in a PR.

The capture smoke requires an explicit JSONL input file before any other check and accepts installed release binaries:

```sh
CG_SMOKE_REAL_JSONL=/absolute/path/to/session.jsonl CG_SMOKE_SKIP_BUILD=1 CG=/absolute/path/to/cg CONTEXTDB=/absolute/path/to/contextdb bash scripts/cg-capture-live-smoke.sh
```

Supply `OPENROUTER_API_KEY` through your environment. Authenticated sync qualification additionally requires an isolated hub ticket and tenant; without them the capture smoke does not establish release readiness. The other scripts are `scripts/llm-byok-live-smoke.sh` (hosted and local compatible endpoints) and `scripts/cg-deterministic-llm-live-smoke.sh` (hosted extraction and an installed provider CLI). Consult their usage blocks for endpoint, credential-variable and binary settings before invoking them.
