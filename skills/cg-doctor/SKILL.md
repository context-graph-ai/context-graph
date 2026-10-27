---
name: cg-doctor
description: Check provider setup.
---

# Check provider setup

## Prerequisite

Install `cg`.

Install the provider CLI you intend to check.

## Recipe

1. Run `cg doctor`.

2. Read the findings and fix hints.

3. Run `cg doctor --json` for structured findings.

## Installed-binary example

Install `cg` and Python 3; use a POSIX shell.

Inspect the available options before checking a provider.

<!-- smoke: skill -->
```sh
cg_example=$(mktemp -d)
export HOME="$cg_example/home"
export CG_DATA_ROOT="$cg_example/data"
mkdir -p "$HOME"
cg doctor --help
```

### Example output

```text
Usage: cg doctor [--json]

Examples:
  cg doctor
  cg doctor --json
```

## Failure to handle

Doctor exits 1 on provider flag drift. <!-- enforced by: `cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing` -->

Doctor makes no completion call. <!-- enforced by: `cg_doctor_runs_only_free_checks_names_drift_and_spends_nothing` -->

Apply the named fix before repeating the check.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
