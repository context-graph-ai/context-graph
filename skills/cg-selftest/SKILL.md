---
name: cg-selftest
description: Test a model connection.
---

# Test a model connection

## Prerequisite

Install `cg`.

Connect a model first; review and record spend consent.

## Recipe

1. Run `cg model status` and check the selected connection.

2. Run `cg selftest --format json` to test that connection.

3. Inspect the reply, pass status and failure bucket.

Selftest with a consented connection spends one live completion call. <!-- enforced by: `selftest_spends_exactly_one_live_call` -->

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
cg selftest --format json
```

## Failure to handle

Missing consent refuses selftest before any HTTP request. <!-- enforced by: `cli_selftest_without_consent_is_a_distinct_consent_refused_condition_and_zero_requests` -->

Review the selected connection before recording consent.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
