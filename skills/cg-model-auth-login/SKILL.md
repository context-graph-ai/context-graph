---
name: cg-model-auth-login
description: Authorize an existing provider login.
---

# Authorize a provider session

## Prerequisite

Install `cg`.

Log into the provider CLI you intend to use.

## Recipe

1. Run `cg model auth login`.

2. Review the selected account and spend notice.

3. For explicit noninteractive consent, use the command below.

4. Inspect the connection with `cg model status`.

The consent prompt names the account and asks you to type yes; after consent, the command output names the account and its subscription-quota spend. <!-- enforced by: `model_auth_login_consent_names_account_and_spend_before_anything_spends` -->

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
cg model auth login --connection claude --consent --format json
```

## Failure to handle

Noninteractive login without `--consent` leaves consent unrecorded. <!-- enforced by: `model_auth_login_non_interactive_without_consent_flag_records_no_consent` -->

Review the selected account before repeating login with `--consent`.

See [CLI reference](../../docs/cli.md) and [configuration](../../docs/configuration.md).
