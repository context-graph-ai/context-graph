<!-- concept -->
Kind: `<official | community>` · Status: `<NOT EVIDENCE | descriptive only | acceptance rows reported below>`

# Performance result: <target> on <hardware label>, <run date>

<!-- concept -->
Copy this file to `docs/performance/results/<date>-<target>-<short-candidate-sha>.md`
and replace every `<…>` from reviewed values only. See
[the guide](../README.md#publish-a-reviewed-summary).

Rules for filling it in:

<!-- concept -->
- **Public only.** Never paste raw receipts, logs, absolute paths, hostnames,
  user names or secrets. Raw material stays private.
- **Never invent a value.** If a field was not recorded, write `unknown`; if it
  does not apply, write `not applicable` and say why.
- **Descriptive is not acceptance.** Numbers go in the descriptive table, which
  has no verdict. The acceptance table lists only a cited context-graph product
  requirement with its number, statistic and exact scope. A benchmark
  assertion or a completed run is never an acceptance verdict.
- **Missing proof is not a pass.** An acceptance row that was not run is
  `NOT RUN`; one whose run does not match the row's scope, or whose provenance
  is partly unknown, is `NOT EVIDENCE`.

## Classification

<!-- concept -->
- Kind: `<official | community>`
- Measured by: `<public name or handle>`; approved for publication by: `<maintainer>`
- Run date: `<YYYY-MM-DD>`
- Run status: `<completed | failed | incomplete>`

## Source

<!-- concept -->
- Baseline: `<full context-graph SHA>`; locked ContextDB: `<full SHA>`
- Candidate: `<full context-graph SHA>`; locked ContextDB: `<full SHA>`
- Benchmark target and selectors: `<target, selectors>`

## Workload and configuration

<!-- concept -->
- Fixture or corpus identity: `<name and digest>`; size: `<events, passages, Contexts>`
- Model identity: `<model and digest, or not applicable>`
- Build: `<profile, toolchain, flags>`
- Exact command (local paths replaced by placeholders): `<command>`

## Environment

<!-- concept -->
- Hardware (public label): `<CPU, cores, memory>`
- Filesystem and storage (public): `<type and configuration>`
- OS / kernel: `<…>`
- Resource limits: `<…>`
- Isolation: `<what else was stopped during the runs>`

## Acceptance (cited product requirements only)

<!-- concept -->
If this entry reports on none, write "None reported." and leave the table out.

| Product requirement (cited, with exact scope) | Verdict | Evidence and coverage |
|---|---|---|
| `<requirement text and its source>` | `<PASS / FAIL / NOT RUN / NOT EVIDENCE>` | `<what was measured, statistic and unit, what was not covered>` |

## Descriptive measurements (no verdict)

| Target / selector | Statistic | Unit | Baseline | Candidate | Change |
|---|---|---|---|---|---|
| `<…>` | `<…>` | `<…>` | `<value>` | `<value>` | `<value>` |

## Missing metadata

<!-- concept -->
- [ ] Baseline revision
- [ ] Candidate revision
- [ ] Run date
- [ ] Fixture, corpus or model identity
- [ ] Hardware, OS or toolchain
- [ ] Resource limits or isolation
- [ ] Exact command

Provenance: `<complete | partly unknown: list above>`

## Limitations

- `<anything that could affect the comparison>`
