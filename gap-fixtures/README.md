# Gap-report adversarial fixtures

Each fixture is a pair: `<name>.txt` (the transcript chunk fed to extraction, verbatim) and
`<name>.expected.yaml` (what a faithful extraction must and must not contain). These four classes
come from the seed's decision-gate spec: reversal-within-chunk, pure-tool-noise (empty IS correct),
buried-signal, stated-wrong-type.

Scoring (per model, per fixture, per category — intentions/decisions/observations/entities):
- `must_contain`: fuzzy match (case-insensitive substring or ≥0.6 token overlap) against extracted
  item descriptions/names. Each hit = 1, miss = 0.
- `must_not_contain`: any match = a faithfulness violation (counts negatively).
- `allow_empty: true` fixtures: score 1.0 iff ALL categories empty (the laziest-valid-completion
  trap — schema-valid empty output is CORRECT here and only here).
- Entity typing scored separately: `entities[].type` must equal expected type (the probe's known
  weak spot).
- Grounding: every extracted item's span must appear verbatim in the chunk (the deterministic
  grounding check already enforces this — record the flag rate per model).

The report aggregates per-category faithfulness for local vs hosted on: these fixtures + N real
session chunks (hosted output = reference baseline for reals; agreement scored the same fuzzy way).
