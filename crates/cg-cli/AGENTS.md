# AGENTS.md — `cg-cli` crate (the `cg` binary)

The operator-facing surface: argument parsing, help text, output rendering, and the one mapping
from a typed library condition to an exit code. Package `context-graph-cli`; binary `cg`
(`src/main.rs`); the library target `cg_cli` (`src/lib.rs`) is a re-export shelf so tests and
sibling crates reach the same names the binary uses. Carries no product intent — what
context-graph is for lives in the root product documents.

## What this crate owns, and must not own

Owns: every command's arguments, help text and usage refusal; the two output formats (`--format
json|text`) and the exact envelope each answer is rendered in; the mapping from a typed condition
to an exit code (`map_store_open_err`, `memory_error`, `failure`, `usage_error`); the declared
environment inventory.

Must not own: behavior. A rule that decides what a read costs, what a total counts, whether a
model may be used, or what a store does belongs in `context-graph` or `cg-catalog` and reaches
here already typed. Must not own a second vocabulary for a condition another command already has
one for, and must not re-derive the model-cache root: it asks the layer below for that one. The
data root is the exception, and it is this binary's own: `data_root` in `src/main.rs` resolves
`CG_DATA_ROOT`, else `$HOME/.cg`, and passes the resolved path down, so nothing below re-derives
it either — `cg-catalog` takes the root it is handed and resolves none of its own, which
`the_catalog_never_resolves_the_data_root_for_itself` in
`crates/cg-catalog/tests/production_guards.rs` keeps true.

## Seams

- **`main` dispatch** (`src/main.rs`) — one `handle_*` function per command. A new command is a new
  `handle_*` plus its help entry, never a second parser.
- **Condition to exit code** (`map_store_open_err`, `memory_error` in `src/main.rs`) — matched by
  typed variant, never by substring-matching an engine `Debug` dump. A store another process is
  writing is `StoreBusy` and exit 4 on every command that opens the store. The numeric code for a
  reader-held store is deliberately unsettled: do not assign one.
- **`failure` / `usage_error`** (`src/main.rs`) — the two refusal renderers. Every refusal goes
  through one of them so it answers in the format the caller asked for.
- **Locations** (`data_root`, `home_dir` in `src/main.rs`) — the data root is resolved here and
  passed down. The model-cache root is never resolved here: `cg_catalog::default_cache_root`
  delegates to `context_graph::embedder::nomic::default_cache_root`, and that is the whole rule.
- **Re-export shelf** (`src/lib.rs`) — public names from `cg-catalog` and `cg-cli-conventions`.
  Nothing is defined here; a new behavior does not become CLI-owned by being re-exported.

## Invariants and their guarding tests

- **The environment surface is exactly what the binary declares.** `CG_DATA_ROOT` and `HOME` are
  locations; the three development switches are named in the inventory comment on `data_root`;
  nothing in the environment decides whether something runs or what a read may spend. Adding an
  `env::var` fails the test until the comment is updated. Guards:
  `the_binary_reads_exactly_the_environment_it_declares`,
  `the_inventory_comment_names_every_variable_the_binary_reads`,
  `the_owner_plane_location_is_read_once_in_the_layer_that_owns_it`,
  `the_fault_injection_read_is_compiled_out_of_every_build_that_is_not_a_test_build`, in
  `tests/every_environment_read_is_in_the_declared_inventory.rs`.
- **A store another process holds is told to the operator, never reported as empty.** Guards:
  `tests/a_busy_store_is_told_to_the_operator.rs`,
  `tests/cg_status_never_reports_a_refused_store_as_empty.rs`,
  `tests/a_reader_the_store_could_not_name_is_still_a_holder.rs`.
- **A failed read is never rendered as an answer.** A refused status read surfaces the condition;
  a store that really holds nothing still answers with its absences. Guards:
  `a_refused_status_read_is_reported_as_a_refusal_and_never_as_an_answer`,
  `a_store_that_really_holds_nothing_still_answers_with_its_absences`,
  `every_refusable_status_read_is_reached_by_the_fixture`, in
  `tests/a_failed_status_read_is_never_rendered_as_an_answer.rs`.
- **A refusal answers in the format the caller asked for.** Guard:
  `tests/a_refusal_answers_in_the_format_the_caller_asked_for.rs`.
- **Help agrees with the shipped documentation.** Every command `crates/context-graph/README.md`
  documents is named in `cg --help`, and `cg status` help names the memory store beside the model
  connection. Guards: `tests/help_names_every_readme_documented_command.rs`,
  `tests/status_help_says_it_reads_memory_too.rs`.
- **`cg model fetch` finds every registration `cg model show` reads,** including when the operator
  points `CG_DATA_ROOT` and `CG_MODEL_CACHE_ROOT` at different directories. Guard:
  `tests/cg_model_fetch_finds_a_registration_that_cg_model_show_reads_when_data_and_cache_roots_differ.rs`.
- **A stored credential never reaches output.** Guard: `tests/cli_model_byok.rs`.
- **`cg status` stays bounded.** Guard: `tests/status_is_bounded.rs`.
- **The fault-injection door is compiled out of every shipped build.** It is enabled only by this
  package's dev-dependency on itself (`Cargo.toml`), so `cargo build` never carries it. Guard:
  `the_fault_injection_feature_is_turned_on_by_nothing_but_this_package_testing_itself` in
  `tests/every_environment_read_is_in_the_declared_inventory.rs`.

## Where a change lives

| Change | Lands in |
|---|---|
| A new command, a changed message, changed help text | `src/main.rs`, in that command's `handle_*` function and its help entry |
| A changed exit code, or a newly mapped condition | `map_store_open_err` / `memory_error` in `src/main.rs` — matched by typed variant, never by string |
| A refusal's rendering, in either format | `failure` / `usage_error` in `src/main.rs` |
| A new environment read | `src/main.rs` plus the inventory comment on `data_root` AND `DECLARED_INVENTORY` in `tests/every_environment_read_is_in_the_declared_inventory.rs` |
| Where registrations or artifact bytes live | `crates/cg-catalog/src/lib.rs`; the CLI passes the data root it resolved and never resolves the model-cache root itself |
| What a status total counts, or what a read may spend | `crates/context-graph/src/status_reads.rs` and `owner_control::context_graph_read_limits`, not here |
| The shape a check, finding, severity or report is rendered in | `crates/cg-cli-conventions/` |
| The operator-facing prose a help entry must agree with | `crates/context-graph/README.md`, in the same change |

## Fast test command

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`. The repository gate that adds the wall-clock and documentation-claim
checks is `scripts/gate.sh`; the root `AGENTS.md` describes both.
