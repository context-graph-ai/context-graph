# AGENTS.md — `cg-connector-claude-code` crate

The Claude Code connector: it finds a project's Claude Code history on this machine
(`src/history.rs`), decodes each JSONL record into the canonical Claude occurrence
(`src/jsonl.rs`, `src/occurrence.rs`) and assembles the ordered `EventDraft` group the generic
event core stores (`src/assembler.rs`). Carries no product intent — what context-graph is for lives
in the root product documents.

## What this crate owns, and must not own

Owns: Claude-specific field navigation, the Claude project-store key, the connector id, adapter
and anchor-declaration versions (`src/declaration.rs`), and the mapping from a Claude record to
provider-neutral render roles and kinds.

Must not own: event identity, sequence, hashing, checkpoints, bindings, projection or search. Those
belong to `context-graph`, which never depends on this crate; a future Codex or other connector uses
the same generic event tables and APIs.

## Rules for maintenance

- Provider mapping and declaration versions stay in this crate. A changed mapping is a new adapter
  or anchor-declaration version, never an edit to events already stored.
- Fixtures are independently authored and synthetic. No owner transcript, machine path or private
  label becomes a fixture, default or constant.
- A record's retained `source_remainder` is data, never code vocabulary: nothing read from captured
  content may create a new mapping, kind or repair rule.
- Historical JSONL and a future live front end both decode to the same canonical occurrence and call
  the one assembler; neither builds `EventDraft` values itself.
- The CLI (`crates/cg-cli/src/history/`) opens files and calls this crate; it navigates no Claude
  JSON of its own.

## Fast test command

`cargo test -p cg-connector-claude-code` and
`cargo clippy -p cg-connector-claude-code --all-targets -- -D warnings`.
