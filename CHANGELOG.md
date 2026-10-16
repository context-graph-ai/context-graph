# Changelog

## Unreleased

- **Added.** Local Claude Code history capture, stable event identities and resumable checkpoints.
- **Added.** Searchable passages with the local Nomic text embedder; find, source expansion, status and history deletion.
- **Added.** A Rust memory library for intentions, decisions, entities, observations, evidence and digests.
- **Added.** Embedding-space registration and a bounded read-only consumer surface.
- **Added.** Model artifacts, hosted connection metadata and credential references.
- **Added.** Contributor checks, documentation claim bindings and agent recipes.

- **Limitation.** The standalone journey is single-machine Claude Code history search.
- **Limitation.** History import creates events and passages, not structured graph rows.
- **Limitation.** Capture can finish before search is ready; search coverage defaults to ten recent sessions.
- **Limitation.** Invalidated decisions cannot return to active status.
