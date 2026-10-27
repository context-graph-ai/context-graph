# Synthetic Claude source oracle

These files are independently authored synthetic inputs for source and adapter fidelity. They do
not copy, transform, name, or locate any private transcript. `expected-occurrences.json` is a
source-fact oracle, not a serialized public event schema: it states every accepted occurrence's
source fields, ordered semantic mapping, exact retained text or structured source value, and
coverage expectation. A later regression author binds those facts to the released connector contract.

`compact-rich.jsonl` is the primary valid stream. It deliberately includes accounts, a legal matter,
literal embedding-prefix-looking text, Unicode, a fence, a tab, multipart messages, linked and
unmatched tools, textual and refused-media attachments, opaque top-level rows, an opaque nested
block, nested subagent attribution, duplicate message IDs, a missing timestamp, and a bad timestamp.
`second-stream-same-session.jsonl` is a distinct source instance with the same session identifier.

The minimal variants cover malformed newline-terminated JSON, an unterminated pending tail completed
on a subsequent source version, and whitespace-only-to-first-record progression. `generate_oversized_tail.py`
deterministically writes an oversized valid record followed by an unterminated tail; it avoids a
stochastic flood. Run `python3 verify_source_oracle.py` after generation. The verifier uses only the
standard library; it checks JSON validity, malformed and pending-tail locations, exact UTF-8 raw-text
offsets, semantic sentinels, and the fixture manifest checksums.

Unspecified on purpose: exact canonical JSON spelling after storage, normalized payload object shape,
anchor key set/version, adapter version, source-instance hash representation, store UUID/hash/sequence,
and receipt rendering. The source fields and values required to account for each accepted occurrence
are frozen here; none of those unspecified implementation details may remove or alter an accepted
source fact.
