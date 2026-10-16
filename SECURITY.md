# Security

Report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/context-graph-ai/context-graph/security/advisories/new). Include the affected revision, reproduction steps and likely impact. Do not put secrets or conversation excerpts in public issues.

The store contains raw conversation text in accepted events under the data root (`~/.cg` by default, or `CG_DATA_ROOT`). Treat that directory, copies and backups as sensitive. Hosted understanding sends selected event-derived input to the chosen provider; consult [privacy and data movement](docs/privacy-and-data-movement.md) before connecting one.

Credentials stay off model entries: connections hold credential references, and list/show/status output excludes secrets. Protect the credential store and environment separately from model artifact caches.
