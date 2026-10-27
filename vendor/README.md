# Candle archive compatibility backport

These directories contain the official published Candle 0.10.2 core, NN, and
transformers archives. The crate licenses and `.cargo_vcs_info.json` files are
preserved. All three packages identify upstream commit
`7c7a8c570e6a16ea24cc30a7501c8ffbbdb51680` in
[Hugging Face Candle](https://github.com/huggingface/candle).

| Published archive | SHA-256 (verified against crates.io version metadata) |
| --- | --- |
| [candle-core 0.10.2](https://static.crates.io/crates/candle-core/candle-core-0.10.2.crate) | `6bd9895436c1ba5dc1037a19935d084b838db066ff4e15ef7dded020b7c12a4a` |
| [candle-nn 0.10.2](https://static.crates.io/crates/candle-nn/candle-nn-0.10.2.crate) | `a9317a09d6530b758990ed7f625ac69ff43653bc9ee28b0464644ad1169ada87` |
| [candle-transformers 0.10.2](https://static.crates.io/crates/candle-transformers/candle-transformers-0.10.2.crate) | `f59d08c89e9f4af9c464e2f3a8e16199e7cc601e6f34538c2cfbb42b623b1783` |

[CANDLE-SOURCES.json](CANDLE-SOURCES.json) records exact distribution and metadata
URLs, checksums, declared `MIT OR Apache-2.0` licenses, and source identities.
The core archive's [`LICENSE`](candle-core/LICENSE) contains Apache-2.0 text and
remains unchanged; [LICENSE-CANDLE](LICENSE-CANDLE) copies it for this cohort.
The published NN and transformers archives contain no standalone license file;
their `MIT OR Apache-2.0` declarations and all published notices are preserved.

Only these published files differ:

- `candle-core/Cargo.toml`: require exactly zip 9.0.4 with default features
  disabled, retaining the previous empty zip feature set.
- `candle-core/src/npy.rs`: propagate `ZipFile::name()` errors and borrow its
  returned `Cow<str>` while stripping `.npy` in eager and lazy reads.
- `candle-core/src/pickle.rs`: collect fallible archive filenames before reading
  PTH tensor metadata, propagating the existing typed ZIP error.
- `candle-nn/Cargo.toml`: direct the dependency named `candle` (package
  `candle-core`) to the sibling core directory.
- `candle-transformers/Cargo.toml`: direct its `candle` alias and `candle-nn`
  dependency to the same sibling cohort.

The filename adapter is the narrow 0.10.2 backport of
[upstream compatibility change 4041](https://github.com/huggingface/candle/pull/4041),
reference commit `70f72e9cfdbfc570846ca363ead2135394af3e9a`. This reference is an
unmerged contribution, not the source of the vendored packages. Tensor, model,
math, device, and runtime code is unchanged; NN and transformers Rust files are
byte-identical to their published archives. `Cargo.toml.orig` and upstream
`Cargo.lock` files are retained as provenance; Cargo uses the patched standalone
`Cargo.toml` manifests and this repository's root lockfile.

The root workspace uses direct dependency paths and excludes all three packages
from membership. The paths also apply to Git/path library consumers; they do not
depend on an application inheriting a root `[patch]` table. A separate dependency
on registry Candle has a distinct Rust type identity. Context Graph's public
embedding interfaces exchange project types and vectors, not raw Candle tensors.
Registry publication of these path dependencies requires a compatible published
upstream release; the workspace packages currently have `publish = false`.

Archive regressions run in the existing `context-graph` unit-test binary under
its default `nomic-embedder` feature. The PTH test reuses the byte-identical
1165-byte `candle-core/tests/test.pt` fixture shipped in the verified core archive;
no model or Python/PyTorch execution is needed. NPZ fixtures are generated through
Candle's writer during the tests. zip 9.0.4 accepts arbitrary filename bytes as
UTF-8 or CP437; tests exercise structurally truncated filename metadata for typed
ZIP errors rather than claiming non-UTF-8 alone is invalid.

Remove this cohort and its direct path edges when adopting an official Candle
release compatible with a fixed zip version. Before removal, verify a single
Candle cohort for downstream consumers, archive regression coverage, the strict
dependency policy, and existing embedding behavior with unchanged model fixtures.
