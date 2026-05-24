#!/usr/bin/env bash
set -euo pipefail

expanded_source="$(mktemp)"
cargo_tree_output="$(mktemp)"
trap 'rm -f "$expanded_source" "$cargo_tree_output"' EXIT

cargo expand --lib -p context-graph >"$expanded_source"
if rg -ni 'embeddinggemma|candle_embeddinggemma' "$expanded_source"; then
  echo "EmbeddingGemma runtime symbol leaked into cargo-expanded context-graph library source" >&2
  exit 1
fi

cargo tree -p context-graph --target all -e normal,build >"$cargo_tree_output"

banned_crates=(
  candle-embeddinggemma
  pyo3
  cpython
  tch
  ort
  tensorflow
  tflite
  onnxruntime
  libtorch
  reqwest
  openai
  anthropic-sdk
  anthropic
  cohere
  openssl-sys
  mkl-sys
  cuda
  cudnn
  tensorrt
)

resolved_crates="$(sed -E 's/^[^[:alnum:]]*//' "$cargo_tree_output" | awk '{print tolower($1)}')"
for crate_name in "${banned_crates[@]}"; do
  if grep -Fxq "$crate_name" <<<"$resolved_crates"; then
    echo "Banned crate resolved in context-graph normal/build dependency tree: $crate_name" >&2
    cat "$cargo_tree_output" >&2
    exit 1
  fi
done
