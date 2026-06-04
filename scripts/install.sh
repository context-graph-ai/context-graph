#!/usr/bin/env bash
set -euo pipefail

prefix="/usr/local"
dry_run=0
prefetch_defaults=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h)
      echo "Usage: install.sh [--prefix DIR] [--dry-run] [--prefetch-defaults]"
      echo "--prefix DIR         Install cg under DIR/bin (default: /usr/local)"
      echo "--dry-run            Print completion status after staging files"
      echo "--prefetch-defaults  Download Apache-licensed default model artifacts"
      exit 0
      ;;
    --prefix)
      prefix="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --prefetch-defaults)
      prefetch_defaults=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$prefix" ]; then
  echo "missing --prefix value" >&2
  exit 2
fi

cache_root="${CG_MODEL_CACHE_ROOT:-$HOME/.cache/cg/models}"
source_override_config="$cache_root/cg/catalog-sources.toml"
nomic_id="nomic-embed-text-v1.5"
nomic_bytes="546938168"
nomic_revision="e9b6763023c676ca8431644204f50c2b100d9aab"
nomic_production_model_sha="9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c"
nomic_production_tokenizer_sha="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
nomic_production_config_sha="9ab00bd92cee80a569f708140b7b6c1661a65891ff3765b1519e181ba2f2c92b"
nomic_revision_sha="b6e8c34dc906161922aedff26e55cc6ec85162b83826c43ef604452489c407f1"

mkdir -p "$prefix/bin" "$HOME/.config/cg"

python_clean_hooks='
import json, pathlib, sys
agent = sys.argv[1]
for raw in sys.argv[2:]:
    path = pathlib.Path(raw)
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    hooks = data.get("hooks", [])
    out = []
    skipping = False
    begin = f"# BEGIN cg-injected-hook-{agent}"
    end = f"# END cg-injected-hook-{agent}"
    for hook in hooks:
        if isinstance(hook, str) and hook.startswith(begin):
            skipping = True
            continue
        if skipping:
            if isinstance(hook, str) and hook.startswith(end):
                skipping = False
            continue
        if hook == f"cg observe --agent {agent}":
            continue
        out.append(hook)
    data["hooks"] = out
    path.write_text(json.dumps(data, indent=2) + "\n")
'

inject_claude_hook() {
  local config="$HOME/agent-config/claude-code/settings.json"
  if [ ! -f "$config" ]; then
    return 0
  fi
  python3 - "$config" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
hooks = data.get("hooks", [])
if not hooks:
    sys.exit(0)
out = []
skipping = False
for hook in hooks:
    if isinstance(hook, str) and hook.startswith("# BEGIN cg-injected-hook-claude-code"):
        skipping = True
        continue
    if skipping:
        if isinstance(hook, str) and hook.startswith("# END cg-injected-hook-claude-code"):
            skipping = False
        continue
    if hook == "cg observe --agent claude-code":
        continue
    out.append(hook)
out.extend([
    "# BEGIN cg-injected-hook-claude-code-0",
    "# END cg-injected-hook-claude-code-0",
])
data["hooks"] = out
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

cg_bin="$prefix/bin/cg"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiled_cg=""
debug_cg="$repo_root/target/debug/cg"
release_cg="$repo_root/target/release/cg"
if [ -x "$release_cg" ] && { [ ! -x "$debug_cg" ] || [ "$release_cg" -nt "$debug_cg" ]; }; then
  compiled_cg="$release_cg"
elif [ -x "$debug_cg" ]; then
  compiled_cg="$debug_cg"
fi
if [ -z "$compiled_cg" ]; then
  cargo_cmd="${CARGO:-}"
  if [ -z "$cargo_cmd" ] && command -v cargo >/dev/null 2>&1; then
    cargo_cmd="$(command -v cargo)"
  fi
  if [ -z "$cargo_cmd" ] && [ -x "$HOME/.cargo/bin/cargo" ]; then
    cargo_cmd="$HOME/.cargo/bin/cargo"
  fi
  if [ -z "$cargo_cmd" ]; then
    echo "could not find cargo and no prebuilt cg binary exists under target/" >&2
    exit 1
  fi
  "$cargo_cmd" build --manifest-path "$repo_root/Cargo.toml" -p context-graph-cli --bin cg --release
  compiled_cg="$repo_root/target/release/cg"
fi
install -m 755 "$compiled_cg" "$cg_bin"

cat >"$prefix/cg-uninstall.sh" <<UNINSTALL
#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="$cache_root"
CG_BINARY="$prefix/bin/cg"
CG_HOOK_MARKER="# BEGIN cg-injected-hook-"

clean_all_hooks() {
  python3 - "\$HOME" <<'PY'
import json, pathlib, sys
home = pathlib.Path(sys.argv[1])
agents = ["claude-code", "cursor", "gemini-cli"]
paths = list(home.glob("**/agent-config/claude-code/settings.json"))
paths += [home / ".cursor/hooks.json", home / ".gemini/settings.json"]
for path in paths:
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    hooks = data.get("hooks", [])
    out = []
    skipping = None
    for hook in hooks:
        if isinstance(hook, str) and hook.startswith("# BEGIN cg-injected-hook-"):
            skipping = hook.removeprefix("# BEGIN cg-injected-hook-").rsplit("-", 1)[0]
            continue
        if skipping is not None:
            if isinstance(hook, str) and hook.startswith(f"# END cg-injected-hook-{skipping}"):
                skipping = None
            continue
        if any(hook == f"cg observe --agent {agent}" for agent in agents):
            continue
        out.append(hook)
    data["hooks"] = out
    path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

case "\${1:-}" in
  --help)
    echo "Usage: cg-uninstall.sh [--purge]"
    echo "Default uninstall removes cg binaries, model cache, user config, and preserves project .cg directories."
    echo "--purge prompts before removing preserved .cg project directories."
    exit 0
    ;;
esac

rm -f "\$CG_BINARY"
rm -rf "\$CACHE_ROOT" "\$HOME/.config/cg"
clean_all_hooks

if [ "\${1:-}" = "--purge" ]; then
  printf 'Remove all .cg directories under %s? Type yes to continue: ' "\$HOME"
  read -r answer
  if [ "\$answer" = "yes" ]; then
    find "\$HOME" -type d -name .cg -prune -exec rm -rf {} +
  else
    echo "Preserved project .cg directories."
  fi
else
  echo "Preserved project .cg directories. To inspect: find \$HOME -type d -name .cg"
fi
UNINSTALL
chmod 755 "$prefix/cg-uninstall.sh"

inject_claude_hook

download_artifact() {
  local url="$1"
  local destination="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 -o "$destination" "$url"
  else
    python3 - "$url" "$destination" <<'PY'
import pathlib, sys, urllib.request
url, destination = sys.argv[1], pathlib.Path(sys.argv[2])
with urllib.request.urlopen(url, timeout=60) as response:
    destination.write_bytes(response.read())
PY
  fi
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

validate_sha_any() {
  local path="$1"
  local label="$2"
  shift 2
  local actual
  actual="$(sha256_file "$path")"
  for expected in "$@"; do
    if [ "$actual" = "$expected" ]; then
      return 0
    fi
  done
  echo "nomic $label SHA mismatch: expected one of $*, got $actual" >&2
  return 1
}

validate_nomic_artifacts() {
  local model_dir="$1"
  local model_shas=("$nomic_production_model_sha")
  local tokenizer_shas=("$nomic_production_tokenizer_sha")
  local config_shas=("$nomic_production_config_sha")
  local override_sha
  if override_sha="$(read_source_override_sha "$nomic_id" "model.safetensors")"; then
    model_shas=("$override_sha")
  fi
  if override_sha="$(read_source_override_sha "$nomic_id" "tokenizer.json")"; then
    tokenizer_shas=("$override_sha")
  fi
  if override_sha="$(read_source_override_sha "$nomic_id" "config.json")"; then
    config_shas=("$override_sha")
  fi
  validate_sha_any "$model_dir/model.safetensors" "model.safetensors" "${model_shas[@]}"
  validate_sha_any "$model_dir/tokenizer.json" "tokenizer.json" "${tokenizer_shas[@]}"
  validate_sha_any "$model_dir/config.json" "config.json" "${config_shas[@]}"
  local revision_shas=("$nomic_revision_sha")
  local revision_sha_overridden=0
  if override_sha="$(read_source_override_sha "$nomic_id" "REVISION")"; then
    revision_shas=("$override_sha")
    revision_sha_overridden=1
  fi
  validate_sha_any "$model_dir/REVISION" "REVISION" "${revision_shas[@]}"
  local revision_text
  revision_text="$(tr -d '\r\n' <"$model_dir/REVISION")"
  if [ -z "$revision_text" ]; then
    echo "nomic REVISION is empty" >&2
    return 1
  fi
  if [ "$revision_sha_overridden" = "0" ] && [ "$revision_text" != "$revision" ]; then
    echo "nomic REVISION mismatch: expected $revision, got $revision_text" >&2
    return 1
  fi
}

write_nomic_manifest() {
  local model_dir="$1"
  local source_revision="${2:-$nomic_revision}"
  local model_sha="${3:-}"
  local tokenizer_sha="${4:-}"
  local config_sha="${5:-}"
  local revision_sha="${6:-}"
  if [ -z "$model_sha" ]; then
    model_sha="$(sha256_file "$model_dir/model.safetensors")"
  fi
  if [ -z "$tokenizer_sha" ]; then
    tokenizer_sha="$(sha256_file "$model_dir/tokenizer.json")"
  fi
  if [ -z "$config_sha" ]; then
    config_sha="$(sha256_file "$model_dir/config.json")"
  fi
  if [ -z "$revision_sha" ]; then
    revision_sha="$(sha256_file "$model_dir/REVISION")"
  fi
  cat >"$model_dir/catalog.toml" <<EOF
model_id = "$nomic_id"
kind = "TextEmbedder"
dimension = 768
input_modality = "text"
license_spdx = "Apache-2.0"
license_url = "https://www.apache.org/licenses/LICENSE-2.0"
source_repo = "nomic-ai/nomic-embed-text-v1.5"
source_revision = "$source_revision"
byte_size = $nomic_bytes
cache_status = "cached"
artifact_filenames = ["model.safetensors", "tokenizer.json", "config.json", "REVISION", "catalog.toml"]

[per_file_sha256]
model.safetensors = "$model_sha"
tokenizer.json = "$tokenizer_sha"
config.json = "$config_sha"
REVISION = "$revision_sha"
EOF
}

read_source_override() {
  local model_id="$1"
  local config="$source_override_config"
  if [ ! -f "$config" ]; then
    return 1
  fi
  python3 - "$config" "$model_id" <<'PY'
import pathlib, sys, tomllib
config = pathlib.Path(sys.argv[1])
model_id = sys.argv[2]
data = tomllib.loads(config.read_text())
source = data.get("sources", {}).get(model_id)
if not source:
    sys.exit(1)
print(f"{source['base_url']}\t{source['revision']}")
PY
}

read_source_override_sha() {
  local model_id="$1"
  local file="$2"
  local config="$source_override_config"
  if [ ! -f "$config" ]; then
    return 1
  fi
  python3 - "$config" "$model_id" "$file" <<'PY'
import pathlib, sys, tomllib
config = pathlib.Path(sys.argv[1])
model_id = sys.argv[2]
file = sys.argv[3]
data = tomllib.loads(config.read_text())
sha = data.get("sources", {}).get(model_id, {}).get("sha256", {}).get(file)
if not sha:
    sys.exit(1)
print(sha)
PY
}

if [ "$prefetch_defaults" = "1" ]; then
  model_dir="$cache_root/$nomic_id"
  temp_model_dir="$cache_root/.install-$nomic_id.tmp.$$"
  rm -rf "$temp_model_dir"
  mkdir -p "$temp_model_dir"
  base_url="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/$nomic_revision"
  revision="$nomic_revision"
  source_override=0
  if override="$(read_source_override "$nomic_id")"; then
    IFS="$(printf '\t')" read -r base_url revision <<EOF
$override
EOF
    source_override=1
  fi
  download_artifact "$base_url/model.safetensors" "$temp_model_dir/model.safetensors"
  download_artifact "$base_url/tokenizer.json" "$temp_model_dir/tokenizer.json"
  download_artifact "$base_url/config.json" "$temp_model_dir/config.json"
  if [ "$source_override" = "1" ]; then
    download_artifact "$base_url/REVISION" "$temp_model_dir/REVISION"
  else
    printf '%s\n' "$nomic_revision" >"$temp_model_dir/REVISION"
  fi
  validate_nomic_artifacts "$temp_model_dir"
  source_revision="$(tr -d '\r\n' <"$temp_model_dir/REVISION")"
  model_sha="$(sha256_file "$temp_model_dir/model.safetensors")"
  tokenizer_sha="$(sha256_file "$temp_model_dir/tokenizer.json")"
  config_sha="$(sha256_file "$temp_model_dir/config.json")"
  revision_sha="$(sha256_file "$temp_model_dir/REVISION")"
  mkdir -p "$model_dir"
  mv -f "$temp_model_dir/model.safetensors" "$model_dir/model.safetensors"
  mv -f "$temp_model_dir/tokenizer.json" "$model_dir/tokenizer.json"
  mv -f "$temp_model_dir/config.json" "$model_dir/config.json"
  mv -f "$temp_model_dir/REVISION" "$model_dir/REVISION"
  write_nomic_manifest "$model_dir" "$source_revision" "$model_sha" "$tokenizer_sha" "$config_sha" "$revision_sha"
  rm -rf "$temp_model_dir"
fi

echo "Installed cg to $prefix/bin/cg"
echo "Bundled default model: $nomic_id"
echo "Default model byte budget: $nomic_bytes"
echo "Prefetch with: cg model fetch --defaults"

if [ "$dry_run" = "1" ]; then
  echo "Dry run complete; files were staged for review in the requested prefix."
fi
