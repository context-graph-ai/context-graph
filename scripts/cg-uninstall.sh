#!/usr/bin/env bash
set -euo pipefail

prefix="${CG_INSTALL_PREFIX:-/usr/local}"
# The canonical resolver is context_graph::embedder::nomic::default_cache_root: the
# override first, then $XDG_CACHE_HOME, then $HOME/.cache. Resolving it any other way
# here would seed a directory cg never reads.
cache_root="${CG_MODEL_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/cg/models}"
cg_binary="${CG_BINARY:-$prefix/bin/cg}"

clean_all_hooks() {
  python3 - "$HOME" <<'PY'
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

case "${1:-}" in
  --help)
    echo "Usage: cg-uninstall.sh [--purge]"
    echo "Default uninstall removes cg binaries, model cache, user config, and preserves project .cg directories."
    echo "--purge prompts before removing preserved .cg project directories."
    exit 0
    ;;
esac

rm -f "$cg_binary"
rm -rf "$cache_root" "$HOME/.config/cg"
clean_all_hooks

if [ "${1:-}" = "--purge" ]; then
  printf 'Remove all .cg directories under %s? Type yes to continue: ' "$HOME"
  read -r answer
  if [ "$answer" = "yes" ]; then
    find "$HOME" -type d -name .cg -prune -exec rm -rf {} +
  else
    echo "Preserved project .cg directories."
  fi
else
  echo "Preserved project .cg directories. To inspect: find $HOME -type d -name .cg"
fi
