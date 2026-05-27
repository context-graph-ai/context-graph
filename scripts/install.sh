#!/usr/bin/env bash
set -euo pipefail

prefix="/usr/local"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --prefix)
      prefix="${2:-}"
      shift 2
      ;;
    --dry-run|--prefetch-defaults)
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [ -z "$prefix" ]; then
  echo "missing --prefix value" >&2
  exit 2
fi

mkdir -p "$prefix/bin" "$HOME/.config/cg"

cat >"$prefix/bin/cg" <<'CG_STUB'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  disconnect|connect)
    exit 0
    ;;
  *)
    echo "cg command stub"
    exit 0
    ;;
esac
CG_STUB
chmod 755 "$prefix/bin/cg"

cat >"$prefix/cg-uninstall.sh" <<'UNINSTALL_STUB'
#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="/tmp/cg/models"
CG_BINARY="/tmp/cg/bin/cg"
CG_HOOK_MARKER="# wrong-cg-hook-marker"

if [ "${1:-}" = "--help" ]; then
  echo "stub uninstall help"
  exit 0
fi

echo "cg uninstall stub"
UNINSTALL_STUB
chmod 755 "$prefix/cg-uninstall.sh"

echo "context graph installer stub"
