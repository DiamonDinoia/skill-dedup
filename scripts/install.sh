#!/bin/bash
# Install dedup-check, its hook scripts and its embedding model from this clone.
# Prints the harness hook config. Harness files are only edited with --apply.
set -eu

usage() {
    cat <<'EOF'
usage: install.sh [--apply] [--no-model]

Installs the dedup-check CLI from this clone (uv tool install, or a venv with
pip when uv is not on PATH), links hermes-prereview into ~/.local/bin, copies the hook scripts to
$XDG_DATA_HOME/skill-dedup/hooks (~/.local/share by default), fetches the
embedding model (about 210 MB), and prints the harness hook config.
With --apply each existing harness file is backed up first as
<name>.bak.<utc-timestamp> and the Stop hook is merged in, idempotently.
EOF
}

APPLY=0
MODEL=1
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --no-model) MODEL=0 ;;
        --help) usage; exit 0 ;;
        *) echo "install.sh: bad flag: $arg" >&2; exit 2 ;;
    esac
done

if [ -z "${HOME:-}" ]; then
    echo "install.sh: HOME is not set; export HOME and retry" >&2
    exit 1
fi

# The hook config and --apply both run apply_hooks.py. Fail before any write.
if ! command -v python3 >/dev/null 2>&1; then
    echo "install.sh: python3 not found; install python3 >= 3.11 and retry" >&2
    exit 1
fi

PY_GATE='import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'

# shellcheck disable=SC1007  # env-prefixed cd: keep a poisoned CDPATH out of pwd
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_DIR="$DATA_HOME/skill-dedup"
HOOK_DIR="$APP_DIR/hooks"
LOCAL_BIN="$HOME/.local/bin"
STOP_HOOK="$APP_DIR/hooks/stop_hook.py"

if command -v uv >/dev/null 2>&1; then
    uv tool install --force "$REPO_DIR"
else
    if ! python3 -c "$PY_GATE"; then
        echo "install.sh: python3 >= 3.11 is required; the oldest python3 on PATH is older" >&2
        exit 1
    fi
    if ! python3 -c 'import ensurepip, venv'; then
        echo "install.sh: python3 cannot create a venv; install the python3-venv package (Debian/Ubuntu: apt install python3-venv)" >&2
        exit 1
    fi
    VENV="$APP_DIR/venv"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet "$REPO_DIR"
    mkdir -p "$LOCAL_BIN"
    ln -sf "$VENV/bin/dedup-check" "$LOCAL_BIN/dedup-check"
fi

# The hooks run python3, whichever way the CLI was installed.
if ! command -v python3 >/dev/null 2>&1 || ! python3 -c "$PY_GATE"; then
    echo "install.sh: warning: the hooks need python3 >= 3.11 on PATH; dedup-check itself is installed" >&2
fi

case ":$PATH:" in
    *":$LOCAL_BIN:"*) ;;
    *) echo "install.sh: warning: $LOCAL_BIN is not on PATH; add it to run dedup-check" >&2 ;;
esac

mkdir -p "$HOOK_DIR"
cp "$REPO_DIR/hooks/stop_hook.py" "$REPO_DIR/hooks/dedup_contract.py" \
   "$REPO_DIR/hooks/hermes-prereview" "$HOOK_DIR/"
chmod +x "$HOOK_DIR/stop_hook.py" "$HOOK_DIR/hermes-prereview"
# Same bin dir as dedup-check, so hermes-adversarial-review finds it by bare name.
mkdir -p "$LOCAL_BIN"
ln -sf "$HOOK_DIR/hermes-prereview" "$LOCAL_BIN/hermes-prereview"

DEDUP_CHECK=$LOCAL_BIN/dedup-check
[ -x "$DEDUP_CHECK" ] || DEDUP_CHECK=dedup-check
if [ "$MODEL" -eq 1 ]; then
    "$DEDUP_CHECK" --fetch-model
fi

cat <<EOF

Merge the first block into ~/.claude/settings.json and the second block
into ~/.codex/hooks.json. Codex asks to trust the changed file on the next
start. The Codex entry runs stop_hook.py with --harness codex.
EOF
# One builder of the command: apply_hooks.py prints the exact JSON it writes.
python3 "$REPO_DIR/scripts/apply_hooks.py" --print "$STOP_HOOK"
cat <<EOF

Hermes: $HOOK_DIR/hermes-prereview is installed and linked as
$LOCAL_BIN/hermes-prereview. To wire it into
hermes-adversarial-review, rerun with --apply, or run the patch by hand:
  patch -p1 -d $LOCAL_BIN < $REPO_DIR/hooks/hermes-adversarial-review.patch

The three hook scripts stay in one directory ($HOOK_DIR): both scripts
import dedup_contract from their own real directory. All pieces need
dedup-check on PATH.
EOF

if [ "$APPLY" -eq 1 ]; then
    python3 "$REPO_DIR/scripts/apply_hooks.py" "$STOP_HOOK"
    REVIEW="$LOCAL_BIN/hermes-adversarial-review"
    if [ -f "$REVIEW" ] && patch -R --dry-run -p1 -d "$LOCAL_BIN" \
            < "$REPO_DIR/hooks/hermes-adversarial-review.patch" >/dev/null 2>&1; then
        echo "install.sh: hermes-adversarial-review.patch is already applied to $REVIEW" >&2
    elif [ -f "$REVIEW" ] && patch --dry-run -p1 -d "$LOCAL_BIN" \
            < "$REPO_DIR/hooks/hermes-adversarial-review.patch" >/dev/null 2>&1; then
        patch -p1 -d "$LOCAL_BIN" < "$REPO_DIR/hooks/hermes-adversarial-review.patch"
        echo "install.sh: applied hermes-adversarial-review.patch to $REVIEW" >&2
    else
        echo "install.sh: skip hermes patch: $REVIEW missing or patch does not apply" >&2
    fi
fi
