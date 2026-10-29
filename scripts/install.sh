#!/bin/bash
# Install dedup-check, its hook scripts and its embedding model from this clone.
# Prints the harness hook config. Harness files are only edited with --apply.
set -eu

usage() {
    cat <<'EOF'
usage: install.sh [--apply] [--no-model] [--cpu] [--stop-hook]

Installs the dedup-check CLI from this clone (uv tool install, or a venv with
pip when uv is not on PATH). When `nvidia-smi -L` succeeds, the script
installs the gpu extra (fastembed-gpu, onnxruntime-gpu with CUDA 13 and cuDNN
wheels, about 2.5 GB) and excludes the CPU fastembed and onnxruntime.
Otherwise, or with --cpu, it installs the CPU build. The pip fallback is
CPU only. It links hermes-prereview into ~/.local/bin, copies the hook
scripts to
$XDG_DATA_HOME/skill-dedup/hooks (~/.local/share by default), and fetches the
embedding model (about 210 MB).
--apply patches ~/.local/bin/hermes-adversarial-review so the review runs
hermes-prereview once and gets the findings in its prompt. It also upgrades a
script that carries the previous version of the patch.
--stop-hook is optional and not recommended on big repos. It prints the Stop
hook config for Claude Code and Codex. With --stop-hook --apply each existing
harness file is backed up first as <name>.bak.<utc-timestamp> and the Stop
hook is merged in, idempotently.
EOF
}

APPLY=0
STOP_HOOK_OPT=0
MODEL=1
FORCE_CPU=0
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --no-model) MODEL=0 ;;
        --cpu) FORCE_CPU=1 ;;
        --stop-hook) STOP_HOOK_OPT=1 ;;
        --help) usage; exit 0 ;;
        *) echo "install.sh: bad flag: $arg" >&2; exit 2 ;;
    esac
done

if [ -z "${HOME:-}" ]; then
    echo "install.sh: HOME is not set; export HOME and retry" >&2
    exit 1
fi

# The hook scripts and apply_hooks.py run python3. Fail before any write.
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

# A working GPU, not just the binary: nvidia-smi -L must exit 0. The CUDA 13
# wheels of onnxruntime-gpu need driver 580 or newer (CUDA 13.x, Linux).
MIN_DRIVER=580
GPU=0
if [ "$FORCE_CPU" -eq 0 ] && nvidia-smi -L >/dev/null 2>&1; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
    DRIVER_MAJOR=${DRIVER%%.*}
    case "$DRIVER_MAJOR" in
        ''|*[!0-9]*) echo "install.sh: cannot read the NVIDIA driver version; installing the CPU build" >&2 ;;
        *) if [ "$DRIVER_MAJOR" -ge "$MIN_DRIVER" ]; then GPU=1
           else echo "install.sh: NVIDIA driver $DRIVER is older than $MIN_DRIVER (CUDA 13); installing the CPU build" >&2
           fi ;;
    esac
fi
if [ "$GPU" -eq 1 ]; then
    echo "install.sh: GPU found; installing the gpu extra (about 2.5 GB). Use --cpu to skip it." >&2
fi

if command -v uv >/dev/null 2>&1; then
    if [ "$GPU" -eq 1 ]; then
        # fastembed-gpu replaces fastembed; both ship the onnxruntime module
        EXCLUDES=$(mktemp)
        printf 'fastembed\nonnxruntime\n' > "$EXCLUDES"
        uv tool install --force "$REPO_DIR[gpu]" --excludes "$EXCLUDES"
        rm -f "$EXCLUDES"
    else
        uv tool install --force "$REPO_DIR"
    fi
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
    # the pip fallback is CPU only
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

if [ "$STOP_HOOK_OPT" -eq 1 ]; then
    cat <<EOF

Merge the first block into ~/.claude/settings.json and the second block
into ~/.codex/hooks.json. Codex asks to trust the changed file on the next
start. The Codex entry runs stop_hook.py with --harness codex.
EOF
    # One builder of the command: apply_hooks.py prints the exact JSON it writes.
    python3 "$REPO_DIR/scripts/apply_hooks.py" --print "$STOP_HOOK"
fi
cat <<EOF

Hermes: $HOOK_DIR/hermes-prereview is installed and linked as
$LOCAL_BIN/hermes-prereview. To wire it into
hermes-adversarial-review, rerun with --apply, or run the patch by hand:
  patch -p1 -d $LOCAL_BIN < $REPO_DIR/hooks/hermes-adversarial-review.patch

The hook scripts stay in one directory ($HOOK_DIR): they
import dedup_contract from their own real directory. All pieces need
dedup-check on PATH.
EOF

if [ "$APPLY" -eq 1 ]; then
    REVIEW="$LOCAL_BIN/hermes-adversarial-review"
    PATCH="$REPO_DIR/hooks/hermes-adversarial-review.patch"
    PATCH_V0="$REPO_DIR/hooks/hermes-adversarial-review.v0.patch"
    SCRATCH=$(mktemp -d)
    trap 'rm -rf "$SCRATCH"' EXIT
    # Patch state of the live script: new patch, previous patch (v0), or none.
    if [ ! -f "$REVIEW" ]; then
        echo "install.sh: skip hermes patch: $REVIEW missing" >&2
    elif patch -R --dry-run -p1 -d "$LOCAL_BIN" < "$PATCH" >/dev/null 2>&1; then
        echo "install.sh: hermes-adversarial-review.patch is already applied to $REVIEW" >&2
    elif patch --dry-run -p1 -d "$LOCAL_BIN" < "$PATCH" >/dev/null 2>&1; then
        patch -p1 -d "$LOCAL_BIN" < "$PATCH"
        echo "install.sh: applied hermes-adversarial-review.patch to $REVIEW" >&2
    elif cp "$REVIEW" "$SCRATCH/" \
            && patch -R -p1 -d "$SCRATCH" < "$PATCH_V0" >/dev/null 2>&1 \
            && patch --dry-run -p1 -d "$SCRATCH" < "$PATCH" >/dev/null 2>&1; then
        # the previous patch aborted the review on a blocking duplicate: replace it
        patch -R -p1 -d "$LOCAL_BIN" < "$PATCH_V0"
        patch -p1 -d "$LOCAL_BIN" < "$PATCH"
        echo "install.sh: upgraded hermes-adversarial-review.patch on $REVIEW" >&2
    else
        echo "install.sh: skip hermes patch: patch does not apply to $REVIEW" >&2
    fi
    if [ "$STOP_HOOK_OPT" -eq 1 ]; then
        python3 "$REPO_DIR/scripts/apply_hooks.py" "$STOP_HOOK"
    fi
fi
