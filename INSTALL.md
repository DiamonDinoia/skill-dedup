# Install skill-dedup

This file gives an agent the full install. Run the steps in order and adapt
them to the host. Each command comes from the install script that this file
replaces. Keep the values as they are.

The old script had four flags. Each one is a choice in this file: `--apply`,
`--stop-hook`, `--no-model` and `--cpu`.

## 1. Check the host

1. Check that HOME is set. Run `echo "$HOME"`. An empty answer means HOME is
   not set. Export HOME and stop.
2. Check python3. Run `command -v python3`. The install and the hooks need
   python3 3.11 or later on PATH. Check the version.

   ```
   python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
   ```

   A non-zero exit means the oldest python3 on PATH is too old. Stop and
   report it.
3. Check that python3 can create a venv. Skip this check when uv is on PATH.

   ```
   python3 -c 'import ensurepip, venv'
   ```

   A non-zero exit means the python3-venv package is missing. On Debian and
   Ubuntu install it with `apt install python3-venv`. Stop and report.
4. Check uv. Run `command -v uv`. With uv the install uses `uv tool install`.
   Without uv step 5 makes a venv with pip. The pip route is CPU only.
5. Check the GPU driver only on a host with an NVIDIA GPU. See step 4.

## 2. Get the source

```
git clone https://github.com/DiamonDinoia/skill-dedup
cd skill-dedup
```

Run all later commands from the clone root.

## 3. Set the paths

```
REPO_DIR=$(pwd)
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_DIR="$DATA_HOME/skill-dedup"
HOOK_DIR="$APP_DIR/hooks"
LOCAL_BIN="$HOME/.local/bin"
STOP_HOOK="$APP_DIR/hooks/stop_hook.py"
```

`DATA_HOME` follows XDG_DATA_HOME when it is set. Otherwise it is
~/.local/share. With uv the tool venv is under ~/.local/share/uv/tools/skill-dedup.
Without uv the venv is $APP_DIR/venv.

## 4. Choose the GPU or CPU build

The `gpu` extra adds fastembed-gpu and onnxruntime-gpu[cuda,cudnn], the CUDA 13
wheels. The footprint is about 2.5 GB. fastembed and fastembed-gpu both ship
the onnxruntime module, so the CPU build would shadow the GPU build. The base
install keeps fastembed, so a plain install works with no extra step. The GPU
install must exclude the CPU packages fastembed and onnxruntime.

Check the driver.

```
nvidia-smi -L
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1
```

Use the GPU build when the first command exits 0 and the driver major version
is 580 or newer. The check needs a working GPU, not only the binary. 580 is
the NVIDIA minimum for CUDA 13.x on Linux x86_64, from the CUDA Toolkit
release notes. A driver major below 580 means the CPU build and one stderr
line. A driver version that is not a number means the CPU build and one stderr
line. Choose the CPU build by hand with the `--cpu` choice when the GPU is not
wanted. The pip route is CPU only.

At run time dedup/embed.py checks onnxruntime.get_available_providers(). With
CUDAExecutionProvider it calls onnxruntime.preload_dlls() and asks for CUDA
with a CPU fallback. If the CUDA session fails, or runs on CPU, the embedder
falls back to the CPU path with 4 threads at most. The model does not become
unavailable.

Measured on an RTX 4070 Laptop with 8 GB and 2000 Markdown chunks: 6.2 s on
CUDA. The CPU build did not finish them in 11 minutes on 4 threads. CUDA and
CPU vectors differ. On 200 chunks the cosine between the two vectors of a
chunk is 0.999992 at worst. A pair score moves by up to 1e-3. That is enough
to cross a threshold, so the cache key of a CUDA run ends in `-cuda`. A CPU run
keeps its old key.

## 5. Install the CLI

GPU route, with uv.

```
EXCLUDES=$(mktemp)
printf 'fastembed\nonnxruntime\n' > "$EXCLUDES"
uv tool install --force "$REPO_DIR[gpu]" --excludes "$EXCLUDES"
rm -f "$EXCLUDES"
```

The excludes file lists the CPU packages. The tool venv then has no CPU build.

CPU route, with uv.

```
uv tool install --force "$REPO_DIR"
```

Pip route, CPU only, when uv is absent.

```
VENV="$APP_DIR/venv"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet "$REPO_DIR"
mkdir -p "$LOCAL_BIN"
ln -sf "$VENV/bin/dedup-check" "$LOCAL_BIN/dedup-check"
```

A manual install is the same two routes from the clone root: `uv tool install .`
or the venv commands above.

Check the PATH at the end. When uv was used and its tool bin directory is not
~/.local/bin, that directory must be on PATH to run dedup-check. When $LOCAL_BIN
is not on PATH, print one warning line and add it.

## 6. Check the hook runtime

The hook scripts run python3. They need python3 3.11 or later on PATH, whatever
way the CLI was installed. Print one warning line when this fails.
dedup-check itself is installed at this point.

## 7. Install the hook scripts

```
mkdir -p "$HOOK_DIR"
cp "$REPO_DIR/hooks/stop_hook.py" "$REPO_DIR/hooks/dedup_contract.py" \
   "$REPO_DIR/hooks/hermes-prereview" "$HOOK_DIR/"
chmod +x "$HOOK_DIR/stop_hook.py" "$HOOK_DIR/hermes-prereview"
mkdir -p "$LOCAL_BIN"
ln -sf "$HOOK_DIR/hermes-prereview" "$LOCAL_BIN/hermes-prereview"
```

The copy out of the clone keeps the harness config free of clone paths.
stop_hook.py, hermes-prereview and dedup_contract.py must stay in one
directory. Symlinks are fine: both scripts import dedup_contract from their
own real directory. The review runs hermes-prereview by bare name from the
bin dir of dedup-check. All pieces need dedup-check on PATH.

## 8. Choose the model download

The model is BAAI/bge-base-en-v1.5, about 210 MB. It goes to
$XDG_CACHE_HOME/skill-dedup/models, ~/.cache/skill-dedup/models by default.
Download it.

```
DEDUP_CHECK=$LOCAL_BIN/dedup-check
[ -x "$DEDUP_CHECK" ] || DEDUP_CHECK=dedup-check
"$DEDUP_CHECK" --fetch-model
```

Skip the download with the `--no-model` choice on an offline host. Run
`dedup-check --fetch-model` later to get the model. When the model cannot load,
the command prints one stderr message and goes on with lexical checks only.

Model selection. Text chunks use bge-base-en-v1.5. Its test recall is 0.95 on
paraphrase pairs, against 0.79 for the lexical matcher. The paraphrase recall
comes from 19 test pairs. The text model is English only. No code embedder
ships. On the pair set a code embedder added 0.06 refactored recall at a union
precision of 0.71. [eval/decisions.md](eval/decisions.md) has the numbers and
the command that prints them. The full shoot-out is in
[eval/results.md](eval/results.md).

## 9. Wire the hooks

Both steps are optional. Nothing else in this file edits a harness file.

### 9.1 The `--apply` step, Hermes

The patch to ~/.local/bin/hermes-adversarial-review makes the review run
hermes-prereview once before the review and append its findings to the review
prompt. Take a backup of $REVIEW before the first patch command.

```
REVIEW="$LOCAL_BIN/hermes-adversarial-review"
PATCH="$REPO_DIR/hooks/hermes-adversarial-review.patch"
PATCH_V0="$REPO_DIR/hooks/hermes-adversarial-review.v0.patch"
SCRATCH=$(mktemp -d)
```

Apply the patch in one of four states.

1. Missing. When $REVIEW does not exist, print
   `skip hermes patch: $REVIEW missing` and stop this step. Do not create the
   file.
2. Already applied. Check with the next command. A zero exit means the patch
   is in place. Print `already applied` and change nothing. An applied patch
   is not applied twice.

   ```
   patch -R --dry-run -p1 -d "$LOCAL_BIN" < "$PATCH"
   ```
3. Fits. Check with the next command. A zero exit means the patch fits. Apply
   it and print `applied hermes-adversarial-review.patch`.

   ```
   patch --dry-run -p1 -d "$LOCAL_BIN" < "$PATCH"
   patch -p1 -d "$LOCAL_BIN" < "$PATCH"
   ```
4. Previous patch. When the script carries the previous patch
   hooks/hermes-adversarial-review.v0.patch, upgrade it. The previous patch
   aborted the review on a blocking duplicate, so it must go. Test in a
   scratch dir first.

   ```
   cp "$REVIEW" "$SCRATCH/"
   patch -R -p1 -d "$SCRATCH" < "$PATCH_V0"
   patch --dry-run -p1 -d "$SCRATCH" < "$PATCH"
   ```

   When both commands succeed, upgrade the real script and print
   `upgraded hermes-adversarial-review.patch`.

   ```
   patch -R -p1 -d "$LOCAL_BIN" < "$PATCH_V0"
   patch -p1 -d "$LOCAL_BIN" < "$PATCH"
   ```
5. No fit. When no check passes, print
   `skip hermes patch: patch does not apply to $REVIEW` and leave the file
   alone.

### 9.2 The `--stop-hook` step, Claude Code and Codex

First print the two JSON blocks.

```
python3 "$REPO_DIR/scripts/apply_hooks.py" --print "$STOP_HOOK"
```

The first block is the entry for ~/.claude/settings.json. The second block is
the entry for ~/.codex/hooks.json and it has `--harness codex` on the command.
Codex drops Stop output with hookSpecificOutput, so the hook prints the Codex
shape only with that flag.

Then merge the entries with the script. A script merge with backups is safer
than a hand edit.

```
python3 "$REPO_DIR/scripts/apply_hooks.py" "$STOP_HOOK"
```

The script backs up each harness file first. The backup has the same file name
plus `.bak.` and a UTC timestamp. The merge keeps the other keys and the hooks
of other tools, and is idempotent. A symlinked file is resolved first: the file
behind the link is updated and the link stays. A dangling symlink at
~/.claude/settings.json gets its missing target created. A new harness file
gets mode 0644, an existing file keeps its mode. A read-only 0444 target file
in a writable directory is still replaced. Its backup keeps the content. When
a write of a later file fails, one stderr line names the updated files and
their backups.

The Stop hook is optional and not recommended on big repos. It runs the check
after every agent turn. One run on a 21,873-file repo takes longer than the
570 s timeout on CPU. Install it only on small repos, or when the `gpu` extra
is in use.

### 9.3 Manual merge, fallback

Use this fallback when the script cannot edit the harness files. Replace
/ABS/PATH/TO in each snippet with the real path of this repo.

1. Claude Code: merge the Stop entry of hooks/claude-settings.json into
   ~/.claude/settings.json under "hooks". Keep existing entries.
2. Codex: merge the Stop entry of hooks/codex-hooks.json into
   ~/.codex/hooks.json under "hooks". Keep existing entries. Codex will ask to
   trust the changed file on next start.
3. The snippets only add a Stop hook that runs stop_hook.py with a 600 s
   timeout. They do not remove other hooks.

## 10. Verify the install

1. `command -v dedup-check` prints one path.
2. `command -v hermes-prereview` prints one path.
3. Run `dedup-check` in a git repo with a change set. It exits 0 and prints one
   line for each result.
4. When the Stop hook was installed, send one Stop event JSON to stop_hook.py
   on stdin and check the exit code and stdout.

   ```
   echo '{}' | python3 "$STOP_HOOK"
   echo '{}' | python3 "$STOP_HOOK" --harness codex
   ```

   The Codex entry prints `{"systemMessage": ...}` for advice.

## 11. Uninstall

```
uv tool uninstall skill-dedup
rm -f ~/.local/bin/dedup-check
rm -rf ~/.local/share/skill-dedup
rm -f ~/.local/bin/hermes-prereview
rm -rf ~/.cache/skill-dedup
```

With XDG_DATA_HOME set, the app dir is $XDG_DATA_HOME/skill-dedup instead of
~/.local/share/skill-dedup. Remove the dedup Stop entries from the harness
configs by hand. A moved clone leaves its old hook entry in the harness files.
Remove it by hand too.
