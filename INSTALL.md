# Install skill-dedup

This file gives an agent the full install. Run the steps in order and adapt
them to the host. Each command comes from the install script that this file
replaces. Keep the values as they are.

The old script had four flags. Each one is a choice in this file: apply the
Hermes patch, install the Stop hook, skip the model download and force the
CPU build.

## 1. Check the host

1. Check that HOME is set. Run `echo "$HOME"`. An empty answer means HOME is
   not set. Export HOME and stop.
2. Check python3. Run `command -v python3`. Check the version.

   ```
   python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
   ```

   The pip route needs python3 3.11 or later on PATH. On that route a
   non-zero exit means the oldest python3 on PATH is too old. Stop and
   report it. The uv route brings its own Python, so do not stop there.
   Only the hooks need this python3 on the uv route. Section 6 has the
   warning for that case.
3. Check uv. Run `command -v uv`. With uv the install uses `uv tool install`.
   Without uv, install uv for the current user:

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   The installer puts uv in ~/.local/bin. When the curl command is
   denied or curl is missing, download the uv binary from the GitHub
   release with python3 urllib instead:

   ```
   python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tgz && tar -xzf /tmp/uv.tgz -C /tmp && install -D /tmp/uv-x86_64-unknown-linux-gnu/uv ~/.local/bin/uv
   ```

   The asset name is for x86_64 Linux. For aarch64 Linux use
   `uv-aarch64-unknown-linux-gnu.tar.gz` and change the
   `uv-x86_64-unknown-linux-gnu` path in the command to match it. For
   other platforms take the matching asset name from
   https://github.com/astral-sh/uv/releases. When the install still
   fails, take the pip route in step 4. The pip route is CPU only.
4. Pip route only. Check that python3 can create a venv.

   ```
   python3 -c 'import ensurepip, venv'
   ```

   A non-zero exit means the python3-venv package is missing. On Debian and
   Ubuntu install it with `apt install python3-venv` when the user is root
   or has sudo. When neither is available, stop and report.
5. Check the GPU driver only on a host with an NVIDIA GPU. See step 3.

## 2. Get the source

```
git clone https://github.com/DiamonDinoia/skill-dedup
cd skill-dedup
```

Run all later commands from the clone root. When the repo is already
cloned, skip the clone and use its path as REPO_DIR.

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
line. Force the CPU build by hand when the GPU is not
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

When any install above fails because onnxruntime or fastembed has no wheel
for the running Python, rerun the uv command with `--python 3.13`. uv
downloads Python 3.13 on first use. On the pip route, make the venv with
`python3.13` or `python3.12` when one exists. When neither exists, install
uv with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or the
python3 urllib fallback from step 3) and take the uv route.

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

Skip the download on an offline host. Run
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

### 9.1 Apply the Hermes patch

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

### 9.2 Install the Stop hook, Claude Code and Codex

First print the two JSON blocks.

```
python3 "$REPO_DIR/scripts/apply_hooks.py" --print "$STOP_HOOK"
```

The first block is the entry for ~/.claude/settings.json. The second block is
the entry for ~/.codex/hooks.json and it has `--harness codex` on the command.
Codex drops Stop output with hookSpecificOutput, so the hook prints the Codex
shape only with that flag.

The Stop hook is optional. The agent asks the user whether to install it and
states the cost in one sentence. It runs dedup-check after every agent turn,
and one run on a 21,873-file repo exceeds the 570 s timeout on CPU. On a yes,
run the merge below. On a no, skip it. When the run cannot receive an answer,
such as `claude -p` or `codex exec`, skip the question and the merge, and name
the install command in the final message:
`python3 "$REPO_DIR/scripts/apply_hooks.py" "$STOP_HOOK"`.

Then merge the entries with the script on a yes. A script merge with backups
is safer than a hand edit.

```
python3 "$REPO_DIR/scripts/apply_hooks.py" "$STOP_HOOK"
```

The script backs up each harness file first. The backup has the same file name
plus `.bak.` and a UTC timestamp. The merge keeps the other keys and the hooks
of other tools. A re-run replaces the entry that it installed. This also
applies to a hand edit with an env prefix such as `PATH=`: any NAME=value assignment before the command. An entry that changed in other
ways stays as a second dedup entry. Delete it first when the hook must run
once. A symlinked file is resolved first: the file
behind the link is updated and the link stays. A dangling symlink at
~/.claude/settings.json gets its missing target created. A new harness file
gets mode 0644, an existing file keeps its mode. A read-only 0444 target file
in a writable directory is still replaced. Its backup keeps the content. When
a write of a later file fails, one stderr line names the updated files and
their backups.

Wire the Stop hook only after dedup-check is on PATH. When the CLI
install is blocked, state the blocker and keep the hook install command
in the final message even if the user answered yes.

Codex skips untrusted hooks. After the apply_hooks.py merge, the agent
must relay all three of these facts to the user, for Codex only:

1. Codex records trust against the hook's hash and skips a new or
   changed hook until it is trusted. The source is
   https://developers.openai.com/codex/hooks.
2. The docs page says to open /hooks to review and trust hooks. At the next
   interactive start, if the hook needs review, Codex prints a warning that
   tells you to open /hooks. The startup dialog we probed offers the option
   "Trust all and continue".
3. `codex exec --dangerously-bypass-hook-trust` runs the hook without
   stored trust for one invocation.

The agent does not edit config.toml itself.

### 9.3 Manual merge, fallback

Use this fallback when the script cannot edit the harness files. Replace
/ABS/PATH/TO in each snippet with the real path of this repo.

1. Claude Code: merge the Stop entry of hooks/claude-settings.json into
   ~/.claude/settings.json under "hooks". Keep existing entries.
2. Codex: merge the Stop entry of hooks/codex-hooks.json into
   ~/.codex/hooks.json under "hooks". Keep existing entries. Codex marks the
   changed hook for review and skips it until trusted. The docs page says to
   open /hooks to trust it, and notes Codex prints a warning at the next
   interactive start that tells you to open /hooks. The startup dialog we
   probed offers "Trust all and continue".
3. The snippets only add a Stop hook that runs stop_hook.py with a 600 s
   timeout. They do not remove other hooks.

## 10. Verify the install

1. `command -v dedup-check` prints one path.
2. `command -v hermes-prereview` prints one path.
3. Run the seeded check in a scratch git repo. It exits 0 and prints a
   line ending in `exact 1.000 BLOCK` or `lexical 1.000 BLOCK`.

```
D=$(mktemp -d)
cd "$D"
git init -q
git config user.name tester
git config user.email tester@example.com
cat > a.py <<'EOF'
def word_frequencies(text):
    words = text.lower().split()
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    limit = 10
    top = ranked[:limit]
    return dict(top)
EOF
git add a.py
git commit -q -m seed
cp a.py b.py
dedup-check --repo "$D"
```
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
