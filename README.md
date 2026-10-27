# skill-dedup

Find duplicate code and text in a change set.

## Install

Run the install script from a clone of the repo:

```
git clone https://github.com/DiamonDinoia/skill-dedup
./skill-dedup/scripts/install.sh
```

The script does these steps:

1. Install the command. With `uv` on PATH the script runs
   `uv tool install --force` on the clone directory. Without `uv` it makes a
   venv at ~/.local/share/skill-dedup/venv, installs the package with pip,
   and adds `dedup-check` to ~/.local/bin. The pip path must have python3
   3.11 or a subsequent release, and the python3-venv package on Debian and
   Ubuntu.
2. Copy the hook scripts to ~/.local/share/skill-dedup/hooks. Then the
   harness config does not refer to the clone. Set XDG_DATA_HOME to use a
   different parent directory.
3. Download the model, unless --no-model. The model is
   BAAI/bge-base-en-v1.5, about 210 MB. It goes to
   $XDG_CACHE_HOME/skill-dedup/models, ~/.cache by default. Use
   `install.sh --no-model` to skip the download. Run
   `dedup-check --fetch-model` after the install to get the model.
4. Print the Stop hook JSON for Claude Code and Codex, and the Hermes step.
   The script does not edit a harness file. The blocks are the output of
   `scripts/apply_hooks.py --print`, so each printed command is the same
   string that --apply writes into that harness file. The second block has
   `--harness codex` on the command: Codex drops Stop output with
   hookSpecificOutput, so the hook prints the Codex shape only with that
   flag.
5. With --apply, back up each harness file first, then merge the Stop hook.
   The backup has the same file name plus `.bak.` and a UTC timestamp. The
   targets are ~/.claude/settings.json and ~/.codex/hooks.json. A symlinked
   file is resolved first: the file behind the link is updated and the link
   stays. The merge keeps the other keys and the hooks of other tools, and
   is idempotent. A new harness file gets mode 0644, an existing file keeps
   its mode. When a write of a later file fails, one stderr line names the
   updated files and their backups. For Hermes the script applies
   hooks/hermes-adversarial-review.patch to
   ~/.local/bin/hermes-adversarial-review when that file is available and
   `patch --dry-run` gives no error. If not, the script prints one skip
   line. Run `install.sh --help` for the flag list.

Manual install. From the clone root run `uv tool install .`, or make the
venv and use pip by hand:

```
python3 -m venv ~/.local/share/skill-dedup/venv
~/.local/share/skill-dedup/venv/bin/pip install .
ln -sf ~/.local/share/skill-dedup/venv/bin/dedup-check ~/.local/bin/dedup-check
```

Then run `dedup-check --fetch-model` and read hooks/install.md for the
harness config.

## Uninstall

```
uv tool uninstall skill-dedup  # or: rm ~/.local/bin/dedup-check
rm -rf ~/.local/share/skill-dedup
rm -rf ~/.cache/skill-dedup
```

Remove the dedup Stop entries from the harness configs by hand.

## What it does

The `dedup-check` command reads the change set of a git worktree: the diff
from the base commit to the worktree, plus untracked files that git does not
ignore. It splits the new and changed lines into chunks. Then it compares each
new chunk with all chunks of the same repo at the base commit.

The command does three checks in sequence:

1. Exact match: the two normalized texts are the same.
2. Lexical match: MinHash and Jaccard similarity on normalized token shingles.
3. Semantic match: cosine similarity on embeddings. Text chunks only. The
   model is BAAI/bge-base-en-v1.5 through fastembed, ONNX on CPU.

## Use

```
dedup-check [--base REF] [--json] [--repo PATH]
```

- `--base REF` sets the base commit. The default is the merge-base of HEAD
  and its upstream, or HEAD when the branch has no upstream.
- `--json` prints the results as one JSON object.
- `--repo PATH` selects the repo. The default is the current directory.

Plain output is one line for each result:

```
dedup/lexical.py:49-88 ~ dedup/cli.py:30-69 lexical 0.970 BLOCK
```

The JSON object has the keys base and findings. Each finding has the new
chunk, the existing chunk, the match type, the score and a block flag. The new
chunk has the keys path, start, end and kind. The existing chunk has the keys
path, start and end. A match type is exact, lexical or
semantic.

```
dedup-check --json
```

Exit codes:

- 0: the check ran. Results do not change the exit code.
- 2: an error, for example a bad ref or a bad flag.

Each error and message is one line on stderr with the prefix `dedup-check:`.
In a directory that is not a git repo the command prints an empty result and
exits 0.

The first semantic run downloads the model. When the model cannot load, for
example on an offline host, the command prints one stderr message and goes on
with lexical checks only.

## Decision table

| match | condition | result |
|---|---|---|
| exact | normalized texts the same | block |
| lexical, code | Jaccard 0.95 or more | block |
| lexical, text | Jaccard 0.95 or more | block |
| lexical, code | Jaccard 0.40 or more | advise |
| lexical, text | Jaccard 0.60 or more | advise |
| semantic, text | cosine 0.8065 or more | advise |

Semantic matches do not block. They advise only. Findings in .json, .jsonl,
.ndjson, .csv and .tsv files advise only and never block. A run reports 20
findings at most: blockers first, then by score.

## Model selection

Text chunks use bge-base-en-v1.5: test recall 0.95 on paraphrase pairs,
against 0.79 for the lexical matcher. No code embedder ships: on the pair set
a code embedder added 0.06 refactored recall at a union precision of 0.71
([eval/decisions.md](eval/decisions.md) has the numbers and the command that prints them).
The full shoot-out is in eval/results.md. `uv run --extra eval python eval/run.py`
regenerates eval/results.json and eval/results.md.

## Known limits

- The matchers do not catch most refactored code with a new structure and not
  many shared tokens: only 1 of 17 test refactors has Jaccard 0.40 or more.
- A cold run keeps all eligible base blobs in memory at once
  (dedup/cli.py:254, `_cat_blobs`).
- The command captures the full diff before size filtering: a 49 MB rewrite
  peaked at 335 MB RSS.
- All string and number literals tokenize to LIT, so chunks that differ only
  in values match fully.
- The brace chunker can glue a preceding declaration such as `int value = 0;`
  onto the next function.
- Line counts differ from git on U+2028 and form feed.
- A corrupt but readable .npz cache entry is ignored and the result is
  recomputed (dedup/cli.py:413).
- The calibrated semantic block threshold (dedup/embed.py:19) is unused:
  semantic findings never block.
- The command skips chunks below 5 non-blank code lines or 20 words.
- The command skips the generated lock files uv.lock, poetry.lock,
  Pipfile.lock, package-lock.json, yarn.lock, pnpm-lock.yaml, Cargo.lock,
  Gemfile.lock, composer.lock and go.sum.
- The text model is English only.
- The first semantic run downloads about 210 MB to
  $XDG_CACHE_HOME/skill-dedup/models.
- The check stays in one repo: a new chunk can match only chunks of the same
  repo at the base commit.
- A dangling symlink at ~/.claude/settings.json gets its missing target
  created.
- A read-only (0444) target file in a writable directory is still replaced.
  Its backup keeps the content.
- With uv, install.sh needs uv's tool bin directory on PATH when it is not
  ~/.local/bin.
- A moved clone leaves its old hook entry in the harness files. Remove it
  by hand.
- The paraphrase recall comes from 19 test pairs.

## Development

Run the tests from the repo root. Use `python -m pytest`, because a bare
`pytest` can import a stale wheel.

```
uv run python -m pytest -q
```

The model test uses the real embedding model. Set DEDUP_TEST_MODEL to 1.

```
DEDUP_TEST_MODEL=1 uv run python -m pytest -q -m model
```

Check the pair set.

```
uv run python eval/pairs/validate.py
```

Run the model shoot-out again. This takes more than 20 minutes.

```
uv run --extra eval python eval/run.py
```
