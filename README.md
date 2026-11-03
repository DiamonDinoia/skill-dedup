# skill-dedup

Find duplicate code and text in a change set.

## What it does

The `dedup-check` command reads the change set of a git worktree. It takes the diff from the base commit plus untracked files that git does not ignore, chunks the new and changed lines, and compares each new chunk with all chunks of the same repo at the base commit.

Three checks run in sequence.

1. Exact match. The two normalized texts are the same.
2. Lexical match. MinHash and Jaccard similarity on normalized token shingles.
3. Semantic match. Cosine similarity on embeddings. Text chunks only. The model is BAAI/bge-base-en-v1.5 through fastembed, ONNX on CPU or CUDA.

## Install

Ask your agent to install it. Paste this sentence:

`Install skill-dedup by following https://github.com/DiamonDinoia/skill-dedup/blob/main/INSTALL.md`

[INSTALL.md](INSTALL.md) holds the agent steps, from the prerequisites to the uninstall.

## Use

```
dedup-check [--base REF] [--json] [--repo PATH]
```

`--base REF` sets the base commit. The default is the merge-base of HEAD and its upstream, or HEAD when the branch has no upstream. `--json` prints the results as one JSON object. `--repo PATH` selects the repo. The default is the current directory.

```
dedup/lexical.py:49-88 ~ dedup/cli.py:30-69 lexical 0.970 BLOCK
```

Plain output is one line for each result. The JSON object has the keys base and findings, and each finding has the new chunk, the existing chunk, the match type, the score and a block flag.

The exit code is 0 on a completed run and 2 on an error such as a bad ref or a bad flag. Results do not change the exit code. Each error is one line on stderr with the prefix `dedup-check:`. In a directory that is not a git repo the command prints an empty result and exits 0. When the model cannot load, for example on an offline host, the command goes on with lexical checks only.

## Run once before a review

The default design runs the check once, right before an adversarial review. `hermes-adversarial-review` runs `hermes-prereview --repo PATH` and appends the findings to the review prompt, under the line "Duplicate check (dedup-check, run once before this review)". The reviewer verifies each candidate and treats `[block]` items as high severity. A blocking result with exit code 3 does not stop the review. It goes into the prompt. An error, crash, missing binary or timeout prints one stderr line and adds a note to the prompt that the check did not run. The review goes on. `--no-dedup` skips the check.

## Decision table

| match | condition | result |
|---|---|---|
| exact | normalized texts the same | block |
| lexical, code | Jaccard 0.95 or more | block |
| lexical, text | Jaccard 0.95 or more | block |
| lexical, code | Jaccard 0.40 or more | advise |
| lexical, text | Jaccard 0.60 or more | advise |
| semantic, text | cosine 0.8065 or more | advise |

Semantic matches do not block. They advise only. Findings in .json, .jsonl, .ndjson, .csv and .tsv files advise only and never block. A run reports 20 findings at most: blockers first, then by score.

## Known limits

- The matchers miss most refactored code with a new structure. Only 1 of 17 test refactors has Jaccard 0.40 or more.
- A cold run keeps all eligible base blobs in memory at once. See `_cat_blobs` in dedup/cli.py:254.
- The command captures the full diff before size filtering. A 49 MB rewrite peaked at 335 MB RSS.
- All string and number literals tokenize to LIT, so chunks that differ only in values match fully.
- The command skips chunks below 5 non-blank code lines or 20 words, and the generated lock files uv.lock, poetry.lock, Pipfile.lock, package-lock.json, yarn.lock, pnpm-lock.yaml, Cargo.lock, Gemfile.lock, composer.lock and go.sum.
- The check stays in one repo. A new chunk can match only chunks of the same repo at the base commit.

## Development

Run the tests from the repo root. Use `python -m pytest`, because a bare `pytest` can import a stale wheel. The model test uses the real embedding model.

    uv run python -m pytest -q
    DEDUP_TEST_MODEL=1 uv run python -m pytest -q -m model

Check the pair set with `uv run python eval/pairs/validate.py`. The full shoot-out is in eval/results.md. `uv run --extra eval python eval/run.py` regenerates eval/results.json and eval/results.md. It takes more than 20 minutes.
