---
name: dedup
description: Use when a change set can contain a duplicate of code or text in its own repo. Runs dedup-check on the change set and reports block or advise results.
---

The `dedup-check` command finds duplicate code and text in the change set of a
git worktree. It chunks the new and changed lines, then compares each new
chunk with all chunks of the same repo at the base commit. Three checks run:
exact match, lexical match (MinHash and Jaccard on token shingles), and
semantic match (text only, BAAI/bge-base-en-v1.5 through fastembed on CPU).

Run it in the repo root:

```
dedup-check [--base REF] [--json] [--repo PATH]
```

The default base is the merge-base of HEAD and its upstream, or HEAD when the
branch has no upstream. Use `--json` for machine-readable output: one object
with the keys base and findings. Each finding has the new chunk, the existing
chunk, the match type, the score and a block flag.

Exact matches, and lexical matches at Jaccard 0.95 or more, block. Lexical
matches from 0.40 on code and 0.60 on text advise. Semantic matches from
cosine 0.8065 advise. Semantic matches do not block. The exit code is 0 on a
completed run and 2 on an error. Each error is one line on stderr.

An offline host keeps lexical checks only: the command prints one stderr
message and goes on. The first semantic run downloads about 210 MB to
$XDG_CACHE_HOME/skill-dedup/models.

Limits: the matchers do not catch most refactored code with not many shared
tokens, the command skips chunks below 5 non-blank code lines or 20 words, and
the text model is English only.

The full use, the decision table and the development commands are in
README.md. Install from the repo root with scripts/install.sh: it installs
`dedup-check`, downloads the model, and prints the hook config. The manual
hook setup is in hooks/install.md.
