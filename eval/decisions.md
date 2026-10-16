# Eval decisions, recorded 2026-10-15

## Why eval/results.md says "pick: none" for the two classes

eval/run.py applies a limit of 200 ms/chunk (LIMITS, eval/run.py line 59). Each candidate model is slower than this limit on this CPU. The run stops over-limit models and picks nothing.

## Text embedder: bge-base-en-v1.5

Manager decision, 2026-10-15: the text limit becomes 350 ms and the CLI ships bge-base-en-v1.5.
- Test recall 0.9474, against 0.7895 for the lexical matcher (eval/results.json, text rows).
- 309.26 ms/chunk in eval/results.json. A manager run measured 229 ms/chunk. The range 229 to 309 is below the 350 ms limit.
- RSS 935 MB (eval/results.json).
- nomic-embed-text-v1.5 has the same recall 0.9474 at RSS 1853 MB (eval/results.json). bge is smaller at equal recall.

## Semantic matches advise only

User default, 2026-10-15. A semantic match advises. Only exact and lexical matches block. Shipped in dedup/cli.py.

## No code embedder

A code embedder adds +0.06 refactored recall at a union test precision of 0.7083 (`uv run python eval/analysis/union.py`: refactored recall is 0.9412 for lexical alone and 1.0 for the union). This increase does not pay for a model of 1.3 GB or more (eval/results.json, code rows, weights_mb).

## fastembed models are not revision-pinned

Sentence-transformers models load at a pinned revision (eval/run.py, MODELS). fastembed 0.9.0 has no revision parameter. It gets the current revision from the hub at download time. The cache dir comes from FASTEMBED_CACHE_PATH or the fastembed default. eval/run.py can thus not pin the fastembed models.

## Code lexical advise cutoff 0.40

0.40 is the lowest cutoff with dev precision and test precision of 0.90 or more (`uv run python eval/analysis/lexcut.py`). dedup/cli.py ships this cutoff as LEXICAL_ADVISE["code"].

The rule that picks the cutoff reads the dev split and the test split. The test precision 0.90 is thus not a clean measure. It comes from pairs that the rule used.

Since this commit, the eval lexical scores use the same per-language tokenizer as the CLI. One pair (chn-py-021) moved by 0.00006. The cutoff stays 0.40.

For publication the absolute path of the pairs file in eval/results.json was replaced with the same path relative to the user's home, written with a leading `~/`. This was the only edit to the raw results.
