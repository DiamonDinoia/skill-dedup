# Evaluation pairs

One JSON object per line: `{"id", "kind", "lang", "a", "b", "label", "subset", "split", "source"}`. `kind` is `code` (python, cpp) or `text` (markdown). `label`: 1 duplicate, 0 not.
Subsets: code positives `exact`, `renamed`, `refactored`; code negatives `hard_negative`, `random_negative`; text positives `paraphrase`, `near_copy`; text negatives `hard_negative`, `random_negative`. Items: code 5-60 lines, text 20-200 words. `source` is `path:start-end` of the original.
The renamer in `eval/pairs/build/build_pairs.py` renames only user identifiers, each to a natural word from its own list or synonym map. It never touches keywords, stdlib or builtin names, ALL_CAPS macros, module names, or string literals, and it adds no generated suffix.
The split is by group: the same script assigns every whole source file to dev or test (seed 11), so no file appears in both splits and each subset keeps 40-60% of its pairs in dev.
Run `uv run python eval/pairs/validate.py`; it checks counts, fields, sizes, the renamed structure (a one-to-one identifier map), the group split, and the paraphrase/refactored medians. Exits non-zero on failure.
Originals come from this user's repos (`skills/*`, `finufft`, `simd-labs`) and the CPython 3.14 stdlib of the project uv venv; refactored and paraphrased sides were written for this set.
