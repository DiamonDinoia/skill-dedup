## code: pick none (all candidates failed limits or loading)
| model | sem recall | AUC | ms/chunk |
|---|---|---|---|
| lexical (baseline) | 0.9655 | 0.8066 | None |
| CodeRankEmbed | 0.8276 | 0.8993 | 697.45 |
| jina-embeddings-v2-base-code | 0.7931 | 0.888 | 1162.83 |
| codesage-small-v2 (UNRUN: ImportError: cannot import name 'Conv1D' from 'transformers.modeling_utils') | - | - | - |
## text: pick none (all candidates failed limits or loading)
| model | sem recall | AUC | ms/chunk |
|---|---|---|---|
| lexical (baseline) | 0.7895 | 0.9588 | None |
| Qwen3-Embedding-0.6B (UNRUN: over-limit: ms/chunk > 200 (killed at 693 s)) | - | - | - |
| bge-base-en-v1.5 | 0.9474 | 1.0 | 309.26 |
| nomic-embed-text-v1.5 | 0.9474 | 1.0 | 289.18 |
