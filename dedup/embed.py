"""Embed chunk text; load() returns None when the model is unavailable."""

from __future__ import annotations

import os
import warnings

import numpy as np

# A kind with no entry here gets no embedder and no CLI note. A code model is
# one more entry. The fastembed model truncates each input to 512 tokens.
MODELS = {
    "text": "BAAI/bge-base-en-v1.5",  # fastembed name, dim 768
}

# Source: git show e067dfafbf1d352752204975a24b5ae68960bb27:eval/results.json,
# bge-base-en-v1.5 row, dev precision >= 0.95 (advise) and dev precision 1.0
# (block, 0.8202).
THRESHOLDS = {"text": (0.8064655181434274, 0.8201913417129902)}

_BATCH = 16

# First line of the last load failure per kind, for the CLI note.
ERRORS: dict[str, str] = {}


class Embedder:
    def __init__(self, fe, model_id):
        self._fe = fe
        self.model_id = model_id

    def embed(self, texts):
        rows = []
        for i in range(0, len(texts), _BATCH):
            batch = list(self._fe.embed(list(texts[i : i + _BATCH])))
            v = np.asarray(batch, np.float32)
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            rows.append(v / np.maximum(norm, 1e-12))
        return np.vstack(rows) if rows else np.zeros((0, 0), np.float32)


def _model_cache_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(root, "skill-dedup", "models")


def load(kind) -> Embedder | None:
    model = MODELS.get(kind)
    if model is None:
        return None
    try:
        # fastembed re-enables the hub progress bars in its retry loop, so they
        # must be locked off with the env var before the hub import, and the
        # warnings that the lock causes in enable/disable_progress_bars filtered
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        warnings.filterwarnings("ignore", message="Cannot (en|dis)able progress bars")
        from fastembed import TextEmbedding
        import fastembed
        from loguru import logger

        # the CLI owns stderr; keep fastembed progress and error logs out of it
        logger.disable("fastembed")
        fe = TextEmbedding(
            model_name=model,
            cache_dir=_model_cache_dir(),
            # the hook runs beside the agent, so it gets a small thread budget
            threads=min(4, os.cpu_count() or 1),
        )
        # load the weights and the ONNX session now, so an offline or corrupt
        # model fails here and not in the CLI match loop
        list(fe.embed(["warmup"]))
    except Exception as e:
        msg = str(e)
        ERRORS[kind] = msg.splitlines()[0] if msg.strip() else e.__class__.__name__
        return None
    model_id = f"{model}-fe{fastembed.__version__}"
    return Embedder(fe, model_id)
