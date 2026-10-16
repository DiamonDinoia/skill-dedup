"""Model shoot-out for the duplicate detector.

Main process computes the lexical baseline, spawns one child per embedding
model (clean RSS per model), scores pairs, picks a winner per class, and
writes results.json / results.md. Child mode: --one NAME prints result JSON.

Child wall budget: LOAD_WALL_S + 2 x 200 ms x item count. A child over the
budget is killed and the model is recorded as over-limit.
"""
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

HERE = Path(__file__).parent
PAIRS_DEFAULT = HERE / "pairs" / "pairs.jsonl"
CACHE = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
LOAD_WALL_S = 600
WALL_S_PER_ITEM = 0.4  # 2 passes (full + truncated) x the 200 ms/chunk limit
FE_THREADS = 6

# Document-side prefixes from each HF model card; both sides of a symmetric
# pair get the document prefix. St revisions pinned per the shoot-out decision;
# fe models are not revision-pinnable (eval/decisions.md).
MODELS = {
    "code": [
        dict(name="CodeRankEmbed", hf="nomic-ai/CodeRankEmbed",
             rev="3c4b60807d71f79b43f3c4363786d9493691f8b1", backend="st",
             prefix="", params=136731648),
        dict(name="jina-embeddings-v2-base-code", hf="jinaai/jina-embeddings-v2-base-code",
             backend="fe",
             prefix="", params=160869120),
        dict(name="codesage-small-v2", hf="codesage/codesage-small-v2",
             rev="4844c2f24b25e181aa43ca058cc73dd2622565c1", backend="st",
             prefix="", params=None),  # safetensors total missing on the Hub; counted at load
    ],
    "text": [
        dict(name="Qwen3-Embedding-0.6B", hf="Qwen/Qwen3-Embedding-0.6B",
             rev="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", backend="st",
             prefix="", params=595776512),
        dict(name="bge-base-en-v1.5", hf="BAAI/bge-base-en-v1.5",
             backend="fe",
             prefix="", params=109482752),
        dict(name="nomic-embed-text-v1.5", hf="nomic-ai/nomic-embed-text-v1.5",
             backend="fe",
             prefix="search_document: ", params=136731648),
    ],
}
SEMANTIC = {"code": {"renamed", "refactored"}, "text": {"paraphrase"}}
LIMITS = dict(params=600e6, rss_mb=4000, ms_per_chunk=200)


def load_pairs(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def lexical_scores(pairs):
    from dedup.lexical import shingles
    out = {}
    for p in pairs:
        lang = ".py" if p.get("lang") == "python" else None
        sa, sb = shingles(p["a"], p["kind"], lang), shingles(p["b"], p["kind"], lang)
        shared = len(sa & sb)
        out[p["id"]] = shared / (len(sa) + len(sb) - shared) if sa or sb else 0.0
    return out


def fe_weights_repo(model_name):
    """fastembed maps some model names to a different HF repo for weights."""
    from fastembed import TextEmbedding
    for m in TextEmbedding.list_supported_models():
        if m["model"] == model_name:
            return (m.get("sources") or {}).get("hf") or model_name
    return model_name


def weights_mb(hf_id):
    # Weights are symlinks into a shared blobs dir; count each real file once.
    pat = "models--" + hf_id.replace("/", "--")
    seen, total = set(), 0
    for root in (CACHE, CACHE / "hub"):
        for d in root.glob(f"**/{pat}"):
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                st = f.stat()  # follows symlinks to the blob
                if (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    return round(total / 1e6, 1)


def run_one(model, pairs):
    """Child: load one model, embed all pair sides, print result JSON."""
    t0 = time.perf_counter()
    if model["backend"] == "st":
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model["hf"], revision=model["rev"],
                                 trust_remote_code=True, device="cpu")
        params = sum(p.numel() for p in st.parameters())

        def embed(texts):
            return st.encode(texts, batch_size=16, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)

        tok = st.tokenizer
        enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
        dec = tok.decode
        max_len = st.max_seq_length
        weights_repo = model["hf"]
    else:
        from fastembed import TextEmbedding
        # cache_dir=None: fastembed honours FASTEMBED_CACHE_PATH, else its default
        fe = TextEmbedding(model["hf"], threads=FE_THREADS)

        def embed(texts):
            return np.stack(list(fe.embed(texts, batch_size=16))).astype(np.float64)

        tok = fe.model.tokenizer
        enc = lambda t: tok.encode(t, add_special_tokens=False).ids
        dec = tok.decode
        max_len = getattr(fe.model, "max_length", None)
        params = model["params"]
        weights_repo = fe_weights_repo(model["hf"])
    load_s = time.perf_counter() - t0

    test_ids = {p["id"] for p in pairs if p["split"] == "test"}
    seen, timed_texts, rest = set(), [], []
    for p in sorted(pairs, key=lambda p: p["id"] not in test_ids):  # test first
        for side in ("a", "b"):
            if p[side] in seen:
                continue
            seen.add(p[side])
            (timed_texts if p["id"] in test_ids else rest).append(p[side])

    token_counts = [len(enc(t)) for t in timed_texts + rest]
    mean_tokens = round(sum(token_counts) / len(token_counts), 1)
    trunc_texts = [dec(ids[:512]) for ids in (enc(t) for t in timed_texts)]

    t1 = time.perf_counter()
    emb = dict(zip(timed_texts, embed([model["prefix"] + t for t in timed_texts])))
    dt = time.perf_counter() - t1
    if rest:
        emb.update(zip(rest, embed([model["prefix"] + t for t in rest])))
    t2 = time.perf_counter()
    embed([model["prefix"] + t for t in trunc_texts])
    dt512 = time.perf_counter() - t2

    # fastembed models here already output unit vectors; re-normalize anyway
    scores = {}
    for p in pairs:
        va, vb = emb[p["a"]], emb[p["b"]]
        va = va / np.linalg.norm(va)
        vb = vb / np.linalg.norm(vb)
        scores[p["id"]] = float(va @ vb)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # MB
    json.dump(dict(scores=scores, load_s=round(load_s, 2),
                   ms_per_chunk=round(dt / len(timed_texts) * 1000, 2),
                   ms_per_chunk_trunc512=round(dt512 / len(trunc_texts) * 1000, 2),
                   rss_mb=round(rss, 0), weights_mb=weights_mb(weights_repo),
                   params=params, max_seq_length=max_len,
                   mean_tokens=mean_tokens), sys.stdout)


def auc(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    _, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    ranks[order] = avg[inv]
    np_, nn = y.sum(), len(y) - y.sum()
    if np_ == 0 or nn == 0:
        return None
    return round(float((ranks[y == 1].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)), 4)


def dev_threshold(dev_ids, labels, scores):
    """Lowest threshold with dev precision >= 0.95; else the max-precision one."""
    cands = sorted({scores[i] for i in dev_ids})
    def prec(t):
        pred = [i for i in dev_ids if scores[i] >= t]
        return sum(labels[i] for i in pred) / len(pred) if pred else 0.0
    ok = [t for t in cands if prec(t) >= 0.95]
    return min(ok) if ok else max(cands, key=prec)


def block_threshold(dev_ids, labels, scores):
    neg = [scores[i] for i in dev_ids if labels[i] == 0]
    return math.nextafter(max(neg), math.inf) if neg else math.nextafter(1.0, math.inf)


def metrics(kind, pairs, scores):
    test = [p for p in pairs if p["split"] == "test"]
    dev = [p for p in pairs if p["split"] == "dev"]
    lab = {p["id"]: p["label"] for p in dev + test}
    dev_ids = [p["id"] for p in dev]
    t = dev_threshold(dev_ids, lab, scores)
    tb = block_threshold(dev_ids, lab, scores)
    sem = [p for p in test if p["subset"] in SEMANTIC[kind]]
    sem_recall = sum(scores[p["id"]] >= t for p in sem) / len(sem) if sem else 0.0
    at_tb = [p for p in test if scores[p["id"]] >= tb]
    tp = sum(p["label"] for p in at_tb)
    return dict(
        auc=auc([p["label"] for p in test], [scores[p["id"]] for p in test]),
        dev_threshold=t, block_threshold=tb,
        sem_recall=round(sem_recall, 4),
        block_precision=round(tp / len(at_tb), 4) if at_tb else None,
        block_recall=round(tp / sum(p["label"] for p in test), 4),
        hard_neg_max=round(max((scores[p["id"]] for p in test
                                if p["subset"] == "hard_negative"), default=0.0), 4),
    )


def pick(candidates):
    ok = [c for c in candidates if "scores" in c
          and c["params"] <= LIMITS["params"] and c["rss_mb"] <= LIMITS["rss_mb"]
          and c["ms_per_chunk"] <= LIMITS["ms_per_chunk"]]
    if not ok:
        return None
    top = max(c["sem_recall"] for c in ok)
    near = [c for c in ok if c["sem_recall"] >= top - 0.03]
    return min(near, key=lambda c: c["weights_mb"])


def error_line(stderr):
    """First line that names the exception type; else the first non-empty line."""
    import re
    for line in stderr.splitlines():
        if re.search(r"\w*(Error|Exception)\b", line):
            return line.strip()[:200]
    return next((l.strip()[:200] for l in stderr.splitlines() if l.strip()), "unknown error")


def child(model, pairs_path, n_items, extra_env=None):
    budget = LOAD_WALL_S + WALL_S_PER_ITEM * n_items
    env = dict(os.environ, **(extra_env or {}))
    try:
        r = subprocess.run([sys.executable, __file__, "--one", model["name"],
                            "--pairs", pairs_path],
                           capture_output=True, text=True, timeout=budget, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"over-limit: ms/chunk > 200 (killed at {budget:.0f} s)",
                "killed_at_s": round(budget)}
    if r.returncode != 0:
        return {"error": error_line(r.stderr)}
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--pairs", default=str(PAIRS_DEFAULT))
    ap.add_argument("--mutate-lexical", action="store_true",
                    help="replace every embedding score with the lexical score")
    args = ap.parse_args()
    pairs = load_pairs(args.pairs)
    if args.one:
        kind = next(k for k, ms in MODELS.items() for m in ms if m["name"] == args.one)
        model = next(m for m in MODELS[kind] if m["name"] == args.one)
        run_one(model, [p for p in pairs if p["kind"] == kind])
        return

    results = {"mutated": args.mutate_lexical, "pairs": str(args.pairs),
               "cores": sorted(os.sched_getaffinity(0)), "revisions": {}, "classes": {}}
    md = []
    for kind in ("code", "text"):
        kp = [p for p in pairs if p["kind"] == kind]
        lex = lexical_scores(kp)
        n_items = len({t for p in kp for t in (p["a"], p["b"])})
        rows = [dict(name="lexical (baseline)", **metrics(kind, kp, lex),
                     load_s=None, ms_per_chunk=None, ms_per_chunk_trunc512=None,
                     rss_mb=None, weights_mb=None)]
        cands = []
        for m in MODELS[kind]:
            results["revisions"][m["hf"]] = m.get("rev")  # None: fe is not revision-pinned
            if args.mutate_lexical:
                r = dict(scores=lex, load_s=0.0, ms_per_chunk=0.0,
                         ms_per_chunk_trunc512=0.0, rss_mb=0.0,
                         weights_mb=weights_mb(m["hf"]) or 0.0, params=0)
            else:
                r = child(m, args.pairs, n_items)
                if "error" in r and "over-limit" not in r["error"]:
                    r = child(m, args.pairs, n_items, extra_env={"HF_HUB_OFFLINE": "1"})
            if "error" in r:
                row = dict(name=m["name"] + " (UNRUN: " + r["error"] + ")")
                if "killed_at_s" in r:
                    row["killed_at_s"] = r["killed_at_s"]
                rows.append(row)
                continue
            met = metrics(kind, kp, r["scores"])
            # r.get: mutate and partial results can lack load-time fields
            rows.append(dict(name=m["name"], **met, **{k: r.get(k) for k in (
                "load_s", "ms_per_chunk", "ms_per_chunk_trunc512", "rss_mb",
                "weights_mb", "max_seq_length", "mean_tokens")}))
            cands.append(dict(name=m["name"], scores=r["scores"], params=r["params"],
                              rss_mb=r["rss_mb"], ms_per_chunk=r["ms_per_chunk"],
                              weights_mb=r["weights_mb"], sem_recall=met["sem_recall"]))
        w = pick(cands)
        results["classes"][kind] = dict(rows=rows, pick=w["name"] if w else None,
                                        beats_lexical=bool(w) and w["sem_recall"] > rows[0]["sem_recall"])
        if w:
            pick_row = next(r for r in rows if r["name"] == w["name"])
            md.append(f"## {kind}: pick {w['name']} (dev thr {pick_row['dev_threshold']:.4f}, "
                      f"block thr {pick_row['block_threshold']:.4f})")
        else:
            md.append(f"## {kind}: pick none (all candidates failed limits or loading)")
        md.append("| model | sem recall | AUC | ms/chunk |")
        md.append("|---|---|---|---|")
        for r in rows:
            if "sem_recall" not in r:
                md.append(f"| {r['name']} | - | - | - |")
            else:
                md.append(f"| {r['name']} | {r['sem_recall']} | {r['auc']} | {r['ms_per_chunk']} |")
    ok = all(c["pick"] and c["beats_lexical"] for c in results["classes"].values())
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
