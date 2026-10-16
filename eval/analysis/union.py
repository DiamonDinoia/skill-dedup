"""Print lexical-alone vs union (lexical or code embedder) metrics on code pairs.

Scores come from eval/results.json when it has per-pair scores (key
"per_pair_scores": {model: {pair_id: score}}); the shipped results.json has
none, so this reads the committed eval/analysis/code_scores.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMBEDDERS = ("CodeRankEmbed", "jina")


def dev_threshold(dev_ids, label, scores):
    # run.py advise rule: lowest candidate with dev precision >= 0.95
    cands = sorted({scores[i] for i in dev_ids})
    def prec(t):
        pred = [i for i in dev_ids if scores[i] >= t]
        return sum(label[i] for i in pred) / len(pred) if pred else 0.0
    ok = [t for t in cands if prec(t) >= 0.95]
    return min(ok) if ok else max(cands, key=prec)


def main():
    pairs = [json.loads(l) for l in (ROOT / "eval/pairs/pairs.jsonl").read_text().splitlines()]
    code = [p for p in pairs if p["kind"] == "code"]
    label = {p["id"]: p["label"] for p in code}
    dev = [p["id"] for p in code if p["split"] == "dev"]
    test = [p for p in code if p["split"] == "test"]

    res = json.loads((ROOT / "eval/results.json").read_text())
    scores = res.get("per_pair_scores")
    if scores is None:
        scores = json.loads((ROOT / "eval/analysis/code_scores.json").read_text())

    lex = scores["lexical"]
    tlex = dev_threshold(dev, label, lex)
    ref = [p["id"] for p in test if p["subset"] == "refactored"]
    ren = [p["id"] for p in test if p["subset"] == "renamed"]

    def metrics(flag):
        tp = sum(label[i] for i in flag)
        prec = tp / len(flag) if flag else 0.0
        return prec, sum(i in flag for i in ref) / len(ref), sum(i in flag for i in ren) / len(ren)

    lp, lr, ln = metrics({i for i in [p['id'] for p in test] if lex[i] >= tlex})
    print(f"lexical alone: advise={tlex:.4f} test precision={lp:.4f} refactored={lr:.4f} renamed={ln:.4f}")
    for name in EMBEDDERS:
        te = dev_threshold(dev, label, scores[name])
        p, r, n = metrics({i for i in [p['id'] for p in test] if lex[i] >= tlex or scores[name][i] >= te})
        print(f"union lex+{name}: emb advise={te:.4f} test precision={p:.4f} "
              f"refactored={r:.4f} ({r - lr:+.2f}) renamed={n:.4f}")


if __name__ == "__main__":
    main()
