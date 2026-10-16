"""Print the code-pair lexical cutoff table and the CLI's advise cutoff."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dedup import lexical

CUTS = [0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
MIN_PRECISION = 0.90  # advise rule: dev and test precision must both be >= this


def jaccard(a, b, lang=None):
    sa, sb = lexical.shingles(a, "code", lang), lexical.shingles(b, "code", lang)
    shared = len(sa & sb)
    return shared / (len(sa) + len(sb) - shared) if sa or sb else 0.0


def main():
    pairs = [json.loads(l) for l in (ROOT / "eval/pairs/pairs.jsonl").read_text().splitlines()]
    code = [p for p in pairs if p["kind"] == "code"]
    score = {p["id"]: jaccard(p["a"], p["b"], ".py" if p["lang"] == "python" else None)
             for p in code}

    def stats(split):
        sp = [p for p in code if p["split"] == split]
        out = {}
        ref = [p for p in sp if p["subset"] == "refactored"]
        ren = [p for p in sp if p["subset"] == "renamed"]
        for t in CUTS:
            flag = [p for p in sp if score[p["id"]] >= t]
            out[t] = dict(
                prec=sum(p["label"] for p in flag) / len(flag) if flag else 0.0,
                ref=sum(score[p["id"]] >= t for p in ref) / len(ref),
                ren=sum(score[p["id"]] >= t for p in ren) / len(ren),
                hn=sum(1 for p in flag if p["subset"] == "hard_negative"),
            )
        return out

    dev, test = stats("dev"), stats("test")
    for t in CUTS:
        d, s = dev[t], test[t]
        print(f"{t:.2f} dev prec={d['prec']:.4f} ref={d['ref']:.4f} ren={d['ren']:.4f} hn={d['hn']} | "
              f"test prec={s['prec']:.4f} ref={s['ref']:.4f} ren={s['ren']:.4f} hn={s['hn']}")

    ok = [t for t in CUTS if dev[t]["prec"] >= MIN_PRECISION and test[t]["prec"] >= MIN_PRECISION]
    if ok:
        print(f"CHOSEN: {min(ok):.2f} (lowest cutoff with dev and test precision >= {MIN_PRECISION:.2f})")
    else:
        print(f"CHOSEN: none (no cutoff has dev and test precision >= {MIN_PRECISION:.2f})")


if __name__ == "__main__":
    main()
