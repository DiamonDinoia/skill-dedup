#!/usr/bin/env python3
r"""Validate eval/pairs/pairs.jsonl. Usage: python validate.py [path-to-pairs.jsonl]

Exits 1 when a subset count is below the minimum, a field is missing or malformed,
a duplicate id exists, a and b are identical outside subset "exact", sizes break
the bounds (code 5-60 lines, text 20-200 words), fewer than 20 code pairs exist
for python or cpp, the positives are too easy (median word 3-shingle Jaccard over
paraphrase >= 0.30, or median token 3-shingle Jaccard over refactored >= 0.50),
a renamed pair fails the structure check (a) or the suffix check (b), or the
split leaks (c). Prints per-subset counts and medians.

(a) renamed structure: regex-tokenize a and b; token lists have equal length,
    non-identifier tokens are equal at each position, and the identifier map
    from a to b is one-to-one and consistent.
(b) no b matches the old generated suffix pattern [A-Za-z_]\w*X\dq\b.
(c) no source file (the part of source before ':') appears in both splits, and
    each subset keeps between 40% and 60% of its pairs in dev.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MINS = {
    ("code", "exact"): 10,
    ("code", "renamed"): 25,
    ("code", "refactored"): 25,
    ("code", "hard_negative"): 40,
    ("code", "random_negative"): 20,
    ("text", "paraphrase"): 40,
    ("text", "near_copy"): 15,
    ("text", "hard_negative"): 40,
    ("text", "random_negative"): 20,
}
FIELDS = {"id": str, "kind": str, "lang": str, "a": str, "b": str,
          "label": int, "subset": str, "split": str, "source": str}
SRC_RE = re.compile(r"^\S+:\d+-\d+$")
BAD_SUFFIX = re.compile(r"[A-Za-z_]\w*X\dq\b")

# One stdlib regex tokenizer for the renamed-structure check: strings, numbers,
# identifiers, single-symbol operators. eval/pairs/build uses the same regex.
TOK_RE = re.compile(r"""
      (?P<string>[rRbBfFuU]{0,3}(?:\"\"\".*?\"\"\"|'''.*?'''|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'))
    | (?P<number>\d[\w.]*)
    | (?P<ident>[A-Za-z_]\w*)
    | (?P<op>[^\w\s'\"])
""", re.X | re.S)


def renamed_structure_error(a, b):
    """Check (a): token shapes of a and b agree; identifier map a->b is a bijection."""
    ta = [(m.lastgroup, m.group()) for m in TOK_RE.finditer(a)]
    tb = [(m.lastgroup, m.group()) for m in TOK_RE.finditer(b)]
    if len(ta) != len(tb):
        return f"token count differs: {len(ta)} vs {len(tb)}"
    fwd, rev = {}, {}
    for i, ((ka, sa), (kb, sb)) in enumerate(zip(ta, tb)):
        if ka != kb:
            return f"token {i}: kind differs: {ka} {sa!r} vs {kb} {sb!r}"
        if ka != "ident":
            if sa != sb:
                return f"token {i}: {sa!r} vs {sb!r}"
        else:
            if fwd.get(sa, sb) != sb:
                return f"identifier {sa!r} maps to {fwd.get(sa)!r} and {sb!r}"
            if rev.get(sb, sa) != sa:
                return f"identifier {sb!r} is the image of {rev.get(sb)!r} and {sa!r}"
            fwd[sa] = sb
            rev[sb] = sa
    return None


def shingles(words):
    return set(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))


def jaccard(a, b, code):
    if code:
        wa, wb = re.findall(r"\w+|[^\w\s]", a), re.findall(r"\w+|[^\w\s]", b)
    else:
        wa = re.findall(r"[A-Za-z0-9']+", a.lower())
        wb = re.findall(r"[A-Za-z0-9']+", b.lower())
    sa, sb = shingles(wa), shingles(wb)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "pairs.jsonl"
    errors = []
    rows = []
    for ln, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {ln}: bad JSON: {e}")
            continue
        for f, t in FIELDS.items():
            if f not in r:
                errors.append(f"line {ln}: missing field {f!r}")
            elif not isinstance(r[f], t):
                errors.append(f"line {ln}: field {f!r} wrong type")
        rows.append(r)

    ids = Counter(r.get("id") for r in rows)
    for i, n in ids.items():
        if n > 1:
            errors.append(f"duplicate id {i!r} x{n}")

    counts = Counter()
    splits = defaultdict(Counter)
    langs = Counter()
    jac_para, jac_ref = [], []
    for r in rows:
        kind, subset = r.get("kind"), r.get("subset")
        if kind not in ("code", "text"):
            errors.append(f"{r.get('id')}: bad kind {kind!r}")
            continue
        counts[(kind, subset)] += 1
        splits[(kind, subset)][r.get("split")] += 1
        if kind == "code":
            langs[r.get("lang")] += 1
        if r.get("label") not in (0, 1):
            errors.append(f"{r['id']}: label not 0/1")
        if r.get("split") not in ("dev", "test"):
            errors.append(f"{r['id']}: bad split")
        if BAD_SUFFIX.search(r.get("b", "")):
            errors.append(f"{r['id']}: check (b): b matches generated suffix pattern X<digit>q")
        if not SRC_RE.match(r.get("source", "")):
            errors.append(f"{r['id']}: bad source {r.get('source')!r}")
        if "a" in r and "b" in r and isinstance(r.get("a"), str) and isinstance(r.get("b"), str):
            if r["a"] == r["b"] and subset != "exact":
                errors.append(f"{r['id']}: a == b outside exact")
            if kind == "code":
                for side in ("a", "b"):
                    nl = r[side].count("\n") + 1
                    if not 5 <= nl <= 60:
                        errors.append(f"{r['id']}: {side} has {nl} lines, need 5-60")
            else:
                for side in ("a", "b"):
                    nw = len(r[side].split())
                    if not 20 <= nw <= 200:
                        errors.append(f"{r['id']}: {side} has {nw} words, need 20-200")
            if subset == "renamed":
                err = renamed_structure_error(r["a"], r["b"])
                if err:
                    errors.append(f"{r['id']}: check (a) renamed structure: {err}")
            if subset == "paraphrase":
                jac_para.append(jaccard(r["a"], r["b"], code=False))
            if subset == "refactored":
                jac_ref.append(jaccard(r["a"], r["b"], code=True))

    files_by_split = defaultdict(set)
    for r in rows:
        if r.get("split") in ("dev", "test"):
            files_by_split[r["split"]].add(r.get("source", ":").split(":")[0])
    for f in sorted(files_by_split["dev"] & files_by_split["test"]):
        errors.append(f"check (c): source file in both splits: {f}")

    for key, mn in MINS.items():
        if counts[key] < mn:
            errors.append(f"subset {key[0]}/{key[1]} has {counts[key]}, needs >= {mn}")
    for key in sorted(splits):
        c = splits[key]
        n = c["dev"] + c["test"]
        if n and not 0.4 <= c["dev"] / n <= 0.6:
            errors.append(f"check (c): subset {key[0]}/{key[1]} dev share "
                          f"{c['dev']}/{n} outside 40-60%")
    for lang in ("python", "cpp"):
        if langs[lang] < 20:
            errors.append(f"code lang {lang} has {langs[lang]} pairs, needs >= 20")

    med_para = median(jac_para) if jac_para else float("nan")
    med_ref = median(jac_ref) if jac_ref else float("nan")
    if jac_para and med_para >= 0.30:
        errors.append(f"paraphrase median word 3-shingle Jaccard {med_para:.3f} >= 0.30")
    if jac_ref and med_ref >= 0.50:
        errors.append(f"refactored median token 3-shingle Jaccard {med_ref:.3f} >= 0.50")

    print(f"file: {path}")
    for key in sorted(MINS):
        req = MINS[key]
        got = counts[key]
        sp = splits[key]
        print(f"{key[0]:5s}/{key[1]:15s} n={got:3d} (min {req:2d}) dev/test={sp['dev']}/{sp['test']}")
    print(f"code langs: python={langs['python']} cpp={langs['cpp']}")
    print(f"median word 3-shingle Jaccard paraphrase: {med_para:.3f} (max 0.30 allowed)")
    print(f"median token 3-shingle Jaccard refactored: {med_ref:.3f} (max 0.50 allowed)")
    if errors:
        for e in errors[:30]:
            print(f"ERROR: {e}")
        print(f"{len(errors)} error(s)")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
