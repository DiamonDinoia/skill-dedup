"""Find duplicate chunks between the worktree change set and a base commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

from dedup import lexical
from dedup.chunk import _MD_EXTS, chunk_file
from dedup.lexical import LexicalIndex

CACHE_VERSION = 4
# shortcut: hard cap on reported findings, upgrade by paging if repos need more
MAX_FINDINGS = 20
# Measured on eval/pairs code pairs (e067dfa): 0.40 is the lowest code cutoff with dev precision 1.000 and test precision 0.900.
LEXICAL_ADVISE = {"code": 0.40, "text": 0.60}
LEXICAL_BLOCK = 0.95
MAX_BLOB_BYTES = 1 << 20
SEM_BLOCK = 4096

# Generated lock files give no chunks: every literal tokenizes to LIT,
# so any two windows of one lock file score lexical 1.0.
_GENERATED_NAMES = {
    "uv.lock", "poetry.lock", "Pipfile.lock", "package-lock.json",
    "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "Gemfile.lock",
    "composer.lock", "go.sum",
}

_HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_KIND_CODE = {"code": 0, "text": 1}
_KINDS = ("code", "text")


@dataclass
class _Ex:
    path: str
    start: int
    end: int
    kind: str
    sha: str  # "" marks a worktree new-chunk candidate
    j: int  # index among the same-kind chunks of its base blob


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _lex_lang(c) -> str | None:
    # The tokenizer's code rules: the file extension, except a markdown
    # fence chunk, which follows its fence language ('python' -> '.py').
    if c.kind != "code":
        return None
    ext = _ext(c.path)
    if ext in _MD_EXTS:
        return ".py" if c.lang == "python" else None
    return ext


def _git(top: str, *args: str) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", "-C", top, *args], capture_output=True)
    if p.returncode != 0:
        # contract: one stderr line, so keep the first non-empty git line.
        err = next(
            (l for l in p.stderr.decode("utf-8", "replace").splitlines() if l.strip()),
            "",
        )
        raise RuntimeError(err or f"git {args[0]} failed")
    return p


def _git_ok(repo: str, *args: str) -> bool:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True
    ).returncode == 0


def _resolve_base(repo: str, ref: str | None) -> str:
    if ref is not None:
        p = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--verify", f"{ref}^{{commit}}"],
            capture_output=True,
        )
        if p.returncode != 0:
            raise RuntimeError(f"bad ref: {ref}")
        return p.stdout.decode().strip()
    if _git_ok(repo, "rev-parse", "--abbrev-ref", "@{upstream}"):
        return _git(repo, "merge-base", "HEAD", "@{upstream}").stdout.decode().strip()
    return _git(repo, "rev-parse", "HEAD").stdout.decode().strip()


_ESC = {ord("t"): 9, ord("n"): 10, ord('"'): 34, ord("\\"): 92}


def _unquote_git_path(s: bytes) -> str:
    # C-quoted path: octal escapes and \t \n \" \\ become raw bytes.
    out = bytearray()
    i = 1
    end = len(s) - (1 if s.endswith(b'"') else 0)
    while i < end:
        c = s[i]
        i += 1
        if c != 0x5C:  # backslash
            out.append(c)
        elif 0x30 <= s[i] <= 0x33:  # octal escape, three digits
            out.append(int(s[i:i + 3], 8))
            i += 3
        else:
            out.append(_ESC.get(s[i], s[i]))
            i += 1
    return bytes(out).decode("utf-8", "surrogateescape")


def _diff_path(s: bytes) -> str:
    # s is the path part of a ---/+++ line, with an a/ or b/ prefix.
    if s.endswith(b"\t"):
        s = s[:-1]
    p = _unquote_git_path(s) if s.startswith(b'"') else os.fsdecode(s)
    return p[2:]


def _diff_ranges(top: str, base: str):
    # shortcut: patch captured whole before size filtering (49 MB rewrite
    # peaked at 335 MB RSS), stream from a Popen pipe if that matters
    out = _git(
        top, "-c", "core.quotePath=false",
        "diff", "--text", "--no-color", "--no-ext-diff", "--no-textconv",
        "--src-prefix=a/", "--dst-prefix=b/", "--no-relative", "-U0", base,
    ).stdout
    added: dict[str, list[tuple[int, int]]] = {}
    removed: dict[str, list[tuple[int, int]]] = {}
    deleted: set[str] = set()
    split: dict[str, list[int]] = {}
    old = new = None
    lines = out.split(b"\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(b"diff --git "):
            old = new = None  # header lines count only until the first @@
        elif line.startswith(b"--- "):
            s = line[4:]
            old = None if s == b"/dev/null" else _diff_path(s)
        elif line.startswith(b"+++ "):
            s = line[4:]
            new = None if s == b"/dev/null" else _diff_path(s)
            if new is None and old is not None:
                deleted.add(old)
        elif (m := _HUNK_RE.match(line)):
            a, b = int(m.group(1)), int(m.group(2) or 1)
            c, d = int(m.group(3)), int(m.group(4) or 1)
            if old is not None:
                if b:
                    removed.setdefault(old, []).append((a, a + b - 1))
                else:
                    # Insertion after old line a: only a chunk spanning lines
                    # a and a+1 is cut by it; boundary chunks stay existing.
                    split.setdefault(old, []).append(a)
            if new is not None:
                # A pure deletion +c,0 still disturbs the new-side chunk at
                # lines c and c+1 of the file that kept the hunk.
                added.setdefault(new, []).append(
                    (c, c + d - 1) if d else (c, c + 1)
                )
            rem_o, rem_n = b, d
            i += 1
            while i < n and (rem_o or rem_n):
                body = lines[i]
                if body.startswith(b"\\"):  # "\ No newline at end of file"
                    i += 1
                    continue
                rem_o -= not body.startswith(b"+")
                rem_n -= not body.startswith(b"-")
                i += 1
            continue
        i += 1
    return added, removed, deleted, split


def _overlap(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= hi and lo <= end for lo, hi in ranges)


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _read_worktree(repo: str, path: str) -> str | None:
    if os.path.basename(path) in _GENERATED_NAMES:
        return None  # generated lock file: no chunk
    full = os.path.join(repo, path)
    try:
        if not stat.S_ISREG(os.lstat(full).st_mode):
            return None  # symlinks and devices are never read
        with os.fdopen(os.open(full, os.O_RDONLY | _O_NOFOLLOW), "rb") as f:
            data = f.read(MAX_BLOB_BYTES + 1)
    except OSError:
        return None
    if len(data) > MAX_BLOB_BYTES or b"\x00" in data:
        return None
    return data.decode("utf-8", "replace")


def _ls_tree(top: str, base: str) -> list[tuple[str, str]]:
    out = _git(top, "ls-tree", "-r", "-z", base).stdout
    entries = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        _, typ, sha = meta.split()
        if typ == b"blob":
            name = os.fsdecode(path)
            if os.path.basename(name) not in _GENERATED_NAMES:
                entries.append((name, sha.decode()))
    return entries


def _batch_check(repo: str, shas: list[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if not shas:
        return sizes
    proc = subprocess.Popen(
        ["git", "-C", repo, "cat-file", "--batch-check"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    for sha in shas:
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().decode().split()
        if len(header) >= 3 and header[1] == "blob":
            sizes[sha] = int(header[2])
        else:
            # "<sha> missing", or an empty read on a dead process.
            proc.kill()
            proc.wait()
            raise RuntimeError(f"unreadable base object {sha}")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("git cat-file --batch-check failed")
    return sizes


def _cat_blobs(repo: str, shas: list[str]) -> dict[str, bytes]:
    # shortcut: cold run holds every eligible base blob in memory, stream per
    # blob when eligible blobs exceed a few hundred MB
    sizes = _batch_check(repo, shas)
    small = [s for s in shas if sizes.get(s, MAX_BLOB_BYTES + 1) <= MAX_BLOB_BYTES]
    data: dict[str, bytes] = {}
    if not small:
        return data
    proc = subprocess.Popen(
        ["git", "-C", repo, "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    for sha in small:
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().decode().split()
        if len(header) < 3 or header[1] != "blob":
            # The object was there at --batch-check time; fail, don't skip.
            proc.kill()
            proc.wait()
            raise RuntimeError(f"unreadable base object {sha}")
        size = int(header[2])
        raw = proc.stdout.read(size)
        proc.stdout.read(1)
        if b"\x00" not in raw:
            data[sha] = raw
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("git cat-file --batch failed")
    return data


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def _norm_hash(text: str) -> bytes:
    return hashlib.sha1(_norm(text).encode("utf-8")).digest()


_warned: set[str] = set()


def _note(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"dedup-check: {msg}", file=sys.stderr)


def _load_embedder(kind: str):
    try:
        from dedup import embed
    except ImportError as e:
        msg = str(e)
        _note(
            f"{kind} model unavailable ("
            f"{msg.splitlines()[0] if msg.strip() else 'import failed'}); lexical only"
        )
        return None
    if kind not in embed.MODELS:
        return None  # no model for this kind by design: no note
    emb = embed.load(kind)
    if emb is None:
        _note(f"{kind} model unavailable ({embed.ERRORS.get(kind, 'unknown')}); lexical only")
        return None
    return emb, *embed.THRESHOLDS[kind]


def _cache_dir() -> str:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(root, "skill-dedup")


def _save_cache(path: str, **arrays) -> None:
    """Write npz arrays atomically; remove the temp file on any failure."""
    cache_dir = os.path.dirname(path)
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _cached_vecs(cache_dir: str, model_id: str, sha: str, ext: str, kchunks, emb):
    spans = np.array([[c.start, c.end] for c in kchunks], dtype=np.int64)
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)
    xname = re.sub(r"[^A-Za-z0-9_.-]", "_", ext)
    path = os.path.join(cache_dir, f"{name}-{CACHE_VERSION}-{xname}-{sha}.npz")
    try:
        with np.load(path) as z:
            if z["spans"].shape == spans.shape and bool((z["spans"] == spans).all()):
                return z["vecs"].astype(np.float32, copy=True)
    except Exception:
        pass  # a corrupt or unreadable cache file is recomputed
    vecs = np.asarray(emb.embed([c.text for c in kchunks]), dtype=np.float32)
    try:
        _save_cache(path, spans=spans, vecs=vecs)
    except OSError:
        pass  # the cache must never break a run
    return vecs


def _kind_matrix(exs: list[_Ex], blob_chunks: dict, cache_dir: str, emb):
    vecs: dict[tuple[str, str], np.ndarray] = {}
    for e in exs:
        key = (e.sha, _ext(e.path))
        if key not in vecs:
            kchunks = [c for c in blob_chunks[key] if c.kind == e.kind]
            vecs[key] = _cached_vecs(
                cache_dir, emb.model_id, key[0], key[1], kchunks, emb
            )
    return np.stack([vecs[(e.sha, _ext(e.path))][e.j] for e in exs])


def _new_chunks(repo: str, added, untracked: list[str]):
    out = []
    for p in sorted(set(added) | set(untracked)):
        text = _read_worktree(repo, p)
        if text is None:
            continue
        ranges = added.get(p)
        for c in chunk_file(p, text):
            if ranges is None or _overlap(c.start, c.end, ranges):
                out.append(c)
    return out


def _lex_cache_path(cache_dir: str, ext: str, sha: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", ext)
    return os.path.join(cache_dir, f"lex-{CACHE_VERSION}-{name}-{sha}.npz")


def _load_lex(path: str, num_perm: int, scheme: str):
    """Return (spans, kinds, normh, off, ints, hv) or None on miss or corrupt."""
    try:
        with np.load(path) as z:
            n = len(z["kinds"])
            ok = (
                z["spans"].shape == (n, 2)
                and z["normh"].shape == (n, 20)
                and len(z["off"]) == n + 1
                and int(z["off"][0]) == 0
                and int(z["off"][n]) == len(z["ints"])
                and len(z["hv"]) == n * num_perm
                and int(z["nperm"]) == num_perm
                and str(z["scheme"]) == scheme
            )
            if not ok:
                return None  # stale shape: recompute
            return z["spans"], z["kinds"], z["normh"], z["off"], z["ints"], z["hv"]
    except Exception:
        return None  # a corrupt or unreadable cache file is recomputed


def _build_lex(cache_dir: str, first_path: str, ext: str, sha: str,
               text: str, num_perm: int, scheme: str):
    """Chunk one base blob, compute its lexical data, and cache it by (blob, ext)."""
    chunks = chunk_file(first_path, text)
    n = len(chunks)
    spans = np.array([[c.start, c.end] for c in chunks], np.int64).reshape(n, 2)
    kinds = np.array([_KIND_CODE[c.kind] for c in chunks], np.int8)
    normh = np.frombuffer(
        b"".join(_norm_hash(c.text) for c in chunks), np.uint8
    ).reshape(n, 20)
    flat = []
    off = [0]
    hvs = []
    for c in chunks:
        lang = _lex_lang(c)
        sh = (
            lexical.shingles(c.text, c.kind)
            if lang is None
            else lexical.shingles(c.text, c.kind, lang)
        )
        ints, hv = lexical.cached_minhash(sh, num_perm)
        flat.extend(ints)
        off.append(len(flat))
        hvs.append(hv)
    ints = np.array(flat, np.uint64).reshape(len(flat))
    off = np.array(off, np.int64)
    hv = np.concatenate(hvs) if hvs else np.zeros(0, np.uint32)
    try:
        _save_cache(
            _lex_cache_path(cache_dir, ext, sha),
            spans=spans, kinds=kinds, normh=normh, off=off,
            ints=ints, hv=hv, nperm=np.int64(num_perm), scheme=scheme,
        )
    except OSError:
        pass  # the cache must never break a run
    return spans, kinds, normh, off, ints, hv


# Findings in these data files never block: strings and numbers tokenize to
# LIT, so same-shape rows with different values score lexical 1.0.
_DATA_EXTS = (".json", ".jsonl", ".ndjson", ".csv", ".tsv")

def _finding(nc, ex: _Ex, match: str, score: float, block: bool) -> dict:
    return {
        "new": {"path": nc.path, "start": nc.start, "end": nc.end, "kind": nc.kind},
        "existing": {"path": ex.path, "start": ex.start, "end": ex.end},
        "match": match,
        "score": float(score),
        "block": bool(block) and not nc.path.lower().endswith(_DATA_EXTS),
    }


def _match(repo: str, new_chunks, entries, removed, deleted, split) -> list[dict]:
    # LSH recall needs the lowest cutoff; the per-kind cut runs at query time.
    lex = LexicalIndex(threshold=min(LEXICAL_ADVISE.values()))
    cache_dir = _cache_dir()
    # Chunking depends on the path extension, so every per-blob record
    # (lexical cache, build, semantic lookup) is keyed by (blob sha, ext).
    first: dict[tuple[str, str], str] = {}
    for path, sha in entries:
        first.setdefault((sha, _ext(path)), path)
    # Per-record lexical data; blob bytes are read only on a cache miss.
    recs: dict[tuple[str, str], tuple] = {}
    missing = []
    new_kinds = {c.kind for c in new_chunks}
    if new_chunks:
        for sha, ext in first:
            r = _load_lex(
                _lex_cache_path(cache_dir, ext, sha), lex.num_perm, lex._tmpl.scheme
            )
            recs[(sha, ext)] = r
            if r is None:
                missing.append((sha, ext))
        if missing:
            data = _cat_blobs(repo, [s for s, _ in missing])
            for sha, ext in missing:
                raw = data.get(sha)
                if raw is None:
                    continue
                recs[(sha, ext)] = _build_lex(
                    cache_dir, first[(sha, ext)], ext, sha,
                    raw.decode("utf-8", "replace"), lex.num_perm, lex._tmpl.scheme,
                )
    existing: list[_Ex] = []
    exact: dict[tuple[str, bytes], int] = {}
    for path, sha in entries:
        r = recs.get((sha, _ext(path)))
        if r is None:
            continue
        spans, kinds, normh, off, ints, hv = r
        counts: dict[str, int] = {}
        rmap = removed.get(path)
        pts = split.get(path)
        for ci in range(len(kinds)):
            kind = _KINDS[kinds[ci]]
            j = counts.get(kind, 0)
            counts[kind] = j + 1
            if kind not in new_kinds:
                continue  # no new chunk of this kind can match it
            start, end = int(spans[ci, 0]), int(spans[ci, 1])
            if (
                path in deleted
                or (rmap and _overlap(start, end, rmap))
                or (pts and any(start <= a < end for a in pts))
            ):
                continue
            i = len(existing)
            np_ = lex.num_perm
            exact.setdefault((kind, normh[ci].tobytes()), i)
            lex.add_cached(
                str(i), kind,
                ints[off[ci]:off[ci + 1]],
                hv[ci * np_:(ci + 1) * np_],
            )
            existing.append(_Ex(path, start, end, kind, sha, j))
    # New chunks also match each other: the chunk earlier in (path, start)
    # order plays `existing` and points at the worktree file.
    new_side = sorted(new_chunks, key=lambda c: (c.path, c.start))
    npos = {id(c): k for k, c in enumerate(new_side)}
    nbase = len(existing)
    for nc in new_side:
        exact.setdefault((nc.kind, _norm_hash(nc.text)), len(existing))
        lex.add(str(len(existing)), nc.text, nc.kind, _lex_lang(nc))
        existing.append(_Ex(nc.path, nc.start, nc.end, nc.kind, "", 0))

    def _ok(nc, i: int) -> bool:
        e = existing[i]
        if e.sha:
            return True  # BASE candidates are always valid
        return i - nbase < npos[id(nc)] and not (
            e.path == nc.path and e.start <= nc.end and nc.start <= e.end
        )
    findings = []
    pending = []
    for nc in new_chunks:
        i = exact.get((nc.kind, _norm_hash(nc.text)))
        if i is not None and _ok(nc, i):
            findings.append(_finding(nc, existing[i], "exact", 1.0, True))
            continue
        hits = [
            h
            for h in lex.query(nc.text, nc.kind, _lex_lang(nc))
            if h[1] >= LEXICAL_ADVISE[nc.kind] and _ok(nc, int(h[0]))
        ]
        if hits:
            i, score = int(hits[0][0]), hits[0][1]
            findings.append(
                _finding(nc, existing[i], "lexical", score, score >= LEXICAL_BLOCK)
            )
            continue
        pending.append(nc)
    by_kind: dict[str, list] = {}
    for nc in pending:
        by_kind.setdefault(nc.kind, []).append(nc)
    for kind, ncs in by_kind.items():
        exs = [e for e in existing if e.kind == kind]
        if not exs:
            continue  # no model load when there is nothing to match against
        res = _load_embedder(kind)
        if res is None:
            continue
        emb, advise, blk = res
        base_exs = [e for e in exs if e.sha]
        new_cols = [c for c in new_side if c.kind == kind]
        mats = []
        if base_exs:
            # Text is needed only here; read it late, in one batch.
            need = sorted({(e.sha, _ext(e.path)) for e in base_exs})
            data = _cat_blobs(repo, [s for s, _ in need])
            blob_chunks = {
                key: chunk_file(first[key], data[key[0]].decode("utf-8", "replace"))
                for key in need
                if key[0] in data
            }
            for key in need:
                blob_chunks.setdefault(key, [])
            mats.append(_kind_matrix(base_exs, blob_chunks, cache_dir, emb))
        if new_cols:
            # Worktree text is already in hand; no cache key exists for it.
            mats.append(
                np.asarray(emb.embed([c.text for c in new_cols]), dtype=np.float32)
            )
        mat = mats[0] if len(mats) == 1 else np.vstack(mats)
        nv = np.asarray(emb.embed([c.text for c in ncs]), dtype=np.float32)
        # New-candidate columns are invalid for self, for a chunk later in
        # (path, start) order, and for a same-file chunk whose span overlaps.
        nb = len(base_exs)
        bad = None
        if new_cols:
            bad = np.zeros((len(ncs), len(new_cols)), dtype=bool)
            for row, nc in enumerate(ncs):
                q = npos[id(nc)]
                for t, c in enumerate(new_cols):
                    bad[row, t] = npos[id(c)] >= q or (
                        c.path == nc.path
                        and c.start <= nc.end
                        and nc.start <= c.end
                    )
        # Score block by block so the full new-by-existing matrix never
        # materializes; the per-new-chunk best score and index stay the same.
        best_score = np.full(len(ncs), -np.inf, dtype=np.float64)
        best_ex = np.zeros(len(ncs), dtype=np.int64)
        for lo in range(0, mat.shape[0], SEM_BLOCK):
            sims = nv @ mat[lo:lo + SEM_BLOCK].T
            if bad is not None and lo + sims.shape[1] > nb:
                s0 = max(nb - lo, 0)
                b0 = max(lo - nb, 0)
                sims[:, s0:] = np.where(
                    bad[:, b0:b0 + sims.shape[1] - s0],
                    -np.inf,
                    sims[:, s0:],
                )
            idx = sims.argmax(axis=1)
            row_best = sims[np.arange(len(ncs)), idx]
            better = row_best > best_score
            best_score[better] = row_best[better]
            best_ex[better] = lo + idx[better]
        for row, nc in enumerate(ncs):
            score = float(best_score[row])
            if score >= advise:
                findings.append(
                    # Semantic matches advise only; only exact and lexical block.
                    _finding(nc, exs[int(best_ex[row])], "semantic", score, False)
                )
    findings.sort(key=lambda f: (not f["block"], -f["score"]))
    return findings[:MAX_FINDINGS]


class _OneLineParser(argparse.ArgumentParser):
    # contract: a bad flag is one stderr line and exit 2, no usage text.
    def error(self, message):
        print(f"dedup-check: {message}", file=sys.stderr)
        raise SystemExit(2)


def main(argv=None) -> int:
    ap = _OneLineParser(prog="dedup-check")
    ap.add_argument("--base", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--fetch-model", action="store_true",
                    help="download the text embedding model and exit")
    args = ap.parse_args(argv)
    _warned.clear()  # one note per run, not one per process
    if args.fetch_model:
        from dedup import embed
        if embed.load("text") is None:
            print(f"dedup-check: model fetch failed:"
                  f" {embed.ERRORS.get('text', 'unknown')}", file=sys.stderr)
            return 2
        print(f"dedup-check: model ready at {embed._model_cache_dir()}")
        return 0
    if not _git_ok(args.repo, "rev-parse", "--git-dir"):
        if args.json:
            print(json.dumps({"base": None, "findings": []}))
        return 0
    try:
        # All git commands and all paths are relative to the top level.
        top = (
            _git(args.repo, "rev-parse", "--show-toplevel")
            .stdout.decode("utf-8", "replace").strip()
        )
        base = _resolve_base(top, args.base)
        added, removed, deleted, split = _diff_ranges(top, base)
        untracked = [
            os.fsdecode(p)
            for p in _git(top, "ls-files", "-z", "--others", "--exclude-standard")
            .stdout.split(b"\0")
            if p
        ]
        entries = _ls_tree(top, base)
        new_chunks = _new_chunks(top, added, untracked)
        findings = _match(top, new_chunks, entries, removed, deleted, split)
    except Exception as e:
        msg = str(e)
        print(f"dedup-check: {msg.splitlines()[0] if msg.strip() else e.__class__.__name__}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"base": base, "findings": findings}))
    else:
        for f in findings:
            n, e = f["new"], f["existing"]
            line = (
                f"{n['path']}:{n['start']}-{n['end']}"
                f" ~ {e['path']}:{e['start']}-{e['end']}"
                f" {f['match']} {f['score']:.3f}"
            )
            if f["block"]:
                line += " BLOCK"
            print(line)
    return 0
