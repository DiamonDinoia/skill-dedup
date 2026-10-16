"""Lexical duplicate detection with MinHash over token shingles."""
import hashlib
import keyword
import re

import numpy as np
from datasketch import MinHash, MinHashLSH

_WORD_RE = re.compile(r"[a-z0-9]+")

_KEYWORDS = frozenset(keyword.kwlist) | frozenset(
    # C/C++/JS keywords
    "auto break case catch char class const consteval constexpr continue default "
    "delete do double else enum explicit export extends extern float for friend goto "
    "if inline int long namespace new noexcept operator private protected public "
    "register requires return short signed sizeof static static_assert struct switch "
    "template this throw try typedef typename union unsigned using virtual void "
    "volatile while let var function typeof instanceof in of yield async await "
    "null nullptr true false boolean".split()
)

_DIRECTIVE_WORDS = frozenset(
    "include define undef if ifdef ifndef elif else endif pragma error warning line".split()
)
_WS = frozenset(chr(i) for i in range(0x3001) if chr(i).isspace())
_IDENT_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_CONT = _IDENT_START | frozenset("0123456789")
_OP_CLASS = frozenset(
    chr(i) for i in range(128) if chr(i) not in _WS and not chr(i).isalnum() and chr(i) != "_"
)


def _scan_string(text: str, i: int, quote: str, qcount: int, multiline: bool) -> int:
    # Past the opening quote(s); returns the index after the closer or -1.
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote and text[i:i + qcount] == quote * qcount:
            return i + qcount
        if c == "\n" and not multiline:
            return -1
        i += 1
    return -1


def _code_tokens(text: str, lang: str | None = None) -> list[str]:
    # One left-to-right hand scan producing the same tokens as the old
    # alternation regex: strings first, then block and line comments,
    # preprocessor directives at line start, '##', hash comments,
    # identifiers/keywords, numeric literals, and operator runs. The regex
    # backtracked exponentially on string literals full of escapes
    # (skills/ste/scripts/ste.py chunked to 89 s); this scan is linear.
    # lang is a lower-case file extension. For Python sources ('.py', '.pyi',
    # '.pyw') '//' and '/*' are operators and '#' always opens a comment, with
    # no directives and no '##'; any other value keeps the C-family rules.
    py = lang in (".py", ".pyi", ".pyw")
    out = []
    i, n = 0, len(text)
    bol = True  # at the start of a line
    dead = {}  # quote kind -> position of the newline that ended its failed scan
    while i < n:
        c = text[i]
        if c == "\n":
            bol = True
            i += 1
            continue
        if c in _WS:
            i += 1
            continue
        if c in "\"'`":
            if text[i:i + 3] in ('"""', "'''"):
                j = _scan_string(text, i + 3, c, 3, True)
                if j != -1:
                    out.append("LIT")
                    i = j
                    bol = False
                    continue
            # A backtick string runs to the next unescaped backtick across
            # newlines (JS templates, markdown fences), as the regex did.
            if c == "`":
                if i < dead.get(c, -1):
                    k = -1
                else:
                    k = _scan_string(text, i + 1, c, 1, True)
                    if k == -1:
                        # A multiline scan fails only at end of text, so every
                        # later backtick fails too. Same memo as the quotes.
                        dead[c] = n
            elif i < dead.get(c, -1):
                k = -1
            else:
                k = _scan_string(text, i + 1, c, 1, False)
                if k == -1:
                    # Every later same-kind quote before this newline is an
                    # escape-second of the failed scan, so its rescan fails
                    # too. Skip the rescans that made this quadratic.
                    eol = text.find("\n", i)
                    dead[c] = n if eol == -1 else eol
            if k != -1:
                out.append("LIT")
                i = k
                bol = False
                continue
            # Unterminated: fall through to the operator run, as the regex did.
        if c == "/" and not py and text[i:i + 2] == "//":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and not py and text[i:i + 2] == "/*":
            j = text.find("*/", i + 2)
            if j != -1:
                i = j + 2
                bol = False
                continue
            # Unterminated block comment: fall through to the operator run.
        if c == "#":
            if not py and bol:
                j = i + 1
                while j < n and text[j] in " \t":
                    j += 1
                k = j
                while k < n and (text[k] in _IDENT_CONT):
                    k += 1
                word = text[j:k]
                if word in _DIRECTIVE_WORDS:
                    out.extend(("#", word))
                    i = k
                    bol = False
                    continue
            if not py and text[i:i + 2] == "##":
                out.append("##")
                i += 2
                bol = False
                continue
            j = text.find("\n", i)
            i = n if j == -1 else j  # hash comment, skipped like //
            continue
        if c in _IDENT_START:
            j = i + 1
            while j < n and text[j] in _IDENT_CONT:
                j += 1
            tok = text[i:j]
            out.append(tok if tok in _KEYWORDS else "ID")
            i = j
            bol = False
            continue
        if c.isdigit():
            j = i + 1
            while j < n and (text[j] in _IDENT_CONT or text[j] == "."):
                j += 1
            # shortcut: all literals map to LIT, so chunks differing only in strings score 1.0; upgrade if string-heavy false positives show up
            out.append("LIT")
            i = j
            bol = False
            continue
        # Operator run, including quotes and '#'-ish starts that matched
        # nothing. The run stops before a quote, '#' and (outside Python) '//'
        # or '/*', so the string and comment branches see their openers at
        # every token boundary.
        j = i + 1
        while j < n and text[j] in _OP_CLASS:
            if text[j] in "\"'`#":
                break
            if not py and text[j:j + 2] in ("//", "/*"):
                break
            j += 1
        out.append(text[i:j])
        i = j
        bol = False
    return out


def shingles(text: str, kind: str, lang: str | None = None) -> set[str]:
    if kind == "code":
        tokens = _code_tokens(text, lang)
        width = 5
    elif kind == "text":
        tokens = _WORD_RE.findall(text.lower())
        width = 3
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    return {" ".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def _shingle_ints(shingle_set: set[str]) -> list[int]:
    # First 8 digest bytes little-endian; the low 32 bits equal datasketch's
    # sha1_hash32 of the shingle bytes.
    return [
        int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "little")
        for s in shingle_set
    ]


def cached_minhash(shingle_set: set[str], num_perm: int):
    """Return (shingle ints, MinHash hashvalues) for one chunk.

    The ints keep exact Jaccard scoring possible from a cache; the hashvalues
    let a later run rebuild the LSH entry without the text.
    """
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode("utf-8") for s in shingle_set])
    return sorted(_shingle_ints(shingle_set)), m.hashvalues


class LexicalIndex:
    def __init__(self, threshold: float = 0.5, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        # Kinds never match each other because each has its own LSH index.
        self._lsh = {
            "code": MinHashLSH(threshold=threshold, num_perm=num_perm),
            "text": MinHashLSH(threshold=threshold, num_perm=num_perm),
        }
        # Shingle sets per (kind, key), for exact Jaccard scoring. The same
        # key may exist once per kind.
        self._sets: dict[tuple[str, str], set[int]] = {}
        self._tmpl = MinHash(num_perm=num_perm)

    def add(self, key: str, text: str, kind: str, lang: str | None = None) -> None:
        # Pass lang only when set: two-argument stubs of shingles keep working.
        sh = shingles(text, kind) if lang is None else shingles(text, kind, lang)
        ints, hv = cached_minhash(sh, self.num_perm)
        self.add_cached(key, kind, ints, hv)

    def add_cached(self, key: str, kind: str, ints, hashvalues) -> None:
        """Insert a chunk from precomputed shingle ints and MinHash values."""
        if (kind, key) in self._sets:
            # Re-adding a key replaces the old entry.
            self._lsh[kind].remove(key)
            del self._sets[(kind, key)]
        if not len(ints):
            return
        m = MinHash(
            hashvalues=np.asarray(hashvalues, dtype=np.uint32),
            permutations=self._tmpl.permutations,
            scheme=self._tmpl.scheme,
        )
        self._lsh[kind].insert(key, m)
        self._sets[(kind, key)] = set(int(int(v)) for v in ints)

    def query(self, text: str, kind: str, lang: str | None = None) -> list[tuple[str, float]]:
        # Pass lang only when set: two-argument stubs of shingles keep working.
        shingle_set = shingles(text, kind) if lang is None else shingles(text, kind, lang)
        if not shingle_set:
            return []
        ints, hv = cached_minhash(shingle_set, self.num_perm)
        m = MinHash(
            hashvalues=hv,
            permutations=self._tmpl.permutations,
            scheme=self._tmpl.scheme,
        )
        qset = set(ints)
        hits = []
        # shortcut: candidate recall is bounded by LSH, so a pair just above the threshold can be missed; upgrade by lowering the LSH threshold below the reporting threshold if recall matters
        # shortcut: the LSH candidate list is uncapped, revisit if a query
        # against a big index gets slow
        for key in self._lsh[kind].query(m):
            other = self._sets[(kind, key)]
            # MinHash is only the candidate filter. The reported score is the
            # exact Jaccard of the shingle sets.
            shared = len(qset & other)
            score = shared / (len(qset) + len(other) - shared)
            if score >= self.threshold:
                hits.append((key, score))
        hits.sort(key=lambda kv: kv[1], reverse=True)
        return hits
