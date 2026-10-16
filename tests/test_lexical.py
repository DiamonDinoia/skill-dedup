"""Tests for the MinHash lexical stage."""
import re
import time

from dedup.lexical import _KEYWORDS, LexicalIndex, _code_tokens, shingles

CODE_A = """
def compute_total(price, count):
    subtotal = price * count
    tax = subtotal * 0.2
    return subtotal + tax
"""

# Identical structure, every identifier renamed; literals and keywords unchanged.
CODE_B = """
def sum_amount(rate, units):
    base = rate * units
    fee = base * 0.2
    return base + fee
"""

UNRELATED = """
import os
for path in os.listdir("/tmp"):
    if os.path.isdir(path):
        print(path)
"""

_WORDS = [f"word{i:02d}" for i in range(60)]
TEXT_A = "Sample report. " + " ".join(_WORDS) + " End."
_EDITED = _WORDS.copy()
_EDITED[10] = "swappedone"
_EDITED[40] = "swappedtwo"
TEXT_B = "sample report " + " ".join(_EDITED) + " end"


def _only(scores, key):
    for k, _ in scores:
        if k != key:
            raise AssertionError(f"unexpected match {k!r} in {scores}")
    return dict(scores)[key]


def test_exact_copy():
    idx = LexicalIndex(threshold=0.5)
    idx.add("a", CODE_A, "code")
    score = _only(idx.query(CODE_A, "code"), "a")
    assert score >= 0.95, score


def test_renamed_copy():
    idx = LexicalIndex(threshold=0.5)
    idx.add("a", CODE_A, "code")
    score = _only(idx.query(CODE_B, "code"), "a")
    assert score >= 0.8, score


def test_unrelated_absent():
    idx = LexicalIndex(threshold=0.5)
    idx.add("a", CODE_A, "code")
    scores = idx.query(UNRELATED, "code")
    assert "a" not in dict(scores), scores


def test_light_text_edit():
    idx = LexicalIndex(threshold=0.5)
    idx.add("t", TEXT_A, "text")
    score = _only(idx.query(TEXT_B, "text"), "t")
    assert score >= 0.7, score


def test_kinds_never_match():
    idx = LexicalIndex(threshold=0.1)
    idx.add("c", CODE_A, "code")
    idx.add("t", TEXT_A, "text")
    assert idx.query(TEXT_A, "code") == []
    assert idx.query(CODE_A, "text") == []


def test_empty_shingles():
    idx = LexicalIndex()
    idx.add("empty", "x", "code")  # one token, no 5-token windows
    assert idx.query("x", "code") == []


# Differ only in the #include lines: B has one extra include.
C_INC_A = """\
#include <stdio.h>
int main(void) { printf("hi there\\n"); return 0; }
"""
C_INC_B = """\
#include <stdio.h>
#include <stdlib.h>
int main(void) { printf("hi there\\n"); return 0; }
"""

PY_COMMENT = """\
# local helper, not code
x = 1
y = x + 2
z = y * 3
w = z - 1
v = w + x
"""
PY_PLAIN = PY_COMMENT.replace("# local helper, not code\n", "")


def _exact(a, b, kind):
    sa, sb = shingles(a, kind), shingles(b, kind)
    assert sa and sb
    return len(sa & sb) / len(sa | sb)


def test_readd_replaces_entry():
    idx = LexicalIndex(threshold=0.5)
    idx.add("k", CODE_A, "code")
    idx.add("k", UNRELATED, "code")
    assert "k" not in dict(idx.query(CODE_A, "code"))
    assert "k" in dict(idx.query(UNRELATED, "code"))


def test_hash_comments_keep_directives():
    idx = LexicalIndex(threshold=0.5)
    idx.add("a", C_INC_A, "code")
    idx.add("b", C_INC_B, "code")
    scores = dict(idx.query(C_INC_A, "code"))
    assert scores["a"] == 1.0
    assert "b" in scores
    assert scores["b"] < 0.95, scores
    # A plain Python comment is still stripped.
    idx2 = LexicalIndex(threshold=0.5)
    idx2.add("p", PY_COMMENT, "code")
    assert dict(idx2.query(PY_PLAIN, "code"))["p"] == 1.0


def test_scores_are_exact_jaccard():
    # 60 words each, 38 shared 3-grams: exact Jaccard 38/78 = 0.487,
    # just below the threshold, close enough that MinHash estimates can
    # land above it.
    a = " ".join(f"a{i:02d}" for i in range(60))
    b = " ".join([f"a{i:02d}" for i in range(40)] + [f"b{i:02d}" for i in range(20)])
    exact_ab = _exact(a, b, "text")
    assert exact_ab < 0.5, exact_ab
    idx = LexicalIndex(threshold=0.5)
    idx.add("a", a, "text")
    assert "a" not in dict(idx.query(b, "text"))
    # Returned scores are the exact Jaccard, not the MinHash estimate.
    idx.add("t", TEXT_A, "text")
    scores = dict(idx.query(TEXT_B, "text"))
    assert scores["t"] == _exact(TEXT_A, TEXT_B, "text")


def test_same_key_in_both_kinds():
    idx = LexicalIndex(threshold=0.5)
    idx.add("k", CODE_A, "code")
    idx.add("k", TEXT_A, "text")
    assert dict(idx.query(CODE_A, "code"))["k"] == 1.0
    assert dict(idx.query(TEXT_A, "text"))["k"] == 1.0


CODE_TOKEN_ROWS = [
    # '#' or '//' inside a string literal are not comments.
    ('x = "a # b" + y', ["ID", "=", "LIT", "+", "ID"]),
    ('f"val={a# }" # t', ["ID", "LIT"]),
    ('const char* s = "#include";', ["const", "char", "*", "ID", "=", "LIT", ";"]),
    # '##' in a string is not the token-paste operator.
    ('x = "a ## b # c"', ["ID", "=", "LIT"]),
    # A one-line block comment with '#' inside is fully skipped.
    ("/* note # hash */", []),
    # A preprocessor directive keeps '#' and the directive word.
    ("#include <stdio.h>", ["#", "include", "<", "ID", ".", "ID", ">"]),
    # The '##' token-paste operator in a macro body survives the scan.
    ("#define CAT(a, b) a ## b",
     ["#", "define", "ID", "(", "ID", ",", "ID", ")", "ID", "##", "ID"]),
    # A plain Python comment line yields nothing.
    ("# comment", []),
    ("s = 'it''s'", ["ID", "=", "LIT", "LIT"]),
    # '//' inside a string stays one literal.
    ('x = "a // b"', ["ID", "=", "LIT"]),
    # A multi-line block comment does not eat the code after it.
    ("/* a\n * b\n * c */\ny = 2", ["ID", "=", "LIT"]),
    # A triple-quoted string with '#' inside stays one literal.
    ('t = """a # b\nc"""', ["ID", "=", "LIT"]),
    # // inside a JS backtick string stays one literal.
    ("const t = `a // b`;", ["const", "ID", "=", "LIT", ";"]),
    # A JS template literal spanning three lines, with '//' and '#' inside,
    # is one literal, as the 8868bb7 regex tokenized it.
    ("const t = `a // b\nc # d\ne`;", ["const", "ID", "=", "LIT", ";"]),
    # An escaped backtick inside a backtick string does not close it.
    ("`a\\`b`", ["LIT"]),
]


# The 8868bb7 regex tokenizer, kept as ground truth for string edge cases.
_OLD_TOKEN_RE = re.compile(
    r'''
      (?P<string>
          """(?:\\.|(?!""")[\s\S])*"""
        | \'\'\'(?:\\.|(?!\'\'\')[\s\S])*\'\'\'
        | "(?:\\.|[^"\\\n])*"
        | '(?:\\.|[^'\\\n])*'
        | `(?:\\.|[^`\\])*`
      )
    | (?P<block>/\*.*?\*/)
    | (?P<comment>//[^\n]*)
    | (?P<directive>
          ^[ \t]*\#[ \t]*
          (?P<dword>include|define|undef|if|ifdef|ifndef|elif|else|endif
                   |pragma|error|warning|line)\b
      )
    | (?P<paste>\#\#)
    | (?P<hash>\#[^\n]*)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<number>\d[\w.]*)
    | (?P<op>[^\sA-Za-z0-9_]+)
    ''',
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)


def _old_code_tokens(text):
    out = []
    for m in _OLD_TOKEN_RE.finditer(text):
        g = m.groupdict()
        if g["string"] or g["number"]:
            out.append("LIT")
        elif g["directive"]:
            out.extend(("#", g["dword"]))
        elif g["ident"]:
            tok = g["ident"]
            out.append(tok if tok in _KEYWORDS else "ID")
        elif g["paste"] or g["op"]:
            out.append(m.group())
    return out


def test_code_tokens_unterminated_backtick():
    # An unterminated backtick string must finish in linear time and give
    # the same tokens as the 8868bb7 regex (the backtick as an operator).
    src = "`unterminated\n" + "".join(
        f"x{i} = compute(y{i}, z{i})\n" for i in range(200)
    )
    t0 = time.perf_counter()
    got = _code_tokens(src)
    assert time.perf_counter() - t0 < 0.5
    assert got == _old_code_tokens(src)


def test_code_tokens_unterminated_string_linear():
    # '\"' plus 100k escaped quotes: the old fallback restarts the failed
    # scan at every quote and rescans the suffix, which is quadratic.
    src = '"' + '\\"' * 100000
    for lang in (None, ".py"):
        t0 = time.perf_counter()
        _code_tokens(src, lang)
        assert time.perf_counter() - t0 < 2.0


def test_code_tokens_unterminated_escaped_backticks_linear():
    # Backticks share the dead memo with the quotes: without it each escaped
    # backtick restarts the failed scan and rescans the suffix (quadratic;
    # m=16000 took 11.5 s before the memo).
    src = "`" + "x\\`" * 16000
    t0 = time.perf_counter()
    _code_tokens(src, None)
    assert time.perf_counter() - t0 < 2.0


def test_code_tokens():
    for src, expected in CODE_TOKEN_ROWS:
        assert _code_tokens(src) == expected, src


CODE_LANG_TOKEN_ROWS = [
    # Spacing between an operator run and a string opener is invisible.
    ('f("hello")', None, ["ID", "(", "LIT", ")"]),
    ('f( "hello" )', None, ["ID", "(", "LIT", ")"]),
    # A comment starting right after an operator is still a comment.
    ("x=1;//c", None, ["ID", "=", "LIT", ";"]),
    ("x=1;/*c*/", None, ["ID", "=", "LIT", ";"]),
    # Python: '//' and '//=' are operators, every token after them stays.
    ("return a // b if c else d", ".py",
     ["return", "ID", "//", "ID", "if", "ID", "else", "ID"]),
    # lang=None keeps the C-family rules: '//' opens a line comment.
    ("return a // b if c else d", None, ["return", "ID"]),
]


def test_code_tokens_lang():
    for src, lang, expected in CODE_LANG_TOKEN_ROWS:
        assert _code_tokens(src, lang) == expected, (src, lang)


def test_empty_readd_removes_old():
    idx = LexicalIndex(threshold=0.5)
    idx.add("k", CODE_A, "code")
    # One token: no 5-token window, so the shingle set is empty.
    idx.add("k", "x", "code")
    assert idx.query(CODE_A, "code") == []
