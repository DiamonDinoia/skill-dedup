#!/usr/bin/env python3
"""Regenerate eval/pairs/pairs.jsonl: renamed sides and the group split.

Reads the committed pairs.jsonl. Rewrites each absolute source path as a path
relative to the parent of its corpus repo (the path parts after "<group>",
below the .../repos root), for example `skill-ste/skills/ste/SKILL.md`.
Regenerates every renamed pair b from its a:
only user identifiers are renamed (never keywords, stdlib or builtin names,
ALL_CAPS macros, or module names), each to a natural word, with no generated
suffix. Assigns every whole source file to dev or test with one seeded search,
so no file sits in both splits and each subset keeps 40-60% of its pairs in dev.
Then rewrites pairs.jsonl. Run: python eval/pairs/build/build_pairs.py
"""
import builtins
import io
import json
import keyword
import random
import re
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from validate import TOK_RE, renamed_structure_error  # noqa: E402

PAIRS = HERE.parent / "pairs.jsonl"

BAD_SUFFIX = re.compile(r"[A-Za-z_]\w*X\dq\b")

PYSTOP = set(keyword.kwlist) | set(dir(builtins)) | {
    "self", "cls", "np", "pd", "plt", "os", "sys", "re", "io", "json", "math",
    "time", "pytest", "torch", "tf", "nn", "F", "np2", "Path"}
CPPSTOP = set("""alignas alignof and asm auto bitand bitor bool break case catch char
char8_t char16_t char32_t class compl concept const consteval constexpr constinit
const_cast continue co_await co_return co_yield decltype default delete do double
dynamic_cast else enum explicit export extern false final float for friend goto if
inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq
override private protected public register reinterpret_cast requires return short
signed sizeof static static_assert static_cast struct switch template this
thread_local throw true try typedef typeid typename union unsigned using virtual
void volatile wchar_t while xor xor_eq
std size_t ptrdiff_t ssize_t intptr_t uintptr_t int8_t int16_t int32_t int64_t
uint8_t uint16_t uint32_t uint64_t double_t float_t complex imaginary
printf fprintf sprintf snprintf scanf fscanf malloc calloc realloc free memcpy
memmove memset memcmp strlen strcmp strncmp strcpy strncpy strcat strdup strchr
abs labs fabs sqrt pow exp log log2 floor ceil round fmin fmax min max hypot
assert errno FILE NULL nullptr_t va_list va_start va_end va_arg div_t clock_t time_t
string wstring string_view vector map set unordered_map unordered_set pair tuple
array deque list bitset optional variant shared_ptr unique_ptr weak_ptr make_pair
make_shared make_unique move forward swap begin end rbegin rend sort find count
accumulate iota iterator ostream istream iostream cout cin cerr endl stringstream
ostringstream istringstream exception runtime_error logic_error out_of_range
length_error invalid_argument overflow_error numeric_limits allocator function
bind ref cref less greater hash equal_to void_t enable_if conditional is_same
integral_constant true_type false_type decay remove_reference remove_const
initializer_list reverse_iterator size begin_data FLT CPX BIGINT VEC fftw_plan
Sleep MessageBoxA MessageBoxW LPVOID DWORD WINAPI HRESULT BOOL BYTE WORD
__m128 __m128d __m128i __m256 __m256d __m256i""".split())

# Frequent-identifier word list for natural replacements.
WORDS = """account axis batch block board border branch bridge buffer button cache
canvas cell center chain chamber channel charge chart check chip circle
clause client clock cluster coast code coil column compass convoy copy corner
count course cover crane credit curve cycle damper dealer deck degree delta
depth device diagram digit domain donor draft driver edge effort entry fabric
factor feather field figure filter finger flag flight floor folder format
frame funnel garden gauge glass grade grain graph grid ground group guard guide
hammer handle harbor header hearth height helper hinge holder hook horizon host
hotel hour house hub hunter index input item jacket joint journey judge kernel
kettle ladder lane laptop layer ledger length letter level lever library limit
linen list loader lobby lock lodge logic loop lumber magnet manner marble margin
marker master match matrix meadow measure member memory menu merchant mesh meter
method mirror model module moment monitor morning motion motor mount mouse needle
network node notebook notice number nurse object ocean offset onion option orbit
order origin outlet output owner packet paddle page painter panel paper parcel
parent parse part party patch path pattern pause pearl pencil period permit phase
phone photo piece pilot pillar pilot pitch place plain plane planet plate platform
player plot pocket point policy portal position post power prefix price prime
print prison probe product profile program project prompt proof pulse pump puzzle
quarter queue radar rail range rank rate ratio reader reason record region register
relay remark reply report rescue result rhythm ridge ring river rocket role roof
room root rope rotor round route row ruler runner saddle sale sample scale scanner
scene scheme school scope score scrap screen script search season seat section
sector seed segment sensor series server session shade shadow shaft shape share
sheet shell shift ship shop shore shoulder signal silver singer sketch sleeve
slide slot socket soldier solution sonata sort source space span spare speaker
sphere spirit sponsor spot spring square stack stage stamp stand star state
station status steam steel step stone storage store storm story strand stream
street string strip structure studio style subject sugar sum summit supply survey
switch symbol system table talent target task teacher temple tenant tenor term
text theater theme thread ticket tier tile timber timer title toast token tool
topic tower track trade trailer train travel tray treaty tree trial truck tunnel
turn tutor type union unit update valley value vault vendor verse vessel view
village virtue visitor volume voter wagon walker wall water wave wealth weather
weight wheel widow width window winner wire wonder word work worker wound writer
yard year yield zone""".split()
WORDS = [w for w in WORDS if w]

SYN = {"count": "total", "cnt": "count", "num": "count", "n": "count",
       "i": "index", "j": "index", "k": "index", "idx": "pos", "pos": "offset",
       "tmp": "temp", "temp": "scratch", "val": "value", "ret": "result",
       "res": "result", "buf": "buffer", "cfg": "config", "args": "params",
       "kwargs": "options", "ctx": "context", "acc": "total", "cur": "current",
       "prev": "previous", "dst": "target", "src": "source", "sz": "size",
       "ptr": "pointer", "fn": "func", "cb": "callback", "msg": "message",
       "err": "error", "req": "request", "resp": "response", "descr": "summary",
       "desc": "summary", "init": "setup", "calc": "compute", "arr": "array",
       "vec": "vector", "mat": "matrix", "elem": "element", "tot": "total",
       "len": "length", "x": "point", "c": "coeff", "t": "step", "rng": "gen"}


def style_of(name):
    if "_" in name:
        return "snake"
    if name[0].isupper():
        return "pascal"
    if name.lower() != name:
        return "camel"
    return "lower"


def build_mapping(names, text, stop, salt):
    """Map each name to a natural word. Deterministic; avoids every identifier
    already present in text (regex token view) so the map stays one-to-one."""
    used = {m.group() for m in TOK_RE.finditer(text) if m.lastgroup == "ident"}
    used |= stop
    mapping = {}
    wi = salt
    for name in names:
        cands = []
        if name in SYN:
            cands.append(SYN[name])
        st = style_of(name)
        k = 0
        while len(cands) < 400:
            w1 = WORDS[(wi + k * 37) % len(WORDS)]
            w2 = WORDS[(wi + k * 37 + 53 + k) % len(WORDS)]
            k += 1
            if st == "snake":
                cands.append(f"{w1}_{w2}")
            elif st == "pascal":
                cands.append(w1.capitalize() + w2.capitalize())
            elif st == "camel":
                cands.append(w1 + w2.capitalize())
            else:
                cands.append(w1)
        for cand in cands:
            if cand in used or cand in mapping.values():
                continue
            if cand in stop or cand == name or BAD_SUFFIX.match(cand):
                continue
            if keyword.iskeyword(cand):
                continue
            mapping[name] = cand
            wi += 1
            break
        else:
            raise SystemExit(f"no replacement found for {name!r}")
    return mapping


def replace_spans(text, edits):
    """edits: list of (start, end, new) absolute offsets, source order."""
    out, pos = [], 0
    for s, e, new in edits:
        out.append(text[pos:s])
        out.append(new)
        pos = e
    out.append(text[pos:])
    return "".join(out)


if not hasattr(tokenize, "FSTRING_START"):
    raise SystemExit("build_pairs.py needs Python 3.12+ (tokenize.FSTRING_START); "
                     "without it, references inside f-strings would not be renamed")
FSTRING_START = tokenize.FSTRING_START
FSTRING_END = tokenize.FSTRING_END


def rename_py(text, salt):
    lines = text.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    fstr = set()
    depth = 0
    import_words = set()
    comment_words = set()
    for i, t in enumerate(toks):
        if FSTRING_START is not None and t.type == FSTRING_START:
            depth += 1
        elif FSTRING_END is not None and t.type == FSTRING_END:
            depth = max(0, depth - 1)
        elif t.type == tokenize.NAME and depth:
            fstr.add(t.string)
        if t.type == tokenize.COMMENT:
            comment_words |= set(re.findall(r"[A-Za-z_]\w*", t.string))
        if t.type == tokenize.NAME and t.string in ("import", "from"):
            row = t.start[0]
            if lines[row - 1].lstrip().startswith(("import ", "from ")):
                import_words |= set(re.findall(r"[A-Za-z_]\w*", lines[row - 1]))
                import_words -= {"import", "from", "as"}
    prev_significant = None
    occ = []  # (name, start, end, ok)
    depth = 0
    for t in toks:
        if FSTRING_START is not None and t.type == FSTRING_START:
            depth += 1
            continue
        if FSTRING_END is not None and t.type == FSTRING_END:
            depth = max(0, depth - 1)
            continue
        if t.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER):
            if t.type == tokenize.NAME and not depth:
                n = t.string
                ok = (
                    n not in PYSTOP
                    and n not in import_words
                    and n not in comment_words
                    and n not in fstr
                    and n != "_"
                    and not n.isupper()
                    and not (n.startswith("__") and n.endswith("__"))
                    and prev_significant != "."
                )
                after = text[offs[t.end[0] - 1] + t.end[1]:]
                if after.lstrip()[:1] == ".":
                    ok = False
                occ.append([n, offs[t.start[0] - 1] + t.start[1],
                            offs[t.end[0] - 1] + t.end[1], ok])
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                          tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING):
            prev_significant = t.string
    # A name skipped anywhere is skipped everywhere, so the map stays consistent.
    bad = {n for n, _, _, ok in occ if not ok}
    cand = [(n, s, e) for n, s, e, ok in occ if n not in bad]
    names = []
    for n, _, _ in cand:
        if n not in names:
            names.append(n)
    if not names:
        return text
    mapping = build_mapping(names, text, PYSTOP, salt)
    edits = [(s, e, mapping[n]) for n, s, e in cand if n in mapping]
    return replace_spans(text, edits)


CPP_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
CPP_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def rename_cpp(text, salt):
    prot = [m.span() for m in CPP_COMMENT.finditer(text)]
    prot += [m.span() for m in CPP_STRING.finditer(text)]

    def protected(s, e):
        return any(s < pe and e > ps for ps, pe in prot)

    comment_words = set()
    for m in CPP_COMMENT.finditer(text):
        comment_words |= set(re.findall(r"[A-Za-z_]\w*", m.group()))
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            comment_words |= set(re.findall(r"[A-Za-z_]\w*", ln))

    occ = []
    for m in re.finditer(r"(?<!\w)[A-Za-z_]\w*", text):
        n = m.group()
        s, e = m.span()
        ok = not protected(s, e) and n not in CPPSTOP and not n.isupper() \
            and n not in comment_words
        before = text[:s].rstrip()
        after = text[e:].lstrip()
        if before.endswith((".", "->", "::")) or after.startswith("::"):
            ok = False
        occ.append([n, s, e, ok])
    # A name skipped anywhere is skipped everywhere, so the map stays consistent.
    bad = {n for n, _, _, ok in occ if not ok}
    cand = [(n, s, e) for n, s, e, ok in occ if n not in bad]
    names = []
    for n, _, _ in cand:
        if n not in names:
            names.append(n)
    if not names:
        return text
    mapping = build_mapping(names, text, CPPSTOP, salt)
    edits = [(s, e, mapping[n]) for n, s, e in cand if n in mapping]
    return replace_spans(text, edits)


def _rel_source(r):
    """Absolute /.../repos/<group>/<repo>/PATH -> <repo>/PATH, keeping the span.

    The corpus repos sit at /.../repos/<group>/<repo>, so the path after
    <group> is relative to the parent of the corpus repo. Already-relative
    sources pass through unchanged.
    """
    path, sep, span = r["source"].rpartition(":")
    parts = path.split("/")
    if "repos" in parts and len(parts) > parts.index("repos") + 2:
        i = parts.index("repos")
        path = "/".join(parts[i + 2:])
    r = dict(r, source=path + sep + span)
    return r


def assign_splits(rows):
    """Whole-file dev/test assignment; each subset keeps 40-60% dev."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["kind"], r["subset"])].append(r)
    rng = random.Random(11)
    for attempt in range(5000):
        assign = {}
        for key in sorted(buckets):
            bucket = buckets[key]
            n = len(bucket)
            dev = 0
            for r in bucket:
                f = r["source"].split(":")[0]
                if f not in assign:
                    assign[f] = "dev" if rng.random() < 0.5 else "test"
                dev += assign[f] == "dev"
            if not 0.4 <= dev / n <= 0.6:
                break
        else:
            return assign
    raise SystemExit("no valid split found in 5000 tries")


def main():
    rows = [json.loads(l) for l in PAIRS.open() if l.strip()]
    rows = [_rel_source(r) for r in rows]
    for nth, r in enumerate(r for r in rows if r["subset"] == "renamed"):
        salt = 1 + nth * 97
        b = rename_py(r["a"], salt) if r["lang"] == "python" else rename_cpp(r["a"], salt)
        if b == r["a"]:
            raise SystemExit(f"{r['id']}: nothing renamed")
        err = renamed_structure_error(r["a"], b)
        if err:
            raise SystemExit(f"{r['id']}: self-check failed: {err}")
        n_a = {m.group() for m in TOK_RE.finditer(r["a"]) if m.lastgroup == "ident"}
        n_b = {m.group() for m in TOK_RE.finditer(b) if m.lastgroup == "ident"}
        changed = n_a - n_b
        if len(changed) < 1:
            raise SystemExit(f"{r['id']}: no identifiers renamed")
        print(f"{r['id']}: {len(changed)} identifiers renamed")
        r["b"] = b
    assign = assign_splits(rows)
    for r in rows:
        r["split"] = assign[r["source"].split(":")[0]]
    out = "".join(json.dumps(r) + "\n" for r in rows)
    PAIRS.write_text(out)
    print(f"wrote {len(rows)} pairs to {PAIRS}")


if __name__ == "__main__":
    main()
