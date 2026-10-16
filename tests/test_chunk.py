from dedup.chunk import MIN_CODE_LINES, _window, chunk_file

PY_SRC = """\
import os


def top(a):
    x = a + 1
    y = x * 2
    z = y - 3
    return x + y + z


class A:
    a1 = 1
    a2 = 2
    a3 = 3
    a4 = 4

    def m1(self, q):
        a = q
        b = a + 1
        c = b + 2
        return c

    def m2(self):
        u = 1
        v = 2
        w = 3
        return u + v + w
"""


def spans(chunks):
    return sorted((c.start, c.end, c.kind, c.lang) for c in chunks)


def test_python_functions_methods_and_class_body():
    chunks = chunk_file("m.py", PY_SRC)
    assert spans(chunks) == [
        (4, 8, "code", "python"),     # top
        (11, 16, "code", "python"),   # class frame before the methods
        (17, 21, "code", "python"),   # m1
        (23, 27, "code", "python"),   # m2
    ]
    top = next(c for c in chunks if c.start == 4)
    assert top.text.startswith("def top(a):")


CPP_SRC = """\
#include <string>

std::string s1() {
    std::string a = "{not a brace";
    std::string b = "ok";
    std::string c = a;
    return a + b + c;
}

int s2() {
    int x = 1;
    int y = 2;
    int z = x + y;
    return z;
}
"""


def test_cpp_two_functions_brace_in_string():
    chunks = chunk_file("s.cpp", CPP_SRC)
    assert spans(chunks) == [
        (3, 8, "code", "cpp"),
        (10, 15, "code", "cpp"),
    ]


PARA1 = ("Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango.")
PARA2 = ("One two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty.")

MD_SRC = f"""\
# Title

{PARA1}

{PARA2}

```python
v = 0
w = 0
x = 1
y = 2
z = 3
```
"""


def test_markdown_heading_paragraphs_and_fence():
    chunks = chunk_file("d.md", MD_SRC)
    assert spans(chunks) == [
        (3, 3, "text", "markdown"),
        (5, 5, "text", "markdown"),
        (7, 13, "code", "python"),
    ]
    fence = next(c for c in chunks if c.kind == "code")
    assert fence.lang == "python"
    assert (fence.start, fence.end) == (7, 13)  # span covers the whole fence
    assert fence.text == "v = 0\nw = 0\nx = 1\ny = 2\nz = 3"


def test_short_chunk_is_skipped():
    assert chunk_file("s.py", "def f():\n    return 1\n") == []


def test_window_no_contained_duplicates():
    for n in (28, 40, 41, 60, 100):
        lines = [f"line {i}" for i in range(n)]
        wins = [(c.start, c.end) for c in _window("w.other", lines, 1, n, "other")]
        for i, (s, e) in enumerate(wins):
            assert e - s + 1 <= 40, (n, s, e)
            for j, (s2, e2) in enumerate(wins):
                if i != j:
                    assert not (s2 <= s and e <= e2), (n, (s, e), (s2, e2))
        covered = set()
        for s, e in wins:
            covered.update(range(s, e + 1))
        assert covered == set(range(1, n + 1)), n

def test_nul_byte_returns_empty():
    assert chunk_file("x.py", "a\x00b") == []


CLASS_SRC = """\
class B:
    x = 1
    y = 2
    z = 3
    w = 4
    v = 5

    def m(self):
        a = 1
        b = 2
        c = 3
        return a + b + c

    tail = 1
    t2 = 2
    t3 = 3
"""


def test_class_frame_chunks_match_their_lines():
    chunks = chunk_file("m.py", CLASS_SRC)
    lines = CLASS_SRC.splitlines()
    for c in chunks:
        assert c.text == "\n".join(lines[c.start - 1:c.end])
    assert spans(chunks) == [
        (1, 7, "code", "python"),    # class line plus attributes
        (8, 12, "code", "python"),   # method m
    ]
    frame = next(c for c in chunks if c.start == 1)
    assert frame.text.splitlines()[0] == "class B:"
    assert "def m" not in frame.text


MOD_SRC = """\
import os
import re
import sys
import json
import math
import pathlib


def f():
    return 1


if __name__ == "__main__":
    total = 0
    for name in (os, re, sys):
        total += 1
    print(total)
"""


def test_module_level_lines_are_chunked():
    chunks = chunk_file("m.py", MOD_SRC)
    body = "\n".join(c.text for c in chunks)
    assert "import pathlib" in body
    assert 'if __name__ == "__main__":' in body


DEC_SRC = """\
_registry = []


def register(cls):
    _registry.append(cls)
    return cls


@register
class C:
    a = 1
    b = 2
    c = 3
    d = 4
"""


def test_class_decorator_line_is_in_frame():
    chunks = chunk_file("m.py", DEC_SRC)
    frame = next(c for c in chunks if "class C:" in c.text)
    assert frame.text.splitlines()[0] == "@register"


FANCY_MD = """\
# Doc

```python title=x
a1 = 1
a2 = 2
a3 = 3
a4 = 4
a5 = 5
```

  ```python
b1 = 1
b2 = 2
b3 = 3
b4 = 4
b5 = 5
```
"""


def test_fence_info_words_and_indent():
    chunks = chunk_file("d.md", FANCY_MD)
    fences = [c for c in chunks if c.kind == "code" and c.lang == "python"]
    assert len(fences) == 2


STRAY_CPP = """\
}

int f() {
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    return a + b + c + d;
}
"""


def test_stray_brace_does_not_lose_function():
    chunks = chunk_file("s.cpp", STRAY_CPP)
    assert (3, 9, "code", "cpp") in spans(chunks)


C3_SRC = """\
int f1(int a)
{
    int x = a + 1;
    int y = x * 2;
    return x + y;
}
int f2(int b)
{
    int u = b - 1;
    int v = u + 3;
    return u * v;
}
int f3(int c)
{
    int p = c * 2;
    int q = p + 1;
    return p - q;
}
"""


def test_adjacent_c_functions_no_blank_lines():
    chunks = chunk_file("t.c", C3_SRC)
    lines = C3_SRC.splitlines()
    assert spans(chunks) == [
        (1, 6, "code", "c"),
        (7, 12, "code", "c"),
        (13, 18, "code", "c"),
    ]
    for c in chunks:
        sigs = [s for s in ("int f1", "int f2", "int f3") if s in c.text]
        assert len(sigs) == 1, c.text
        assert c.text == "\n".join(lines[c.start - 1:c.end])


FENCE_PY_SRC = """\
Some intro words here.

```python
def add(a, b):
    total = a + b
    total += 1
    total *= 2
    return total
```
"""

TILDE_UNCLOSED_SRC = """\
~~~c
int g(void)
{
    int r = 1;
    int s = 2;
    return r + s;
}"""


def test_fenced_python_text_excludes_fence_lines():
    chunks = chunk_file("d.md", FENCE_PY_SRC)
    fence = next(c for c in chunks if c.kind == "code")
    assert (fence.start, fence.end) == (3, 9)  # span covers the fence lines
    assert fence.text == (
        "def add(a, b):\n    total = a + b\n    total += 1\n"
        "    total *= 2\n    return total"
    )


def test_tilde_and_unclosed_fence_text_excludes_fence_lines():
    fence = chunk_file("d.md", TILDE_UNCLOSED_SRC)[0]
    assert (fence.start, fence.end) == (1, 7)
    assert fence.text == "int g(void)\n{\n    int r = 1;\n    int s = 2;\n    return r + s;\n}"
    assert fence.lang == "c"


def test_r1_and_coverage_on_stdlib():
    import importlib.util

    for name in ("ast", "difflib"):
        origin = importlib.util.find_spec(name).origin
        with open(origin) as fh:
            text = fh.read()
        lines = text.splitlines()
        chunks = chunk_file(name + ".py", text)
        for c in chunks:  # R1: every chunk is one contiguous line range
            assert c.text == "\n".join(lines[c.start - 1:c.end]), (name, c.start)
        covered = [False] * (len(lines) + 1)
        for c in chunks:
            for k in range(c.start, c.end + 1):
                covered[k] = True
        run = 0  # non-blank lines in the current uncovered run
        for ln in range(1, len(lines) + 2):
            if ln <= len(lines) and not covered[ln]:
                if lines[ln - 1].strip():
                    run += 1
            else:
                assert run < MIN_CODE_LINES, (name, ln)
                run = 0
