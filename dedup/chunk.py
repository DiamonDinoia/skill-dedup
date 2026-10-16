"""Split a source file into chunks for duplicate detection."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass

WINDOW = 40
STEP = 20
MIN_CODE_LINES = 5
MIN_TEXT_WORDS = 20

_EXT_LANG = {
    ".py": "python",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".js": "js", ".jsx": "js", ".mjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".go": "go",
    ".rs": "rust",
    ".sh": "shell", ".bash": "shell",
}
_MD_EXTS = {".md", ".rst", ".txt"}
_BRACE_LANGS = {"c", "cpp", "js", "ts", "go", "rust"}

_FENCE_ALIAS = {
    "py": "python", "python3": "python",
    "c++": "cpp", "cplusplus": "cpp",
    "javascript": "js", "typescript": "ts", "golang": "go", "rs": "rust",
    "sh": "shell", "bash": "shell", "zsh": "shell",
    "md": "markdown",
}
_VALID_LANGS = {"python", "cpp", "c", "js", "ts", "go", "rust", "shell", "markdown"}


@dataclass(frozen=True)
class Chunk:
    path: str
    start: int   # 1-based first line, inclusive
    end: int     # 1-based last line, inclusive
    kind: str    # "code" or "text"
    lang: str
    text: str    # the chunk content


def chunk_file(path: str, text: str) -> list[Chunk]:
    if "\x00" in text:
        return []
    ext = os.path.splitext(path)[1].lower()
    lines = text.splitlines()
    if ext in _MD_EXTS:
        chunks = _chunk_markdown(path, lines)
    else:
        lang = _EXT_LANG.get(ext, "other")
        if lang == "python":
            chunks = _chunk_python(path, text, lines)
        elif lang in _BRACE_LANGS:
            chunks = _chunk_braces(path, lines, lang)
        else:
            chunks = _window(path, lines, 1, len(lines), lang)
    return [c for c in chunks if _keep(c)]


def _keep(chunk: Chunk) -> bool:
    # shortcut: spans under the min size are dropped, so tiny functions, lone headings and short fences are never compared; upgrade by merging small neighbours if real duplicates hide there
    if chunk.kind == "code":
        return sum(1 for line in chunk.text.splitlines() if line.strip()) >= MIN_CODE_LINES
    return len(chunk.text.split()) >= MIN_TEXT_WORDS


def _mk(path: str, start: int, end: int, kind: str, lang: str, lines: list[str]) -> Chunk:
    return Chunk(path, start, end, kind, lang, "\n".join(lines[start - 1:end]))


def _window(path: str, lines: list[str], a: int, b: int, lang: str) -> list[Chunk]:
    """Chunks of WINDOW lines, step STEP, over absolute 1-based lines a..b."""
    if b - a < WINDOW:
        return [_mk(path, a, b, "code", lang, lines)]
    out = []
    for s in (*range(a, b - WINDOW + 2, STEP), b - WINDOW + 1):
        if out and out[-1].start <= s and s + WINDOW - 1 <= out[-1].end:
            continue  # identical to or contained in the previous window
        out.append(_mk(path, s, s + WINDOW - 1, "code", lang, lines))
    return out


def _func_span(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[int, int]:
    start = node.lineno
    if node.decorator_list:
        start = min(start, min(d.lineno for d in node.decorator_list))
    return start, node.end_lineno or start


def _chunk_python(path: str, text: str, lines: list[str]) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _window(path, lines, 1, len(lines), "python")
    chunks = []
    covered = [False] * (len(lines) + 1)  # 1-based

    def take(s: int, e: int) -> None:
        c = _mk(path, s, e, "code", "python", lines)
        if not _keep(c):
            return  # short spans fall back to the leftover windows
        chunks.append(c)
        for k in range(s, e + 1):
            covered[k] = True

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            take(*_func_span(node))
        elif isinstance(node, ast.ClassDef):
            cs, ce = _func_span(node)
            mspans = sorted(
                _func_span(m) for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            for ms, me in mspans:
                take(ms, me)
            run = None
            for ln in range(cs, ce + 1):
                if any(ms <= ln <= me for ms, me in mspans):
                    if run is not None:
                        take(run, ln - 1)
                        run = None
                elif run is None:
                    run = ln
            if run is not None:
                take(run, ce)
    run = None
    for ln in range(1, len(lines) + 2):
        if ln <= len(lines) and not covered[ln]:
            if run is None:
                run = ln
        elif run is not None:
            chunks.extend(_window(path, lines, run, ln - 1, "python"))
            run = None
    return chunks


def _clean_lines(lines: list[str]) -> list[str]:
    # shortcut: no raw strings, template literals, or line continuations;
    # upgrade to a real lexer if brace chunks misfire on those.
    # shortcut: JS/TS regex literals (for example `/}/g`) are not lexed, so a brace inside one can move a chunk boundary; upgrade with a regex-literal rule if JS repos show it.
    out = []
    in_block = False
    for line in lines:
        buf = []
        i, n = 0, len(line)
        while i < n:
            if in_block:
                k = line.find("*/", i)
                if k == -1:
                    i = n
                else:
                    in_block = False
                    i = k + 2
                continue
            c = line[i]
            nxt = line[i + 1] if i + 1 < n else ""
            if c == "/" and nxt == "/":
                break
            if c == "/" and nxt == "*":
                in_block = True
                i += 2
                continue
            if c in "\"'":
                i += 1
                while i < n:
                    if line[i] == "\\":
                        i += 2
                    elif line[i] == c:
                        i += 1
                        break
                    else:
                        i += 1
                continue
            buf.append(c)
            i += 1
        out.append("".join(buf))
    return out


def _chunk_braces(path: str, lines: list[str], lang: str) -> list[Chunk]:
    clean = _clean_lines(lines)
    chunks = []
    covered = [False] * (len(lines) + 1)  # 1-based
    depth = 0
    pending = None      # first line of the current depth-0 signature
    chunk_start = None
    for ln, code in enumerate(clean, start=1):
        for ch in code:
            if ch == "{":
                if depth == 0:
                    chunk_start = pending if pending is not None else ln
                    pending = None  # consumed: the next signature starts fresh
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
                if depth == 0 and chunk_start is not None:
                    c = _mk(path, chunk_start, ln, "code", lang, lines)
                    if _keep(c):
                        chunks.append(c)
                        for k in range(chunk_start, ln + 1):
                            covered[k] = True
                    chunk_start = None
                    pending = None
            elif ch == ";" and depth == 0:
                pending = None
        if depth == 0 and chunk_start is None:
            if code.strip() and code.strip() != "}":
                if pending is None:
                    pending = ln
            else:
                pending = None
    run_start = None
    for ln in range(1, len(lines) + 2):
        if ln <= len(lines) and not covered[ln]:
            if run_start is None:
                run_start = ln
        elif run_start is not None:
            chunks.extend(_window(path, lines, run_start, ln - 1, lang))
            run_start = None
    return chunks


_FENCE_RE = re.compile(r"^ {0,3}(```+|~~~+)\s*(\S*)")


def _fence_lang(info: str) -> str:
    name = _FENCE_ALIAS.get(info.strip().lower(), info.strip().lower())
    return name if name in _VALID_LANGS else "other"


def _chunk_markdown(path: str, lines: list[str]) -> list[Chunk]:
    # shortcut: only '#' headings are split on; rst underline headings go to
    # the paragraph rule. Upgrade if rst files need section-level chunks.
    chunks = []
    para: list[int] = []  # 1-based line numbers

    def flush():
        if para:
            body = "\n".join(lines[ln - 1] for ln in para)
            chunks.append(Chunk(path, para[0], para[-1], "text", "markdown", body))
            para.clear()

    i = 0
    while i < len(lines):
        m = _FENCE_RE.match(lines[i])
        if m:
            flush()
            mark = m.group(1)
            j = i + 1
            closer = re.compile(r"^ {0,3}" + re.escape(mark[0]) + "{" + str(len(mark)) + r",}\s*$")
            while j < len(lines) and not closer.match(lines[j]):
                j += 1
            end = min(j, len(lines) - 1)  # include the closing fence if present
            body = "\n".join(lines[i + 1 : j])  # text without the fence lines
            chunks.append(Chunk(path, i + 1, end + 1, "code", _fence_lang(m.group(2)), body))
            i = j + 1
            continue
        line = lines[i]
        if line.startswith("#"):
            flush()
            para.append(i + 1)
        elif line.strip():
            para.append(i + 1)
        else:
            flush()
        i += 1
    flush()
    return chunks
