"""Tests for dedup.cli against real throwaway git repos."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dedup import cli
from dedup import lexical

CONTRACT = Path(__file__).resolve().parent.parent / "hooks" / "dedup_contract.py"


def load_contract():
    spec = importlib.util.spec_from_file_location("dedup_contract", CONTRACT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FUNC_A = '''\
def fetch_rows(cursor, table):
    query = "select * from " + table
    cursor.execute(query)
    rows = cursor.fetchall()
    names = [r["name"] for r in rows]
    total = len(names)
    return names, total
'''

FUNC_A_RENAMED = '''\
def grab_items(handle, source):
    stmt = "select * from " + source
    handle.execute(stmt)
    recs = handle.fetchall()
    labels = [r["name"] for r in recs]
    amount = len(labels)
    return labels, amount
'''

# Structurally different from FUNC_A so the lexical score stays below 0.6.
FUNC_B = '''\
def clamp_window(values, low, high):
    out = []
    for v in values:
        if v < low:
            out.append(low)
        elif v > high:
            out.append(high)
        else:
            out.append(v)
    return out
'''


def git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@t.t")
    git(r, "config", "user.name", "t")
    git(r, "config", "commit.gpgsign", "false")
    return r


def commit_all(repo, msg="c"):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


def write(repo, rel, text):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def run_cli(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


class FakeEmbedder:
    """Maps every text to the same vector so all pairs have cosine 1.0."""

    model_id = "fake-model"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0


def fake_loader(fake, advise=0.5, block=0.99):
    return lambda kind: (fake, advise, block)


class CosineEmbedder:
    """Base chunks get one vector, new chunks another; their cosine is `cos`."""

    model_id = "fake-cosine"

    def __init__(self, cos):
        b = np.array([float(cos), float((1.0 - float(cos) ** 2) ** 0.5)])
        b /= np.linalg.norm(b)
        self._a = np.array([1.0, 0.0], np.float32)
        self._b = b.astype(np.float32)

    def embed(self, texts):
        # FUNC_A chunks form the base side, FUNC_B chunks the new side.
        return np.stack([self._a if "fetch_rows" in t else self._b for t in texts])


def shingle_jaccard(a, b, kind):
    s1, s2 = lexical.shingles(a, kind), lexical.shingles(b, kind)
    return len(s1 & s2) / len(s1 | s2)


def _spaced(tokens):
    lines = [" ".join(tokens[i:i + 10]) for i in range(0, len(tokens), 10)]
    return "\n".join(lines) + "\n"


# Disjoint operator-run tokens keep every 5-gram unique to its region, so the
# shared and unique token counts set the Jaccard directly.
def code_pair(n_shared, n_uniq):
    shared = ["+" * i for i in range(1, n_shared + 1)]
    uniq1 = ["-" * i for i in range(1, n_uniq + 1)]
    uniq2 = ["*" * i for i in range(1, n_uniq + 1)]
    return _spaced(shared + uniq1), _spaced(shared + uniq2)


def text_pair(n_shared, n_uniq):
    shared = ["s%03d" % i for i in range(n_shared)]
    uniq1 = ["u%03d" % i for i in range(n_uniq)]
    uniq2 = ["v%03d" % i for i in range(n_uniq)]
    return _spaced(shared + uniq1), _spaced(shared + uniq2)


# Identical but for the expression after '//': without Python rules '//'
# opens a comment and the token streams (and scores) are identical.
FUNC_DIV_A = '''\
def divide_span(a, b, c, mode):
    span = len(mode)
    total = a // b
    digits = str(total)
    count = len(digits)
    pair = (total, count)
    return pair, span
'''

FUNC_DIV_B = FUNC_DIV_A.replace("a // b\n", "a // b if c else d\n")


# The tokenizer gets the file language, so a '//' difference keeps the pair
# under the 0.95 block cutoff instead of scoring a false 1.0.
def test_python_floor_div_not_identical(repo, capsys):
    write(repo, "a.py", FUNC_DIV_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_DIV_B)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    pair = [f for f in fs if f["new"]["path"] == "b.py" and f["existing"]["path"] == "a.py"]
    assert all(not f["block"] and f["score"] < 1.0 for f in pair)


# A fenced python block in markdown tokenizes by its fence language, so a
# renamed copy still matches the base Python function.
def test_fenced_python_matches_base_function(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "notes.md", "# guide\n\n```python\n" + FUNC_A_RENAMED + "```\n")
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "lexical"
    assert f["new"]["path"] == "notes.md"
    assert f["existing"]["path"] == "a.py"


# (a) + (j): unchanged copy into another file is exact/block and passes the contract.
def test_exact_copy_blocks(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    load_contract().validate(data)
    assert len(data["base"]) == 40
    assert len(data["findings"]) == 1
    f = data["findings"][0]
    assert f["match"] == "exact"
    assert f["block"] is True
    assert f["score"] == 1.0
    assert f["new"]["path"] == "b.py"
    assert f["existing"]["path"] == "a.py"


# (b): a renamed-identifier copy matches lexically.
def test_renamed_copy_is_lexical(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A_RENAMED)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    assert fs[0]["match"] == "lexical"
    assert fs[0]["score"] >= 0.95
    assert fs[0]["block"] is True


# (c): editing a function in place leaves no untouched existing copy.
def test_edit_in_place_no_finding(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "a.py", FUNC_A.replace("total = len(names)", "total = len(names) + 1"))
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["findings"] == []


# (d): moving a function to another file gives no finding.
def test_move_no_finding(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "a.py", "")
    write(repo, "b.py", FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["findings"] == []


# (e): an untracked file is checked.
def test_untracked_file_checked(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "loose.py", FUNC_A)
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        == "loose.py"
    )
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1 and fs[0]["match"] == "exact"


# A lock-file copy is skipped; an exact .py copy in the same diff still blocks.
def test_generated_lock_files_skipped(repo, capsys):
    write(repo, "a.py", FUNC_A)
    write(repo, "vendor/uv.lock", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    write(repo, "copy/uv.lock", FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["new"]["path"] == "b.py" and f["match"] == "exact" and f["block"]


# (f): outside a git repo prints null base and exits 0.
def test_outside_repo(tmp_path, capsys):
    code, out, _ = run_cli(["--repo", str(tmp_path), "--json"], capsys)
    assert code == 0
    assert json.loads(out) == {"base": None, "findings": []}
    code, out, _ = run_cli(["--repo", str(tmp_path)], capsys)
    assert code == 0
    assert out == ""


# (g): a bad --base exits 2 with empty stdout.
def test_bad_base(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    code, out, err = run_cli(["--repo", str(repo), "--base", "nope-nope", "--json"], capsys)
    assert code == 2
    assert out == ""
    assert err.count("\n") == 1 or (err and "\n" not in err.rstrip("\n"))


# (h) + (j): same-vector texts match semantically; JSON passes the contract.
def test_semantic_match(repo, capsys, monkeypatch):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_B)
    fake = FakeEmbedder()
    monkeypatch.setattr(cli, "_load_embedder", fake_loader(fake))
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    load_contract().validate(data)
    assert len(data["findings"]) == 1
    f = data["findings"][0]
    assert f["match"] == "semantic"
    assert f["block"] is False  # semantic matches advise only, even at 1.0
    assert f["score"] == pytest.approx(1.0)


def test_semantic_advise_only(repo, capsys, monkeypatch):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_B)
    fake = FakeEmbedder()
    monkeypatch.setattr(cli, "_load_embedder", fake_loader(fake, advise=0.5, block=1.5))
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    assert fs[0]["match"] == "semantic"
    assert fs[0]["block"] is False


# (i): a second run embeds only new chunks; a corrupt cache file is recomputed.
def test_embedding_cache(repo, capsys, monkeypatch, tmp_path):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_B)
    fake = FakeEmbedder()
    monkeypatch.setattr(cli, "_load_embedder", fake_loader(fake))

    code, _, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0 and not err
    first_calls = list(fake.calls)
    assert any(any("def fetch_rows" in t for t in c) for c in first_calls)
    assert any(any("def clamp_window" in t for t in c) for c in first_calls)

    fake.calls.clear()
    code, _, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert fake.calls  # the new side is still embedded every run
    for call in fake.calls:
        assert all("def fetch_rows" not in t for t in call), "base blob re-embedded"
    assert any(any("def clamp_window" in t for t in c) for c in fake.calls)

    cache_files = list((tmp_path / "cache" / "skill-dedup").glob("*.npz"))
    assert cache_files
    for f in cache_files:
        f.write_bytes(b"garbage")
    fake.calls.clear()
    code, _, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert any(any("def fetch_rows" in t for t in c) for c in fake.calls)


# (l): a second run reads the lexical cache: base blobs are not re-shingled,
# and a corrupt lexical cache file is recomputed.
def test_lexical_cache(repo, capsys, monkeypatch, tmp_path):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A_RENAMED)
    from dedup import lexical
    real = lexical.shingles
    calls = []
    def spy(text, kind, lang=None):
        calls.append(text)
        return real(text, kind, lang)
    monkeypatch.setattr(lexical, "shingles", spy)

    code, _, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert any("def fetch_rows" in t for t in calls)  # base blob shingled once
    assert any("def grab_items" in t for t in calls)  # new chunk shingled
    n_first = len(calls)

    calls.clear()
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1 and fs[0]["match"] == "lexical"
    assert not any("def fetch_rows" in t for t in calls), "base blob re-shingled"
    assert any("def grab_items" in t for t in calls)  # new chunks are still shingled
    assert len(calls) < n_first

    cache_files = list((tmp_path / "cache" / "skill-dedup").glob("lex-*.npz"))
    assert cache_files
    for f in cache_files:
        f.write_bytes(b"garbage")
    calls.clear()
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1 and fs[0]["match"] == "lexical"
    assert any("def fetch_rows" in t for t in calls), "corrupt cache not recomputed"


# (k): without the embed module, lexical findings still appear and stderr notes it.
def test_no_embed_module(repo, capsys, monkeypatch):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A_RENAMED)
    write(repo, "c.py", FUNC_B)
    # a None entry in sys.modules plus no attribute on the package makes the
    # import raise ImportError
    import dedup
    monkeypatch.setitem(__import__("sys").modules, "dedup.embed", None)
    monkeypatch.delattr(dedup, "embed", raising=False)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert any(f["match"] == "lexical" for f in fs)
    assert "lexical" in err


def test_text_output(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo)], capsys)
    assert code == 0
    lines = out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("b.py:1-7 ~ a.py:1-7 exact 1.000 BLOCK")


# git output parsing: paths with spaces or non-ASCII survive.

def test_untracked_path_with_space(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "sub dir/with space.py", FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    assert fs[0]["match"] == "exact"
    assert fs[0]["new"]["path"] == "sub dir/with space.py"


def test_untracked_path_with_utf8(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "ünï/cødé.py", FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    assert fs[0]["new"]["path"] == "ünï/cødé.py"


def test_tracked_path_with_space_second_hunk(repo, capsys):
    helper = """\
def top_helper(items):
    total = 0
    for it in items:
        total += 1
    return total
"""
    write(repo, "a.py", FUNC_A)
    write(repo, "my file.py", helper)
    commit_all(repo)
    write(repo, "my file.py", helper.replace("total += 1", "total += 2") + FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1 and fs[0]["match"] == "exact"
    assert fs[0]["new"]["path"] == "my file.py"
    assert fs[0]["existing"]["path"] == "a.py"


# A removed content line that starts with "-- " looks like a header: hunk
# line counts must keep it out of header parsing. Paragraphs use disjoint
# word sets so no accidental lexical match hides a misparsed section.
def test_removed_line_looks_like_header(repo, capsys):
    base = (
        "red orange yellow green blue indigo violet black white gray\n"
        "brown pink purple gold silver copper bronze iron steel stone\n"
        "\n"
        "navy teal aqua coral olive mint peach plum lilac salmon\n"
        "-- kilo lima mike november oscar papa quebec romeo sierra\n"
        "khaki beige cream ivory tan rust jade pearl ruby amber\n"
        "\n"
        "zebra giraffe tiger lion leopard cheetah puma jaguar ocelot lynx\n"
        "bobcat cougar panther wolf fox bear moose elk deer bison\n"
    )
    write(repo, "file.txt", base)
    commit_all(repo)
    write(
        repo, "file.txt",
        base.replace(
            "red orange yellow green blue indigo violet black white gray",
            "uno dos tres cuatro cinco seis siete ocho nueve diez",
        )
        .replace("-- kilo lima mike november oscar papa quebec romeo sierra\n", "")
        .replace(
            "zebra giraffe tiger lion leopard cheetah puma jaguar ocelot lynx\n",
            "zebra giraffe tiger lion leopard cheetah puma jaguar ocelot lynx  \n",
        ),
    )
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["findings"] == []


def test_repo_subdir_matches_root(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "sub/new.py", FUNC_A)
    code, out_root, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    code, out_sub, _ = run_cli(["--repo", str(repo / "sub"), "--json"], capsys)
    assert code == 0
    assert out_sub == out_root
    fs = json.loads(out_sub)["findings"]
    assert len(fs) == 1
    assert fs[0]["new"]["path"] == "sub/new.py"


def test_no_commits_one_error_line(repo, capsys):
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 2
    assert out == ""
    assert err.startswith("dedup-check: ")
    assert len(err.strip().splitlines()) == 1


def test_bad_flag_one_error_line(repo, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["--repo", str(repo), "--bogus"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("dedup-check: ")
    assert len(err.strip().splitlines()) == 1


# A cosine-0.999 semantic match still does not block.
def test_semantic_score_0_999_never_blocks(repo, capsys, monkeypatch):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_B)
    fake = CosineEmbedder(0.999)
    monkeypatch.setattr(cli, "_load_embedder", fake_loader(fake, advise=0.5, block=0.99))
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    load_contract().validate(data)
    fs = data["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "semantic"
    assert f["score"] == pytest.approx(0.999, abs=1e-3)
    assert f["block"] is False


# A code pair just above the 0.40 code cutoff is a lexical advise finding.
def test_code_pair_at_0_45_is_lexical_advise(repo, capsys):
    base_text, new_text = code_pair(65, 38)
    j = shingle_jaccard(base_text, new_text, "code")
    assert 0.43 < j < 0.47  # the fixture must sit just above the 0.40 cutoff
    write(repo, "base.algo", base_text)
    commit_all(repo)
    write(repo, "new.algo", new_text)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    load_contract().validate(data)
    fs = data["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "lexical"
    assert f["score"] == pytest.approx(j)
    assert f["block"] is False  # 0.45 is under the 0.95 lexical block


# A code pair below the 0.40 code cutoff gives no finding.
def test_code_pair_at_0_35_no_finding(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_embedder", lambda kind: None)
    base_text, new_text = code_pair(58, 48)
    j = shingle_jaccard(base_text, new_text, "code")
    assert 0.34 < j < 0.38  # the fixture must sit below the 0.40 cutoff
    write(repo, "base.algo", base_text)
    commit_all(repo)
    write(repo, "new.algo", new_text)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["findings"] == []


# A text pair above the code cutoff but below the 0.60 text cutoff gives no
# lexical finding.
def test_text_pair_at_0_5_no_lexical_finding(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_embedder", lambda kind: None)
    base_text, new_text = text_pair(135, 69)
    j = shingle_jaccard(base_text, new_text, "text")
    assert 0.47 < j < 0.53  # the fixture must sit between the two cutoffs
    write(repo, "base.txt", base_text)
    commit_all(repo)
    write(repo, "new.txt", new_text)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["findings"] == []


# Worktree fix 1: symlinks, devices and oversized worktree files are skipped.
def test_untracked_symlinks_and_oversize_skipped(repo, capsys, tmp_path):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    outside = tmp_path / "outside.py"
    outside.write_text(FUNC_A)
    os.symlink(outside, repo / "zleak.py")
    os.symlink("/dev/zero", repo / "zzero.py")
    write(repo, "big.py", FUNC_A * 30000)  # over MAX_BLOB_BYTES
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    assert json.loads(out)["findings"] == []


# Worktree fix 2: a blob over MAX_BLOB_BYTES is skipped by size; its bytes are
# never streamed through cat-file --batch.
def test_oversized_base_blob_never_read(repo, capsys, monkeypatch):
    write(repo, "a.py", FUNC_A)
    write(repo, "big.py", "".join("pad%06d = tail\n" % i for i in range(200000)))
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    streamed = []
    real_popen = subprocess.Popen

    class Count:
        def __init__(self, f):
            self.f = f
            self.n = 0

        def read(self, n=-1):
            b = self.f.read(n)
            self.n += len(b)
            return b

        def readline(self):
            b = self.f.readline()
            self.n += len(b)
            return b

    def popen_spy(*args, **kwargs):
        p = real_popen(*args, **kwargs)
        if "--batch" in args[0]:
            p.stdout = Count(p.stdout)
            streamed.append(p.stdout)
        return p

    monkeypatch.setattr(cli.subprocess, "Popen", popen_spy)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1 and fs[0]["match"] == "exact"  # a.py <-> b.py still found
    assert streamed, "no --batch content was streamed at all"
    # big.py is over 2 MiB; only headers and the small a.py blob may stream.
    assert sum(c.n for c in streamed) < 4096


# Fix 3: repo diff config (noprefix, mnemonicPrefix, relative) must not change
# how the patch header paths are read.
def test_diff_config_prefixes_ignored(repo, capsys):
    git(repo, "config", "diff.noprefix", "true")
    git(repo, "config", "diff.mnemonicPrefix", "true")
    git(repo, "config", "diff.relative", "true")
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "a.py", FUNC_A + FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact"
    assert (f["new"]["path"], f["new"]["start"], f["new"]["end"]) == ("a.py", 8, 14)
    assert (f["existing"]["path"], f["existing"]["start"], f["existing"]["end"]) == ("a.py", 1, 7)


FUNC_A_EXTRA = FUNC_A.replace(
    "    total = len(names)\n", "    total = len(names)\n    total += 0\n"
)


# Fix 4: deleting the one differing statement turns b.py into an exact
# duplicate of a.py even though the hunk has a zero-length new range.
def test_deletion_only_duplicate_found(repo, capsys):
    write(repo, "a.py", FUNC_A)
    write(repo, "b.py", FUNC_A_EXTRA)
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact"
    assert f["new"]["path"] == "b.py"
    assert f["existing"]["path"] == "a.py"


# Fix 5: appending a copy of the last function to the same file keeps the
# untouched base chunk as an existing candidate.
def test_append_same_file_duplicate_at_eof(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "a.py", FUNC_A + FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact"
    assert (f["new"]["path"], f["new"]["start"], f["new"]["end"]) == ("a.py", 8, 14)
    assert (f["existing"]["path"], f["existing"]["start"], f["existing"]["end"]) == ("a.py", 1, 7)


WORDS_LINE = " ".join("w%02d" % i for i in range(20)) + "\n"


# Fix 6: one blob under two extensions chunks twice; the .md record must exist
# even when .algo (which chunks the blob to nothing) sorts first.
def test_existing_chunks_keyed_by_extension(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_embedder", lambda kind: None)
    write(repo, "a.algo", WORDS_LINE)
    write(repo, "b.md", WORDS_LINE)
    commit_all(repo)
    write(repo, "c.md", WORDS_LINE)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact"
    assert f["new"]["path"] == "c.md"
    assert f["existing"]["path"] == "b.md"


# Disconnected operator tokens keep every 5-gram unique across windows so the
# base chunks stay below the lexical cutoff of any new chunk.
_OPS = "+-*%<>=!&|^~?:"


def op_lines(n_lines, seed):
    lines = []
    for i in range(n_lines):
        toks = []
        for k in range(10):
            x = seed + i * 10 + k
            t = ""
            for _ in range(6):
                t = _OPS[x % 13] + t
                x //= 13
            toks.append(t)
        lines.append(" ".join(toks))
    return lines


# Fix 7: scoring in blocks of existing vectors must give the same findings as
# the full similarity matrix. The second run reuses the caches, so the same
# repo serves both block sizes.
def test_semantic_blocked_matches_unblocked(repo, capsys, monkeypatch):
    write(repo, "base.algo", "\n".join(op_lines(200040, 0)) + "\n")
    commit_all(repo)
    write(repo, "new.algo", "\n".join(op_lines(50, 3000000)) + "\n")
    # The base blob must stay a blob under test, so lift the size cap.
    monkeypatch.setattr(cli, "MAX_BLOB_BYTES", 1 << 26)
    fake = FakeEmbedder()
    monkeypatch.setattr(cli, "_load_embedder", fake_loader(fake))
    monkeypatch.setattr(cli, "SEM_BLOCK", 1024)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    blocked = json.loads(out)["findings"]
    assert len(blocked) == 2  # both window chunks of new.algo
    assert all(f["match"] == "semantic" for f in blocked)
    assert any(len(call) >= 10000 for call in fake.calls)  # many existing vectors
    monkeypatch.setattr(cli, "SEM_BLOCK", 10**9)  # one block: the full matrix
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    unblocked = json.loads(out)["findings"]
    assert blocked == unblocked


# (A): a comment-only edit drops F from BASE, so the copy in b.py must match
# the edited new chunk in a.py.
def test_edit_plus_copy_matches_new_pair(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_embedder", lambda kind: None)
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    edited = FUNC_A.replace(
        "    rows = cursor.fetchall()\n", "    # fetch\n    rows = cursor.fetchall()\n"
    )
    write(repo, "a.py", edited)
    write(repo, "b.py", edited)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact" and f["block"] is True
    assert f["new"]["path"] == "b.py"
    assert f["existing"]["path"] == "a.py"


WORDS = "\n".join("word%02d" % i for i in range(30)) + "\n"


# (A): two identical untracked files report the pair once; the later file in
# (path, start) order is `new`.
def test_two_identical_untracked_files_one_finding(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_embedder", lambda kind: None)
    write(repo, "README.md", WORDS)
    commit_all(repo)
    write(repo, "x.py", FUNC_A)
    write(repo, "y.py", FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact" and f["block"] is True
    assert f["new"]["path"] == "y.py"
    assert f["existing"]["path"] == "x.py"


# (A): a lone untracked file never matches itself.
def test_lone_untracked_file_no_finding(repo, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_load_embedder", lambda kind: None)
    write(repo, "README.md", WORDS)
    commit_all(repo)
    write(repo, "x.py", FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    assert json.loads(out)["findings"] == []


# (B): "*.py -diff" must not hide a tracked edit turned copy.
def test_diff_attribute_off_file_still_checked(repo, capsys):
    write(repo, ".gitattributes", "*.py -diff\n")
    write(repo, "a.py", FUNC_A)
    write(repo, "b.py", FUNC_B)
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0, err
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "exact" and f["block"] is True
    assert f["new"]["path"] == "b.py"
    assert f["existing"]["path"] == "a.py"


# (C): a missing base blob is one stderr line and exit 2.
def test_missing_base_blob_is_error(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD:a.py"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    (repo / ".git" / "objects" / sha[:2] / sha[2:]).unlink()
    write(repo, "b.py", FUNC_B)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 2
    assert out == ""
    assert err.startswith("dedup-check: ")
    assert len(err.splitlines()) == 1


# (n): new chunks match each other semantically; the chunk later in
# (path, start) order is the `new` side. A lone chunk never matches itself.
def test_semantic_new_vs_new(repo, capsys, monkeypatch, tmp_path):
    write(repo, "seed.txt", "tiny seed\n")
    commit_all(repo)
    a_text, b_text = text_pair(0, 30)  # disjoint words: lexical Jaccard is 0
    assert shingle_jaccard(a_text, b_text, "text") < 0.60
    write(repo, "a.md", a_text)
    write(repo, "b.md", b_text)
    fake = FakeEmbedder()  # every text gets the same vector: cosine 1.0
    monkeypatch.setattr(cli, "_load_embedder", fake_loader(fake))
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["match"] == "semantic"
    assert f["block"] is False  # semantic matches advise only
    assert (f["new"]["path"], f["new"]["start"]) == ("b.md", 1)
    assert (f["existing"]["path"], f["existing"]["start"]) == ("a.md", 1)

    # A single new file with one paragraph gives one chunk; its span
    # overlaps only itself, so no finding may come out.
    r2 = tmp_path / "r2"
    r2.mkdir()
    git(r2, "init", "-q")
    git(r2, "config", "user.email", "t@t.t")
    git(r2, "config", "user.name", "t")
    git(r2, "config", "commit.gpgsign", "false")
    write(r2, "seed.txt", "tiny seed\n")
    commit_all(r2)
    write(r2, "one.md", text_pair(0, 30)[0])
    code, out, _ = run_cli(["--repo", str(r2), "--json"], capsys)
    assert code == 0
    assert json.loads(out)["findings"] == []


# Same-shape data rows that differ only in values tokenize to LIT everywhere
# and score 1.0, so findings in data files advise only and must never block.
def test_jsonl_values_only_finding_never_blocks(repo, capsys):
    rows_a = "".join(
        f'{{"id": {i}, "name": "alpha{i}", "score": {i}.5}}\n' for i in range(8)
    )
    rows_b = "".join(
        f'{{"id": {i + 100}, "name": "beta{i}", "score": {i}.9}}\n'
        for i in range(8)
    )
    write(repo, "a.jsonl", rows_a)
    commit_all(repo)
    write(repo, "b.jsonl", rows_b)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    f = fs[0]
    assert f["new"]["path"] == "b.jsonl"
    assert f["existing"]["path"] == "a.jsonl"
    assert f["score"] == 1.0
    assert f["block"] is False


# The data-file rule changes block only: an exact copy in a code file blocks.
def test_exact_py_copy_still_blocks(repo, capsys):
    write(repo, "a.py", FUNC_A)
    commit_all(repo)
    write(repo, "b.py", FUNC_A)
    code, out, _ = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    fs = json.loads(out)["findings"]
    assert len(fs) == 1
    assert fs[0]["match"] == "exact" and fs[0]["block"] is True


# --fetch-model loads the embedder (which downloads the model on a cold
# cache), prints the cache dir and exits 0; a load failure exits 2 with one
# stderr line. The suite stubs embed.load: no download here.
def test_fetch_model_ready(capsys, monkeypatch):
    from dedup import embed
    monkeypatch.setattr(embed, "load", lambda kind: object())
    code, out, err = run_cli(["--fetch-model"], capsys)
    assert code == 0
    assert out.startswith("dedup-check: model ready at "), out
    assert err == ""


def test_fetch_model_failure_exits_2(capsys, monkeypatch):
    from dedup import embed
    embed.ERRORS.clear()

    def fail(kind):
        embed.ERRORS["text"] = "offline"
        return None

    monkeypatch.setattr(embed, "load", fail)
    code, out, err = run_cli(["--fetch-model"], capsys)
    assert code == 2
    lines = err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("dedup-check: ")
    assert out == ""
