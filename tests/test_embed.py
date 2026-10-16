"""Tests for dedup.embed and its CLI wiring."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dedup import cli, embed


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


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch, request):
    # the model test uses the caller's cache: an empty one re-downloads the model
    if request.node.get_closest_marker("model"):
        return
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


CODE_A = '''\
def fetch_rows(cursor, table):
    query = "select * from " + table
    cursor.execute(query)
    rows = cursor.fetchall()
    names = [r["name"] for r in rows]
    total = len(names)
    return names, total
'''

# Structurally different from CODE_A so the lexical score stays below 0.6.
CODE_B = '''\
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

CODE_C = '''\
def rotate_left(bits, count):
    width = 32
    count = count % width
    mask = (1 << width) - 1
    shifted = (bits << count) & mask
    wrapped = bits >> (width - count)
    return shifted | wrapped
'''


def run_cli(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_load_code_is_none():
    assert "code" not in embed.MODELS
    assert embed.load("code") is None


def test_offline_failure_returns_none_and_cli_notes_once(repo, capsys, monkeypatch):
    import fastembed

    def boom(*args, **kwargs):
        raise RuntimeError("offline kaboom\nsecond line")

    monkeypatch.setattr(fastembed, "TextEmbedding", boom)
    assert embed.load("text") is None
    assert embed.ERRORS["text"] == "offline kaboom"

    # a pending text chunk makes the CLI try the text embedder; a lexical
    # finding on the code side still comes through and the exit stays 0
    (repo / "a.py").write_text(CODE_A)
    (repo / "intro.md").write_text(
        "# G\n\nepsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
        "rho sigma tau upsilon phi chi psi omega aleph beth\n"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c")
    (repo / "b.py").write_text(CODE_B)
    (repo / "notes.md").write_text(
        "# F\n\ncrimson amber scarlet lemon orange tangerine apricot mango "
        "papaya banana melon cherry plum grape fig date quince kiwi lychee guava\n"
    )
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert err.startswith("dedup-check: text model unavailable (offline kaboom); lexical only")
    assert len(err.strip().splitlines()) == 1
    fs = json.loads(out)["findings"]
    assert fs == []


def test_no_note_for_kind_without_entry(repo, capsys):
    # only code chunks, no lexical match: the semantic stage has no code entry
    # by design, so stderr stays empty
    (repo / "a.py").write_text(CODE_A)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "c")
    (repo / "b.py").write_text(CODE_B)
    (repo / "c.py").write_text(CODE_C)
    code, out, err = run_cli(["--repo", str(repo), "--json"], capsys)
    assert code == 0
    assert err == ""
    assert json.loads(out)["findings"] == []


SENT_A = (
    "Rising shipping costs and a shortage of warehouse space have pushed the "
    "retailer to move its regional distribution center closer to the port, "
    "where labor is cheaper."
)
SENT_B = (
    "Because shipping costs keep climbing and warehouse space is scarce, the "
    "retailer has decided to relocate its regional distribution center nearer "
    "to the port, taking advantage of lower labor costs there."
)
SENT_C = (
    "Preheat the oven to two hundred degrees, then simmer the sauce slowly "
    "while you knead the dough until it turns smooth and elastic before "
    "letting it rise in a warm corner of the kitchen for one full hour."
)


@pytest.mark.model
def test_real_model_paraphrase_scores():
    if not os.environ.get("DEDUP_TEST_MODEL"):
        pytest.skip("DEDUP_TEST_MODEL unset")
    emb = embed.load("text")
    assert emb is not None, embed.ERRORS.get("text")
    advise, block = embed.THRESHOLDS["text"]
    v = emb.embed([SENT_A, SENT_B, SENT_C])
    assert v.dtype == np.float32
    assert v.shape == (3, 768)
    np.testing.assert_allclose(
        np.linalg.norm(v, axis=1), 1.0, rtol=1e-5, atol=1e-6
    )
    assert float(v[0] @ v[1]) >= advise
    assert float(v[0] @ v[2]) < advise
