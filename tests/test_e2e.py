"""End-to-end acceptance: a seeded repo reports the copy and the paraphrase."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dedup import lexical

# 14 lines, so the function is its own code chunk.
FUNC = '''\
def summarize_events(events, limit):
    counts = {}
    skipped = 0
    for event in events:
        kind = event.get("kind")
        if kind is None:
            skipped += 1
            continue
        counts[kind] = counts.get(kind, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: pair[1])
    ranked.reverse()
    top = ranked[:limit]
    report = {name: total for name, total in top}
    return report, skipped
'''

# Identifiers renamed, structure equal: the lexical tokenizer maps both to "ID".
FUNC_RENAMED = '''\
def tally_records(records, cap):
    totals = {}
    dropped = 0
    for record in records:
        tag = record.get("kind")
        if tag is None:
            dropped += 1
            continue
        totals[tag] = totals.get(tag, 0) + 1
    ordered = sorted(totals.items(), key=lambda pair: pair[1])
    ordered.reverse()
    head = ordered[:cap]
    summary = {name: total for name, total in head}
    return summary, dropped
'''

# Same length, different job: lexical Jaccard with FUNC is 0.07, under 0.40.
FUNC_UNRELATED = '''\
def format_table(headers, rows):
    widths = []
    for header in headers:
        widths.append(len(str(header)))
    for row in rows:
        for index, cell in enumerate(row):
            text = str(cell)
            if len(text) > widths[index]:
                widths[index] = len(text)
    border = "+".join("-" * (w + 2) for w in widths)
    lines = [border]
    lines.append("| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append(border)
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    lines.append(border)
    return "\\n".join(lines)
'''

PARA = (
    "The nightly job copies each merged change into the review bucket, where a "
    "second worker checks the archive against the source tree and writes a "
    "short report that lists every file whose hash no longer matches the one "
    "recorded in the manifest."
)

# Same claims, new wording: text Jaccard is 0.13, so only the semantic stage
# can find it; the real model scores the pair at cosine 0.91.
PARAPHRASE = (
    "Each night the job duplicates every merged change into the review "
    "bucket, and there a second worker compares the archive with the source "
    "tree, writing a short report that names any file whose hash differs "
    "from the one the manifest records."
)


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


@pytest.fixture
def seeded(repo):
    (repo / "mod_a.py").write_text(FUNC)
    (repo / "guide.md").write_text(PARA + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    (repo / "copy_a.py").write_text(FUNC_RENAMED)
    (repo / "notes.md").write_text(PARAPHRASE + "\n")
    (repo / "unrelated_c.py").write_text(FUNC_UNRELATED)
    return repo


def run_cli_sub(repo):
    # the real CLI in its own process, with the same --json --repo call shape
    # as the in-process tests
    return subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from dedup import cli; "
            "raise SystemExit(cli.main(sys.argv[1:]))",
            "--json", "--repo", str(repo),
        ],
        capture_output=True, text=True,
    )


def shingle_jaccard(a, b, kind):
    s1, s2 = lexical.shingles(a, kind), lexical.shingles(b, kind)
    return len(s1 & s2) / len(s1 | s2)


@pytest.mark.model
def test_seeded_repo_reports_copy_and_paraphrase(seeded):
    if not os.environ.get("DEDUP_TEST_MODEL"):
        pytest.skip("DEDUP_TEST_MODEL unset")
    assert shingle_jaccard(PARA, PARAPHRASE, "text") < 0.60
    proc = run_cli_sub(seeded)
    assert proc.returncode == 0, proc.stderr
    fs = json.loads(proc.stdout)["findings"]

    # (a) the renamed copy matches on the function's span
    a = [f for f in fs if f["new"]["path"] == "copy_a.py"]
    assert len(a) == 1
    f = a[0]
    assert f["match"] in ("exact", "lexical")
    assert f["existing"]["path"] == "mod_a.py"
    span = (f["new"]["start"], f["new"]["end"])
    assert span == (1, len(FUNC_RENAMED.splitlines()))

    # (b) the paraphrase is a semantic finding only; semantic never blocks
    b = [f for f in fs if f["new"]["path"] == "notes.md"]
    assert len(b) == 1, f"paraphrase finding missing: {fs}"
    assert b[0]["match"] == "semantic"
    assert b[0]["block"] is False
    assert b[0]["existing"]["path"] == "guide.md"

    # (c) unrelated new code is no finding
    assert not [f for f in fs if f["new"]["path"] == "unrelated_c.py"]


def test_seeded_repo_lexical_only(seeded, monkeypatch):
    # no embedder: an empty cache plus offline mode makes the model load fail,
    # so the subprocess runs the lexical stage only
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    proc = run_cli_sub(seeded)
    assert proc.returncode == 0, proc.stderr
    fs = json.loads(proc.stdout)["findings"]
    a = [f for f in fs if f["new"]["path"] == "copy_a.py"]
    assert len(a) == 1
    assert a[0]["match"] in ("exact", "lexical")
    assert not [f for f in fs if f["new"]["path"] == "unrelated_c.py"]
