"""Tests for the dedup Stop hooks against fake dedup-check scripts."""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STOP_HOOK = ROOT / "hooks" / "stop_hook.py"
PREREVIEW = ROOT / "hooks" / "hermes-prereview"

sys.path.insert(0, str(ROOT / "hooks"))
import dedup_contract  # noqa: E402  (hook scripts are not a package)

CLAUDE_EVENT = {"session_id": "abc", "cwd": "/repo", "hook_event_name": "Stop",
                "stop_hook_active": False, "last_assistant_message": "done"}
CODEX_EVENT = {"turn_id": "t1", "hook_event_name": "Stop",
               "stop_hook_active": False, "last_assistant_message": "done"}

BLOCKING = {"base": "deadbeef", "findings": [
    {"new": {"path": "src/new.py", "start": 10, "end": 42, "kind": "code"},
     "existing": {"path": "lib/old.py", "start": 3, "end": 35},
     "match": "exact", "score": 0.99, "block": True}]}
NON_BLOCKING = {"base": "deadbeef", "findings": [
    {"new": {"path": "docs/a.md", "start": 1, "end": 5, "kind": "text"},
     "existing": {"path": "docs/b.md", "start": 9, "end": 13},
     "match": "lexical", "score": 0.72, "block": False}]}
EMPTY = {"base": "deadbeef", "findings": []}


@pytest.fixture
def fake(tmp_path):
    """Write a fake dedup-check and return a runner for the hook scripts."""

    def make(payload=None, exit_code=0, raw=None, err=None):
        script = tmp_path / f"dedup-{len(list(tmp_path.iterdir()))}"
        if raw is not None:
            octal = "".join(f"\\{b:03o}" for b in raw)
            script.write_text(f"#!/bin/sh\nprintf '{octal}'\nexit {exit_code}\n")
            script.chmod(0o755)
            return script
        body = json.dumps(payload) if payload is not None else ""
        if err is None:
            err = "boom" if exit_code else ""
        script.write_text(f"#!/bin/sh\necho '{err}' >&2\ncat <<'EOF'\n{body}\nEOF\nexit {exit_code}\n")
        script.chmod(0o755)
        return script

    def run(target, event, script, *args, cwd=None):
        env = dict(os.environ, DEDUP_CHECK=str(script))
        return subprocess.run([sys.executable, str(target), *args], input=json.dumps(event),
                              capture_output=True, text=True, env=env, cwd=cwd)

    return make, run


def stdout_json(p):
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_no_findings_claude_exits_0_without_block(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(EMPTY)))
    assert out == {} and "decision" not in out


def test_blocking_finding_blocks_once_naming_both_paths(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(BLOCKING)))
    assert out["decision"] == "block"
    reason = out["reason"]
    assert "src/new.py:10-42" in reason and "lib/old.py:3-35" in reason
    assert "(exact, 0.99)" in reason
    assert "Reuse or merge the existing code or text" in reason


def test_blocking_finding_not_blocked_when_stop_hook_active(fake):
    make_f, run = fake
    event = dict(CLAUDE_EVENT, stop_hook_active=True)
    out = stdout_json(run(STOP_HOOK, event, make_f(BLOCKING)))
    assert out.get("decision") != "block"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "src/new.py" in ctx and "lib/old.py" in ctx


def test_non_blocking_finding_does_not_block(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(NON_BLOCKING)))
    assert out.get("decision") != "block"
    assert "docs/a.md" in out["hookSpecificOutput"]["additionalContext"]


def test_dedup_check_error_does_not_block(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(None, exit_code=2)))
    assert out.get("decision") != "block"
    assert "boom" in out["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("payload,exit_code", [
    (EMPTY, 0), (BLOCKING, 0), (NON_BLOCKING, 0), (None, 2)])
def test_codex_event_gets_valid_json_stdout_on_every_path(fake, payload, exit_code):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CODEX_EVENT, make_f(payload, exit_code)))
    assert isinstance(out, dict)


# Codex 0.161.0 parses Stop output with deny_unknown_fields and drops the run
# on any other key. Every output path must stay inside the Codex schema.
CODEX_KEYS = {"continue", "stopReason", "suppressOutput", "systemMessage",
              "decision", "reason"}

@pytest.mark.parametrize("payload,exit_code,event_kw,want", [
    (BLOCKING, 0, {}, "decision"),
    (NON_BLOCKING, 0, {}, "systemMessage"),
    (BLOCKING, 0, {"stop_hook_active": True}, "systemMessage"),
    (EMPTY, 0, {}, None),
    (None, 2, {}, "systemMessage"),
], ids=["block", "advise", "stop_hook_active", "no-findings", "error"])
def test_codex_output_keys_stay_inside_the_codex_schema(fake, payload, exit_code,
                                                        event_kw, want):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, dict(CODEX_EVENT, **event_kw),
                          make_f(payload, exit_code), "--harness", "codex"))
    assert set(out) <= CODEX_KEYS
    if want:
        assert want in out
    if "decision" in out:
        assert out["decision"] == "block"


def test_hermes_prereview_exit_0_without_blocking_findings(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(NON_BLOCKING), "--repo", ".")
    assert p.returncode == 0 and "docs/a.md" in p.stdout


def test_hermes_prereview_exit_3_with_blocking_findings(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(BLOCKING), "--repo", ".")
    assert p.returncode == 3 and "[block]" in p.stdout


def test_hermes_prereview_exit_2_on_error(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(None, exit_code=2), "--repo", ".")
    assert p.returncode == 2 and "boom" in p.stderr


# A noisy dedup-check must not flood the one-line warning: first stderr line,
# capped at 200 characters, never more.
def test_stop_hook_dedup_error_stderr_is_capped_one_line(fake):
    make_f, run = fake
    long_err = "e" * 500
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(None, exit_code=2, err=long_err)))
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "\n" not in ctx and long_err[:200] in ctx and "e" * 201 not in ctx


def test_hermes_prereview_dedup_error_stderr_is_capped_one_line(fake):
    make_f, run = fake
    long_err = "e" * 500
    p = run(PREREVIEW, {}, make_f(None, exit_code=2, err=long_err), "--repo", ".")
    assert p.returncode == 2
    line = p.stderr.strip()
    assert "\n" not in line and long_err[:200] in line and "e" * 201 not in line


# The Stop event carries the session cwd: dedup-check must run there, not in
# the process cwd. A non-string or bad cwd must fall back, never crash.
def make_cwd_fake(tmp_path):
    """A fake dedup-check that reports BLOCKING only in a dir holding copy.txt."""
    script = tmp_path / f"dedup-cwd-{len(list(tmp_path.iterdir()))}"
    script.write_text(
        "#!/bin/sh\n"
        "if [ -f \"$PWD/copy.txt\" ]; then\n"
        f"  printf '%s\\n' '{json.dumps(BLOCKING)}'\n"
        "else\n"
        f"  printf '%s\\n' '{json.dumps(EMPTY)}'\n"
        "fi\n")
    script.chmod(0o755)
    return script


def repo_with_copy(tmp_path, name):
    repo = tmp_path / name
    repo.mkdir()
    (repo / "copy.txt").write_text("duplicate\n")
    return repo


def test_stop_hook_runs_dedup_check_in_the_event_cwd(fake, tmp_path):
    make_f, run = fake
    repo = repo_with_copy(tmp_path, "event-repo")
    other = tmp_path / "elsewhere"  # process cwd: no copy.txt here
    other.mkdir()
    event = dict(CLAUDE_EVENT, cwd=str(repo))
    out = stdout_json(run(STOP_HOOK, event, make_cwd_fake(tmp_path), cwd=str(other)))
    assert out["decision"] == "block"


def test_stop_hook_missing_event_cwd_keeps_inherited_cwd(fake, tmp_path):
    make_f, run = fake
    repo = repo_with_copy(tmp_path, "proc-repo")
    event = {k: v for k, v in CLAUDE_EVENT.items() if k != "cwd"}
    out = stdout_json(run(STOP_HOOK, event, make_cwd_fake(tmp_path), cwd=str(repo)))
    assert out["decision"] == "block"


@pytest.mark.parametrize("bad_cwd", [123, "no-such-dir"])
def test_stop_hook_bad_event_cwd_falls_back_without_crash(fake, tmp_path, bad_cwd):
    make_f, run = fake
    repo = repo_with_copy(tmp_path, "fallback-repo")
    if bad_cwd == "no-such-dir":
        bad_cwd = str(tmp_path / bad_cwd)
    event = dict(CLAUDE_EVENT, cwd=bad_cwd)
    out = stdout_json(run(STOP_HOOK, event, make_cwd_fake(tmp_path), cwd=str(repo)))
    assert out["decision"] == "block"  # fell back to the inherited cwd


NOT_AN_OBJECT = ["just", "a", "list"]
FINDING_WITHOUT_NEW = {"base": "deadbeef", "findings": [
    {"existing": {"path": "lib/old.py", "start": 3, "end": 35},
     "match": "exact", "score": 0.99, "block": True}]}


def test_stop_hook_non_object_json_gives_advice_and_exit_0(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(NOT_AN_OBJECT)))
    assert out.get("decision") != "block"
    assert "Duplicate check skipped" in out["hookSpecificOutput"]["additionalContext"]


def test_stop_hook_finding_without_new_gives_advice_and_exit_0(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(FINDING_WITHOUT_NEW)))
    assert out.get("decision") != "block"
    assert "Duplicate check skipped" in out["hookSpecificOutput"]["additionalContext"]


def test_stop_hook_non_object_stdin_exits_0_with_hook_output(fake):
    make_f, run = fake
    p = run(STOP_HOOK, ["not", "an", "object"], make_f(EMPTY))
    out = stdout_json(p)
    assert out == {}


def test_stop_hook_non_object_stdin_with_blocking_finding_blocks_once(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, ["not", "an", "object"], make_f(BLOCKING)))
    assert out["decision"] == "block"


def test_hermes_prereview_non_object_json_exit_2(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(NOT_AN_OBJECT), "--repo", ".")
    assert p.returncode == 2 and "rejected" in p.stderr


def test_hermes_prereview_finding_without_new_exit_2(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(FINDING_WITHOUT_NEW), "--repo", ".")
    assert p.returncode == 2 and "rejected" in p.stderr


# Non-UTF-8 stdout: decode failure takes the error path, it must never crash a hook.
def test_stop_hook_non_utf8_stdout_gives_advice_and_exit_0(fake):
    make_f, run = fake
    p = run(STOP_HOOK, CLAUDE_EVENT, make_f(raw=b"\xff\xfe{"))
    out = stdout_json(p)
    assert "decision" not in out
    assert "Duplicate check skipped" in out["hookSpecificOutput"]["additionalContext"]


def test_hermes_prereview_non_utf8_stdout_exit_2(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(raw=b"\xff\xfe{"), "--repo", ".")
    assert p.returncode == 2 and "rejected" in p.stderr


# The shared contract validator: one bad field must raise with the field path.
GOOD = {"base": "deadbeef", "findings": [
    {"new": {"path": "a.py", "start": 1, "end": 2, "kind": "code"},
     "existing": {"path": "b.py", "start": 3, "end": 4},
     "match": "lexical", "score": 0.9, "block": False}]}


def broke(edit):
    p = copy.deepcopy(GOOD)
    edit(p)
    return p


BAD_ROWS = [
    ("block-as-string", broke(lambda p: p["findings"][0].update(block="true"))),
    ("block-as-int", broke(lambda p: p["findings"][0].update(block=1))),
    ("score-as-string", broke(lambda p: p["findings"][0].update(score="0.9"))),
    ("score-as-bool", broke(lambda p: p["findings"][0].update(score=True))),
    ("start-as-string", broke(lambda p: p["findings"][0]["new"].update(start="3"))),
    ("start-as-bool", broke(lambda p: p["findings"][0]["new"].update(start=True))),
    ("kind-missing", broke(lambda p: p["findings"][0]["new"].pop("kind"))),
    ("kind-prose", broke(lambda p: p["findings"][0]["new"].update(kind="prose"))),
    ("match-fuzzy", broke(lambda p: p["findings"][0].update(match="fuzzy"))),
    ("score-nan", broke(lambda p: p["findings"][0].update(score=float("nan")))),
    ("score-infinity", broke(lambda p: p["findings"][0].update(score=float("inf")))),
    ("score-huge-int", broke(lambda p: p["findings"][0].update(score=10**400))),
    ("new-start-zero", broke(lambda p: p["findings"][0]["new"].update(start=0))),
    ("new-start-negative", broke(lambda p: p["findings"][0]["new"].update(start=-5))),
    ("new-span-reversed", broke(lambda p: p["findings"][0]["new"].update(start=10, end=5))),
    ("existing-start-zero", broke(lambda p: p["findings"][0]["existing"].update(start=0))),
    ("existing-start-negative", broke(lambda p: p["findings"][0]["existing"].update(start=-5))),
    ("existing-span-reversed", broke(lambda p: p["findings"][0]["existing"].update(start=10, end=5))),
    ("findings-as-dict", broke(lambda p: p.update(findings={"a.py": 1}))),
    ("base-as-int", broke(lambda p: p.update(base=5))),
]


@pytest.mark.parametrize("payload", BAD_ROWS, ids=[n for n, _ in BAD_ROWS])
def test_contract_rejects_bad_payload(payload):
    _, bad = payload
    with pytest.raises(ValueError) as e:
        dedup_contract.validate(bad)
    assert any(k in str(e.value) for k in ("findings", "base", "top level"))


def test_contract_accepts_valid_payload():
    data = dedup_contract.validate(copy.deepcopy(GOOD))
    assert data["findings"][0]["block"] is False


BLOCK_AS_STRING = broke(lambda p: p["findings"][0].update(block="true"))


def test_stop_hook_block_as_string_never_blocks(fake):
    make_f, run = fake
    out = stdout_json(run(STOP_HOOK, CLAUDE_EVENT, make_f(BLOCK_AS_STRING)))
    assert "decision" not in out
    assert "Duplicate check skipped" in out["hookSpecificOutput"]["additionalContext"]


def test_hermes_prereview_block_as_string_exit_2(fake):
    make_f, run = fake
    p = run(PREREVIEW, {}, make_f(BLOCK_AS_STRING), "--repo", ".")
    assert p.returncode == 2 and "block" in p.stderr


# Tests for the hermes-adversarial-review patch, applied to a scratch copy of
# the committed fixture: the suite runs on a clean checkout. A copy of the live
# script is patched only when DEDUP_TEST_LIVE_REVIEW=1; it is never modified.
PATCH = ROOT / "hooks" / "hermes-adversarial-review.patch"
FIXTURE_REVIEW = ROOT / "tests" / "fixtures" / "review-script"
LIVE_REVIEW = Path(os.environ.get(
    "DEDUP_LIVE_REVIEW", Path.home() / ".local/bin/hermes-adversarial-review"))


def patch_copy(source, target):
    shutil.copy(source, target)
    p = subprocess.run(["patch", "-p1", "-i", str(PATCH)], cwd=target.parent,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return target


@pytest.fixture
def patched_review(tmp_path):
    return patch_copy(FIXTURE_REVIEW, tmp_path / "hermes-adversarial-review")


@pytest.fixture
def patched_live_review(tmp_path):
    if os.environ.get("DEDUP_TEST_LIVE_REVIEW") != "1":
        pytest.skip("set DEDUP_TEST_LIVE_REVIEW=1 to patch a copy of the live script")
    if not LIVE_REVIEW.exists():
        pytest.skip(f"live script not present at {LIVE_REVIEW}")
    return patch_copy(LIVE_REVIEW, tmp_path / "hermes-adversarial-review")


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "a.txt").write_text("old\n")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], check=True)
    (repo / "new.txt").write_text("a new untracked file\n")
    return repo


def run_review(patched_review, tmp_path, *, prereview_rc=None, timeout=None):
    bindir = tmp_path / f"bin-{'none' if prereview_rc is None else prereview_rc}"
    bindir.mkdir(exist_ok=True)
    if prereview_rc is not None:
        # exec: the stub IS the sleeper, so the review's timeout kill hits it
        body = ("#!/bin/sh\nexec sleep 30\n" if prereview_rc == "sleep"
                else f"#!/bin/sh\nexit {prereview_rc}\n")
        pre = bindir / "hermes-prereview"
        pre.write_text(body)
        pre.chmod(0o755)
    hermes = bindir / "hermes"
    hermes.write_text("#!/bin/sh\nexit 0\n")
    hermes.chmod(0o755)
    env = dict(os.environ, PATH=f"{bindir}:/usr/bin:/bin",
               HERMES_HOME=str(tmp_path / "hh"))
    # timeout bounds the run inside pytest so a missing patch timeout cannot hang the suite
    return subprocess.run([sys.executable, str(patched_review),
                           "--repo", str(make_repo(tmp_path)), "--model", "test-model"],
                          capture_output=True, text=True, env=env, timeout=timeout)


def test_patch_exits_4_on_blocking_prereview(tmp_path, patched_review):
    p = run_review(patched_review, tmp_path, prereview_rc=3)
    assert p.returncode == 4


def test_patch_continues_warns_on_prereview_exit_2(tmp_path, patched_review):
    p = run_review(patched_review, tmp_path, prereview_rc=2)
    assert p.returncode != 4
    assert "dedup pre-check exited 2; continuing" in p.stderr


def test_patch_continues_warns_when_prereview_missing(tmp_path, patched_review):
    p = run_review(patched_review, tmp_path, prereview_rc=None)
    assert p.returncode != 4
    assert "dedup pre-check failed to start" in p.stderr


def test_patched_main_has_no_dedup_block_named_exit_4(patched_review):
    src = patched_review.read_text()
    assert "--no-dedup" in src
    dedup_block = src.split("if a.repo and not a.no_dedup:")[1].split("reviews = ")[0]
    assert "sys.exit(4)" in dedup_block
    assert "sys.exit(rc)" not in dedup_block and "sys.exit(3)" not in dedup_block
    assert "sys.exit(1)" not in dedup_block


def test_patch_prereview_timeout_warns_and_continues(tmp_path, patched_review):
    src = patched_review.read_text()
    assert "DEDUP_PREREVIEW_TIMEOUT = 600" in src
    patched_review.write_text(src.replace(
        "DEDUP_PREREVIEW_TIMEOUT = 600", "DEDUP_PREREVIEW_TIMEOUT = 2"))
    t0 = time.monotonic()
    p = run_review(patched_review, tmp_path, prereview_rc="sleep", timeout=15)
    assert time.monotonic() - t0 < 15
    assert "dedup pre-check timed out after 2 s; continuing" in p.stderr
    assert p.returncode != 4  # fail open: the review continues past the dedup step


# Opt-in check against a copy of the live script: same patch, same behaviour.
def test_live_review_patch_exits_4_on_blocking_prereview(tmp_path, patched_live_review):
    p = run_review(patched_live_review, tmp_path, prereview_rc=3, timeout=120)
    assert p.returncode == 4
