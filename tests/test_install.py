"""Tests for scripts/apply_hooks.py, driven directly."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APPLY_HOOKS = ROOT / "scripts" / "apply_hooks.py"

sys.path.insert(0, str(ROOT / "scripts"))
import apply_hooks  # noqa: E402  (scripts are not a package)


def base_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
    env["HOME"] = str(home)
    return env


def ours_entries(path: Path, command: str):
    data = json.loads(path.read_text())
    return [g for g in data["hooks"]["Stop"]
            if any(h.get("command") == command for h in g.get("hooks", []))]


def run_apply_hooks(home: Path, hook: Path):
    env = base_env(home)
    return subprocess.run([sys.executable, str(APPLY_HOOKS), str(hook)],
                          capture_output=True, text=True, env=env)


def test_wrong_shapes_give_one_line_and_write_nothing(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    hook = tmp_path / "stop_hook.py"
    bad = ("[1, 2]",
           json.dumps({"hooks": ["not an object"]}),
           json.dumps({"hooks": {"Stop": {"not": "a list"}}}))
    for text in bad:
        settings.write_text(text)
        p = run_apply_hooks(tmp_path, hook)
        assert p.returncode == 1
        assert p.stderr.count("\n") == 1
        assert p.stderr.startswith(f"apply_hooks: {settings}: expected ")
        assert settings.read_text() == text
        assert not list(claude_dir.glob("settings.json.bak.*"))
        assert not (tmp_path / ".codex").exists()


def test_second_write_failure_names_updated_files_and_backups(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.write_text(json.dumps({"theme": "dark"}))
    (tmp_path / ".codex").write_text("not a directory")
    hook = tmp_path / "stop_hook.py"
    p = run_apply_hooks(tmp_path, hook)
    assert p.returncode == 1
    assert p.stderr.count("\n") == 1
    assert "apply_hooks:" in p.stderr and "Traceback" not in p.stderr
    assert str(settings) in p.stderr
    assert "updated" in p.stderr and "backup" in p.stderr
    command = f"python3 {shlex.quote(str(hook))}"
    assert len(ours_entries(settings, command)) == 1
    (backup,) = claude_dir.glob("settings.json.bak.*")
    assert str(backup) in p.stderr  # the line names each backup path


@pytest.mark.parametrize("home_value", [None, ""], ids=["unset", "empty"])
def test_apply_hooks_without_home_exits_1_and_writes_nothing(tmp_path, home_value):
    hook = tmp_path / "stop_hook.py"
    env = {"PATH": os.environ["PATH"]}
    if home_value is not None:
        env["HOME"] = home_value
    p = subprocess.run([sys.executable, str(APPLY_HOOKS), str(hook)],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 1
    assert p.stderr == "apply_hooks: HOME is not set\n"
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".codex").exists()


# A hand edit that prefixes PATH=... to make dedup-check findable is still
# our entry: a merge re-run must replace it, not add a second one.
def test_is_ours_accepts_leading_env_assignments():
    hook = {"command": "PATH=$HOME/.local/bin:$PATH python3 /x/stop_hook.py"}
    assert apply_hooks.is_ours(hook, "python3 /x/stop_hook.py") is True
    assert apply_hooks.is_ours(hook, "python3 /x/stop_hook.py --harness codex") is False

def test_is_ours_still_rejects_another_tool():
    hook = {"command": "PATH=$HOME/.local/bin:$PATH python3 /other/stop_hook.py"}
    assert apply_hooks.is_ours(hook, "python3 /x/stop_hook.py") is False
    trailing = {"command": "python3 /x/stop_hook.py FOO=bar"}
    assert apply_hooks.is_ours(trailing, "python3 /x/stop_hook.py") is False
