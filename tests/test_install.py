"""Tests for scripts/install.sh: a temp HOME, a fake uv, and a fake python3."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "install.sh"
APPLY_HOOKS = ROOT / "scripts" / "apply_hooks.py"

# Minimal PATH that has the core tools but not uv: forces the pip branch.
NO_UV_PATH = "/usr/local/bin:/usr/bin:/bin"


def base_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
    env["HOME"] = str(home)
    return env


def fake_nvidia_smi(bindir: Path, works: bool):
    """A nvidia-smi that lists one GPU (exit 0) or fails like a missing driver."""
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/bin/sh\n" + ("echo 'GPU 0: Fake (UUID: x)'\n" if works
                                    else "echo 'NVIDIA-SMI has failed' >&2\nexit 9\n"))
    smi.chmod(0o755)


def run_install(home: Path, *args: str, gpu: bool = False):
    """Run install.sh against a temp HOME with a fake uv first on PATH."""
    bindir = home / "bin"
    bindir.mkdir(exist_ok=True)
    fake_nvidia_smi(bindir, gpu)
    log = home / "uv.log"
    uv = bindir / "uv"
    # keep a copy of the --excludes file: install.sh deletes it after the call
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n'
                  f'while [ $# -gt 0 ]; do [ "$1" = --excludes ] && cp "$2" "{home}/excludes.copy"; shift; done\n')
    uv.chmod(0o755)
    env = base_env(home)
    env["PATH"] = f"{bindir}{os.pathsep}{os.environ['PATH']}"
    p = subprocess.run(["bash", str(INSTALL), *args],
                       capture_output=True, text=True, env=env)
    return p, log.read_text().split()


FAKE_PYTHON3 = """#!/bin/sh
# Fake python3 for the pip branch: real interpreter for -c, stub venv.
if [ -n "$FAKE_PY_SITE" ]; then
    export PYTHONPATH="$FAKE_PY_SITE"
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    if [ "$3" = "--help" ]; then exit 0; fi
    mkdir -p "$3/bin"
    printf '#!/bin/sh\\nexit 0\\n' > "$3/bin/python"
    chmod +x "$3/bin/python"
    exit 0
fi
exec "{real}" "$@"
"""


def make_fake_python3(home: Path, spoof_310: bool):
    """A python3 on PATH that is real (>= 3.11) or spoofs version 3.10."""
    bindir = home / "pybin"
    bindir.mkdir(exist_ok=True)
    env = base_env(home)
    env.pop("FAKE_PY_SITE", None)
    if spoof_310:
        site = home / "pysite"
        site.mkdir(exist_ok=True)
        (site / "sitecustomize.py").write_text(
            "import sys; sys.version_info = (3, 10, 8, 'final', 0)\n")
        env["FAKE_PY_SITE"] = str(site)
    fake_nvidia_smi(bindir, False)
    py = bindir / "python3"
    py.write_text(FAKE_PYTHON3.replace("{real}", sys.executable))
    py.chmod(0o755)
    env["PATH"] = f"{bindir}{os.pathsep}{NO_UV_PATH}"
    return env


def run_pip_install(home: Path, spoof_310: bool, *args: str):
    env = make_fake_python3(home, spoof_310)
    return subprocess.run(["bash", str(INSTALL), *args],
                          capture_output=True, text=True, env=env)


def ours_command(home: Path, data_home: "Path | None" = None, harness: str = "claude") -> str:
    data = data_home or home / ".local/share"
    cmd = f"python3 {shlex.quote(str(data / 'skill-dedup/hooks/stop_hook.py'))}"
    return cmd + (" --harness codex" if harness == "codex" else "")


def ours_entries(path: Path, command: str):
    data = json.loads(path.read_text())
    return [g for g in data["hooks"]["Stop"]
            if any(h.get("command") == command for h in g.get("hooks", []))]


def run_apply_hooks(home: Path, hook: Path):
    env = base_env(home)
    return subprocess.run([sys.executable, str(APPLY_HOOKS), str(hook)],
                          capture_output=True, text=True, env=env)


def run_apply_without_python3(home: Path):
    bindir = home / "nobin"
    bindir.mkdir(exist_ok=True)
    env = base_env(home)
    env["PATH"] = str(bindir)
    return subprocess.run(["/bin/bash", str(INSTALL), "--apply", "--stop-hook", "--no-model"],
                          capture_output=True, text=True, env=env)


def printed_jsons(stdout: str) -> list:
    """All top level JSON blocks in the install.sh output, in order."""
    dec, docs, i = json.JSONDecoder(), [], 0
    while True:
        start = stdout.find("{", i)
        if start < 0:
            return docs
        doc, end = dec.raw_decode(stdout, start)
        docs.append(doc)
        i = end


def test_no_apply_installs_with_uv_and_writes_no_harness_files(tmp_path):
    p, uv_args = run_install(tmp_path, "--no-model")
    assert p.returncode == 0, p.stderr
    i = uv_args.index("tool")
    assert uv_args[i:i + 3] == ["tool", "install", "--force"]
    assert uv_args[i + 3] == str(ROOT)  # the clone dir, from the script path
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".codex").exists()
    hook_dir = tmp_path / ".local/share/skill-dedup/hooks"
    for name in ("stop_hook.py", "dedup_contract.py", "hermes-prereview"):
        assert (hook_dir / name).exists()
    # the Stop hook is opt-in: no JSON printed without --stop-hook
    assert "stop_hook.py" not in p.stdout and "hookSpecificOutput" not in p.stdout
    # hermes-adversarial-review runs "hermes-prereview" by bare name from the same bin dir
    link = tmp_path / ".local/bin/hermes-prereview"
    assert link.is_symlink() and link.resolve() == (hook_dir / "hermes-prereview").resolve()


def install_spec(uv_args):
    return uv_args[uv_args.index("install") + 2]


def test_working_gpu_installs_the_gpu_extra(tmp_path):
    p, uv_args = run_install(tmp_path, "--no-model", gpu=True)
    assert p.returncode == 0, p.stderr
    assert install_spec(uv_args) == f"{ROOT}[gpu]"
    assert uv_args[uv_args.index("--excludes") - 1] == f"{ROOT}[gpu]"
    # the CPU build must not resolve beside the GPU one
    assert (tmp_path / "excludes.copy").read_text().split() == ["fastembed", "onnxruntime"]
    assert "2.5 GB" in p.stderr


def test_cpu_flag_overrides_a_working_gpu(tmp_path):
    p, uv_args = run_install(tmp_path, "--no-model", "--cpu", gpu=True)
    assert p.returncode == 0, p.stderr
    assert install_spec(uv_args) == str(ROOT) and "--excludes" not in uv_args


def test_broken_nvidia_smi_installs_the_cpu_extra(tmp_path):
    # the binary exists but exits non-zero: no working GPU
    p, uv_args = run_install(tmp_path, "--no-model", gpu=False)
    assert (tmp_path / "bin/nvidia-smi").exists()
    assert install_spec(uv_args) == str(ROOT) and "--excludes" not in uv_args


def make_review(home: Path, patch_name: str | None) -> Path:
    """The fixture review script in a temp HOME, patched with hooks/<patch_name> or unpatched."""
    review = home / ".local/bin/hermes-adversarial-review"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text((ROOT / "tests/fixtures/review-script").read_text())
    if patch_name:
        subprocess.run(["patch", "-p1", "-d", str(review.parent), "-i", str(ROOT / "hooks" / patch_name)],
                       check=True, capture_output=True)
    return review


def test_apply_patches_hermes_only_and_is_idempotent(tmp_path):
    review = make_review(tmp_path, None)
    p, _ = run_install(tmp_path, "--no-model", "--apply")
    assert p.returncode == 0 and "applied hermes-adversarial-review.patch" in p.stderr
    patched = review.read_text()
    assert "dedup_section" in patched
    assert not (tmp_path / ".claude").exists() and not (tmp_path / ".codex").exists()
    p, _ = run_install(tmp_path, "--no-model", "--apply")
    assert p.returncode == 0 and "already applied" in p.stderr
    assert review.read_text() == patched


def test_apply_upgrades_a_script_with_the_previous_patch(tmp_path):
    review = make_review(tmp_path, "hermes-adversarial-review.v0.patch")
    assert "sys.exit(4)" in review.read_text() and "dedup_section" not in review.read_text()
    p, _ = run_install(tmp_path, "--no-model", "--apply")
    assert p.returncode == 0, p.stderr
    assert "upgraded hermes-adversarial-review.patch" in p.stderr
    new = review.read_text()
    assert "dedup_section" in new and "sys.exit(4)" not in new
    # same bytes as a direct application of the new patch to the unpatched script
    direct = make_review(tmp_path / "ref", "hermes-adversarial-review.patch")
    assert new == direct.read_text()
    p, _ = run_install(tmp_path, "--no-model", "--apply")
    assert "already applied" in p.stderr


def test_unknown_review_script_is_skipped_untouched(tmp_path):
    review = make_review(tmp_path, None)
    review.write_text("#!/bin/sh\necho unrelated\n")
    p, _ = run_install(tmp_path, "--no-model", "--apply")
    assert p.returncode == 0 and "skip hermes patch" in p.stderr
    assert review.read_text() == "#!/bin/sh\necho unrelated\n"


def test_stop_hook_flag_prints_the_hook_config(tmp_path):
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook")
    assert p.returncode == 0 and "stop_hook.py" in p.stdout
    assert not (tmp_path / ".claude").exists()


def test_apply_twice_is_idempotent_and_backs_up(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.write_text(json.dumps({"theme": "dark"}))
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    hooks_file = codex_dir / "hooks.json"
    hooks_file.write_text(json.dumps({"note": "keep me"}))
    for _ in range(2):
        p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
        assert p.returncode == 0, p.stderr
        # The hermes review script is absent in the temp HOME.
        assert "skip hermes patch" in p.stderr
    for f, command in ((settings, ours_command(tmp_path)),
                       (hooks_file, ours_command(tmp_path, harness="codex"))):
        entries = ours_entries(f, command)
        assert len(entries) == 1
        assert entries[0]["hooks"][0]["command"] == command
    assert json.loads(settings.read_text())["theme"] == "dark"
    assert json.loads(hooks_file.read_text())["note"] == "keep me"
    assert list(claude_dir.glob("settings.json.bak.*"))
    assert list(codex_dir.glob("hooks.json.bak.*"))


def test_version_gate_accepts_current_python_and_rejects_310(tmp_path):
    # The current interpreter is >= 3.11: the pip branch must pass the gate.
    p = run_pip_install(tmp_path, False, "--no-model")
    assert p.returncode == 0, p.stderr
    assert (tmp_path / ".local/share/skill-dedup/venv").is_dir()

    home2 = tmp_path / "old"
    home2.mkdir()
    # A python3 that behaves as 3.10: the gate must fail before any venv.
    p = run_pip_install(home2, True, "--no-model")
    assert p.returncode != 0
    assert "3.11" in p.stderr
    assert not (home2 / ".local/share/skill-dedup/venv").exists()


def test_no_ensurepip_fails_before_creating_venv(tmp_path):
    # On Debian without python3-venv, `python3 -m venv --help` still works,
    # so only an import check catches it before anything is created.
    env = make_fake_python3(tmp_path, False)
    site = tmp_path / "pysite_noensurepip"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(
        "import sys; sys.version_info = (3, 12, 0, 'final', 0)\n")
    (site / "ensurepip.py").write_text(
        "raise ImportError(\"No module named 'ensurepip'\")\n")
    env["FAKE_PY_SITE"] = str(site)
    p = subprocess.run(["bash", str(INSTALL), "--no-model"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 1
    assert "install.sh: python3 cannot create a venv; install the " \
           "python3-venv package (Debian/Ubuntu: apt install python3-venv)" \
           in p.stderr
    assert not (tmp_path / ".local/share/skill-dedup/venv").exists()


def test_other_tools_stop_hook_is_kept(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    other = {"hooks": [{"type": "command",
                        "command": "python3 /other-tool/stop_hook.py"}]}
    settings.write_text(json.dumps({
        "theme": "dark",
        "hooks": {"Stop": [other]},
    }))
    for _ in range(2):
        p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
        assert p.returncode == 0, p.stderr
    data = json.loads(settings.read_text())
    stop = data["hooks"]["Stop"]
    assert other in stop  # the other tool's hook survives
    assert len(ours_entries(settings, ours_command(tmp_path))) == 1
    assert data["theme"] == "dark"


def test_hook_path_with_space_is_quoted(tmp_path):
    data_home = tmp_path / "data with space"
    env_home = tmp_path / "home"
    env_home.mkdir()
    bindir = env_home / "bin"
    bindir.mkdir()
    log = env_home / "uv.log"
    uv = bindir / "uv"
    # keep a copy of the --excludes file: install.sh deletes it after the call
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n'
                  f'while [ $# -gt 0 ]; do [ "$1" = --excludes ] && cp "$2" "{env_home}/excludes.copy"; shift; done\n')
    uv.chmod(0o755)
    env = base_env(env_home)
    env["XDG_DATA_HOME"] = str(data_home)
    env["PATH"] = f"{bindir}{os.pathsep}{os.environ['PATH']}"
    p = subprocess.run(["bash", str(INSTALL), "--no-model", "--stop-hook", "--apply"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    hook = data_home / "skill-dedup/hooks/stop_hook.py"
    assert hook.exists()
    command = ours_command(env_home, data_home)
    settings = env_home / ".claude/settings.json"
    assert len(ours_entries(settings, command)) == 1
    # The written command, run through sh, finds the hook file.
    r = subprocess.run(["sh", "-c", command], input="{}",
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()  # the hook answered with one JSON object


def test_invalid_codex_json_leaves_claude_untouched(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.write_text(json.dumps({"theme": "dark"}))
    before = settings.read_text()
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "hooks.json").write_text("{not json")
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode != 0
    assert "apply_hooks:" in p.stderr and "Traceback" not in p.stderr
    # Claude settings were parsed ok but never written nor backed up.
    assert settings.read_text() == before
    assert not list(claude_dir.glob("settings.json.bak.*"))


def test_two_runs_in_one_second_keep_both_backups(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.write_text(json.dumps({"theme": "dark"}))
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    # A real change, still inside the same second: the second write must not
    # overwrite the first backup.
    settings.write_text(json.dumps({**json.loads(settings.read_text()),
                                    "theme": "light"}))
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    backups = list(claude_dir.glob("settings.json.bak.*"))
    assert len(backups) == 2
    assert len({b.name for b in backups}) == 2


def test_noop_apply_skips_the_write_and_keeps_user_bytes(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"note": "em dash — and ‘quotes’"}))
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir()
    hooks_file.write_text(json.dumps({"note": "keep me"}))
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    first = {f: f.read_bytes() for f in (settings, hooks_file)}
    # A merge writes the user's characters back as bytes, not \uXXXX escapes.
    assert "—".encode() in first[settings]
    assert "\\u2014" not in first[settings].decode()
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    for f in (settings, hooks_file):
        assert f"SKIP {f.resolve()}" in p.stdout
        assert f.read_bytes() == first[f]  # byte-identical on repeat runs
    # The no-op run wrote nothing, so it made no new backups either.
    baks = [b for d in (".claude", ".codex")
            for b in (tmp_path / d).glob("*.bak.*")]
    assert len(baks) == 2  # one per file, from the first run only


def test_symlinked_settings_target_is_updated_and_link_stays(tmp_path):
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    target = dotfiles / "claude-settings.json"
    target.write_text(json.dumps({"theme": "dark"}))
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    settings.symlink_to(target)
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    assert settings.is_symlink()
    data = json.loads(target.read_text())
    assert data["theme"] == "dark"  # earlier keys are kept
    assert len(ours_entries(target, ours_command(tmp_path))) == 1
    assert list(dotfiles.glob("claude-settings.json.bak.*"))


def stop_commands(path: Path) -> list:
    stop = json.loads(path.read_text())["hooks"]["Stop"]
    return [h.get("command") for g in stop for h in g.get("hooks", [])]


def test_printed_entries_are_what_apply_writes(tmp_path):
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook")
    assert p.returncode == 0, p.stderr
    claude_block, codex_block = printed_jsons(p.stdout)
    claude_cmd = claude_block["hooks"]["Stop"][0]["hooks"][0]["command"]
    codex_cmd = codex_block["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert claude_cmd == ours_command(tmp_path)
    assert codex_cmd == ours_command(tmp_path, harness="codex")
    # Merge the printed entries into the harness files by hand.
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({
        "theme": "dark",
        "hooks": {"Stop": list(claude_block["hooks"]["Stop"])},
    }))
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    hooks_file = codex_dir / "hooks.json"
    hooks_file.write_text(json.dumps({
        "note": "keep me",
        "hooks": {"Stop": list(codex_block["hooks"]["Stop"])},
    }))
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    assert stop_commands(settings) == [claude_cmd]
    assert stop_commands(hooks_file) == [codex_cmd]


def test_mixed_group_keeps_the_other_tool_hook(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    other = {"type": "command", "command": "python3 /other-tool/stop_hook.py"}
    ours = {"type": "command", "command": ours_command(tmp_path)}
    settings.write_text(json.dumps({
        "theme": "dark",
        "hooks": {"Stop": [{"hooks": [other, ours]}]},
    }))
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    stop = json.loads(settings.read_text())["hooks"]["Stop"]
    flat = [h for g in stop for h in g.get("hooks", [])]
    assert other in flat  # the group is edited, not deleted
    assert sum(1 for h in flat if h.get("command") == ours["command"]) == 1


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


def test_apply_without_python3_exits_before_any_write(tmp_path):
    p = run_apply_without_python3(tmp_path)
    assert p.returncode == 1
    assert "python3 not found" in p.stderr
    assert p.stderr.count("\n") == 1
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".local").exists()


def test_new_files_get_0644_and_existing_keeps_its_mode(tmp_path):
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    for f in (tmp_path / ".claude" / "settings.json",
              tmp_path / ".codex" / "hooks.json"):
        assert f.stat().st_mode & 0o777 == 0o644
    home2 = tmp_path / "keep"
    home2.mkdir()
    settings = home2 / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"theme": "dark"}))
    os.chmod(settings, 0o600)
    p, _ = run_install(home2, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    assert settings.stat().st_mode & 0o777 == 0o600


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


def test_apply_replaces_the_codex_entry_written_before_the_harness_flag(tmp_path):
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    hooks_file = codex_dir / "hooks.json"
    # An old install wrote the same command without --harness codex: it runs
    # the hook in claude mode and prints output Codex rejects. It must go.
    old = ours_command(tmp_path)
    hooks_file.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
        {"type": "command", "command": old, "timeout": 600}]}]}}))
    p, _ = run_install(tmp_path, "--no-model", "--stop-hook", "--apply")
    assert p.returncode == 0, p.stderr
    assert ours_entries(hooks_file, old) == []
    assert len(ours_entries(hooks_file, ours_command(tmp_path, harness="codex"))) == 1
