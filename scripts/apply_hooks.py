#!/usr/bin/env python3
"""Merge the dedup Stop hook into the Claude Code and Codex hook files.

Usage: apply_hooks.py [--print] /abs/path/to/hooks/stop_hook.py

--print writes the two harness JSON blocks to stdout and touches nothing:
the Claude block first, then the Codex block (--harness codex). With no
flag each existing file is backed up to <name>.bak.<utc-timestamp> and
updated in place. A symlinked file is resolved first: the target is backed
up and replaced, the symlink stays. Idempotent: re-runs leave exactly one
dedup Stop entry; the other keys and the hooks of other tools are kept.
Called by install.sh.
"""
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def dedup_command(stop_hook: str, harness: str) -> str:
    # The one builder of the command. --print and the write both use it.
    # Only the Codex entry takes --harness codex: Codex needs the codex shape.
    base = f"python3 {shlex.quote(stop_hook)}"
    return base + (" --harness codex" if harness == "codex" else "")


def dedup_group(command: str) -> dict:
    return {"hooks": [{
        "type": "command",
        "command": command,
        "timeout": 600,
    }]}


def is_ours(hook, command: str) -> bool:
    # Compare the split words, not the strings: an older entry may quote our
    # path differently. A substring match would claim another tool's own
    # stop_hook.py and drop it.
    if not isinstance(hook, dict):
        return False
    value = hook.get("command")
    if not isinstance(value, str):
        return False
    try:
        return shlex.split(value) == shlex.split(command)
    except ValueError:
        return False


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as err:
        raise ValueError(f"{path}: not valid JSON: {err}") from err
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level")
    hooks = data.get("hooks")
    if hooks is not None and not isinstance(hooks, dict):
        raise ValueError(f"{path}: expected hooks to be a JSON object")
    if isinstance(hooks, dict):
        stop = hooks.get("Stop")
        if stop is not None and not isinstance(stop, list):
            raise ValueError(f"{path}: expected hooks.Stop to be a JSON array")
    return data


def add_ours(data: dict, command: str, ours: tuple) -> None:
    """Put one dedup group into data and take our old entries out.

    Inside each Stop group only our hook dicts are removed. ours carries
    both entry shapes: without it an entry from an install before the
    --harness flag would stay and run the wrong harness mode. A group is
    dropped when its hook list becomes empty. The other dicts stay as they
    are.
    """
    hooks = data.setdefault("hooks", {})
    stop = []
    for group in hooks.get("Stop", []):
        inner = group.get("hooks") if isinstance(group, dict) else None
        if isinstance(inner, list):
            kept = [h for h in inner if not any(is_ours(h, c) for c in ours)]
            if not kept:
                continue
            group["hooks"] = kept
        stop.append(group)
    stop.append(dedup_group(command))
    hooks["Stop"] = stop


def backup(path: Path) -> Path:
    # Microseconds: two runs in one second must not overwrite one backup.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = path.with_name(path.name + ".bak." + stamp)
    shutil.copy(path, out)
    return out


def render(data: dict) -> bytes:
    # ensure_ascii=False: the user's own characters must come back as they
    # were, not as \uXXXX escapes.
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_atomic(path: Path, content: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> int:
    args = sys.argv[1:]
    printing = bool(args) and args[0] == "--print"
    if printing:
        args = args[1:]
    if len(args) != 1:
        print("usage: apply_hooks.py [--print] /abs/path/to/hooks/stop_hook.py",
              file=sys.stderr)
        return 2
    ours = tuple(dedup_command(args[0], h) for h in ("claude", "codex"))
    if printing:
        for command in ours:
            print(json.dumps({"hooks": {"Stop": [dedup_group(command)]}}, indent=2))
        return 0
    home_env = os.environ.get("HOME")
    if not home_env:
        # Path.home() would fall back to the real home and write there.
        print("apply_hooks: HOME is not set", file=sys.stderr)
        return 1
    home = Path(home_env)
    paths = [((home / ".claude" / "settings.json").resolve(), ours[0]),
             ((home / ".codex" / "hooks.json").resolve(), ours[1])]
    built = []
    try:
        # Parse and build everything before any file is written.
        for p, command in paths:
            data = load(p)
            add_ours(data, command, ours)
            built.append((p, data))
    except (OSError, ValueError) as err:
        print(f"apply_hooks: {err}", file=sys.stderr)
        return 1
    updated = []
    backups = []
    for p, data in built:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            content = render(data)
            if p.exists() and p.read_bytes() == content:
                # No change: a rewrite only churns mtime and backups.
                print(f"SKIP {p}")
                continue
            if p.exists():
                backups.append(str(backup(p)))
            write_atomic(p, content)
            updated.append(str(p))
        except OSError as err:
            msg = f"apply_hooks: {err}"
            if updated:
                msg += (f"; updated {', '.join(updated)}"
                        f"; backups: {', '.join(backups) or 'none'}")
            print(msg, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
