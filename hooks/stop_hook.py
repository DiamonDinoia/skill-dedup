#!/usr/bin/env python3
"""Shared Stop hook for Claude Code and Codex: block once on blocking duplicates.

Reads the Stop event JSON on stdin, runs `dedup-check --json` (override with the
DEDUP_CHECK env var, tests only) and prints one JSON object on stdout:
- {"decision": "block", "reason": ...} when a finding has block=true and
  stop_hook_active is false. Works for both harnesses.
- Non-blocking advice. With --harness claude it is
  {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": ...}}.
  Codex rejects unknown fields on Stop, so --harness codex prints only
  {"systemMessage": ...}. The default is claude: only the Codex entry takes
  --harness codex.
- {} on the no-findings path: valid no-action output for both. Codex demands
  JSON: "Stop expects JSON on stdout when it exits 0."
A dedup-check error or timeout never blocks; it becomes one advice line.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from dedup_contract import validate

TIMEOUT = 570  # hook timeout is 600 s
MAX_SHOWN = 40


def reply(obj):
    json.dump(obj, sys.stdout)
    sys.stdout.write("\n")


def advice(text, harness):
    if not text:
        reply({})
    elif harness == "codex":
        # Codex 0.161.0 drops the whole output on any field it does not know.
        reply({"systemMessage": text})
    else:
        reply({"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": text}})


def fmt(f):
    n, e = f["new"], f["existing"]
    return (f"{n['path']}:{n['start']}-{n['end']} ~ "
            f"{e['path']}:{e['start']}-{e['end']} ({f.get('match')}, {f.get('score')})")


def listing(findings):
    lines = [fmt(f) for f in findings[:MAX_SHOWN]]
    if len(findings) > MAX_SHOWN:
        lines.append(f"... and {len(findings) - MAX_SHOWN} more")
    return "\n".join(lines)


def parse_findings(out):
    """Parse and validate dedup-check output via the shared contract. Returns
    (findings, None) or (None, error): a contract break must never break the hook."""
    try:
        # stdout is bytes; UnicodeDecodeError is a ValueError, same error path
        data = json.loads(out.decode())
    except ValueError:
        return None, "dedup-check printed invalid JSON"
    try:
        validate(data)
    except ValueError as err:
        return None, str(err)
    return data["findings"], None


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--harness", choices=("claude", "codex"), default="claude")
    harness = ap.parse_args().harness
    try:
        raw = json.load(sys.stdin)
    except (ValueError, EOFError):
        raw = None
    event = raw if isinstance(raw, dict) else {}
    cmd = [os.environ.get("DEDUP_CHECK", "dedup-check"), "--json"]
    # Claude Code and Codex send the session cwd in the Stop event; run there.
    cwd = event.get("cwd")
    if not isinstance(cwd, str) or not cwd or not os.path.isdir(cwd):
        cwd = None  # missing or unusable: keep the inherited cwd
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as err:
        advice(f"Duplicate check skipped: dedup-check failed ({err}).", harness)
        return 0
    if p.returncode != 0:
        # first non-empty line only, capped so a noisy dedup-check stays one line
        err = (p.stderr.decode(errors="replace").strip().splitlines() or [""])[0][:200]
        advice(f"Duplicate check skipped: dedup-check exited {p.returncode}: {err}", harness)
        return 0
    findings, err = parse_findings(p.stdout)
    if err:
        advice(f"Duplicate check skipped: {err}.", harness)
        return 0
    if not findings:
        reply({})
        return 0
    blocking = [f for f in findings if f["block"]]  # validated bool: "true" never reaches here
    if blocking and not event.get("stop_hook_active"):
        reply({"decision": "block",
               "reason": ("New work duplicates existing content:\n" + listing(blocking) +
                          "\nReuse or merge the existing code or text, "
                          "or explain why the duplicate stays.")})
        return 0
    advice("Possible duplicates of existing work:\n" + listing(findings), harness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
