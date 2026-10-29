# Install the optional dedup Stop hooks and the Hermes pre-review step by hand

scripts/install.sh automates these steps: it installs the CLI, downloads the
model, copies the hook scripts out of the clone, and patches
hermes-adversarial-review with --apply. It merges the harness config only with
`--stop-hook --apply`. Use the steps below for a fully manual setup.

1. Replace /ABS/PATH/TO in each snippet with the real path of this repo.
2. Claude Code: merge the Stop entry of claude-settings.json into ~/.claude/settings.json under "hooks". Keep existing entries.
3. Codex: merge the Stop entry of codex-hooks.json into ~/.codex/hooks.json under "hooks". Keep existing entries. Codex will ask to trust the changed file on next start.
4. The snippets only add a Stop hook that runs stop_hook.py with a 600 s timeout. They do not remove other hooks.
5. Hermes: link hermes-prereview into the bin dir of dedup-check, for example `ln -sf ~/.local/share/skill-dedup/hooks/hermes-prereview ~/.local/bin/hermes-prereview`. The review runs it by bare name.
6. stop_hook.py, hermes-prereview and dedup_contract.py must stay in one directory (symlinks are fine): both scripts import dedup_contract from their own real directory.
7. Then apply hermes-adversarial-review.patch: `patch -p1 -d ~/.local/bin < hooks/hermes-adversarial-review.patch`. Take a backup first.
8. The patch runs hermes-prereview once before the review and appends its findings to the review prompt. A blocking result (exit 3) does not abort the review: the reviewer sees the `[block]` items as high severity. A prereview error, crash, missing binary or timeout prints one stderr line, adds a "did not run" note to the prompt, and the review continues. Pass --no-dedup to skip.
9. All pieces need `dedup-check` on PATH, run from a git repo.
10. Test one harness before use: send a Stop event JSON to stop_hook.py on stdin and check the exit code and stdout. Pass --harness codex for the Codex entry: it prints {"systemMessage": ...} for advice.
