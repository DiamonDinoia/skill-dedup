"""eval/run.py --mutate-lexical writes complete rows with no model loads."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CODE_A = "def compute_total(price, count):\n    subtotal = price * count\n    tax = subtotal * 0.2\n    return subtotal + tax\n"
CODE_B = "def sum_amount(rate, units):\n    base = rate * units\n    fee = base * 0.2\n    return base + fee\n"
CODE_Z = "import os\nfor path in os.listdir('/tmp'):\n    print(path)\n"
TEXT_A = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
TEXT_B = "alpha bravo charlie delta echo foxtrot golf hotel india kilo"
TEXT_Z = "zebra yucca xylem widget vortex umber tundra spline"


def _pairs(kind, a, b, z):
    return [
        dict(id=f"{kind}-dev-pos", kind=kind, split="dev", subset="train", label=1, a=a, b=b),
        dict(id=f"{kind}-dev-neg", kind=kind, split="dev", subset="train", label=0, a=a, b=z),
        dict(id=f"{kind}-test-pos", kind=kind, split="test",
             subset="refactored" if kind == "code" else "paraphrase", label=1, a=a, b=b),
        dict(id=f"{kind}-test-neg", kind=kind, split="test",
             subset="hard_negative", label=0, a=b, b=z),
    ]


def _load_run():
    spec = importlib.util.spec_from_file_location("eval_run", ROOT / "eval" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mutate_lexical_writes_rows(tmp_path, monkeypatch):
    run = _load_run()
    pairs = _pairs("code", CODE_A, CODE_B, CODE_Z) + _pairs("text", TEXT_A, TEXT_B, TEXT_Z)
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("\n".join(json.dumps(p) for p in pairs))

    monkeypatch.setattr(run, "HERE", tmp_path)          # results land in tmp_path
    monkeypatch.setattr(run, "CACHE", tmp_path / "no-cache")

    def _no_child(*a, **kw):
        raise AssertionError("mutate mode must not spawn a model child")

    monkeypatch.setattr(run, "child", _no_child)
    monkeypatch.setattr(sys, "argv", ["run.py", "--mutate-lexical", "--pairs", str(pairs_path)])

    with pytest.raises(SystemExit) as exc:  # mutated models tie lexical: beats_lexical False
        run.main()
    assert exc.value.code == 1

    results = json.loads((tmp_path / "results.json").read_text())
    assert results["mutated"] is True
    load_keys = ("load_s", "ms_per_chunk", "ms_per_chunk_trunc512", "rss_mb",
                 "weights_mb", "max_seq_length", "mean_tokens")
    for kind in ("code", "text"):
        rows = results["classes"][kind]["rows"]
        assert rows[0]["name"] == "lexical (baseline)"
        assert [r["name"] for r in rows[1:]] == [m["name"] for m in run.MODELS[kind]]
        for row in rows[1:]:
            assert "sem_recall" in row
            for k in load_keys:  # mutate rows have no load-time fields; they must not KeyError
                assert k in row
            assert row["max_seq_length"] is None
            assert row["mean_tokens"] is None
    md = (tmp_path / "results.md").read_text()
    assert "lexical (baseline)" in md
