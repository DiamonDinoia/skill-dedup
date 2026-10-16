#!/usr/bin/env python3
"""The one validator for the dedup-check --json contract.

Every consumer (stop_hook.py, hermes-prereview) calls validate(data) on the
parsed JSON. It returns the data or raises ValueError naming the field path.
Anything outside the contract is a dedup-check error and must never block.
"""


import math
from typing import NoReturn


def _err(path, msg) -> NoReturn:
    raise ValueError(f"{path} {msg}")


def _check_int(v, path):
    # bool is an int subclass; reject it with type identity, not isinstance
    if type(v) is not int:
        _err(path, f"must be an int, got {type(v).__name__}")


def _check_span(seg, path, want_kind):
    if not isinstance(seg, dict):
        _err(path, f"must be an object, got {type(seg).__name__}")
    if not isinstance(seg.get("path"), str):
        _err(f"{path}.path", f"must be a string, got {type(seg.get('path')).__name__}")
    _check_int(seg.get("start"), f"{path}.start")
    _check_int(seg.get("end"), f"{path}.end")
    if seg["start"] < 1:
        _err(f"{path}.start", "must be >= 1")
    if seg["end"] < seg["start"]:
        _err(f"{path}.end", "must be >= start")
    if want_kind and seg.get("kind") not in ("code", "text"):
        _err(f"{path}.kind", f"must be \"code\" or \"text\", got {seg.get('kind')!r}")


def validate(data):
    if not isinstance(data, dict):
        _err("top level", f"must be an object, got {type(data).__name__}")
    base = data.get("base")
    if base is not None and not isinstance(base, str):
        _err("base", f"must be a string or null, got {type(base).__name__}")
    findings = data.get("findings")
    if not isinstance(findings, list):
        _err("findings", f"must be a list, got {type(findings).__name__}")
    for i, f in enumerate(findings):
        p = f"findings[{i}]"
        if not isinstance(f, dict):
            _err(p, f"must be an object, got {type(f).__name__}")
        _check_span(f.get("new"), f"{p}.new", want_kind=True)
        _check_span(f.get("existing"), f"{p}.existing", want_kind=False)
        if f.get("match") not in ("exact", "lexical", "semantic"):
            _err(f"{p}.match", f"must be exact, lexical or semantic, got {f.get('match')!r}")
        score = f.get("score")
        if type(score) not in (int, float):
            _err(f"{p}.score", f"must be a number, got {type(score).__name__}")
        # math.isfinite raises OverflowError on an int too large for float
        try:
            finite = math.isfinite(score)
        except OverflowError:
            finite = False
        if not finite:
            _err(f"{p}.score", "must be finite")
        # block decides whether a finding blocks: truthiness is not enough,
        # "true" or 1 from a broken dedup-check must raise, never block
        if not type(f.get("block")) is bool:
            _err(f"{p}.block", f"must be a bool, got {type(f.get('block')).__name__}")
    return data
