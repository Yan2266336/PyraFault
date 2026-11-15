"""`scripts/13_run_verifier.py`'s startup bank check, in both directions.

The loop that loads window banks used to reject a bank file naming no context in this
freeze (one direction) but never checked the reverse -- a context with no bank at all.
Stage 12 has no resume, so an interrupted build leaves a directory that names a real
subset of contexts and passes the one-directional check cleanly, then dies hours into
a real run with a bare `KeyError` from `zoomtrace.py` the moment `localise` looks up a
bankless context. `_load_banks` is the factored-out startup check `main()` calls;
these tests exercise it directly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyrafault.geom import BBoxL0
from pyrafault.verify.bank import WindowBank, bank_key

ROOT = Path(__file__).resolve().parents[1]


def _load_stage13():
    path = ROOT / "scripts" / "13_run_verifier.py"
    spec = importlib.util.spec_from_file_location(
        "stage13_run_verifier_bank_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage13 = _load_stage13()


def _write_bank(bank_dir: Path, context_id: str) -> None:
    WindowBank(context_id, side=8, origins=[(0, 0)],
              embeddings=np.ones((1, 4), dtype=np.float16)).save(
        bank_dir / f"{context_id}.npz"
    )


def _ctx_meta(*context_ids) -> dict:
    return {cid: ("S1", BBoxL0(0, 0, 100, 100)) for cid in context_ids}


def test_a_context_with_no_bank_raises_naming_the_count_and_a_sample(tmp_path):
    bank_dir = tmp_path / "window_bank"
    bank_dir.mkdir()
    _write_bank(bank_dir, "c0")
    ctx_meta = _ctx_meta("c0", "c1")  # c1 has no bank file
    with pytest.raises(SystemExit, match="1.*c1"):
        stage13._load_banks(bank_dir, ctx_meta)


def test_the_raise_names_every_missing_context_up_to_the_bound(tmp_path):
    bank_dir = tmp_path / "window_bank"
    bank_dir.mkdir()
    ctx_meta = _ctx_meta(*[f"c{i}" for i in range(15)])  # none have banks
    with pytest.raises(SystemExit) as exc:
        stage13._load_banks(bank_dir, ctx_meta)
    assert "15" in str(exc.value)


def test_a_bank_naming_no_context_in_the_set_is_now_tolerated(tmp_path):
    """This replaces an assertion that it raises, deliberately.

    `ctx_meta` is scoped to the split being run, while the bank directory holds whatever
    has been built -- for the lung freeze, banks for `calibration` and `test` side by side,
    and `dev` not built at all. A stray-bank check cannot tell "a bank for the other split"
    from "a bank for a context that does not exist", so it would force one bank directory
    per split to buy nothing. The direction that protects a run is the other one, which the
    tests above cover.
    """
    bank_dir = tmp_path / "window_bank"
    bank_dir.mkdir()
    _write_bank(bank_dir, "c0")
    _write_bank(bank_dir, "other_split_ctx")

    banks = stage13._load_banks(bank_dir, _ctx_meta("c0"))

    assert len(banks) == 1


def test_a_complete_bank_set_loads_without_error(tmp_path):
    bank_dir = tmp_path / "window_bank"
    bank_dir.mkdir()
    _write_bank(bank_dir, "c0")
    _write_bank(bank_dir, "c1")
    ctx_meta = _ctx_meta("c0", "c1")
    bank = stage13._load_banks(bank_dir, ctx_meta)
    assert set(bank) == {bank_key("S1", BBoxL0(0, 0, 100, 100))}


def test_no_banks_at_all_raises_the_original_message(tmp_path):
    bank_dir = tmp_path / "window_bank"
    bank_dir.mkdir()
    with pytest.raises(SystemExit, match="run stage 12 first"):
        stage13._load_banks(bank_dir, {})



class TestSplitScopedCompleteness:
    """The check must cover the contexts the run will touch, not every context in the
    freeze. The lung freeze holds 24,113 contexts of which 16,644 are `dev`; demanding a
    bank for those would make an evaluation unrunnable without three times the
    bank-building it needs, for contexts no evaluated split ever reads. The scoping is the
    caller's -- it restricts `ctx_meta` to the split -- so what these pin is that
    `_load_banks` demands exactly what it is given and tolerates extra bank files.
    """

    def test_a_bank_for_a_context_outside_the_given_set_is_tolerated(self, tmp_path):
        bank_dir = tmp_path / "window_bank"
        bank_dir.mkdir()
        _write_bank(bank_dir, "c_test")
        _write_bank(bank_dir, "c_dev")      # another split's bank, left over on disk

        banks = stage13._load_banks(bank_dir, _ctx_meta("c_test"))

        assert len(banks) == 1

    def test_a_missing_bank_inside_the_given_set_still_raises(self, tmp_path):
        bank_dir = tmp_path / "window_bank"
        bank_dir.mkdir()
        _write_bank(bank_dir, "c_a")

        with pytest.raises(SystemExit, match="c_b"):
            stage13._load_banks(bank_dir, _ctx_meta("c_a", "c_b"))
