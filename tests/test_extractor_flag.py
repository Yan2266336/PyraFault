"""`--extractor` on stage 12 and stage 13, and the check that stops them disagreeing.

The ablation runs the same freeze under uni, dinov2 and handcrafted. Each needs its own
window banks, and the run that retrieves against them has to be using the same extractor
that built them -- otherwise every similarity is computed between vectors from two
different models. Nothing about that is visible at runtime, so it is checked.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pyrafault.verify.bank import bank_dir, write_bank_manifest

ROOT = Path(__file__).resolve().parents[1]


def _load(stage: str):
    spec = importlib.util.spec_from_file_location(f"s_{stage}", ROOT / "scripts" / stage)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage13 = _load("13_run_verifier.py")


class TestTheTwoStagesAgreeOnWhereBanksLive:
    def test_neither_builds_the_path_by_hand(self):
        """Stage 12 writes and stage 13 reads; if they each join the path themselves they
        can drift. Both must call the one helper.

        The pattern banned is the path join, not the bare word: stage 13 legitimately
        compares a verifier's `name` against the string `window_bank` to keep a plugin
        out of the directory stage 12 owns.
        """
        for name in ("12_build_window_bank.py", "13_run_verifier.py"):
            src = (ROOT / "scripts" / name).read_text()
            assert '"06_verify" / "window_bank"' not in src, (
                f"{name} still joins the bank path by hand"
            )

    def test_the_helper_is_what_they_import(self):
        for name in ("12_build_window_bank.py", "13_run_verifier.py"):
            assert "bank_dir" in (ROOT / "scripts" / name).read_text()


class TestTheManifestGuard:
    """`_check_bank_extractor` is the gate stage 13 runs before retrieving."""

    def test_a_matching_manifest_passes(self, tmp_path):
        d = bank_dir(tmp_path, "uni")
        write_bank_manifest(d, extractor="uni", n_banks=3)
        stage13._check_bank_extractor(d, "uni")      # no raise

    def test_a_mismatched_manifest_raises_naming_both(self, tmp_path):
        d = bank_dir(tmp_path, "uni")
        write_bank_manifest(d, extractor="dinov2", n_banks=3)
        with pytest.raises(SystemExit, match="dinov2"):
            stage13._check_bank_extractor(d, "uni")

    def test_a_directory_with_no_manifest_raises_rather_than_being_trusted(self, tmp_path):
        """Banks built before the manifest existed are not silently accepted: the whole
        point is that an unlabelled bank cannot be told apart from another model's."""
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        with pytest.raises(SystemExit, match="manifest"):
            stage13._check_bank_extractor(d, "uni")

    def test_the_refusal_says_how_to_fix_it(self, tmp_path):
        d = bank_dir(tmp_path, "uni")
        d.mkdir(parents=True)
        with pytest.raises(SystemExit) as exc:
            stage13._check_bank_extractor(d, "uni")
        assert "12" in str(exc.value)      # points at the stage that writes it


class TestTheFlagDefaultsToTheConfig:
    def test_both_stages_declare_the_flag(self):
        for name in ("12_build_window_bank.py", "13_run_verifier.py"):
            assert "--extractor" in (ROOT / "scripts" / name).read_text()

    def test_the_default_is_the_config_not_a_literal(self):
        """Hard-coding `uni` as the default would make every existing command change
        meaning the day a freeze is built with something else."""
        for name in ("12_build_window_bank.py", "13_run_verifier.py"):
            src = (ROOT / "scripts" / name).read_text()
            i = src.index("--extractor")
            window = src[i:i + 400]
            assert "default=None" in window, f"{name}: --extractor must default to None"
            assert "visual_sampling.extractor" in src


class TestRomaIsSelectable:
    """`--matcher roma` has to exist in the base environment even though RoMa does not.

    `RomaMatcher` defers `import romatch` into `__init__`, so the class can be imported
    and registered anywhere; only constructing it needs the overlay. That split is what
    lets the matcher appear in `--matcher`'s choices on every machine while the overlay
    stays optional.
    """

    def test_roma_is_a_choice(self):
        assert "roma" in stage13.MATCHERS

    def test_the_three_matchers_are_distinct_classes(self):
        assert len({id(c) for c in stage13.MATCHERS.values()}) == len(stage13.MATCHERS)

    def test_importing_the_class_does_not_need_the_overlay(self):
        """Importing stage 13 at all would fail otherwise, on every machine."""
        from pyrafault.verify.matchers import RomaMatcher
        assert RomaMatcher.name == "roma"

    def test_constructing_it_without_the_overlay_says_what_to_do(self):
        import importlib.util
        if importlib.util.find_spec("romatch") is not None:
            pytest.skip("overlay present; the helpful-failure path cannot be exercised")
        from pyrafault.verify.matchers import RomaMatcher
        with pytest.raises(ImportError, match="requirements-roma"):
            RomaMatcher(device="cpu")
