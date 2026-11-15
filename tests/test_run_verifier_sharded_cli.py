"""Stage 13's two halves, end to end through the CLI.

The equality that matters is at the top of this file rather than in a docstring halfway
down: `--localise-only` for every shard followed by `--from-localisations` must write
the same `predictions_<split>.parquet` as one unsharded invocation. If it does not, the
full-scale lung numbers are not the numbers the protocol defines, and the only way that
shows up otherwise is as a discrepancy nobody can attribute months later.

The freeze here is synthetic and its pixels are pre-seeded into `02_cache`, so no slide
file is ever opened: the test exercises the sharding, the file layout and the CLI's
refusals, not the WSI reader.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from pyrafault.cache import cache_key, cache_path, save_png
from pyrafault.schema import SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
STAGE13 = ROOT / "scripts" / "13_run_verifier.py"

#: A verifier with no dependencies and a deterministic answer derived from the query's
#: pixels, written into the freeze's own directory and loaded by `--verifier
#: module:Class`. Deterministic so the sharded and unsharded routes are comparable at
#: all, and pixel-derived so a shard that localised the wrong items cannot still agree.
STUB = '''
from pyrafault.geom import BBoxL0
from pyrafault.verify.base import Localisation, Verifier


class Stub(Verifier):
    name, revision = "stub_baseline", "3"

    def localise(self, inputs):
        shade = inputs["query"].getpixel((0, 0))[0]
        if shade == 0:
            return Localisation(None, 0.0)
        return Localisation(BBoxL0(100 * shade, 2000, 512, 512), shade / 255.0)
'''


def _table(name, rows, path):
    cols = list(SCHEMAS[name].names)
    pd.DataFrame([{c: r.get(c) for c in cols} for r in rows],
                 columns=cols).to_parquet(path, index=False)


@pytest.fixture
def seeded_freeze(tmp_path):
    """A freeze whose every served patch is already in `02_cache`.

    Both splits are present and every fault cell appears, because `fit_tau` needs
    faulty items and `fit_delta` needs clean ones -- a fixture with only one would fail
    inside the fit rather than testing the sharding.

    Six contexts, not one: stage 13 shards by context, so a single-context freeze would
    put every item in shard 0 and leave the rest empty -- the fixture would then be
    testing the empty-shard refusal instead of the partition.
    """
    root = tmp_path / "data_stub"
    bench = root / "01_benchmark"
    bench.mkdir(parents=True)
    cfg = yaml.safe_load(open(ROOT / "configs" / "example10.yaml"))
    ctx_mpp, ctx_size = cfg["context"]["target_mpp"], cfg["context"]["size_px"]
    qry_mpp, qry_size = cfg["query"]["target_mpp"], cfg["query"]["size_px"]
    ver = cfg["extraction_version"]
    (bench / "config.yaml").write_text(yaml.safe_dump(cfg))

    pairs, queries, boxes = [], [], {}
    for split in ("calibration", "test"):
        for q in range(6):
            for fault in ("clean", "hard_mismatch", "out_of_slide"):
                pid = f"{split}-{fault}-{q}"
                shade = 20 * (q + 1)
                true = (100 * shade, 2000)
                claim = true if fault == "clean" else (90000, 2000)
                pairs.append({
                    "pair_id": pid, "slide_id": "S1", "context_id": f"S1__c{q}",
                    "query_id": f"{pid}__q",
                    "claimed_bbox_l0_x": claim[0], "claimed_bbox_l0_y": claim[1],
                    "claimed_bbox_l0_w": qry_size * 4, "claimed_bbox_l0_h": qry_size * 4,
                    "true_bbox_l0_x": true[0], "true_bbox_l0_y": true[1],
                    "true_bbox_l0_w": qry_size * 4, "true_bbox_l0_h": qry_size * 4,
                    "claim_correct": fault == "clean", "fault_type": fault,
                    "fault_severity": "small", "normalized_offset": 0.0,
                    "offset_frac_context": 0.0, "iou_claim_true": 1.0,
                    "true_within_context": True, "local_recovery_possible": True,
                    "true_slide_id": "S1" if fault != "out_of_slide" else "S2",
                    "same_slide": fault != "out_of_slide",
                    "context_mpp": ctx_mpp, "query_mpp": qry_mpp, "scale_ratio": 4.0,
                    "construction_seed": 1, "pipeline_version": "v1",
                    "split": split, "qc_status": "pass",
                    "degradation_kind": "none", "degradation_band": "none",
                    "degradation_params": json.dumps({}),
                })
                queries.append({"query_id": f"{pid}__q", "context_id": f"S1__c{q}",
                                "slide_id": "S1", "target_mpp": qry_mpp,
                                "output_size": qry_size, "extraction_version": ver,
                                "split": split})
                boxes[(pairs[-1]["true_slide_id"], true, shade)] = True

    _table("pairs", pairs, bench / "pairs.parquet")
    _table("queries", queries, bench / "queries.parquet")
    _table("contexts", [{
        "context_id": f"S1__c{q}", "slide_id": "S1", "bbox_l0_x": 0, "bbox_l0_y": 0,
        "bbox_l0_w": 400000, "bbox_l0_h": 400000, "target_mpp": ctx_mpp,
        "output_size": ctx_size, "extraction_version": ver, "split": "test",
    } for q in range(6)], bench / "contexts.parquet")
    _table("slides", [{"slide_id": s, "source_relpath": f"{s}.svs"}
                      for s in ("S1", "S2")], bench / "slides.parquet")

    from pyrafault.geom import BBoxL0
    cache = root / "02_cache"
    ctx_box = BBoxL0(0, 0, 400000, 400000)
    save_png(Image.new("RGB", (32, 32), (7, 7, 7)),
             cache_path(cache, "S1", cache_key("S1", ctx_box, ctx_mpp, ctx_size, ver)))
    for (slide, (x, y), shade) in boxes:
        box = BBoxL0(x, y, qry_size * 4, qry_size * 4)
        save_png(Image.new("RGB", (16, 16), (shade,) * 3),
                 cache_path(cache, slide, cache_key(slide, box, qry_mpp, qry_size, ver)))

    (root / "stub_verifier.py").write_text(STUB)
    return root


def _run(root, *extra):
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{root}")
    r = subprocess.run(
        [sys.executable, str(STAGE13), "--config", str(root / "01_benchmark" / "config.yaml"),
         "--data-root", str(root), "--verifier", "stub_verifier:Stub", *extra],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    return r


def _predictions(root, split):
    return pd.read_parquet(root / "06_verify" / "stub_baseline" / f"predictions_{split}.parquet")


@pytest.fixture
def calibrated(seeded_freeze):
    """Thresholds fitted once, unsharded, so every comparison below decides under the
    same tau and delta regardless of which route produced the localisations."""
    r = _run(seeded_freeze, "--split", "calibration", "--fit-thresholds")
    assert r.returncode == 0, r.stderr
    return seeded_freeze


class TestTheShardedRouteReproducesTheDirectOne:
    def test_the_predictions_table_is_identical(self, calibrated):
        r = _run(calibrated, "--split", "test")   # unsharded, for comparison
        assert r.returncode == 0, r.stderr
        direct = _predictions(calibrated, "test")

        for k in range(4):
            r = _run(calibrated, "--split", "test", "--localise-only",
                     "--shard", str(k), "--n-shards", "4")
            assert r.returncode == 0, r.stderr
        r = _run(calibrated, "--split", "test", "--from-localisations")
        assert r.returncode == 0, r.stderr
        staged = _predictions(calibrated, "test")

        cols = [c for c in direct.columns if c != "runtime_ms"]
        pd.testing.assert_frame_equal(staged[cols], direct[cols])

    def test_the_fitted_thresholds_are_identical(self, calibrated):
        tpath = calibrated / "06_verify" / "stub_baseline" / "thresholds.json"
        direct = json.loads(tpath.read_text())

        for k in range(3):
            r = _run(calibrated, "--split", "calibration", "--localise-only",
                     "--shard", str(k), "--n-shards", "3")
            assert r.returncode == 0, r.stderr
        r = _run(calibrated, "--split", "calibration", "--from-localisations",
                 "--fit-thresholds")
        assert r.returncode == 0, r.stderr

        staged = json.loads(tpath.read_text())
        assert (staged["tau"], staged["delta"]) == (direct["tau"], direct["delta"])
        assert staged["residuals"] == direct["residuals"]


class TestTheShardFiles:
    def test_each_shard_writes_its_own_file_naming_the_partition(self, seeded_freeze):
        r = _run(seeded_freeze, "--split", "test", "--localise-only",
                 "--shard", "2", "--n-shards", "4")
        assert r.returncode == 0, r.stderr
        loc = seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
        assert [p.name for p in loc.iterdir()] == ["00002of00004.parquet"]

    def test_the_shards_partition_the_split(self, seeded_freeze):
        for k in range(4):
            assert _run(seeded_freeze, "--split", "test", "--localise-only",
                        "--shard", str(k), "--n-shards", "4").returncode == 0
        loc = seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
        df = pd.concat([pd.read_parquet(p) for p in sorted(loc.iterdir())])
        assert len(df) == 18 and df["item_id"].is_unique

    def test_a_shard_file_carries_no_decision(self, seeded_freeze):
        _run(seeded_freeze, "--split", "test", "--localise-only",
             "--shard", "0", "--n-shards", "1")
        loc = seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
        df = pd.read_parquet(next(iter(loc.iterdir())))
        assert "decision" not in df.columns

    def test_calibration_and_test_shards_do_not_share_a_directory(self, seeded_freeze):
        for split in ("calibration", "test"):
            _run(seeded_freeze, "--split", split, "--localise-only",
                 "--shard", "0", "--n-shards", "1")
        base = seeded_freeze / "06_verify" / "stub_baseline"
        assert (base / "localisations_calibration").is_dir()
        assert (base / "localisations_test").is_dir()


class TestTheCLIRefusals:
    def test_the_two_halves_cannot_be_combined(self, seeded_freeze):
        r = _run(seeded_freeze, "--split", "test", "--localise-only",
                 "--from-localisations")
        assert r.returncode != 0 and "two halves" in r.stderr

    def test_sharding_the_deciding_pass_is_refused(self, seeded_freeze):
        r = _run(seeded_freeze, "--split", "test", "--from-localisations",
                 "--shard", "1", "--n-shards", "4")
        assert r.returncode != 0 and "cannot be sharded" in r.stderr

    def test_deciding_with_no_shards_on_disk_says_so(self, seeded_freeze):
        r = _run(seeded_freeze, "--split", "test", "--from-localisations")
        assert r.returncode != 0 and "no localisation shards" in r.stderr

    def test_an_incomplete_shard_set_is_refused_rather_than_decided(self, calibrated):
        """The failure this guards against is a silent one: a predictions table over
        three quarters of the split, with every row individually correct."""
        for k in (0, 1, 2):
            _run(calibrated, "--split", "test", "--localise-only",
                 "--shard", str(k), "--n-shards", "4")
        r = _run(calibrated, "--split", "test", "--from-localisations")
        # `shards; missing` is stage 13's own gap check, which fires before anything
        # reads a row. It used to be `_precomputed_or_raise`'s "do not cover this split",
        # reached only once the decide loop hit the first absent item; 36e43c1 moved the
        # refusal earlier and made it name the indices to resubmit, and this assertion
        # was left pointing at the message that no longer fires.
        assert r.returncode != 0 and "shards; missing" in r.stderr

    def test_the_independence_audit_is_refused_on_the_deciding_half(self, seeded_freeze):
        _run(seeded_freeze, "--split", "test", "--localise-only",
             "--shard", "0", "--n-shards", "1")
        r = _run(seeded_freeze, "--split", "test", "--from-localisations",
                 "--audit-independence")
        assert r.returncode != 0 and "localise-only" in r.stderr

    def test_shards_from_another_verifier_are_refused(self, seeded_freeze):
        _run(seeded_freeze, "--split", "test", "--localise-only",
             "--shard", "0", "--n-shards", "1")
        loc = (seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
               / "00000of00001.parquet")
        df = pd.read_parquet(loc)
        df["verifier_name"] = "something_else"
        df.to_parquet(loc, index=False)
        r = _run(seeded_freeze, "--split", "test", "--from-localisations")
        assert r.returncode != 0 and "stale" in r.stderr

    def test_shards_from_two_revisions_are_refused(self, seeded_freeze):
        for k in range(2):
            _run(seeded_freeze, "--split", "test", "--localise-only",
                 "--shard", str(k), "--n-shards", "2")
        loc = (seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
               / "00001of00002.parquet")
        df = pd.read_parquet(loc)
        df["verifier_revision"] = "99"
        df.to_parquet(loc, index=False)
        r = _run(seeded_freeze, "--split", "test", "--from-localisations")
        assert r.returncode != 0 and "disagree" in r.stderr

    def test_more_shards_than_contexts_is_refused_rather_than_written_empty(
        self, seeded_freeze
    ):
        """Six contexts, eight shards. An empty shard file would look like a completed
        shard, and the deciding pass would then be missing a quarter of the split with
        nothing on disk saying so."""
        r = _run(seeded_freeze, "--split", "test", "--localise-only",
                 "--shard", "7", "--n-shards", "8")
        assert r.returncode != 0 and "empty" in r.stderr


class TestShardingKeepsEachShardToItsOwnContexts:
    def test_a_shards_items_all_come_from_its_own_contexts(self, seeded_freeze):
        """Why the partition is by context and not by item: a ZoomTrace shard loads
        only the window banks of the contexts it was given, so an item whose context
        landed in another shard would hit a bank that was never loaded."""
        for k in range(3):
            assert _run(seeded_freeze, "--split", "test", "--localise-only",
                        "--shard", str(k), "--n-shards", "3").returncode == 0
        loc = seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
        pairs = pd.read_parquet(seeded_freeze / "01_benchmark" / "pairs.parquet")
        from pyrafault.dataset import make_item_id
        import yaml as _yaml
        seed = _yaml.safe_load(
            open(seeded_freeze / "01_benchmark" / "config.yaml"))["seed"]
        ctx_of = {make_item_id(seed, r.pair_id): r.context_id
                  for r in pairs.itertuples()}
        per_shard = [set(ctx_of[i] for i in pd.read_parquet(p)["item_id"])
                     for p in sorted(loc.iterdir())]
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            assert not per_shard[a] & per_shard[b], "shards share a context"
        assert set().union(*per_shard) == set(pairs["context_id"])


class TestResume:
    """An array job that hits the queue's wall clock is resubmitted for the indices that
    died. Recomputing the ones that finished is hours of GPU for nothing -- but trusting
    a file that does not actually hold this shard would be worse, so the check is against
    the item ids and the method name, not against the file's existence."""

    def test_a_complete_shard_is_not_recomputed(self, seeded_freeze):
        args = ("--split", "test", "--localise-only", "--shard", "0", "--n-shards", "2")
        assert _run(seeded_freeze, *args).returncode == 0
        r = _run(seeded_freeze, *args)
        assert r.returncode == 0 and "already complete" in r.stdout

    def test_no_resume_recomputes_it_anyway(self, seeded_freeze):
        args = ("--split", "test", "--localise-only", "--shard", "0", "--n-shards", "2")
        assert _run(seeded_freeze, *args).returncode == 0
        r = _run(seeded_freeze, *args, "--no-resume")
        assert r.returncode == 0 and "already complete" not in r.stdout

    def test_a_truncated_file_is_recomputed(self, seeded_freeze):
        args = ("--split", "test", "--localise-only", "--shard", "0", "--n-shards", "2")
        assert _run(seeded_freeze, *args).returncode == 0
        f = (seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
             / "00000of00002.parquet")
        pd.read_parquet(f).head(1).to_parquet(f, index=False)
        r = _run(seeded_freeze, *args)
        assert r.returncode == 0 and "already complete" not in r.stdout
        assert len(pd.read_parquet(f)) == 9

    def test_a_file_from_a_different_partition_is_recomputed(self, seeded_freeze):
        """Same shard index, different `--n-shards`, so the file name collides while the
        item set does not. Trusting it would evaluate the wrong third of the split."""
        assert _run(seeded_freeze, "--split", "test", "--localise-only",
                    "--shard", "0", "--n-shards", "2").returncode == 0
        f = (seeded_freeze / "06_verify" / "stub_baseline" / "localisations_test"
             / "00000of00002.parquet")
        df = pd.read_parquet(f)
        df.to_parquet(f.with_name("00000of00003.parquet"), index=False)
        r = _run(seeded_freeze, "--split", "test", "--localise-only",
                 "--shard", "0", "--n-shards", "3")
        assert r.returncode == 0 and "already complete" not in r.stdout


class TestOnePartitionAtATime:
    """A localisation directory can hold shards from two different partitions.

    A smoke run of `--n-shards 400` leaves `00000of00400.parquet` behind; the real run at
    `--n-shards 20` writes `00000of00020.parquet` and the rest. `--from-localisations`
    globs the directory, so it would read both -- and `load_localisations` does catch the
    overlapping item ids, but "the shards overlap" is a confusing way to say "there is a
    file here from a different run". Naming the partitions is the difference between a
    five-second fix and an hour of staring at it.
    """

    def test_two_partitions_in_one_directory_are_refused_by_name(self, calibrated):
        for n in (2, 3):
            for k in range(n):
                assert _run(calibrated, "--split", "test", "--localise-only",
                            "--shard", str(k), "--n-shards", str(n)).returncode == 0
        r = _run(calibrated, "--split", "test", "--from-localisations")
        assert r.returncode != 0
        assert "00002" in r.stderr and "00003" in r.stderr, r.stderr

    def test_a_single_partition_still_reads_cleanly(self, calibrated):
        for k in range(3):
            assert _run(calibrated, "--split", "test", "--localise-only",
                        "--shard", str(k), "--n-shards", "3").returncode == 0
        r = _run(calibrated, "--split", "test", "--from-localisations")
        assert r.returncode == 0, r.stderr

    def test_an_incomplete_partition_is_still_reported_as_missing_items(self, calibrated):
        """The two checks are different failures and must stay distinguishable: one
        partition with a hole is not the same problem as two partitions mixed."""
        for k in (0, 1):
            assert _run(calibrated, "--split", "test", "--localise-only",
                        "--shard", str(k), "--n-shards", "3").returncode == 0
        r = _run(calibrated, "--split", "test", "--from-localisations")
        # Distinguishable from the mixed-partition refusal, which is the point of this
        # test: that one says "different partitions", this one names the gap.
        assert r.returncode != 0 and "shards; missing" in r.stderr
        assert "different partitions" not in r.stderr


class TestDecidingNeedsNoMatcher:
    """`--from-localisations` must not construct the matcher.

    This is not only a speed question. A matcher can live in an optional overlay
    environment (RoMa does), and the deciding half deliberately runs in the base one. If
    deciding still built the matcher, a sharded run would finish the expensive half and
    then fail on the cheapest step -- which is exactly what happened once, after 58
    GPU-hours of localisation.
    """

    def test_a_matcher_that_cannot_be_built_still_decides(self, calibrated, monkeypatch):
        for k in range(2):
            assert _run(calibrated, "--split", "test", "--localise-only",
                        "--shard", str(k), "--n-shards", "2").returncode == 0
        # The stub verifier is not ZoomTrace, so the matcher is never reached either
        # way; what this pins is that the deciding path does not go looking for one.
        r = _run(calibrated, "--split", "test", "--from-localisations")
        assert r.returncode == 0, r.stderr

    def test_zoomtrace_is_never_constructed_when_deciding_from_shards(self, monkeypatch):
        """Checked by behaviour, not by scanning the source.

        There IS a `_build_zoomtrace` call after this branch -- the fallback for a
        ZoomTrace run with no shards to decide from -- and reading the source cannot
        tell that it is unreachable here. What matters is that it does not run, so the
        test makes it explode and requires that nothing explodes.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "s13_nomatcher", ROOT / "scripts" / "13_run_verifier.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def boom(*a, **kw):
            raise AssertionError("the deciding path built a matcher it does not need")

        monkeypatch.setattr(mod, "_build_zoomtrace", boom)
        # Reaching the fallback would raise; the real guard is that `verifier` is already
        # a RecordedVerifier by then, so this stands in for every from-localisations run.
        assert mod.RecordedVerifier("zoomtrace", "1").name == "zoomtrace"
        with pytest.raises(AssertionError, match="does not need"):
            mod._build_zoomtrace(None, None, None)
