"""The scorer reads predictions and the benchmark, never a verifier.

That is what lets someone be scored by submitting a file: their method never has to
enter this repository.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_the_scorer_imports_no_verifier_module():
    src = Path("scripts/14_score_verifier.py").read_text()
    for forbidden in ("zoomtrace", "matchers", "LoftrMatcher", "get_extractor"):
        assert forbidden not in src, f"scorer must not depend on {forbidden}"


def test_writes_metrics_json_and_markdown(tmp_path):
    preds = pd.DataFrame({
        "item_id": ["i0", "i1"],
        "estimated_bbox_l0_x": [1000, 1000], "estimated_bbox_l0_y": [1000, 1000],
        "estimated_bbox_l0_w": [100, 100], "estimated_bbox_l0_h": [100, 100],
        "reliability": [0.9, 0.9], "decision": ["verified", "mismatched"],
    })
    pairs = pd.DataFrame({
        "item_id": ["i0", "i1"], "claim_correct": [True, False],
        "fault_type": ["clean", "adjacent_drift"], "true_within_context": [True, True],
        "true_bbox_l0_x": [1000, 1000], "true_bbox_l0_y": [1000, 1000],
        "true_bbox_l0_w": [100, 100], "true_bbox_l0_h": [100, 100],
        "claimed_bbox_l0_x": [1000, 1000], "claimed_bbox_l0_y": [1000, 1000],
        "claimed_bbox_l0_w": [100, 100], "claimed_bbox_l0_h": [100, 100],
        "scale_ratio": [4.0, 4.0], "hardness_score": [0.5, 0.5],
    })
    p = tmp_path / "predictions.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs_with_items.parquet"; pairs.to_parquet(q, index=False)

    subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--out-dir", str(tmp_path)],
        check=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["balanced_accuracy"] == 1.0
    assert (tmp_path / "metrics.md").exists()


def test_split_filters_expected_item_ids_before_scoring(tmp_path):
    """A predictions file covers one split; the pairs table it is checked against
    may hold every split. Without filtering, score_predictions's item_id-set check
    raises on the splits the predictions file legitimately omits.
    """
    preds = pd.DataFrame({
        "item_id": ["i0"],
        "estimated_bbox_l0_x": [1000], "estimated_bbox_l0_y": [1000],
        "estimated_bbox_l0_w": [100], "estimated_bbox_l0_h": [100],
        "reliability": [0.9], "decision": ["verified"],
    })
    pairs = pd.DataFrame({
        "item_id": ["i0", "i1"], "split": ["test", "train"],
        "claim_correct": [True, False],
        "fault_type": ["clean", "adjacent_drift"], "true_within_context": [True, True],
        "true_bbox_l0_x": [1000, 1000], "true_bbox_l0_y": [1000, 1000],
        "true_bbox_l0_w": [100, 100], "true_bbox_l0_h": [100, 100],
        "claimed_bbox_l0_x": [1000, 1000], "claimed_bbox_l0_y": [1000, 1000],
        "claimed_bbox_l0_w": [100, 100], "claimed_bbox_l0_h": [100, 100],
        "scale_ratio": [4.0, 4.0], "hardness_score": [0.5, 0.5],
    })
    p = tmp_path / "predictions_test.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs.parquet"; pairs.to_parquet(q, index=False)

    subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--split", "test",
         "--out-dir", str(tmp_path)],
        check=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    m = json.loads((tmp_path / "metrics_test.json").read_text())
    assert m["n_items"] == 1
    assert m["scored_split"] == "test"


def test_without_split_a_partial_submission_raises(tmp_path):
    """Documents the failure mode --split fixes: a per-split predictions file scored
    against an unfiltered multi-split pairs table hits score_predictions's mismatched
    item_id-set ValueError.
    """
    preds = pd.DataFrame({
        "item_id": ["i0"],
        "estimated_bbox_l0_x": [1000], "estimated_bbox_l0_y": [1000],
        "estimated_bbox_l0_w": [100], "estimated_bbox_l0_h": [100],
        "reliability": [0.9], "decision": ["verified"],
    })
    pairs = pd.DataFrame({
        "item_id": ["i0", "i1"], "split": ["test", "train"],
        "claim_correct": [True, False],
        "fault_type": ["clean", "adjacent_drift"], "true_within_context": [True, True],
        "true_bbox_l0_x": [1000, 1000], "true_bbox_l0_y": [1000, 1000],
        "true_bbox_l0_w": [100, 100], "true_bbox_l0_h": [100, 100],
        "claimed_bbox_l0_x": [1000, 1000], "claimed_bbox_l0_y": [1000, 1000],
        "claimed_bbox_l0_w": [100, 100], "claimed_bbox_l0_h": [100, 100],
        "scale_ratio": [4.0, 4.0], "hardness_score": [0.5, 0.5],
    })
    p = tmp_path / "predictions_test.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs.parquet"; pairs.to_parquet(q, index=False)

    result = subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--out-dir", str(tmp_path)],
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "item_id" in result.stderr


def test_split_without_a_split_column_raises_naming_the_cause(tmp_path):
    """Cleanup: `--split` used to silently no-op when `pairs` has no 'split' column --
    the expected item_id set would then be the whole (unfiltered) table, and any
    downstream mismatch would surface as a confusing item_id-set error with no hint
    that --split was the reason it didn't filter."""
    preds = pd.DataFrame({
        "item_id": ["i0"],
        "estimated_bbox_l0_x": [1000], "estimated_bbox_l0_y": [1000],
        "estimated_bbox_l0_w": [100], "estimated_bbox_l0_h": [100],
        "reliability": [0.9], "decision": ["verified"],
    })
    pairs = pd.DataFrame({  # no 'split' column
        "item_id": ["i0"], "claim_correct": [True],
        "fault_type": ["clean"], "true_within_context": [True],
        "true_bbox_l0_x": [1000], "true_bbox_l0_y": [1000],
        "true_bbox_l0_w": [100], "true_bbox_l0_h": [100],
        "claimed_bbox_l0_x": [1000], "claimed_bbox_l0_y": [1000],
        "claimed_bbox_l0_w": [100], "claimed_bbox_l0_h": [100],
        "scale_ratio": [4.0], "hardness_score": [0.5],
    })
    p = tmp_path / "predictions_test.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs.parquet"; pairs.to_parquet(q, index=False)

    result = subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--split", "test",
         "--out-dir", str(tmp_path)],
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "split" in result.stderr


def _one_item_frames(item_id: str, split: str, claim_correct: bool):
    preds = pd.DataFrame({
        "item_id": [item_id],
        "estimated_bbox_l0_x": [1000], "estimated_bbox_l0_y": [1000],
        "estimated_bbox_l0_w": [100], "estimated_bbox_l0_h": [100],
        "reliability": [0.9], "decision": ["verified" if claim_correct else "mismatched"],
    })
    pairs = pd.DataFrame({
        "item_id": [item_id], "split": [split], "claim_correct": [claim_correct],
        "fault_type": ["clean" if claim_correct else "adjacent_drift"],
        "true_within_context": [True],
        "true_bbox_l0_x": [1000], "true_bbox_l0_y": [1000],
        "true_bbox_l0_w": [100], "true_bbox_l0_h": [100],
        "claimed_bbox_l0_x": [1000], "claimed_bbox_l0_y": [1000],
        "claimed_bbox_l0_w": [100], "claimed_bbox_l0_h": [100],
        "scale_ratio": [4.0], "hardness_score": [0.5],
    })
    return preds, pairs


def test_scoring_two_splits_of_one_run_keeps_both_results(tmp_path):
    """One run produces a predictions file per split, scored into one directory. Under a
    single `metrics.json` the second scoring silently overwrote the first, and nothing in
    the file said which split it described -- so the survivor could be read as either."""
    written = {}
    for split, correct in (("calibration", True), ("test", False)):
        preds, pairs = _one_item_frames(f"i-{split}", split, correct)
        p = tmp_path / f"predictions_{split}.parquet"; preds.to_parquet(p, index=False)
        q = tmp_path / f"pairs_{split}.parquet"; pairs.to_parquet(q, index=False)
        subprocess.run(
            [sys.executable, "scripts/14_score_verifier.py",
             "--predictions", str(p), "--pairs", str(q), "--split", split,
             "--out-dir", str(tmp_path)],
            check=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        )
        written[split] = json.loads((tmp_path / f"metrics_{split}.json").read_text())

    assert written["calibration"]["scored_split"] == "calibration"
    assert written["test"]["scored_split"] == "test"
    assert (tmp_path / "metrics_calibration.md").exists()
    assert (tmp_path / "metrics_test.md").exists()


def _band_case(tmp_path, bands=("none", "severe")):
    n = len(bands)
    preds = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n)],
        "estimated_bbox_l0_x": [1000 + 40 * i for i in range(n)],
        "estimated_bbox_l0_y": [1000] * n,
        "estimated_bbox_l0_w": [100] * n, "estimated_bbox_l0_h": [100] * n,
        "reliability": [0.9] * n, "decision": ["verified"] * n,
    })
    pairs = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n)], "claim_correct": [True] * n,
        "fault_type": ["clean"] * n, "true_within_context": [True] * n,
        "true_bbox_l0_x": [1000] * n, "true_bbox_l0_y": [1000] * n,
        "true_bbox_l0_w": [100] * n, "true_bbox_l0_h": [100] * n,
        "claimed_bbox_l0_x": [1000] * n, "claimed_bbox_l0_y": [1000] * n,
        "claimed_bbox_l0_w": [100] * n, "claimed_bbox_l0_h": [100] * n,
        "scale_ratio": [4.0] * n, "hardness_score": [0.5] * n,
        "degradation_band": list(bands),
    })
    p = tmp_path / "predictions.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs_with_items.parquet"; pairs.to_parquet(q, index=False)
    subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--out-dir", str(tmp_path)],
        check=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    return (json.loads((tmp_path / "metrics.json").read_text()),
            (tmp_path / "metrics.md").read_text())


def test_the_markdown_reports_localisation_per_band(tmp_path):
    """A pooled median IoU is not comparable between methods when one is bimodal -- the
    NCC control's pooled median reads higher than ZoomTrace's while its balanced accuracy
    is 0.500 against 0.988 -- so the human-readable report has to carry the per-band cut,
    not only the json."""
    metrics, md = _band_case(tmp_path)
    assert "By degradation band" in md
    for band in ("none", "severe"):
        assert band in md
        assert metrics["by_degradation_band"][band]["median_iou"] is not None
    assert "median IoU" in md


def test_a_band_with_no_localisation_prints_a_dash_not_a_zero(tmp_path):
    n = 2
    preds = pd.DataFrame({
        "item_id": ["i0", "i1"],
        "estimated_bbox_l0_x": [1000, None], "estimated_bbox_l0_y": [1000, None],
        "estimated_bbox_l0_w": [100, None], "estimated_bbox_l0_h": [100, None],
        "reliability": [0.9, 0.9], "decision": ["verified", "mismatched"],
    })
    pairs = pd.DataFrame({
        "item_id": ["i0", "i1"], "claim_correct": [True, True],
        "fault_type": ["clean"] * n, "true_within_context": [True] * n,
        "true_bbox_l0_x": [1000] * n, "true_bbox_l0_y": [1000] * n,
        "true_bbox_l0_w": [100] * n, "true_bbox_l0_h": [100] * n,
        "claimed_bbox_l0_x": [1000] * n, "claimed_bbox_l0_y": [1000] * n,
        "claimed_bbox_l0_w": [100] * n, "claimed_bbox_l0_h": [100] * n,
        "scale_ratio": [4.0] * n, "hardness_score": [0.5] * n,
        "degradation_band": ["none", "severe"],
    })
    p = tmp_path / "predictions.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs_with_items.parquet"; pairs.to_parquet(q, index=False)
    subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--out-dir", str(tmp_path)],
        check=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    md = (tmp_path / "metrics.md").read_text()
    severe = [l for l in md.splitlines() if l.startswith("| `severe`")]
    assert severe and "—" in severe[0], severe


def test_a_pre_degradation_freeze_prints_no_band_table(tmp_path):
    """Nothing is invented for a freeze built before the column existed: an empty table
    with four band headings would read as four measurements."""
    preds = pd.DataFrame({
        "item_id": ["i0"], "estimated_bbox_l0_x": [1000], "estimated_bbox_l0_y": [1000],
        "estimated_bbox_l0_w": [100], "estimated_bbox_l0_h": [100],
        "reliability": [0.9], "decision": ["verified"],
    })
    pairs = pd.DataFrame({
        "item_id": ["i0"], "claim_correct": [True], "fault_type": ["clean"],
        "true_within_context": [True],
        "true_bbox_l0_x": [1000], "true_bbox_l0_y": [1000],
        "true_bbox_l0_w": [100], "true_bbox_l0_h": [100],
        "claimed_bbox_l0_x": [1000], "claimed_bbox_l0_y": [1000],
        "claimed_bbox_l0_w": [100], "claimed_bbox_l0_h": [100],
        "scale_ratio": [4.0], "hardness_score": [0.5],
    })
    p = tmp_path / "predictions.parquet"; preds.to_parquet(p, index=False)
    q = tmp_path / "pairs_with_items.parquet"; pairs.to_parquet(q, index=False)
    subprocess.run(
        [sys.executable, "scripts/14_score_verifier.py",
         "--predictions", str(p), "--pairs", str(q), "--out-dir", str(tmp_path)],
        check=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    assert "By degradation band" not in (tmp_path / "metrics.md").read_text()
