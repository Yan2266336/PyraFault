from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import check_baseline_assets as assets


def _init_repo(path: Path, required: str) -> str:
    target = path / required
    target.parent.mkdir(parents=True)
    target.write_text("pinned\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
                   check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "PyraFault Test"],
                   check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def test_inventory_is_exactly_the_eight_handover_methods():
    assert assets.METHODS == (
        "ncc", "efficient-loftr", "matchanything-eloftr", "roma", "lightglue",
        "minima-lightglue", "sue", "statistical-uncertainty",
    )


def test_ncc_needs_no_external_assets():
    results = assets.check_methods(["ncc"], environ={})
    assert results == [assets.Result("ncc", "assets", True,
                                      "no external source or weight assets")]


def test_missing_environment_variable_is_a_clear_failure():
    results = assets.check_methods(["efficient-loftr"], environ={})
    assert {result.check for result in results if not result.passed} == {
        "EFFICIENT_LOFTR_SOURCE", "EFFICIENT_LOFTR_CHECKPOINT",
    }
    assert all("not set" in result.detail for result in results)


def test_file_check_reports_expected_and_actual_sha(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"native checkpoint")
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    hashes = assets._Hashes()

    good = assets._check_file(
        "x", assets.FileAsset("WEIGHTS", actual), {"WEIGHTS": str(checkpoint)}, hashes)
    bad = assets._check_file(
        "x", assets.FileAsset("WEIGHTS", "0" * 64), {"WEIGHTS": str(checkpoint)}, hashes)

    assert good[0].passed
    assert not bad[0].passed
    assert f"actual={actual}" in bad[0].detail
    assert f"expected={'0' * 64}" in bad[0].detail


def test_source_check_enforces_commit_required_file_and_clean_tracked_tree(tmp_path):
    repo = tmp_path / "upstream"
    revision = _init_repo(repo, "package/model.py")
    spec = assets.SourceAsset("SOURCE", revision, ("package/model.py",))

    clean = assets._check_source("x", spec, {"SOURCE": str(repo)})
    assert clean and all(result.passed for result in clean)
    assert f"actual={revision}" in clean[0].detail

    (repo / "package/model.py").write_text("modified\n", encoding="utf-8")
    dirty = assets._check_source("x", spec, {"SOURCE": str(repo)})
    tree = next(result for result in dirty if result.check.endswith(":tracked-tree"))
    assert not tree.passed
    assert "tracked changes" in tree.detail


def test_source_check_reports_wrong_revision_and_missing_required_file(tmp_path):
    repo = tmp_path / "upstream"
    _init_repo(repo, "present.py")
    spec = assets.SourceAsset("SOURCE", "f" * 40, ("missing.py",))
    results = assets._check_source("x", spec, {"SOURCE": str(repo)})
    assert not next(r for r in results if r.check.endswith(":git-revision")).passed
    assert not next(r for r in results if r.check.endswith(":missing.py")).passed


def _dino_snapshot(tmp_path: Path) -> Path:
    from pyrafault.verify.plugins import DINO_REVISION
    snapshot = tmp_path / DINO_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "dinov2", "hidden_size": 768}), encoding="utf-8")
    (snapshot / "preprocessor_config.json").write_text(
        json.dumps({"do_resize": True, "size": 224}), encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"offline model")
    return snapshot


def test_dino_snapshot_checks_revision_layout_and_prints_every_actual_hash(tmp_path):
    snapshot = _dino_snapshot(tmp_path)
    results = assets._check_dino(
        "sue", {"PYRAFAULT_DINOV2_SNAPSHOT": str(snapshot)}, assets._Hashes())

    assert all(result.passed for result in results)
    hashed = {result.check.rsplit(":", 1)[-1]: result.detail for result in results
              if ":sha256:" in result.check}
    assert set(hashed) == {"config.json", "preprocessor_config.json", "model.safetensors"}
    assert all("actual=" in detail for detail in hashed.values())


def test_dino_snapshot_rejects_wrong_commit_directory_and_config(tmp_path):
    snapshot = tmp_path / "main"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "bert", "hidden_size": 768}), encoding="utf-8")
    (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "pytorch_model.bin").write_bytes(b"model")
    results = assets._check_dino(
        "sue", {"PYRAFAULT_DINOV2_SNAPSHOT": str(snapshot)}, assets._Hashes())
    assert not next(r for r in results if r.check.endswith(":revision-directory")).passed
    assert not next(r for r in results if r.check.endswith(":model-config")).passed


def test_frozen_module_check_distinguishes_eval_from_trainable_parameters():
    torch = pytest.importorskip("torch")
    frozen = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    trainable = torch.nn.Linear(2, 2).eval()

    good = assets._check_frozen_modules("x", [("model", frozen)])
    bad = assets._check_frozen_modules("x", [("model", trainable)])

    assert all(result.passed for result in good)
    assert next(r for r in bad if r.check.endswith(":eval")).passed
    assert not next(r for r in bad if r.check.endswith(":frozen")).passed


def test_cli_ncc_is_offline_and_succeeds_without_environment():
    result = subprocess.run(
        [sys.executable, "scripts/check_baseline_assets.py", "--method", "ncc"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0
    assert "[PASS] ncc/assets" in result.stdout
    assert "0 failed" in result.stdout


def test_load_must_be_among_explicitly_selected_methods():
    with pytest.raises(SystemExit):
        assets.main(["--method", "ncc", "--load", "lightglue"])
