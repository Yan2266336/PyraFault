#!/usr/bin/env python
"""Offline preflight for the eight handover baselines.

The default mode only reads paths supplied through the baseline environment
variables.  It checks pinned Git revisions, tracked-file cleanliness, required
source files, checkpoint SHA-256 digests, and the pinned DINOv2 snapshot layout.
It never imports an upstream model, reads WSI data, or accesses the network.

Examples::

    python scripts/check_baseline_assets.py
    python scripts/check_baseline_assets.py --method lightglue --method minima-lightglue
    python scripts/check_baseline_assets.py --load lightglue

``--load`` is deliberately singular: it constructs just that plugin, then checks
that every expected torch module is in eval mode with frozen parameters.  This
avoids retaining all models in accelerator memory at once.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


# Make the handover command work exactly as printed when invoked from the checkout;
# it must not depend on an editable install or a caller-supplied PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ROMA_CHECKPOINT_SHA256 = (
    "c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba"
)
ROMA_DINOV2_SHA256 = (
    "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
)


@dataclass(frozen=True)
class Result:
    method: str
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SourceAsset:
    variable: str
    revision: str
    required: tuple[str, ...]


@dataclass(frozen=True)
class FileAsset:
    variable: str
    sha256: str


@dataclass(frozen=True)
class MethodAssets:
    sources: tuple[SourceAsset, ...] = ()
    files: tuple[FileAsset, ...] = ()
    dino_snapshot: bool = False


class _Hashes:
    """Avoid reading a shared multi-hundred-MB checkpoint twice in one run."""

    def __init__(self) -> None:
        self._cache: dict[Path, str] = {}

    def __call__(self, path: Path) -> str:
        path = path.resolve()
        if path not in self._cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self._cache[path] = digest.hexdigest()
        return self._cache[path]


def _method_assets() -> dict[str, MethodAssets]:
    # Import constants from the adapters so the preflight cannot silently drift
    # from what model construction enforces.  These modules do not load a model.
    from pyrafault.verify.efficient_loftr import (
        OFFICIAL_CHECKPOINT_SHA256 as ELOFTR_SHA,
        OFFICIAL_SOURCE_REVISION as ELOFTR_REV,
    )
    from pyrafault.verify.lightglue import (
        LIGHTGLUE_SHA256,
        OFFICIAL_SOURCE_REVISION as LIGHTGLUE_REV,
        SUPERPOINT_SHA256,
    )
    from pyrafault.verify.matchanything_eloftr import (
        OFFICIAL_CHECKPOINT_SHA256 as MATCHANYTHING_SHA,
        OFFICIAL_SOURCE_REVISION as MATCHANYTHING_REV,
    )
    from pyrafault.verify.minima_lightglue import MINIMA_LIGHTGLUE_SHA256
    from pyrafault.verify.roma import OFFICIAL_SOURCE_REVISION as ROMA_REV

    lightglue_source = SourceAsset(
        "LIGHTGLUE_SOURCE", LIGHTGLUE_REV,
        ("lightglue/superpoint.py", "lightglue/lightglue.py"),
    )
    superpoint = FileAsset("SUPERPOINT_CHECKPOINT", SUPERPOINT_SHA256)
    return {
        "ncc": MethodAssets(),
        "efficient-loftr": MethodAssets(
            sources=(SourceAsset("EFFICIENT_LOFTR_SOURCE", ELOFTR_REV,
                                 ("src/loftr/__init__.py",)),),
            files=(FileAsset("EFFICIENT_LOFTR_CHECKPOINT", ELOFTR_SHA),),
        ),
        "matchanything-eloftr": MethodAssets(
            sources=(SourceAsset(
                "MATCHANYTHING_ELOFTR_SOURCE", MATCHANYTHING_REV,
                ("src/__init__.py", "src/config/default.py",
                 "src/loftr/__init__.py", "configs/models/eloftr_model.py"),
            ),),
            files=(FileAsset("MATCHANYTHING_ELOFTR_CHECKPOINT", MATCHANYTHING_SHA),),
        ),
        "roma": MethodAssets(
            sources=(SourceAsset("ROMA_SOURCE", ROMA_REV,
                                 ("romatch/models/model_zoo/__init__.py",)),),
            files=(FileAsset("ROMA_CHECKPOINT", ROMA_CHECKPOINT_SHA256),
                   FileAsset("ROMA_DINOV2_WEIGHTS", ROMA_DINOV2_SHA256)),
        ),
        "lightglue": MethodAssets(
            sources=(lightglue_source,),
            files=(superpoint, FileAsset("LIGHTGLUE_CHECKPOINT", LIGHTGLUE_SHA256)),
        ),
        "minima-lightglue": MethodAssets(
            sources=(lightglue_source,),
            files=(superpoint,
                   FileAsset("MINIMA_LIGHTGLUE_CHECKPOINT", MINIMA_LIGHTGLUE_SHA256)),
        ),
        "sue": MethodAssets(dino_snapshot=True),
        "statistical-uncertainty": MethodAssets(dino_snapshot=True),
    }


METHODS = tuple(_method_assets())


def _path_from_env(method: str, variable: str, environ: Mapping[str, str]) -> tuple[Path | None, list[Result]]:
    value = environ.get(variable)
    if not value:
        return None, [Result(method, variable, False, f"environment variable {variable} is not set")]
    return Path(value).expanduser().resolve(), []


def _git_state(path: Path) -> tuple[str, str]:
    """Return HEAD and tracked-file status, matching the adapters' invariant."""
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("not a readable Git checkout") from exc
    return head, status


def _check_source(method: str, asset: SourceAsset, environ: Mapping[str, str]) -> list[Result]:
    path, results = _path_from_env(method, asset.variable, environ)
    if path is None:
        return results
    if not path.is_dir():
        return [Result(method, asset.variable, False, f"directory does not exist: {path}")]
    try:
        actual, status = _git_state(path)
    except ValueError as exc:
        return [Result(method, asset.variable, False, f"{path}: {exc}")]
    results.append(Result(
        method, f"{asset.variable}:git-revision", actual == asset.revision,
        f"expected={asset.revision} actual={actual}",
    ))
    results.append(Result(
        method, f"{asset.variable}:tracked-tree", not status,
        "tracked tree clean" if not status else f"tracked changes={status!r}",
    ))
    for relative in asset.required:
        candidate = path / relative
        results.append(Result(
            method, f"{asset.variable}:{relative}", candidate.is_file(),
            f"required file {'present' if candidate.is_file() else 'missing'}: {candidate}",
        ))
    return results


def _check_file(method: str, asset: FileAsset, environ: Mapping[str, str],
                hashes: _Hashes) -> list[Result]:
    path, results = _path_from_env(method, asset.variable, environ)
    if path is None:
        return results
    if not path.is_file():
        return [Result(method, asset.variable, False, f"file does not exist: {path}")]
    actual = hashes(path)
    return [Result(
        method, f"{asset.variable}:sha256", actual == asset.sha256,
        f"path={path} expected={asset.sha256} actual={actual}",
    )]


def _dino_weight_files(snapshot: Path) -> tuple[list[Path], str | None]:
    """Resolve a direct or sharded PyTorch/safetensors Transformers checkpoint."""
    direct = [snapshot / name for name in ("model.safetensors", "pytorch_model.bin")
              if (snapshot / name).is_file()]
    if direct:
        return direct, None
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = snapshot / name
        if not index.is_file():
            continue
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            shards = sorted({snapshot / value for value in data["weight_map"].values()})
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return [], f"invalid checkpoint index {index}: {exc}"
        missing = [str(path) for path in shards if not path.is_file()]
        if missing:
            return [], f"checkpoint index references missing shard(s): {missing}"
        return [index, *shards], None
    return [], "missing model.safetensors/pytorch_model.bin (or a valid sharded index)"


def _check_dino(method: str, environ: Mapping[str, str], hashes: _Hashes) -> list[Result]:
    from pyrafault.verify.plugins import DINO_REVISION

    variable = "PYRAFAULT_DINOV2_SNAPSHOT"
    snapshot, results = _path_from_env(method, variable, environ)
    if snapshot is None:
        return results
    if not snapshot.is_dir():
        return [Result(method, variable, False, f"directory does not exist: {snapshot}")]
    results.append(Result(
        method, f"{variable}:revision-directory", snapshot.name == DINO_REVISION,
        f"expected directory name={DINO_REVISION} actual={snapshot.name}",
    ))
    required = [snapshot / "config.json", snapshot / "preprocessor_config.json"]
    for path in required:
        results.append(Result(
            method, f"{variable}:{path.name}", path.is_file(),
            f"required file {'present' if path.is_file() else 'missing'}: {path}",
        ))
    if required[0].is_file():
        try:
            config = json.loads(required[0].read_text(encoding="utf-8"))
            valid = config.get("model_type") == "dinov2" and config.get("hidden_size") == 768
            detail = (f"model_type={config.get('model_type')!r} "
                      f"hidden_size={config.get('hidden_size')!r}")
        except (OSError, json.JSONDecodeError) as exc:
            valid, detail = False, f"invalid config.json: {exc}"
        results.append(Result(method, f"{variable}:model-config", valid, detail))
    model_files, error = _dino_weight_files(snapshot)
    results.append(Result(
        method, f"{variable}:model-files", error is None,
        error or f"found {len(model_files)} checkpoint/index file(s)",
    ))
    # There is intentionally no invented expected DINO file hash.  Report the
    # actual digest of every required config and resolved model file for provenance.
    for path in [*required, *model_files]:
        if path.is_file():
            results.append(Result(
                method, f"{variable}:sha256:{path.relative_to(snapshot)}", True,
                f"path={path} actual={hashes(path)}",
            ))
    return results


def check_methods(methods: Sequence[str], environ: Mapping[str, str] | None = None) -> list[Result]:
    environ = os.environ if environ is None else environ
    specs, hashes = _method_assets(), _Hashes()
    results: list[Result] = []
    for method in methods:
        if method == "ncc":
            results.append(Result(method, "assets", True, "no external source or weight assets"))
        spec = specs[method]
        for source in spec.sources:
            results.extend(_check_source(method, source, environ))
        for file in spec.files:
            results.extend(_check_file(method, file, environ, hashes))
        if spec.dino_snapshot:
            results.extend(_check_dino(method, environ, hashes))
    return results


def _plugin_for(method: str):
    if method == "ncc":
        from pyrafault.verify.ncc_baseline import NccBaseline
        return NccBaseline()
    from pyrafault.verify.plugins import (
        EfficientLoFTRPlugin, LightGluePlugin, MatchAnythingELoFTRPlugin,
        MinimaLightGluePlugin, RoMaPlugin, StatisticalUncertaintyPlugin, SUEPlugin,
    )
    classes = {
        "efficient-loftr": EfficientLoFTRPlugin,
        "matchanything-eloftr": MatchAnythingELoFTRPlugin,
        "roma": RoMaPlugin,
        "lightglue": LightGluePlugin,
        "minima-lightglue": MinimaLightGluePlugin,
        "sue": SUEPlugin,
        "statistical-uncertainty": StatisticalUncertaintyPlugin,
    }
    return classes[method]()


def _expected_torch_modules(method: str, backend) -> list[tuple[str, object]]:
    if method in {"efficient-loftr", "matchanything-eloftr"}:
        return [("matcher.model", backend._matcher._model)]
    if method == "roma":
        matcher = backend._matcher
        return [("matcher.model", matcher._model),
                ("matcher.dinov2", matcher._model.encoder.dinov2_vitl14[0])]
    if method in {"lightglue", "minima-lightglue"}:
        matcher = backend._matcher
        return [("matcher.superpoint", matcher._extractor),
                ("matcher.lightglue", matcher._model)]
    if method in {"sue", "statistical-uncertainty"}:
        return [("extractor.model", backend._extractor.model)]
    return []


def _check_frozen_modules(method: str, modules: Sequence[tuple[str, object]]) -> list[Result]:
    results: list[Result] = []
    for label, module in modules:
        training = bool(getattr(module, "training", True))
        parameters = list(module.parameters())
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        results.append(Result(method, f"load:{label}:eval", not training,
                              f"training={training}"))
        results.append(Result(method, f"load:{label}:frozen", trainable == 0,
                              f"trainable_parameters={trainable}"))
    return results


def check_loaded(method: str) -> list[Result]:
    """Construct one selected plugin; no forward pass or image/WSI access."""
    if method == "ncc":
        plugin = _plugin_for(method)
        return [Result(method, "load", plugin.name == "ncc_baseline",
                       f"constructed {plugin.__class__.__module__}:{plugin.__class__.__name__}")]
    try:
        import torch
        torch.set_float32_matmul_precision("highest")
        plugin = _plugin_for(method)
        revision = plugin.revision  # lazy construction happens here, once
        backend = plugin._backend_instance
        modules = _expected_torch_modules(method, backend)
        results = [Result(method, "load:revision", True, f"actual={revision}")]
        if not modules:
            results.append(Result(method, "load:modules", False,
                                  "no expected torch modules were found"))
        else:
            results.extend(_check_frozen_modules(method, modules))
        del modules, backend, plugin
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    except Exception as exc:  # present one clear FAIL instead of a handover traceback
        return [Result(method, "load", False, f"{type(exc).__name__}: {exc}")]


def _print_results(results: Sequence[Result]) -> None:
    for result in results:
        state = "PASS" if result.passed else "FAIL"
        print(f"[{state}] {result.method}/{result.check}: {result.detail}")
    failed = sum(not result.passed for result in results)
    state = "PASS" if not failed else "FAIL"
    print(f"[{state}] summary: {len(results) - failed} passed, {failed} failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", choices=METHODS,
                        help="check only this method; repeat for more than one")
    parser.add_argument("--load", choices=METHODS, metavar="METHOD",
                        help="also construct exactly this one plugin and verify eval/frozen state")
    args = parser.parse_args(argv)
    methods = list(dict.fromkeys(args.method or ([args.load] if args.load else METHODS)))
    if args.load and args.load not in methods:
        parser.error("--load METHOD must also be among the selected --method values")
    results = check_methods(methods)
    if args.load:
        if any(not result.passed for result in results if result.method == args.load):
            results.append(Result(args.load, "load", False,
                                  "skipped because the selected method's asset checks failed"))
        else:
            results.extend(check_loaded(args.load))
    _print_results(results)
    return int(any(not result.passed for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
