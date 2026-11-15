"""Load, validate and freeze the pipeline configuration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the configuration is internally inconsistent."""


@dataclass(frozen=True)
class SourceCfg:
    """The ingest adapter that produced `slides.parquet` (spec §6 "Stage 00 is an
    adapter, not the foundation"). `mode` selects which fields apply:

    - `mode: gdc` (default, so an absent key means a config written before `mode`
      existed keeps its digest) needs `gdc_dir` plus the three GDC export patterns.
    - `mode: directory` needs `slides_dir`, `pattern`, `case_id_from`.

    Everything downstream of stage 00 only ever reads three columns of
    `slides.parquet` (`slide_id`, `case_id`, `source_relpath`) and joins the third
    against `root_dir` to open the file — it never looks at `mode` again.
    """
    mode: str = "gdc"
    # mode: gdc
    gdc_dir: str | None = None
    manifest: str | None = None
    sample_sheet: str | None = None
    metadata_json: str | None = None
    verify_md5: bool = True
    # mode: directory
    slides_dir: str | None = None
    pattern: str | None = None
    case_id_from: str | None = None

    @property
    def root_dir(self) -> str:
        """Directory that `source_relpath` in `slides.parquet` is relative to.

        `gdc_dir` under `mode: gdc`, `slides_dir` under `mode: directory` — every
        stage from 01 onward joins this with `source_relpath` to open a slide, and
        that join is the entirety of their dependency on which adapter ran stage 00.
        """
        return self.gdc_dir if self.mode == "gdc" else self.slides_dir


@dataclass(frozen=True)
class SlideCfg:
    require_mpp: bool
    max_mpp_l0: float


@dataclass(frozen=True)
class ThumbnailCfg:
    target_mpp: float


@dataclass(frozen=True)
class TissueCfg:
    method: str
    sat_floor: float
    gray_max: int
    close_radius_um: float
    open_radius_um: float
    min_component_area_um2: float


@dataclass(frozen=True)
class ContextCfg:
    target_mpp: float
    size_px: int
    stride_frac: float
    min_tissue_fraction: float
    num_per_slide: int
    selection: str


@dataclass(frozen=True)
class QueryCfg:
    target_mpp: float
    size_px: int
    stride_frac: float
    boundary_margin_frac: float
    min_tissue_fraction: float
    num_per_context: int


@dataclass(frozen=True)
class QualityCfg:
    blur_var_flag: float
    blur_var_exclude: float
    mean_intensity_flag: tuple[float, float]
    mean_intensity_exclude: tuple[float, float]
    pen_mark_flag_frac: float


@dataclass(frozen=True)
class VisualSamplingCfg:
    enabled: bool
    extractor: str
    fallback_extractor: str
    strategy: str
    n_clusters: int
    self_similarity_k: int


@dataclass(frozen=True)
class FeaturesCfg:
    uni: dict[str, Any]
    dinov2: dict[str, Any]
    batch_size: int
    device: str


@dataclass(frozen=True)
class FaultsCfg:
    clean: bool
    drift_enabled: bool
    drift_normalize_by: str
    severity_bands: dict[str, tuple[float, float]]
    drift_balance: str
    max_theta_retries: int
    random_enabled: bool
    random_max_iou: float
    random_min_center_dist: float
    hard_enabled: bool
    hard_pool: str
    hard_max_iou: float
    hard_min_center_dist: float
    hard_top_k: int
    out_of_context_enabled: bool
    # `out_of_context` (spec table 3 rows 4-5). The claim stays inside `C` -- it is the
    # agent's own zoom request (proposal §4.1: fault injection holds `ℓ` fixed and swaps
    # the returned pixels), so a verifier can never reject one by a containment test.
    # What moves outside `C` is `ℓ*`, the true source of the patch actually served.
    # `selection` picks how that source is chosen, and the two policies are the two
    # distinct sample types the proposal asks for, not a tuning knob:
    #   random -> `out_of_context`      (row 5: parent-level provenance failure)
    #   hard   -> `out_of_context_hard` (row 4: same-slide tissue self-similarity)
    #   both   -> emit one pair of each per query
    out_of_context_selection: str = "both"
    # Minimum gap between `C` and the substituted source, in context widths. 0.0 means
    # "must not overlap `C`" and nothing more; raising it pushes the true source further
    # away, which makes the fault easier. Contexts overlap each other at stride 0.25, so
    # a candidate drawn from a neighbouring context is NOT automatically outside `C` --
    # this is enforced geometrically, never by "belongs to a different context_id".
    out_of_context_min_gap: float = 0.0
    # `out_of_slide`: the patch is served from a DIFFERENT slide entirely, while the
    # claim stays inside `C` as always. This is the cache/concurrency failure the
    # proposal's introduction names -- a tool handing back another slide's pixels --
    # and the source of its same-slide/out-of-slide stratification marker.
    #
    # The substitute is drawn from the same SPLIT by construction. Drawing across
    # splits would put dev pixels inside test items, which is the contamination the
    # case-level split exists to prevent; a fault type is not a licence to undo it.
    out_of_slide_enabled: bool = False
    out_of_slide_selection: str = "random"


@dataclass(frozen=True)
class SplitsCfg:
    """`dev`/`calibration`/`test` are either three absolute case counts (`mode ==
    "absolute"`) or three fractions of the cohort summing to 1.0 (`mode ==
    "fraction"`) — see spec §8. Config load can only check what it can see (types,
    non-negativity, fraction sum); whether an absolute count matches the real cohort
    size is checked later, in stage 03's gather, via
    `pyrafault.sampling.splits.resolve_split_counts`.
    """
    level: str
    dev: float
    calibration: float
    test: float
    min_contexts_per_slide: dict[str, int]
    max_reshuffle_attempts: int
    mode: str = "absolute"


@dataclass(frozen=True)
class StorageCfg:
    canonical: str
    materialize_pilot_png: bool
    large_scale_patch_format: str


@dataclass(frozen=True)
class QcCfg:
    overlays_per_slide: int
    verify_source_md5: bool
    verify_materialized_hash: bool
    reconstruction_sample_frac: float


@dataclass(frozen=True)
class Config:
    seed: int
    pipeline_version: str
    extraction_version: str
    source: SourceCfg
    slide: SlideCfg
    thumbnail: ThumbnailCfg
    tissue: TissueCfg
    context: ContextCfg
    query: QueryCfg
    quality: QualityCfg
    visual_sampling: VisualSamplingCfg
    features: FeaturesCfg
    faults: FaultsCfg
    splits: SplitsCfg
    storage: StorageCfg
    qc: QcCfg
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    config_digest: str = field(repr=False, default="")

    def digest(self) -> str:
        return self.config_digest

    def section_digests(self) -> dict[str, str]:
        """One digest per top-level section of the raw config.

        `digest()` stays what it was: the identity of the whole config, which is what
        `benchmark_manifest.json` records. These are for asking a narrower question --
        did the part of the config a given stage actually reads change? -- which is
        the difference between a data-side change (e.g. adding a `degradation:`
        section) costing hours and costing days: under a single whole-config digest,
        adding ANY section forces a rebuild from stage 00, re-embedding every query
        candidate with a vision transformer to change three columns that move no
        coordinate. That happened once already, when the last fault types landed.
        """
        out = {}
        for name, value in sorted(self.raw.items()):
            payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
            out[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return out


def _bands(raw: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def _freeze(obj: Any) -> Any:
    """Recursively make dicts/lists read-only: dict -> MappingProxyType, list/tuple -> tuple.

    Used for fields that must not be mutable after `load_config` returns, since
    `Config` is frozen at the top level but plain dicts/lists nested inside a frozen
    dataclass field are otherwise still mutable in place.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    return obj


def _plain(obj: Any) -> Any:
    """Invert `_freeze`: `MappingProxyType` -> dict, tuple -> list, recursively.

    `Config.raw` is frozen (see `_freeze`) so nothing downstream can mutate it in
    place, but `json.dumps` -- used by `section_digests()` to hash a section's
    content -- doesn't know how to serialise a `MappingProxyType` at all, and would
    silently serialise a tuple as a JSON array anyway, which is fine, but only once
    every nested dict has also been converted back. Kept symmetric with `_freeze` for
    the same reason `_freeze` exists: a section digest must depend only on the data,
    never on which container type happens to be holding it at the moment.
    """
    if isinstance(obj, MappingProxyType):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def _canonical_digest(raw: dict[str, Any]) -> str:
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return raw[name]
    except KeyError as e:
        raise ConfigError(f"config missing required section: '{name}'") from e


def _build(cls: type, name: str, **kwargs: Any) -> Any:
    try:
        return cls(**kwargs)
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid fields in '{name}' config: {e}") from e


def _splits_mode_and_values(splits_raw: dict[str, Any]) -> tuple[str, float, float, float]:
    """Classify `splits.dev/calibration/test` as absolute case counts or fractions.

    All three must share a type: `int` (`mode = "absolute"`) or `float` (`mode =
    "fraction"`). YAML already distinguishes `6` from `6.0`, so this is a plain
    `type()` check on the parsed values — no numeric heuristics. Whether the
    *values* make sense (non-negative, fractions summing to 1.0, an absolute total
    matching the real cohort) is checked separately: the parts knowable at config
    load in `_validate_splits`, and the cohort-size match in stage 03's gather
    (`pyrafault.sampling.splits.resolve_split_counts`), since only that stage knows
    how many cases actually exist.
    """
    try:
        values = {k: splits_raw[k] for k in ("dev", "calibration", "test")}
    except KeyError as e:
        raise ConfigError(f"invalid or missing fields in 'splits' config: {e}") from e
    types = {k: type(v) for k, v in values.items()}
    if all(t is int for t in types.values()):
        mode = "absolute"
    elif all(t is float for t in types.values()):
        mode = "fraction"
    else:
        raise ConfigError(
            "splits.dev/calibration/test must be either all integers (absolute case "
            f"counts) or all floats (fractions summing to 1.0); got mixed types {types}"
        )
    return mode, float(values["dev"]), float(values["calibration"]), float(values["test"])


def _validate_splits(splits: SplitsCfg) -> None:
    if splits.dev < 0 or splits.calibration < 0 or splits.test < 0:
        raise ConfigError(
            "splits.dev/calibration/test must be non-negative, got "
            f"dev={splits.dev}, calibration={splits.calibration}, test={splits.test}"
        )
    if splits.calibration == 0 or splits.test == 0:
        raise ConfigError(
            "splits.calibration and splits.test must not be zero — a split must "
            f"never be empty, got calibration={splits.calibration}, test={splits.test}"
        )
    if splits.mode == "fraction":
        total = splits.dev + splits.calibration + splits.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                "splits.dev/calibration/test fractions must sum to 1.0, got "
                f"{total} (dev={splits.dev}, calibration={splits.calibration}, "
                f"test={splits.test})"
            )


def _section_or_empty(raw: dict, name: str) -> dict:
    """A `faults` subsection that may be absent entirely.

    `_section` raises on a missing key, which is right for sections every config must
    have. The two `out_of_slide` fault types postdate the earliest frozen configs, so
    requiring the section would invalidate a config that a published freeze still has
    to load and agree with. An absent section therefore means
    "disabled", not "invalid".
    """
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"faults.{name} must be a mapping, got {type(value).__name__}")
    return value


def _validate_source(source: SourceCfg) -> None:
    if source.mode not in ("gdc", "directory"):
        raise ConfigError(f"source.mode must be 'gdc' or 'directory', got {source.mode!r}")
    if source.mode == "gdc":
        required = ("gdc_dir", "manifest", "sample_sheet", "metadata_json")
        missing = [k for k in required if getattr(source, k) is None]
        if missing:
            raise ConfigError(
                f"source.mode=gdc requires {list(required)}; missing: {missing}"
            )
    else:
        required = ("slides_dir", "pattern", "case_id_from")
        missing = [k for k in required if getattr(source, k) is None]
        if missing:
            raise ConfigError(
                f"source.mode=directory requires {list(required)}; missing: {missing}"
            )


def _validate(cfg: Config) -> None:
    _validate_source(cfg.source)
    if cfg.query.target_mpp > cfg.context.target_mpp:
        raise ConfigError(
            f"query.target_mpp ({cfg.query.target_mpp}) must be finer than "
            f"context.target_mpp ({cfg.context.target_mpp})"
        )
    if cfg.slide.max_mpp_l0 > cfg.query.target_mpp:
        raise ConfigError(
            f"slide.max_mpp_l0 ({cfg.slide.max_mpp_l0}) exceeds query.target_mpp "
            f"({cfg.query.target_mpp}); such a slide cannot be read without upsampling"
        )
    order = ["small", "medium", "large"]
    bands = cfg.faults.severity_bands
    missing = [k for k in order if k not in bands]
    if missing:
        raise ConfigError(f"severity_bands missing: {missing}")
    for k in order:
        lo, hi = bands[k]
        if lo >= hi:
            raise ConfigError(f"severity_bands[{k}] is empty: [{lo}, {hi})")
    for a, b in zip(order, order[1:]):
        if bands[a][1] != bands[b][0]:
            raise ConfigError(
                f"severity_bands must be contiguous: {a} ends at {bands[a][1]} "
                f"but {b} starts at {bands[b][0]}"
            )
    if cfg.visual_sampling.n_clusters != cfg.query.num_per_context:
        raise ConfigError(
            f"visual_sampling.n_clusters ({cfg.visual_sampling.n_clusters}) must equal "
            f"query.num_per_context ({cfg.query.num_per_context})"
        )
    _validate_splits(cfg.splits)
    if cfg.faults.hard_pool not in ("context", "slide"):
        raise ConfigError(f"faults.hard_same_slide.pool must be context|slide, got {cfg.faults.hard_pool}")
    if cfg.faults.out_of_context_selection not in ("random", "hard", "both"):
        raise ConfigError(
            "faults.out_of_context.selection must be random|hard|both, got "
            f"{cfg.faults.out_of_context_selection!r}"
        )
    if cfg.faults.out_of_slide_selection not in ("random", "hard", "both"):
        raise ConfigError(
            "faults.out_of_slide.selection must be random|hard|both, got "
            f"{cfg.faults.out_of_slide_selection!r}"
        )
    if cfg.faults.out_of_context_min_gap < 0:
        raise ConfigError(
            "faults.out_of_context.min_gap_context_widths must be >= 0, got "
            f"{cfg.faults.out_of_context_min_gap}"
        )
    if not 0.0 < cfg.context.stride_frac <= 1.0:
        raise ConfigError(f"context.stride_frac out of range: {cfg.context.stride_frac}")


def load_config(source: str | Path | dict[str, Any]) -> Config:
    if isinstance(source, dict):
        raw = source
    else:
        with open(source, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

    # Computed from the untouched, still-mutable `raw` so recreating this exact
    # config always yields the same digest regardless of how `raw` is later frozen
    # for storage on the Config object.
    config_digest = _canonical_digest(raw)

    try:
        seed = int(raw["seed"])
        pipeline_version = str(raw["pipeline_version"])
        extraction_version = str(raw["extraction_version"])
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid or missing top-level fields: {e}") from e

    faults_raw = _section(raw, "faults")
    d = _section(faults_raw, "adjacent_drift")
    r = _section(faults_raw, "random_same_context")
    h = _section(faults_raw, "hard_same_slide")
    o = _section(faults_raw, "out_of_context")
    q = _section(raw, "quality")
    features_raw = _section(raw, "features")
    splits_raw = _section(raw, "splits")

    try:
        quality = QualityCfg(
            blur_var_flag=float(q["blur_var_flag"]),
            blur_var_exclude=float(q["blur_var_exclude"]),
            mean_intensity_flag=tuple(q["mean_intensity_flag"]),
            mean_intensity_exclude=tuple(q["mean_intensity_exclude"]),
            pen_mark_flag_frac=float(q["pen_mark_flag_frac"]),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid or missing fields in 'quality' config: {e}") from e

    try:
        faults = FaultsCfg(
            clean=bool(faults_raw["clean"]),
            drift_enabled=bool(d["enabled"]),
            drift_normalize_by=str(d["normalize_by"]),
            severity_bands=_freeze(_bands(d["severity_bands"])),
            drift_balance=str(d["balance"]),
            max_theta_retries=int(d["max_theta_retries"]),
            random_enabled=bool(r["enabled"]),
            random_max_iou=float(r["max_iou_with_true"]),
            random_min_center_dist=float(r["min_center_dist_query_widths"]),
            hard_enabled=bool(h["enabled"]),
            hard_pool=str(h["pool"]),
            hard_max_iou=float(h["max_iou_with_true"]),
            hard_min_center_dist=float(h["min_center_dist_query_widths"]),
            hard_top_k=int(h["top_k"]),
            out_of_context_enabled=bool(o["enabled"]),
            out_of_context_selection=str(o.get("selection", "both")),
            out_of_context_min_gap=float(o.get("min_gap_context_widths", 0.0)),
            out_of_slide_enabled=bool(_section_or_empty(faults_raw, "out_of_slide").get("enabled", False)),
            out_of_slide_selection=str(_section_or_empty(faults_raw, "out_of_slide").get("selection", "random")),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid or missing fields in 'faults' config: {e}") from e

    try:
        features = FeaturesCfg(
            uni=_freeze(features_raw["uni"]),
            dinov2=_freeze(features_raw["dinov2"]),
            batch_size=int(features_raw["batch_size"]),
            device=str(features_raw["device"]),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid or missing fields in 'features' config: {e}") from e

    splits_mode, splits_dev, splits_cal, splits_test = _splits_mode_and_values(splits_raw)
    try:
        splits = SplitsCfg(
            level=str(splits_raw["level"]),
            dev=splits_dev,
            calibration=splits_cal,
            test=splits_test,
            min_contexts_per_slide=_freeze(splits_raw["min_contexts_per_slide"]),
            max_reshuffle_attempts=int(splits_raw["max_reshuffle_attempts"]),
            mode=splits_mode,
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid or missing fields in 'splits' config: {e}") from e

    cfg = Config(
        seed=seed,
        pipeline_version=pipeline_version,
        extraction_version=extraction_version,
        source=_build(SourceCfg, "source", **_section(raw, "source")),
        slide=_build(SlideCfg, "slide", **_section(raw, "slide")),
        thumbnail=_build(ThumbnailCfg, "thumbnail", **_section(raw, "thumbnail")),
        tissue=_build(TissueCfg, "tissue", **_section(raw, "tissue")),
        context=_build(ContextCfg, "context", **_section(raw, "context")),
        query=_build(QueryCfg, "query", **_section(raw, "query")),
        quality=quality,
        visual_sampling=_build(VisualSamplingCfg, "visual_sampling", **_section(raw, "visual_sampling")),
        features=features,
        faults=faults,
        splits=splits,
        storage=_build(StorageCfg, "storage", **_section(raw, "storage")),
        qc=_build(QcCfg, "qc", **_section(raw, "qc")),
        raw=_freeze(raw),
        config_digest=config_digest,
    )
    _validate(cfg)
    return cfg
