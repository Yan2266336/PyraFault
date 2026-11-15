import pytest
from pyrafault.config import load_config, ConfigError

CFG = "configs/example10.yaml"


def test_loads_pilot_config():
    cfg = load_config(CFG)
    assert cfg.seed == 20260812
    assert cfg.extraction_version == "v1"
    assert cfg.context.target_mpp == 2.0
    assert cfg.context.size_px == 2048
    assert cfg.context.stride_frac == 0.25
    assert cfg.context.num_per_slide == 30
    assert cfg.query.target_mpp == 0.5
    assert cfg.query.size_px == 512
    assert cfg.query.num_per_context == 5
    assert cfg.query.boundary_margin_frac == 1.0
    # All eight fault types are on in the shipped config: the three magnification
    # freezes are meant to share one label vocabulary, so a consumer can concatenate
    # them and stratify by scale ratio without first discovering that one of them is
    # missing two classes.
    assert cfg.faults.out_of_context_enabled is True
    assert cfg.faults.out_of_slide_enabled is True
    assert cfg.faults.hard_pool == "context"
    assert cfg.splits.min_contexts_per_slide == {"test": 15, "calibration": 15}


def test_config_is_frozen():
    cfg = load_config(CFG)
    with pytest.raises(Exception):
        cfg.context.size_px = 1024


def test_digest_is_stable_and_sensitive(tmp_path, write_config):
    a = load_config(CFG).digest()
    b = load_config(CFG).digest()
    assert a == b and len(a) == 64

    p = write_config(tmp_path / "mod.yaml", context_size_px=1024)
    assert load_config(p).digest() != a


def test_rejects_query_mpp_coarser_than_context():
    import yaml
    raw = yaml.safe_load(open(CFG))
    raw["query"]["target_mpp"] = 4.0
    with pytest.raises(ConfigError, match="query.target_mpp"):
        load_config(raw)


def test_rejects_severity_bands_with_gap():
    import yaml
    raw = yaml.safe_load(open(CFG))
    raw["faults"]["adjacent_drift"]["severity_bands"]["medium"] = [1.0, 2.0]
    with pytest.raises(ConfigError, match="severity_bands"):
        load_config(raw)


def test_raw_and_nested_dicts_are_immutable():
    cfg = load_config(CFG)
    with pytest.raises(TypeError):
        cfg.raw["seed"] = 1
    with pytest.raises(TypeError):
        cfg.faults.severity_bands["small"] = (0.0, 1.0)


def test_digest_unaffected_by_mutation_attempt():
    cfg = load_config(CFG)
    original = cfg.digest()
    with pytest.raises(TypeError):
        cfg.raw["seed"] = 999999
    assert cfg.digest() == original


def test_rejects_missing_section():
    import yaml
    raw = yaml.safe_load(open(CFG))
    del raw["source"]
    with pytest.raises(ConfigError, match="source"):
        load_config(raw)


def test_rejects_unknown_key_in_section():
    import yaml
    raw = yaml.safe_load(open(CFG))
    raw["source"]["unexpected_key"] = "oops"
    with pytest.raises(ConfigError, match="source"):
        load_config(raw)


# ---------------------------------------------------------------------------
# splits: cohort-agnostic — integer counts or fraction, per spec §8.
# The "does the total match the real cohort" check is NOT here: it cannot be,
# since config load has no way to know the cohort size. That check lives in
# stage 03's gather (see tests/test_splits.py).
# ---------------------------------------------------------------------------

def _raw():
    import yaml
    return yaml.safe_load(open(CFG))


def test_splits_accepts_integer_case_counts():
    raw = _raw()
    raw["splits"].update(dev=6, calibration=2, test=2)
    cfg = load_config(raw)
    assert cfg.splits.mode == "absolute"
    assert (cfg.splits.dev, cfg.splits.calibration, cfg.splits.test) == (6, 2, 2)


def test_splits_accepts_float_fractions_summing_to_one():
    raw = _raw()
    raw["splits"].update(dev=0.6, calibration=0.2, test=0.2)
    cfg = load_config(raw)
    assert cfg.splits.mode == "fraction"
    assert (cfg.splits.dev, cfg.splits.calibration, cfg.splits.test) == (0.6, 0.2, 0.2)


def test_splits_rejects_mixed_types():
    raw = _raw()
    raw["splits"].update(dev=6, calibration=0.2, test=0.2)
    with pytest.raises(ConfigError, match="mixed types"):
        load_config(raw)


def test_splits_rejects_negative_integer_count():
    raw = _raw()
    raw["splits"].update(dev=12, calibration=-1, test=2)
    with pytest.raises(ConfigError, match="non-negative"):
        load_config(raw)


def test_splits_rejects_negative_fraction():
    raw = _raw()
    raw["splits"].update(dev=0.9, calibration=-0.1, test=0.2)
    with pytest.raises(ConfigError, match="non-negative"):
        load_config(raw)


def test_splits_rejects_fractions_not_summing_to_one():
    raw = _raw()
    raw["splits"].update(dev=0.5, calibration=0.2, test=0.2)
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(raw)


def test_splits_rejects_zero_test_count():
    raw = _raw()
    raw["splits"].update(dev=8, calibration=2, test=0)
    with pytest.raises(ConfigError, match="never be empty"):
        load_config(raw)


def test_splits_rejects_zero_calibration_fraction():
    raw = _raw()
    raw["splits"].update(dev=1.0, calibration=0.0, test=0.0)
    with pytest.raises(ConfigError, match="never be empty"):
        load_config(raw)


def test_splits_does_not_validate_cohort_size_at_load_time():
    """A config for a 50-case cohort must load fine at this layer — the mismatch
    against the pilot's real 10 cases is a stage-03 concern, not a config concern."""
    raw = _raw()
    raw["splits"].update(dev=40, calibration=5, test=5)
    cfg = load_config(raw)
    assert (cfg.splits.dev, cfg.splits.calibration, cfg.splits.test) == (40, 5, 5)


# ---------------------------------------------------------------------------
# Regression guard: generalising must not silently change what the shipped
# pilot config means.
# ---------------------------------------------------------------------------

def test_pilot_config_still_describes_the_pilot_cohort():
    cfg = load_config(CFG)
    assert cfg.splits.mode == "absolute"
    assert (cfg.splits.dev, cfg.splits.calibration, cfg.splits.test) == (6, 2, 2)
    from pyrafault.sampling.splits import resolve_split_counts
    assert resolve_split_counts(cfg.splits, 10) == {"dev": 6, "calibration": 2, "test": 2}


# ---------------------------------------------------------------------------
# source: exact filename or glob (spec §12)
# ---------------------------------------------------------------------------

def test_pilot_config_source_filenames_are_globs():
    cfg = load_config(CFG)
    assert cfg.source.manifest == "gdc_manifest.*.txt"
    assert cfg.source.sample_sheet == "gdc_sample_sheet.*.tsv"
    assert cfg.source.metadata_json == "metadata.cart.*.json"


# ---------------------------------------------------------------------------
# source.mode: gdc | directory (spec §6 "Stage 00 is an adapter, not the
# foundation"). `mode` must default to "gdc" so configs.yaml written before
# this key existed — including the frozen configs/example10.yaml — keep their
# meaning and, critically, their digest.
# ---------------------------------------------------------------------------

# Pinned so an unintended edit to configs/example10.yaml fails here rather than at the
# end of a rebuild. Changing it is legitimate -- it was changed deliberately when
# the out-of-context and out-of-slide fault types were enabled -- but it must be a
# decision, taken together with re-freezing every data root built from this config,
# because stage 09 compares this digest three ways (archived copy, feature manifest,
# live config) and any freeze built under the old one now fails config_drift.
PILOT_DIGEST = "75184108727e48022b2da34b7f568c4045b49fdfdc783a7329871f8e97b218e4"


def test_pilot_config_still_loads_and_means_mode_gdc():
    cfg = load_config(CFG)
    assert cfg.source.mode == "gdc"
    assert cfg.source.gdc_dir == "data/00_source/gdc_download"
    assert cfg.source.root_dir == cfg.source.gdc_dir


def test_pilot_config_digest_is_unchanged():
    """The digest is compared three ways by stage 09's config_drift check
    (archived copy, feature manifest, live config); if this ever changes,
    the frozen benchmark's provenance chain breaks and stage 05's GPU
    embedding freeze would need re-running for nothing."""
    assert load_config(CFG).digest() == PILOT_DIGEST


def test_mode_absent_defaults_to_gdc():
    raw = _raw()
    assert "mode" not in raw["source"]
    cfg = load_config(raw)
    assert cfg.source.mode == "gdc"


def test_directory_mode_config_with_no_gdc_keys_loads():
    raw = _raw()
    raw["source"] = {
        "mode": "directory",
        "slides_dir": "/data/some/cohort",
        "pattern": "*.svs",
        "case_id_from": "slide_id",
    }
    cfg = load_config(raw)
    assert cfg.source.mode == "directory"
    assert cfg.source.gdc_dir is None
    assert cfg.source.manifest is None
    assert cfg.source.root_dir == "/data/some/cohort"


def test_directory_mode_missing_required_field_is_a_config_error():
    raw = _raw()
    raw["source"] = {"mode": "directory", "slides_dir": "/data/x", "pattern": "*.svs"}
    with pytest.raises(ConfigError, match="case_id_from"):
        load_config(raw)


def test_gdc_mode_missing_required_field_is_a_config_error():
    raw = _raw()
    del raw["source"]["gdc_dir"]
    with pytest.raises(ConfigError, match="gdc_dir"):
        load_config(raw)


def test_unknown_source_mode_is_a_config_error():
    raw = _raw()
    raw["source"]["mode"] = "s3"
    with pytest.raises(ConfigError, match="mode"):
        load_config(raw)


class TestOutOfContextConfig:
    """`faults.out_of_context` gained `selection` and `min_gap_context_widths`. Both
    default, so `configs/example10.yaml` -- and the frozen benchmark's config digest --
    stay byte-identical; see `test_pilot_digest_unchanged`."""

    def _cfg(self, tmp_path, **ooc):
        import yaml
        raw = yaml.safe_load(open("configs/example10.yaml"))
        raw["faults"]["out_of_context"] = {"enabled": True, **ooc}
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(raw, sort_keys=False))
        return p

    def test_selection_defaults_to_both(self, tmp_path):
        from pyrafault.config import load_config
        assert load_config(self._cfg(tmp_path)).faults.out_of_context_selection == "both"

    @pytest.mark.parametrize("value", ["random", "hard", "both"])
    def test_valid_selections_load(self, tmp_path, value):
        from pyrafault.config import load_config
        cfg = load_config(self._cfg(tmp_path, selection=value))
        assert cfg.faults.out_of_context_selection == value

    def test_unknown_selection_is_rejected_by_name(self, tmp_path):
        from pyrafault.config import ConfigError, load_config
        with pytest.raises(ConfigError, match="random|hard|both"):
            load_config(self._cfg(tmp_path, selection="nearest"))

    def test_negative_gap_is_rejected(self, tmp_path):
        from pyrafault.config import ConfigError, load_config
        with pytest.raises(ConfigError, match="min_gap_context_widths"):
            load_config(self._cfg(tmp_path, min_gap_context_widths=-1.0))
