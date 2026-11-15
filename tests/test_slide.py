import pytest
from pyrafault.io.slide import select_level, UpsamplingError

APERIO = (1.0, 4.0, 16.0, 32.0)          # the pilot slides
APERIO_SHORT = (1.0, 4.0, 16.0)          # TCGA-55-8094
WITH_EIGHT = (1.0, 2.0, 4.0, 8.0, 16.0)  # hypothetical scanner exposing native 5x


def test_context_5x_reads_level1_on_pilot_pyramid():
    need = 2.0 / 0.2520          # 7.937
    assert select_level(APERIO, need) == 1
    assert select_level(APERIO_SHORT, need) == 1


def test_query_20x_reads_level0_on_pilot_pyramid():
    need = 0.5 / 0.2520          # 1.984
    assert select_level(APERIO, need) == 0


def test_prefers_a_native_level_when_one_matches_exactly():
    assert select_level(WITH_EIGHT, 8.0) == 3      # native ds 8.0, not the finer ds 4.0
    assert select_level(WITH_EIGHT, 2.0) == 1


def test_never_upsamples_to_reach_a_nearby_native_level():
    """ds 8.0 is 0.8% coarser than 2.0/0.2520; taking it would upscale 2032 -> 2048."""
    assert select_level(WITH_EIGHT, 2.0 / 0.2520) == 2      # ds 4.0, then downscale
    assert select_level(WITH_EIGHT, 0.5 / 0.2520) == 0      # ds 1.0, then downscale


def test_exact_match_is_accepted_within_tolerance():
    assert select_level(APERIO, 4.0) == 1
    assert select_level(APERIO, 4.0 * (1 - 1e-9)) == 1


def test_refuses_to_upsample():
    with pytest.raises(UpsamplingError):
        select_level(APERIO, 0.5)            # need_ds < 1
    with pytest.raises(UpsamplingError):
        select_level((4.0, 16.0), 2.0)       # no level at or below need_ds


def test_unsorted_downsamples_still_pick_the_coarsest_eligible():
    assert select_level((16.0, 1.0, 4.0), 7.937) == 2


@pytest.mark.slow
def test_real_slide_metadata(pilot_slide_path):
    from pyrafault.io.slide import open_slide
    with open_slide(pilot_slide_path, "TCGA-55-8094-01Z-00-DX1") as s:
        m = s.meta
        assert (m.width_l0, m.height_l0) == (35492, 31380)
        assert m.mpp_x == pytest.approx(0.2520, abs=1e-4)
        assert m.objective_power == 40
        assert m.level_downsamples[0] == 1.0
        assert m.vendor == "aperio"
