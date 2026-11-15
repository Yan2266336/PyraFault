import numpy as np
import pytest
from PIL import Image

from pyrafault.geom import BBoxL0
from pyrafault.qc.overlay import (
    CLAIMED_COLOR, TRUE_COLOR, bbox_to_context_px, render_pair,
)

CTX = BBoxL0(1000, 2000, 16254, 16254)
OUT = 2048


def test_context_origin_maps_to_zero():
    assert bbox_to_context_px(BBoxL0(1000, 2000, 1016, 1016), CTX, OUT)[:2] == (0, 0)


def test_query_footprint_is_128_px_in_a_2048_context():
    x0, y0, x1, y1 = bbox_to_context_px(BBoxL0(1000, 2000, 1016, 1016), CTX, OUT)
    assert (x1 - x0, y1 - y0) == (128, 128)


def test_mapping_is_not_transposed():
    """A box offset only in x must move only in x."""
    box = BBoxL0(1000 + 4 * 1016, 2000, 1016, 1016)
    x0, y0, _, _ = bbox_to_context_px(box, CTX, OUT)
    assert y0 == 0 and x0 == pytest.approx(512, abs=1)


def test_far_corner_maps_to_the_far_corner():
    box = BBoxL0(CTX.x1 - 1016, CTX.y1 - 1016, 1016, 1016)
    x0, y0, x1, y1 = bbox_to_context_px(box, CTX, OUT)
    assert x1 == OUT and y1 == OUT


def test_render_draws_green_true_and_red_claimed():
    ctx = Image.fromarray(np.full((OUT, OUT, 3), 255, np.uint8))
    qry = Image.fromarray(np.full((512, 512, 3), 200, np.uint8))
    out = render_pair(ctx, qry, (100, 100, 228, 228), (600, 600, 728, 728), "cap")
    arr = np.asarray(out)
    assert any(np.all(arr.reshape(-1, 3) == TRUE_COLOR, axis=1))
    assert any(np.all(arr.reshape(-1, 3) == CLAIMED_COLOR, axis=1))


def test_render_places_the_query_to_the_right_of_the_context():
    ctx = Image.fromarray(np.full((OUT, OUT, 3), 255, np.uint8))
    qry = Image.fromarray(np.full((512, 512, 3), 200, np.uint8))
    out = render_pair(ctx, qry, (0, 0, 128, 128), (0, 0, 128, 128), "cap")
    assert out.width > out.height


def test_clean_pair_boxes_coincide_exactly():
    box = BBoxL0(1000 + 3 * 1016, 2000 + 5 * 1016, 1016, 1016)
    assert bbox_to_context_px(box, CTX, OUT) == bbox_to_context_px(box, CTX, OUT)
