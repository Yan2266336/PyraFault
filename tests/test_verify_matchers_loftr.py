"""LoFTR at the sizes this benchmark actually uses.

Marked slow: it downloads and runs a network. Measured before the design committed to
it -- at 128 px a true window yields 144 matches against 2-24 for a random one.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _textured(seed, size=128):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size // 8, size // 8), dtype=np.uint8)
    return np.kron(small, np.ones((8, 8), dtype=np.uint8))


def test_loftr_separates_a_matching_window_from_an_unrelated_one():
    from pyrafault.verify.matchers import LoftrMatcher
    m = LoftrMatcher()
    img = _textured(0)
    assert m.match(img, img).score > m.match(img, _textured(999)).score


def test_loftr_satisfies_the_matcher_interface():
    from pyrafault.verify.matchers import LoftrMatcher, Matcher
    assert isinstance(LoftrMatcher(), Matcher)


def test_loftr_handles_the_smallest_shipped_window_without_raising():
    """The 5x->40x freeze uses 65 px windows -- far below LoFTR's usual ~480."""
    from pyrafault.verify.matchers import LoftrMatcher
    r = LoftrMatcher().match(_textured(0, 65), _textured(0, 65))
    assert 0.0 <= r.score <= 1.0
