"""The retrieval grid.

Two properties matter more than the tiling itself. The grid must not reproduce the
query lattice the benchmark was built on -- a retrieval grid that IS the construction
grid hands the method the construction's prior. And the stride must be half the window,
because a true position a full window away from every grid point is indistinguishable
from noise (measured: 18 matches at offset 64 against 2-24 for a random window).
"""
import numpy as np
import pytest

from pyrafault.verify.bank import WindowBank, bank_key, window_origins, window_side


class _Cfg:
    class context: size_px, target_mpp = 2048, 2.0
    class query: size_px, target_mpp = 512, 0.5


class TestGeometry:
    def test_window_side_is_the_query_footprint_in_context_pixels(self):
        assert window_side(_Cfg) == 128

    def test_a_finer_query_gives_a_smaller_footprint(self):
        class C(_Cfg):
            class query: size_px, target_mpp = 512, 0.2527
        assert window_side(C) == 65

    def test_origins_tile_the_context_at_the_given_stride(self):
        o = window_origins(2048, side=128, stride=64)
        assert len(o) == 31 * 31
        assert (0, 0) in o and (1920, 1920) in o

    def test_no_window_escapes_the_context(self):
        for x, y in window_origins(2048, side=128, stride=64):
            assert 0 <= x <= 2048 - 128 and 0 <= y <= 2048 - 128

    def test_stride_defaults_to_half_the_window(self):
        assert window_origins(2048, side=128) == window_origins(2048, side=128, stride=64)


class TestBankKey:
    def test_the_key_includes_the_slide(self):
        """bbox alone collides: the pilot freeze's 187 contexts occupy 136 distinct
        boxes, so a bbox-only index would serve 27% of items another context's bank."""
        from pyrafault.geom import BBoxL0
        box = BBoxL0(0, 0, 100, 100)
        assert bank_key("S1", box) != bank_key("S2", box)

    def test_the_key_is_hashable_and_stable(self):
        from pyrafault.geom import BBoxL0
        box = BBoxL0(4, 8, 100, 100)
        assert bank_key("S1", box) == bank_key("S1", BBoxL0(4, 8, 100, 100))
        assert isinstance(hash(bank_key("S1", box)), int)


class TestPersistence:
    def test_round_trips(self, tmp_path):
        bank = WindowBank("ctx0", side=128,
                          origins=[(0, 0), (64, 0)],
                          embeddings=np.ones((2, 4), dtype=np.float16))
        p = tmp_path / "ctx0.npz"
        bank.save(p)
        back = WindowBank.load(p)
        assert back.context_id == "ctx0" and back.side == 128
        assert back.origins == [(0, 0), (64, 0)]
        assert np.array_equal(back.embeddings, bank.embeddings)

    def test_rejects_a_row_count_that_does_not_match_its_origins(self):
        with pytest.raises(ValueError, match="origins"):
            WindowBank("ctx0", side=128, origins=[(0, 0)],
                       embeddings=np.ones((2, 4), dtype=np.float16))


class TestNormalizedEmbeddings:
    """Finding 2: the normalised matrix must be computed once and cached, not rebuilt
    on every retrieval call -- that is the amortised-cost premise the bank exists for
    (spec S3.1)."""

    def test_rows_are_unit_norm(self):
        bank = WindowBank("ctx0", side=128, origins=[(0, 0), (64, 0)],
                          embeddings=np.array([[3, 4, 0, 0], [1, 1, 1, 1]],
                                              dtype=np.float16))
        norms = np.linalg.norm(bank.normalized_embeddings, axis=1)
        assert norms == pytest.approx([1.0, 1.0], abs=1e-5)

    def test_a_zero_row_normalizes_to_zero_not_nan(self):
        bank = WindowBank("ctx0", side=128, origins=[(0, 0)],
                          embeddings=np.zeros((1, 4), dtype=np.float16))
        assert np.all(np.isfinite(bank.normalized_embeddings))

    def test_computed_once_and_cached(self):
        """A frozen dataclass's cached_property must actually cache -- verified by
        object identity across two accesses -- because on a (900-3844, 1024) bank
        recomputing this every query is exactly the cost finding 2 removes."""
        bank = WindowBank("ctx0", side=128, origins=[(0, 0)],
                          embeddings=np.ones((1, 4), dtype=np.float16))
        assert bank.normalized_embeddings is bank.normalized_embeddings

    def test_matches_the_original_per_query_cosine_similarity(self):
        """Regression: the cached matrix must reproduce the same ranking the old
        per-call `emb @ q / (norm(emb) * norm(q))` formula gave."""
        emb = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float16)
        bank = WindowBank("ctx0", side=128, origins=[(0, 0), (1, 0), (2, 0)],
                          embeddings=emb)
        q = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        old = (emb.astype(np.float32) @ q) / (
            np.linalg.norm(emb.astype(np.float32), axis=1) * np.linalg.norm(q) + 1e-12
        )
        new = bank.normalized_embeddings @ (q / (np.linalg.norm(q) + 1e-12))
        assert new == pytest.approx(old, abs=1e-5)
