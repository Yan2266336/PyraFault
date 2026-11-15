import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from pyrafault.sampling.diversity import cluster_stratified, self_similarity


def _blobs(k=4, per=25, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim)) * 5
    x = np.concatenate([c + rng.normal(scale=0.2, size=(per, dim)) for c in centres])
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), k, per


def test_empty_cluster_backfill_on_a_degenerate_embedding_matrix():
    """A degenerate pool (all embeddings identical) forces KMeans to collapse every
    point onto one cluster, leaving k-1 clusters empty -- this exercises the backfill
    branch (real index arithmetic that freezes queries.parquet) which otherwise had
    zero test coverage. Must still return exactly k unique, in-range indices."""
    n, k, dim = 10, 5, 4
    emb = np.ones((n, dim), dtype=np.float32)
    # Expected and intentional for this deliberately-degenerate input, not a real
    # gap: sklearn warns that fewer distinct clusters than k exist among duplicate
    # points, which is exactly the empty-cluster condition this test targets.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        idx, labels, dist = cluster_stratified(emb, k=k, seed=0)
    assert len(idx) == k
    assert len(set(idx)) == k, "backfill must not pick the same candidate twice"
    assert all(0 <= i < n for i in idx)
    assert idx == sorted(idx)
    assert labels.shape == (n,) and dist.shape == (n,)


def test_empty_cluster_backfill_is_deterministic():
    emb = np.ones((10, 4), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        a = cluster_stratified(emb, k=5, seed=0)[0]
        b = cluster_stratified(emb, k=5, seed=0)[0]
    assert a == b


def test_picks_one_representative_per_cluster():
    emb, k, per = _blobs()
    idx, labels, dist = cluster_stratified(emb, k=4, seed=1)
    assert len(idx) == 4
    assert len({i // per for i in idx}) == 4, "one pick from each true blob"
    assert labels.shape == (len(emb),) and dist.shape == (len(emb),)


def test_selected_points_are_nearest_to_their_centroid():
    emb, k, per = _blobs()
    idx, labels, dist = cluster_stratified(emb, k=4, seed=1)
    for i in idx:
        same = np.flatnonzero(labels == labels[i])
        assert dist[i] == pytest.approx(dist[same].min())


def test_is_deterministic_for_a_given_seed():
    emb, _, _ = _blobs()
    a = cluster_stratified(emb, 4, seed=7)[0]
    b = cluster_stratified(emb, 4, seed=7)[0]
    assert a == b


def test_returns_all_when_fewer_candidates_than_clusters():
    emb = np.eye(3, 8, dtype=np.float32)
    idx, labels, dist = cluster_stratified(emb, k=4, seed=1)
    assert sorted(idx) == [0, 1, 2]


def test_self_similarity_is_high_inside_a_tight_blob():
    emb, k, per = _blobs()
    ss = self_similarity(emb, indices=[0, per, 2 * per], k=8)
    assert ss.shape == (3,)
    assert (ss > 0.9).all(), "neighbours within a tight blob are near-identical"


def test_self_similarity_excludes_the_point_itself():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    ss = self_similarity(emb, indices=[0], k=1)
    assert ss[0] == pytest.approx(1.0)      # its twin, not itself
    ss2 = self_similarity(emb, indices=[2], k=1)
    assert ss2[0] == pytest.approx(0.0)


def test_self_similarity_handles_k_larger_than_the_pool():
    emb = np.eye(3, 4, dtype=np.float32)
    ss = self_similarity(emb, indices=[0], k=100)
    assert np.isfinite(ss).all()
