"""Visual diversity sampling (guideline §8.2, §8.3).

This controls visual coverage only. Clusters are numbered, never named — assigning them
medical meaning would turn the benchmark into an annotation task, which §8.1 explicitly
rules out.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans


def cluster_stratified(
    embeddings: np.ndarray, k: int, seed: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Pick the candidate nearest each of `k` KMeans centroids.

    Returns `(selected_indices, cluster_label_per_candidate, distance_to_own_centroid)`.
    """
    n = len(embeddings)
    if n == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    if n <= k:
        return (
            list(range(n)),
            np.arange(n, dtype=np.int64),
            np.zeros(n, dtype=np.float64),
        )

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings).astype(np.int64)
    dist = np.linalg.norm(embeddings - km.cluster_centers_[labels], axis=1)

    chosen: list[int] = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        if members.size == 0:
            continue
        chosen.append(int(members[np.argmin(dist[members])]))

    # A degenerate empty cluster would leave fewer than k picks; backfill with the
    # unchosen candidates furthest from everything already selected.
    if len(chosen) < k:
        remaining = [i for i in range(n) if i not in set(chosen)]
        while len(chosen) < k and remaining:
            sims = embeddings[remaining] @ embeddings[chosen].T
            far = int(np.argmin(sims.max(axis=1)))
            chosen.append(remaining.pop(far))

    return sorted(chosen), labels, dist


def self_similarity(embeddings: np.ndarray, indices: Sequence[int], k: int) -> np.ndarray:
    """Mean cosine similarity to the `k` nearest neighbours, excluding the point itself.

    High self-similarity marks repetitive tissue — glands, dense stroma — which §8.3 calls
    out as a source of benchmark difficulty rather than duplicate data to be removed.
    """
    if len(embeddings) < 2 or not len(indices):
        return np.zeros(len(indices), dtype=np.float64)

    sims = embeddings[np.asarray(indices)] @ embeddings.T
    for row, i in enumerate(indices):
        sims[row, i] = -np.inf
    kk = min(k, embeddings.shape[0] - 1)
    top = np.partition(sims, -kk, axis=1)[:, -kk:]
    return top.mean(axis=1).astype(np.float64)
