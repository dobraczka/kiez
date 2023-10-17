import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.neighbors import NMSLIB
from kiez.neighbors.util import available_ann_algorithms

APPROXIMATE_ALGORITHMS = available_ann_algorithms()
if NMSLIB not in APPROXIMATE_ALGORITHMS:
    skip = True
else:
    skip = False
skip_reason = "NMSLIB not installed"

rng = np.random.RandomState(2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_metric():
    with pytest.raises(ValueError) as exc_info:
        NMSLIB(metric="jibberish")
    assert "Unknown" in str(exc_info.value)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_sqeuclidean(n_samples=20, n_features=5, n_neighbors=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    hnsw1 = NMSLIB(n_candidates=n_neighbors, metric="sqeuclidean")
    hnsw1.fit(source, target)
    d, i = hnsw1.kneighbors(
        query=source[
            :5,
        ]
    )
    hnsw2 = NMSLIB(n_candidates=n_neighbors)
    hnsw2.fit(source, target)
    i2 = hnsw2.kneighbors(
        query=source[
            :5,
        ],
        return_distance=False,
    )
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_cosine(n_samples=20, n_features=5, n_neighbors=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    hnsw1 = NMSLIB(n_candidates=n_neighbors, metric="cosine")
    hnsw1.fit(source, target)
    d, i = hnsw1.kneighbors(
        query=source[
            :5,
        ]
    )
    hnsw2 = NMSLIB(n_candidates=n_neighbors, metric="cosinesimil")
    hnsw2.fit(source, target)
    i2 = hnsw2.kneighbors(
        query=source[
            :5,
        ],
        return_distance=False,
    )
    assert_array_equal(i, i2)
