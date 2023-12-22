import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.neighbors import NMSLIB
from kiez.neighbors.util import available_nn_algorithms

NN_ALGORITHMS = available_nn_algorithms()
if NMSLIB not in NN_ALGORITHMS:
    skip = True
else:
    skip = False
skip_reason = "NMSLIB not installed"

rng = np.random.RandomState(2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_metric():
    with pytest.raises(ValueError, match="Unknown"):
        NMSLIB(metric="jibberish")


@pytest.mark.skipif(skip, reason=skip_reason)
def test_sqeuclidean(source_target, n_neighbors=5):
    source, target = source_target
    hnsw1 = NMSLIB(n_candidates=n_neighbors, metric="sqeuclidean")
    hnsw1.fit(source, target)
    d, i = hnsw1.kneighbors()
    hnsw2 = NMSLIB(n_candidates=n_neighbors)
    hnsw2.fit(source, target)
    i2 = hnsw2.kneighbors(return_distance=False)
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_cosine(source_target, n_neighbors=5):
    source, target = source_target
    hnsw1 = NMSLIB(n_candidates=n_neighbors, metric="cosine")
    hnsw1.fit(source, target)
    d, i = hnsw1.kneighbors()
    hnsw2 = NMSLIB(n_candidates=n_neighbors, metric="cosinesimil")
    hnsw2.fit(source, target)
    i2 = hnsw2.kneighbors(return_distance=False)
    assert_array_equal(i, i2)
