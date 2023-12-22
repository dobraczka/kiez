import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.neighbors import NNG
from kiez.neighbors.util import available_nn_algorithms

NN_ALGORITHMS = available_nn_algorithms()
skip = NNG not in NN_ALGORITHMS
skip_reason = "NNG not installed"


rng = np.random.RandomState(2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_metric():
    with pytest.raises(ValueError, match="Unknown"):
        NNG(metric="jibberish")


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_dir(source_target):
    source, _ = source_target
    with pytest.raises(TypeError, match="NNG requires"):
        NNG(index_dir=1)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_right_dir(tmp_path, source_target):
    source, target = source_target
    nng = NNG(index_dir=str(tmp_path))
    nng.fit(source, target)
    assert nng is not None


@pytest.mark.skipif(skip, reason=skip_reason)
def test_none_dir(source_target):
    source, target = source_target
    nng = NNG(index_dir=None)
    nng.fit(source, target)
    assert nng is not None


@pytest.mark.skipif(skip, reason=skip_reason)
def test_self_query(source_target, n_neighbors=5):
    source, _ = source_target
    nng = NNG(index_dir=None, n_candidates=n_neighbors, epsilon=0.00001)
    nng.fit(source, source)
    d, i = nng.kneighbors()
    i2 = nng.kneighbors(return_distance=False)
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_query(source_target, n_neighbors=5):
    source, target = source_target
    nng = NNG(index_dir=None, n_candidates=n_neighbors, epsilon=0.00001)
    nng.fit(source, target)
    d, i = nng.kneighbors()
    i2 = nng.kneighbors(return_distance=False)
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_sqeuclidean(source_target, n_neighbors=5):
    source, target = source_target
    nng1 = NNG(index_dir=None, n_candidates=n_neighbors, metric="sqeuclidean")
    nng1.fit(source, target)
    d, i = nng1.kneighbors()
    nng2 = NNG(index_dir=None, n_candidates=n_neighbors)
    nng2.fit(source, target)
    i2 = nng2.kneighbors(return_distance=False)
    assert_array_equal(i, i2)
