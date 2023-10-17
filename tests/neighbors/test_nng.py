import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.neighbors import NNG
from kiez.neighbors.util import available_ann_algorithms

APPROXIMATE_ALGORITHMS = available_ann_algorithms()
if NNG not in APPROXIMATE_ALGORITHMS:
    skip = True
else:
    skip = False
skip_reason = "NNG not installed"


rng = np.random.RandomState(2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_metric():
    with pytest.raises(ValueError) as exc_info:
        NNG(metric="jibberish")
        assert "Unknown" in exc_info


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_dir(n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    with pytest.raises(TypeError) as exc_info:
        nng = NNG(index_dir=1)
        nng.fit(source)
        assert "NNG requires" in exc_info


@pytest.mark.skipif(skip, reason=skip_reason)
def test_right_dir(tmp_path, n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    nng = NNG(index_dir=str(tmp_path))
    nng.fit(source, target)
    assert nng is not None


@pytest.mark.skipif(skip, reason=skip_reason)
def test_none_dir(n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    nng = NNG(index_dir=None)
    nng.fit(source, target)
    assert nng is not None


@pytest.mark.skipif(skip, reason=skip_reason)
def test_self_query(n_samples=20, n_features=5, n_neighbors=5):
    source = rng.rand(n_samples, n_features)
    nng = NNG(index_dir=None, n_candidates=n_neighbors, epsilon=0.00001)
    nng.fit(source, source)
    d, i = nng.kneighbors()
    i2 = nng.kneighbors(return_distance=False)
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_query(n_samples=20, n_features=5, n_neighbors=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    nng = NNG(index_dir=None, n_candidates=n_neighbors, epsilon=0.00001)
    nng.fit(source, target)
    d, i = nng.kneighbors(
        query=source[
            :5,
        ]
    )
    i2 = nng.kneighbors(
        query=source[
            :5,
        ],
        return_distance=False,
    )
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_sqeuclidean(n_samples=20, n_features=5, n_neighbors=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    nng1 = NNG(index_dir=None, n_candidates=n_neighbors, metric="sqeuclidean")
    nng1.fit(source, target)
    d, i = nng1.kneighbors(
        query=source[
            :5,
        ]
    )
    nng2 = NNG(index_dir=None, n_candidates=n_neighbors)
    nng2.fit(source, target)
    i2 = nng2.kneighbors(
        query=source[
            :5,
        ],
        return_distance=False,
    )
    assert_array_equal(i, i2)
