import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.neighbors import Annoy
from kiez.neighbors.util import available_nn_algorithms

NN_ALGORITHMS = available_nn_algorithms()
if Annoy not in NN_ALGORITHMS:
    skip = True
else:
    skip = False
skip_reason = "Annoy not installed"


rng = np.random.RandomState(2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_wrong_metric():
    with pytest.raises(ValueError) as exc_info:
        Annoy(metric="jibberish")
    assert "Unknown" in str(exc_info.value)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_minkowski_metric():
    annoy = Annoy(metric="minkowski")
    assert annoy.metric == "euclidean"


@pytest.mark.skipif(skip, reason=skip_reason)
def test_self_query(source_target, n_neighbors=5):
    source, _ = source_target
    annoy = Annoy(n_candidates=n_neighbors)
    annoy.fit(source, source)
    d, i = annoy.kneighbors()
    i2 = annoy.kneighbors(return_distance=False)
    assert_array_equal(i, i2)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_query(tmp_path, source_target, n_neighbors=5):
    source, target = source_target
    annoy = Annoy(n_candidates=n_neighbors, metric="euclidean")
    annoy.fit(source, target)
    d, i = annoy.kneighbors()
    annoy2 = Annoy(n_candidates=n_neighbors, metric="minkowski")
    annoy2.fit(source, target)
    i2 = annoy2.kneighbors(return_distance=False)
    assert_array_equal(i, i2)
    annoy3 = Annoy(n_candidates=n_neighbors, mmap_dir=str(tmp_path))
    annoy3.fit(source, target)
    i3 = annoy3.kneighbors(return_distance=False)
    assert_array_equal(i, i3)
    annoy4 = Annoy(n_candidates=n_neighbors, mmap_dir=None)
    annoy4.fit(source, target)
    i4 = annoy4.kneighbors(return_distance=False)
    assert_array_equal(i, i4)


@pytest.mark.skipif(skip, reason=skip_reason)
def test_inner_kneighbors(tmp_path, source_target, n_neighbors=5):
    source, target = source_target
    annoy = Annoy(n_candidates=n_neighbors)
    annoy.fit(source, target)
    with pytest.raises(AssertionError) as exc_info:
        annoy._kneighbors_part(
            k=n_neighbors,
            query=source,
            index="test",
            return_distance=True,
            is_self_querying=False,
        )
    assert "Internal" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        annoy._kneighbors_part(
            k=n_neighbors,
            query=source,
            index=("test", ""),
            return_distance=True,
            is_self_querying=False,
        )
    assert "Internal" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        annoy._kneighbors_part(
            k=n_neighbors,
            query=source,
            index=("test", 2, ""),
            return_distance=True,
            is_self_querying=False,
        )
    assert "Internal" in str(exc_info.value)
