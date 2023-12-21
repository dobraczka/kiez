import numpy as np
from numpy.testing import assert_array_equal

from kiez.neighbors import SklearnNN

rng = np.random.RandomState(2)


def test_self_query(source_target, n_neighbors=5):
    source, _ = source_target
    sklearnnn = SklearnNN()
    sklearnnn.fit(source, source)
    d, i = sklearnnn.kneighbors()
    i2 = sklearnnn.kneighbors(return_distance=False)
    assert_array_equal(i, i2)
