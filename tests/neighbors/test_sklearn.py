import numpy as np
from kiez.neighbors import SklearnNN
from numpy.testing import assert_array_equal

rng = np.random.RandomState(2)


def test_self_query(n_samples=20, n_features=5, n_neighbors=5):
    source = rng.rand(n_samples, n_features)
    sklearnnn = SklearnNN()
    sklearnnn.fit(source, source)
    d, i = sklearnnn.kneighbors()
    i2 = sklearnnn.kneighbors(return_distance=False)
    assert_array_equal(i, i2)
