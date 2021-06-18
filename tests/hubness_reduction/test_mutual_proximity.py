import numpy as np
import pytest
from kiez import Kiez
from kiez.hubness_reduction import MutualProximity
from numpy.testing import assert_array_equal

rng = np.random.RandomState(2)


def test_wrong_input():
    with pytest.raises(ValueError) as exc_info:
        MutualProximity(method="wrong")
    assert "not recognized" in str(exc_info.value)


def test_sqeuclidean(n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    k_inst = Kiez(hubness=MutualProximity())
    k_inst.fit(source, target)
    ndist, nind = k_inst.kneighbors(k=1)
    out_dist, out_nind = k_inst.hubness.transform(ndist, nind, None)
    assert_array_equal(ndist, out_dist)
    assert_array_equal(nind, out_nind)
