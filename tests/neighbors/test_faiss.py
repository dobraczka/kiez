import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.neighbors import Faiss
from kiez.neighbors.util import available_nn_algorithms

NN_ALGORITHMS = available_nn_algorithms()
if Faiss not in NN_ALGORITHMS:
    skip = True
else:
    skip = False


@pytest.mark.skipif(skip, reason="Faiss not installed")
@pytest.mark.parametrize("single_source", [True, False])
def test_different_instantiations(single_source, source_target):
    rng = np.random.RandomState(2)
    source, target = source_target
    for same_config in [
        (
            {"metric": "l2"},
            {"metric": "l2", "index_key": "Flat"},
        ),
        (
            {"metric": "euclidean"},
            {"metric": "euclidean", "index_key": "Flat"},
        ),
    ]:
        auto = Faiss(**same_config[0])
        manual = Faiss(**same_config[1])
        if single_source:
            auto.fit(source)
            manual.fit(source)
        else:
            auto.fit(source, target)
            manual.fit(source, target)
        auto_res = auto.kneighbors()
        manual_res = manual.kneighbors()
        assert_array_equal(auto_res[0], manual_res[0])
        assert_array_equal(auto_res[1], manual_res[1])

        # check if repr runs without error
        print(auto.__repr__())
        print(manual.__repr__())
