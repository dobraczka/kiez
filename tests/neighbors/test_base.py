import numpy as np
import pytest

from kiez.neighbors import SklearnNN
from kiez.neighbors.util import available_nn_algorithms

NN_ALGORITHMS = available_nn_algorithms()

rng = np.random.RandomState(2)


@pytest.mark.parametrize("algo_cls", NN_ALGORITHMS)
def test_str_rep(algo_cls, source_target):
    source, _ = source_target
    algo = algo_cls()
    assert "is unfitted" in str(algo._describe_source_target_fitted())
    algo.fit(source, source)
    assert "is fitted" in str(algo._describe_source_target_fitted())


def test_check_k_value():
    with pytest.raises(ValueError, match="Expected"):
        SklearnNN()._check_k_value(k=-1, needed_space=2)

    with pytest.raises(TypeError, match="integer"):
        SklearnNN()._check_k_value(k="test", needed_space=2)

    checked = SklearnNN()._check_k_value(k=3, needed_space=2)
    assert checked == 2
