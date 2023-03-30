import numpy as np
import pytest

from kiez.neighbors import NMSLIB, NNG, Annoy, Faiss, SklearnNN

rng = np.random.RandomState(2)


@pytest.mark.parametrize("algo_cls", [NMSLIB, SklearnNN, NNG, Annoy, Faiss])
def test_str_rep(algo_cls, n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    algo = algo_cls()
    assert "is unfitted" in str(algo._describe_source_target_fitted())
    algo.fit(source, source)
    assert "is fitted" in str(algo._describe_source_target_fitted())


def test_check_k_value():
    with pytest.raises(ValueError) as exc_info:
        SklearnNN()._check_k_value(k=-1, needed_space=2)
    assert "Expected" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        SklearnNN()._check_k_value(k="test", needed_space=2)
    assert "integer" in str(exc_info.value)

    checked = SklearnNN()._check_k_value(k=3, needed_space=2)
    assert checked == 2
