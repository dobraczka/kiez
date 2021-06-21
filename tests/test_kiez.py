import numpy as np
import pytest
from kiez import Kiez
from kiez.hubness_reduction import DisSimLocal
from kiez.neighbors import HNSW, SklearnNN

rng = np.random.RandomState(2)


def test_wrong_kcandidates(n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    k_inst = Kiez()
    k_inst.fit(source, target)
    nn_ind = k_inst._kcandidates(source, k=1, return_distance=False)
    assert nn_ind.shape == (20, 5)


def test_non_default_kneighbors(n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    k_inst = Kiez()
    k_inst.fit(source, target)
    nn_ind = k_inst.kneighbors(source, k=1, return_distance=False)
    assert nn_ind.shape == (20, 1)


def test_n_neighbors_wrong():
    with pytest.raises(ValueError) as exc_info:
        Kiez(n_neighbors=-1)
    assert "Expected" in str(exc_info.value)


def test_n_neighbors_wrong_type():
    with pytest.raises(TypeError) as exc_info:
        Kiez(n_neighbors="1")
    assert "does not" in str(exc_info.value)


def test_dis_sim_local_wrong():
    with pytest.raises(ValueError) as exc_info:
        Kiez(algorithm=SklearnNN(p=1), hubness=DisSimLocal())
    assert "only supports" in str(exc_info.value)


def test_dis_sim_local_wrong_metric():
    with pytest.raises(ValueError) as exc_info:
        Kiez(algorithm=SklearnNN(metric="cosine"), hubness=DisSimLocal())
    assert "only supports" in str(exc_info.value)


def test_dis_sim_local_squaring():
    k_inst = Kiez(algorithm=HNSW(metric="sqeuclidean"), hubness=DisSimLocal())
    assert k_inst.hubness.squared
