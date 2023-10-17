import pathlib
from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors

from kiez import Kiez
from kiez.hubness_reduction import (
    DisSimLocal,
    HubnessReduction,
    LocalScaling,
    NoHubnessReduction,
)
from kiez.neighbors import NMSLIB, NNAlgorithm, SklearnNN
from kiez.neighbors.util import available_ann_algorithms

APPROXIMATE_ALGORITHMS = available_ann_algorithms()

HERE = pathlib.Path(__file__).parent.resolve()
rng = np.random.RandomState(2)


class CustomHubness(HubnessReduction):
    """Test class to make sure user created classes work"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        pass  # pragma: no cover

    def __repr__(self):
        return "NoHubnessReduction"

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query,
        assume_sorted=True,
        return_distance=True,
        *args,
        **kwargs,
    ):
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


class CustomAlgorithm(NNAlgorithm):
    """Test class to make sure user created classes work"""

    valid_metrics = ["minkowski"]

    def __init__(
        self,
        n_candidates=5,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs)
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params

    def _fit(self, data, is_source: bool):
        nn = NearestNeighbors(
            n_neighbors=self.n_candidates,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        nn.fit(data)
        return nn

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        if is_self_querying:
            return index.kneighbors(
                X=None, n_neighbors=k, return_distance=return_distance
            )
        return index.kneighbors(X=query, n_neighbors=k, return_distance=return_distance)


def test_hubness_resolver(n_samples=20, n_features=5):
    source = rng.rand(n_samples, n_features)
    target = rng.rand(n_samples, n_features)
    res = []
    for algo in [
        SklearnNN(),
        SklearnNN,
        "SklearnNN",
        CustomAlgorithm,
        CustomAlgorithm(),
    ]:
        for hub in [
            NoHubnessReduction(),
            NoHubnessReduction,
            None,
            "NoHubnessReduction",
            CustomHubness,
            CustomHubness(),
        ]:
            k_inst = Kiez(algorithm=algo, hubness=hub)
            k_inst.fit(source, target)
            res.append(k_inst.kneighbors(source, k=1))
    for i in range(len(res) - 1):
        assert_array_equal(res[i][0], res[i + 1][0])
        assert_array_equal(res[i][1], res[i + 1][1])


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
    if NMSLIB in APPROXIMATE_ALGORITHMS:
        k_inst = Kiez(algorithm=NMSLIB(metric="sqeuclidean"), hubness=DisSimLocal())
        assert k_inst.hubness.squared


def test_from_config():
    if NMSLIB in APPROXIMATE_ALGORITHMS:
        path = HERE.joinpath("example_conf.json")
        kiez = Kiez.from_path(path)
        assert kiez.hubness is not None
        assert isinstance(kiez.hubness, HubnessReduction)
        assert isinstance(kiez.hubness, LocalScaling), f"wrong hubness: {kiez.hubness}"
        assert kiez.algorithm is not None
        assert isinstance(kiez.algorithm, NNAlgorithm)
        assert isinstance(kiez.algorithm, NMSLIB), f"wrong algorithm: {kiez.algorithm}"


def mock_make(name, algorithm_kwargs):
    if name == "Faiss":
        raise ImportError
    else:
        return SklearnNN()


@mock.patch("kiez.kiez.nn_algorithm_resolver.make", mock_make)
def test_no_faiss():
    kiez = Kiez()
    assert isinstance(kiez.algorithm, SklearnNN)
