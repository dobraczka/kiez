import pathlib
from unittest import mock

import pytest

from kiez import Kiez
from kiez.hubness_reduction import HubnessReduction, LocalScaling
from kiez.neighbors import NMSLIB, NNAlgorithm, SklearnNN
from kiez.neighbors.util import available_nn_algorithms

NN_ALGORITHMS = available_nn_algorithms()

MP = [("MutualProximity", {"method": method}) for method in ["normal", "empiric"]]
LS = [("LocalScaling", {"method": method}) for method in ["standard", "nicdm"]]
DSL = [("DisSimLocal", {"squared": val}) for val in [True, False]]
HUBNESS_AND_KWARGS = [(None, {}), ("CSLS", {}), *MP, *LS, *DSL]


HERE = pathlib.Path(__file__).parent.resolve()


def test_no_hub(source_target):
    source, target = source_target
    n_cand = 10
    k_inst = Kiez(n_candidates=n_cand)
    k_inst.fit(source, target)
    # check only created target index
    assert not hasattr(k_inst.algorithm, "source_index")
    k_inst.algorithm = SklearnNN()
    assert "f{k_inst}"
    assert (
        Kiez(
            n_candidates=n_cand,
            algorithm="SklearnNN",
            algorithm_kwargs={"metric": "minkowski"},
        ).algorithm.n_candidates
        == n_cand
    )


def assert_different_neighbors(k_inst, n_cand):
    dist, neigh = k_inst.kneighbors()
    assert neigh.shape[1] == n_cand
    assert dist.shape[1] == n_cand

    neigh = k_inst.kneighbors(return_distance=False)
    assert neigh.shape[1] == n_cand

    dist, neigh = k_inst.kneighbors(k=1)
    assert neigh.shape[1] == 1
    assert dist.shape[1] == 1

    dist, neigh = k_inst.kneighbors(k=20)
    assert neigh.shape[1] == n_cand
    assert dist.shape[1] == n_cand


@pytest.mark.parametrize("algo", NN_ALGORITHMS)
def test_algo_resolver(source_target, algo, n_cand=5):
    source, target = source_target
    k_inst = Kiez(algorithm=algo, n_candidates=n_cand)
    k_inst.fit(source, target)
    assert_different_neighbors(k_inst, n_cand)


@pytest.mark.parametrize("hub,hubkwargs", HUBNESS_AND_KWARGS)
def test_hubness_resolver(hub, hubkwargs, source_target, n_cand=5):
    source, target = source_target
    k_inst = Kiez(
        algorithm="SklearnNN",
        n_candidates=n_cand,
        hubness=hub,
        hubness_kwargs=hubkwargs,
    )
    assert f"{k_inst}" is not None
    k_inst.fit(source, target)
    assert_different_neighbors(k_inst, n_cand)
    k_inst.fit(source, None)
    assert_different_neighbors(k_inst, n_cand)
    with pytest.raises(ValueError) as exc_info:
        k_inst = Kiez(
            algorithm="SklearnNN",
            n_candidates=1,
            hubness=hub,
            hubness_kwargs=hubkwargs,
        )
    assert "Cannot" in str(exc_info.value)


def test_n_candidates_wrong():
    with pytest.raises(ValueError) as exc_info:
        Kiez(n_candidates=-1)
    assert "Expected" in str(exc_info.value)


def test_n_candidates_wrong_type():
    with pytest.raises(TypeError) as exc_info:
        Kiez(n_candidates="1")
    assert "does not" in str(exc_info.value)


def test_dis_sim_local_wrong():
    with pytest.raises(ValueError) as exc_info:
        Kiez(algorithm=SklearnNN(p=1), hubness="DisSimLocal")
    assert "only supports" in str(exc_info.value)


def test_dis_sim_local_wrong_metric():
    with pytest.raises(ValueError) as exc_info:
        Kiez(algorithm=SklearnNN(metric="cosine"), hubness="DisSimLocal")
    assert "only supports" in str(exc_info.value)


def test_dis_sim_local_squaring():
    if NMSLIB in NN_ALGORITHMS:
        k_inst = Kiez(algorithm=NMSLIB(metric="sqeuclidean"), hubness="DisSimLocal")
        assert k_inst.hubness.squared


def test_from_config():
    if NMSLIB in NN_ALGORITHMS:
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
