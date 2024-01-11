import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kiez import Kiez
from kiez.neighbors import Faiss
from kiez.neighbors.util import available_nn_algorithms

try:
    import torch
except ImportError:
    torch = None

NN_ALGORITHMS = available_nn_algorithms()
skip = Faiss not in NN_ALGORITHMS
skip2 = skip or not torch or not torch.cuda.is_available()


@pytest.mark.faiss()
@pytest.mark.skipif(skip, reason="Faiss not installed")
@pytest.mark.parametrize("single_source", [True, False])
def test_different_instantiations(single_source, source_target):
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


@pytest.mark.faiss()
@pytest.mark.skipif(skip, reason="Faiss not installed")
@pytest.mark.parametrize("metric", Faiss.valid_metrics)
def test_metrics_smoketest(metric, source_target):
    k = 3
    source, target = source_target
    nn_inst_np = Faiss(metric=metric, index_key="Flat")
    kiez_inst = Kiez(
        n_candidates=5,
        algorithm=nn_inst_np,
    )
    kiez_inst.fit(source, target)
    np_dist, np_ind = kiez_inst.kneighbors(k)


@pytest.mark.faiss()
@pytest.mark.skipif(skip2, reason="Faiss or PyTorch not installed or no GPU")
@pytest.mark.parametrize(
    ("hubness", "hubness_kwargs"),
    [
        ("NoHubnessReduction", {}),
        ("LocalScaling", {"method": "ls"}),
        ("LocalScaling", {"method": "nicdm"}),
        ("MutualProximity", {"method": "normal"}),
        ("DisSimLocal", {}),
        ("CSLS", {}),
    ],
)
@pytest.mark.parametrize("metric", Faiss.valid_metrics)
def test_torch_gpu(hubness, hubness_kwargs, metric, source_target):
    if metric != "euclidean" and hubness == "DisSimLocal":
        return
    k = 3
    source, target = source_target
    nn_inst_np = Faiss(metric=metric, index_key="Flat")
    kiez_inst = Kiez(
        n_candidates=5,
        algorithm=nn_inst_np,
        hubness=hubness,
        hubness_kwargs=hubness_kwargs,
    )
    kiez_inst.fit(source, target)
    np_dist, np_ind = kiez_inst.kneighbors(k)

    source = torch.tensor(source).to(torch.float32).cuda()
    target = torch.tensor(target).to(torch.float32).cuda()
    device_type = source.device.type
    nn_inst = Faiss(metric=metric, index_key="Flat", use_gpu=True)
    kiez_inst = Kiez(
        n_candidates=5,
        algorithm=nn_inst,
        hubness=hubness,
        hubness_kwargs=hubness_kwargs,
    )
    kiez_inst.fit(source, target)
    dist, ind = kiez_inst.kneighbors(k)
    assert type(dist) == torch.Tensor
    assert type(ind) == torch.Tensor
    assert dist.device.type == device_type
    assert ind.device.type == device_type
    assert dist.shape == (len(source), k)
    assert ind.shape == (len(source), k)

    tolerance = 1.0e-6 if hubness != "MutualProximity" else 1.0e-1
    assert_allclose(np_dist, dist.cpu().numpy(), rtol=tolerance, atol=tolerance)
    assert_array_equal(np_ind, ind.cpu().numpy())


@pytest.mark.faiss()
@pytest.mark.skipif(skip2, reason="Faiss or PyTorch not installed or no GPU")
@pytest.mark.parametrize(
    ("hubness", "hubness_kwargs"),
    [
        ("MutualProximity", {"method": "empiric"}),
    ],
)
def test_torch_gpu_use_np(hubness, hubness_kwargs, source_target):
    k = 3
    source, target = source_target
    nn_inst_np = Faiss(metric="l2", index_key="Flat")
    kiez_inst = Kiez(
        n_candidates=5,
        algorithm=nn_inst_np,
        hubness=hubness,
        hubness_kwargs=hubness_kwargs,
    )
    kiez_inst.fit(source, target)
    np_dist, np_ind = kiez_inst.kneighbors(k)

    source = torch.tensor(source).to(torch.float32).cuda()
    target = torch.tensor(target).to(torch.float32).cuda()
    nn_inst = Faiss(metric="l2", index_key="Flat", use_gpu=True)
    kiez_inst = Kiez(
        n_candidates=5,
        algorithm=nn_inst,
        hubness=hubness,
        hubness_kwargs=hubness_kwargs,
    )
    kiez_inst.fit(source, target)
    dist, ind = kiez_inst.kneighbors(k)
    assert dist.shape == (len(source), k)
    assert ind.shape == (len(source), k)

    tolerance = 1.0e-6
    assert_allclose(np_dist, dist, rtol=tolerance, atol=tolerance)
    assert_array_equal(np_ind, ind)
