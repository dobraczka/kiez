import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kiez import Kiez
from kiez.hubness_reduction import hubness_reduction_resolver
from kiez.neighbors import SklearnNN, nn_algorithm_resolver

try:
    import torch
except ImportError:
    torch = None

skip = not torch


class TorchNN(SklearnNN):
    def _fit(self, data, is_source: bool):
        return super()._fit(data=data.numpy(), is_source=is_source)

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        res = super()._kneighbors(
            k=k,
            query=query,
            index=index,
            return_distance=return_distance,
            is_self_querying=is_self_querying,
        )
        if return_distance:
            return torch.from_numpy(res[0]).to(dtype=torch.float32), torch.from_numpy(
                res[1]
            )
        return torch.from_numpy(res)


@pytest.mark.all()
@pytest.mark.skipif(skip, reason="Torch not installed")
@pytest.mark.parametrize("algo", ["SklearnNN", "NNG", "NMSLIB", "Annoy"])
def test_forbidden_torch(algo, source_target):
    source, target = source_target
    # calculated tests with torch tensors
    torch_s, torch_t = torch.from_numpy(source), torch.from_numpy(target)
    try:
        nn_algo = nn_algorithm_resolver.make(algo)
    except ImportError:
        return
    kinst = Kiez(algorithm=nn_algo)
    with pytest.raises(ValueError, match="Not implemented"):
        kinst.fit(torch_s, torch_t)


@pytest.mark.torch()
@pytest.mark.skipif(skip, reason="Torch not installed")
@pytest.mark.parametrize(
    ("hubness", "hubness_kwargs"),
    [
        ("NoHubnessReduction", {}),
        ("LocalScaling", {"method": "ls"}),
        ("LocalScaling", {"method": "nicdm"}),
        ("MutualProximity", {"method": "normal"}),
        ("MutualProximity", {"method": "empiric"}),
        ("DisSimLocal", {}),
        ("CSLS", {}),
    ],
)
def test_transform_torch(hubness, hubness_kwargs, source_target):
    source, target = source_target

    # calculated expected with numpy arrays
    nn_inst = SklearnNN(metric="euclidean")
    hubness_kwargs["nn_algo"] = nn_inst
    hub = hubness_reduction_resolver.make(hubness, hubness_kwargs)
    hub.fit(source, target)
    dist, ind = hub.kneighbors()

    # calculated tests with torch tensors
    torch_s, torch_t = torch.from_numpy(source), torch.from_numpy(target)
    nn_inst = TorchNN(metric="euclidean")
    hubness_kwargs["nn_algo"] = nn_inst
    hub = hubness_reduction_resolver.make(hubness, hubness_kwargs)
    hub.fit(torch_s, torch_t)
    tdist, tind = hub.kneighbors()

    assert isinstance(tdist, torch.Tensor)
    assert isinstance(tind, torch.Tensor)
    dist = torch.from_numpy(dist).to(dtype=torch.float32).round(decimals=6)
    tdist = tdist.to(dtype=torch.float32).round(decimals=6)
    ind = torch.from_numpy(ind)

    assert_array_equal(tind, ind)
    tolerance = (
        1e-01
        if "method" in hubness_kwargs and hubness_kwargs["method"] == "normal"
        else 1e-06
    )
    assert_allclose(tdist, dist, rtol=tolerance, atol=tolerance)
