import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kiez.hubness_reduction import hubness_reduction_resolver
from kiez.neighbors import SklearnNN

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
            return torch.from_numpy(res[0]), torch.from_numpy(res[1])
        return torch.from_numpy(res)


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

    if "method" in hubness_kwargs and hubness_kwargs["method"] == "empiric":
        assert isinstance(tdist, np.ndarray)
        assert isinstance(tind, np.ndarray)
    else:
        assert isinstance(tdist, torch.Tensor)
        assert isinstance(tind, torch.Tensor)
        tdist = tdist.numpy()
        tind = tind.numpy()

    tolerance = 1.0e-6 if hubness != "MutualProximity" else 1.0e-1
    assert_allclose(tdist, dist, rtol=tolerance, atol=tolerance)
    assert_array_equal(tind, ind)
