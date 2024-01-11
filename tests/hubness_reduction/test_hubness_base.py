import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.hubness_reduction.base import HubnessReduction

try:
    import torch
except ImportError:
    torch = None

skip = not torch


@pytest.mark.skipif(skip, reason="PyTorch not installed")
def test_sorting():
    rng = np.random.default_rng(seed=42)
    size = (100, 10)
    dist = rng.random(size)
    ind = rng.integers(low=0, high=200, size=size)
    np_dist, np_ind = HubnessReduction._sort(dist, ind, size[1])
    assert isinstance(np_dist, np.ndarray)
    assert isinstance(np_ind, np.ndarray)

    tdist = torch.tensor(dist)
    tind = torch.tensor(ind)
    torch_dist, torch_ind = HubnessReduction._sort(tdist, tind, size[1])
    assert isinstance(torch_dist, torch.Tensor)
    assert isinstance(torch_ind, torch.Tensor)

    assert_array_equal(torch_dist.numpy(), np_dist)
    assert_array_equal(torch_ind.numpy(), np_ind)
