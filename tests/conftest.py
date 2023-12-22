import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def source_target(request):
    rng = np.random.RandomState(42)
    n_samples = 20
    n_samples2 = 50
    n_features = 5
    return rng.rand(n_samples, n_features), rng.rand(n_samples2, n_features)
