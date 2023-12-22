import numpy as np
import pytest

from kiez.hubness_reduction import LocalScaling, MutualProximity
from kiez.neighbors import SklearnNN

rng = np.random.RandomState(2)


def test_wrong_input_mp():
    with pytest.raises(ValueError, match="not recognized"):
        MutualProximity(nn_algo=SklearnNN(), method="wrong")


def test_wrong_input_ls():
    with pytest.raises(ValueError, match="Invalid"):
        LocalScaling(nn_algo=SklearnNN(), method="wrong")
