import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez import Kiez
from kiez.hubness_reduction import LocalScaling, MutualProximity
from kiez.neighbors import SklearnNN

rng = np.random.RandomState(2)


def test_wrong_input_mp():
    with pytest.raises(ValueError) as exc_info:
        MutualProximity(nn_algo=SklearnNN(), method="wrong")
    assert "not recognized" in str(exc_info.value)


def test_wrong_input_ls():
    with pytest.raises(ValueError) as exc_info:
        LocalScaling(nn_algo=SklearnNN(), method="wrong")
    assert "Invalid" in str(exc_info.value)
