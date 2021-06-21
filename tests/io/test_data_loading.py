import numpy as np
import pytest
from kiez.io.data_loading import _seperate_common_embedding
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            (
                np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6], [5, 6, 7]]),
                {0: "s1", 1: "s2"},
                {2: "t1", 3: "t2"},
                {"s1": "t1", "s2": "t2"},
            ),
            (
                np.array([[1, 2, 3], [2, 3, 4]]),
                np.array([[4, 5, 6], [5, 6, 7]]),
                {"s1": 0, "s2": 1},
                {"t1": 0, "t2": 1},
                {0: 0, 1: 1},
            ),
        ),
        (
            (
                np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6], [5, 6, 7]]),
                {0: "s1", 2: "s2"},
                {1: "t1", 3: "t2"},
                {"s1": "t1", "s2": "t2"},
            ),
            (
                np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[2, 3, 4], [5, 6, 7]]),
                {"s1": 0, "s2": 1},
                {"t1": 0, "t2": 1},
                {0: 0, 1: 1},
            ),
        ),
    ],
)
def test_seperate_common_embedding(input, expected):
    emb1, emb2, ids1, ids2, ent_links = _seperate_common_embedding(*input)
    assert_array_equal(expected[0], emb1)
    assert_array_equal(expected[1], emb2)
    assert expected[2] == ids1
    assert expected[3] == ids2
    assert expected[4] == ent_links
