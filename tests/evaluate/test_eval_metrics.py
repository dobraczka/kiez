import pytest
from kiez.evaluate import hits


@pytest.mark.parametrize(
    "nn_ind, gold, k, expected",
    [
        (
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
            {0: 2, 1: 4, 2: 3, 3: 4},
            [1, 2, 3],
            {1: 0.5, 2: 0.75, 3: 1.0},
        ),
        (
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
            {0: 5, 1: 6, 2: 7, 3: 8},
            None,
            {1: 0.0, 5: 0.0, 10: 0.0},
        ),
        (
            {0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5], 3: [4, 5, 6]},
            {0: 2, 1: 4, 2: 3, 3: 4},
            [1, 2, 3],
            {1: 0.5, 2: 0.75, 3: 1.0},
        ),
        (
            {0: [1, 2, 3], 1: [2, 3, 4], 2: [3, 4, 5], 3: [4, 5, 6]},
            {0: 5, 1: 6, 2: 7, 3: 8},
            None,
            {1: 0.0, 5: 0.0, 10: 0.0},
        ),
        (
            {
                "0": ["1", "2", "3"],
                "1": ["2", "3", "4"],
                "2": ["3", "4", "5"],
                "3": ["4", "5", "6"],
            },
            {"0": "2", "1": "4", "2": "3", "3": "4"},
            [1, 2, 3],
            {1: 0.5, 2: 0.75, 3: 1.0},
        ),
    ],
)
def test_hits(nn_ind, gold, k, expected):
    res = hits(nn_ind, gold, k)
    assert res == expected
