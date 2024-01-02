#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness https://github.com/VarIr/scikit-hubness/

import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kiez.analysis import hubness_score
from kiez.analysis.estimation import _calc_atkinson_index, _calc_gini_index

PRE_CALC_NEIGHBORS = np.load("tests/nn_ind.npy")


# fmt: off
K_OCC = np.array([3, 0, 5, 3, 0, 5, 4, 1, 0, 1, 1, 0, 0, 2, 0, 1, 0,
                  2, 2, 1, 0, 2, 0, 5, 2, 1, 0, 1, 0, 0, 4, 2, 3, 6,
                  1, 0, 3, 0, 0, 0, 2, 2, 3, 4, 3, 3, 2, 1, 0, 0, 1,
                  5, 2, 3, 0, 10, 0, 1, 0, 3, 1, 3, 5, 1, 1, 2, 6,
                  1, 3, 3, 3, 2, 2, 2, 0, 5, 2, 1, 1, 4, 0, 2, 2, 8,
                  1, 0, 7, 1, 2, 0, 0, 2, 0, 0, 3, 3, 3, 2, 9, 1, ])
# fmt: on

rng = np.random.RandomState(2)


@pytest.fixture()
def get_expected():
    expected = {}
    for k in [2, 5, 10, 50]:
        with open(f"tests/expected_k{k}_hub_scores.pkl", "rb") as handle:
            expected[k] = pickle.load(handle)
    return expected


@pytest.mark.parametrize("verbose", [-1, 0, 1, 2, 3, None])
def test_hubness(verbose):
    hubness_true = 0.9128709291752769
    neighbors = np.array([[0, 2], [1, 0], [2, 0], [3, 1], [4, 0]])
    score = hubness_score(neighbors, 5)
    np.testing.assert_almost_equal(score["k_skewness"], hubness_true, decimal=10)


def test_limiting_factor():
    # Different implementations of Gini index calculation should give the same result.
    gini_space = _calc_gini_index(K_OCC, limiting="space")
    gini_time = _calc_gini_index(K_OCC, limiting="time")
    gini_naive = _calc_gini_index(K_OCC, limiting=None)

    assert gini_space == gini_time == gini_naive


@pytest.mark.parametrize("k", [2, 5, 10, 50])
def test_all(k, get_expected):
    measures = hubness_score(
        PRE_CALC_NEIGHBORS,
        1000,
        k=k,
        return_value="all",
        store_k_occurrence=True,
    )
    for score_key, v in get_expected[k].items():
        if isinstance(v, np.ndarray):
            assert_array_equal(v, measures[score_key])
        else:
            assert pytest.approx(v) == measures[score_key]


def test_atkinson():
    atkinson_0999 = _calc_atkinson_index(K_OCC, eps=0.999)
    atkinson_1000 = _calc_atkinson_index(K_OCC, eps=1)
    np.testing.assert_almost_equal(atkinson_0999, atkinson_1000, decimal=3)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_hubness_return_values_are_self_consistent(k):
    # Test that the three returned values fit together
    n_samples = 1000
    scores = hubness_score(
        PRE_CALC_NEIGHBORS,
        1000,
        k=k,
        store_k_occurrence=True,
    )
    skew = scores["k_skewness"]
    occ = scores["k_occurrence"]
    occ_true = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        occ_true[i] = (PRE_CALC_NEIGHBORS[:, :k] == i).sum()
    np.testing.assert_array_equal(occ, occ_true)
    # Calculate skewness (different method than in module)
    x0 = occ - occ.mean()
    s2 = (x0**2).mean()
    m3 = (x0**3).mean()
    skew_true = m3 / (s2**1.5)
    np.testing.assert_equal(skew, skew_true)


def test_negative_indices():
    neighbors = np.array([[1, 2, 3], [-1, 4, 5]])
    score = hubness_score(neighbors, 5)
    assert score is not None


def test_k_too_large():
    neighbors = np.array([[1, 2, 3], [-1, 4, 5]])
    with pytest.warns(None, match="k > nn_ind.shape[1], k will be set to 3"):
        score = hubness_score(neighbors, 5, k=10)
        assert score is not None


def test_wrong_neighbors():
    with pytest.raises(ValueError, match="no negative"):
        hubness_score(np.array([[np.inf], [0]]), 1)
