# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from itertools import product

import numpy as np
import pytest
from kiez.alignment.alignment import NeighborhoodAlignment
from kiez.hubness_reduction import hubness_algorithms
from kiez.utils.platform import available_ann_algorithms_on_current_platform
from numpy.testing import assert_array_almost_equal, assert_array_equal

P = (1, 3, 4, np.inf, 2)  # Euclidean last, for tests against approx NN
rng = np.random.RandomState(2)
APPROXIMATE_ALGORITHMS = available_ann_algorithms_on_current_platform()

HUBNESS_ALGORITHMS = hubness_algorithms
MP_PARAMS = tuple({"method": method} for method in ["normal", "empiric"])
LS_PARAMS = tuple({"method": method} for method in ["standard", "nicdm"])
DSL_PARAMS = tuple({"squared": val} for val in [True, False])
HUBNESS_ALGORITHMS_WITH_PARAMS = (
    (None, {}),
    ("csls", {}),
    *product(["mp"], MP_PARAMS),
    *product(["ls"], LS_PARAMS),
    *product(["dsl"], DSL_PARAMS),
)


@pytest.mark.parametrize("hubness_and_params", HUBNESS_ALGORITHMS_WITH_PARAMS)
def test_alignment_source_equals_target(
    hubness_and_params,
    n_samples=20,
    n_features=5,
    n_query_pts=10,
    n_neighbors=5,
):
    hubness, hub_params = hubness_and_params
    source = rng.rand(n_samples, n_features)
    query = rng.rand(n_query_pts, n_features)

    for p in P:
        results = []
        results_nodist = []

        for algo in ["auto", "kd_tree", "ball_tree", "brute"]:
            align = NeighborhoodAlignment(
                n_neighbors=n_neighbors,
                algorithm=algo,
                algorithm_params={"n_candidates": n_neighbors},
                hubness=hubness,
                hubness_params=hub_params,
                p=p,
            )
            if hubness == "dsl" and p != 2:
                with pytest.warns(UserWarning):
                    align.fit(source, source)
            else:
                align.fit(source, source)
            results.append(
                align.kneighbors(
                    source_query_points=query, return_distance=True
                )
            )
            results_nodist.append(
                align.kneighbors(
                    source_query_points=query, return_distance=False
                )
            )
        for i in range(len(results) - 1):
            assert_array_almost_equal(results_nodist[i], results[i][1])
            assert_array_almost_equal(results[i][0], results[i + 1][0])
            assert_array_almost_equal(results[i][1], results[i + 1][1])
    # Test approximate NN against exact NN with Euclidean distances
    assert p == 2, f"Internal: last parameter p={p}, should have been 2"
    for algo in APPROXIMATE_ALGORITHMS:
        align = NeighborhoodAlignment(
            n_neighbors=n_neighbors,
            algorithm=algo,
            algorithm_params={"n_candidates": n_neighbors},
            hubness=hubness,
            hubness_params=hub_params,
            p=p,
        )
        align.fit(source, source)
        results_approx = align.kneighbors(
            source_query_points=query, return_distance=True
        )
        results_approx_nodist = align.kneighbors(
            source_query_points=query, return_distance=False
        )
        assert_array_equal(results_approx_nodist, results_approx[1])
        if algo in ["rptree"]:  # quite imprecise
            assert_array_almost_equal(
                results_approx[0], results[1][0], decimal=0
            )
            for i in range(len(results_approx[1])):
                assert (
                    np.intersect1d(results_approx[1][i], results[1][1][i]).size
                    >= 1
                )
        else:
            # assert_array_almost_equal(
            #     results_approx[0], results[1][0], decimal=6
            # )
            for ra, r in zip(results_approx[1], results[1][1]):
                assert set(ra) == set(r)


@pytest.mark.parametrize("hubness_and_params", HUBNESS_ALGORITHMS_WITH_PARAMS)
def test_alignment(
    hubness_and_params,
    n_samples=20,
    n_features=5,
    n_query_pts=10,
    n_neighbors=5,
):
    hubness, hub_params = hubness_and_params
    source = rng.rand(n_query_pts, n_features)
    target = rng.rand(n_samples, n_features)

    for p in P:
        results = []
        results_nodist = []

        for algo in ["auto", "kd_tree", "ball_tree", "brute"]:
            align = NeighborhoodAlignment(
                n_neighbors=n_neighbors,
                algorithm=algo,
                algorithm_params={"n_candidates": n_neighbors},
                hubness=hubness,
                hubness_params=hub_params,
                p=p,
            )
            if hubness == "dsl" and p != 2:
                with pytest.warns(UserWarning):
                    align.fit(source, target)
            else:
                align.fit(source, target)
            results.append(align.kneighbors(return_distance=True))
            results_nodist.append(align.kneighbors(return_distance=False))
        for i in range(len(results) - 1):
            try:
                assert_array_almost_equal(results_nodist[i], results[i][1])
                assert_array_almost_equal(results[i][0], results[i + 1][0])
                assert_array_almost_equal(results[i][1], results[i + 1][1])
            except AssertionError as error:
                # empiric mp with ball tree can give slightly different results
                # because slight differences in distance provided by ball_tree
                if i != 3 and hubness != "mp" and "empiric" not in hub_params:
                    raise error
    # Test approximate NN against exact NN with Euclidean distances
    assert p == 2, f"Internal: last parameter p={p}, should have been 2"
    for algo in APPROXIMATE_ALGORITHMS:
        align = NeighborhoodAlignment(
            n_neighbors=n_neighbors,
            algorithm=algo,
            algorithm_params={"n_candidates": n_neighbors},
            hubness=hubness,
            hubness_params=hub_params,
            p=p,
        )
        align.fit(source, target)
        results_approx = align.kneighbors(
            source_query_points=source, return_distance=True
        )
        results_approx_nodist = align.kneighbors(
            source_query_points=source, return_distance=False
        )
        assert_array_equal(results_approx_nodist, results_approx[1])
        if algo in ["rptree"]:  # quite imprecise
            assert_array_almost_equal(
                results_approx[0], results[1][0], decimal=0
            )
            for i in range(len(results_approx[1])):
                try:
                    assert (
                        np.intersect1d(
                            results_approx[1][i], results[1][1][i]
                        ).size
                        >= 1
                    ), f"{algo} failed with {hubness}, and {hub_params}"
                except AssertionError as error:
                    if hubness != "mp":
                        raise error
        else:
            # assert_array_almost_equal(
            #     results_approx[0], results[1][0], decimal=6
            # )
            for ra, r in zip(results_approx[1], results[1][1]):
                assert set(ra) == set(r)
