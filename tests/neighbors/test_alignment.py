import numpy as np
import pytest
from kiez import Kiez
from kiez.hubness_reduction import CSLS, DisSimLocal, LocalScaling, MutualProximity
from kiez.neighbors import NMSLIB, NNG, Annoy, Faiss, SklearnNN
from kiez.utils.platform import available_ann_algorithms_on_current_platform
from numpy.testing import assert_array_almost_equal, assert_array_equal

P = (1, 3, 4, np.inf, 2)  # Euclidean last, for tests against approx NN
rng = np.random.RandomState(2)
APPROXIMATE_ALGORITHMS = available_ann_algorithms_on_current_platform()

MP = [MutualProximity(method=method) for method in ["normal", "empiric"]]
LS = [LocalScaling(method=method) for method in ["standard", "nicdm"]]
DSL = [DisSimLocal(squared=val) for val in [True, False]]
HUBNESS = [None, CSLS(), *MP, *LS, *DSL]
APPROXIMATE_ALGORITHMS = [NMSLIB, NNG, Annoy, Faiss]


@pytest.mark.parametrize("hubness", HUBNESS)
def test_alignment_source_equals_target(
    hubness,
    n_samples=20,
    n_features=5,
    n_query_pts=10,
    n_neighbors=5,
):
    source = rng.rand(n_samples, n_features)
    query = rng.rand(n_query_pts, n_features)
    exactalgos = [
        SklearnNN(n_candidates=n_neighbors, algorithm=algo)
        for algo in ["auto", "kd_tree", "ball_tree", "brute"]
    ]
    exactalgos.append(
        Faiss(n_candidates=n_neighbors, metric="euclidean", index_key="Flat")
    )

    for p in P:
        results = []
        results_nodist = []

        for algo in exactalgos:
            if hubness == "dsl" and p != 2:
                with pytest.raises(ValueError):
                    align = Kiez(
                        n_neighbors=n_neighbors, algorithm=algo, hubness=hubness
                    )
                    continue
            align = Kiez(n_neighbors=n_neighbors, algorithm=algo, hubness=hubness)
            align.fit(source)
            results.append(
                align.kneighbors(source_query_points=query, return_distance=True)
            )
            results_nodist.append(
                align.kneighbors(source_query_points=query, return_distance=False)
            )
        for i in range(len(results) - 1):
            assert_array_almost_equal(results_nodist[i], results[i][1])
            assert_array_almost_equal(results[i][0], results[i + 1][0], decimal=3)
            assert_array_almost_equal(results[i][1], results[i + 1][1])
    # Test approximate NN against exact NN with Euclidean distances
    assert p == 2, f"Internal: last parameter p={p}, should have been 2"

    ann_algos = [
        algo_cls(n_candidates=n_neighbors, metric="euclidean")
        for algo_cls in APPROXIMATE_ALGORITHMS
    ]
    for algo in ann_algos:
        align = Kiez(
            n_neighbors=n_neighbors,
            algorithm=algo,
            hubness=hubness,
        )
        align.fit(source)
        results_approx = align.kneighbors(
            source_query_points=query, return_distance=True
        )
        results_approx_nodist = align.kneighbors(
            source_query_points=query, return_distance=False
        )
        assert_array_equal(results_approx_nodist, results_approx[1])
        if isinstance(algo, Annoy):  # quite imprecise
            assert_array_almost_equal(results_approx[0], results[1][0], decimal=0)
            for i in range(len(results_approx[1])):
                assert np.intersect1d(results_approx[1][i], results[1][1][i]).size >= 1
        else:
            assert_array_almost_equal(results_approx[0], results[1][0], decimal=3)
            for ra, r in zip(results_approx[1], results[1][1]):
                assert set(ra) == set(r)


@pytest.mark.parametrize("hubness", HUBNESS)
def test_alignment(
    hubness,
    n_samples=20,
    n_features=5,
    n_query_pts=10,
    n_neighbors=5,
):
    source = rng.rand(n_query_pts, n_features)
    target = rng.rand(n_samples, n_features)

    exactalgos = [
        SklearnNN(n_candidates=n_neighbors, algorithm=algo)
        for algo in ["auto", "kd_tree", "ball_tree", "brute"]
    ]
    exactalgos.append(
        Faiss(n_candidates=n_neighbors, metric="euclidean", index_key="Flat")
    )

    for p in P:
        results = []
        results_nodist = []

        for algo in exactalgos:
            if hubness == "dsl" and p != 2:
                with pytest.raises(ValueError):
                    align = Kiez(
                        n_neighbors=n_neighbors, algorithm=algo, hubness=hubness
                    )
                    continue
            align = Kiez(n_neighbors=n_neighbors, algorithm=algo, hubness=hubness)
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
                if not (
                    isinstance(hubness, MutualProximity) and hubness.method == "empiric"
                ):
                    raise error
    # Test approximate NN against exact NN with Euclidean distances
    assert p == 2, f"Internal: last parameter p={p}, should have been 2"
    ann_algos = [
        algo_cls(n_candidates=n_neighbors, metric="euclidean")
        for algo_cls in APPROXIMATE_ALGORITHMS
    ]
    for algo in ann_algos:
        align = Kiez(
            n_neighbors=n_neighbors,
            algorithm=algo,
            hubness=hubness,
        )
        align.fit(source, target)
        results_approx = align.kneighbors(
            source_query_points=source, return_distance=True
        )
        results_approx_nodist = align.kneighbors(
            source_query_points=source, return_distance=False
        )
        assert_array_equal(results_approx_nodist, results_approx[1])
        if isinstance(algo, Annoy):  # quite imprecise
            assert_array_almost_equal(results_approx[0], results[1][0], decimal=0)
            for i in range(len(results_approx[1])):
                try:
                    assert (
                        np.intersect1d(results_approx[1][i], results[1][1][i]).size >= 1
                    ), f"{algo} failed with {hubness}"
                except AssertionError as error:
                    # empiric mp with ball tree can give slightly different results
                    # because slight differences in distance provided by ball_tree
                    if not (
                        isinstance(hubness, MutualProximity)
                        and hubness.method == "empiric"
                    ):
                        raise error
        else:
            for ra, r in zip(results_approx[1], results[1][1]):
                assert set(ra) == set(r), f"{algo} failed with {hubness}"
