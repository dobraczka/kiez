# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# loosely adapted from skhubness: https://github.com/VarIr/scikit-hubness

import warnings
from functools import partial

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from kiez.hubness_reduction import (
    CSLS,
    DisSimLocal,
    LocalScaling,
    MutualProximity,
    hubness_algorithms,
    hubness_algorithms_long,
)
from kiez.hubness_reduction.base import NoHubnessReduction
from sklearn.metrics.pairwise import (
    PAIRWISE_DISTANCE_FUNCTIONS,
    pairwise_distances_chunked,
)
from sklearn.neighbors import BallTree, KDTree
from sklearn.utils import gen_even_slices

from .approximate_neighbors import UnavailableANN
from .hnsw import HNSW
from .nng import NNG
from .random_projection_trees import RandomProjectionTree

VALID_METRICS = dict(
    nng=NNG.valid_metrics if not issubclass(NNG, UnavailableANN) else [],
    hnsw=HNSW.valid_metrics,
    rptree=RandomProjectionTree.valid_metrics,
    ball_tree=BallTree.valid_metrics,
    kd_tree=KDTree.valid_metrics,
    # The following list comes from the
    # sklearn.metrics.pairwise doc string
    brute=(
        list(PAIRWISE_DISTANCE_FUNCTIONS.keys())
        + [
            "braycurtis",
            "canberra",
            "chebyshev",
            "correlation",
            "cosine",
            "dice",
            "hamming",
            "jaccard",
            "kulsinski",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
            "wminkowski",
        ]
    ),
)

EXACT_ALG = (
    "brute",
    "kd_tree",
    "ball_tree",
)

ANN_ALG = (
    "hnsw",
    "rptree",
    "nng",
)

ALLOWED_FIT_METHODS = [*EXACT_ALG, *ANN_ALG]


class NeighborhoodAlignment:
    def __init__(
        self,
        n_neighbors=None,
        algorithm="auto",
        algorithm_params: dict = None,
        hubness: str = None,
        hubness_params: dict = None,
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
        verbose: int = 0,
    ):
        if algorithm_params is None:
            n_candidates = 1 if hubness is None else 100
            algorithm_params = {"n_candidates": n_candidates, "metric": metric}
        if n_jobs is not None and "n_jobs" not in algorithm_params:
            algorithm_params["n_jobs"] = n_jobs
        if "verbose" not in algorithm_params:
            algorithm_params["verbose"] = verbose
        hubness_params = hubness_params if hubness_params is not None else {}
        if "verbose" not in hubness_params:
            hubness_params["verbose"] = verbose

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        self.hubness = hubness
        self.hubness_params = hubness_params
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _initialize(self, source, target):
        if self._fit_method == "ball_tree":
            self._tree_source = BallTree(
                source,
                self.leaf_size,
                metric=self.effective_metric_,
                **self.effective_metric_params_,
            )
            self._tree_target = BallTree(
                target,
                self.leaf_size,
                metric=self.effective_metric_,
                **self.effective_metric_params_,
            )
            self._index_source = None
            self._index_target = None
        elif self._fit_method == "kd_tree":
            self._tree_source = KDTree(
                source,
                self.leaf_size,
                metric=self.effective_metric_,
                **self.effective_metric_params_,
            )
            self._tree_target = KDTree(
                target,
                self.leaf_size,
                metric=self.effective_metric_,
                **self.effective_metric_params_,
            )
            self._index_source = None
            self._index_target = None
        elif self._fit_method == "brute":
            self._tree_source = None
            self._tree_target = None
            self._index_source = None
            self._index_target = None
        elif self._fit_method == "hnsw":
            self._tree_source = None
            self._tree_target = None
            self._index_source = HNSW(**self.algorithm_params)
            self._index_source.fit(source)
            self._index_target = HNSW(**self.algorithm_params)
            self._index_target.fit(target)
        elif self._fit_method == "nng":
            self._tree_source = None
            self._tree_target = None
            self._index_source = NNG(**self.algorithm_params)
            self._index_source.fit(source)
            self._index_target = NNG(**self.algorithm_params)
            self._index_target.fit(target)
        elif self._fit_method == "rptree":
            self._tree_source = None  # it's not a sklearn tree
            self._tree_target = None
            self._index_source = RandomProjectionTree(**self.algorithm_params)
            self._index_source.fit(source)
            self._index_target = RandomProjectionTree(**self.algorithm_params)
            self._index_target.fit(target)

    def _kneighbors_reduce_func(
        self, dist, start, n_neighbors, return_distance
    ):
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[
            sample_range, np.argsort(dist[sample_range, neigh_ind])
        ]
        if return_distance:
            if self.effective_metric_ == "euclidean":
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = dist[sample_range, neigh_ind], neigh_ind
        else:
            result = neigh_ind
        return result

    def kcandidates(
        self, query_points, s_to_t=True, n_neighbors=None, return_distance=True
    ) -> np.ndarray or (np.ndarray, np.ndarray):
        if n_neighbors is None:
            try:
                n_neighbors = self.algorithm_params["n_candidates"]
            except KeyError:
                n_neighbors = 1 if self.hubness is None else 100
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" % type(n_neighbors)
                )

        # The number of candidates must not be less than the number of neighbors used downstream
        if self.n_neighbors is not None and n_neighbors < self.n_neighbors:
            n_neighbors = self.n_neighbors

        if s_to_t:
            fit = self._fit_target
            index = self._index_target
            tree = self._tree_target
        else:
            fit = self._fit_source
            index = self._index_source
            tree = self._tree_source

        try:
            train_size = fit.shape[0]
        except AttributeError:
            train_size = index.n_samples_fit_
        if n_neighbors > train_size:
            warnings.warn(
                "n_candidates > n_samples. Setting n_candidates = n_samples."
            )
            n_neighbors = train_size
        n_samples, _ = query_points.shape

        n_jobs = effective_n_jobs(self.n_jobs)

        if self._fit_method not in ALLOWED_FIT_METHODS:
            raise ValueError(
                f"internal: _fit_method not recognized: {self._fit_method}."
            )
        elif self._fit_method == "brute":

            # TODO handle sparse matrices here
            reduce_func = partial(
                self._kneighbors_reduce_func,
                n_neighbors=n_neighbors,
                return_distance=return_distance,
            )

            # for efficiency, use squared euclidean distances
            kwds = (
                {"squared": True}
                if self.effective_metric_ == "euclidean"
                else self.effective_metric_params_
            )

            result = pairwise_distances_chunked(
                query_points,
                fit,
                reduce_func=reduce_func,
                metric=self.effective_metric_,
                n_jobs=n_jobs,
                **kwds,
            )

        elif self._fit_method in ["ball_tree", "kd_tree"]:
            delayed_query = delayed(tree.query)
            parallel_kwargs = {"prefer": "threads"}
            result = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(query_points[s], n_neighbors, return_distance)
                for s in gen_even_slices(query_points.shape[0], n_jobs)
            )
        elif self._fit_method in [
            "rptree",
            "nng",
        ]:
            # assume joblib>=0.12
            delayed_query = delayed(index.kneighbors)
            parallel_kwargs = {"prefer": "threads"}
            result = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(
                    query_points[s],
                    n_candidates=n_neighbors,
                    return_distance=True,
                )
                for s in gen_even_slices(query_points.shape[0], n_jobs)
            )
        elif self._fit_method in ["hnsw"]:
            # XXX nmslib supports multiple threads natively, so no joblib used here
            # Must pack results into list to match the output format of joblib
            result = index.kneighbors(
                query_points, n_candidates=n_neighbors, return_distance=True
            )
            result = [
                result,
            ]

        if return_distance:
            dist, neigh_ind = zip(*result)
            result = [
                np.atleast_2d(arr)
                for arr in [np.vstack(dist), np.vstack(neigh_ind)]
            ]
        else:
            result = np.atleast_2d(np.vstack(result))
        return result

    def _check_hubness_algorithm(self):
        if self.hubness not in [
            *hubness_algorithms,
            *hubness_algorithms_long,
            None,
        ]:
            raise ValueError(f"Unrecognized hubness algorithm: {self.hubness}")

        # Users are allowed to use various identifiers for the algorithms,
        # but here we normalize them to the short abbreviations used downstream
        if self.hubness in ["mp", "mutual_proximity"]:
            self.hubness = "mp"
        elif self.hubness in ["ls", "local_scaling"]:
            self.hubness = "ls"
        elif self.hubness in ["dsl", "dis_sim_local"]:
            self.hubness = "dsl"
        elif self.hubness is None:
            pass

    def _check_algorithm_metric(self):
        if self.algorithm not in ["auto", *EXACT_ALG, *ANN_ALG]:
            raise ValueError("unrecognized algorithm: '%s'" % self.algorithm)

        if self.algorithm == "auto":
            if self.metric == "precomputed":
                alg_check = "brute"
            elif (
                callable(self.metric)
                or self.metric in VALID_METRICS["ball_tree"]
            ):
                alg_check = "ball_tree"
            else:
                alg_check = "brute"
        else:
            alg_check = self.algorithm

        if callable(self.metric):
            if self.algorithm in ["kd_tree"]:
                # callable metric is only valid for brute force and ball_tree
                raise ValueError(
                    f"{self.algorithm} algorithm does not support callable metric '{self.metric}'"
                )
        elif self.metric not in VALID_METRICS[alg_check]:
            raise ValueError(
                f"Metric '{self.metric}' not valid. Use "
                f"sorted(skhubness.neighbors.VALID_METRICS['{alg_check}']) "
                f"to get valid options. "
                f"Metric can also be a callable function."
            )

        if self.metric_params is not None and "p" in self.metric_params:
            warnings.warn(
                "Parameter p is found in metric_params. "
                "The corresponding parameter from __init__ "
                "is ignored.",
                SyntaxWarning,
                stacklevel=3,
            )
            effective_p = self.metric_params["p"]
        else:
            effective_p = self.p

        if self.metric in ["wminkowski", "minkowski"] and effective_p <= 0:
            raise ValueError("p must be greater than zero for minkowski metric")

    def _check_algorithm_hubness_compatibility(self):
        if self.hubness == "dsl":
            if self.metric in ["euclidean", "minkowski"]:
                self.metric = (
                    "euclidean"  # DSL input must still be squared Euclidean
                )
                self.hubness_params["squared"] = False
                if self.p != 2:
                    warnings.warn(
                        f"DisSimLocal only supports squared Euclidean distances: Ignoring p={self.p}."
                    )
            elif self.metric in ["sqeuclidean"]:
                self.hubness_params["squared"] = True
            else:
                warnings.warn(
                    f"DisSimLocal only supports squared Euclidean distances: Ignoring metric={self.metric}."
                )
                self.metric = "euclidean"
                self.hubness_params["squared"] = True

    def _set_hubness_reduction(self, source, target):
        if self._hubness_reduction_method is None:
            self._hubness_reduction = NoHubnessReduction()
        else:
            n_candidates = self.algorithm_params["n_candidates"]
            neigh_t_to_s = self.kcandidates(
                target,
                s_to_t=False,
                n_neighbors=n_candidates,
                return_distance=True,
            )
            neigh_dist_t_to_s = neigh_t_to_s[0]  # [:, 1:]
            neigh_ind_t_to_s = neigh_t_to_s[1]  # [:, 1:]
            if (
                self._hubness_reduction_method not in hubness_algorithms
                and self._hubness_reduction_method
                not in hubness_algorithms_long
            ):
                raise ValueError(
                    f'Hubness reduction algorithm = "{self._hubness_reduction_method}" not recognized.'
                )
            elif self._hubness_reduction_method == "csls":
                self._hubness_reduction = CSLS(**self.hubness_params)
            elif self._hubness_reduction_method == "ls":
                self._hubness_reduction = LocalScaling(**self.hubness_params)
            elif self._hubness_reduction_method == "mp":
                self._hubness_reduction = MutualProximity(**self.hubness_params)
            elif self._hubness_reduction_method == "dsl":
                self._hubness_reduction = DisSimLocal(**self.hubness_params)
            self._hubness_reduction.fit(
                neigh_dist_t_to_s,
                neigh_ind_t_to_s,
                source,
                target,
                assume_sorted=False,
            )

    def fit(self, source, target):
        self._check_algorithm_metric()
        self._check_hubness_algorithm()
        self._check_algorithm_hubness_compatibility()
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        effective_p = self.effective_metric_params_.get("p", self.p)
        if self.metric in ["wminkowski", "minkowski"]:
            self.effective_metric_params_["p"] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == "minkowski":
            p = self.effective_metric_params_.pop("p", 2)
            if p <= 0:
                raise ValueError(
                    "p must be greater than one for minkowski metric, or in ]0, 1[ for fractional norms."
                )
            elif p == 1:
                self.effective_metric_ = "manhattan"
            elif p == 2:
                self.effective_metric_ = "euclidean"
            elif p == np.inf:
                self.effective_metric_ = "chebyshev"
            else:
                self.effective_metric_params_["p"] = p

        self._fit_method = self.algorithm
        self._fit_target = target
        self._fit_source = source
        self._hubness_reduction_method = self.hubness

        if self._fit_method == "auto":
            # A tree approach is better for small number of neighbors,
            # and KDTree is generally faster when available
            if (
                self.n_neighbors is None
                or self.n_neighbors < self._fit_target.shape[0] // 2
            ) and self.metric != "precomputed":
                if self.effective_metric_ in VALID_METRICS["kd_tree"]:
                    self._fit_method = "kd_tree"
                elif (
                    callable(self.effective_metric_)
                    or self.effective_metric_ in VALID_METRICS["ball_tree"]
                ):
                    self._fit_method = "ball_tree"
                else:
                    self._fit_method = "brute"
            else:
                self._fit_method = "brute"
            self._index = None

        self._initialize(source, target)

        # Fit hubness reduction method
        self._set_hubness_reduction(source, target)

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    f"Expected n_neighbors > 0. Got {self.n_neighbors:d}"
                )
            else:
                if not np.issubdtype(type(self.n_neighbors), np.integer):
                    raise TypeError(
                        f"n_neighbors does not take {type(self.n_neighbors)} value, "
                        f"enter integer value"
                    )

        return self

    def kneighbors(
        self,
        k=None,
        source_query_points=None,
        return_distance=True,
    ):
        if k is None:
            n_neighbors = self.n_neighbors
        elif self.n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        else:
            if not np.issubdtype(type(self.n_neighbors), np.integer):
                raise TypeError(
                    f"n_neighbors does not take {type(n_neighbors)} value, enter integer value"
                )
            n_neighbors = k
        if source_query_points is None:
            source_query_points = self._fit_source

        # First obtain candidate neighbors
        query_dist, query_ind = self.kcandidates(
            source_query_points, return_distance=True
        )
        query_dist = np.atleast_2d(query_dist)
        query_ind = np.atleast_2d(query_ind)

        # Second, reduce hubness
        (
            hubness_reduced_query_dist,
            query_ind,
        ) = self._hubness_reduction.transform(
            query_dist,
            query_ind,
            source_query_points,
            assume_sorted=True,
        )
        # Third, sort hubness reduced candidate neighbors to get the final k neighbors
        kth = np.arange(n_neighbors)
        mask = np.argpartition(hubness_reduced_query_dist, kth=kth)[
            :, :n_neighbors
        ]
        hubness_reduced_query_dist = np.take_along_axis(
            hubness_reduced_query_dist, mask, axis=1
        )
        query_ind = np.take_along_axis(query_ind, mask, axis=1)

        if return_distance:
            result = hubness_reduced_query_dist, query_ind
        else:
            result = query_ind
        return result
