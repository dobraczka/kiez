# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from kiez.hubness_reduction import DisSimLocal
from kiez.hubness_reduction.base import NoHubnessReduction
from kiez.neighbors import SklearnNN


class Kiez:
    def __init__(
        self,
        n_neighbors=5,
        algorithm=None,
        hubness: str = None,
    ):
        if not np.issubdtype(type(n_neighbors), np.integer):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, enter"
                " integer value"
            )
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        self.n_neighbors = n_neighbors
        self.algorithm = (
            SklearnNN(n_candidates=n_neighbors) if algorithm is None else algorithm
        )
        self.hubness = NoHubnessReduction() if hubness is None else hubness
        self._check_algorithm_hubness_compatibility()

    def __repr__(self):
        return (
            f"Kiez(n_neighbors: {self.n_neighbors}, algorithm: {self.algorithm},"
            f" hubness: {self.hubness}"
        )

    def _kcandidates(
        self, query_points, *, s_to_t=True, k=None, return_distance=True
    ) -> np.ndarray or (np.ndarray, np.ndarray):
        if k is None:
            k = self.algorithm.n_candidates

        # The number of candidates must not be less than the number of neighbors used downstream
        if k < self.n_neighbors:
            k = self.n_neighbors
        return self.algorithm.kneighbors(
            k=k,
            query=query_points,
            s_to_t=s_to_t,
            return_distance=return_distance,
        )

    def _check_algorithm_hubness_compatibility(self):
        if isinstance(self.hubness, DisSimLocal):
            if self.algorithm.metric in ["euclidean", "minkowski"]:
                self.hubness.squared = False
                if hasattr(self.algorithm, "p"):
                    if self.algorithm.p != 2:
                        raise ValueError(
                            "DisSimLocal only supports squared Euclidean distances. If"
                            " the provided NNAlgorithm has a `p` parameter it must be"
                            f" set to p=2. Now it is p={self.algorithm.p}"
                        )
            elif self.algorithm.metric in ["sqeuclidean"]:
                self.hubness.squared = True
            else:
                raise ValueError(
                    "DisSimLocal only supports squared Euclidean distances, not"
                    f" metric={self.algorithm.metric}."
                )

    def fit(self, source, target):
        self.algorithm.fit(source, target)
        neigh_dist_t_to_s, neigh_ind_t_to_s = self._kcandidates(
            target,
            s_to_t=False,
            k=self.algorithm.n_candidates,
            return_distance=True,
        )
        self.hubness.fit(
            neigh_dist_t_to_s,
            neigh_ind_t_to_s,
            source,
            target,
            assume_sorted=False,
        )
        return self

    def kneighbors(
        self,
        source_query_points=None,
        k=None,
        return_distance=True,
    ):
        # function loosely adapted from skhubness: https://github.com/VarIr/scikit-hubness

        if k is None:
            n_neighbors = self.n_neighbors
        else:
            n_neighbors = k
        # First obtain candidate neighbors
        query_dist, query_ind = self._kcandidates(
            source_query_points, return_distance=True
        )
        query_dist = np.atleast_2d(query_dist)
        query_ind = np.atleast_2d(query_ind)

        # Second, reduce hubness
        hubness_reduced_query_dist, query_ind = self.hubness.transform(
            query_dist,
            query_ind,
            source_query_points,
            assume_sorted=True,
        )
        # Third, sort hubness reduced candidate neighbors to get the final k neighbors
        kth = np.arange(n_neighbors)
        mask = np.argpartition(hubness_reduced_query_dist, kth=kth)[:, :n_neighbors]
        hubness_reduced_query_dist = np.take_along_axis(
            hubness_reduced_query_dist, mask, axis=1
        )
        query_ind = np.take_along_axis(query_ind, mask, axis=1)

        if return_distance:
            result = hubness_reduced_query_dist, query_ind
        else:
            result = query_ind
        return result
