# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from class_resolver import HintOrType

from kiez.hubness_reduction import DisSimLocal, hubness_reduction_resolver
from kiez.hubness_reduction.base import HubnessReduction
from kiez.neighbors import NNAlgorithm, SklearnNN, nn_algorithm_resolver


class Kiez:
    """Performs hubness reduced nearest neighbor search for entity alignment

    Use the given algorithm to :meth:`fit` the data and calculate the
    :meth:`kneighbors`.

    Parameters
    ----------
    n_neighbors : int, default=5
        number of nearest neighbors used in search
    algorithm : :obj:`~kiez.neighbors.NNAlgorithm`, default = None
        initialised `NNAlgorithm` object that will be used for neighbor search
        If no algorithm is provided :obj:`~kiez.neighbors.SklearnNN`
        is used with default values
    algorithm_kwargs :
        A dictionary of keyword arguments to pass to the :obj:`~kiez.neighbors.NNAlgorithm`
        if given as a class in the ``algorithm`` argument.

    hubness :
        Either an instance of a :obj:`~kiez.hubness_reduction.base.HubnessReduction`,
        the class for a :obj:`~kiez.hubness_reduction.base.HubnessReduction` that should
        be instantiated, the name of the hubness reduction method, or if None, defaults
        to no hubness reduction.
    hubness_kwargs :
        A dictionary of keyword arguments to pass to the :obj:`~kiez.hubness_reduction.base.HubnessReduction`
        if given as a class in the ``hubness`` argument.

    Examples
    --------
    >>> from kiez import Kiez
    >>> import numpy as np
    >>> # create example data
    >>> rng = np.random.RandomState(0)
    >>> source = rng.rand(100,50)
    >>> target = rng.rand(100,50)
    >>> # fit and get neighbors
    >>> k_inst = Kiez()
    >>> k_inst.fit(source, target)
    >>> nn_dist, nn_ind = k_inst.kneighbors()

    Using a specific algorithm and hubness reduction

    >>> from kiez import Kiez
    >>> import numpy as np
    >>> # create example data
    >>> rng = np.random.RandomState(0)
    >>> source = rng.rand(100,50)
    >>> target = rng.rand(100,50)
    >>> # prepare algorithm and hubness reduction
    >>> from kiez.neighbors import HNSW
    >>> hnsw = HNSW(n_candidates=10)
    >>> from kiez.hubness_reduction import CSLS
    >>> hr = CSLS()
    >>> # fit and get neighbors
    >>> k_inst = Kiez(n_neighbors=5, algorithm=hnsw, hubness=hr)
    >>> k_inst.fit(source, target)
    >>> nn_dist, nn_ind = k_inst.kneighbors()
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        algorithm: HintOrType[NNAlgorithm] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        hubness: HintOrType[HubnessReduction] = None,
        hubness_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not np.issubdtype(type(n_neighbors), np.integer):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, enter"
                " integer value"
            )
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        self.n_neighbors = n_neighbors
        if algorithm is None and algorithm_kwargs is None:
            algorithm_kwargs = dict(n_candidates=n_neighbors)
        self.algorithm = nn_algorithm_resolver.make(algorithm, algorithm_kwargs)
        self.hubness = hubness_reduction_resolver.make(hubness, hubness_kwargs)
        self._check_algorithm_hubness_compatibility()

    def __repr__(self):
        return (
            f"Kiez(n_neighbors: {self.n_neighbors}, algorithm: {self.algorithm},"
            f" hubness: {self.hubness})"
            f" {self.algorithm._describe_source_target_fitted()}"
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

    def fit(self, source, target=None) -> Kiez:
        """Fits the algorithm and hubness reduction method

        Parameters
        ----------
        source : matrix of shape (n_samples, n_features)
            embeddings of source entities
        target : matrix of shape (m_samples, n_features)
            embeddings of target entities. If none given, uses the source.

        Returns
        -------
        Kiez
            Fitted kiez instance
        """
        self.algorithm.fit(source, target)
        if target is None:
            target = source
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
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Retrieves the k-nearest neighbors using the supplied nearest neighbor algorithm and hubness reduction method.

        Parameters
        ----------
        source_query_points : matrix of shape (n_samples, n_features), default = None
            subset of source entity embeddings
            if `None` all source entities are used for querying
        k : int, default = None
            number of nearest neighbors to search for
        return_distance : bool, default = True
            Whether to return distances
            If `False` only indices are returned

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the distance between source and target entities
            only present if return_distance=True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
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
