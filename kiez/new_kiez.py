# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from class_resolver import HintOrType

from kiez.hubness_reduction import new_hubness_reduction_resolver
from kiez.hubness_reduction.new_base import NewHubnessReduction, NewNoHubnessReduction
from kiez.neighbors import NNAlgorithm, nn_algorithm_resolver


class NewKiez:
    """Performs hubness reduced nearest neighbor search for entity alignment

    Use the given algorithm to :meth:`fit` the data and calculate the
    :meth:`kneighbors`.

    Parameters
    ----------
    n_neighbors : int, default=5
        number of nearest neighbors used in search
    algorithm : :obj:`~kiez.neighbors.NNAlgorithm`, default = None
        initialised `NNAlgorithm` object that will be used for neighbor search
        If no algorithm is provided :obj:`~kiez.neighbors.Faiss` is used if available else
        :obj:`~kiez.neighbors.SklearnNN` is used with default values
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
    >>> from kiez.neighbors import NMSLIB
    >>> hnsw = NMSLIB(n_candidates=10)
    >>> from kiez.hubness_reduction import CSLS
    >>> hr = CSLS()
    >>> # fit and get neighbors
    >>> k_inst = Kiez(n_neighbors=5, algorithm=hnsw, hubness=hr)
    >>> k_inst.fit(source, target)
    >>> nn_dist, nn_ind = k_inst.kneighbors()

    You can also initalize Kiez via a json file

    >>> kiez = Kiez.from_path("tests/example_conf.json")
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        algorithm: HintOrType[NNAlgorithm] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        hubness: HintOrType[NewHubnessReduction] = None,
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
            algorithm_kwargs = {"n_candidates": n_neighbors}
        if algorithm is None:
            try:
                algorithm = nn_algorithm_resolver.make("Faiss", algorithm_kwargs)
            except ImportError:
                algorithm = nn_algorithm_resolver.make("SklearnNN", algorithm_kwargs)
        else:
            algorithm = nn_algorithm_resolver.make(algorithm, algorithm_kwargs)
        assert algorithm
        if hubness_kwargs is None:
            hubness_kwargs = dict()
        hubness_kwargs["nn_algo"] = algorithm
        self.hubness = new_hubness_reduction_resolver.make(hubness, hubness_kwargs)

    def __repr__(self):
        return (
            f"Kiez(n_neighbors: {self.n_neighbors}, algorithm: {self.hubness.nn_algo},"
            f" hubness: {self.hubness})"
            f" {self.algorithm._describe_source_target_fitted()}"
        )

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> Kiez:
        """Load a Kiez instance from configuration in a JSON file, based on its path."""
        with open(path) as file:
            return cls(**json.load(file))

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
        self.hubness.fit(source, target)
        return self

    def kneighbors(
        self,
        return_distance=True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Retrieves the k-nearest neighbors using the supplied nearest neighbor algorithm and hubness reduction method.

        Parameters
        ----------
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
        hubness_reduced_query_dist, query_ind = self.hubness.k_neighbors(
            self.n_neighbors
        )

        if return_distance:
            result = hubness_reduced_query_dist, query_ind
        else:
            result = query_ind
        return result
