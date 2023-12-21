import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..neighbors import NNAlgorithm


class HubnessReduction(ABC):
    """Base class for hubness reduction."""

    def __init__(self, nn_algo: NNAlgorithm, verbose: int = 0, **kwargs):
        self.nn_algo = nn_algo
        self.verbose = verbose
        if nn_algo.n_candidates == 1:
            raise ValueError(
                "Cannot perform hubness reduction with a single candidate per query!"
            )

    @abstractmethod
    def _fit(self, neigh_dist, neigh_ind, source, target):
        pass  # pragma: no cover

    def fit(self, source, target=None):
        self.nn_algo.fit(source, target)
        if target is None:
            target = source
        neigh_dist_t_to_s, neigh_ind_t_to_s = self.nn_algo.kneighbors(
            k=self.nn_algo.n_candidates,
            query=target,
            s_to_t=False,
            return_distance=True,
        )
        self._fit(
            neigh_dist_t_to_s,
            neigh_ind_t_to_s,
            source,
            target,
        )

    @abstractmethod
    def transform(self, neigh_dist, neigh_ind, query) -> Tuple[np.ndarray, np.ndarray]:
        pass  # pragma: no cover

    def _set_k_if_needed(self, k: Optional[int] = None) -> int:
        if k is None:
            warnings.warn(
                f"No k supplied, setting to n_candidates = {self.nn_algo.n_candidates}"
            )
            return self.nn_algo.n_candidates
        if k > self.nn_algo.n_candidates:
            warnings.warn(
                "k > n_candidates supplied! Setting to n_candidates ="
                f" {self.nn_algo.n_candidates}"
            )
            return self.nn_algo.n_candidates
        return k

    def kneighbors(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        n_neighbors = self._set_k_if_needed(k)
        # First obtain candidate neighbors
        query_dist, query_ind = self.nn_algo.kneighbors(
            query=None, k=self.nn_algo.n_candidates, return_distance=True
        )

        # Second, reduce hubness
        hubness_reduced_query_dist, query_ind = self.transform(
            query_dist,
            query_ind,
            self.nn_algo.source_,
        )
        # Third, sort hubness reduced candidate neighbors to get the final k neighbors
        kth = np.arange(n_neighbors)
        mask = np.argpartition(hubness_reduced_query_dist, kth=kth)[:, :n_neighbors]
        hubness_reduced_query_dist = np.take_along_axis(
            hubness_reduced_query_dist, mask, axis=1
        )
        query_ind = np.take_along_axis(query_ind, mask, axis=1)
        return hubness_reduced_query_dist, query_ind


class NoHubnessReduction(HubnessReduction):
    """Base class for hubness reduction."""

    def _fit(self, neigh_dist, neigh_ind, source, target):
        pass  # pragma: no cover

    def fit(self, source, target=None):
        self.nn_algo.fit(source, target, only_fit_target=True)

    def transform(self, neigh_dist, neigh_ind, query) -> Tuple[np.ndarray, np.ndarray]:
        pass  # pragma: no cover

    def kneighbors(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        n_neighbors = self._set_k_if_needed(k)
        return self.nn_algo.kneighbors(query=None, k=n_neighbors, return_distance=True)
