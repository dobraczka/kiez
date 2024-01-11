import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TypeVar

import numpy as np

from ..neighbors import NNAlgorithm

try:
    import torch
except ImportError:
    torch = None

T = TypeVar("T")


class HubnessReduction(ABC):
    """Base class for hubness reduction."""

    def __init__(self, nn_algo: NNAlgorithm, verbose: int = 0, **kwargs):
        self.nn_algo = nn_algo
        self.verbose = verbose
        self._use_torch = False
        if nn_algo.n_candidates == 1:
            raise ValueError(
                "Cannot perform hubness reduction with a single candidate per query!"
            )

    @abstractmethod
    def _fit(self, neigh_dist: T, neigh_ind: T, source: T, target: T):
        pass  # pragma: no cover

    def fit(self, source: T, target: Optional[T] = None):
        self.nn_algo.fit(source, target)
        if target is None:
            target = source
        neigh_dist_t_to_s, neigh_ind_t_to_s = self.nn_algo.kneighbors(
            k=self.nn_algo.n_candidates,
            query=target,
            s_to_t=False,
            return_distance=True,
        )
        if torch and isinstance(neigh_dist_t_to_s, torch.Tensor):
            self._use_torch = True
        self._fit(
            neigh_dist_t_to_s,
            neigh_ind_t_to_s,
            source,
            target,
        )

    @abstractmethod
    def transform(self, neigh_dist, neigh_ind, query) -> Tuple[T, T]:
        pass  # pragma: no cover

    def _set_k_if_needed(self, k: Optional[int] = None) -> int:
        if k is None:
            warnings.warn(
                f"No k supplied, setting to n_candidates = {self.nn_algo.n_candidates}",
                stacklevel=2,
            )
            return self.nn_algo.n_candidates
        if k > self.nn_algo.n_candidates:
            warnings.warn(
                "k > n_candidates supplied! Setting to n_candidates ="
                f" {self.nn_algo.n_candidates}",
                stacklevel=2,
            )
            return self.nn_algo.n_candidates
        return k

    @staticmethod
    def _sort(hubness_reduced_query_dist, query_ind, n_neighbors: int) -> Tuple[T, T]:
        if torch and isinstance(hubness_reduced_query_dist, torch.Tensor):
            mask = torch.argsort(hubness_reduced_query_dist)[:, :n_neighbors]
            hubness_reduced_query_dist = torch.take_along_dim(
                hubness_reduced_query_dist, mask, dim=1
            )
            query_ind = torch.take_along_dim(query_ind, mask, dim=1)
        else:
            kth = np.arange(n_neighbors)
            mask = np.argpartition(hubness_reduced_query_dist, kth=kth)[:, :n_neighbors]
            hubness_reduced_query_dist = np.take_along_axis(
                hubness_reduced_query_dist, mask, axis=1
            )
            query_ind = np.take_along_axis(query_ind, mask, axis=1)
        return hubness_reduced_query_dist, query_ind

    def kneighbors(self, k: Optional[int] = None) -> Tuple[T, T]:
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
        return HubnessReduction._sort(
            hubness_reduced_query_dist, query_ind, n_neighbors
        )


class NoHubnessReduction(HubnessReduction):
    """Base class for hubness reduction."""

    def _fit(self, neigh_dist, neigh_ind, source, target):
        pass  # pragma: no cover

    def fit(self, source, target=None):
        self.nn_algo.fit(source, target, only_fit_target=True)

    def transform(self, neigh_dist, neigh_ind, query) -> Tuple[T, T]:
        return neigh_dist, neigh_ind

    def kneighbors(self, k: Optional[int] = None) -> Tuple[T, T]:
        n_neighbors = self._set_k_if_needed(k)
        return self.nn_algo.kneighbors(query=None, k=n_neighbors, return_distance=True)
