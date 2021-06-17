import warnings
from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted


class NNAlgorithm(ABC):
    valid_metrics = []

    def __init__(self, n_candidates, metric, n_jobs):
        self.n_candidates = n_candidates
        self.metric = metric
        self.n_jobs = n_jobs

    def _describe_source_target_fitted(self):
        if hasattr(self, "source_"):
            return (
                f" is fitted with: source.shape={self.source_.shape} and"
                f" target.shape={self.target_.shape}"
            )
        return " is unfitted"

    @property
    def valid_metrics(self):
        pass

    @abstractmethod
    def _fit(self, data, is_source: bool):
        pass

    def fit(self, source, target):
        self.source_equals_target = np.array_equal(source, target)
        if self.source_equals_target:
            self.source_index = self._fit(source, True)
            self.target_index = self.source_index
        else:
            self.source_index = self._fit(source, True)
            self.target_index = self._fit(target, False)
        self.source_ = source
        self.target_ = target

    def _check_k_value(self, k, needed_space):
        if k <= 0:
            raise ValueError(f"Expected k > 0. Got {k}")
        if not np.issubdtype(type(k), np.integer):
            raise TypeError(f"k does not take {type(k)} value, enter integer value")
        if k > needed_space:
            warnings.warn(
                f"k={k} is larger than number of samples in indexed space.\n"
                + "Setting to k={target_space_size}"
            )
            return needed_space
        return k

    @abstractmethod
    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        pass

    def kneighbors(self, k=None, query=None, s_to_t=True, return_distance=True):
        check_is_fitted(self, ["source_index", "target_index"])
        k = self.n_candidates if k is None else k
        is_self_querying = query is None and self.source_equals_target

        if s_to_t:
            query = self.source_ if query is None else query
            index = self.target_index
            needed_space = self.target_.shape[0]
        else:
            query = self.target_ if query is None else query
            index = self.source_index
            needed_space = self.source_.shape[0]
        k = self._check_k_value(k, needed_space)
        return self._kneighbors(
            k=k,
            query=query,
            index=index,
            return_distance=return_distance,
            is_self_querying=is_self_querying,
        )


class NNAlgorithmWithJoblib(NNAlgorithm):
    @abstractmethod
    def _kneighbors_part(self, k, query, index, return_distance, is_self_querying):
        pass

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        n_jobs = effective_n_jobs(self.n_jobs)
        # assume joblib>=0.12
        delayed_query = delayed(self._kneighbors_part)
        parallel_kwargs = {"prefer": "threads"}
        result = Parallel(n_jobs, **parallel_kwargs)(
            delayed_query(
                k=k,
                query=query[s],
                index=index,
                return_distance=return_distance,
                is_self_querying=is_self_querying,
            )
            for s in gen_even_slices(query.shape[0], n_jobs)
        )
        if return_distance:
            dist, neigh_ind = zip(*result)
            result = [
                np.atleast_2d(arr) for arr in [np.vstack(dist), np.vstack(neigh_ind)]
            ]
        else:
            result = np.atleast_2d(np.vstack(result))
        return result
