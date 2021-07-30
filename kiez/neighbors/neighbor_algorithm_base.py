import warnings
from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted


class NNAlgorithm(ABC):
    """Base class for nearest neighbor algorithms"""

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
        pass  # pragma: no cover

    @abstractmethod
    def _fit(self, data, is_source: bool):
        pass  # pragma: no cover

    def fit(self, source: np.ndarray, target: np.ndarray = None):
        """Indexes the given data using the underlying algorithm

        Parameters
        ----------
        source : matrix of shape (n_samples, n_features)
            embeddings of source entities
        target : matrix of shape (m_samples, n_features)
            embeddings of target entities or None in a single-source use case

        Raises
        ------
        ValueError
            If source and target have a different number of features
        """
        self.source_equals_target = target is None
        if self.source_equals_target:
            self.source_index = self._fit(source, True)
            self.target_index = self.source_index
            target = source
        else:
            if source.shape[1] != target.shape[1]:
                raise ValueError(
                    "Expected source and target to have the same number of features,"
                    f" but got source.shape: {source.shape} and target.shape:"
                    f" {target.shape}"
                )
            self.source_index = self._fit(source, True)
            self.target_index = self._fit(target, False)
        self.source_ = source
        self.target_ = target

    def _check_k_value(self, k, needed_space):
        if not np.issubdtype(type(k), np.integer):
            raise TypeError(f"k does not take {type(k)} value, enter integer value")
        if k <= 0:
            raise ValueError(f"Expected k > 0. Got {k}")
        if k > needed_space:
            warnings.warn(
                f"k={k} is larger than number of samples in indexed space.\n"
                + f"Setting to k={needed_space}"
            )
            return needed_space
        return k

    @abstractmethod
    def _kneighbors(self, query, k, index, return_distance, is_self_querying):
        pass  # pragma: no cover

    def kneighbors(self, query=None, k=None, s_to_t=True, return_distance=True):
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
        pass  # pragma: no cover

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
