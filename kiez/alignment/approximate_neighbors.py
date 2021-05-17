# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from typing import Tuple, Union

import numpy as np


class ApproximateNearestNeighbor(ABC):
    """Abstract base class for approximate nearest neighbor search methods.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.
    """

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "sqeuclidean",
        n_jobs: int = 1,
        verbose: int = 0,
        *args,
        **kwargs,
    ):
        self.n_candidates = n_candidates
        self.metric = metric
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def fit(self, x):
        pass  # pragma: no cover

    @abstractmethod
    def kneighbors(
        self, query_points=None, n_candidates=None, return_distance=True
    ) -> Union[Tuple[np.array, np.array], np.array]:
        pass  # pragma: no cover


class UnavailableANN(ApproximateNearestNeighbor):
    """Placeholder for ANN methods that are not available on specific platforms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "The chosen approximate nearest neighbor method is not supported on your platform."
        )

    def fit(self, x):
        pass

    def kneighbors(
        self, query_points=None, n_candidates=None, return_distance=True
    ):
        pass
