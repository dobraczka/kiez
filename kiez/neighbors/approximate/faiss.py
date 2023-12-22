import logging
from typing import Optional

import numpy as np

from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class Faiss(NNAlgorithm):
    """Wrapper for `faiss` library.

    Faiss implements a number of (A)NN algorithms and enables the use of GPUs.

    Parameters
    ----------
    n_candidates: int, default = 5
        number of nearest neighbors used in search
    metric: str, default = 'l2'
        distance measure used in search
        possible measures are found in :obj:`Faiss.valid_metrics`
        Euclidean is the same as l2, expect for taking the sqrt of the result
    index_key: str, default = None
        index name to use
        If none is provided will determine the best automatically
        Else will use it as input for :meth:`faiss.index_factory`
    index_param: str, default = None
        Hyperparameter string for the indexing algorithm
        See https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning#auto-tuning-the-runtime-parameters for info
    use_gpu: bool
        If true uses all available gpus

    Examples
    --------
    >>> import numpy as np
    >>> from kiez import Kiez
    >>> source = np.random.rand(1000, 512)
    >>> target = np.random.rand(100, 512)
    >>> k_inst = Kiez(algorithm="Faiss")
    >>> k_inst.fit(source, target)

    >>> k_inst = Kiez(algorithm="Faiss",algorithm_kwargs={"metric":"euclidean","index_key":"Flat"})

    supply hyperparameters for indexing algorithm

    >>> k_inst = Kiez(algorithm="Faiss",algorithm_kwargs={"index_key":"HNSW32","index_param":"efSearch=16383"})

    Notes
    -----
    For details about configuring faiss consult their wiki: https://github.com/facebookresearch/faiss/wiki
    """

    valid_metrics = ("l2", "euclidean")
    valid_spaces = "l2"

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "l2",
        index_key: str = "Flat",
        index_param: Optional[str] = None,
        use_gpu: bool = False,
        verbose: int = logging.WARNING,
    ):
        if faiss is None:  # pragma: no cover
            raise ImportError(
                "Please install the `faiss` package, before using this class.\nSee here"
                " for installation instructions:"
                " https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
            )
        if metric not in self.__class__.valid_metrics:
            raise ValueError(
                f"Unknown metric {metric}, please use one of {self.valid_metrics}"
            )
        if metric not in self.__class__.valid_spaces:
            if metric == "euclidean":
                self.space = "l2"
        else:
            self.space = metric
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=None)
        self.index_key = index_key
        self.index_param = index_param
        self.use_gpu = use_gpu
        self.verbose = verbose

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_candidates={self.n_candidates},"
            + f"metric={self.metric},"
            + f"index_key={self.index_key},"
            + f"index_param={{{self.index_param}}},"
            + f"use_gpu={self.use_gpu})"
        )

    def _fit(self, data, is_source: bool):
        dim = data.shape[1]
        index = faiss.index_factory(dim, self.index_key)
        params = faiss.ParameterSpace()
        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
            params = faiss.GpuParameterSpace()
        if self.index_param is not None:
            params.set_index_parameters(index, self.index_param)
        index.add(data)
        return index

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        if is_self_querying:
            dist, ind = index.search(self.source_, k)
        else:
            dist, ind = index.search(query, k)
        if return_distance:
            if self.metric == "euclidean":
                dist = np.sqrt(dist)
            return dist, ind
        return ind
