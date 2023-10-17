import logging
import warnings
from typing import Optional

import numpy as np

from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None
try:
    import autofaiss
except ImportError:  # pragma: no cover
    autofaiss = None


class Faiss(NNAlgorithm):
    """Wrapper for `faiss` library.

    Faiss implements a number of (A)NN algorithms and enables the use of GPUs.
    If it is installed and you let it, kiez utilizes the `autofaiss` package to
    find the appropriate indexing structure and optimizes the hyperparameters of
    the algorithm

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

    get info about selected indices

    >>> k_inst.algorithm.source_index_infos["index_key"]
    'HNSW15'

    >>> k_inst = Kiez(algorithm="Faiss",algorithm_kwargs={"metric":"euclidean","index_key":"Flat"})


    supply hyperparameters for indexing algorithm

    >>> k_inst = Kiez(algorithm="Faiss",algorithm_kwargs={"index_key":"HNSW32","index_param":"efSearch=16383"})

    Notes
    -----
    For details about configuring faiss consult their wiki: https://github.com/facebookresearch/faiss/wiki
    For details about autofaiss see their documentation: https://criteo.github.io/autofaiss/
    """

    valid_metrics = ["l2", "euclidean"]
    valid_spaces = ["l2"]

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "l2",
        index_key: Optional[str] = None,
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
        use_auto_tune = autofaiss is not None
        # check index string
        if index_key:
            try:
                faiss.index_factory(1, index_key)
            except RuntimeError as exc:
                raise ValueError(
                    f'Could not parse index "{index_key}".\n Please consult the faiss'
                    " wiki to create a correct instruction:"
                    " https://github.com/facebookresearch/faiss/wiki/The-index-factory"
                ) from exc
            # user seems to know what they want so no tuning
            if index_param or index_key == "Flat":
                use_auto_tune = False
        elif index_param:
            warnings.warn(
                "Index key not set but hyperparameter given. Are you sure this is"
                " intended?"
            )
        else:
            # no index and no hyperparams so check
            # if autofaiss is available
            if autofaiss is None:  # pragma: no cover
                warnings.warn(
                    "Please install the `autofaiss` package, to enable automatic index"
                    " selection.\nYou can install `autofaiss` via: pip install"
                    " autofaiss\n Will use `Flat` index for now, but there are probably"
                    " better choices..."
                )
                use_auto_tune = False
        self.index_key = index_key
        self.index_param = index_param
        self.use_auto_tune = use_auto_tune
        self.use_gpu = use_gpu
        self.index_infos = None
        self.verbose = verbose

    def _source_target_repr(self, is_source: bool):
        ret_str = f"{self.__class__.__name__}(n_candidates={self.n_candidates},metric={self.metric},"
        if is_source:
            ret_str += (
                f"index_key={self.source_index_key},"
                f" index_param={{{self.source_index_param}}},"
            )
        else:
            ret_str += (
                f"index_key={self.target_index_key},"
                f" index_param={{{self.target_index_param}}},"
            )
        ret_str += f"use_auto_tune={self.use_auto_tune}, use_gpu={self.use_gpu})"
        return ret_str

    def __repr__(self):
        if hasattr(self, "source_index_key") and hasattr(self, "target_index_key"):
            ret_str = (
                f"Source: {self._source_target_repr(True)}, "
                f"Target: {self._source_target_repr(False)}"
            )
        elif hasattr(self, "source_index_key"):
            ret_str = f"{self._source_target_repr(True)}"
        else:
            ret_str = (
                f"{self.__class__.__name__}(n_candidates={self.n_candidates},"
                + f"metric={self.metric},"
                + f"index_key={self.index_key},"
                + f"index_param={{{self.index_param}}},"
                + f"use_auto_tune={self.use_auto_tune},"
                + f"use_gpu={self.use_gpu})"
            )
        return ret_str

    def _to_float32(self, data):
        if not data.dtype == "float32":
            return data.astype("float32")
        return data

    def _fit(self, data, is_source: bool):
        dim = data.shape[1]
        if self.use_auto_tune:
            index, index_infos = autofaiss.build_index(
                self._to_float32(data),
                index_key=self.index_key,
                index_param=self.index_param,
                metric_type=self.space,
                save_on_disk=False,
                use_gpu=self.use_gpu,
                verbose=self.verbose,
            )
            if is_source:
                self.source_index_key = index_infos["index_key"]
                self.source_index_param = index_infos["index_param"]
                self.source_index_infos = index_infos
            else:
                self.target_index_key = index_infos["index_key"]
                self.target_index_param = index_infos["index_param"]
                self.target_index_infos = index_infos
        else:
            index = faiss.index_factory(dim, self.index_key)
            params = faiss.ParameterSpace()
            if self.use_gpu:
                index = faiss.index_cpu_to_all_gpus(index)
                params = faiss.GpuParameterSpace()
            if self.index_param is not None:
                params.set_index_parameters(index, self.index_param)
            index.add(self._to_float32(data))
        return index

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        if is_self_querying:
            dist, ind = index.search(self._to_float32(self.source_), k)
        else:
            dist, ind = index.search(self._to_float32(query), k)
        if return_distance:
            if self.metric == "euclidean":
                dist = np.sqrt(dist)
            return dist, ind
        else:
            return ind
