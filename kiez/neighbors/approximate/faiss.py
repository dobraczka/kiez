import numpy as np
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class Faiss(NNAlgorithm):
    valid_metrics = ["l2"]

    def __init__(
        self,
        n_candidates=5,
        metric="euclidean",
        index_type="auto",
    ):
        if faiss is None:  # pragma: no cover
            raise ImportError(
                "Please install the `faiss` package, before using this class.\nSee here"
                " for installation instructions:"
                " https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
            )
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=None)
        self.index_type = index_type

    def __repr__(self):
        ret_str = (
            f"{self.__class__.__name__}(n_candidates={self.n_candidates},"
            + f"metric={self.metric},"
            + f"index_type={self.index_type})"
        )
        return ret_str

    def _to_float32(self, data):
        if not data.dtype == "float32":
            return data.astype("float32")
        return data

    def _fit(self, data, is_source: bool):
        dim = data.shape[1]
        index = faiss.IndexFlatL2(dim)
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
