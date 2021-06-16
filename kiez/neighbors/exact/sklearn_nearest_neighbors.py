from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm
from sklearn.neighbors import VALID_METRICS, NearestNeighbors


class SklearnNN(NNAlgorithm):
    valid_metrics = VALID_METRICS

    def __init__(
        self,
        n_candidates=5,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_candidates=n_candidates, metric=metric, n_jobs=n_jobs
        )
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params

    def _fit(self, data, is_source: bool):
        nn = NearestNeighbors(
            n_neighbors=self.n_candidates,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        nn.fit(data)
        return nn

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        if is_self_querying:
            return index.kneighbors(
                X=None, n_neighbors=k, return_distance=return_distance
            )
        return index.kneighbors(
            X=query, n_neighbors=k, return_distance=return_distance
        )
