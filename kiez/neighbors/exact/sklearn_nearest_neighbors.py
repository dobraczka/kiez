# documentation copied in parts from scikit-learn
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm
from sklearn.neighbors import VALID_METRICS, NearestNeighbors


class SklearnNN(NNAlgorithm):
    """Wrapper for scikit learn's NearestNeighbors class

    Parameters
    ----------
    n_candidates: int
        number of nearest neighbors used in search
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`sklearn.neighbors.BallTree`
        - 'kd_tree' will use :class:`sklearn.neighbors.KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`~kiez.neighbors.neighbor_algorithm_base.NNAlgorithm.fit` method.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric: str, default = 'minkowski'
        distance measure used in search
        default is minkowski with p=2, which is equivlanet to euclidean
        possible measures are found in :obj:`SklearnNN.valid_metrics`
    p: int, default=2
        Parameter for the Minkowski metric.
        When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used.
    metric_params: dict, default=None
        Additional keyword arguments for the metric function.
        metric_params
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    Notes
    -----
    See also scikit learn's guide: https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors
    """

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
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs)
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric_params = metric_params

    def __repr__(self):
        ret_str = (
            f"{self.__class__.__name__}(n_candidates={self.n_candidates},"
            + f"algorithm={self.algorithm},"
            + f"leaf_size={self.leaf_size},"
            + f"metric={self.metric},"
            + f"n_jobs={self.n_jobs} "
        )
        if (
            hasattr(self, "source_index")
            and self.source_index._fit_method != self.algorithm
        ):
            return ret_str + f" and effective algo is {self.source_index._fit_method}"
        ret_str += ")"
        return ret_str

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
        return index.kneighbors(X=query, n_neighbors=k, return_distance=return_distance)
