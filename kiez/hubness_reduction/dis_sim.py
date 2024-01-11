# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from typing import Tuple, TypeVar

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted

from .base import HubnessReduction

T = TypeVar("T")

_DESIRED_P_VALUE = 2
_MINIMUM_DIST = 0.0

try:
    import torch
except ImportError:
    torch = None


class DisSimLocal(HubnessReduction):
    """Hubness reduction with DisSimLocal.

    Uses the formula presented in [1]_.

    Parameters
    ----------
    squared: bool, default = True
        DisSimLocal operates on squared Euclidean distances.
        If True, return (quasi) squared Euclidean distances;
        if False, return (quasi) Eucldean distances.

    References
    ----------
    .. [1] Hara K, Suzuki I, Kobayashi K, Fukumizu K, Radovanović M (2016)
           Flattening the density gradient for eliminating spatial centrality to reduce hubness.
           In: Proceedings of the 30th AAAI conference on artificial intelligence, pp 1659-1665.
           https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12055
    """

    def __init__(self, squared: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.squared = squared
        if self.nn_algo.metric in ["euclidean", "minkowski"]:
            self.squared = False
            if hasattr(self.nn_algo, "p") and self.nn_algo.p != _DESIRED_P_VALUE:
                raise ValueError(
                    "DisSimLocal only supports squared Euclidean distances. If"
                    " the provided NNAlgorithm has a `p` parameter it must be"
                    f" set to p=2. Now it is p={self.nn_algo.p}"
                )
        elif self.nn_algo.metric in ["sqeuclidean"]:
            self.squared = True
        else:
            raise ValueError(
                "DisSimLocal only supports squared Euclidean distances, not"
                f" metric={self.nn_algo.metric}."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(squared = {self.squared})"

    def _fit(
        self,
        neigh_dist,
        neigh_ind,
        source,
        target,
    ) -> "DisSimLocal":
        """Fit the model using target, neigh_dist, and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (colums).
        neigh_ind: shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.
        source: shape (n_samples, n_features)
            source embedding, where n_samples is the number of vectors,
            and n_features their dimensionality (number of features).
        target: shape (n_samples, n_features)
            Target embedding, where n_samples is the number of vectors,
            and n_features their dimensionality (number of features).

        Returns
        -------
        DisSimLocal
            Fitted DisSimLocal
        """
        # Calculate local neighborhood centroids among the training points
        knn = neigh_ind
        centroids = source[knn].mean(axis=1)
        if self._use_torch:
            # see https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py#L87C21-L87C48
            X = target - centroids
            dist_to_cent = torch.einsum("ij,ij->i", X, X)
        else:
            dist_to_cent = row_norms(target - centroids, squared=True)

        self.source_ = source
        self.target_ = target
        self.target_centroids_ = centroids
        self.target_dist_to_centroids_ = dist_to_cent
        return self

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query,
    ) -> Tuple[T, T]:
        """Transform distance between test and training data with DisSimLocal.

        Parameters
        ----------
        neigh_dist: shape (n_query, n_neighbors)
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).
        neigh_ind: shape (n_query, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist
        query: shape (n_query, n_features)
            Query entities that were used to obtain neighbors
            If none is provided use source that was provided in fit step

        Returns
        -------
        hub_reduced_dist, neigh_ind
            DisSimLocal distances, and corresponding neighbor indices

        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        """
        check_is_fitted(
            self,
            ["target_", "target_centroids_", "target_dist_to_centroids_"],
        )
        # Calculate local neighborhood centroids for source objects among target objects
        if self._use_torch:
            # pairwise squared euclidean distance between each query vector and knn
            # unsqueeze to enable batching
            hub_reduced_dist = (
                torch.cdist(torch.unsqueeze(query, 1), self.target_[neigh_ind])
                .pow(2)
                .squeeze()
            )
        else:
            hub_reduced_dist = np.empty_like(neigh_dist)
            for i, ind in enumerate(neigh_ind):
                hub_reduced_dist[i, :] = euclidean_distances(
                    query[i].reshape(1, -1), self.target_[ind], squared=True
                )

        centroids = self.target_[neigh_ind].mean(axis=1)
        source_minus_centroids = query - centroids
        source_minus_centroids **= 2
        source_dist_to_centroids = source_minus_centroids.sum(axis=1)
        target_dist_to_centroids = self.target_dist_to_centroids_[neigh_ind]

        hub_reduced_dist -= source_dist_to_centroids.reshape(-1, 1)
        hub_reduced_dist -= target_dist_to_centroids

        # DisSimLocal can yield negative dissimilarities, which can cause problems with
        # certain scikit-learn routines (e.g. in metric='precomputed' usages).
        # We, therefore, shift dissimilarities to non-negative values, if necessary.
        min_dist = hub_reduced_dist.min()
        if min_dist < _MINIMUM_DIST:
            hub_reduced_dist += -min_dist

        # Return Euclidean or squared Euclidean distances?
        if not self.squared:
            hub_reduced_dist **= 1 / 2

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
