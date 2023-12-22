from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from .base import HubnessReduction


class CSLS(HubnessReduction):
    """Hubness reduction with Cross-domain similarity local scaling.

    Uses the formula presented in [1]_.

    References
    ----------
    .. [1] Lample, G., Conneau, A., Ranzato, M., Denoyer, L., & Jégou, H. (2018)
           Word translation without parallel data
           In: 6th International Conference on Learning Representations,
           ICLR 2018 - Conference Track Proceedings.
           https://openreview.net/forum?id=H196sainb
    """

    def __repr__(self):
        return f"{self.__class__.__name__}(verbose = {self.verbose})"

    def _fit(
        self,
        neigh_dist,
        neigh_ind,
        source=None,
        target=None,
    ) -> CSLS:
        """Fit the model using target, neigh_dist, and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (colums).
        neigh_ind: np.ndarray, shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.
        source
            ignored
        target
            ignored

        Returns
        -------
        CSLS
            Fitted CSLS
        """
        self.r_dist_train_ = neigh_dist
        self.r_ind_train_ = neigh_ind
        return self

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform distance between test and training data with CSLS.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_query, n_neighbors)
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).
        neigh_ind: np.ndarray, shape (n_query, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist
        query
            Ignored

        Returns
        -------
        hub_reduced_dist, neigh_ind
            CSLS distances, and corresponding neighbor indices

        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        """
        check_is_fitted(self, "r_dist_train_")

        n_test, n_indexed = neigh_dist.shape

        # Find average distances to the k nearest neighbors
        r_dist_test = neigh_dist

        hub_reduced_dist = np.empty_like(neigh_dist)

        # Optionally show progress of local scaling loop
        disable_tqdm = not self.verbose
        range_n_test = tqdm(
            range(n_test),
            desc="CSLS",
            disable=disable_tqdm,
        )

        r_train = self.r_dist_train_.mean(axis=1)
        r_test = r_dist_test.mean(axis=1)
        for i in range_n_test:
            hub_reduced_dist[i, :] = (
                2 * neigh_dist[i] - r_test[i] - r_train[neigh_ind[i]]
            )
        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
