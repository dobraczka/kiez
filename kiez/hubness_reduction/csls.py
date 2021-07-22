from __future__ import annotations

import warnings

import numpy as np
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from tqdm.auto import tqdm

from .base import HubnessReduction


class CSLS(HubnessReduction):
    """Hubness reduction with Cross-domain similarity local scaling.

    Uses the formula presented in [1]_.

    Parameters
    ----------
    k: int, default = 5
        Number of neighbors to consider for mean distance of k-nearest neighbors
    verbose: int, default= 0
        Verbosity level

    References
    ----------
    .. [1] Lample, G., Conneau, A., Ranzato, M., Denoyer, L., & Jégou, H. (2018)
           Word translation without parallel data
           In: 6th International Conference on Learning Representations,
           ICLR 2018 - Conference Track Proceedings.
           https://openreview.net/forum?id=H196sainb
    """

    def __init__(self, k: int = 5, verbose: int = 0, *args, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.verbose = verbose

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k}, verbose = {self.verbose})"

    def fit(
        self,
        neigh_dist,
        neigh_ind,
        source=None,
        target=None,
        assume_sorted=None,
        *args,
        **kwargs,
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
        assume_sorted: bool, default=True #noqa: DAR103
            Assume input matrices are sorted according to neigh_dist.
            If False, these are sorted here.
        *args
            Ignored
        **kwargs
            Ignored
        Returns
        -------
        CSLS
            Fitted CSLS
        Raises
        ------
        ValueError
            If self.k < 0
        TypeError
            If self.k not int
        """
        # Check equal number of rows and columns
        check_consistent_length(neigh_ind, neigh_dist)
        check_consistent_length(neigh_ind.T, neigh_dist.T)
        try:
            if self.k <= 0:
                raise ValueError(f"Expected k > 0. Got {self.k}")
        except TypeError:
            raise TypeError(f"Expected k: int > 0. Got {self.k}")

        # increment to include the k-th element in slicing
        k = self.k + 1

        if assume_sorted:
            self.r_dist_train_ = neigh_dist[:, :k]
            self.r_ind_train_ = neigh_ind[:, :k]
        else:
            kth = np.arange(self.k)
            mask = np.argpartition(neigh_dist, kth=kth)[:, :k]
            self.r_dist_train_ = np.take_along_axis(neigh_dist, mask, axis=1)
            self.r_ind_train_ = np.take_along_axis(neigh_ind, mask, axis=1)
        return self

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query,
        assume_sorted: bool = True,
        *args,
        **kwargs,
    ) -> (np.ndarray, np.ndarray):
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
        assume_sorted: bool
            ignored
        *args
            Ignored
        **kwargs
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

        if n_indexed == 1:
            warnings.warn(
                "Cannot perform hubness reduction with a single neighbor per query. "
                "Skipping hubness reduction, and returning untransformed distances."
            )
            return neigh_dist, neigh_ind

        k = self.k

        # Find average distances to the k nearest neighbors
        if assume_sorted:
            r_dist_test = neigh_dist[:, :k]
        else:
            kth = np.arange(self.k)
            mask = np.argpartition(neigh_dist, kth=kth)[:, :k]
            r_dist_test = np.take_along_axis(neigh_dist, mask, axis=1)

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
