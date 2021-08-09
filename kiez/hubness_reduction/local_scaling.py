# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from __future__ import annotations

import warnings

import numpy as np
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from tqdm.auto import tqdm

from .base import HubnessReduction


class LocalScaling(HubnessReduction):
    """Hubness reduction with Local Scaling.

    Uses the formula presented in [1]_.

    Parameters
    ----------
    k: int, default = 5
        Number of neighbors to consider for the rescaling
    method: 'standard' or 'nicdm', default = 'standard'
        Perform local scaling with the specified variant:
        - 'standard' or 'ls' rescale distances using the distance to the k-th neighbor
        - 'nicdm' rescales distances using a statistic over distances to k neighbors
    verbose: int, default = 0
        If verbose > 0, show progress bar.
    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """

    def __init__(
        self, k: int = 5, method: str = "standard", verbose: int = 0, **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.method = method.lower()
        if self.method not in ["ls", "standard", "nicdm"]:
            raise ValueError(
                f"Internal: Invalid method {self.method}. Try 'ls' or 'nicdm'."
            )
        self.verbose = verbose

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(k={self.k}, method = {self.method}, verbose ="
            f" {self.verbose})"
        )

    def fit(
        self,
        neigh_dist,
        neigh_ind,
        source,
        target,
        assume_sorted: bool = True,
        *args,
        **kwargs,
    ) -> LocalScaling:
        """Fit the model using neigh_dist and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (colums).
        neigh_ind: np.ndarray, shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.
        source
            Ignored
        target
            Ignored
        assume_sorted: bool, default = True #noqa: DAR103
            Assume input matrices are sorted according to neigh_dist.
            If False, these are sorted here.
        *args
            Ignored
        **kwargs
            Ignored
        Returns
        -------
        LocalScaling
            Fitted LocalScaling
        """
        # Check equal number of rows and columns
        check_consistent_length(neigh_ind, neigh_dist)
        check_consistent_length(neigh_ind.T, neigh_dist.T)

        # increment to include the k-th element in slicing
        k = self.k + 1

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        if assume_sorted:
            self.r_dist_t_to_s_ = neigh_dist[:, :k]
            self.r_ind_t_to_s_ = neigh_ind[:, :k]
        else:
            kth = np.arange(self.k)
            mask = np.argpartition(neigh_dist, kth=kth)[:, :k]
            self.r_dist_t_to_s_ = np.take_along_axis(neigh_dist, mask, axis=1)
            self.r_ind_t_to_s_ = np.take_along_axis(neigh_ind, mask, axis=1)
        return self

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        source=None,
        assume_sorted: bool = True,
    ) -> (np.ndarray, np.ndarray):
        """Transform distance between test and training data with Mutual Proximity.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_query, n_neighbors)
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).
        neigh_ind: np.ndarray, shape (n_query, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist
        source
            Ignored
        assume_sorted: bool, default = True #noqa: DAR103
            Assume input matrices are sorted according to neigh_dist.
            If False, these are partitioned here.
            NOTE: The returned matrices are never sorted.
        Returns
        -------
        hub_reduced_dist, neigh_ind
            Local scaling distances, and corresponding neighbor indices
        Raises
        ------
        ValueError
            If wrong self.method was supplied
        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        """
        check_is_fitted(self, "r_dist_t_to_s_")

        n_test, n_indexed = neigh_dist.shape

        if n_indexed == 1:
            warnings.warn(
                "Cannot perform hubness reduction with a single neighbor per query. "
                "Skipping hubness reduction, and returning untransformed distances."
            )
            return neigh_dist, neigh_ind

        # increment to include the k-th element in slicing
        k = self.k + 1

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        if assume_sorted:
            r_dist_s_to_t = neigh_dist[:, :k]
        else:
            kth = np.arange(self.k)
            mask = np.argpartition(neigh_dist, kth=kth)[:, :k]
            r_dist_s_to_t = np.take_along_axis(neigh_dist, mask, axis=1)

        # Calculate LS or NICDM
        hub_reduced_dist = np.empty_like(neigh_dist)

        # Optionally show progress of local scaling loop
        disable_tqdm = not self.verbose
        range_n_test = tqdm(
            range(n_test),
            desc=f"LS {self.method}",
            disable=disable_tqdm,
        )

        # Perform standard local scaling...
        if self.method not in ["ls", "standard", "nicdm"]:
            raise ValueError(
                f"Internal: Invalid method {self.method}. Try 'ls' or 'nicdm'."
            )
        elif self.method in ["ls", "standard"]:
            r_t_to_s = self.r_dist_t_to_s_[:, -1]
            r_s_to_t = r_dist_s_to_t[:, -1]
            for i in range_n_test:
                hub_reduced_dist[i, :] = 1.0 - np.exp(
                    -1 * neigh_dist[i] ** 2 / (r_s_to_t[i] * r_t_to_s[neigh_ind[i]])
                )
        # ...or use non-iterative contextual dissimilarity measure
        elif self.method == "nicdm":
            r_t_to_s = self.r_dist_t_to_s_.mean(axis=1)
            r_s_to_t = r_dist_s_to_t.mean(axis=1)
            for i in range_n_test:
                hub_reduced_dist[i, :] = neigh_dist[i] / np.sqrt(
                    (r_s_to_t[i] * r_t_to_s[neigh_ind[i]])
                )

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
