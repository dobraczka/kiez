# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
)
from tqdm.auto import tqdm

from .base import HubnessReduction


class MutualProximity(HubnessReduction):
    """Hubness reduction with Mutual Proximity.

    Uses the formula presented in [1]_.

    Parameters
    ----------
    method: 'normal' or 'empiric', default = 'normal'
        Model distance distribution with 'method'.
        - 'normal' or 'gaussi' model distance distributions with independent Gaussians (fast)
        - 'empiric' or 'exact' model distances with the empiric distributions (slow)
    verbose: int, default = 0
        If verbose > 0, show progress bar.
    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """

    def __init__(self, method: str = "normal", verbose: int = 0, **kwargs):
        super().__init__(**kwargs)
        if method not in ["exact", "empiric", "normal", "gaussi"]:
            raise ValueError(
                f'Mutual proximity method "{method}" not recognized. Try "normal"'
                ' or "empiric".'
            )
        if method in ["exact", "empiric"]:
            self.method = "empiric"
        elif method in ["normal", "gaussi"]:
            self.method = "normal"
        self.verbose = verbose

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(method = {self.method}, verbose ="
            f" {self.verbose})"
        )

    def fit(
        self, neigh_dist, neigh_ind, source, target, assume_sorted=None
    ) -> MutualProximity:
        """Fit the model using neigh_dist and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (columns).
        neigh_ind: np.ndarray, shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.
        source
            Ignored
        target
            Ignored
        assume_sorted
            Ignored
        Returns
        ------
        MutualProximity

        Raises
        ------
        ValueError
            If self.method is unknown
        """
        # Check equal number of rows and columns
        check_consistent_length(neigh_ind, neigh_dist)
        check_consistent_length(neigh_ind.T, neigh_dist.T)
        check_array(neigh_dist, force_all_finite=False)
        check_array(neigh_ind)

        self.n_train = neigh_dist.shape[0]

        if self.method not in ["normal", "empiric"]:
            raise ValueError(f"Internal: Invalid method {self.method}.")
        elif self.method == "empiric":
            self.neigh_dist_t_to_s_ = neigh_dist
            self.neigh_ind_t_to_s_ = neigh_ind
        elif self.method == "normal":
            self.mu_t_to_s_ = np.nanmean(neigh_dist, axis=1)
            self.sd_t_to_s_ = np.nanstd(neigh_dist, axis=1, ddof=0)
        return self

    def transform(self, neigh_dist, neigh_ind, query, assume_sorted=None):
        """Transform distance between test and training data with Mutual Proximity.

        Parameters
        ----------
        neigh_dist: np.ndarray
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).
        neigh_ind: np.ndarray
            Neighbor indices corresponding to the values in neigh_dist
        query
            Ignored
        assume_sorted
            Ignored
        Returns
        -------
        hub_reduced_dist, neigh_ind
            Mutual Proximity distances, and corresponding neighbor indices
        Raises
        ------
        ValueError
            if self.method is unknown
        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        """
        check_is_fitted(
            self,
            [
                "mu_t_to_s_",
                "sd_t_to_s_",
                "neigh_dist_t_to_s_",
                "neigh_ind_t_to_s_",
            ],
            all_or_any=any,
        )
        check_array(neigh_dist, force_all_finite="allow-nan")
        check_array(neigh_ind)

        n_test, n_indexed = neigh_dist.shape

        if n_indexed == 1:
            warnings.warn(
                "Cannot perform hubness reduction with a single neighbor per query. "
                "Skipping hubness reduction, and returning untransformed distances."
            )
            return neigh_dist, neigh_ind

        hub_reduced_dist = np.empty_like(neigh_dist)

        # Show progress in hubness reduction loop
        disable_tqdm = not self.verbose
        range_n_test = tqdm(
            range(n_test),
            desc=f"MP ({self.method})",
            disable=disable_tqdm,
        )

        # Calculate MP with independent Gaussians
        if self.method not in ["normal", "empiric"]:
            raise ValueError(f"Internal: Invalid method {self.method}.")
        elif self.method == "normal":
            mu_t_to_s = self.mu_t_to_s_
            sd_t_to_s_ = self.sd_t_to_s_
            for i in range_n_test:
                j_mom = neigh_ind[i]
                mu = np.nanmean(neigh_dist[i])
                sd = np.nanstd(neigh_dist[i], ddof=0)
                p1 = stats.norm.sf(neigh_dist[i, :], mu, sd)
                p2 = stats.norm.sf(
                    neigh_dist[i, :], mu_t_to_s[j_mom], sd_t_to_s_[j_mom]
                )
                hub_reduced_dist[i, :] = (1 - p1 * p2).ravel()
        # Calculate MP empiric (slow)
        elif self.method == "empiric":
            max_ind = max(self.neigh_ind_t_to_s_.max(), neigh_ind.max())
            for i in range_n_test:
                d_i = neigh_dist[i, :][np.newaxis, :]  # broadcasted afterwards
                d_j = np.zeros((d_i.size, n_indexed))
                for j in range(n_indexed):
                    tmp = np.zeros(max_ind + 1) + (
                        self.neigh_dist_t_to_s_[neigh_ind[i, j], -1] + 1e-6
                    )
                    tmp[
                        self.neigh_ind_t_to_s_[neigh_ind[i, j]]
                    ] = self.neigh_dist_t_to_s_[neigh_ind[i, j]]
                    d_j[j, :] = tmp[neigh_ind[i]]
                d = d_i.T
                hub_reduced_dist[i, :] = 1.0 - (
                    np.sum((d_i > d) & (d_j > d), axis=1) / n_indexed
                )

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
