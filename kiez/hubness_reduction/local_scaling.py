# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/

from typing import Tuple, TypeVar

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import HubnessReduction

T = TypeVar("T")

try:
    import torch
except ImportError:
    torch = None


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
           Learning Research, 13(1), 2871-2902.
    """

    def __init__(self, method: str = "standard", **kwargs):
        super().__init__(**kwargs)
        self.method = method.lower()
        if self.method not in ["ls", "standard", "nicdm"]:
            raise ValueError(
                f"Internal: Invalid method {self.method}. Try 'ls' or 'nicdm'."
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(method = {self.method}, verbose ="
            f" {self.verbose})"
        )

    def _fit(
        self,
        neigh_dist,
        neigh_ind,
        source,
        target,
    ) -> "LocalScaling":
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

        Returns
        -------
        LocalScaling
            Fitted LocalScaling
        """
        self.r_dist_t_to_s_ = neigh_dist
        self.r_ind_t_to_s_ = neigh_ind
        return self

    def _exp(self, inner_exp):
        if self._use_torch:
            return torch.exp(inner_exp)
        return np.exp(inner_exp)

    def _sqrt(self, value):
        if self._use_torch:
            return torch.sqrt(value)
        return np.sqrt(value)

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query=None,
    ) -> Tuple[T, T]:
        """Transform distance between test and training data with Mutual Proximity.

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

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        r_dist_s_to_t = neigh_dist

        # Perform standard local scaling...
        if self.method in ["ls", "standard"]:
            r_t_to_s = self.r_dist_t_to_s_[:, -1]
            r_s_to_t = r_dist_s_to_t[:, -1].reshape(-1, 1)
            inner_exp = -1 * neigh_dist**2 / (r_s_to_t * r_t_to_s[neigh_ind])
            exp = self._exp(inner_exp)
            hub_reduced_dist = 1.0 - exp
        # ...or use non-iterative contextual dissimilarity measure
        elif self.method == "nicdm":
            r_t_to_s = self.r_dist_t_to_s_.mean(axis=1)
            r_s_to_t = r_dist_s_to_t.mean(axis=1).reshape(-1, 1)
            inner_sqrt = r_s_to_t * r_t_to_s[neigh_ind]
            sqrt = self._sqrt(inner_sqrt)
            hub_reduced_dist = neigh_dist / sqrt

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
