#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# adapted from skhubness: https://github.com/VarIr/scikit-hubness/
"""
Estimate hubness in datasets
"""


from __future__ import annotations

import logging
import warnings
from typing import Union

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

# Available hubness measures
VALID_HUBNESS_MEASURES = [
    "all",
    "all_but_gini",
    "k_skewness",
    "k_skewness_truncnorm",
    "atkinson",
    "gini",
    "robinhood",
    "antihubs",
    "antihub_occurrence",
    "hubs",
    "hub_occurrence",
    "groupie_ratio",
    "k_occurrence",
]


def _calc_skewness_truncnorm(k_occurrence: np.ndarray) -> float:
    """Hubness measure; corrected for non-negativity of k-occurrence.

    Hubness as skewness of truncated normal distribution
    estimated from k-occurrence histogram.

    Parameters
    ----------
    k_occurrence: ndarray
        Reverse nearest neighbor count for each object.
    Returns
    -------
    skew_truncnorm
    """
    clip_left = 0
    clip_right = np.iinfo(np.int64).max
    k_occurrence_mean = k_occurrence.mean()
    k_occurrence_std = k_occurrence.std(ddof=1)
    a = (clip_left - k_occurrence_mean) / k_occurrence_std
    b = (clip_right - k_occurrence_mean) / k_occurrence_std
    skew_truncnorm = stats.truncnorm(a, b).moment(3)
    return skew_truncnorm


def _calc_gini_index(
    k_occurrence: np.ndarray, limiting="memory", verbose: int = 0
) -> float:
    """Hubness measure; Gini index
    Parameters
    ----------
    k_occurrence: ndarray
        Reverse nearest neighbor count for each object.
    limiting: 'memory' or 'cpu'
        If 'cpu', use fast implementation with high memory usage,
        if 'memory', use slightly slower, but memory-efficient implementation,
        otherwise use naive implementation (slow, low memory usage)
    verbose: int
        control verbosity
    Returns
    -------
    gini_index
    """
    n = k_occurrence.size
    if limiting in ["memory", "space"]:
        numerator = 0
        for i in tqdm(range(n), disable=not verbose, desc="Gini"):
            numerator += np.sum(np.abs(k_occurrence[:] - k_occurrence[i]))
    elif limiting in ["time", "cpu"]:
        numerator = np.sum(
            np.abs(k_occurrence.reshape(1, -1) - k_occurrence.reshape(-1, 1))
        )
    else:  # slow naive implementation
        n = k_occurrence.size
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += np.abs(k_occurrence[i] - k_occurrence[j])
    denominator = 2 * n * np.sum(k_occurrence)
    return numerator / denominator


def _calc_robinhood_index(k_occurrence: np.ndarray) -> float:
    """Hubness measure; Robin hood/Hoover/Schutz index.

    Parameters
    ----------
    k_occurrence: ndarray
        Reverse nearest neighbor count for each object.
    Returns
    -------
    robinhood_index

    Notes
    -----
    The Robin Hood index was proposed in [1]_ and is especially suited
    for hubness estimation in large data sets. Additionally, it offers
    straight-forward interpretability by answering the question:
    What share of k-occurrence must be redistributed, so that all objects
    are equally often nearest neighbors to others?

    References
    ----------
    .. [1] `Feldbauer, R.; Leodolter, M.; Plant, C. & Flexer, A.
            Fast approximate hubness reduction for large high-dimensional data.
            IEEE International Conference of Big Knowledge (2018).`
    """
    numerator = 0.5 * float(np.sum(np.abs(k_occurrence - k_occurrence.mean())))
    denominator = float(np.sum(k_occurrence))
    return numerator / denominator


def _calc_atkinson_index(k_occurrence: np.ndarray, eps: float = 0.5) -> float:
    """Hubness measure; Atkinson index.

    Parameters
    ----------
    k_occurrence: ndarray
        Reverse nearest neighbor count for each object.
    eps: float, default = 0.5 # noqa: DAR103
        'Income' weight. Turns the index into a normative measure.
    Returns
    -------
    atkinson_index
    """
    if eps == 1:
        term = np.prod(k_occurrence) ** (1.0 / k_occurrence.size)
    else:
        term = np.mean(k_occurrence ** (1 - eps)) ** (1 / (1 - eps))
    return 1.0 - 1.0 / k_occurrence.mean() * term


def _calc_antihub_occurrence(k_occurrence: np.ndarray) -> (np.array, float):
    """Proportion of antihubs in data set.

    Antihubs are objects that are never among the nearest neighbors
    of other objects.

    Parameters
    ----------
    k_occurrence: ndarray
        Reverse nearest neighbor count for each object.
    Returns
    -------
    antihubs, antihub_occurrence
    """
    antihubs = np.argwhere(k_occurrence == 0).ravel()
    antihub_occurrence = antihubs.size / k_occurrence.size
    return antihubs, antihub_occurrence


def _calc_hub_occurrence(
    k: int, k_occurrence: np.ndarray, n_test: int, hub_size: float = 2
):
    """Proportion of nearest neighbor slots occupied by hubs.

    Parameters
    ----------
    k: int
        Specifies the number of nearest neighbors
    k_occurrence: ndarray
        Reverse nearest neighbor count for each object.
    n_test: int
        Number of queries (or objects in a test set)
    hub_size: float
        Factor to determine hubs
    Returns
    -------
    hubs, hub_occurrence
    """
    hubs = np.argwhere(k_occurrence >= hub_size * k).ravel()
    hub_occurrence = k_occurrence[hubs].sum() / k / n_test
    return hubs, hub_occurrence


def hubness_score(
    nn_ind: np.ndarray,
    target_samples: int,
    *,
    k: int = None,
    hub_size: float = 2.0,
    shuffle_equal: bool = True,
    random_state=None,
    verbose: int = 0,
    return_value: str = "all_but_gini",
    store_k_occurrence: bool = False,
) -> Union[float, dict]:
    """Calculates hubness scores from given neighbor indices

    Utilizes findings from [1]_ and [2]_.

    Parameters
    ----------
    nn_ind : np.ndarray
       Neighbor index matrix
    target_samples: int
        number of entities in the target space
    k : int
        number of k for k-nearest neighbor
    hub_size : float
        Hubs are defined as objects with k-occurrence > hub_size * k.
    shuffle_equal : bool
        If true shuffle neighbors with identical distances
        to avoid artifact hubness.
        NOTE: This is especially useful for secondary distance measures
        with a finite number of possible values
    random_state: int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int
        Level of output messages
    return_value : str
        Hubness measure to return
        By default, return all but gini,
        because gini is slow on large datasets
        Use "all" to return a dict of all available measures,
        or check `kiez.analysis.VALID_HUBNESS_MEASURE`
        for available measures.
    store_k_occurrence: bool
        Whether to save the k-occurrence. Requires O(n_test) memory.

    Returns
    -------
    hubness_measure: float or dict
            Return the hubness measure as indicated by `return_value`.
            if return_value is 'all',
            a dict of all hubness measures is returned.

    Raises
    ------
    ValueError
        If nn_ind has wrong type

    References
    ----------
    .. [1] `Radovanović, M.; Nanopoulos, A. & Ivanovic, M.
            Hubs in space: Popular nearest neighbors in high-dimensional data.
            Journal of Machine Learning Research, 2010, 11, 2487-2531`
    .. [2] `Feldbauer, R.; Leodolter, M.; Plant, C. & Flexer, A.
            Fast approximate hubness reduction for large high-dimensional data.
            IEEE International Conference of Big Knowledge (2018).`

    Examples
    --------
    >>> from kiez import Kiez
    >>> from kiez.analysis import hubness_score
    >>> import numpy as np
    >>> # create example data
    >>> rng = np.random.RandomState(0)
    >>> source = rng.rand(100,50)
    >>> target = rng.rand(100,50)
    >>> # fit and get neighbors
    >>> k_inst = Kiez()
    >>> k_inst.fit(source, target)
    >>> nn_ind = k_inst.kneighbors(return_distance=False)
    >>> # get hubness
    >>> hubness_score(nn_ind, target.shape[1])
        {
            "k_skewness": 1.0243818877407802,
            "k_skewness_truncnorm": 0.705309555084711,
            "atkinson": 0.1846908928840305,
            "robinhood": 0.31,
            "antihubs": array([14, 34, 37, 45, 54, 57, 67, 74]),
            "antihub_occurrence": 0.08,
            "hubs": array([31, 39, 46, 56, 62, 66, 68, 70]),
            "hub_occurrence": 0.436,
            "groupie_ratio": 0.076,
        }
    IGNORE:
    # noqa: DAR002
    IGNORE
    """
    n_train = nn_ind.shape[0]
    n_test = target_samples
    k_neighbors = nn_ind.copy()
    if k is None:
        k = nn_ind.shape[1]
    else:
        if k < k_neighbors.shape[1]:
            k_neighbors = k_neighbors[:, :k]
        elif k > k_neighbors.shape[1]:
            k = nn_ind.shape[1]
            warnings.warn(f"k > nn_ind.shape[1], k will be set to {k}")

    # Negative indices can occur, when ANN does not find enough neighbors,
    # and must be removed
    mask = k_neighbors < 0
    if np.any(mask):
        k_neighbors = k_neighbors[~mask]
        del mask

    try:
        k_occurrence = np.bincount(k_neighbors.astype(int).ravel(), minlength=n_train)
    except ValueError as e:
        logging.info(f"k_occurence failed with the following neighbors: {k_neighbors}")
        raise e

    # traditional skewness measure
    k_skewness = stats.skew(k_occurrence)

    # new skewness measure (truncated normal distribution)
    k_skewness_truncnorm = _calc_skewness_truncnorm(k_occurrence)

    # Gini index
    if return_value in ["gini", "all"]:
        limiting = "space" if k_occurrence.shape[0] > 10_000 else "time"
        gini_index = _calc_gini_index(k_occurrence, limiting, verbose=verbose)
    else:
        gini_index = np.nan

    # Robin Hood index
    robinhood_index = _calc_robinhood_index(k_occurrence)

    # Atkinson index
    atkinson_index = _calc_atkinson_index(k_occurrence)

    # anti-hub occurrence
    antihubs, antihub_occurrence = _calc_antihub_occurrence(k_occurrence)

    # hub occurrence
    hubs, hub_occurrence = _calc_hub_occurrence(
        k=k,
        k_occurrence=k_occurrence,
        n_test=n_test,
        hub_size=hub_size,
    )

    # Largest hub
    groupie_ratio = k_occurrence.max() / n_test / k

    # Dictionary of all hubness measures
    hubness_measures = {
        "k_skewness": k_skewness,
        "k_skewness_truncnorm": k_skewness_truncnorm,
        "atkinson": atkinson_index,
        "gini": gini_index,
        "robinhood": robinhood_index,
        "antihubs": antihubs,
        "antihub_occurrence": antihub_occurrence,
        "hubs": hubs,
        "hub_occurrence": hub_occurrence,
        "groupie_ratio": groupie_ratio,
    }

    if store_k_occurrence:
        hubness_measures["k_occurrence"] = k_occurrence
    if return_value == "all":
        return hubness_measures
    elif return_value == "all_but_gini":
        del hubness_measures["gini"]
        return hubness_measures
    else:
        return hubness_measures[return_value]
