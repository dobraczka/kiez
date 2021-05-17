# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Tom Dupre la Tour (original work)
#         Roman Feldbauer (adaptions for scikit-hubness)
# PEP 563: Postponed Evaluation of Annotations

from __future__ import annotations

import logging
from typing import Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm.auto import tqdm

from ..io.temp_file_handling import create_tempfile_preferably_in_dir
from ..utils.check import check_n_candidates
from .approximate_neighbors import ApproximateNearestNeighbor

try:
    import annoy  # noqa: autoimport

except ImportError:
    annoy = None  # pragma: no cover

__all__ = [
    "RandomProjectionTree",
]


class RandomProjectionTree(BaseEstimator, ApproximateNearestNeighbor):
    """Wrapper for using annoy.AnnoyIndex

    Annoy is an approximate nearest neighbor library,
    that builds a forest of random projections trees.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    n_trees: int, default = 10
        Build a forest of n_trees trees. More trees gives higher precision when querying,
        but are more expensive in terms of build time and index size.
    search_k: int, default = -1
        Query will inspect search_k nodes. A larger value will give more accurate results,
        but will take longer time.
    mmap_dir: str, default = 'auto'
        Memory-map the index to the given directory.
        This is required to make the the class pickleable.
        If None, keep everything in main memory (NON pickleable index),
        if mmap_dir is a string, it is interpreted as a directory to store the index into,
        if 'auto', create a temp dir for the index, preferably in /dev/shm on Linux.
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """

    valid_metrics = [
        "angular",
        "euclidean",
        "manhattan",
        "hamming",
        "dot",
        "minkowski",
    ]

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "euclidean",
        n_trees: int = 10,
        search_k: int = -1,
        mmap_dir: str = "auto",
        n_jobs: int = 1,
        verbose: int = 0,
    ):

        if annoy is None:  # pragma: no cover
            raise ImportError(
                "Please install the `annoy` package, before using this class.\n"
                "$ pip install annoy"
            ) from None

        super().__init__(
            n_candidates=n_candidates,
            metric=metric,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.n_trees = n_trees
        self.search_k = search_k
        self.mmap_dir = mmap_dir

    def fit(self, data, y_ignored=None) -> RandomProjectionTree:
        """Build the annoy.Index and insert data.

        Parameters
        ----------
        data: np.array
            Data to be indexed
        y_ignored: any
            Ignored

        Returns
        -------
        self: RandomProjectionTree
            An instance of RandomProjectionTree with a built index
        """
        if y_ignored is None:
            data = check_array(data)

        self.n_samples_fit_ = data.shape[0]
        self.n_features_ = data.shape[1]
        self.data_dtype_ = data.dtype
        if self.metric == "minkowski":  # for compatibility
            self.metric = "euclidean"
        metric = self.metric if self.metric != "sqeuclidean" else "euclidean"
        self.effective_metric_ = metric
        annoy_index = annoy.AnnoyIndex(data.shape[1], metric=metric)
        if self.mmap_dir == "auto":
            self.annoy_ = create_tempfile_preferably_in_dir(
                prefix="skhubness_", suffix=".annoy", directory="/dev/shm"
            )
            logging.warning(
                f"The index will be stored in {self.annoy_}. "
                f"It will NOT be deleted automatically, when this instance is destructed."
            )
        elif isinstance(self.mmap_dir, str):
            self.annoy_ = create_tempfile_preferably_in_dir(
                prefix="skhubness_", suffix=".annoy", directory=self.mmap_dir
            )
        else:  # e.g. None
            self.mmap_dir = None

        for i, x in tqdm(
            enumerate(data),
            desc="Build RPtree",
            disable=not self.verbose,
        ):
            annoy_index.add_item(i, x.tolist())
        annoy_index.build(self.n_trees)

        if self.mmap_dir is None:
            self.annoy_ = annoy_index
        else:
            annoy_index.save(
                self.annoy_,
            )

        return self

    def kneighbors(
        self, query=None, n_candidates=None, return_distance=True
    ) -> Union[Tuple[np.array, np.array], np.array]:
        """Retrieve k nearest neighbors.

        Parameters
        ----------
        query: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        n_candidates: int or None, optional, default = None
            Number of neighbors to retrieve.
            If None, use the value passed during construction.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.
        """
        check_is_fitted(self, "annoy_")
        if query is not None:
            query = check_array(query)

        n_test = self.n_samples_fit_ if query is None else query.shape[0]
        dtype = self.data_dtype_ if query is None else query.dtype

        if n_candidates is None:
            n_candidates = self.n_candidates
        n_candidates = check_n_candidates(n_candidates)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        if query is None:
            n_neighbors = n_candidates + 1
            start = 1
        else:
            n_neighbors = n_candidates
            start = 0

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        neigh_ind = -np.ones((n_test, n_candidates), dtype=np.int32)
        neigh_dist = np.empty_like(neigh_ind, dtype=dtype) * np.nan

        # Load memory-mapped annoy.Index, unless it's already in main memory
        if isinstance(self.annoy_, str):
            annoy_index = annoy.AnnoyIndex(
                self.n_features_, metric=self.effective_metric_
            )
            annoy_index.load(self.annoy_)
        elif isinstance(self.annoy_, annoy.AnnoyIndex):
            annoy_index = self.annoy_
        assert isinstance(
            annoy_index, annoy.AnnoyIndex
        ), "Internal error: unexpected type for annoy index"

        disable_tqdm = not self.verbose
        if query is None:
            n_items = annoy_index.get_n_items()

            for i in tqdm(
                range(n_items),
                desc="Query RPtree",
                disable=disable_tqdm,
            ):
                ind, dist = annoy_index.get_nns_by_item(
                    i,
                    n_neighbors,
                    self.search_k,
                    include_distances=True,
                )
                ind = ind[start:]
                dist = dist[start:]
                neigh_ind[i, : len(ind)] = ind
                neigh_dist[i, : len(dist)] = dist
        else:  # if query was provided
            for i, x in tqdm(
                enumerate(query),
                desc="Query RPtree",
                disable=disable_tqdm,
            ):
                ind, dist = annoy_index.get_nns_by_vector(
                    x.tolist(),
                    n_neighbors,
                    self.search_k,
                    include_distances=True,
                )
                ind = ind[start:]
                dist = dist[start:]
                neigh_ind[i, : len(ind)] = ind
                neigh_dist[i, : len(dist)] = dist

        if self.metric == "sqeuclidean":
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
