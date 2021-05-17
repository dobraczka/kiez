# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer (adaptions for scikit-hubness)
# PEP 563: Postponed Evaluation of Annotations

from __future__ import annotations

import logging
import pathlib
from typing import Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm.auto import tqdm

from ..io.temp_file_handling import create_tempfile_preferably_in_dir
from ..utils.check import check_n_candidates
from .approximate_neighbors import ApproximateNearestNeighbor

try:
    import ngtpy  # noqa: autoimport

except ImportError:
    ngtpy = None  # pragma: no cover


__all__ = [
    "NNG",
]


class NNG(BaseEstimator, ApproximateNearestNeighbor):
    """Wrapper for ngtpy and NNG variants.

    By default, the graph is an ANNG. Only when the `optimize` parameter is set,
    the graph is optimized to obtain an ONNG.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are 'manhattan', 'L1', 'euclidean', 'L2', 'minkowski',
        'Angle', 'Normalized Angle', 'Hamming', 'Jaccard', 'Cosine' or 'Normalized Cosine'.
    index_dir: str, default = 'auto'
        Store the index in the given directory.
        If None, keep the index in main memory (NON pickleable index),
        If index_dir is a string, it is interpreted as a directory to store the index into,
        if 'auto', create a temp dir for the index, preferably in /dev/shm on Linux.
        Note: The directory/the index will NOT be deleted automatically.
    optimize: bool, default = False
        Use ONNG method by optimizing the ANNG graph.
        May require long time for index creation.
    edge_size_for_creation: int, default = 80
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    edge_size_for_search: int, default = 40
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    epsilon: float, default 0.1
        Trade-off in ANNG between higher accuracy (larger epsilon) and shorter query time (smaller epsilon)
    num_incoming: int
        Number of incoming edges in ONNG graph
    num_outgoing: int
        Number of outgoing edges in ONNG graph
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures

    Notes
    -----
    NNG stores the index to a directory specified in `index_dir`.
    The index is persistent, and will NOT be deleted automatically.
    It is the user's responsibility to take care of deletion,
    when required.
    """

    valid_metrics = [
        "manhattan",
        "L1",
        "euclidean",
        "L2",
        "minkowski",
        "sqeuclidean",
        "Angle",
        "Normalized Angle",
        "Cosine",
        "Normalized Cosine",
        "Hamming",
        "Jaccard",
    ]
    internal_distance_type = {
        "manhattan": "L1",
        "euclidean": "L2",
        "minkowski": "L2",
        "sqeuclidean": "L2",
    }

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "euclidean",
        index_dir: str = "auto",
        optimize: bool = False,
        edge_size_for_creation: int = 80,
        edge_size_for_search: int = 40,
        num_incoming: int = -1,
        num_outgoing: int = -1,
        epsilon: float = 0.1,
        n_jobs: int = 1,
        verbose: int = 0,
    ):

        if ngtpy is None:  # pragma: no cover
            raise ImportError(
                "Please install the `ngt` package, before using this class.\n"
                "$ pip3 install ngt"
            ) from None

        super().__init__(
            n_candidates=n_candidates,
            metric=metric,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.index_dir = index_dir
        self.optimize = optimize
        self.edge_size_for_creation = edge_size_for_creation
        self.edge_size_for_search = edge_size_for_search
        self.num_incoming = num_incoming
        self.num_outgoing = num_outgoing
        self.epsilon = epsilon

    def _index_dir_plausibility_check(self):
        if not (self.index_dir is None or isinstance(self.index_dir, str)):
            raise TypeError(
                "NNG requires to write an index to the filesystem. "
                "Please provide a valid path with parameter `index_dir`."
            )

    def fit(self, data, y_ignore=None) -> NNG:
        """Build the ngtpy.Index and insert data.

        Parameters
        ----------
        data: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: NNG
            An instance of NNG with a built index
        """
        if y_ignore is None:
            data = check_array(data)

        self.n_samples_fit_ = data.shape[0]
        self.n_features_ = data.shape[1]
        self.data_dtype_ = data.dtype

        # Map common distance names to names used by ngt
        try:
            self.effective_metric_ = NNG.internal_distance_type[self.metric]
        except KeyError:
            self.effective_metric_ = self.metric
        if self.effective_metric_ not in NNG.valid_metrics:
            raise ValueError(
                f"Unknown distance/similarity measure: {self.effective_metric_}. "
                f"Please use one of: {NNG.valid_metrics}."
            )

        # Set up a directory to save the index to
        prefix = "skhubness_"
        suffix = ".anng"
        self._index_dir_plausibility_check()
        if self.index_dir in ["auto"]:
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix, directory="/dev/shm"
            )
            logging.warning(
                f"The index will be stored in {index_path}. "
                f"It will NOT be deleted automatically, when this instance is destructed."
            )
        elif isinstance(self.index_dir, str):
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix, directory=self.index_dir
            )
        elif self.index_dir is None:
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix
            )

        # Create the ANNG index, insert data
        ngtpy.create(
            path=index_path,
            dimension=self.n_features_,
            edge_size_for_creation=self.edge_size_for_creation,
            edge_size_for_search=self.edge_size_for_search,
            distance_type=self.effective_metric_,
        )
        index_obj = ngtpy.Index(index_path)
        index_obj.batch_insert(data, num_threads=self.n_jobs)
        index_obj.save()

        # Convert ANNG top ONNG
        if self.optimize:
            optimizer = ngtpy.Optimizer()
            optimizer.set(
                num_of_outgoings=self.num_outgoing,
                num_of_incomings=self.num_incoming,
            )
            index_path_onng = str(pathlib.Path(index_path).with_suffix(".onng"))
            optimizer.execute(index_path, index_path_onng)
            index_path = index_path_onng

        # Keep index in memory or store in path
        if self.index_dir is None:
            self.index_ = index_obj
        else:
            # index_obj.save()
            self.index_ = index_path

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
        check_is_fitted(self, "index_")
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
        if return_distance:
            neigh_dist = np.empty_like(neigh_ind, dtype=dtype) * np.nan

        if isinstance(self.index_, str):
            index = ngtpy.Index(self.index_)
        else:
            index = self.index_

        disable_tqdm = not self.verbose
        if query is None:
            for i in tqdm(
                range(n_test),
                desc="Query NNG",
                disable=disable_tqdm,
            ):
                query = index.get_object(i)
                response = index.search(
                    query=query,
                    size=n_neighbors,
                    with_distance=return_distance,
                    epsilon=self.epsilon,
                )
                if return_distance:
                    ind, dist = [np.array(arr) for arr in zip(*response)]
                else:
                    ind = response
                ind = ind[start:]
                neigh_ind[i, : len(ind)] = ind
                if return_distance:
                    dist = dist[start:]
                    neigh_dist[i, : len(dist)] = dist
        else:  # if query was provided
            for i, x in tqdm(
                enumerate(query),
                desc="Query NNG",
                disable=disable_tqdm,
            ):
                response = index.search(
                    query=x,
                    size=n_neighbors,
                    with_distance=return_distance,
                    epsilon=self.epsilon,
                )
                if return_distance:
                    ind, dist = [np.array(arr) for arr in zip(*response)]
                else:
                    ind = response
                ind = ind[start:]
                neigh_ind[i, : len(ind)] = ind
                if return_distance:
                    dist = dist[start:]
                    neigh_dist[i, : len(dist)] = dist

        if return_distance and self.metric == "sqeuclidean":
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
