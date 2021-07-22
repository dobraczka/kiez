# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer (adaptions for scikit-hubness)
#         Daniel Obraczka (adaptions for kiez)
# PEP 563: Postponed Evaluation of Annotations

from __future__ import annotations

import logging

import numpy as np
from kiez.io.temp_file_handling import create_tempfile_preferably_in_dir
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithmWithJoblib
from tqdm.auto import tqdm

try:
    import ngtpy  # noqa: autoimport

except ImportError:  # pragma: no cover
    ngtpy = None


__all__ = [
    "NNG",
]


class NNG(NNAlgorithmWithJoblib):
    """
    Wrapper for NGT's graph based approximate nearest neighbor search

    Parameters
    ----------
    n_candidates: int
        number of nearest neighbors used in search
    metric: str, default = 'euclidean'
        distance measure used in search
        possible measures are found in :obj:`NNG.valid_metrics`
    index_dir: str default = 'auto'
        Store the index in the given directory.
        If None, keep the index in main memory (NON pickleable index),
        If index_dir is a string, it is interpreted as a directory to store the index into,
        if 'auto', create a temp dir for the index, preferably in /dev/shm on Linux.
        Note: The directory/the index will NOT be deleted automatically.
    edge_size_for_creation: int, default = 80
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    edge_size_for_search: int, default = 40
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    epsilon: float, default 0.1
        Trade-off in ANNG between higher accuracy (larger epsilon) and shorter query time (smaller epsilon)
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Notes
    -----
    See the NGT documentation for more details: https://github.com/yahoojapan/NGT/blob/master/python/README-ngtpy.md

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
    _internal_distance_type = {
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
        edge_size_for_creation: int = 80,
        edge_size_for_search: int = 40,
        epsilon: float = 0.1,
        n_jobs: int = 1,
        verbose: int = 0,
    ):

        if ngtpy is None:  # pragma: no cover
            raise ImportError(
                "Please install the `ngt` package, before using this class.\n"
                "$ pip3 install ngt"
            ) from None
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs)
        # Map common distance names to names used by ngt
        try:
            self.effective_metric_ = NNG._internal_distance_type[self.metric]
        except KeyError:
            self.effective_metric_ = self.metric
        if self.effective_metric_ not in NNG.valid_metrics:
            raise ValueError(
                f"Unknown distance/similarity measure: {self.effective_metric_}. "
                f"Please use one of: {NNG.valid_metrics}."
            )
        self.verbose = verbose
        self.index_dir = index_dir
        self._index_dir_plausibility_check()
        self.edge_size_for_creation = edge_size_for_creation
        self.edge_size_for_search = edge_size_for_search
        self.epsilon = epsilon
        self.index_path_source = None
        self.index_path_target = None

    def __repr__(self):
        ret_str = (
            f"{self.__class__.__name__}(n_candidates={self.n_candidates},"
            + f"index_dir = {self.index_dir},"
            + f"edge_size_for_creation = {self.edge_size_for_creation},"
            + f"edge_size_for_search = {self.edge_size_for_search},"
            + f"epsilon = {self.epsilon},"
            + f"n_jobs = {self.n_jobs},"
            + f"verbose = {self.verbose}"
        )
        if self.index_path_source is not None:
            ret_str += (
                f" source index path={self.index_path_source} and target index"
                f" path={self.index_path_target}"
            )
        if self.metric != self.effective_metric_:
            return ret_str + f" and effective algo is {self.effective_metric_}"
        ret_str += ")"
        return ret_str

    def _index_dir_plausibility_check(self):
        if not (self.index_dir is None or isinstance(self.index_dir, str)):
            raise TypeError(
                "NNG requires to write an index to the filesystem. "
                "Please provide a valid path with parameter `index_dir`."
            )

    def _fit(self, data, is_source: bool):
        if is_source:
            prefix = "kiez_source"
        else:
            prefix = "kiez_target"

        index_path = None
        # Set up a directory to save the index to
        suffix = ".anng"
        if self.index_dir in ["auto"]:
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix, directory="/dev/shm"
            )
            logging.warning(
                f"The index will be stored in {index_path}. It will NOT be deleted"
                " automatically, when this instance is destructed."
            )
        elif isinstance(self.index_dir, str):
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix, directory=self.index_dir
            )
        elif self.index_dir is None:
            index_path = create_tempfile_preferably_in_dir(prefix=prefix, suffix=suffix)

        # Create the ANNG index, insert data
        ngtpy.create(
            path=index_path,
            dimension=data.shape[1],
            edge_size_for_creation=self.edge_size_for_creation,
            edge_size_for_search=self.edge_size_for_search,
            distance_type=self.effective_metric_,
        )
        index_obj = ngtpy.Index(index_path)
        index_obj.batch_insert(data, num_threads=self.n_jobs)
        index_obj.save()

        if index_path is not None:
            if is_source:
                self.index_path_source = index_path
            else:
                self.index_path_target = index_path

        # Keep index in memory or store in path
        if self.index_dir is None:
            return index_obj
        return index_path

    def _kneighbors_part(self, k, query, index, return_distance, is_self_querying):
        index = (
            ngtpy.Index(index) if isinstance(index, str) else index  # load if is path
        )
        n_query = query.shape[0]
        query_dtype = query.dtype

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        if is_self_querying:
            k = k + 1
            start = 1
        else:
            start = 0

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        neigh_ind = -np.ones((n_query, k), dtype=np.int32)
        if return_distance:
            neigh_dist = np.empty_like(neigh_ind, dtype=query_dtype) * np.nan

        disable_tqdm = not self.verbose
        if is_self_querying:
            for i in tqdm(
                range(n_query),
                desc="Query NNG",
                disable=disable_tqdm,
            ):
                query = index.get_object(i)
                response = index.search(
                    query=query,
                    size=k,
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
                    size=k,
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
