# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer (adaptions for scikit-hubness)
#         Daniel Obraczka (adaptions for kiez)
# PEP 563: Postponed Evaluation of Annotations

from __future__ import annotations

import logging
import pathlib

import numpy as np
from kiez.io.temp_file_handling import create_tempfile_preferably_in_dir
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithmWithJoblib
from tqdm.auto import tqdm

try:
    import ngtpy  # noqa: autoimport

except ImportError:
    ngtpy = None  # pragma: no cover


__all__ = [
    "NNG",
]


class NNG(NNAlgorithmWithJoblib):
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
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs)
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
        self.verbose = verbose
        self.index_dir = index_dir
        self._index_dir_plausibility_check()
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

    def _fit(self, data, is_source: bool):
        if is_source:
            prefix = "kiez_source"
        else:
            prefix = "kiez_target"

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

        # Convert ANNG to ONNG
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
