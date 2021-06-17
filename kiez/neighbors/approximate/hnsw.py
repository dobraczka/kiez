# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer (version for scikit-hubness)
#         Daniel Obraczka (adaptation for kiez)
# PEP 563: Postponed Evaluation of Annotations

from __future__ import annotations

import numpy as np
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm


class HNSW(NNAlgorithm):
    # put import here to avoid displaying nmslib info message
    # when importing kiez
    try:
        import nmslib
    except ImportError:
        nmslib = None  # pragma: no cover

    valid_metrics = [
        "euclidean",
        "l2",
        "minkowski",
        "squared_euclidean",
        "sqeuclidean",
        "cosine",
        "cosinesimil",
    ]

    def __init__(
        self,
        n_candidates: int = 5,
        metric: str = "euclidean",
        method: str = "hnsw",
        M: int = 16,  # noqa: N803
        post_processing: int = 2,
        ef_construction: int = 200,
        n_jobs: int = 1,
        verbose: int = 0,
    ):

        if HNSW.nmslib is None:  # pragma: no cover
            raise ImportError(
                "Please install the `nmslib` package, before using this class.\n"
                "$ pip install nmslib"
            )

        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs)
        self.verbose = verbose
        self.method = method
        self.M = M  # noqa: N803
        self.post_processing = post_processing
        self.ef_construction = ef_construction
        self.space = None

    def _fit(self, data, is_source: bool):
        method = self.method
        post_processing = self.post_processing
        big_m = self.M
        ef_construction = self.ef_construction

        if self.metric not in self.valid_metrics:
            raise ValueError(
                f'Invalid metric "{self.metric}". Please try "euclidean" or "cosine".'
            )
        elif self.metric in [
            "euclidean",
            "l2",
            "minkowski",
            "squared_euclidean",
            "sqeuclidean",
        ]:
            if self.metric in ["squared_euclidean", "sqeuclidean"]:
                self.metric = "sqeuclidean"
            else:
                self.metric = "euclidean"
            self.space = "l2"
        elif self.metric in ["cosine", "cosinesimil"]:
            self.space = "cosinesimil"
        assert self.space in [
            "l2",
            "cosinesimil",
        ], f"Internal: self.space={self.space} not allowed"

        hnsw_index = HNSW.nmslib.init(method=method, space=self.space)
        hnsw_index.addDataPointBatch(data)
        hnsw_index.createIndex(
            {
                "M": big_m,
                "efConstruction": ef_construction,
                "post": post_processing,
                "indexThreadQty": self.n_jobs,
            },
            print_progress=(self.verbose >= 2),
        )
        return hnsw_index

    def _kneighbors(self, k, query, index, return_distance, is_self_querying):
        # Fetch the neighbor candidates
        neigh_ind_dist = index.knnQueryBatch(query, k=k, num_threads=self.n_jobs)

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        n_query = query.shape[0]
        neigh_ind = -np.ones((n_query, k), dtype=np.int32)
        neigh_dist = np.empty_like(neigh_ind, dtype=query.dtype) * np.nan

        for i, (ind, dist) in enumerate(neigh_ind_dist):
            neigh_ind[i, : ind.size] = ind
            neigh_dist[i, : dist.size] = dist

        # Convert cosine similarities to cosine distances
        if self.space == "cosinesimil":
            neigh_dist *= -1
            neigh_dist += 1
        elif self.space == "l2" and self.metric == "sqeuclidean":
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
