# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Tom Dupre la Tour (original work)
#         Roman Feldbauer (adaptions for scikit-hubness)
#         Daniel Obraczka (adaptions for kiez)
# PEP 563: Postponed Evaluation of Annotations

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from kiez.io.temp_file_handling import create_tempfile_preferably_in_dir
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithmWithJoblib
from tqdm.auto import tqdm

try:
    import annoy  # noqa: autoimport

except ImportError:
    annoy = None  # pragma: no cover


class Annoy(NNAlgorithmWithJoblib):
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
        )
        if self.metric == "minkowski":  # for compatibility
            self.metric = "euclidean"
        metric = self.metric if self.metric != "sqeuclidean" else "euclidean"
        self.effective_metric_ = metric
        self.verbose = verbose
        self.n_trees = n_trees
        self.search_k = search_k
        self.mmap_dir = mmap_dir

    def _fit(self, data, is_source: bool):
        if is_source:
            self._source = data
            prefix = "kiez_source"
        else:
            self._target = data
            prefix = "kiez_target"
        suffix = ".annoy"
        annoy_index = annoy.AnnoyIndex(data.shape[1], metric=self.effective_metric_)
        if self.mmap_dir == "auto":
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix, directory="/dev/shm"
            )
            logging.warning(
                f"The index will be stored in {index_path}. It will NOT be deleted"
                " automatically, when this instance is destructed."
            )
        elif isinstance(self.mmap_dir, str):
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix, suffix=suffix, directory=self.mmap_dir
            )
        else:  # e.g. None
            self.mmap_dir = None

        for i, x in tqdm(
            enumerate(data),
            desc="Build Annoy",
            disable=not self.verbose,
        ):
            annoy_index.add_item(i, x.tolist())
        annoy_index.build(self.n_trees)

        if self.mmap_dir is not None:
            annoy_index.save(index_path)
        else:
            annoy_index = (index_path, data.shape[1])

        return annoy_index

    def _kneighbors_part(self, k, query, index, return_distance, is_self_querying):
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
        neigh_dist = np.empty_like(neigh_ind, dtype=query_dtype) * np.nan

        if isinstance(index, Tuple):
            err_msg = (
                "Internal error: annoy index must be either annoy.AnnoyIndex"
                + f"or Tuple[str,int], but found {index}"
            )
            try:
                index, n_features = index
                if not isinstance(index, str) or not isinstance(n_features, int):
                    raise ValueError(err_msg)
            except ValueError:
                raise ValueError(err_msg)
            # Load memory-mapped annoy.Index, unless it's already in main memory
            if isinstance(index, str):
                annoy_index = annoy.AnnoyIndex(
                    n_features, metric=self.effective_metric_
                )
                annoy_index.load(index)
        elif isinstance(index, annoy.AnnoyIndex):
            annoy_index = index
        assert isinstance(
            index, annoy.AnnoyIndex
        ), "Internal error: unexpected type for annoy index"

        disable_tqdm = not self.verbose
        if is_self_querying:
            n_items = annoy_index.get_n_items()

            for i in tqdm(
                range(n_items),
                desc="Query Annoy",
                disable=disable_tqdm,
            ):
                ind, dist = annoy_index.get_nns_by_item(
                    i,
                    k,
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
                desc="Query Annoy",
                disable=disable_tqdm,
            ):
                ind, dist = annoy_index.get_nns_by_vector(
                    x.tolist(),
                    k,
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
