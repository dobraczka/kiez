# -*- coding: utf-8 -*-
# adapted from skhubness
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod


class HubnessReduction(ABC):
    """Base class for hubness reduction."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(
        self, neigh_dist, neigh_ind, source, target, assume_sorted, *args, **kwargs
    ):
        pass  # pragma: no cover

    @abstractmethod
    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query,
        assume_sorted,
        return_distance=True,
    ):
        pass  # pragma: no cover


class NoHubnessReduction(HubnessReduction):
    """Compatibility class for neighbor search without hubness reduction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        pass  # pragma: no cover

    def __repr__(self):
        return "NoHubnessReduction"

    def transform(
        self,
        neigh_dist,
        neigh_ind,
        query,
        assume_sorted=True,
        return_distance=True,
        *args,
        **kwargs
    ):
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
