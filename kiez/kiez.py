import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union, overload

import numpy as np
from class_resolver import HintOrType

from kiez.hubness_reduction import (
    hubness_reduction_resolver,
)
from kiez.hubness_reduction.base import HubnessReduction
from kiez.neighbors import NNAlgorithm, nn_algorithm_resolver
from kiez.neighbors.util import available_nn_algorithms

T = TypeVar("T")


class Kiez:
    """Performs hubness reduced nearest neighbor search for entity alignment.

    Use the given algorithm to :meth:`fit` the data and calculate the
    :meth:`kneighbors`.

    Parameters
    ----------
    n_candidates : int, default=10
        number of nearest neighbors used for candidate search
    algorithm : :obj:`~kiez.neighbors.NNAlgorithm`, default = None
        initialised `NNAlgorithm` object that will be used for neighbor search
        If no algorithm is provided :obj:`~kiez.neighbors.Faiss` is used if available else
        :obj:`~kiez.neighbors.SklearnNN` is used with default values
    algorithm_kwargs :
        A dictionary of keyword arguments to pass to the :obj:`~kiez.neighbors.NNAlgorithm`
        if given as a class in the ``algorithm`` argument.

    hubness :
        Either an instance of a :obj:`~kiez.hubness_reduction.base.HubnessReduction`,
        the class for a :obj:`~kiez.hubness_reduction.base.HubnessReduction` that should
        be instantiated, the name of the hubness reduction method, or if None, defaults
        to no hubness reduction.
    hubness_kwargs :
        A dictionary of keyword arguments to pass to the :obj:`~kiez.hubness_reduction.base.HubnessReduction`
        if given as a class in the ``hubness`` argument.

    Examples
    --------
    >>> from kiez import Kiez
    >>> import numpy as np
    >>> # create example data
    >>> rng = np.random.RandomState(0)
    >>> source = rng.rand(100,50)
    >>> target = rng.rand(100,50)
    >>> # fit and get neighbors
    >>> k_inst = Kiez()
    >>> k_inst.fit(source, target)
    >>> nn_dist, nn_ind = k_inst.kneighbors(5)

    Using a specific algorithm and hubness reduction

    >>> from kiez import Kiez
    >>> import numpy as np
    >>> # create example data
    >>> rng = np.random.RandomState(0)
    >>> source = rng.rand(100,50)
    >>> target = rng.rand(100,50)
    >>> # prepare algorithm and hubness reduction
    >>> k_inst = Kiez(n_candidates=10, algorithm="Faiss", hubness="CSLS")
    >>> # fit and get neighbors
    >>> k_inst.fit(source, target)
    >>> nn_dist, nn_ind = k_inst.kneighbors(5)

    You can investigate which NN algos are installed and which hubness methods are implemented with:

    >>> Kiez.show_hubness_options()
    >>> Kiez.show_algorithm_options()

    Beginning with version 0.5.0 torch can be used, when using `Faiss` as NN algorithm:

    >>> from kiez import Kiez
    >>> import torch
    >>> source = torch.randn((100,10))
    >>> target = torch.randn((200,10))
    >>> k_inst = Kiez(algorithm="Faiss", hubness="CSLS")
    >>> k_inst.fit(source, target)
    >>> nn_dist, nn_ind = k_inst.kneighbors()

    You can also utilize tensors and NN calculations on the GPU:

    >>> k_inst = Kiez(algorithm="Faiss", algorithm_kwargs={"use_gpu":True}, hubness="CSLS")
    >>> k_inst.fit(source.cuda(), target.cuda())
    >>> nn_dist, nn_ind = k_inst.kneighbors()

    You can also initalize Kiez via a json file

    >>> kiez = Kiez.from_path("tests/example_conf.json")
    """

    def __init__(
        self,
        n_candidates: int = 10,
        algorithm: HintOrType[NNAlgorithm] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        hubness: HintOrType[HubnessReduction] = None,
        hubness_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not np.issubdtype(type(n_candidates), np.integer):
            raise TypeError(
                f"n_neighbors does not take {type(n_candidates)} value, enter"
                " integer value"
            )

        if n_candidates <= 0:
            raise ValueError(f"Expected n_candidates > 0. Got {n_candidates}")
        if algorithm_kwargs is None:
            algorithm_kwargs = {"n_candidates": n_candidates}
        elif "n_candidates" not in algorithm_kwargs:
            algorithm_kwargs["n_candidates"] = n_candidates
        if algorithm is None:
            try:
                algorithm = nn_algorithm_resolver.make("Faiss", algorithm_kwargs)
            except ImportError:
                algorithm = nn_algorithm_resolver.make("SklearnNN", algorithm_kwargs)
        else:
            algorithm = nn_algorithm_resolver.make(algorithm, algorithm_kwargs)
        assert algorithm
        if hubness_kwargs is None:
            hubness_kwargs = {}
        hubness_kwargs["nn_algo"] = algorithm
        self.hubness = hubness_reduction_resolver.make(hubness, hubness_kwargs)

    @staticmethod
    def show_algorithm_options() -> List[str]:
        return available_nn_algorithms(as_string=True)

    @staticmethod
    def show_hubness_options() -> List[str]:
        return list(hubness_reduction_resolver.options)

    @property
    def algorithm(self):
        return self.hubness.nn_algo

    @algorithm.setter
    def algorithm(self, value):
        self.hubness.nn_algo = value

    def __repr__(self):
        return (
            f"Kiez(algorithm: {self.algorithm},"
            f" hubness: {self.hubness})"
            f" {self.algorithm._describe_source_target_fitted()}"
        )

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "Kiez":
        """Load a Kiez instance from configuration in a JSON file, based on its path."""
        with open(path) as file:
            return cls(**json.load(file))

    def fit(self, source: T, target: Optional[T] = None) -> "Kiez":
        """Fits the algorithm and hubness reduction method.

        Parameters
        ----------
        source : matrix of shape (n_samples, n_features)
            embeddings of source entities
        target : matrix of shape (m_samples, n_features)
            embeddings of target entities. If none given, uses the source.

        Returns
        -------
        Kiez
            Fitted kiez instance
        """
        self.hubness.fit(source, target)
        return self

    @overload
    def kneighbors(
        self,
        k: Optional[int] = None,
        return_distance: Literal[True] = True,
    ) -> Tuple[T, T]:
        ...

    @overload
    def kneighbors(
        self,
        k: Optional[int] = None,
        return_distance: Literal[False] = False,
    ) -> Any:
        ...

    def kneighbors(
        self,
        k: Optional[int] = None,
        return_distance: bool = True,
    ) -> Union[T, Tuple[T, T]]:
        """Retrieve the k-nearest neighbors using the supplied nearest neighbor algorithm and hubness reduction method.

        Parameters
        ----------
        k : Optional[int], default = None
            k-nearest neighbors, if None is set to number of n_candidates
        return_distance : bool, default = True
            Whether to return distances
            If `False` only indices are returned

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the distance between source and target entities
            only present if return_distance=True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        hubness_reduced_query_dist, query_ind = self.hubness.kneighbors(k)

        if return_distance:
            result = hubness_reduced_query_dist, query_ind
        else:
            result = query_ind
        return result
