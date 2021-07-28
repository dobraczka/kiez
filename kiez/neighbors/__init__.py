from class_resolver import Resolver

from kiez.neighbors.approximate.hnsw import HNSW
from kiez.neighbors.approximate.nng import NNG
from kiez.neighbors.approximate.random_projection_trees import Annoy
from kiez.neighbors.exact.sklearn_nearest_neighbors import SklearnNN
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm, NNAlgorithmWithJoblib

__all__ = ["nn_algorithm_resolver", "HNSW", "NNG", "Annoy", "SklearnNN", "NNAlgorithm"]

nn_algorithm_resolver = Resolver.from_subclasses(
    base=NNAlgorithm,
    skip={  # Skip being able to resolve intermediate base classes
        NNAlgorithmWithJoblib,
    },
    default=SklearnNN,
)
