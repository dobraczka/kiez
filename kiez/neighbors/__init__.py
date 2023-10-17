from class_resolver import ClassResolver

from kiez.neighbors.approximate.faiss import Faiss
from kiez.neighbors.approximate.nmslib import NMSLIB
from kiez.neighbors.approximate.nng import NNG
from kiez.neighbors.approximate.random_projection_trees import Annoy
from kiez.neighbors.exact.sklearn_nearest_neighbors import SklearnNN
from kiez.neighbors.neighbor_algorithm_base import NNAlgorithm, NNAlgorithmWithJoblib

__all__ = [
    "nn_algorithm_resolver",
    "NMSLIB",
    "NNG",
    "Annoy",
    "SklearnNN",
    "Faiss",
    "NNAlgorithm",
]

nn_algorithm_resolver = ClassResolver.from_subclasses(
    base=NNAlgorithm,
    skip={  # Skip being able to resolve intermediate base classes
        NNAlgorithmWithJoblib,
    },
    default=SklearnNN,
)
