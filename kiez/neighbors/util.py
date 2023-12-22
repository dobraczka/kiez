from typing import List, Type

from kiez.neighbors import NNAlgorithm, nn_algorithm_resolver


def available_nn_algorithms() -> List[Type[NNAlgorithm]]:
    """Get available (approximate) nearest neighbor algorithms.

    Returns
    -------
    algorithms: List[Type[NNAlgorithm]]
        A tuple of available algorithms
    """
    possible = ["NMSLIB", "NNG", "Annoy", "Faiss", "SklearnNN"]
    available = []
    for ann in possible:
        try:
            nn_algorithm_resolver.make(ann)
            available.append(nn_algorithm_resolver.lookup(ann))
        except ImportError:
            pass
    return available
