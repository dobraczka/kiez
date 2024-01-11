from typing import List, Literal, Type, Union, overload

from kiez.neighbors import NNAlgorithm, nn_algorithm_resolver


@overload
def available_nn_algorithms(as_string: Literal[True]) -> List[str]:
    ...


@overload
def available_nn_algorithms(
    as_string: Literal[False] = False,
) -> List[Type[NNAlgorithm]]:
    ...


def available_nn_algorithms(
    as_string: bool = False,
) -> Union[List[str], List[Type[NNAlgorithm]]]:
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
            if as_string:
                available.append(ann.lower())
            else:
                available.append(nn_algorithm_resolver.lookup(ann))
        except ImportError:  # noqa: PERF203
            pass
    return available
