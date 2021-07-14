"""
Calculate evaluation metrics such as hits@k
"""
from typing import Any, Dict, List, Union

import numpy as np


def _hits_from_ndarray(nn_ind, gold, k, hits_counter):
    for hits_at_k in k:
        for i, _ in enumerate(nn_ind):
            if i in gold and gold[i] in nn_ind[i][:hits_at_k]:
                hits_counter[hits_at_k] += 1
    return hits_counter


def _hits_from_dict(nn_ind, gold, k, hits_counter):
    for hits_at_k in k:
        for i, v in nn_ind.items():
            if i in gold and gold[i] in v[:hits_at_k]:
                hits_counter[hits_at_k] += 1
    return hits_counter


def hits(
    nn_ind: Union[np.ndarray, Dict[Any, List]],
    gold: Dict[Any, Any],  # source -> target
    k=None,
) -> Dict[int, float]:
    """Show hits@k

    Parameters
    ----------
    nn_ind:  array or dict
        Contains the indices of the nearest neighbors for source entities.
    gold: dict
        Map of source indices to the respective target indices
    k: array
        Which ks should be evaluated

    Returns
    -------
    hits: dict
        k: relative hits@k values

    Examples
    --------
    >>> from kiez.evaluate import hits
    >>> import numpy as np
    >>> nn_ind = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    >>> gold = {0: 2, 1: 4, 2: 3, 3: 4}
    >>> hits(nn_ind, gold)
    {1: 0.5, 5: 1.0, 10: 1.0}
    """
    if k is None:
        k = [1, 5, 10]
    k.sort()
    hits_counter = {hits_at_k: 0 for hits_at_k in k}
    if isinstance(nn_ind, (np.ndarray, list)):
        hits_counter = _hits_from_ndarray(nn_ind, gold, k, hits_counter)
    elif isinstance(nn_ind, Dict):
        hits_counter = _hits_from_dict(nn_ind, gold, k, hits_counter)
    return {hits_at_k: k_val / len(gold) for hits_at_k, k_val in hits_counter.items()}
